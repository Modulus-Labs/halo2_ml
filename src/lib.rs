pub mod nn_chip;
pub mod nn_ops;

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error as PlonkError, Instance}, pasta::Fp,
};
use nn_chip::{ForwardLayerChip, ForwardLayerConfig, LayerParams, NNLayerInstructions};

use crate::nn_ops::eltwise_ops::{ReluConfig, EltwiseOp};
use crate::nn_ops::lookup_ops::DecompTable;

#[derive(Clone, Debug)]
///Config for Neural Net Chip
pub struct NeuralNetConfig<F: FieldExt> {
    input: Column<Instance>,
    output: Column<Instance>,
    range_table: DecompTable<F, 1024>,
    layers: Vec<ForwardLayerConfig<F, 1024, ReluConfig<F, 1024>>>,
}

#[derive(Default)]
pub struct NNCircuit<F: FieldExt> {
    pub layers: Vec<LayerParams<F>>,
    pub input: Vec<F>,
    //_marker: PhantomData<(F, Elt)>,
}

impl<F: FieldExt> Circuit<F> for NNCircuit<F> {
    type Config = NeuralNetConfig<F>;

    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        const MAX_MAT_WIDTH: usize = 4;
        let input = meta.instance_column();
        meta.enable_equality(input);
        let output = meta.instance_column();
        meta.enable_equality(output);

        let mat_advices: Vec<Column<Advice>> = (0..MAX_MAT_WIDTH + 3)
            .map(|_| {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            })
            .collect();

        const DECOMP_COMPONENTS: usize = <ReluConfig<Fp, 1024> as EltwiseOp<Fp, 1024>>::ADVICE_LEN;
        let elt_advices: Vec<Column<Advice>> = (0..=DECOMP_COMPONENTS+1)
            .map(|_| {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            })
            .collect();

        const MATRICES: [(usize, usize); 2] = [(4, 4), (4, 4)];

        let range_table = DecompTable::configure(meta);

        let layers = MATRICES
            .iter()
            .map(|matrix| {
                ForwardLayerChip::configure(
                    meta,
                    mat_advices[0],
                    mat_advices[1..matrix.1+1].into(),
                    mat_advices[mat_advices.len() - 2],
                    mat_advices[mat_advices.len() - 1],
                    elt_advices[0..=elt_advices.len() - 2].into(),
                    elt_advices[elt_advices.len() - 1],
                    range_table.clone(),
                    [matrix.0, matrix.1],
                )
            })
            .collect();

        NeuralNetConfig {
            input,
            output,
            range_table,
            layers,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), PlonkError> {
        config.range_table.layout(layouter.namespace(|| "range check lookup table"))?;

        let layers: Vec<_> = config
            .layers
            .into_iter()
            .map(|config| ForwardLayerChip::<F, 1024, ReluConfig<F, 1024>>::construct(config))
            .collect();
        let input = layers[0].load_input_instance(
            layouter.namespace(|| "Load input from constant"),
            config.input,
            0,
            self.input.len(),
        )?;
        let output =
            self.layers
                .iter()
                .zip(layers.iter())
                .enumerate().fold(Ok(input), |input, (index, (layer, chip))| {
                    chip.add_layers(layouter.namespace(|| format!("NN Layer {index}")), input?, layer)
                })?;
        for (index, cell) in output.into_iter().enumerate() {
            layouter
                .namespace(|| format!("contrain output at offset {index}"))
                .constrain_instance(cell.0.cell(), config.output, index)?;
        }
        Ok(())
    }
}
