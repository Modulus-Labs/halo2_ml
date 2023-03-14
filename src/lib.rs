#![feature(associated_type_defaults)]
pub mod nn_ops;

use halo2_base::halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error as PlonkError, Fixed, Instance},
};
use nn_ops::{
    vector_ops::{
        linear::fc_old::{FcConfig, FcParams},
        non_linear::eltwise_ops::NormalizeReluChip,
    },
    DefaultDecomp,
};

use crate::nn_ops::{
    lookup_ops::DecompTable,
    vector_ops::{
        linear::fc_old::{FcChip, FcChipConfig},
        non_linear::eltwise_ops::NormalizeChip,
    },
    ColumnAllocator, NNLayer,
};

pub fn felt_from_i64<F: FieldExt>(x: i64) -> F {
    if x.is_positive() {
        F::from(x.unsigned_abs())
    } else {
        F::from(x.unsigned_abs()).neg()
    }
}

pub fn felt_to_i64<F: FieldExt>(x: F) -> i64 {
    if x > F::TWO_INV {
        -(x.neg().get_lower_128() as i64)
    } else {
        x.get_lower_128() as i64
    }
}

#[derive(Clone, Debug)]
///Config for Neural Net Chip
pub struct NeuralNetConfig<F: FieldExt> {
    input: Column<Instance>,
    output: Column<Instance>,
    input_advice: Column<Advice>,
    range_table: DecompTable<F, DefaultDecomp>,
    layers: Vec<FcConfig<F>>,
}

#[derive(Default, Clone)]
pub struct NNCircuit<F: FieldExt> {
    pub layers: Vec<FcParams<F>>,
    pub input: Vec<F>,
    pub output: Vec<F>,
    //_marker: PhantomData<&'a PhantomData<F>>,
}

impl<F: FieldExt> Circuit<F> for NNCircuit<F> {
    type Config = NeuralNetConfig<F>;

    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        const MAX_MAT_WIDTH: usize = 4;
        const INPUT_WIDTH: usize = 4;
        let input = meta.instance_column();
        meta.enable_equality(input);
        let output = meta.instance_column();
        meta.enable_equality(output);

        const DECOMP_COMPONENTS: usize = 10;
        let elt_advices: Vec<Column<Advice>> = (0..=DECOMP_COMPONENTS + 2)
            .map(|_| {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            })
            .collect();

        let range_table = DecompTable::configure(meta);

        let relu_chip = NormalizeChip::<F, 1024, 2>::configure(
            meta,
            elt_advices[0],
            elt_advices[1..elt_advices.len() - 1].into(),
            elt_advices[elt_advices.len() - 1],
            range_table.clone(),
        );

        let config = FcChipConfig {
            weights_height: INPUT_WIDTH,
            weights_width: 4,
            elt_config: relu_chip,
        };

        let mut advice_allocator = ColumnAllocator::<Advice>::new(meta, 1);
        let mut fixed_allocator = ColumnAllocator::<Fixed>::new(meta, 0);

        let layers = vec![
            FcChip::<_, NormalizeChip<F, 1024, 2>>::configure(
                meta,
                config.clone(),
                &mut advice_allocator,
                &mut fixed_allocator,
            ),
            FcChip::<_, NormalizeChip<F, 1024, 2>>::configure(
                meta,
                config,
                &mut advice_allocator,
                &mut fixed_allocator,
            ),
        ];

        NeuralNetConfig {
            input,
            output,
            input_advice: {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            },
            range_table,
            layers,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), PlonkError> {
        config
            .range_table
            .layout(layouter.namespace(|| "range check lookup table"))?;

        let layers: Vec<_> = config
            .layers
            .into_iter()
            .map(|config| FcChip::<_, NormalizeReluChip<F, 1024, 2>>::construct(config))
            .collect();
        // let input = layers[0].load_input_instance(
        //     layouter.namespace(|| "Load input from constant"),
        //     config.input,
        //     0,
        //     self.input.len(),
        // )?;

        let input = layouter.assign_region(
            || "load input",
            |mut region| {
                self.input
                    .iter()
                    .enumerate()
                    .map(|(row, _)| {
                        region.assign_advice_from_instance(
                            || "Load Input",
                            config.input,
                            row,
                            config.input_advice,
                            row,
                        )
                    })
                    .collect()
            },
        )?;
        let output =
            self.layers.iter().zip(layers.iter()).enumerate().fold(
                Ok(input),
                |input, (_index, (layer, chip))| {
                    chip.add_layer(&mut layouter, input?, layer.clone())
                },
            )?;
        for (index, cell) in output.into_iter().enumerate() {
            layouter
                .namespace(|| format!("contrain output at offset {index}"))
                .constrain_instance(cell.cell(), config.output, index)?;
        }
        Ok(())
    }
}

impl<F: FieldExt> NNCircuit<F> {
    pub fn num_instances(&self) -> Vec<usize> {
        vec![self.input.len(), self.output.len()]
    }

    pub fn instances(&self) -> Vec<Vec<F>> {
        vec![self.input.clone(), self.output.clone()]
    }
}
