use std::time::Instant;

use halo2_machinelearning::{
    nn_chip::{ForwardLayerChip, ForwardLayerConfig, LayerParams, NNLayerInstructions},
    nn_ops::{self, eltwise_ops::NormalizeChip},
};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error as PlonkError, Instance}, transcript::{Blake2bRead, TranscriptWriterBuffer, TranscriptReadBuffer},
    poly::{commitment::{Params, ParamsProver, ParamsVerifier}, kzg::{commitment::ParamsKZG, multiopen::{ProverSHPLONK, VerifierSHPLONK}, strategy::SingleStrategy}},
};
use nn_ops::eltwise_ops::{NormalizeReluChip};

use halo2_machinelearning::nn_ops::lookup_ops::DecompTable;

use halo2_proofs::{
    halo2curves::{bn256::Bn256, bn256::Fr},
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof},
    transcript::{Blake2bWrite, Challenge255},
};
use rand::rngs::OsRng;

const DIMS: [[usize; 2]; 5] = [[784, 600], [600, 200], [200, 120], [120, 84], [84, 10]];

#[derive(Clone, Debug)]
///Config for Neural Net Chip
pub struct LenetConfig<F: FieldExt> {
    input: Column<Instance>,
    output: Column<Instance>,
    range_table: DecompTable<F, 1024>,
    //layers: Vec<ForwardLayerConfig<F, NormalizeReluChip<F, 1024, 2>, 16, 16>>,
    layer_1: ForwardLayerConfig<F, NormalizeChip<F, 1024, 2>, 784, 600>,
    layer_2: ForwardLayerConfig<F, NormalizeChip<F, 1024, 2>, 600, 200>,
    layer_3: ForwardLayerConfig<F, NormalizeReluChip<F, 1024, 2>, 200, 120>,
    layer_4: ForwardLayerConfig<F, NormalizeReluChip<F, 1024, 2>, 120, 84>,
    layer_5: ForwardLayerConfig<F, NormalizeChip<F, 1024, 2>, 84, 10>,
}

#[derive(Default)]
pub struct LenetCircuit<F: FieldExt> {
    pub layers: Vec<LayerParams<F>>,
    pub input: Vec<F>,
    //_marker: PhantomData<&'a PhantomData<F>>,
}

impl<F: FieldExt> Circuit<F> for LenetCircuit<F> {
    type Config = LenetConfig<F>;

    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        const MAX_MAT_WIDTH: usize = 784;
        let input = meta.instance_column();
        meta.enable_equality(input);
        let output = meta.instance_column();
        meta.enable_equality(output);

        let mat_advices: Vec<Column<Advice>> = (0..(2*MAX_MAT_WIDTH) + 2)
            .map(|_| {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            })
            .collect();


        const DECOMP_COMPONENTS: usize = 15;
        let elt_advices: Vec<Column<Advice>> = (0..=DECOMP_COMPONENTS + 2)
            .map(|_| {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            })
            .collect();


        let range_table = DecompTable::configure(meta);

        let relu_chip = NormalizeReluChip::construct(NormalizeReluChip::configure(
            meta,
            elt_advices[0].clone(),
            elt_advices[1..elt_advices.len() - 1].into(),
            elt_advices[elt_advices.len() - 1].clone(),
            range_table.clone(),
        ));

        let norm_chip = NormalizeChip::construct(NormalizeChip::configure(
            meta,
            elt_advices[0],
            elt_advices[1..elt_advices.len() - 1].into(),
            elt_advices[elt_advices.len() - 1],
            range_table.clone(),
        ));

        let layer_1 = ForwardLayerChip::configure(
            meta,
            mat_advices[0..DIMS[0][0]].try_into().unwrap(),
            mat_advices[DIMS[0][0]..(2*DIMS[0][0])].try_into().unwrap(),
            mat_advices[mat_advices.len() - 2].clone(),
            mat_advices[mat_advices.len() - 1].clone(),
            norm_chip.clone(),
        );


        let layer_2 = ForwardLayerChip::configure(
            meta,
            mat_advices[0..DIMS[1][0]].try_into().unwrap(),
            mat_advices[DIMS[1][0]..(2*DIMS[1][0])].try_into().unwrap(),
            mat_advices[mat_advices.len() - 2].clone(),
            mat_advices[mat_advices.len() - 1].clone(),
            norm_chip.clone(),
        );


        let layer_3 = ForwardLayerChip::configure(
            meta,
            mat_advices[0..DIMS[2][0]].try_into().unwrap(),
            mat_advices[DIMS[2][0]..(2*DIMS[2][0])].try_into().unwrap(),
            mat_advices[mat_advices.len() - 2].clone(),
            mat_advices[mat_advices.len() - 1].clone(),
            relu_chip.clone(),
        );


        let layer_4 = ForwardLayerChip::configure(
            meta,
            mat_advices[0..DIMS[3][0]].try_into().unwrap(),
            mat_advices[DIMS[3][0]..(2*DIMS[3][0])].try_into().unwrap(),
            mat_advices[mat_advices.len() - 2],
            mat_advices[mat_advices.len() - 1],
            relu_chip.clone(),
        );


        let layer_5 = ForwardLayerChip::configure(
            meta,
            mat_advices[0..DIMS[4][0]].try_into().unwrap(),
            mat_advices[DIMS[4][0]..(2*DIMS[4][0])].try_into().unwrap(),
            mat_advices[mat_advices.len() - 2],
            mat_advices[mat_advices.len() - 1],
            norm_chip.clone(),
        );


        LenetConfig {
            input,
            output,
            range_table,
            layer_1,
            layer_2,
            layer_3,
            layer_4,
            layer_5,
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

        let l1 = ForwardLayerChip::construct(config.layer_1);
        let l2 = ForwardLayerChip::construct(config.layer_2);
        let l3 = ForwardLayerChip::construct(config.layer_3);
        let l4 = ForwardLayerChip::construct(config.layer_4);
        let l5 = ForwardLayerChip::construct(config.layer_5);
        let input = l1.load_input_instance(
            layouter.namespace(|| "Load input from constant"),
            config.input,
            0,
            self.input.len(),
        )?;


        let input_l2 = l1.add_layers(
            layouter.namespace(|| format!("NN Layer 1")),
            input,
            &self.layers[0],
        )?;


        let input_l3 = l2.add_layers(
            layouter.namespace(|| format!("NN Layer 2")),
            input_l2,
            &self.layers[1],
        )?;


        let input_l4 = l3.add_layers(
            layouter.namespace(|| format!("NN Layer 3")),
            input_l3,
            &self.layers[2],
        )?;


        let input_l5 = l4.add_layers(
            layouter.namespace(|| format!("NN Layer 4")),
            input_l4,
            &self.layers[3],
        )?;


        let output = l5.add_layers(
            layouter.namespace(|| format!("NN Layer 5")),
            input_l5,
            &self.layers[4],
        )?;

        for (index, cell) in output.into_iter().enumerate() {
            layouter
                .namespace(|| format!("contrain output at offset {index}"))
                .constrain_instance(cell.cell(), config.output, index)?;
        }
        Ok(())
    }
}

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;


fn main() -> () {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::builder().testing().build();

    let k = 11;

    let (input, layers, output) = get_inputs();

    let circuit = LenetCircuit::<Fr> {
        layers,
        input: input.clone(),
    };

    #[cfg(feature = "mock")]
    {
        use halo2_proofs::dev::MockProver;
        let now = Instant::now();

        MockProver::run(k, &circuit, vec![input.clone(), output.clone()])
            .unwrap()
            .assert_satisfied();

        println!("Mock prover is satisfied in {:?}", now.elapsed().as_secs());

        #[cfg(feature = "dev-graph")]
        {
        use plotters::prelude::*;

        let root = BitMapBackend::new("inner_product.png", (1024, 3096)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let root = root.titled("inner product", ("sans-serif", 60)).unwrap();
        halo2_proofs::dev::CircuitLayout::default()
        .render(k, &circuit, &root)
        .unwrap();
        }
    }

    #[cfg(not(feature = "mock"))]
    {
        let params: ParamsKZG<Bn256> = ParamsProver::new(11);

        let vk = keygen_vk(&params, &circuit).unwrap();
    
        let pk = keygen_pk(&params, vk, &circuit).unwrap();
    
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    
        let now = Instant::now();
    
        create_proof::<_, ProverSHPLONK<Bn256>, _, _, _, _>(
            &params,
            &pk,
            &[circuit],
            &[&[input.as_slice(), output.as_slice()]],
            OsRng,
            &mut transcript,
        )
        .unwrap();
    
        println!("Proof took {:?}", now.elapsed().as_secs());
    
        let proof = transcript.finalize();
        //println!("{:?}", proof);
        let now = Instant::now();
        let strategy = SingleStrategy::new(&params);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);

        assert!(verify_proof::<_, VerifierSHPLONK<Bn256>, _, _, _>(&params, &pk.get_vk(), strategy, &[&[input.as_slice(), output.as_slice()]], &mut transcript).is_ok());

        println!("Verification took {}", now.elapsed().as_secs());
    }

    #[cfg(feature = "dhat-heap")]
    {
        let stats = dhat::HeapStats::get();
        println!("{:?}", stats.max_bytes);
    }
}

fn get_inputs() -> (Vec<Fr>, Vec<LayerParams<Fr>>, Vec<Fr>) {
    let inputs_raw = std::fs::read_to_string(
        "/home/aweso/halo2_machinelearning/network_inputs/0_lenet_lookalike_test.json",
        //"/home/ubuntu/halo2_benches/network_inputs/1_lenet_lookalike_test.json",
    )
    .unwrap();
    let inputs = json::parse(&inputs_raw).unwrap();
    let input: Vec<_> = inputs["input"]
        .members()
        .map(|x| felt_from_i64(x.as_i64().unwrap()))
        .collect();
    let layers: Vec<LayerParams<Fr>> = inputs["layers"]
        .members()
        .map(|layer| LayerParams {
            weights: layer["weight"]
                .members()
                .map(|x| felt_from_i64(x.as_i64().unwrap()))
                .collect(),
            biases: layer["bias"]
                .members()
                .map(|x| felt_from_i64(x.as_i64().unwrap()))
                .collect(),
        })
        .collect();

    let output: Vec<_> = inputs["output"]
        .members()
        .map(|x| felt_from_i64(x.as_i64().unwrap()))
        .collect();

    (input, layers, output)
}

fn felt_from_i64(x: i64) -> Fr {
    if x.is_positive() {
        Fr::from(x.unsigned_abs())
    } else {
        Fr::from(x.unsigned_abs()).neg()
    }
}
