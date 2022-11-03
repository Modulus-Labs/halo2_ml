#![feature(adt_const_params)]
use std::{collections::HashMap, time::Instant};

use halo2_machinelearning::{
    nn_chip::{ForwardLayerChip, ForwardLayerConfig, LayerParams, NNLayerInstructions},
    nn_ops::{self, eltwise_ops::NormalizeChip},
};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, SimpleFloorPlanner},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error as PlonkError, Instance},
    poly::{
        commitment::{ParamsProver},
        kzg::{
            commitment::ParamsKZG,
            multiopen::{ProverSHPLONK, VerifierSHPLONK},
            strategy::SingleStrategy,
        },
    },
    transcript::{Blake2bRead, TranscriptReadBuffer, TranscriptWriterBuffer},
};
use nn_ops::eltwise_ops::NormalizeReluChip;

use halo2_machinelearning::nn_ops::lookup_ops::DecompTable;

use halo2_proofs::{
    halo2curves::{bn256::Bn256, bn256::Fr},
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof},
    transcript::{Blake2bWrite, Challenge255},
};
use rand::rngs::OsRng;

const BASE: usize = 1024;

#[derive(Clone, Debug)]
///Config for Neural Net Chip
pub struct LenetConfig<F: FieldExt> {
    input: Column<Instance>,
    output: Column<Instance>,
    range_table: DecompTable<F, 1024>,
    //layers: Vec<ForwardLayerConfig<F, NormalizeReluChip<F, 1024, 2>, 16, 16>>,
    layers: HashMap<(usize, usize, bool), ForwardLayerConfig<F>>,
}

const DEPTH: usize = 3;

type NetworkArch = [(usize, usize, bool); DEPTH];

#[derive(Default)]
pub struct LenetCircuit<F: FieldExt, const STRUCT: NetworkArch> {
    pub layers: Vec<LayerParams<F>>,
    pub input: Vec<F>,
    //_marker: PhantomData<&'a PhantomData<F>>,
}

impl<F: FieldExt, const NETWORK: NetworkArch> Circuit<F> for LenetCircuit<F, NETWORK> {
    type Config = LenetConfig<F>;

    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        //todo!()
        const MAX_MAT_WIDTH: usize = 200;
        let input = meta.instance_column();
        meta.enable_equality(input);
        let output = meta.instance_column();
        meta.enable_equality(output);

        let mat_advices: Vec<Column<Advice>> = (0..(2 * MAX_MAT_WIDTH) + 2)
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

        let range_table: DecompTable<F, BASE> = DecompTable::configure(meta);

        let relu_chip = NormalizeReluChip::<_, BASE, 2>::configure(
            meta,
            elt_advices[0].clone(),
            elt_advices[1..elt_advices.len() - 1].into(),
            elt_advices[elt_advices.len() - 1].clone(),
            range_table.clone(),
        );

        let norm_chip = NormalizeChip::<_, BASE, 2>::configure(
            meta,
            elt_advices[0],
            elt_advices[1..elt_advices.len() - 1].into(),
            elt_advices[elt_advices.len() - 1],
            range_table.clone(),
        );

        let mut layers = HashMap::new();

        for key in NETWORK {
            if !layers.contains_key(&key) {
                let (width, height, relu) = key;
                if relu {
                    layers.insert(
                        key,
                        ForwardLayerChip::<F, NormalizeReluChip<F, BASE, 2>>::configure(
                            meta,
                            width,
                            height,
                            &mat_advices[0..width],
                            &mat_advices[width..(2 * width)],
                            mat_advices[mat_advices.len() - 2].clone(),
                            mat_advices[mat_advices.len() - 1].clone(),
                            relu_chip.clone(),
                        ),
                    );
                } else {
                    layers.insert(
                        key,
                        ForwardLayerChip::<_, NormalizeChip<F, BASE, 2>>::configure(
                            meta,
                            width,
                            height,
                            mat_advices[0..width].try_into().unwrap(),
                            mat_advices[width..(2 * width)]
                                .try_into()
                                .unwrap(),
                            mat_advices[mat_advices.len() - 2].clone(),
                            mat_advices[mat_advices.len() - 1].clone(),
                            norm_chip.clone(),
                        ),
                    );
                }
            }
        }

        LenetConfig {
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
        config
            .range_table
            .layout(layouter.namespace(|| "range check lookup table"))?;

        let mut input: Option<Vec<AssignedCell<F, F>>> = None;
        for (index, (key, layer)) in NETWORK.iter().zip(self.layers.iter()).enumerate() {
            let (_, _, relu) = key;
            input = if *relu {
                let chip = ForwardLayerChip::<_, NormalizeReluChip<F, BASE, 2>>::construct(config.layers.get(&key).unwrap().clone());
                let inter = input.unwrap_or(chip.load_input_instance(layouter.namespace(|| "Load input from instance"), config.input, 0, self.input.len())?);
                let inter = chip.add_layers(layouter.namespace(|| format!("running layer {}; definintion: {:?}", index, key)), inter, layer)?;
                Some(inter)
            }
            else {
                let chip = ForwardLayerChip::<_, NormalizeChip<F, BASE, 2>>::construct(config.layers.get(&key).unwrap().clone());
                let inter = input.unwrap_or(chip.load_input_instance(layouter.namespace(|| "Load input from instance"), config.input, 0, self.input.len())?);
                let inter = chip.add_layers(layouter.namespace(|| format!("running layer {}; definintion: {:?}", index, key)), inter, layer)?;
                Some(inter)
            }
        }

        for (index, cell) in input.unwrap().into_iter().enumerate() {
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

    let k = 13;

    const NETWORK: NetworkArch = [(32, 100, true), (100, 200, true), (200, 100, false)];

    let (input, layers, output) = get_inputs("FC_3_layers.json");

    let circuit = LenetCircuit::<Fr, NETWORK> {
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
        let params: ParamsKZG<Bn256> = ParamsProver::new(k);

        let vk = keygen_vk(&params, &circuit).unwrap();

        let pk = keygen_pk(&params, vk, &circuit).unwrap();

        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

        let now = Instant::now();

        println!("starting proof!");

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

        assert!(verify_proof::<_, VerifierSHPLONK<Bn256>, _, _, _>(
            &params,
            &pk.get_vk(),
            strategy,
            &[&[input.as_slice(), output.as_slice()]],
            &mut transcript
        )
        .is_ok());

        println!("Verification took {}", now.elapsed().as_secs());
    }

    #[cfg(feature = "dhat-heap")]
    {
        let stats = dhat::HeapStats::get();
        println!("{:?}", stats.max_bytes);
    }
}

fn get_inputs(file_path: &str) -> (Vec<Fr>, Vec<LayerParams<Fr>>, Vec<Fr>) {
    //const PREFIX: &str = "/home/aweso/halo2_machinelearning/bench_objects/";
    const PREFIX: &str = "/home/ubuntu/halo2_benches_new/bench_objects/";
    let inputs_raw = std::fs::read_to_string(PREFIX.to_owned() + file_path)
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
