#![feature(adt_const_params)]
use std::{collections::HashMap, time::Instant};

use halo2_machinelearning::{
    nn_chip::{ForwardLayerChip, ForwardLayerConfig, LayerParams, NNLayerInstructions},
    nn_ops::{self, eltwise_ops::NormalizeChip},
};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, SimpleFloorPlanner, floor_planner::V1},
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
    range_table: DecompTable<F, BASE>,
    //layers: Vec<ForwardLayerConfig<F, NormalizeReluChip<F, 1024, 2>, 16, 16>>,
    layers: HashMap<(usize, usize, bool), ForwardLayerConfig<F>>,
}

type NetworkArch = &'static [(usize, usize, bool)];

#[derive(Default)]
pub struct LenetCircuit<F: FieldExt, const STRUCT: NetworkArch, const MAX_MAT_WIDTH: usize> {
    pub layers: Vec<LayerParams<F>>,
    pub input: Vec<F>,
    //_marker: PhantomData<&'a PhantomData<F>>,
}

impl<F: FieldExt, const NETWORK: NetworkArch, const MAX_MAT_WIDTH: usize> Circuit<F> for LenetCircuit<F, NETWORK, MAX_MAT_WIDTH> {
    type Config = LenetConfig<F>;

    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        //todo!()
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
            if !layers.contains_key(key) {
                let (width, height, relu) = *key;
                if relu {
                    layers.insert(
                        *key,
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
                        *key,
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
        if self.layers.len() == 0 {return Ok(());}
        config
            .range_table
            .layout(layouter.namespace(|| "range check lookup table"))?;

        let mut input: Option<Vec<AssignedCell<F, F>>> = None;
        for (index, (key, layer)) in NETWORK.iter().zip(self.layers.iter()).enumerate() {
            let (_, _, relu) = key;
            input = if *relu {
                let chip = ForwardLayerChip::<_, NormalizeReluChip<F, BASE, 2>>::construct(config.layers.get(&key).unwrap().clone());
                let inter = input.unwrap_or_else(|| chip.load_input_instance(layouter.namespace(|| "Load input from instance"), config.input, 0, self.input.len()).unwrap());
                let inter = chip.add_layers(layouter.namespace(|| format!("running layer {}; definintion: {:?}", index, key)), inter, layer)?;
                Some(inter)
            }
            else {
                let chip = ForwardLayerChip::<_, NormalizeChip<F, BASE, 2>>::construct(config.layers.get(&key).unwrap().clone());
                let inter = input.unwrap_or_else(|| chip.load_input_instance(layouter.namespace(|| "Load input from instance"), config.input, 0, self.input.len()).unwrap());
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

fn bench_layer<const NETWORK: NetworkArch, const MAX_MAT_WIDTH: usize>(file_path: &str, k: u32) -> Result<(), PlonkError> {
    println!("Benches for NN: {:?}", file_path);
    println!("-------------");

    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::builder().testing().build();

    let (input, layers, output) = get_inputs(file_path);

    let circuit = LenetCircuit::<Fr, NETWORK, MAX_MAT_WIDTH> {
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
        )?;

        println!("Proof took {:?}", now.elapsed().as_secs());

        let proof = transcript.finalize();
        //println!("{:?}", proof);
        let now = Instant::now();
        let strategy = SingleStrategy::new(&params);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);

        verify_proof::<_, VerifierSHPLONK<Bn256>, _, _, _>(
            &params,
            &pk.get_vk(),
            strategy,
            &[&[input.as_slice(), output.as_slice()]],
            &mut transcript
        )?;

        println!("Verification took {}", now.elapsed().as_secs());
        println!("-------------");
    }

    #[cfg(feature = "dhat-heap")]
    {
        let stats = dhat::HeapStats::get();
        println!("{:?}", stats.max_bytes);
    }
    Ok(())
}

fn main() {
    const NETWORK0: NetworkArch = &[(784, 95, true), (95, 104, true), (104, 114, true), (114, 126, true), (126, 139, true), (139, 152, true), (152, 168, true), (168, 185, true), (185, 203, true), (203, 224, true), (224, 246, true), (246, 271, true), (271, 298, true), (298, 327, true), (327, 360, true), (360, 396, true), (396, 436, true), (436, 480, true), (480, 528, true), (528, 581, true), (581, 639, true), (639, 703, true), (703, 773, true), (773, 850, true), (850, 935, true), (935, 1029, true), (1029, 1132, true), (1132, 1245, true), (1245, 1369, true), (1369, 1000, false), ];
    bench_layer::<NETWORK0, 1369>("FC_11216646_params.json", 14).unwrap();
    const NETWORK1: NetworkArch = &[(784, 27, true), (27, 29, true), (29, 32, true), (32, 35, true), (35, 39, true), (39, 43, true), (43, 47, true), (47, 52, true), (52, 57, true), (57, 63, true), (63, 70, true), (70, 77, true), (77, 84, true), (84, 93, true), (93, 102, true), (102, 112, true), (112, 124, true), (124, 136, true), (136, 150, true), (150, 165, true), (165, 181, true), (181, 199, true), (199, 219, true), (219, 241, true), (241, 265, true), (265, 292, true), (292, 321, true), (321, 353, true), (353, 389, true), (389, 1000, false), ];
    bench_layer::<NETWORK1, 784>("FC_1195804_params.json", 13).unwrap();
    const NETWORK2: NetworkArch = &[(784, 110, true), (110, 121, true), (121, 133, true), (133, 146, true), (146, 161, true), (161, 177, true), (177, 194, true), (194, 214, true), (214, 235, true), (235, 259, true), (259, 285, true), (285, 313, true), (313, 345, true), (345, 379, true), (379, 417, true), (417, 459, true), (459, 505, true), (505, 555, true), (555, 611, true), (611, 672, true), (672, 740, true), (740, 814, true), (814, 895, true), (895, 984, true), (984, 1083, true), (1083, 1191, true), (1191, 1310, true), (1310, 1442, true), (1442, 1586, true), (1586, 1000, false), ];
    bench_layer::<NETWORK2, 1586>("FC_14773663_params.json", 15).unwrap();
    const NETWORK3: NetworkArch = &[(784, 123, true), (123, 135, true), (135, 148, true), (148, 163, true), (163, 180, true), (180, 198, true), (198, 217, true), (217, 239, true), (239, 263, true), (263, 290, true), (290, 319, true), (319, 350, true), (350, 386, true), (386, 424, true), (424, 467, true), (467, 513, true), (513, 565, true), (565, 621, true), (621, 683, true), (683, 752, true), (752, 827, true), (827, 910, true), (910, 1001, true), (1001, 1101, true), (1101, 1211, true), (1211, 1332, true), (1332, 1465, true), (1465, 1612, true), (1612, 1773, true), (1773, 1000, false), ];
    bench_layer::<NETWORK3, 1773>("FC_18252933_params.json", 15).unwrap();
    const NETWORK4: NetworkArch = &[(784, 39, true), (39, 42, true), (42, 47, true), (47, 51, true), (51, 57, true), (57, 62, true), (62, 69, true), (69, 75, true), (75, 83, true), (83, 91, true), (91, 101, true), (101, 111, true), (111, 122, true), (122, 134, true), (134, 148, true), (148, 162, true), (162, 179, true), (179, 197, true), (197, 216, true), (216, 238, true), (238, 262, true), (262, 288, true), (288, 317, true), (317, 349, true), (349, 384, true), (384, 422, true), (422, 464, true), (464, 511, true), (511, 562, true), (562, 1000, false), ];
    bench_layer::<NETWORK4, 784>("FC_2236477_params.json", 13).unwrap();
    const NETWORK5: NetworkArch = &[(784, 55, true), (55, 60, true), (60, 66, true), (66, 73, true), (73, 80, true), (80, 88, true), (88, 97, true), (97, 107, true), (107, 117, true), (117, 129, true), (129, 142, true), (142, 156, true), (156, 172, true), (172, 189, true), (189, 208, true), (208, 229, true), (229, 252, true), (252, 277, true), (277, 305, true), (305, 336, true), (336, 370, true), (370, 407, true), (407, 447, true), (447, 492, true), (492, 541, true), (541, 595, true), (595, 655, true), (655, 721, true), (721, 793, true), (793, 1000, false), ];
    bench_layer::<NETWORK5, 793>("FC_4107399_params.json", 14).unwrap();
    const NETWORK6: NetworkArch = &[(784, 16, true), (16, 17, true), (17, 19, true), (19, 21, true), (21, 23, true), (23, 25, true), (25, 28, true), (28, 31, true), (31, 34, true), (34, 37, true), (37, 41, true), (41, 45, true), (45, 50, true), (50, 55, true), (55, 60, true), (60, 66, true), (66, 73, true), (73, 80, true), (80, 88, true), (88, 97, true), (97, 107, true), (107, 118, true), (118, 130, true), (130, 143, true), (143, 157, true), (157, 173, true), (173, 190, true), (190, 209, true), (209, 230, true), (230, 1000, false), ];
    bench_layer::<NETWORK6, 784>("FC_517529_params.json", 12).unwrap();
    const NETWORK7: NetworkArch = &[(784, 19, true), (19, 20, true), (20, 22, true), (22, 25, true), (25, 27, true), (27, 30, true), (30, 33, true), (33, 37, true), (37, 40, true), (40, 44, true), (44, 49, true), (49, 54, true), (54, 59, true), (59, 65, true), (65, 72, true), (72, 79, true), (79, 87, true), (87, 96, true), (96, 105, true), (105, 116, true), (116, 127, true), (127, 140, true), (140, 154, true), (154, 170, true), (170, 187, true), (187, 205, true), (205, 226, true), (226, 249, true), (249, 273, true), (273, 1000, false), ];
    bench_layer::<NETWORK7, 784>("FC_676836_params.json", 12).unwrap();
    const NETWORK8: NetworkArch = &[(784, 78, true), (78, 85, true), (85, 94, true), (94, 103, true), (103, 114, true), (114, 125, true), (125, 138, true), (138, 151, true), (151, 167, true), (167, 183, true), (183, 202, true), (202, 222, true), (222, 244, true), (244, 269, true), (269, 296, true), (296, 325, true), (325, 358, true), (358, 394, true), (394, 433, true), (433, 477, true), (477, 524, true), (524, 577, true), (577, 634, true), (634, 698, true), (698, 768, true), (768, 845, true), (845, 929, true), (929, 1022, true), (1022, 1124, true), (1124, 1000, false), ];
    bench_layer::<NETWORK8, 1124>("FC_7770136_params.json", 14).unwrap();
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
