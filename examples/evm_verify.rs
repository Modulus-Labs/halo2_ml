use halo2_machinelearning::{nn_ops::vector_ops::linear::fc::FcParams, NNCircuit};
use halo2_proofs::{circuit::Value, poly::commitment::CommitmentScheme};

use halo2_curves::bn256::{Bn256, Fq, Fr, G1Affine};
use halo2_proofs::{
    dev::MockProver,
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, ProvingKey, VerifyingKey},
    poly::{
        commitment::{Params, ParamsProver},
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverGWC, VerifierGWC},
            strategy::AccumulatorStrategy,
        },
        VerificationStrategy,
    },
    transcript::{EncodedChallenge, TranscriptReadBuffer, TranscriptWriterBuffer},
};
use itertools::Itertools;
use rand::rngs::OsRng;
use snark_verifier::{
    loader::{
        evm::{self, encode_calldata, Address, EvmLoader, ExecutorBuilder},
        native::NativeLoader,
    },
    pcs::kzg::{Gwc19, KzgAs, LimbsEncoding},
    system::halo2::{compile, transcript::evm::EvmTranscript, Config},
    verifier::{self, SnarkVerifier},
};
use std::{io::Cursor, rc::Rc};

use std::fs::File;

use std::io::Write;

const LIMBS: usize = 4;
const BITS: usize = 68;

type As = KzgAs<Bn256, Gwc19>;
type PlonkSuccinctVerifier = verifier::plonk::PlonkSuccinctVerifier<As, LimbsEncoding<LIMBS, BITS>>;
type PlonkVerifier = verifier::plonk::PlonkVerifier<As, LimbsEncoding<LIMBS, BITS>>;

mod aggregation {
    use super::{As, PlonkSuccinctVerifier, BITS, LIMBS};
    use halo2_curves::bn256::{Bn256, Fq, Fr, G1Affine};
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{self, Circuit, ConstraintSystem, Error},
        poly::{commitment::ParamsProver, kzg::commitment::ParamsKZG},
    };
    use halo2_wrong_ecc::{
        integer::rns::Rns,
        maingate::{
            MainGate, MainGateConfig, MainGateInstructions, RangeChip, RangeConfig,
            RangeInstructions, RegionCtx,
        },
        EccConfig,
    };
    use itertools::Itertools;
    use rand::rngs::OsRng;
    use snark_verifier::{
        loader::{self, native::NativeLoader},
        pcs::{
            kzg::{KzgAccumulator, KzgSuccinctVerifyingKey, LimbsEncodingInstructions},
            AccumulationScheme, AccumulationSchemeProver,
        },
        system,
        util::arithmetic::{fe_to_limbs, FieldExt},
        verifier::{plonk::PlonkProtocol, SnarkVerifier},
    };
    use std::rc::Rc;

    const T: usize = 5;
    const RATE: usize = 4;
    const R_F: usize = 8;
    const R_P: usize = 60;

    type Svk = KzgSuccinctVerifyingKey<G1Affine>;
    type BaseFieldEccChip = halo2_wrong_ecc::BaseFieldEccChip<G1Affine, LIMBS, BITS>;
    type Halo2Loader<'a> = loader::halo2::Halo2Loader<'a, G1Affine, BaseFieldEccChip>;
    pub type PoseidonTranscript<L, S> =
        system::halo2::transcript::halo2::PoseidonTranscript<G1Affine, L, S, T, RATE, R_F, R_P>;

    pub struct Snark {
        protocol: PlonkProtocol<G1Affine>,
        instances: Vec<Vec<Fr>>,
        proof: Vec<u8>,
    }

    impl Snark {
        pub fn new(
            protocol: PlonkProtocol<G1Affine>,
            instances: Vec<Vec<Fr>>,
            proof: Vec<u8>,
        ) -> Self {
            Self {
                protocol,
                instances,
                proof,
            }
        }
    }

    impl From<Snark> for SnarkWitness {
        fn from(snark: Snark) -> Self {
            Self {
                protocol: snark.protocol,
                instances: snark
                    .instances
                    .into_iter()
                    .map(|instances| instances.into_iter().map(Value::known).collect_vec())
                    .collect(),
                proof: Value::known(snark.proof),
            }
        }
    }

    #[derive(Clone)]
    pub struct SnarkWitness {
        protocol: PlonkProtocol<G1Affine>,
        instances: Vec<Vec<Value<Fr>>>,
        proof: Value<Vec<u8>>,
    }

    impl SnarkWitness {
        fn without_witnesses(&self) -> Self {
            SnarkWitness {
                protocol: self.protocol.clone(),
                instances: self
                    .instances
                    .iter()
                    .map(|instances| vec![Value::unknown(); instances.len()])
                    .collect(),
                proof: Value::unknown(),
            }
        }

        fn proof(&self) -> Value<&[u8]> {
            self.proof.as_ref().map(Vec::as_slice)
        }
    }

    pub fn aggregate<'a>(
        svk: &Svk,
        loader: &Rc<Halo2Loader<'a>>,
        snarks: &[SnarkWitness],
        as_proof: Value<&'_ [u8]>,
    ) -> KzgAccumulator<G1Affine, Rc<Halo2Loader<'a>>> {
        let assign_instances = |instances: &[Vec<Value<Fr>>]| {
            instances
                .iter()
                .map(|instances| {
                    instances
                        .iter()
                        .map(|instance| loader.assign_scalar(*instance))
                        .collect_vec()
                })
                .collect_vec()
        };

        let accumulators = snarks
            .iter()
            .flat_map(|snark| {
                let protocol = snark.protocol.loaded(loader);
                let instances = assign_instances(&snark.instances);
                let mut transcript =
                    PoseidonTranscript::<Rc<Halo2Loader>, _>::new(loader, snark.proof());
                let proof =
                    PlonkSuccinctVerifier::read_proof(svk, &protocol, &instances, &mut transcript)
                        .unwrap();
                PlonkSuccinctVerifier::verify(svk, &protocol, &instances, &proof).unwrap()
            })
            .collect_vec();

        let acccumulator = {
            let mut transcript = PoseidonTranscript::<Rc<Halo2Loader>, _>::new(loader, as_proof);
            let proof =
                As::read_proof(&Default::default(), &accumulators, &mut transcript).unwrap();
            As::verify(&Default::default(), &accumulators, &proof).unwrap()
        };

        acccumulator
    }

    #[derive(Clone)]
    pub struct AggregationConfig {
        main_gate_config: MainGateConfig,
        range_config: RangeConfig,
    }

    impl AggregationConfig {
        pub fn configure<F: FieldExt>(
            meta: &mut ConstraintSystem<F>,
            composition_bits: Vec<usize>,
            overflow_bits: Vec<usize>,
        ) -> Self {
            let main_gate_config = MainGate::<F>::configure(meta);
            let range_config =
                RangeChip::<F>::configure(meta, &main_gate_config, composition_bits, overflow_bits);
            AggregationConfig {
                main_gate_config,
                range_config,
            }
        }

        pub fn main_gate(&self) -> MainGate<Fr> {
            MainGate::new(self.main_gate_config.clone())
        }

        pub fn range_chip(&self) -> RangeChip<Fr> {
            RangeChip::new(self.range_config.clone())
        }

        pub fn ecc_chip(&self) -> BaseFieldEccChip {
            BaseFieldEccChip::new(EccConfig::new(
                self.range_config.clone(),
                self.main_gate_config.clone(),
            ))
        }
    }

    #[derive(Clone)]
    pub struct AggregationCircuit {
        svk: Svk,
        snarks: Vec<SnarkWitness>,
        instances: Vec<Fr>,
        as_proof: Value<Vec<u8>>,
    }

    impl AggregationCircuit {
        pub fn new(params: &ParamsKZG<Bn256>, snarks: impl IntoIterator<Item = Snark>) -> Self {
            let svk = params.get_g()[0].into();
            let snarks = snarks.into_iter().collect_vec();

            let accumulators = snarks
                .iter()
                .flat_map(|snark| {
                    let mut transcript =
                        PoseidonTranscript::<NativeLoader, _>::new(snark.proof.as_slice());
                    let proof = PlonkSuccinctVerifier::read_proof(
                        &svk,
                        &snark.protocol,
                        &snark.instances,
                        &mut transcript,
                    )
                    .unwrap();
                    PlonkSuccinctVerifier::verify(&svk, &snark.protocol, &snark.instances, &proof)
                        .unwrap()
                })
                .collect_vec();

            let (accumulator, as_proof) = {
                let mut transcript = PoseidonTranscript::<NativeLoader, _>::new(Vec::new());
                let accumulator =
                    As::create_proof(&Default::default(), &accumulators, &mut transcript, OsRng)
                        .unwrap();
                (accumulator, transcript.finalize())
            };

            let KzgAccumulator { lhs, rhs } = accumulator;
            let instances = [lhs.x, lhs.y, rhs.x, rhs.y]
                .map(fe_to_limbs::<_, _, LIMBS, BITS>)
                .concat();

            Self {
                svk,
                snarks: snarks.into_iter().map_into().collect(),
                instances,
                as_proof: Value::known(as_proof),
            }
        }

        pub fn accumulator_indices() -> Vec<(usize, usize)> {
            (0..4 * LIMBS).map(|idx| (0, idx)).collect()
        }

        pub fn num_instance() -> Vec<usize> {
            vec![4 * LIMBS]
        }

        pub fn instances(&self) -> Vec<Vec<Fr>> {
            vec![self.instances.clone()]
        }

        pub fn as_proof(&self) -> Value<&[u8]> {
            self.as_proof.as_ref().map(Vec::as_slice)
        }
    }

    impl Circuit<Fr> for AggregationCircuit {
        type Config = AggregationConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                svk: self.svk,
                snarks: self
                    .snarks
                    .iter()
                    .map(SnarkWitness::without_witnesses)
                    .collect(),
                instances: Vec::new(),
                as_proof: Value::unknown(),
            }
        }

        fn configure(meta: &mut plonk::ConstraintSystem<Fr>) -> Self::Config {
            AggregationConfig::configure(
                meta,
                vec![BITS / LIMBS],
                Rns::<Fq, Fr, LIMBS, BITS>::construct().overflow_lengths(),
            )
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), plonk::Error> {
            let main_gate = config.main_gate();
            let range_chip = config.range_chip();

            range_chip.load_table(&mut layouter)?;

            let accumulator_limbs = layouter.assign_region(
                || "",
                |region| {
                    let ctx = RegionCtx::new(region, 0);

                    let ecc_chip = config.ecc_chip();
                    let loader = Halo2Loader::new(ecc_chip, ctx);
                    let accumulator = aggregate(&self.svk, &loader, &self.snarks, self.as_proof());

                    let accumulator_limbs = [accumulator.lhs, accumulator.rhs]
                        .iter()
                        .map(|ec_point| {
                            loader.ecc_chip().assign_ec_point_to_limbs(
                                &mut loader.ctx_mut(),
                                ec_point.assigned(),
                            )
                        })
                        .collect::<Result<Vec<_>, Error>>()?
                        .into_iter()
                        .flatten();

                    Ok(accumulator_limbs)
                },
            )?;

            for (row, limb) in accumulator_limbs.enumerate() {
                main_gate.expose_public(layouter.namespace(|| ""), limb, row)?;
            }

            Ok(())
        }
    }
}

fn gen_srs(k: u32) -> ParamsKZG<Bn256> {
    ParamsKZG::<Bn256>::setup(k, OsRng)
}

fn gen_pk<C: Circuit<Fr>>(params: &ParamsKZG<Bn256>, circuit: &C) -> ProvingKey<G1Affine> {
    let vk = keygen_vk(params, circuit).unwrap();
    keygen_pk(params, vk, circuit).unwrap()
}

fn gen_proof<
    C: Circuit<Fr>,
    E: EncodedChallenge<G1Affine>,
    TR: TranscriptReadBuffer<Cursor<Vec<u8>>, G1Affine, E>,
    TW: TranscriptWriterBuffer<Vec<u8>, G1Affine, E>,
>(
    params: &ParamsKZG<Bn256>,
    pk: &ProvingKey<G1Affine>,
    circuit: C,
    instances: Vec<Vec<Fr>>,
) -> Vec<u8> {
    MockProver::run(params.k(), &circuit, instances.clone())
        .unwrap()
        .assert_satisfied();

    let instances = instances
        .iter()
        .map(|instances| instances.as_slice())
        .collect_vec();
    let proof = {
        let mut transcript = TW::init(Vec::new());
        create_proof::<KZGCommitmentScheme<Bn256>, ProverGWC<_>, _, _, TW, _>(
            params,
            pk,
            &[circuit],
            &[instances.as_slice()],
            OsRng,
            &mut transcript,
        )
        .unwrap();
        transcript.finalize()
    };

    let accept = {
        let mut transcript = TR::init(Cursor::new(proof.clone()));
        VerificationStrategy::<_, VerifierGWC<_>>::finalize(
            verify_proof::<_, VerifierGWC<_>, _, TR, _>(
                params.verifier_params(),
                pk.get_vk(),
                AccumulatorStrategy::new(params.verifier_params()),
                &[instances.as_slice()],
                &mut transcript,
            )
            .unwrap(),
        )
    };
    assert!(accept);

    proof
}

fn gen_application_snark(params: &ParamsKZG<Bn256>, circuit: &NNCircuit<Fr>) -> aggregation::Snark {
    let pk = gen_pk(params, circuit);
    let protocol = compile(
        params,
        pk.get_vk(),
        Config::kzg().with_num_instance(circuit.num_instances()),
    );

    let proof = gen_proof::<
        _,
        _,
        aggregation::PoseidonTranscript<NativeLoader, _>,
        aggregation::PoseidonTranscript<NativeLoader, _>,
    >(params, &pk, circuit.clone(), circuit.instances());
    aggregation::Snark::new(protocol, circuit.instances(), proof)
}

fn gen_aggregation_evm_verifier(
    params: &ParamsKZG<Bn256>,
    vk: &VerifyingKey<G1Affine>,
    num_instance: Vec<usize>,
    accumulator_indices: Vec<(usize, usize)>,
) -> Vec<u8> {
    let protocol = compile(
        params,
        vk,
        Config::kzg()
            .with_num_instance(num_instance.clone())
            .with_accumulator_indices(Some(accumulator_indices)),
    );
    let vk = (params.get_g()[0], params.g2(), params.s_g2()).into();

    let loader = EvmLoader::new::<Fq, Fr>();
    let protocol = protocol.loaded(&loader);
    let mut transcript = EvmTranscript::<_, Rc<EvmLoader>, _, _>::new(&loader);

    let instances = transcript.load_instances(num_instance);
    let proof = PlonkVerifier::read_proof(&vk, &protocol, &instances, &mut transcript).unwrap();
    PlonkVerifier::verify(&vk, &protocol, &instances, &proof).unwrap();
    //println!("uncompiled code: {:?}", &loader.yul_code());

    evm::compile_yul(&loader.yul_code())
}

fn evm_verify(deployment_code: Vec<u8>, instances: Vec<Vec<Fr>>, proof: Vec<u8>) {
    let calldata = encode_calldata(&instances, &proof);
    let mut f = File::create("calldata").unwrap();

    f.write_all(calldata.as_slice()).unwrap();

    println!("call data is: {calldata:?}");
    let success = {
        let mut evm = ExecutorBuilder::default()
            .with_gas_limit(u64::MAX.into())
            .build();

        let caller = Address::from_low_u64_be(0xfe);
        let verifier = evm
            .deploy(caller, deployment_code.into(), 0.into())
            .address
            .unwrap();
        let result = evm.call_raw(caller, verifier, calldata.into(), 0.into());

        dbg!(result.gas_used);

        !result.reverted
    };
    assert!(success);
}

// fn main() {
//     let params = gen_srs(22);
//     let params_app = {
//         let mut params = params.clone();
//         params.downsize(8);
//         params
//     };

//     let snarks = [(); 3].map(|_| gen_application_snark(&params_app));
//     let agg_circuit = aggregation::AggregationCircuit::new(&params, snarks);
//     let pk = gen_pk(&params, &agg_circuit);
//     let deployment_code = gen_aggregation_evm_verifier(
//         &params,
//         pk.get_vk(),
//         aggregation::AggregationCircuit::num_instance(),
//         aggregation::AggregationCircuit::accumulator_indices(),
//     );

//     let proof = gen_proof::<_, _, EvmTranscript<G1Affine, _, _, _>, EvmTranscript<G1Affine, _, _, _>>(
//         &params,
//         &pk,
//         agg_circuit.clone(),
//         agg_circuit.instances(),
//     );
//     evm_verify(deployment_code, agg_circuit.instances(), proof);
// }

fn main() {
    let layers = vec![
        FcParams {
            weights: vec![1048576; 16]
                .into_iter()
                .map(|x: i64| {
                    if x >= 0 {
                        Value::known(Fr::from(x.unsigned_abs()))
                    } else {
                        Value::known(-Fr::from(x.unsigned_abs()))
                    }
                })
                .collect(),
            biases: vec![1_099_511_627_776; 4]
                .into_iter()
                .map(|x| Value::known(Fr::from(x)))
                .collect(),
        },
        FcParams {
            weights: vec![1048576; 16]
                .into_iter()
                .map(|x| Value::known(Fr::from(x)))
                .collect(),
            biases: vec![1_099_511_627_776; 4]
                .into_iter()
                .map(|x| Value::known(Fr::from(x)))
                .collect(),
        },
    ];

    let input: Vec<Fr> = vec![1048576; 4].into_iter().map(Fr::from).collect();

    let output: Vec<Fr> = vec![22020096; 4].into_iter().map(Fr::from).collect();

    let circuit = NNCircuit::<Fr> {
        layers,
        input: input,
        output: output,
    };

    // MockProver::run(11, &circuit, vec![input, output])
    //     .unwrap()
    //     .assert_satisfied();

    //let params: ParamsKZG<Bn256> = ParamsProver::new(11);

    //let vk = keygen_vk(&params, &circuit).unwrap();

    //let pk = keygen_pk(&params, vk, &circuit).unwrap();

    //let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

    //let now = Instant::now();

    // create_proof::<_, ProverSHPLONK<Bn256>, _, _, _, _>(
    //     &params,
    //     &pk,
    //     &[circuit],
    //     &[&[input.as_slice(), output.as_slice()]],
    //     OsRng,
    //     &mut transcript,
    // )
    // .unwrap();
    let mut f = File::open("KZGParams").unwrap();
    let params = KZGCommitmentScheme::read_params(&mut f).unwrap();

    let params_app = {
        let mut params = params.clone();
        params.downsize(11);
        params
    };

    let snarks = [(); 1].map(|_| gen_application_snark(&params_app, &circuit));
    println!("Generated Snarks!");
    let agg_circuit = aggregation::AggregationCircuit::new(&params, snarks);
    let pk = gen_pk(&params, &agg_circuit);
    println!("PK Generated!");
    let deployment_code = gen_aggregation_evm_verifier(
        &params,
        pk.get_vk(),
        aggregation::AggregationCircuit::num_instance(),
        aggregation::AggregationCircuit::accumulator_indices(),
    );

    let mut f = File::create("contract").unwrap();

    f.write_all(deployment_code.as_slice()).unwrap();

    println!("Deployement code generated");

    let proof = gen_proof::<_, _, EvmTranscript<G1Affine, _, _, _>, EvmTranscript<G1Affine, _, _, _>>(
        &params,
        &pk,
        agg_circuit.clone(),
        agg_circuit.instances(),
    );
    println!("Proof Generated!");
    evm_verify(deployment_code, agg_circuit.instances(), proof);
    println!("Evm Verified Proof!");
}
