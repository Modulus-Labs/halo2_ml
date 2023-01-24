use halo2_machinelearning::{nn_ops::vector_ops::linear::fc::FcParams, NNCircuit};
use halo2_proofs::{
    circuit::Value,
    dev::MockProver,
    halo2curves::{bn256::Bn256, bn256::Fr},
    plonk::{create_proof, keygen_pk, keygen_vk},
    poly::{
        commitment::{Params, ParamsProver},
        kzg::{commitment::ParamsKZG, multiopen::ProverSHPLONK},
    },
    transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
};

use std::time::Instant;

use rand::rngs::OsRng;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() -> () {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::builder().testing().build();

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

    let input: Vec<Fr> = vec![1048576; 4].into_iter().map(|x| Fr::from(x)).collect();

    let output: Vec<Fr> = vec![22020096; 4].into_iter().map(|x| Fr::from(x)).collect();

    let circuit = NNCircuit::<Fr> {
        layers,
        input: input.clone(),
        output: output.clone(),
    };

    // MockProver::run(11, &circuit, vec![input, output])
    //     .unwrap()
    //     .assert_satisfied();

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
    println!("done!");

    #[cfg(feature = "dhat-heap")]
    {
        let stats = dhat::HeapStats::get();
        println!("{:?}", stats.max_bytes);
    }
}
