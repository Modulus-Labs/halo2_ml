use halo2_machinelearning::{nn_chip::LayerParams, NNCircuit};
use halo2_proofs::{
    dev::MockProver,
    halo2curves::{bn256::Bn256, bn256::Fr},
    plonk::{create_proof, keygen_pk, keygen_vk},
    poly::{
        commitment::{Params, ParamsProver},
        kzg::{commitment::ParamsKZG, multiopen::ProverSHPLONK, multiopen::ProverGWC},
    },
    transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
};

use std::time::Instant;

use rand::rngs::OsRng;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn mlp_4d(c: &mut Criterion) -> () {
    let layers = vec![
        LayerParams {
            weights: vec![1048576; 16]
                .into_iter()
                .map(|x: i64| {
                    if x >= 0 {
                        Fr::from(x.unsigned_abs())
                    } else {
                        -Fr::from(x.unsigned_abs())
                    }
                })
                .collect(),
            biases: vec![1_099_511_627_776; 4]
                .into_iter()
                .map(|x| Fr::from(x))
                .collect(),
        },
        LayerParams {
            weights: vec![1048576; 16].into_iter().map(|x| Fr::from(x)).collect(),
            biases: vec![1_099_511_627_776; 4]
                .into_iter()
                .map(|x| Fr::from(x))
                .collect(),
        },
    ];

    let input: Vec<Fr> = vec![1048576; 4].into_iter().map(|x| Fr::from(x)).collect();

    let output: Vec<Fr> = vec![22020096; 4].into_iter().map(|x| Fr::from(x)).collect();

    let circuit = NNCircuit::<Fr> {
        layers,
        input: input.clone(),
    };

    c.bench_function("MLP_4d Mock Prover", |b| {
        b.iter(|| {
            MockProver::run(11, &circuit, vec![input.clone(), output.clone()])
                .unwrap()
                .assert_satisfied()
        })
    });

    let params: ParamsKZG<Bn256> = ParamsProver::new(11);

    let vk = keygen_vk(&params, &circuit).unwrap();

    let pk = keygen_pk(&params, vk, &circuit).unwrap();

    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

    // let now = Instant::now();

    c.bench_function("MLP_4d Real Prover", |b| {
        b.iter(|| {
            create_proof::<_, ProverGWC<Bn256>, _, _, _, _>(
                &params,
                &pk,
                &[circuit.clone()],
                &[&[input.as_slice(), output.as_slice()]],
                OsRng,
                &mut transcript,
            )
            .unwrap()
        })
    });

    // println!("Proof took {:?}", now.elapsed().as_secs());

    // #[cfg(feature = "dhat-heap")]
    // {
    // let stats = dhat::HeapStats::get();
    // println!("{:?}", stats.max_bytes);
    // }
}

criterion_group!(benches, mlp_4d);
criterion_main!(benches);
