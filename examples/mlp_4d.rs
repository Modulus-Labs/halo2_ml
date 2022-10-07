use halo2_machinelearning::{
    nn_chip::{LayerParams},
    NNCircuit,
};
use halo2_proofs::{
    dev::MockProver,
    pasta::{EqAffine, Fp},
    plonk::{create_proof, keygen_pk, keygen_vk},
    poly::commitment::Params,
    transcript::{Blake2bWrite, Challenge255},
};

use std::time::Instant;

use rand::rngs::OsRng;

fn main() -> () {
    let layers = vec![
        LayerParams {
            weights: vec![1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]
                .into_iter()
                .map(|x: i64| {
                    if x >= 0 {
                        Fp::from(x.unsigned_abs())
                    } else {
                        -Fp::from(x.unsigned_abs())
                    }
                })
                .collect(),
            biases: vec![1, 1, 1, 5].into_iter().map(|x| Fp::from(x)).collect(),
        },
        LayerParams {
            weights: vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                .into_iter()
                .map(|x| Fp::from(x))
                .collect(),
            biases: vec![1, 1, 1, 1].into_iter().map(|x| Fp::from(x)).collect(),
        },
    ];

    let input: Vec<Fp> = vec![1, 1, 1, 1].into_iter().map(|x| Fp::from(x)).collect();

    let output: Vec<Fp> = vec![12, 12, 12, 12]
        .into_iter()
        .map(|x| Fp::from(x))
        .collect();

    let circuit = NNCircuit::<Fp> {
        layers,
        input: input.clone(),
    };

    MockProver::run(
        11,
        &circuit,
        vec![input, output]
    ).unwrap().assert_satisfied();

    // let params = Params::<EqAffine>::new(11);

    // let vk = keygen_vk(&params, &circuit).unwrap();

    // let pk = keygen_pk(&params, vk, &circuit).unwrap();

    // let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

    // let now = Instant::now();

    // create_proof(
    //     &params,
    //     &pk,
    //     &[circuit],
    //     &[&[input.as_slice(), output.as_slice()]],
    //     OsRng,
    //     &mut transcript,
    // )
    // .unwrap();

    // println!("Proof took {:?}", now.elapsed().as_secs());
}
