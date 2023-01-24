use halo2_curves::FieldExt;
use halo2_proofs::circuit::{AssignedCell};
use ndarray::Array1;

pub fn gather<F: FieldExt>(inputs: Array1<AssignedCell<F, F>>, index_map: Array1<usize>) -> Array1<AssignedCell<F, F>> {
    index_map.iter().map(|&index| {
        inputs.get(index).unwrap().clone()
    }).collect()
}

#[cfg(test)]
mod tests {
    

    use halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        halo2curves::bn256::Fr,
        plonk::{
            Advice, Circuit, Column, ConstraintSystem, Error as PlonkError, Instance,
        },
    };
    use ndarray::{Array, Array1, Zip};
    use std::marker::PhantomData;

    use super::gather;

    #[derive(Clone, Debug)]
    struct GatherTestConfig<F: FieldExt> {
        input: Column<Instance>,
        input_advice: Column<Advice>,
        output: Column<Instance>,
        _marker: PhantomData<F>,
    }

    struct GatherTestCircuit<F: FieldExt> {
        pub input: Array1<Value<F>>,
        pub gather_indicies: Array1<usize>
    }

    const INPUT_SIZE: usize = 12;
    const OUTPUT_SIZE: usize = 5;

    impl<F: FieldExt> Circuit<F> for GatherTestCircuit<F> {
        type Config = GatherTestConfig<F>;

        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                input: Array::from_shape_simple_fn(INPUT_SIZE, || Value::unknown()),
                gather_indicies: Array::from_shape_fn(OUTPUT_SIZE, |x| x)
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            GatherTestConfig {
                input: {
                    let col = meta.instance_column();
                    meta.enable_equality(col);
                    col
                },
                output: {
                    let col = meta.instance_column();
                    meta.enable_equality(col);
                    col
                },
                input_advice: {
                    let col = meta.advice_column();
                    meta.enable_equality(col);
                    col
                },
                _marker: PhantomData
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), PlonkError> {
            let inputs = layouter.assign_region(
                || "inputs",
                |mut region| {
                    let input_col = config.input;
                    let input_advice = config.input_advice;
                    self.input
                        .iter()
                        .enumerate()
                        .map(|(row, _input)| {
                            region.assign_advice_from_instance(
                                || "assign input",
                                input_col,
                                row,
                                input_advice,
                                row,
                            )
                        })
                        .collect()
                },
            )?;

            let output = gather(inputs, self.gather_indicies.clone());
            for (row, output) in output.iter().enumerate() {
                layouter
                    .constrain_instance(output.cell(), config.output, row)
                    .unwrap();
            }
            Ok(())
        }
    }

    const TEST_OUTPUT: [u64; OUTPUT_SIZE] = [
        2, 8, 4, 1, 7
    ];

    #[test]
    ///test that a simple 8x8 sigmoid works
    fn test_simple_gather() -> Result<(), PlonkError> {
        let input = Array::from_shape_fn(INPUT_SIZE, |x| Fr::from(x as u64));

        let output = Array::from_shape_vec(OUTPUT_SIZE, TEST_OUTPUT.to_vec()).unwrap();
        let gather_indicies = Zip::from(output.view()).map_collect(|&output| output as usize);
        let output = Zip::from(output.view()).map_collect(|&output| Fr::from(output));

        let circuit = GatherTestCircuit {
            input: Zip::from(input.view()).map_collect(|&input| Value::known(input)),
            gather_indicies
        };

        let input_instance: Vec<_> = vec![input.to_vec(), output.to_vec()];

        MockProver::run(11, &circuit, input_instance)
            .unwrap()
            .assert_satisfied();

        Ok(())
    }
}