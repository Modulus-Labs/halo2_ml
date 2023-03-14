use std::marker::PhantomData;

use halo2_base::halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter},
    plonk::{Advice, Column, ConstraintSystem, Error as PlonkError, Fixed, Selector},
    poly::Rotation,
};
use ndarray::{Array, Array1, Array2, Array3, Axis, Zip};

use crate::nn_ops::{ColumnAllocator, InputSizeConfig, NNLayer};

#[derive(Clone, Debug)]
pub struct DistributedMulConfig<F: FieldExt> {
    //pub in_width: usize,
    //pub in_height: usize,
    //pub in_depth: usize,
    pub inputs: Array1<Column<Advice>>,
    pub outputs: Array1<Column<Advice>>,
    pub scalar: Column<Advice>,
    pub selector: Selector,
    _marker: PhantomData<F>,
}

/// Chip for Distrubted Addition by a constant
///
/// Order for ndarrays is Channel-in, Width, Height
pub struct DistributedMulChip<F: FieldExt> {
    config: DistributedMulConfig<F>,
}

impl<F: FieldExt> Chip<F> for DistributedMulChip<F> {
    type Config = DistributedMulConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

// pub struct DistributedAddChipParams<F: FieldExt> {
//     pub inputs: Array3<AssignedCell<F, F>>,
//     pub scalars: Array1<AssignedCell<F, F>>,
// }

impl<F: FieldExt> NNLayer<F> for DistributedMulChip<F> {
    type LayerInput = (Array3<AssignedCell<F, F>>, Array1<AssignedCell<F, F>>);

    type LayerOutput = Array3<AssignedCell<F, F>>;

    fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self { config }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        config: InputSizeConfig,
        advice_allocator: &mut ColumnAllocator<Advice>,
        _fixed_allocator: &mut ColumnAllocator<Fixed>,
    ) -> <Self as Chip<F>>::Config {
        let InputSizeConfig {
            input_height: _,
            input_width,
            input_depth,
        } = config;
        let advice = advice_allocator.take(meta, (input_width * 2) + 1);

        let inputs = Array::from_shape_vec(input_width, advice[0..input_width].to_vec()).unwrap();
        let outputs =
            Array::from_shape_vec(input_width, advice[input_width..(input_width * 2)].to_vec())
                .unwrap();

        let scalar = advice[input_width * 2];

        let selector = meta.selector();
        meta.create_gate("Dist Mul", |meta| {
            let sel = meta.query_selector(selector);
            let scalar = meta.query_advice(scalar, Rotation::cur());
            inputs
                .iter()
                .zip(outputs.iter())
                .map(|(&input, &output)| {
                    let input = meta.query_advice(input, Rotation::cur());
                    let output = meta.query_advice(output, Rotation::cur());
                    sel.clone() * (input * scalar.clone() - output)
                })
                .collect::<Vec<_>>()
        });

        DistributedMulConfig {
            inputs,
            outputs,
            scalar,
            selector,
            _marker: PhantomData,
        }
    }

    fn add_layer(
        &self,
        layouter: &mut impl Layouter<F>,
        inputs: (Array3<AssignedCell<F, F>>, Array1<AssignedCell<F, F>>),
        _params: (),
    ) -> Result<Array3<AssignedCell<F, F>>, PlonkError> {
        let config = &self.config;
        let (inputs, scalars) = inputs;
        let input_dim = inputs.dim();

        let outputs_vec: Result<Vec<_>, _> = inputs
            .axis_iter(Self::DEPTH_AXIS)
            .zip(scalars.iter())
            .map(|(inputs, scalar)| {
                layouter.assign_region(
                    || "Distributed Multiplication",
                    |mut region| {
                        //Copy Inputs
                        inputs
                            .axis_iter(Axis(1))
                            .enumerate()
                            .for_each(|(row, inputs)| {
                                Zip::from(inputs).and(config.inputs.view()).for_each(
                                    |input, &column| {
                                        input
                                            .copy_advice(|| "Copy Input", &mut region, column, row)
                                            .unwrap();
                                    },
                                )
                            });

                        //Assign Scalars
                        let row_count = inputs.dim().1;

                        for row in 0..row_count {
                            // region
                            //     .assign_advice(|| "Assign Scalar", column, row, || scalar)
                            //     .unwrap();
                            scalar
                                .copy_advice(|| "Copy Scalar", &mut region, config.scalar, row)
                                .unwrap();
                        }

                        //Assign Outputs
                        // Ok(Zip::indexed(inputs.view()).and_broadcast(config.outputs.view()).and_broadcast(scalars.view()).map_collect(|(_, _, row), input, &column, scalar| {
                        //     let output = *scalar * input.value();
                        //     region.assign_advice(|| "Assign Output", column, row, || output).unwrap()
                        // }))
                        Ok(
                            // Zip::indexed(inputs.view()).map_collect(|(column, row), input| {
                            //     let output =
                            //         scalar.value().map(|x| *x) + input.value();
                            //     config.selector.enable(&mut region, row).unwrap();
                            //     region
                            //         .assign_advice(
                            //             || "Assign Output",
                            //             *config.outputs.get(column).unwrap(),
                            //             row,
                            //             || output,
                            //         )
                            //         .unwrap()
                            // }),
                            inputs
                                .axis_iter(Axis(0))
                                .zip(config.outputs.iter())
                                .map(|(slice, &column)| {
                                    slice
                                        .iter()
                                        .enumerate()
                                        .map(|(row, input)| {
                                            let output = scalar.value().map(|x| *x) * input.value();
                                            config.selector.enable(&mut region, row).unwrap();
                                            region
                                                .assign_advice(
                                                    || "Assign Output",
                                                    column,
                                                    row,
                                                    || output,
                                                )
                                                .unwrap()
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>(),
                        )
                    },
                )
            })
            .collect();
        let outputs_vec = outputs_vec?.into_iter().flatten().flatten().collect();
        Ok(Array::from_shape_vec(input_dim, outputs_vec).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use crate::nn_ops::{ColumnAllocator, DefaultDecomp, NNLayer};

    use super::{DistributedMulChip, DistributedMulConfig, InputSizeConfig};
    use halo2_base::halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        halo2curves::bn256::Fr,
        plonk::{Advice, Circuit, Column, ConstraintSystem, Error as PlonkError, Fixed, Instance},
    };
    use ndarray::{stack, Array, Array1, Array2, Array3, Axis, Zip};

    #[derive(Clone, Debug)]
    struct DistributedMulTestConfig<F: FieldExt> {
        input: Array2<Column<Instance>>,
        input_advice: Array2<Column<Advice>>,
        scalar_advice: Column<Advice>,
        output: Array2<Column<Instance>>,
        dist_mul_chip: DistributedMulConfig<F>,
    }

    struct DistributedMulTestCircuit<F: FieldExt> {
        pub scalars: Array1<Value<F>>,
        pub input: Array3<Value<F>>,
    }

    const INPUT_WIDTH: usize = 16;
    const INPUT_HEIGHT: usize = 16;

    const DEPTH: usize = 4;

    impl<F: FieldExt> Circuit<F> for DistributedMulTestCircuit<F> {
        type Config = DistributedMulTestConfig<F>;

        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                scalars: Array::from_shape_simple_fn(DEPTH, || Value::unknown()),
                input: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH, INPUT_HEIGHT), || {
                    Value::unknown()
                }),
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let mut advice_allocator = ColumnAllocator::<Advice>::new(meta, 1);
            let mut fixed_allocator = ColumnAllocator::<Fixed>::new(meta, 0);

            let config = InputSizeConfig {
                input_height: INPUT_HEIGHT,
                input_width: INPUT_WIDTH,
                input_depth: DEPTH,
            };

            let dist_mul_chip = DistributedMulChip::configure(
                meta,
                config,
                &mut advice_allocator,
                &mut fixed_allocator,
            );

            DistributedMulTestConfig {
                input: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH), || {
                    let col = meta.instance_column();
                    meta.enable_equality(col);
                    col
                }),
                output: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH), || {
                    let col = meta.instance_column();
                    meta.enable_equality(col);
                    col
                }),
                input_advice: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH), || {
                    let col = meta.advice_column();
                    meta.enable_equality(col);
                    col
                }),
                scalar_advice: {
                    let col = meta.advice_column();
                    meta.enable_equality(col);
                    col
                },
                dist_mul_chip,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), PlonkError> {
            let dist_mul_chip = DistributedMulChip::construct(config.dist_mul_chip);

            let inputs = layouter.assign_region(
                || "inputs",
                |mut region| {
                    let input = config.input.view();
                    let input_advice = config.input_advice.view();
                    let result = stack(
                        Axis(2),
                        &self
                            .input
                            .axis_iter(Axis(2))
                            .enumerate()
                            .map(|(row, slice)| {
                                Zip::from(slice.view())
                                    .and(input)
                                    .and(input_advice)
                                    .map_collect(|_input, instance, column| {
                                        region
                                            .assign_advice_from_instance(
                                                || "assign input",
                                                *instance,
                                                row,
                                                *column,
                                                row,
                                            )
                                            .unwrap()
                                    })
                            })
                            .collect::<Vec<_>>()
                            .iter()
                            .map(|x| x.view())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                    Ok(result)
                },
            )?;

            let scalars = layouter.assign_region(
                || "scalar input assignment",
                |mut region| {
                    self.scalars
                        .iter()
                        .enumerate()
                        .map(|(row, &scalar)| {
                            region.assign_advice(
                                || "assign scalar",
                                config.scalar_advice,
                                row,
                                || scalar,
                            )
                        })
                        .collect()
                },
            )?;

            let inputs = (inputs, scalars);

            let output = dist_mul_chip.add_layer(&mut layouter, inputs, ())?;
            let input = config.output.view();
            for (row, slice) in output.axis_iter(Axis(2)).enumerate() {
                Zip::from(slice.view())
                    .and(input)
                    .for_each(|input, column| {
                        layouter
                            .constrain_instance(input.cell(), *column, row)
                            .unwrap();
                    })
            }
            Ok(())
        }
    }

    #[test]
    ///test that a simple 16x16x4 dist mult works
    fn test_simple_dist_mul() -> Result<(), PlonkError> {
        let circuit = DistributedMulTestCircuit {
            scalars: Array::from_shape_simple_fn(DEPTH, || Value::known(Fr::from(2))),
            input: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH, INPUT_HEIGHT), || {
                Value::known(Fr::one())
            }),
        };

        let mut input_instance = vec![vec![Fr::one(); INPUT_HEIGHT]; INPUT_WIDTH * DEPTH];

        let mut output_instance = vec![vec![Fr::from(2); INPUT_HEIGHT]; INPUT_WIDTH * DEPTH];

        input_instance.append(&mut output_instance);

        MockProver::run(7, &circuit, input_instance)
            .unwrap()
            .assert_satisfied();

        Ok(())
    }
}
