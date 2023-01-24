use std::marker::PhantomData;

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter},
    plonk::{Advice, Column, ConstraintSystem, Error as PlonkError, Fixed, Selector},
    poly::Rotation,
};
use ndarray::{Array, Array1, Array2, Array3, Zip};

use crate::nn_ops::{ColumnAllocator, InputSizeConfig, NNLayer};

#[derive(Clone, Debug)]
pub struct DistributedAddConfig<F: FieldExt> {
    //pub in_width: usize,
    //pub in_height: usize,
    //pub in_depth: usize,
    pub inputs: Array2<Column<Advice>>,
    pub outputs: Array2<Column<Advice>>,
    pub scalars: Array1<Column<Advice>>,
    pub selector: Selector,
    _marker: PhantomData<F>,
}

/// Chip for Distrubted Addition by a constant
///
/// Order for ndarrays is Channel-in, Width, Height
pub struct DistributedAddChip<F: FieldExt> {
    config: DistributedAddConfig<F>,
}

impl<F: FieldExt> Chip<F> for DistributedAddChip<F> {
    type Config = DistributedAddConfig<F>;
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

impl<F: FieldExt> NNLayer<F> for DistributedAddChip<F> {
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
        let advice = advice_allocator.take(meta, input_depth * input_width * 2 + input_depth);

        let inputs = Array::from_shape_vec(
            (input_depth, input_width),
            advice[0..(input_depth * input_width)].to_vec(),
        )
        .unwrap();
        let outputs = Array::from_shape_vec(
            (input_depth, input_width),
            advice[(input_depth * input_width)..(input_depth * input_width) * 2].to_vec(),
        )
        .unwrap();

        let scalars = Array::from_shape_vec(
            input_depth,
            advice[(input_depth * input_width) * 2..(input_depth * input_width) * 2 + input_depth]
                .to_vec(),
        )
        .unwrap();

        let selector = meta.selector();
        meta.create_gate("Dist Add", |meta| {
            let sel = meta.query_selector(selector);
            inputs
                .axis_iter(Self::DEPTH_AXIS)
                .zip(outputs.axis_iter(Self::DEPTH_AXIS))
                .zip(scalars.iter())
                .map(|((inputs, outputs), scalar)| {
                    let scalar = meta.query_advice(*scalar, Rotation::cur());
                    inputs
                        .into_iter()
                        .zip(outputs.into_iter())
                        .map(|(input, output)| {
                            let input = meta.query_advice(*input, Rotation::cur());
                            let output = meta.query_advice(*output, Rotation::cur());
                            sel.clone() * (input + scalar.clone() - output)
                        })
                        .collect::<Vec<_>>()
                })
                .flatten()
                .collect::<Vec<_>>()
        });

        DistributedAddConfig {
            inputs,
            outputs,
            scalars,
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

        layouter.assign_region(
            || "Distributed Addition",
            |mut region| {
                //Copy Inputs
                inputs
                    .axis_iter(Self::ROW_AXIS)
                    .enumerate()
                    .for_each(|(row, inputs)| {
                        Zip::from(inputs)
                            .and(config.inputs.view())
                            .for_each(|input, &column| {
                                input
                                    .copy_advice(|| "Copy Input", &mut region, column, row)
                                    .unwrap();
                            })
                    });

                //Assign Scalars
                let row_count = inputs.dim().2;

                scalars
                    .iter()
                    .zip(config.scalars.iter())
                    .for_each(|(scalar, &column)| {
                        for row in 0..row_count {
                            // region
                            //     .assign_advice(|| "Assign Scalar", column, row, || scalar)
                            //     .unwrap();
                            scalar
                                .copy_advice(|| "Copy Scalar", &mut region, column, row)
                                .unwrap();
                        }
                    });

                //Assign Outputs
                // Ok(Zip::indexed(inputs.view()).and_broadcast(config.outputs.view()).and_broadcast(scalars.view()).map_collect(|(_, _, row), input, &column, scalar| {
                //     let output = *scalar * input.value();
                //     region.assign_advice(|| "Assign Output", column, row, || output).unwrap()
                // }))
                Ok(
                    Zip::indexed(inputs.view()).map_collect(|(channel, column, row), input| {
                        let output =
                            scalars.get(channel).unwrap().value().map(|f| *f) + input.value();
                        config.selector.enable(&mut region, row).unwrap();
                        region
                            .assign_advice(
                                || "Assign Output",
                                *config.outputs.get((channel, column)).unwrap(),
                                row,
                                || output,
                            )
                            .unwrap()
                    }),
                )
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::nn_ops::{ColumnAllocator, NNLayer};

    use super::{DistributedAddChip, DistributedAddConfig, InputSizeConfig};
    use halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        halo2curves::bn256::Fr,
        plonk::{Advice, Circuit, Column, ConstraintSystem, Error as PlonkError, Fixed, Instance},
    };
    use ndarray::{stack, Array, Array1, Array2, Array3, Axis, Zip};

    #[derive(Clone, Debug)]
    struct DistributedAddTestConfig<F: FieldExt> {
        input: Array2<Column<Instance>>,
        input_advice: Array2<Column<Advice>>,
        scalar_advice: Column<Advice>,
        output: Array2<Column<Instance>>,
        dist_mul_chip: DistributedAddConfig<F>,
    }

    struct DistributedAddTestCircuit<F: FieldExt> {
        pub scalars: Array1<Value<F>>,
        pub input: Array3<Value<F>>,
    }

    const INPUT_WIDTH: usize = 16;
    const INPUT_HEIGHT: usize = 16;

    const DEPTH: usize = 4;

    impl<F: FieldExt> Circuit<F> for DistributedAddTestCircuit<F> {
        type Config = DistributedAddTestConfig<F>;

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

            let dist_mul_chip = DistributedAddChip::configure(
                meta,
                config,
                &mut advice_allocator,
                &mut fixed_allocator,
            );

            DistributedAddTestConfig {
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
            let dist_mul_chip = DistributedAddChip::construct(config.dist_mul_chip);

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
    ///test that a simple 16x16x4 dist add works
    fn test_simple_dist_add() -> Result<(), PlonkError> {
        let circuit = DistributedAddTestCircuit {
            scalars: Array::from_shape_simple_fn(DEPTH, || Value::known(Fr::from(2))),
            input: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH, INPUT_HEIGHT), || {
                Value::known(Fr::one())
            }),
        };

        let mut input_instance = vec![vec![Fr::one(); INPUT_HEIGHT]; INPUT_WIDTH * DEPTH];
        let mut output_instance = vec![vec![Fr::from(3); INPUT_HEIGHT]; INPUT_WIDTH * DEPTH];

        input_instance.append(&mut output_instance);

        MockProver::run(7, &circuit, input_instance)
            .unwrap()
            .assert_satisfied();

        Ok(())
    }
}
