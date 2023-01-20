use std::marker::PhantomData;

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter, Value},
    plonk::{
        Advice, Assigned, Column, ConstraintSystem, Error as PlonkError, Expression, Instance,
        Selector,
    },
    poly::Rotation,
};
use ndarray::{
    concatenate, stack, Array, Array1, Array2, Array3, Array4, ArrayBase, Axis, Dim, Zip,
};

#[derive(Clone, Debug)]
pub struct DistrubutedMulConfig<F: FieldExt> {
    //pub in_width: usize,
    //pub in_height: usize,
    //pub in_depth: usize,
    pub inputs: Array2<Column<Advice>>,
    pub outputs: Array2<Column<Advice>>,
    pub scalars: Array1<Column<Advice>>,
    pub selector: Selector,
    _marker: PhantomData<F>,
}

/// Chip for Distrubted Multiplication by a constant
///
/// Order for ndarrays is Channel-in, Width, Height
pub struct DistrubutedMulChip<F: FieldExt> {
    config: DistrubutedMulConfig<F>,
}

impl<F: FieldExt> Chip<F> for DistrubutedMulChip<F> {
    type Config = DistrubutedMulConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: FieldExt> DistrubutedMulChip<F> {
    const DEPTH_AXIS: Axis = Axis(0);
    const COLUMN_AXIS: Axis = Axis(1);
    const ROW_AXIS: Axis = Axis(2);

    pub fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self { config }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        inputs: Array2<Column<Advice>>,
        outputs: Array2<Column<Advice>>,
        scalars: Array1<Column<Advice>>,
    ) -> <Self as Chip<F>>::Config {
        let selector = meta.selector();
        meta.create_gate("Dist Mult", |meta| {
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
                            sel.clone() * (input * scalar.clone() - output)
                        })
                        .collect::<Vec<_>>()
                })
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
        });

        DistrubutedMulConfig {
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
        inputs: &Array3<AssignedCell<F, F>>,
        scalars: &Array1<Value<F>>,
    ) -> Result<Array3<AssignedCell<F, F>>, PlonkError> {
        let config = &self.config;

        layouter.assign_region(
            || "Distributed Multiplication",
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
                    .for_each(|(&scalar, &column)| {
                        for row in 0..row_count {
                            region
                                .assign_advice(|| "Assign Scalar", column, row, || scalar)
                                .unwrap();
                        }
                    });

                //Assign Outputs
                // Ok(Zip::indexed(inputs.view()).and_broadcast(config.outputs.view()).and_broadcast(scalars.view()).map_collect(|(_, _, row), input, &column, scalar| {
                //     let output = *scalar * input.value();
                //     region.assign_advice(|| "Assign Output", column, row, || output).unwrap()
                // }))
                Ok(
                    Zip::indexed(inputs).map_collect(|(channel, column, row), input| {
                        let output = *scalars.get(channel).unwrap() * input.value();
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
    use super::{DistrubutedMulChip, DistrubutedMulConfig};
    use halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{AssignedCell, Chip, Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        halo2curves::bn256::Fr,
        plonk::{
            Advice, Assigned, Assignment, Circuit, Column, ConstraintSystem, Error as PlonkError,
            Expression, Instance, Selector,
        },
        poly::Rotation,
    };
    use ndarray::{array, stack, Array, Array1, Array2, Array3, Array4, ArrayBase, Axis, Zip};

    #[derive(Clone, Debug)]
    struct DistributedMulTestConfig<F: FieldExt> {
        input: Array2<Column<Instance>>,
        input_advice: Array2<Column<Advice>>,
        output: Array2<Column<Instance>>,
        dist_mul_chip: DistrubutedMulConfig<F>,
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
            let inputs = Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH), || {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            });

            let scalars = Array::from_shape_simple_fn(DEPTH, || {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            });

            //let output_width = INPUT_WIDTH + PADDING_WIDTH * 2 - KERNAL_WIDTH + 1;

            let outputs = Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH), || {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            });

            let dist_mul_chip = DistrubutedMulChip::configure(meta, inputs, outputs, scalars);

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
                dist_mul_chip,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), PlonkError> {
            let dist_mul_chip = DistrubutedMulChip::construct(config.dist_mul_chip);

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
                                    .map_collect(|input, instance, column| {
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

            let output = dist_mul_chip.add_layer(&mut layouter, &inputs, &self.scalars)?;
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
    fn test_simple_conv() -> Result<(), PlonkError> {
        let circuit = DistributedMulTestCircuit {
            scalars: Array::from_shape_simple_fn(DEPTH, || Value::known(Fr::from(2))),
            input: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH, INPUT_HEIGHT), || {
                Value::known(Fr::one())
            }),
        };

        let mut input_instance = vec![vec![Fr::one(); INPUT_HEIGHT]; INPUT_WIDTH * DEPTH];
        //let mut output_instance = vec![vec![Fr::one(); output_height]; output_width*DEPTH];
        // let edge: Vec<_> = vec![
        //     16, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 16,
        // ]
        // .iter()
        // .map(|x| Fr::from(*x))
        // .collect();
        // let row: Vec<_> = vec![
        //     24, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 24,
        // ]
        // .iter()
        // .map(|&x| Fr::from(x))
        // .collect();
        // let layer = vec![
        //     edge.clone(),
        //     row.clone(),
        //     row.clone(),
        //     row.clone(),
        //     row.clone(),
        //     row.clone(),
        //     row.clone(),
        //     row.clone(),
        //     row.clone(),
        //     row.clone(),
        //     row.clone(),
        //     row.clone(),
        //     row.clone(),
        //     row.clone(),
        //     row,
        //     edge,
        // ];
        // let mut output_instance: Vec<_> =
        //     vec![layer.clone(), layer.clone(), layer.clone(), layer.clone()]
        //         .into_iter()
        //         .flatten()
        //         .collect();
        let mut output_instance = vec![vec![Fr::from(2); INPUT_HEIGHT]; INPUT_WIDTH * DEPTH];

        input_instance.append(&mut output_instance);

        MockProver::run(7, &circuit, input_instance)
            .unwrap()
            .assert_satisfied();

        Ok(())
    }
}
