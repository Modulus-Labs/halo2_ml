use std::marker::PhantomData;

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter, Value},
    plonk::{
        Advice, Fixed, Column, ConstraintSystem, Error as PlonkError,
        Selector,
    },
    poly::Rotation,
};
use ndarray::{
    Array1, Array2, Array3, Axis, Zip, Array,
};

use crate::nn_ops::{NNLayer, InputSizeConfig, ColumnAllocator};

#[derive(Clone, Debug)]
pub struct DistributedAddMulAddConfig<F: FieldExt> {
    //pub in_width: usize,
    //pub in_height: usize,
    //pub in_depth: usize,
    pub inputs: Array2<Column<Advice>>,
    pub outputs: Array2<Column<Advice>>,
    ///Tuple is (Mult, Add, Bias)
    pub scalars: Array1<(Column<Fixed>, Column<Fixed>, Column<Fixed>)>,
    pub selector: Selector,
    _marker: PhantomData<F>,
}

/// Chip for Distrubted Addition, then Multiplication, then Addition by constants
///
/// Order for ndarrays is Channel-in, Width, Height
/// TODO -> Reduce Column Usage
pub struct DistributedAddMulAddChip<F: FieldExt> {
    config: DistributedAddMulAddConfig<F>,
}

impl<F: FieldExt> Chip<F> for DistributedAddMulAddChip<F> {
    type Config = DistributedAddMulAddConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

pub struct DistributedAddMulAddChipParams<F: FieldExt> {
    pub scalars: Array1<(Value<F>, Value<F>, Value<F>)>,
}

impl<F: FieldExt> NNLayer<F> for DistributedAddMulAddChip<F> {
    type LayerParams = DistributedAddMulAddChipParams<F>;

    type LayerInput = Array3<AssignedCell<F, F>>;

    type LayerOutput = Array3<AssignedCell<F, F>>;

    fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self { config }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        config: InputSizeConfig,
        advice_allocator: &mut ColumnAllocator<Advice>,
        fixed_allocator: &mut ColumnAllocator<Fixed>
        // inputs: Array2<Column<Advice>>,
        // outputs: Array2<Column<Advice>>,
        // scalars: Array1<(Column<Advice>, Column<Advice>, Column<Advice>)>,
    ) -> <Self as Chip<F>>::Config {
        let selector = meta.selector();
        let InputSizeConfig { input_height, input_width, input_depth } = config;
        let advice = advice_allocator.take(meta, input_depth*input_width*2);
        let fixed = fixed_allocator.take(meta, input_depth * 3);

        let inputs = Array::from_shape_vec((input_depth, input_width), advice[0..(input_depth*input_width)].to_vec()).unwrap();
        let outputs = Array::from_shape_vec((input_depth, input_width), advice[(input_depth*input_width)..(input_depth*input_width)*2].to_vec()).unwrap();

        let scalars = Array::from_shape_vec((input_depth, 3), fixed.to_vec()).unwrap();
        let scalars: Array1<_> = scalars.axis_iter(Self::DEPTH_AXIS).map(|scalars| {
            (scalars[0], scalars[1], scalars[2])
        }).collect();

        meta.create_gate("Dist Mult", |meta| {
            let sel = meta.query_selector(selector);
            inputs
                .axis_iter(Self::DEPTH_AXIS)
                .zip(outputs.axis_iter(Self::DEPTH_AXIS))
                .zip(scalars.iter())
                .map(|((inputs, outputs), scalar)| {
                    let mult_scalar = meta.query_fixed(scalar.0, Rotation::cur());
                    let add_scalar = meta.query_fixed(scalar.1, Rotation::cur());
                    let bias_scalar = meta.query_fixed(scalar.2, Rotation::cur());
                    inputs
                        .into_iter()
                        .zip(outputs.into_iter())
                        .map(|(input, output)| {
                            let input = meta.query_advice(*input, Rotation::cur());
                            let output = meta.query_advice(*output, Rotation::cur());
                            sel.clone()
                                * ((input + add_scalar.clone()) * mult_scalar.clone()
                                    + bias_scalar.clone()
                                    - output)
                        })
                        .collect::<Vec<_>>()
                })
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
        });

        DistributedAddMulAddConfig {
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
        inputs: Array3<AssignedCell<F, F>>,
        params: DistributedAddMulAddChipParams<F>,
    ) -> Result<Array3<AssignedCell<F, F>>, PlonkError> {
        let config = &self.config;
        let DistributedAddMulAddChipParams { scalars } = params;

        layouter.assign_region(
            || "Distributed Addition/Multiplication",
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
                                .assign_fixed(|| "Assign Mult Scalar", column.0, row, || scalar.0)
                                .unwrap();
                            region
                                .assign_fixed(|| "Assign Add Scalar", column.1, row, || scalar.1)
                                .unwrap();
                            region
                                .assign_fixed(|| "Assign Bias Scalar", column.2, row, || scalar.2)
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
                        let (mult_scalar, add_scalar, bias_scalar) = scalars.get(channel).unwrap();
                        let output = (*mult_scalar * (*add_scalar + input.value())) + bias_scalar;
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
    use crate::nn_ops::{NNLayer, ColumnAllocator, InputSizeConfig};

    use super::{DistributedAddMulAddChip, DistributedAddMulAddConfig, DistributedAddMulAddChipParams};
    use halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        halo2curves::bn256::Fr,
        plonk::{
            Advice, Circuit, Column, ConstraintSystem, Error as PlonkError, Instance, Fixed,
        },
    };
    use ndarray::{stack, Array, Array1, Array2, Array3, Axis, Zip};

    #[derive(Clone, Debug)]
    struct DistributedAddMulAddTestConfig<F: FieldExt> {
        input: Array2<Column<Instance>>,
        input_advice: Array2<Column<Advice>>,
        output: Array2<Column<Instance>>,
        dist_mul_chip: DistributedAddMulAddConfig<F>,
    }

    struct DistributedAddMulAddTestCircuit<F: FieldExt> {
        pub scalars: Array1<(Value<F>, Value<F>, Value<F>)>,
        pub input: Array3<Value<F>>,
    }

    const INPUT_WIDTH: usize = 16;
    const INPUT_HEIGHT: usize = 16;

    const DEPTH: usize = 4;

    impl<F: FieldExt> Circuit<F> for DistributedAddMulAddTestCircuit<F> {
        type Config = DistributedAddMulAddTestConfig<F>;

        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                scalars: Array::from_shape_simple_fn(DEPTH, || {
                    (Value::unknown(), Value::unknown(), Value::unknown())
                }),
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

            let dist_mul_chip = DistributedAddMulAddChip::configure(meta, config, &mut advice_allocator, &mut fixed_allocator);

            DistributedAddMulAddTestConfig {
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
            let dist_mul_chip = DistributedAddMulAddChip::construct(config.dist_mul_chip);

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

            let params = DistributedAddMulAddChipParams {
                scalars: self.scalars.clone(),
            };

            let output = dist_mul_chip.add_layer(&mut layouter, inputs, params)?;
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
    ///test that a simple 16x16x4 dist add_mult_add works
    fn test_simple_add_mult_add() -> Result<(), PlonkError> {
        let circuit = DistributedAddMulAddTestCircuit {
            scalars: Array::from_shape_simple_fn(DEPTH, || {
                (
                    Value::known(Fr::from(2)),
                    Value::known(Fr::from(1)),
                    Value::known(Fr::from(1)),
                )
            }),
            input: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH, INPUT_HEIGHT), || {
                Value::known(Fr::one())
            }),
        };

        let mut input_instance = vec![vec![Fr::one(); INPUT_HEIGHT]; INPUT_WIDTH * DEPTH];

        let mut output_instance = vec![vec![Fr::from(5); INPUT_HEIGHT]; INPUT_WIDTH * DEPTH];

        input_instance.append(&mut output_instance);

        MockProver::run(7, &circuit, input_instance)
            .unwrap()
            .assert_satisfied();

        Ok(())
    }
}
