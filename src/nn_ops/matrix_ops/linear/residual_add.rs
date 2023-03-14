use std::marker::PhantomData;

use halo2_base::halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, Error as PlonkError, Expression, Fixed, Selector},
    poly::Rotation,
};
use itertools::Itertools;
use ndarray::{concatenate, stack, Array, Array1, Array2, Array3, Array4, Axis, Zip};

use crate::nn_ops::{ColumnAllocator, InputSizeConfig, NNLayer};

#[derive(Clone, Debug)]
pub struct ResidualAdd2DConfig<F: FieldExt> {
    pub inputs: [Array1<Column<Advice>>; 2],
    pub outputs: Array1<Column<Advice>>,
    pub selector: Selector,
    _marker: PhantomData<F>,
}

///Chip for 2-D Residual Addition
///
/// Order for ndarrays is Channel-in, Width, Height, Channel-out
pub struct ResidualAdd2DChip<F: FieldExt> {
    config: ResidualAdd2DConfig<F>,
}

impl<'a, F: FieldExt> Chip<F> for ResidualAdd2DChip<F> {
    type Config = ResidualAdd2DConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<'a, F: FieldExt> NNLayer<F> for ResidualAdd2DChip<F> {
    type LayerInput = [Array3<AssignedCell<F, F>>; 2];

    type LayerOutput = Array3<AssignedCell<F, F>>;

    fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self { config }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        config_params: InputSizeConfig,
        advice_allocator: &mut ColumnAllocator<Advice>,
        _: &mut ColumnAllocator<Fixed>,
    ) -> <Self as Chip<F>>::Config {
        let InputSizeConfig {
            input_height,
            input_width,
            input_depth,
        } = config_params;

        let advice = advice_allocator.take(meta, input_width * 3);
        let inputs_1 = Array1::from_vec(advice[0..input_width].to_vec());
        let inputs_2 = Array1::from_vec(advice[input_width..input_width * 2].to_vec());
        let outputs = Array1::from_vec(advice[input_width * 2..input_width * 3].to_vec());

        let selector = meta.selector();

        meta.create_gate("Residual Addition", |meta| -> Vec<Expression<F>> {
            let sel = meta.query_selector(selector);
            inputs_1
                .iter()
                .zip(inputs_2.iter())
                .zip(outputs.iter())
                .fold(vec![], |mut accum, ((&input_1, &input_2), &output)| {
                    let input_1 = meta.query_advice(input_1, Rotation::cur());
                    let input_2 = meta.query_advice(input_2, Rotation::cur());
                    let output = meta.query_advice(output, Rotation::cur());

                    accum.push(sel.clone() * (input_1 + input_2 - output));
                    accum
                })
        });

        let inputs = [inputs_1, inputs_2];

        ResidualAdd2DConfig {
            inputs,
            outputs,
            selector,
            _marker: PhantomData,
        }
    }

    fn add_layer(
        &self,
        layouter: &mut impl Layouter<F>,
        inputs: [Array3<AssignedCell<F, F>>; 2],
        _: (),
    ) -> Result<Array3<AssignedCell<F, F>>, PlonkError> {
        let config = &self.config;
        let dim = inputs[0].dim();
        let output: Result<Vec<_>, _> = inputs[0]
            .axis_iter(Self::DEPTH_AXIS)
            .zip(inputs[1].axis_iter(Self::DEPTH_AXIS))
            .map(|(inputs_1, inputs_2)| {
                layouter.assign_region(
                    || "Residual Addition",
                    |mut region| {
                        inputs_1
                            .axis_iter(Axis(0))
                            .zip(inputs_2.axis_iter(Axis(0)))
                            .zip(config.inputs[0].iter())
                            .zip(config.inputs[1].iter())
                            .zip(config.outputs.iter())
                            .map(
                                |((((inputs_1, inputs_2), &column_1), &column_2), &output_col)| {
                                    inputs_1
                                        .iter()
                                        .zip(inputs_2.iter())
                                        .enumerate()
                                        .map(|(row, (input_1, input_2))| {
                                            config.selector.enable(&mut region, row)?;
                                            input_1.copy_advice(
                                                || "Assign Input 1",
                                                &mut region,
                                                column_1,
                                                row,
                                            )?;
                                            input_2.copy_advice(
                                                || "Assign Input 2",
                                                &mut region,
                                                column_2,
                                                row,
                                            )?;

                                            let output =
                                                input_1.value().map(|f| *f) + input_2.value();

                                            region.assign_advice(
                                                || "Assign residual output",
                                                output_col,
                                                row,
                                                || output,
                                            )
                                        })
                                        .collect::<Result<Vec<_>, _>>()
                                },
                            )
                            .collect::<Result<Vec<_>, _>>()
                    },
                )
            })
            .collect();
        let output: Vec<_> = output?.into_iter().flatten().flatten().collect();
        Ok(Array::from_shape_vec(dim, output).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use crate::nn_ops::{ColumnAllocator, DefaultDecomp, InputSizeConfig, NNLayer};

    use super::{ResidualAdd2DChip, ResidualAdd2DConfig};
    use halo2_base::halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        halo2curves::bn256::Fr,
        plonk::{Advice, Circuit, Column, ConstraintSystem, Error as PlonkError, Fixed, Instance},
    };
    use ndarray::{stack, Array, Array2, Array3, Array4, Axis, Zip};

    #[derive(Clone, Debug)]
    struct ResidualAddTestConfig<F: FieldExt> {
        input: Array2<Column<Instance>>,
        input_advice: Array2<Column<Advice>>,
        output: Array2<Column<Instance>>,
        residual_chip: ResidualAdd2DConfig<F>,
    }

    struct ResidualAddTestCircuit<F: FieldExt> {
        pub input: [Array3<Value<F>>; 2],
    }

    const INPUT_WIDTH: usize = 8;
    const INPUT_HEIGHT: usize = 8;

    const DEPTH: usize = 4;

    impl<F: FieldExt> Circuit<F> for ResidualAddTestCircuit<F> {
        type Config = ResidualAddTestConfig<F>;

        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                input: [
                    Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH, INPUT_HEIGHT), || {
                        Value::unknown()
                    }),
                    Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH, INPUT_HEIGHT), || {
                        Value::unknown()
                    }),
                ],
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let mut advice_allocator = ColumnAllocator::<Advice>::new(meta, 2);

            let mut fixed_allocator = ColumnAllocator::<Fixed>::new(meta, 0);

            let config_params = InputSizeConfig {
                input_height: INPUT_HEIGHT,
                input_width: INPUT_WIDTH,
                input_depth: DEPTH,
            };

            let residual_chip = ResidualAdd2DChip::configure(
                meta,
                config_params,
                &mut advice_allocator,
                &mut fixed_allocator,
            );

            ResidualAddTestConfig {
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
                residual_chip,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), PlonkError> {
            let residual_chip = ResidualAdd2DChip::construct(config.residual_chip);

            let mut func = |input_value: Array3<Value<F>>,
                            offset: usize|
             -> Result<Array3<AssignedCell<F, F>>, _> {
                layouter.assign_region(
                    || "inputs",
                    |mut region| {
                        let input = config.input.view();
                        let input_advice = config.input_advice.view();
                        let result = stack(
                            Axis(2),
                            &input_value
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
                                                    offset + row,
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
                )
            };

            let input_1 = func(self.input[0].clone(), 0)?;
            let input_2 = func(self.input[1].clone(), 8)?;

            let output = residual_chip.add_layer(&mut layouter, [input_1, input_2], ())?;
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
    ///test that a simple 4x8x8 residual add works; input are all 1
    fn test_simple_conv() -> Result<(), PlonkError> {
        let circuit = ResidualAddTestCircuit {
            input: [
                Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH, INPUT_HEIGHT), || {
                    Value::known(Fr::one())
                }),
                Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH, INPUT_HEIGHT), || {
                    Value::known(Fr::one())
                }),
            ],
        };

        let mut input_instance = vec![vec![Fr::one(); INPUT_HEIGHT * 2]; DEPTH * INPUT_WIDTH];
        let mut output_instance = vec![vec![Fr::from(2); INPUT_HEIGHT]; DEPTH * INPUT_WIDTH];
        input_instance.append(&mut output_instance);

        MockProver::run(7, &circuit, input_instance)
            .unwrap()
            .assert_satisfied();

        Ok(())
    }
}
