use std::marker::PhantomData;

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, Error as PlonkError, Expression, Fixed, Selector},
    poly::Rotation,
};
use ndarray::{
    concatenate, stack, Array, Array1, Array2, Array3, Array4, ArrayView3, ArrayView4, Axis, Zip,
};

use crate::nn_ops::{ColumnAllocator, DecompConfig, NNLayer};

#[derive(Clone, Debug)]
pub struct Conv3DLayerConfig<F: FieldExt> {
    //pub in_width: usize,
    //pub in_height: usize,
    //pub in_depth: usize,
    pub ker_width: usize,
    pub ker_height: usize,
    pub padding_width: usize,
    pub padding_height: usize,
    pub inputs: Array2<Column<Advice>>,
    pub outputs: Array1<Column<Advice>>,
    pub kernals: Array2<Column<Fixed>>,
    pub conv_selectors: Vec<Selector>,
    _marker: PhantomData<F>,
}

///Chip for 2-D Convolution
///
/// Order for ndarrays is Channel-in, Width, Height, Channel-out
pub struct Conv3DLayerChip<F: FieldExt> {
    config: Conv3DLayerConfig<F>,
}

impl<'a, F: FieldExt> Chip<F> for Conv3DLayerChip<F> {
    type Config = Conv3DLayerConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

#[derive(Clone, Debug)]
enum InputOrPadding<F: FieldExt> {
    Input(AssignedCell<F, F>),
    Padding,
}

impl<F: FieldExt> InputOrPadding<F> {
    fn value(&self) -> Value<F> {
        match self {
            InputOrPadding::Input(x) => x.value().map(|x| *x),
            InputOrPadding::Padding => Value::known(F::zero()),
        }
    }
}

pub struct Conv3DLayerConfigParams {
    input_height: usize,
    input_width: usize,
    input_depth: usize,
    ker_height: usize,
    ker_width: usize,
    c_out_size: usize,
    padding_width: usize,
    padding_height: usize,
}

pub struct Conv3DLayerParams<F: FieldExt> {
    kernals: Array4<Value<F>>,
}

impl<'a, F: FieldExt> NNLayer<F> for Conv3DLayerChip<F> {
    const C_OUT_AXIS: Axis = Axis(3);

    type LayerInput = Array3<AssignedCell<F, F>>;

    type ConfigParams = Conv3DLayerConfigParams;
    type LayerParams = Conv3DLayerParams<F>;

    type LayerOutput = Array3<AssignedCell<F, F>>;

    fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self { config }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        config_params: Conv3DLayerConfigParams,
        advice_allocator: &mut ColumnAllocator<Advice>,
        fixed_allocator: &mut ColumnAllocator<Fixed>,
    ) -> <Self as Chip<F>>::Config {
        let Conv3DLayerConfigParams {
            input_height: _,
            input_width,
            input_depth,
            ker_height,
            ker_width,
            padding_width,
            padding_height,
            c_out_size,
        } = config_params;
        let output_width = input_width + padding_width * 2 - ker_width + 1;
        let input_column_count = (input_width + padding_width * 2) * input_depth;
        let output_column_count = output_width;
        let advice_count = input_column_count + output_column_count;
        let fixed_count = ker_width * input_depth;

        let advice = advice_allocator.take(meta, advice_count);
        let fixed = fixed_allocator.take(meta, fixed_count);

        let inputs = Array::from_shape_vec(
            (input_depth, (input_width + padding_width * 2)),
            advice[0..input_column_count].to_vec(),
        )
        .unwrap();
        let outputs = Array::from_shape_vec(
            output_width,
            advice[input_column_count..input_column_count + output_column_count].to_vec(),
        )
        .unwrap();

        let kernals = Array::from_shape_vec((input_depth, ker_width), fixed.to_vec()).unwrap();
        let ker_height_i32: i32 = ker_height.try_into().unwrap();
        let selectors: Vec<Selector> = (0..ker_height_i32)
            .into_iter()
            .map(|offset| {
                let selector = meta.selector();
                meta.create_gate("Conv", |meta| -> Vec<Expression<F>> {
                    let sel = meta.query_selector(selector);

                    //Stack input column along the inner-most axis to represent rows
                    let inputs: Array3<Expression<F>> = stack(
                        Self::ROW_AXIS,
                        &(0..ker_height_i32)
                            .into_iter()
                            .map(|index| {
                                inputs.map(|column| meta.query_advice(*column, Rotation(index)))
                            })
                            .collect::<Vec<_>>()
                            .iter()
                            .map(|item| item.view())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();

                    let kernals: Array3<Expression<F>> = stack(
                        Self::ROW_AXIS,
                        &(0..ker_height_i32)
                            .into_iter()
                            .map(|index| {
                                kernals.map(|column| {
                                    meta.query_fixed(*column, Rotation(index - offset))
                                })
                            })
                            .collect::<Vec<_>>()
                            .iter()
                            .map(|item| item.view())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();

                    let outputs = outputs
                        .iter()
                        .map(|column| meta.query_advice(*column, Rotation::cur()));

                    //window over the columns, and then add everything together
                    //we never need to window over depth because the conv filter is the same depth as the image
                    //we don't window over the rows because this gate only affects a small subset of rows
                    let out_constraints = inputs
                        .axis_windows(Self::COLUMN_AXIS, ker_width)
                        .into_iter()
                        .map(|item| {
                            item.iter()
                                .zip(kernals.iter())
                                .fold(Expression::Constant(F::zero()), |accum, (input, kernal)| {
                                    accum + input.clone() * kernal.clone()
                                })
                        });

                    //enforce equality to output
                    out_constraints
                        .zip(outputs)
                        .map(|(constraint, output)| sel.clone() * (constraint - output))
                        .collect()
                });
                selector
            })
            .collect();
        Conv3DLayerConfig {
            ker_width,
            ker_height,
            padding_width,
            padding_height,
            inputs,
            outputs,
            kernals,
            conv_selectors: selectors,
            _marker: PhantomData,
        }
    }

    fn add_layer(
        &self,
        layouter: &mut impl Layouter<F>,
        inputs: Array3<AssignedCell<F, F>>,
        layer_params: Conv3DLayerParams<F>,
    ) -> Result<Array3<AssignedCell<F, F>>, PlonkError> {
        let Conv3DLayerParams { kernals } = layer_params;

        let config = &self.config;

        //add padding
        let dims = inputs.dim();
        let inputs = inputs.map(|x| InputOrPadding::Input(x.clone()));
        let padding_horizontal =
            Array::from_shape_simple_fn((dims.0, config.padding_width, dims.2), || {
                InputOrPadding::Padding
            });
        let padding_vertical = Array::from_shape_simple_fn(
            (
                dims.0,
                dims.1 + config.padding_width * 2,
                config.padding_height,
            ),
            || InputOrPadding::<F>::Padding,
        );
        let inputs = concatenate(
            Self::COLUMN_AXIS,
            &[
                padding_horizontal.view(),
                inputs.view(),
                padding_horizontal.view(),
            ],
        )
        .unwrap();
        let inputs = concatenate(
            Self::ROW_AXIS,
            &[
                padding_vertical.view(),
                inputs.view(),
                padding_vertical.view(),
            ],
        )
        .unwrap();

        let outputs = kernals
            .axis_iter(Self::C_OUT_AXIS)
            .map(|kernals| {
                layouter.assign_region(
                    || "Conv Layer",
                    |mut region| {
                        //assign inputs and the kernal
                        inputs
                            .axis_iter(Self::ROW_AXIS)
                            .enumerate()
                            .for_each(|(row, inputs)| {
                                Zip::from(inputs).and(config.inputs.view()).for_each(
                                    |input, column| match input {
                                        InputOrPadding::Input(x) => {
                                            x.copy_advice(
                                                || "Copy Input",
                                                &mut region,
                                                *column,
                                                row,
                                            )
                                            .unwrap();
                                        }
                                        InputOrPadding::Padding => {
                                            region
                                                .assign_advice(
                                                    || "Add Padding",
                                                    *column,
                                                    row,
                                                    || Value::known(F::zero()),
                                                )
                                                .unwrap();
                                        }
                                    },
                                );
                            });

                        let mut offset = 0;
                        while offset + config.ker_height < inputs.dim().2 {
                            kernals.axis_iter(Self::ROW_AXIS).enumerate().for_each(
                                |(row, kernal)| {
                                    config.conv_selectors[row]
                                        .enable(&mut region, offset + row)
                                        .unwrap();
                                    Zip::from(kernal).and(config.kernals.view()).for_each(
                                        |kernal, column| {
                                            region
                                                .assign_fixed(
                                                    || "place kernal",
                                                    *column,
                                                    row + offset,
                                                    || *kernal,
                                                )
                                                .unwrap();
                                        },
                                    );
                                },
                            );
                            offset += config.ker_height;
                        }

                        //calculate outputs
                        let outputs =
                            Zip::from(inputs.windows(kernals.dim())).map_collect(|inputs| {
                                inputs
                                    .iter()
                                    .zip(kernals.iter())
                                    .fold(Value::known(F::zero()), |accum, (item, kernal)| {
                                        accum + (*kernal * item.value())
                                    })
                            });

                        debug_assert_eq!(
                            1,
                            outputs.dim().0,
                            "Layers should be reduced to a single layer output"
                        );

                        let outputs = outputs.remove_axis(Self::DEPTH_AXIS);
                        //place outputs
                        Ok(stack(
                            Axis(1),
                            outputs
                                .axis_iter(Axis(1))
                                .enumerate()
                                .map(|(row, outputs)| {
                                    Zip::from(outputs.view())
                                        .and(config.outputs.view())
                                        .map_collect(|output, column| {
                                            region
                                                .assign_advice(
                                                    || "conv outputs",
                                                    *column,
                                                    row,
                                                    || *output,
                                                )
                                                .unwrap()
                                        })
                                })
                                .collect::<Vec<_>>()
                                .iter()
                                .map(|item| item.view())
                                .collect::<Vec<_>>()
                                .as_slice(),
                        )
                        .unwrap())
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(stack(
            Axis(0),
            outputs
                .iter()
                .map(|x| x.view())
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .unwrap())
    }
}

#[cfg(test)]
mod tests {
    use crate::nn_ops::{ColumnAllocator, DefaultDecomp, NNLayer};

    use super::{Conv3DLayerChip, Conv3DLayerConfig, Conv3DLayerConfigParams, Conv3DLayerParams};
    use halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        halo2curves::bn256::Fr,
        plonk::{Advice, Circuit, Column, ConstraintSystem, Error as PlonkError, Fixed, Instance},
    };
    use ndarray::{stack, Array, Array2, Array3, Array4, Axis, Zip};

    #[derive(Clone, Debug)]
    struct ConvTestConfig<F: FieldExt> {
        input: Array2<Column<Instance>>,
        input_advice: Array2<Column<Advice>>,
        output: Array2<Column<Instance>>,
        conv_chip: Conv3DLayerConfig<F>,
    }

    struct ConvTestCircuit<F: FieldExt> {
        pub kernal: Array4<Value<F>>,
        pub input: Array3<Value<F>>,
    }

    const INPUT_WIDTH: usize = 16;
    const INPUT_HEIGHT: usize = 16;

    const KERNAL_WIDTH: usize = 3;
    const KERNAL_HEIGHT: usize = 3;

    const DEPTH: usize = 4;

    const PADDING_WIDTH: usize = 1;
    const PADDING_HEIGHT: usize = 1;

    impl<F: FieldExt> Circuit<F> for ConvTestCircuit<F> {
        type Config = ConvTestConfig<F>;

        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                kernal: Array::from_shape_simple_fn(
                    (DEPTH, KERNAL_WIDTH, KERNAL_HEIGHT, DEPTH),
                    || Value::unknown(),
                ),
                input: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH, INPUT_HEIGHT), || {
                    Value::unknown()
                }),
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            // let inputs =
            //     Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH + PADDING_WIDTH * 2), || {
            //         let col = meta.advice_column();
            //         meta.enable_equality(col);
            //         col
            //     });

            // let kernals = Array::from_shape_simple_fn((DEPTH, KERNAL_WIDTH), || {
            //     let col = meta.advice_column();
            //     meta.enable_equality(col);
            //     col
            // });

            let output_width = INPUT_WIDTH + PADDING_WIDTH * 2 - KERNAL_WIDTH + 1;

            // let outputs = Array::from_shape_simple_fn(output_width, || {
            //     let col = meta.advice_column();
            //     meta.enable_equality(col);
            //     col
            // });

            let mut advice_allocator = ColumnAllocator::<Advice>::new(meta, 2);

            let mut fixed_allocator = ColumnAllocator::<Fixed>::new(meta, 2);

            let config_params = Conv3DLayerConfigParams {
                input_height: INPUT_HEIGHT,
                input_width: INPUT_WIDTH,
                input_depth: DEPTH,
                ker_height: KERNAL_HEIGHT,
                ker_width: KERNAL_WIDTH,
                c_out_size: DEPTH,
                padding_width: PADDING_WIDTH,
                padding_height: PADDING_HEIGHT,
            };

            let conv_chip = Conv3DLayerChip::configure(
                meta,
                config_params,
                &mut advice_allocator,
                &mut fixed_allocator,
            );

            ConvTestConfig {
                input: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH), || {
                    let col = meta.instance_column();
                    meta.enable_equality(col);
                    col
                }),
                output: Array::from_shape_simple_fn((DEPTH, output_width), || {
                    let col = meta.instance_column();
                    meta.enable_equality(col);
                    col
                }),
                input_advice: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH), || {
                    let col = meta.advice_column();
                    meta.enable_equality(col);
                    col
                }),
                conv_chip,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), PlonkError> {
            let conv_chip = Conv3DLayerChip::construct(config.conv_chip);

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

            let layer_params = Conv3DLayerParams {
                kernals: self.kernal.clone(),
            };

            let output = conv_chip.add_layer(&mut layouter, inputs, layer_params)?;
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
    ///test that a simple 16x16x4 w/ 3x3x4 conv works; input and kernal are all 1
    fn test_simple_conv() -> Result<(), PlonkError> {
        let circuit = ConvTestCircuit {
            kernal: Array::from_shape_simple_fn(
                (DEPTH, KERNAL_WIDTH, KERNAL_HEIGHT, DEPTH),
                || Value::known(Fr::one()),
            ),
            input: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH, INPUT_HEIGHT), || {
                Value::known(Fr::one())
            }),
        };

        let _output_width = INPUT_WIDTH + PADDING_WIDTH * 2 - KERNAL_WIDTH + 1;
        let _output_height = INPUT_HEIGHT + PADDING_HEIGHT * 2 - KERNAL_HEIGHT + 1;

        let mut input_instance = vec![vec![Fr::one(); INPUT_HEIGHT]; INPUT_WIDTH * DEPTH];
        //let mut output_instance = vec![vec![Fr::one(); output_height]; output_width*DEPTH];
        let edge: Vec<_> = vec![
            16, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 16,
        ]
        .iter()
        .map(|x| Fr::from(*x))
        .collect();
        let row: Vec<_> = vec![
            24, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 24,
        ]
        .iter()
        .map(|&x| Fr::from(x))
        .collect();
        let layer = vec![
            edge.clone(),
            row.clone(),
            row.clone(),
            row.clone(),
            row.clone(),
            row.clone(),
            row.clone(),
            row.clone(),
            row.clone(),
            row.clone(),
            row.clone(),
            row.clone(),
            row.clone(),
            row.clone(),
            row,
            edge,
        ];
        let mut output_instance: Vec<_> = vec![layer.clone(), layer.clone(), layer.clone(), layer]
            .into_iter()
            .flatten()
            .collect();
        input_instance.append(&mut output_instance);

        MockProver::run(7, &circuit, input_instance)
            .unwrap()
            .assert_satisfied();

        Ok(())
    }
}
