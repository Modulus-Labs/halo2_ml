use std::marker::PhantomData;

use halo2_base::halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, Error as PlonkError, Expression, Fixed, Selector},
    poly::Rotation,
};
use ndarray::{concatenate, stack, Array, Array1, Array2, Array3, Array4, Axis, Zip};

use crate::nn_ops::{ColumnAllocator, NNLayer};

#[derive(Clone, Debug)]
pub struct Conv3DLayerConfig<F: FieldExt> {
    //pub in_width: usize,
    //pub in_height: usize,
    //pub in_depth: usize,
    pub ker_width: usize,
    pub ker_height: usize,
    pub padding_width: usize,
    pub padding_height: usize,
    pub folding_factor: usize,
    pub inputs: Array2<Column<Advice>>,
    pub outputs: Array1<Column<Advice>>,
    pub final_outputs: Array1<Column<Advice>>,
    pub kernals: Array2<Column<Fixed>>,
    pub conv_selectors: Vec<Selector>,
    pub out_selector: Selector,
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
    pub input_height: usize,
    pub input_width: usize,
    pub input_depth: usize,
    pub ker_height: usize,
    pub ker_width: usize,
    pub padding_width: usize,
    pub padding_height: usize,
    pub folding_factor: usize,
}

#[derive(Clone, Debug)]
pub struct Conv3DLayerParams<F: FieldExt> {
    pub kernals: Array4<Value<F>>,
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
            input_height,
            input_width,
            input_depth,
            ker_height,
            ker_width,
            padding_width,
            padding_height,
            folding_factor,
        } = config_params;
        let output_width = input_width + padding_width * 2 - ker_width + 1;
        let input_column_count = (input_width + padding_width * 2) * (input_depth / folding_factor);
        let output_column_count = output_width * 2;
        let advice_count = input_column_count + output_column_count;
        let fixed_count = ker_width * (input_depth / folding_factor);

        let advice = advice_allocator.take(meta, advice_count);
        let fixed = fixed_allocator.take(meta, fixed_count);

        let inputs = Array::from_shape_vec(
            (input_depth / folding_factor, (input_width + padding_width * 2)),
            advice[0..input_column_count].to_vec(),
        )
        .unwrap();
        let outputs = Array::from_shape_vec(
            output_width,
            advice[input_column_count..input_column_count + output_width].to_vec(),
        )
        .unwrap();

        let final_outputs = Array::from_shape_vec(
            output_width,
            advice[input_column_count + output_width..input_column_count + (output_width * 2)]
                .to_vec(),
        )
        .unwrap();

        let kernals = Array::from_shape_vec((input_depth / folding_factor, ker_width), fixed.to_vec()).unwrap();
        let ker_height_i32: i32 = ker_height.try_into().unwrap();
        let selectors: Vec<Selector> = (0..ker_height_i32)
            .map(|offset| {
                let selector = meta.selector();
                meta.create_gate("Conv", |meta| -> Vec<Expression<F>> {
                    let sel = meta.query_selector(selector);

                    //Stack input column along the inner-most axis to represent rows
                    let inputs: Array3<Expression<F>> = stack(
                        Self::ROW_AXIS,
                        &(0..ker_height_i32)
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
                                    accum + (input.clone() * kernal.clone())
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

        let out_selector = meta.selector();
        meta.create_gate("Conv Final Output", |meta| {
            let sel = meta.query_selector(out_selector);

            let outputs: Array1<_> = outputs
                .iter()
                .map(|&out_col| {
                    // (
                    //     meta.query_advice(out_col, Rotation::cur()),
                    //     meta.query_advice(out_col, Rotation(-i32::try_from(input_height + (padding_height * 2)).unwrap())),
                    // )
                    (0..folding_factor).map(|index| {
                        meta.query_advice(out_col, Rotation(-i32::try_from((input_height + (padding_height * 2))*index).unwrap()))
                    }).reduce(|accum, item| accum + item).unwrap()
                })
                .collect();

            let outputs_final: Array1<_> = final_outputs
                .iter()
                .map(|&final_col| meta.query_advice(final_col, Rotation::cur()))
                .collect();

            outputs
                .into_iter()
                .zip(outputs_final.into_iter())
                .map(|(sum, out_final)| sel.clone() * (sum - out_final))
                .collect::<Vec<_>>()
        });
        Conv3DLayerConfig {
            ker_width,
            ker_height,
            padding_width,
            padding_height,
            inputs,
            outputs,
            kernals,
            conv_selectors: selectors,
            final_outputs,
            out_selector,
            folding_factor,
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

        let folding_factor = config.folding_factor;

        //add padding
        let dims = inputs.dim();
        let output_width = dims.1 + config.padding_width * 2 - config.ker_width + 1;
        let output_height = dims.2 + config.padding_height * 2 - config.ker_height + 1;
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
                        let outputs: Result<Vec<_>, _> = inputs
                            .axis_chunks_iter(Self::DEPTH_AXIS, dims.0 / folding_factor)
                            .zip(kernals.axis_chunks_iter(Self::DEPTH_AXIS, dims.0 / folding_factor))
                            .enumerate()
                            .map(|(index, (inputs, kernals))| {
                                let offset = index * (dims.2 + config.padding_height*2);
                                inputs.axis_iter(Self::ROW_AXIS).enumerate().for_each(
                                    |(row, inputs)| {
                                        Zip::from(inputs).and(config.inputs.view()).for_each(
                                            |input, column| match input {
                                                InputOrPadding::Input(x) => {
                                                    x.copy_advice(
                                                        || "Copy Input",
                                                        &mut region,
                                                        *column,
                                                        offset + row,
                                                    )
                                                    .unwrap();
                                                }
                                                InputOrPadding::Padding => {
                                                    region
                                                        .assign_advice(
                                                            || "Add Padding",
                                                            *column,
                                                            offset + row,
                                                            || Value::known(F::zero()),
                                                        )
                                                        .unwrap();
                                                }
                                            },
                                        );
                                    },
                                );

                                let mut offset_ker = 0;
                                while offset_ker + config.ker_height < inputs.dim().2 {
                                    kernals.axis_iter(Self::ROW_AXIS).enumerate().for_each(
                                        |(row, kernal)| {
                                            if offset_ker + row + config.ker_height < inputs.dim().2
                                            {
                                                config.conv_selectors[row]
                                                    .enable(&mut region, offset + offset_ker + row)
                                                    .unwrap();
                                            }
                                            Zip::from(kernal).and(config.kernals.view()).for_each(
                                                |kernal, column| {
                                                    region
                                                        .assign_fixed(
                                                            || "place kernal",
                                                            *column,
                                                            offset + row + offset_ker,
                                                            || *kernal,
                                                        )
                                                        .unwrap();
                                                },
                                            );
                                        },
                                    );
                                    offset_ker += config.ker_height;
                                }

                                //calculate outputs
                                let outputs = Zip::indexed(inputs.windows(kernals.dim()))
                                    .map_collect(|(channel, column, row), inputs| {
                                        inputs.iter().zip(kernals.iter()).fold(
                                            Value::known(F::zero()),
                                            |accum, (item, kernal)| {
                                                accum + (*kernal * item.value())
                                            },
                                        )
                                    });

                                debug_assert_eq!(
                                    1,
                                    outputs.dim().0,
                                    "Layers should be reduced to a single layer output"
                                );

                                let outputs = outputs.remove_axis(Self::DEPTH_AXIS);
                                //place outputs
                                Ok::<_, PlonkError>(
                                    stack(
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
                                                                offset + row,
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
                                    .unwrap(),
                                )
                            })
                            .collect();
                        let outputs = outputs?;
                        // let out_1 = &outputs[0];
                        // let out_2 = &outputs[1];
                        // let outputs: Vec<_> = out_1
                        //     .axis_iter(Axis(0))
                        //     .zip(out_2.axis_iter(Axis(0)))
                        //     .zip(config.final_outputs.iter())
                        //     .map(|((out_1, out_2), &out_col)| {
                        //         out_1
                        //             .iter()
                        //             .zip(out_2.iter())
                        //             .enumerate()
                        //             .map(|(row, (out_1, out_2))| {
                        //                 let final_out = out_1.value().map(|x| *x) + out_2.value();
                        //                 config.out_selector.enable(&mut region, (dims.2 + (config.padding_height*2)) + row)?;
                        //                 region.assign_advice(
                        //                     || "final conv output",
                        //                     out_col,
                        //                     (dims.2 + (config.padding_height*2)) + row,
                        //                     || final_out,
                        //                 )
                        //             })
                        //             .collect::<Result<Vec<_>, _>>()
                        //     })
                        //     .collect::<Result<Vec<_>, _>>()?
                        //     .into_iter()
                        //     .flatten()
                        //     .collect();
                        let outputs: Vec<_> = outputs.into_iter().flat_map(|x| x.into_iter()).collect();
                        let outputs = Array::from_shape_vec((folding_factor, output_width, output_height), outputs).unwrap();
                        let outputs = outputs.axis_iter(Axis(1)).zip(config.final_outputs.iter()).flat_map(|(outputs, &column)| {
                            outputs.axis_iter(Axis(1)).enumerate().map(|(row, outputs)| {
                                let final_out = outputs.iter().fold(Value::known(F::zero()), |accum, item| accum + item.value());
                                config.out_selector.enable(&mut region, (dims.2 + (config.padding_height*2))*(folding_factor-1) + row).unwrap();
                                region.assign_advice(
                                    || "final conv output",
                                    column,
                                    (dims.2 + (config.padding_height*2))*(folding_factor-1) + row,
                                    || final_out,
                                ).unwrap()
                            }).collect::<Vec<_>>()
                        }).collect();
                        Ok(Array::from_shape_vec((output_width, output_height), outputs).unwrap())
                    },
                )
            })
            .collect::<Result<Vec<_>, PlonkError>>()?;

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
    use crate::{
        felt_from_i64,
        nn_ops::{ColumnAllocator, DefaultDecomp, NNLayer},
    };

    use super::{Conv3DLayerChip, Conv3DLayerConfig, Conv3DLayerConfigParams, Conv3DLayerParams};
    use halo2_base::halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        halo2curves::bn256::Fr,
        plonk::{Advice, Circuit, Column, ConstraintSystem, Error as PlonkError, Fixed, Instance},
    };
    use ndarray::{stack, Array, Array2, Array3, Array4, Axis, Zip};

    #[derive(Clone, Debug)]
    struct ConvTestConfig<F: FieldExt> {
        input: Column<Instance>,
        input_advice: Column<Advice>,
        output: Column<Instance>,
        conv_chip: Conv3DLayerConfig<F>,
    }

    struct ConvTestCircuit<F: FieldExt> {
        pub kernal: Array4<Value<F>>,
        pub input: Array3<Value<F>>,
    }

    const INPUT_WIDTH: usize = 8;
    const INPUT_HEIGHT: usize = 8;

    const KERNAL_WIDTH: usize = 3;
    const KERNAL_HEIGHT: usize = 3;

    const DEPTH: usize = 4;
    const C_OUT: usize = 4;

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
                padding_width: PADDING_WIDTH,
                padding_height: PADDING_HEIGHT,
                folding_factor: 4,
            };

            let conv_chip = Conv3DLayerChip::configure(
                meta,
                config_params,
                &mut advice_allocator,
                &mut fixed_allocator,
            );

            ConvTestConfig {
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
                || "input assignment",
                |mut region| {
                    let inputs: Result<Vec<_>, _> = self
                        .input
                        .iter()
                        .enumerate()
                        .map(|(row, _)| {
                            region.assign_advice_from_instance(
                                || "copy input to advice",
                                config.input,
                                row,
                                config.input_advice,
                                row,
                            )
                        })
                        .collect();
                    Ok(Array::from_shape_vec((DEPTH, INPUT_WIDTH, INPUT_HEIGHT), inputs?).unwrap())
                },
            )?;

            let layer_params = Conv3DLayerParams {
                kernals: self.kernal.clone(),
            };

            let output = conv_chip.add_layer(&mut layouter, inputs, layer_params)?;

            for (row, output) in output.iter().enumerate() {
                layouter.constrain_instance(output.cell(), config.output, row)?;
            }

            Ok(())
        }
    }

    const TEST_INPUT: [i64; 256] = [
        -362458, -905050, 769315, 313899, -289525, 888186, -194307, -598785, 126067, -162887,
        608295, 53213, 30329, 389822, 151677, -268616, 286876, 991378, 148567, 989541, 811733,
        121577, -694930, 502978, 846414, 389014, 544706, 344769, -650436, -572481, 947709, 415908,
        890026, 467210, -590251, -67713, 189739, -248293, 592370, 385266, -252665, -541046,
        -714124, 516219, 52376, -751877, -338261, 898630, 218620, -801769, 731073, 716372, -180368,
        512143, -379914, -604460, 451924, 267572, -38889, -937120, 890010, 745294, -55501, -120440,
        394470, -693816, -849786, 451555, -897239, 117217, -927609, 356239, -27439, -425447,
        915691, -301967, -745661, 157321, -961779, 715436, 432323, 498548, -231364, 511527, 943861,
        254205, 739990, 435217, -274280, 681409, -455097, -747453, -510436, 969507, -33861, 960046,
        358011, -241517, 697582, -211135, -665730, 960845, -750280, 400500, 304461, 196507,
        -404510, -969411, 585553, 609438, -797571, -922921, -959823, -559442, 501389, 675141,
        170137, 419695, 283111, -756135, 351289, -405509, 592528, 102645, 679275, -518066, 245268,
        739992, -395461, -807424, 717335, -419104, 966, 658902, -450278, 546880, 192251, 157848,
        -527618, -421702, -112120, -348675, -477854, 804676, -138053, -352890, -619830, -980668,
        45063, -348265, -852352, -332775, 407468, -504504, -978382, -568023, -624053, -885276,
        -341598, 547510, 264936, -233961, 929168, 598566, 196193, 634658, 653348, 278846, -514980,
        -996800, 190407, -645447, 783328, 604651, 471662, 805069, 409806, 477355, 61922, 653800,
        -18961, -711513, 363021, -164058, 816108, -477881, 778082, 721883, -347722, 921109,
        -942910, 660954, 120392, -12674, 968349, -319389, 20993, 745307, 929081, -25928, -726130,
        275359, -575964, 525260, 580201, -805461, -636234, 807898, -725298, 361894, -298026,
        -373793, -149396, 312299, -317157, -585987, -727527, 699568, 952870, -892410, 737021,
        -103405, -712424, -765000, 174313, -851341, -418772, -526982, -551560, 511709, 814502,
        346675, -64069, 250172, -400178, -921422, 94388, 659837, -201475, -728308, 390156, -705505,
        67434, -559851, -105275, -542710, -330592, -190296, -622606, 218819, -858660, 580571,
        -198949, 51263, -200896, 25538,
    ];
    const TEST_KERNAL: [i64; 144] = [
        504919, -847937, 908114, 965823, -478288, -586622, -85633, 186921, 260837, -133152,
        -315440, 425209, -290125, -811961, -260669, -322095, -809977, -574128, -717584, -245858,
        282663, -807441, 561899, -608126, 378378, 386758, -563742, 443224, 12387, 111039, 276799,
        867829, -407021, -669941, 238875, -605913, -969923, 339834, -603683, -565011, 816957,
        35510, 312628, 435935, 854360, 475997, 427707, -623786, 501026, 327346, 92060, 369879,
        842392, 725333, -422075, -460562, 739356, 48485, -614360, 656257, -57681, 27797, -297491,
        299057, 718370, 927533, 819490, -751083, 431313, -687195, 412388, 624574, -334657, -658132,
        97879, -791078, 695169, 664647, -892563, 344985, -847729, 688545, -283742, 659808, 527789,
        -458382, -438244, -736599, 667253, -280230, 750836, 576976, 162300, 575316, -507909,
        -215864, -206319, 310773, 36912, -711359, 967010, 709594, 977438, 8999, 229352, -190115,
        647712, 146154, 593187, -302176, 609715, -250443, -22944, 347391, 982164, -32612, -962321,
        -361797, 90720, -54951, -213536, 783323, -351618, -997265, 257357, 571424, 956639, 501353,
        -843821, -91780, 169663, 410530, 633091, 6958, -41284, -656282, 581076, -437756, 604182,
        -820298, 971447, -980753, -738814, -599567,
    ];
    const TEST_OUTPUT: [i64; 256] = [
        -563840099661,
        -2414827380148,
        -308776614330,
        -1668932624577,
        -1520054498226,
        -3403184590840,
        -924041720475,
        1865766850233,
        -824057477161,
        -2416418933932,
        -1118483223688,
        -1583832064119,
        -857953501637,
        -1628939262416,
        -3140591086395,
        2710477305704,
        -113770190896,
        1570998784315,
        -1657969598027,
        -2844508006264,
        450762434783,
        791458536963,
        -2414299914703,
        2582095144267,
        -667660905585,
        -1550700293346,
        749062874597,
        -4638207658909,
        637007037739,
        3362671749206,
        3412092919771,
        657659981512,
        545136905844,
        -1245652180275,
        -548314131142,
        -913285630301,
        3843576719653,
        3875747527923,
        -1576657631822,
        506957079793,
        145941048431,
        -612464870932,
        650183014905,
        -1499009847088,
        -2038913409359,
        988549866155,
        -642897072767,
        -809124124670,
        569366172573,
        -1860904032477,
        4630775670784,
        -1475865851720,
        2324799572572,
        -28504748780,
        -1719399668729,
        1142755978218,
        -1197799505829,
        2977397277102,
        96525613048,
        4038090882633,
        1070383268225,
        -832329540427,
        128086073669,
        -369575995982,
        1451865380383,
        -94301863121,
        226463430523,
        -968193418649,
        -1688820422440,
        1070031416053,
        -1639084688238,
        1952484926876,
        -1782355762827,
        -592239548907,
        641515442907,
        -4218099682745,
        1017012681276,
        -920228327756,
        -1649323744833,
        740106268901,
        -2482467474367,
        -849337615922,
        -3639825604865,
        -1905255790660,
        -1939728593323,
        -1703364495325,
        412325341772,
        3698342175370,
        -1340606759326,
        -2119383255750,
        828305507516,
        325258990155,
        -2156013433733,
        1380301444519,
        -996610697270,
        355441281519,
        -1398929506857,
        -2666108471927,
        -647535271841,
        -4487567739964,
        -478637046666,
        1824012227765,
        1471118859026,
        -559520115099,
        797319044682,
        992724004319,
        1387267318802,
        1463614989859,
        989269605788,
        2040966566066,
        2441446432306,
        -4357090700331,
        632573000244,
        -824065722993,
        1669372945655,
        -838130680802,
        -562790001378,
        2087075417339,
        671802588717,
        21753835051,
        -449879355870,
        -932905090663,
        1828517817019,
        -1466094901018,
        43767692380,
        -2133577926937,
        89874602868,
        1267387658169,
        -316452740632,
        1847797958696,
        1354476324498,
        -3318198913600,
        1269948217134,
        739158790481,
        -2216025147983,
        2496856546563,
        156496656298,
        127975717733,
        -3321846310254,
        3227183168972,
        -24799632324,
        -1345231292177,
        1032383640418,
        2097691239406,
        -2250433673088,
        -1205423520669,
        -821864008662,
        -4902617993818,
        -636671411388,
        -49289486866,
        -288972320656,
        452295667268,
        241635579858,
        728077218355,
        3388761966874,
        381780724069,
        1219843244694,
        2919708885978,
        -64882020279,
        -1617866508913,
        -2141108589398,
        -1752664918116,
        1574151317401,
        615173503494,
        2203492884055,
        3348291393564,
        -219365514206,
        -969976944747,
        -243818412798,
        -121705054293,
        3444263239702,
        -3934061117541,
        -1312199471369,
        975532080461,
        -19800571335,
        361298866665,
        1793990121398,
        1294807253364,
        -2012591890198,
        -1333260314175,
        1781412737625,
        -2682683850970,
        -2113901706941,
        -5473602048,
        -263272602585,
        -1518171024539,
        -2133978936102,
        1527987309488,
        -1247232887810,
        705107748426,
        -3301121566498,
        -91343858417,
        422109019729,
        533644342682,
        1026691409938,
        -2860885081650,
        1181171838971,
        195411375667,
        2289771349360,
        -1478565520894,
        -932788584518,
        2212692409756,
        -897926413569,
        3576361029724,
        2043144460710,
        -334935776623,
        473408424019,
        3195659186225,
        860577712039,
        -2066621415060,
        -661609825738,
        1468284995122,
        1520350803362,
        -47344739846,
        2195712580849,
        3764761600139,
        876426821958,
        2743627116410,
        -603717793551,
        51126286598,
        5706534281186,
        -5450831855101,
        -38556444299,
        -1324717445297,
        -1558232710884,
        182325498325,
        442921363670,
        2042914131470,
        1464719069377,
        -2269455787550,
        -877703653177,
        1872850585782,
        2241811048185,
        -114891069058,
        2077253057670,
        468102804125,
        1709587565431,
        3113280418507,
        -2059597780275,
        650620288788,
        -536051587330,
        403635390691,
        824367604427,
        -2469480237970,
        347252710401,
        1652633604250,
        491176936326,
        640591881017,
        -237135762612,
        11871595042,
        433270998747,
        2142903980767,
        -3086850228065,
        1088925673006,
        -217120087698,
        -137831297957,
    ];

    #[test]
    ///test that a simple 8x8x4 w/ 3x3x4 conv works; input and kernal are all 1
    fn test_simple_conv() -> Result<(), PlonkError> {
        let kernal = Array::from_shape_vec(
            (DEPTH, KERNAL_WIDTH, KERNAL_HEIGHT, C_OUT),
            TEST_KERNAL
                .iter()
                .map(|&x| Value::known(felt_from_i64(x)))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let input = Array::from_shape_vec(
            (DEPTH, INPUT_WIDTH, INPUT_HEIGHT),
            TEST_INPUT
                .iter()
                .map(|&x| Value::known(felt_from_i64(x)))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let output = TEST_OUTPUT
            .iter()
            .map(|&x| felt_from_i64(x))
            .collect::<Vec<Fr>>();

        let circuit = ConvTestCircuit { kernal, input };

        let _output_width = INPUT_WIDTH + PADDING_WIDTH * 2 - KERNAL_WIDTH + 1;
        let _output_height = INPUT_HEIGHT + PADDING_HEIGHT * 2 - KERNAL_HEIGHT + 1;

        let instances = vec![
            TEST_INPUT
                .iter()
                .map(|&x| felt_from_i64(x))
                .collect::<Vec<Fr>>(),
            output,
        ];

        MockProver::run(9, &circuit, instances)
            .unwrap()
            .assert_satisfied();

        Ok(())
    }
}
