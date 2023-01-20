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

use crate::nn_ops::eltwise_ops::{DecompConfig, NormalizeChip, EltwiseInstructions};

#[derive(Clone, Debug)]
pub struct AvgPool2DConfig<F: FieldExt> {
    //pub in_width: usize,
    //pub in_height: usize,
    //pub in_depth: usize,
    pub inputs: Array1<Column<Advice>>,
    pub output: Column<Advice>,
    pub selector: Selector,
    pub norm_chip: DecompConfig<F>,
    _marker: PhantomData<F>,
}

///Chip for 2-D Convolution (width, height, channel-in, channel-out)
///
/// Order for ndarrays is Channel-in, Width, Height, Channel-out
pub struct AvgPool2DChip<F: FieldExt> {
    config: AvgPool2DConfig<F>,
}

impl<F: FieldExt> Chip<F> for AvgPool2DChip<F> {
    type Config = AvgPool2DConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: FieldExt> AvgPool2DChip<F> {
    const DEPTH_AXIS: Axis = Axis(0);
    const COLUMN_AXIS: Axis = Axis(1);
    const ROW_AXIS: Axis = Axis(2);

    const SCALING_FACTOR: u64 = 1_048_576;

    const ROW_COUNT: usize = 8;

    pub fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self { config }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        inputs: Array1<Column<Advice>>,
        output: Column<Advice>,
        norm_chip: DecompConfig<F>,
    ) -> <Self as Chip<F>>::Config {
        let selector = meta.selector();

        meta.create_gate("Avg Pool", |meta| -> Vec<Expression<F>> {
            let sel = meta.query_selector(selector);

            let sum = inputs.fold(Expression::Constant(F::zero()), |accum, &item| {
                (0..Self::ROW_COUNT).fold(accum, |accum, row| {
                    accum + meta.query_advice(item, Rotation(row as i32))
                })
            });

            let scalar = Expression::Constant(F::from(Self::SCALING_FACTOR / (inputs.dim() as u64 * Self::ROW_COUNT as u64)));

            let output = meta.query_advice(output, Rotation::cur());

            vec![sel * (output - (sum * scalar))]
        });

        AvgPool2DConfig {
            inputs,
            output,
            selector,
            norm_chip,
            _marker: PhantomData,
        }
    }

    pub fn add_layer(
        &self,
        layouter: &mut impl Layouter<F>,
        inputs: &Array3<AssignedCell<F, F>>,
    ) -> Result<Array1<AssignedCell<F, F>>, PlonkError> {
        let config = &self.config;

        let avg_pool_output: Result<Array1<_>, _> = inputs.axis_iter(Self::DEPTH_AXIS).map(|inputs| {
            layouter.assign_region(|| "Avg Pool Layer", |mut region| {
                config.selector.enable(&mut region, 0)?;
                //copy inputs
                inputs.axis_iter(Axis(0)).zip(config.inputs.iter()).for_each(|(inputs, &column)| {
                    inputs.iter().enumerate().for_each(|(row, input)| {
                        input.copy_advice(|| "Copy input to Avg Pool", &mut region, column, row).unwrap();
                    });
                });

                //calculate output
                let sum = inputs.iter().fold(Value::known(F::zero()), |accum, input| {
                    input.value().zip(accum).map(|(&input, accum)| input + accum)
                });

                let scalar = F::from(Self::SCALING_FACTOR / (inputs.dim().0 as u64 * inputs.dim().1 as u64));

                let output = sum.map(|sum| sum * scalar);

                //assign output
                region.assign_advice(|| "Avg Pool Output", config.output, 0, || output)
            })
        }).collect();

        avg_pool_output?.iter().map(|avg_pool_output| {
            let norm_chip = NormalizeChip::<F, 1024, 2>::construct(config.norm_chip.clone());
            norm_chip.apply_elt(layouter.namespace(|| "Norm after avg pool"), avg_pool_output.clone())
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::{nn_ops::{lookup_ops::DecompTable, eltwise_ops::NormalizeChip}, felt_from_i64};

    use super::{AvgPool2DChip, AvgPool2DConfig};
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
    use ndarray::{array, stack, Array, Array2, Array3, Array4, ArrayBase, Axis, Zip};

    #[derive(Clone, Debug)]
    struct AvgPool2DTestConfig<F: FieldExt> {
        input: Array2<Column<Instance>>,
        input_advice: Array2<Column<Advice>>,
        output: Column<Instance>,
        avg_chip: AvgPool2DConfig<F>,
        range_table: DecompTable<F, 1024>
    }

    struct AvgPool2DTestCircuit<F: FieldExt> {
        pub input: Array3<Value<F>>,
    }

    const INPUT_WIDTH: usize = 8;
    const INPUT_HEIGHT: usize = 8;

    const DEPTH: usize = 4;

    impl<F: FieldExt> Circuit<F> for AvgPool2DTestCircuit<F> {
        type Config = AvgPool2DTestConfig<F>;

        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                input: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH, INPUT_HEIGHT), || {
                    Value::unknown()
                }),
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let inputs =
                Array::from_shape_simple_fn(INPUT_WIDTH, || {
                    let col = meta.advice_column();
                    meta.enable_equality(col);
                    col
                });

            let output = {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            };

            const ADVICE_LEN: usize = 10;

            let eltwise_inter = Array::from_shape_simple_fn(ADVICE_LEN + 1, || {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            });

            let range_table: DecompTable<F, 1024> = DecompTable::configure(meta);

            let norm_input = {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            };

            let norm_chip = NormalizeChip::<_, 1024, 2>::configure(
                meta,
                norm_input,
                eltwise_inter.to_vec(),
                output,
                range_table.clone(),
            );


            let avg_chip = AvgPool2DChip::configure(
                meta,
                inputs,
                output,
                norm_chip,
            );

            AvgPool2DTestConfig {
                input: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH), || {
                    let col = meta.instance_column();
                    meta.enable_equality(col);
                    col
                }),
                output: {
                    let col = meta.instance_column();
                    meta.enable_equality(col);
                    col
                },
                input_advice: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH), || {
                    let col = meta.advice_column();
                    meta.enable_equality(col);
                    col
                }),
                avg_chip,
                range_table,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), PlonkError> {
            let avg_chip = AvgPool2DChip::construct(config.avg_chip);
            config
            .range_table
            .layout(layouter.namespace(|| "range check lookup table"))?;


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

            let output = avg_chip.add_layer(&mut layouter, &inputs)?;
            for (row, output) in output.iter().enumerate() {
                layouter
                    .constrain_instance(output.cell(), config.output, row)
                    .unwrap();
            }
            Ok(())
        }
    }

    const TEST_INPUT: [i64; 256] = [-35763, 108446, -818055, 389038, -196141, -455364, 865262, -837986, 660250, -538160, 930815, 14282, 650375, 349994, 671489, -438082, 959393, 225926, -869666, -687261, -18938, 572223, 345842, 917220, -986762, -760786, 489076, -989060, -465216, -929305, 193544, -449730, -78113, 252360, 682841, -452967, 567669, -939688, 653108, 495015, -632829, -884147, -979411, -490706, 623732, -760761, -87121, -303321, 724152, -655787, 671328, 681708, 643070, 875511, 486656, -86903, -76626, 975173, -641742, 398982, -160694, -611420, -625655, -474324, 272281, 455017, -475667, 708693, -19511, 328612, 55757, -544168, 639028, 942355, 925044, 228805, -542241, 412902, 860493, 324998, -706266, 691209, -412395, 662798, -952877, 354722, -978066, 298246, -84889, -819531, 143529, -611893, 363460, 333216, 557436, 928412, -271600, 782553, 919486, -277873, 655204, -608991, -154530, 517664, -425317, -591223, 355582, -129413, 491738, 521777, 488516, 585824, -729604, 48758, -128154, 525182, 709965, 850561, 796821, -342804, 491260, -751298, -775034, -568191, -346796, 629362, 593774, -852791, -979301, -321443, -601615, -943455, -285882, -91438, -213241, 625440, -841792, 513212, -546886, -389338, 503365, 71013, -827587, 695703, -446469, -170582, 752943, 968831, 541215, 454396, 531655, 658512, 668773, 465367, -742235, 887088, 914098, 250547, -128557, 874778, -449337, 616279, -718454, 671127, -231687, -313892, 364366, 307613, -877693, 369286, -786361, -407872, 865085, -496357, -612513, 269715, 299401, 563356, -957634, -874512, -100416, -447938, 31806, 516289, -759883, 912912, 20907, 433279, 567599, 602104, -634049, 61077, 447261, 563173, 936076, 925108, -986145, -51827, 905699, 297809, 874318, 132144, -913554, 890191, -910428, -591779, 954380, -333164, -751763, -966144, -52006, -653361, 213060, 99214, -883560, -282662, 930599, -385201, -568220, 760929, 239783, 783196, -133702, -801911, 289831, 695920, 609369, 143864, 533320, -903284, -265488, -925706, 505169, -470804, 247352, -230551, 776339, 641213, 497880, 272991, -753500, -470839, 233543, 241360, 258923, -721705, -997616, -449684, -251658, -855275, -451871, 393210, 566364, -336961, -568168, 146829];

    const TEST_OUTPUT: [i64; 4] = [-21000, 114842, 25792, -14251];

    #[test]
    ///test that a simple 4x8x8 avg pool works
    fn test_simple_avg_pool() -> Result<(), PlonkError> {
        let inputs = Array::from_shape_vec((DEPTH, INPUT_WIDTH, INPUT_HEIGHT), TEST_INPUT.to_vec()).unwrap();
        let inputs = Zip::from(inputs.view()).map_collect(|&input| felt_from_i64(input));

        let output = Array::from_shape_vec(DEPTH, TEST_OUTPUT.to_vec()).unwrap();
        let output = Zip::from(output.view()).map_collect(|&output| felt_from_i64(output));


        let mut instances = inputs.axis_iter(Axis(0)).fold(vec![], |mut accum, inputs| {
            accum.append(&mut inputs.axis_iter(Axis(0)).map(|inputs| inputs.to_vec()).collect::<Vec<_>>());
            accum
        });

        instances.push(output.to_vec());

        let circuit = AvgPool2DTestCircuit {
            input: Zip::from(inputs.view()).map_collect(|&input| Value::known(input))
        };

        MockProver::<Fr>::run(11, &circuit, instances)
            .unwrap()
            .assert_satisfied();

        Ok(())
    }
}
