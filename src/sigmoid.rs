use std::marker::PhantomData;

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter, Value},
    plonk::{
        Advice, Column, ConstraintSystem, Error as PlonkError, Expression,
        Selector,
    },
    poly::Rotation,
};

use ndarray::{
    Array, Array1, Axis,
};

use crate::{
    felt_from_i64,
    nn_ops::{
        eltwise_ops::{DecompConfig, EltwiseInstructions, NormalizeChip},
        lookup_ops::DecompTable,
    },
};

#[derive(Clone, Debug)]
pub struct SigmoidConfig<F: FieldExt> {
    //pub in_width: usize,
    //pub in_height: usize,
    //pub in_depth: usize,
    pub inputs: Column<Advice>,
    pub outputs: Column<Advice>,
    pub eltwise_inter: Array1<Column<Advice>>,
    pub ranges: Column<Advice>,
    pub comp_signs: Column<Advice>,
    pub comp_selector: Selector,
    pub output_selector: Selector,
    pub norm_chip: DecompConfig<F>,
    _marker: PhantomData<F>,
}

/// Chip for 2d Sigmoid
///
/// Order for ndarrays is Channel-in, Width, Height
pub struct SigmoidChip<F: FieldExt, const BASE: usize> {
    config: SigmoidConfig<F>,
}

impl<F: FieldExt, const BASE: usize> Chip<F> for SigmoidChip<F, BASE> {
    type Config = SigmoidConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: FieldExt, const BASE: usize> SigmoidChip<F, BASE> {
    const COLUMN_AXIS: Axis = Axis(0);
    const ROW_AXIS: Axis = Axis(1);
    const ADVICE_LEN: usize = 10;
    const CEIL: u64 = 2_097_152; //2 * 2^20
    const MAX_VALUE: u64 = 1_099_511_627_776; //1 * (2^20)^2
    const FLOOR: i64 = -2_097_152;
    const MIN_VALUE: u64 = 0;
    const SCALAR: u64 = 262_144; //.25 * 2^20
    const ADDITIVE_BIAS: u64 = 549_755_813_888; //.5 * (2^20)^2

    pub fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self { config }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        inputs: Column<Advice>,
        ranges: Column<Advice>,
        outputs: Column<Advice>,
        eltwise_inter: Array1<Column<Advice>>,
        range_table: DecompTable<F, BASE>,
        norm_chip: DecompConfig<F>,
    ) -> <Self as Chip<F>>::Config {
        let comp_selector = meta.complex_selector();
        let output_selector = meta.selector();

        let max_value = F::from(Self::MAX_VALUE);

        let min_value = F::from(Self::MIN_VALUE);

        let scalar = F::from(Self::SCALAR);
        let additive_bias = F::from(Self::ADDITIVE_BIAS);

        for &item in eltwise_inter.iter() {
            meta.lookup("lookup", |meta| {
                let s_elt = meta.query_selector(comp_selector);
                let word = meta.query_advice(item, Rotation::cur());
                vec![(s_elt * word, range_table.range_check_table)]
            });
        }

        let constant_1 = Expression::Constant(F::from(1));
        let mut eltwise_inter_vec = eltwise_inter.to_vec();
        let comp_signs = eltwise_inter_vec.remove(0);

        meta.create_gate("Sigmoid 2D Comparison", |meta| {
            let sel = meta.query_selector(comp_selector);

            //iterate over all elements to the input
            let base: u64 = BASE.try_into().unwrap();
            assert_eq!(
                Self::ADVICE_LEN, eltwise_inter_vec.len(),
                "Must pass in sufficient advice columns for eltwise intermediate operations: passed in {}, need {}", 
                eltwise_inter_vec.len(), Self::ADVICE_LEN
            );
            let input = meta.query_advice(inputs, Rotation::cur());
            let comp_sign = meta.query_advice(comp_signs, Rotation::cur());
            let iter = eltwise_inter_vec.iter();
            let base = F::from(base);
            let word_sum = iter
                .clone()
                .enumerate()
                .map(|(index, column)| {
                    let b = meta.query_advice(*column, Rotation::cur());
                    let true_base = (0..index).fold(F::from(1), |expr, _input| expr * base);
                    b * true_base
                })
                .reduce(|accum, item| accum + item)
                .unwrap();

            let comp = meta.query_advice(ranges, Rotation::cur());

            let constant_1 = Expression::Constant(F::from(1));

            vec![sel * (word_sum - ((comp_sign.clone() * (input.clone() - comp.clone())) + ((constant_1-comp_sign) * (comp - input))))]
        });

        meta.create_gate("Sigmoid 2D Output", |meta| -> Vec<Expression<F>> {
            let sel = meta.query_selector(output_selector);
            let input = meta.query_advice(inputs, Rotation::cur());
            let output = meta.query_advice(outputs, Rotation::cur());

            let comp_sign_1 = meta.query_advice(comp_signs, Rotation::cur());
            let comp_sign_2 = meta.query_advice(comp_signs, Rotation::next());

            vec![
                sel * (output
                    - (comp_sign_1.clone() * Expression::Constant(max_value)
                        + ((constant_1.clone() - comp_sign_1)
                            * (comp_sign_2.clone()
                                * ((input * Expression::Constant(scalar))
                                    + Expression::Constant(additive_bias))
                                + ((constant_1.clone() - comp_sign_2)
                                    * Expression::Constant(min_value)))))),
            ]
        });

        SigmoidConfig {
            inputs,
            outputs,
            eltwise_inter,
            ranges,
            comp_signs,
            comp_selector,
            output_selector,
            norm_chip,
            _marker: PhantomData,
        }
    }

    pub fn add_layer(
        &self,
        layouter: &mut impl Layouter<F>,
        inputs: &Array1<AssignedCell<F, F>>,
    ) -> Result<Array1<AssignedCell<F, F>>, PlonkError> {
        let base: u128 = BASE.try_into().unwrap();
        let config = &self.config;

        let ciel = F::from(Self::CEIL);
        let max_value = F::from(Self::MAX_VALUE);

        let floor = felt_from_i64(Self::FLOOR);
        let min_value = F::from(Self::MIN_VALUE);

        let scalar = F::from(Self::SCALAR);
        let additive_bias = F::from(Self::ADDITIVE_BIAS);

        let sigmoid_output = layouter.assign_region(
            || "apply 2d sigmoid",
            |mut region| {
                let outputs = inputs
                    .iter()
                    .enumerate()
                    .map(|(row, input)| {
                        let offset = row * 2;
                        let offset_2 = offset + 1;
                        let input_col = config.inputs;
                        let bit_sign_col = config.comp_signs;
                        let eltwise_inter = &config.eltwise_inter;
                        self.config.comp_selector.enable(&mut region, offset)?;
                        self.config.comp_selector.enable(&mut region, offset_2)?;
                        self.config.output_selector.enable(&mut region, offset)?;
                        input.copy_advice(|| "eltwise input", &mut region, input_col, offset)?;
                        input.copy_advice(|| "eltwise input", &mut region, input_col, offset_2)?;

                        let comp_sign_1 = input.value().map(|x| x > &ciel && x < &F::TWO_INV);

                        let comp_sign_2 = input.value().map(|x| x > &floor || x < &F::TWO_INV);

                        // let word_repr: Value<Vec<u32>> = output_i32.map(|x| {
                        //     let str = format!("{:o}", x.abs());
                        //     str.chars()
                        //         .map(|char| char.to_digit(8).unwrap())
                        //         .rev()
                        //         .collect()
                        // });

                        let difference_1 = input.value().map(|x| {
                            if x > &ciel && x < &F::TWO_INV {
                                *x - &ciel
                            } else {
                                ciel - x
                            }
                        });

                        let difference_2 = input.value().map(|x| {
                            if x > &floor || x < &F::TWO_INV {
                                *x - &floor
                            } else {
                                floor - x
                            }
                        });

                        let word_repr_1: Value<Vec<u16>> = difference_1.and_then(|x| {
                            let mut result = vec![];
                            let mut x = x.get_lower_128();

                            loop {
                                let m = x % base;
                                x /= base;

                                result.push(m as u16);
                                if x == 0 {
                                    break;
                                }
                            }

                            Value::known(result)
                        });
                        let word_repr_2: Value<Vec<u16>> = difference_2.and_then(|x| {
                            let mut result = vec![];
                            let mut x = x.get_lower_128();

                            loop {
                                let m = x % base;
                                x /= base;

                                result.push(m as u16);
                                if x == 0 {
                                    break;
                                }
                            }

                            Value::known(result)
                        });

                        region.assign_advice(
                            || "sigmoid comp_sign_1",
                            bit_sign_col,
                            offset,
                            || comp_sign_1.map(|x| F::from(x)),
                        )?;
                        region.assign_advice(
                            || "sigmoid comp_sign_2",
                            bit_sign_col,
                            offset_2,
                            || comp_sign_2.map(|x| F::from(x)),
                        )?;
                        region.assign_advice(
                            || "sigmoid range ciel",
                            config.ranges,
                            offset,
                            || Value::known(ciel),
                        )?;
                        region.assign_advice(
                            || "sigmoid range floor",
                            config.ranges,
                            offset_2,
                            || Value::known(floor),
                        )?;
                        let _: Vec<_> = (0..eltwise_inter.len() - 1)
                            .map(|index_col| {
                                region
                                    .assign_advice(
                                        || "sigmoid word_repr_1",
                                        eltwise_inter[index_col + 1],
                                        offset,
                                        || {
                                            word_repr_1.clone().map(|x| {
                                                match index_col >= x.len() {
                                                    false => F::from(x[index_col] as u64),
                                                    true => F::from(0),
                                                }
                                            })
                                        },
                                    )
                                    .unwrap();
                                region
                                    .assign_advice(
                                        || "sigmoid word_repr_2",
                                        eltwise_inter[index_col + 1],
                                        offset_2,
                                        || {
                                            word_repr_2.clone().map(|x| {
                                                match index_col >= x.len() {
                                                    false => F::from(x[index_col] as u64),
                                                    true => F::from(0),
                                                }
                                            })
                                        },
                                    )
                                    .unwrap();
                            })
                            .collect();
                        region.assign_advice(
                            || "sigmoid_output",
                            config.outputs,
                            offset,
                            || {
                                input.value().map(|&x| {
                                    match (x > ciel && x < F::TWO_INV, x > floor || x < F::TWO_INV)
                                    {
                                        (true, _) => max_value,
                                        (false, true) => (x * scalar) + additive_bias,
                                        (_, false) => min_value,
                                    }
                                })
                            },
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Array::from_vec(outputs))
            },
        )?;

        let norm_chip = NormalizeChip::<F, BASE, 2>::construct(config.norm_chip.clone());
        sigmoid_output
            .into_iter()
            .map(|sigmoid| {
                norm_chip.apply_elt(layouter.namespace(|| "normalize sigmoid output"), sigmoid)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        felt_from_i64,
        nn_ops::{eltwise_ops::NormalizeChip, lookup_ops::DecompTable},
    };

    use super::{SigmoidChip, SigmoidConfig};
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

    #[derive(Clone, Debug)]
    struct SigmoidTestConfig<F: FieldExt> {
        input: Column<Instance>,
        input_advice: Column<Advice>,
        output: Column<Instance>,
        sigmoid_chip: SigmoidConfig<F>,
        range_table: DecompTable<F, 1024>,
    }

    struct SigmoidTestCircuit<F: FieldExt> {
        pub input: Array1<Value<F>>,
    }

    const INPUT_SIZE: usize = 64;

    impl<F: FieldExt> Circuit<F> for SigmoidTestCircuit<F> {
        type Config = SigmoidTestConfig<F>;

        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                input: Array::from_shape_simple_fn(INPUT_SIZE, || Value::unknown()),
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let inputs = {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            };

            //let output_width = INPUT_WIDTH + PADDING_WIDTH * 2 - KERNAL_WIDTH + 1;

            let outputs = {
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

            let ranges = {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            };

            let range_table: DecompTable<F, 1024> = DecompTable::configure(meta);

            let norm_chip = NormalizeChip::<_, 1024, 2>::configure(
                meta,
                inputs,
                eltwise_inter.to_vec(),
                outputs,
                range_table.clone(),
            );

            let sigmoid_chip = SigmoidChip::<_, 1024>::configure(
                meta,
                inputs,
                ranges,
                outputs,
                eltwise_inter,
                range_table.clone(),
                norm_chip,
            );

            SigmoidTestConfig {
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
                sigmoid_chip,
                range_table,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), PlonkError> {
            let norm_chip: SigmoidChip<F, 1024> = SigmoidChip::construct(config.sigmoid_chip);

            config
                .range_table
                .layout(layouter.namespace(|| "range check lookup table"))?;

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

            let output = norm_chip.add_layer(&mut layouter, &inputs)?;
            for (row, output) in output.iter().enumerate() {
                layouter
                    .constrain_instance(output.cell(), config.output, row)
                    .unwrap();
            }
            Ok(())
        }
    }

    const TEST_INPUT: [i64; 64] = [
        -1708196, -1728911, -291931, -1924070, 1189949, 1907001, -1585685, -572931, 2423702,
        597856, 1678723, 1381191, -815078, 1182949, 1968388, -488371, 1018666, 433309, -567488,
        -665819, -1806148, 1780427, 897226, 1052506, 2963539, -2242915, 202210, 867646, -1139101,
        -2182539, -221231, -2096520, 676129, 2451314, -826712, -2111170, 952788, -1333618, -651307,
        -1183385, 1416332, -1714320, 2375800, -532699, 2852778, -2690324, -1652103, -1598632,
        -1369244, 1560568, 1952261, -2940522, -1206972, 26227, -2268278, 2285987, -2390417,
        -2586051, 1881744, 1627420, -1885986, 1460486, -668513, -2133159,
    ];
    const TEST_OUTPUT: [u64; 64] = [
        97239, 92060, 451305, 43270, 821775, 1001038, 127866, 381055, 1048576, 673752, 943968,
        869585, 320518, 820025, 1016385, 402195, 778954, 632615, 382416, 357833, 72751, 969394,
        748594, 787414, 1048576, 0, 574840, 741199, 239512, 0, 468980, 158, 693320, 1048576,
        317610, 0, 762485, 190883, 361461, 228441, 878371, 95708, 1048576, 391113, 1048576, 0,
        111262, 124630, 181977, 914430, 1012353, 0, 222545, 530844, 0, 1048576, 0, 0, 994724,
        931143, 52791, 889409, 357159, 0,
    ];

    #[test]
    ///test that a simple 8x8 sigmoid works
    fn test_simple_sigmoid() -> Result<(), PlonkError> {
        let input = Array::from_shape_vec(INPUT_SIZE, TEST_INPUT.to_vec()).unwrap();
        let input = Zip::from(input.view()).map_collect(|&input| felt_from_i64(input));

        let output = Array::from_shape_vec(INPUT_SIZE, TEST_OUTPUT.to_vec()).unwrap();
        let output = Zip::from(output.view()).map_collect(|&output| Fr::from(output));

        let circuit = SigmoidTestCircuit {
            input: Zip::from(input.view()).map_collect(|&input| Value::known(input)),
        };

        let input_instance: Vec<_> = vec![input.to_vec(), output.to_vec()];

        MockProver::run(11, &circuit, input_instance)
            .unwrap()
            .assert_satisfied();

        Ok(())
    }
}
