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
    stack, Array1, Array2, Axis,
};

use crate::nn_ops::lookup_ops::DecompTable;

#[derive(Clone, Debug)]
pub struct ReluNorm2dConfig<F: FieldExt> {
    //pub in_width: usize,
    //pub in_height: usize,
    //pub in_depth: usize,
    pub inputs: Array1<Column<Advice>>,
    pub outputs: Array1<Column<Advice>>,
    pub eltwise_inter: Array2<Column<Advice>>,
    pub bit_signs: Array1<Column<Advice>>,
    pub selector: Selector,
    _marker: PhantomData<F>,
}

/// Chip for 2d eltwise
///
/// Order for ndarrays is Channel-in, Width, Height
pub struct ReluNorm2dChip<F: FieldExt, const BASE: usize, const K: usize> {
    config: ReluNorm2dConfig<F>,
}

impl<F: FieldExt, const BASE: usize, const K: usize> Chip<F> for ReluNorm2dChip<F, BASE, K> {
    type Config = ReluNorm2dConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: FieldExt, const BASE: usize, const K: usize> ReluNorm2dChip<F, BASE, K> {
    const COLUMN_AXIS: Axis = Axis(0);
    const ROW_AXIS: Axis = Axis(1);
    const ADVICE_LEN: usize = 10;

    pub fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self { config }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        inputs: Array1<Column<Advice>>,
        outputs: Array1<Column<Advice>>,
        eltwise_inter: Array2<Column<Advice>>,
        range_table: DecompTable<F, BASE>,
    ) -> <Self as Chip<F>>::Config {
        let selector = meta.complex_selector();

        for &item in eltwise_inter.iter() {
            meta.lookup("lookup", |meta| {
                let s_elt = meta.query_selector(selector);
                let word = meta.query_advice(item, Rotation::cur());
                vec![(s_elt * word, range_table.range_check_table)]
            });
        }

        let mut bit_signs = vec![];

        meta.create_gate("Eltwise 2d", |meta| -> Vec<Expression<F>> {
            let sel = meta.query_selector(selector);

            //iterate over all elements to the input
            let (expressions, bit_signs_col) = eltwise_inter.axis_iter(Self::COLUMN_AXIS).zip(inputs.iter()).zip(outputs.iter()).fold((vec![], vec![]), |(mut expressions, mut bit_signs), ((eltwise_inter, &input), &output)| {
                let mut eltwise_inter = eltwise_inter.to_vec();
                let bit_sign_col = eltwise_inter.remove(0);
                let base: u64 = BASE.try_into().unwrap();
                assert_eq!(
                    Self::ADVICE_LEN, eltwise_inter.len(),
                    "Must pass in sufficient advice columns for eltwise intermediate operations: passed in {}, need {}", 
                    eltwise_inter.len(), Self::ADVICE_LEN
                );
                let input = meta.query_advice(input, Rotation::cur());
                let output = meta.query_advice(output, Rotation::cur());
                let bit_sign = meta.query_advice(bit_sign_col, Rotation::cur());
                let iter = eltwise_inter.iter();
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
    
                let trunc_sum = iter
                    .clone().skip(K)
                    .enumerate()
                    .map(|(index, column)| {
                        let b = meta.query_advice(*column, Rotation::cur());
                        let true_base = (0..index).fold(F::from(1), |expr, _input| expr * base);
                        b * true_base
                    })
                    .reduce(|accum, item| accum + item)
                    .unwrap();
    
                let constant_1 = Expression::Constant(F::from(1));
                expressions.push(
                    sel.clone()
                        * (bit_sign.clone() * (input.clone() - word_sum.clone())
                            + (constant_1.clone() - bit_sign.clone()) * (input + word_sum)),
                );
                expressions.push(sel.clone() * ((bit_sign.clone()*(output.clone() - trunc_sum))+((constant_1 - bit_sign)*(output))));
    
                bit_signs.push(bit_sign_col);

                (expressions, bit_signs)
            });

            bit_signs = bit_signs_col;
            expressions
        });

        ReluNorm2dConfig {
            inputs,
            outputs,
            eltwise_inter,
            bit_signs: Array1::from_vec(bit_signs),
            selector,
            _marker: PhantomData,
        }
    }

    pub fn add_layer(
        &self,
        layouter: &mut impl Layouter<F>,
        inputs: &Array2<AssignedCell<F, F>>,
    ) -> Result<Array2<AssignedCell<F, F>>, PlonkError> {
        let base: u128 = BASE.try_into().unwrap();
        let config = &self.config;

        layouter.assign_region(
            || "apply 2d normalize",
            |mut region| {
                let outputs = inputs
                    .axis_iter(Self::ROW_AXIS)
                    .enumerate()
                    .map(|(offset, inputs)| {
                        self.config.selector.enable(&mut region, offset)?;
                        let outputs = inputs
                            .iter()
                            .zip(config.inputs.iter())
                            .zip(config.outputs.iter())
                            .zip(config.eltwise_inter.axis_iter(Self::COLUMN_AXIS))
                            .zip(config.bit_signs.iter())
                            .map(
                                |(
                                    (((input, &input_col), &output_col), eltwise_inter),
                                    &bit_sign_col,
                                )| {
                                    let value = input.copy_advice(
                                        || "eltwise input",
                                        &mut region,
                                        input_col,
                                        offset,
                                    )?;
                                    let bit_sign = value.value().map(|x| match *x < F::TWO_INV {
                                        false => 0,
                                        true => 1,
                                    });

                                    // let word_repr: Value<Vec<u32>> = output_i32.map(|x| {
                                    //     let str = format!("{:o}", x.abs());
                                    //     str.chars()
                                    //         .map(|char| char.to_digit(8).unwrap())
                                    //         .rev()
                                    //         .collect()
                                    // });

                                    let output_abs = value.value().map(|x| {
                                        let x = *x;
                                        if x < F::TWO_INV {
                                            x.get_lower_128()
                                        } else {
                                            x.neg().get_lower_128()
                                        }
                                    });
                                    let word_repr: Value<Vec<u16>> =
                                        output_abs.and_then(|mut x| {
                                            let mut result = vec![];

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
                                        || "eltwise_inter bit_sign",
                                        bit_sign_col,
                                        offset,
                                        || bit_sign.map(|x| F::from(x)),
                                    )?;
                                    let _: Vec<_> = (0..eltwise_inter.len() - 1)
                                        .map(|index_col| {
                                            region
                                                .assign_advice(
                                                    || "eltwise_inter word_repr",
                                                    eltwise_inter[index_col + 1],
                                                    offset,
                                                    || {
                                                        word_repr.clone().map(|x| {
                                                            match index_col >= x.len() {
                                                                false => {
                                                                    F::from(x[index_col] as u64)
                                                                }
                                                                true => F::from(0),
                                                            }
                                                        })
                                                    },
                                                )
                                                .unwrap()
                                        })
                                        .collect();
                                    region.assign_advice(
                                        || "eltwise_output",
                                        output_col,
                                        offset,
                                        || {
                                            value.value().map(|x| {
                                                let x = *x;
                                                if x < F::TWO_INV {
                                                    F::from_u128(
                                                        x.get_lower_128()
                                                            / u128::try_from(
                                                                BASE.pow(u32::try_from(K).unwrap()),
                                                            )
                                                            .unwrap(),
                                                    )
                                                } else {
                                                    F::zero()
                                                }
                                            })
                                        },
                                    )
                                },
                            )
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok::<_, PlonkError>(Array1::from_vec(outputs))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(stack(
                    Self::ROW_AXIS,
                    outputs
                        .iter()
                        .map(|item| item.view())
                        .collect::<Vec<_>>()
                        .as_slice(),
                )
                .unwrap())
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::nn_ops::lookup_ops::DecompTable;

    use super::{ReluNorm2dChip, ReluNorm2dConfig};
    use halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        halo2curves::bn256::Fr,
        plonk::{
            Advice, Circuit, Column, ConstraintSystem, Error as PlonkError, Instance,
        },
    };
    use ndarray::{stack, Array, Array1, Array2, Axis, Zip};

    #[derive(Clone, Debug)]
    struct ReluNorm2DTestConfig<F: FieldExt> {
        input: Array1<Column<Instance>>,
        input_advice: Array1<Column<Advice>>,
        output: Array1<Column<Instance>>,
        norm_chip: ReluNorm2dConfig<F>,
        range_table: DecompTable<F, 1024>,
    }

    struct ReluNorm2DTestCircuit<F: FieldExt> {
        pub input: Array2<Value<F>>,
    }

    const INPUT_WIDTH: usize = 8;
    const INPUT_HEIGHT: usize = 8;

    impl<F: FieldExt> Circuit<F> for ReluNorm2DTestCircuit<F> {
        type Config = ReluNorm2DTestConfig<F>;

        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                input: Array::from_shape_simple_fn((INPUT_WIDTH, INPUT_HEIGHT), || {
                    Value::unknown()
                }),
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let inputs = Array::from_shape_simple_fn(INPUT_WIDTH, || {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            });

            //let output_width = INPUT_WIDTH + PADDING_WIDTH * 2 - KERNAL_WIDTH + 1;

            let outputs = Array::from_shape_simple_fn(INPUT_WIDTH, || {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            });

            const ADVICE_LEN: usize = 10;

            let eltwise_inter = Array::from_shape_simple_fn((INPUT_WIDTH, ADVICE_LEN + 1), || {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            });

            let range_table: DecompTable<F, 1024> = DecompTable::configure(meta);

            let norm_chip = ReluNorm2dChip::<_, 1024, 2>::configure(
                meta,
                inputs,
                outputs,
                eltwise_inter,
                range_table.clone(),
            );

            ReluNorm2DTestConfig {
                input: Array::from_shape_simple_fn(INPUT_WIDTH, || {
                    let col = meta.instance_column();
                    meta.enable_equality(col);
                    col
                }),
                output: Array::from_shape_simple_fn(INPUT_WIDTH, || {
                    let col = meta.instance_column();
                    meta.enable_equality(col);
                    col
                }),
                input_advice: Array::from_shape_simple_fn(INPUT_WIDTH, || {
                    let col = meta.advice_column();
                    meta.enable_equality(col);
                    col
                }),
                norm_chip,
                range_table,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), PlonkError> {
            let norm_chip: ReluNorm2dChip<F, 1024, 2> =
                ReluNorm2dChip::construct(config.norm_chip);

            config
                .range_table
                .layout(layouter.namespace(|| "range check lookup table"))?;

            let inputs = layouter.assign_region(
                || "inputs",
                |mut region| {
                    let input = config.input.view();
                    let input_advice = config.input_advice.view();
                    let result = stack(
                        Axis(1),
                        &self
                            .input
                            .axis_iter(Axis(1))
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

            let output = norm_chip.add_layer(&mut layouter, &inputs)?;
            let input = config.output.view();
            for (row, slice) in output.axis_iter(Axis(1)).enumerate() {
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
    ///test that a simple 8x8 relu + normalization works
    fn test_simple_relu() -> Result<(), PlonkError> {
        let circuit = ReluNorm2DTestCircuit {
            input: Array::from_shape_simple_fn((INPUT_WIDTH, INPUT_HEIGHT), || {
                Value::known(Fr::from(1_048_576).neg())
            }),
        };

        let mut input_instance = vec![vec![Fr::from(1_048_576).neg(); INPUT_HEIGHT]; INPUT_WIDTH];
        let mut output_instance = vec![vec![Fr::zero(); INPUT_HEIGHT]; INPUT_WIDTH];

        input_instance.append(&mut output_instance);

        MockProver::run(11, &circuit, input_instance)
            .unwrap()
            .assert_satisfied();

        Ok(())
    }
}
