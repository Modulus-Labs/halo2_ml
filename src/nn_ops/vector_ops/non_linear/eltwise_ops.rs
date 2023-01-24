use std::{fmt::Debug, marker::PhantomData};

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter, Value},
    plonk::{
        Advice, Column, ConstraintSystem, Error as PlonkError, Expression, Expression::Constant,
        Selector,
    },
    poly::Rotation,
};

use crate::nn_ops::DefaultDecomp;

use super::super::super::lookup_ops::DecompTable;

pub trait EltwiseInstructions<F: FieldExt>: Clone + Debug + Chip<F> {
    ///apply the eltwise operation to an `AssignedCell`
    fn apply_elt(
        &self,
        layouter: impl Layouter<F>,
        input: AssignedCell<F, F>,
    ) -> Result<AssignedCell<F, F>, PlonkError>;

    fn construct(config: DecompConfig<F>) -> Self;
}

///Eltwise Op for RELU
#[derive(Clone, Debug)]
pub struct DecompConfig<F: FieldExt> {
    bit_sign: Column<Advice>,
    decomp: Vec<Column<Advice>>,
    input: Column<Advice>,
    output: Column<Advice>,
    selector: Selector,
    _marker: PhantomData<F>,
}

#[derive(Clone, Debug)]
pub struct ReluChip<F: FieldExt, const BASE: usize> {
    config: DecompConfig<F>,
}

impl<F: FieldExt, const BASE: usize> Chip<F> for ReluChip<F, BASE> {
    type Config = DecompConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: FieldExt, const BASE: usize> ReluChip<F, BASE> {
    const ADVICE_LEN: usize = 10;

    pub fn construct(config: DecompConfig<F>) -> Self {
        Self { config }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        input: Column<Advice>,
        mut eltwise_inter: Vec<Column<Advice>>,
        eltwise_output: Column<Advice>,
        range_table: DecompTable<F, DefaultDecomp>,
    ) -> DecompConfig<F> {
        let selector = meta.complex_selector();

        let bit_sign_col = eltwise_inter.remove(0);

        //range check with lookup table for all words items
        for item in eltwise_inter.clone() {
            meta.lookup("lookup", |meta| {
                let s_elt = meta.query_selector(selector);
                let word = meta.query_advice(item, Rotation::cur());
                vec![(s_elt * word, range_table.range_check_table)]
            });
        }

        //check sum of word decomp = abs(item)
        meta.create_gate("Relu Gate", |meta | {
            let base: u64 = BASE.try_into().unwrap();
            assert_eq!(
                Self::ADVICE_LEN, eltwise_inter.len(),
                "Must pass in sufficient advice columns for eltwise intermediate operations: passed in {}, need {}", 
                Self::ADVICE_LEN, eltwise_inter.len()
            );
            let input = meta.query_advice(input, Rotation::cur());
            let bit_sign = meta.query_advice(bit_sign_col, Rotation::cur());
            let iter = eltwise_inter.iter();
            let base = F::from(base);
            let s_elt = meta.query_selector(selector);
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

            let mut expr: Vec<Expression<F>> = Vec::new();
            let constant_1 = Constant(F::from(1));
            let output = meta.query_advice(eltwise_output, Rotation::cur());
            expr.push(
                s_elt.clone()
                    * (bit_sign.clone() * (input.clone() - word_sum.clone())
                        + (constant_1.clone() - bit_sign.clone()) * (input + word_sum.clone())),
            );
            expr.push(s_elt * ((bit_sign.clone()*(output.clone() - word_sum))+((constant_1 - bit_sign)*output)));
            expr
        });
        DecompConfig {
            bit_sign: bit_sign_col,
            input,
            decomp: eltwise_inter,
            output: eltwise_output,
            selector,
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt, const BASE: usize> EltwiseInstructions<F> for ReluChip<F, BASE> {
    fn construct(config: DecompConfig<F>) -> Self {
        Self { config }
    }

    fn apply_elt(
        &self,
        mut layouter: impl Layouter<F>,
        input: AssignedCell<F, F>,
    ) -> Result<AssignedCell<F, F>, PlonkError> {
        let base: u128 = BASE.try_into().unwrap();
        layouter.assign_region(
            || "apply ReLu",
            |mut region| {
                let offset = 0;
                let value = input.copy_advice(
                    || "eltwise input",
                    &mut region,
                    self.config.input,
                    offset,
                )?;
                self.config.selector.enable(&mut region, offset)?;
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
                let word_repr: Value<Vec<u16>> = output_abs.and_then(|mut x| {
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
                    self.config.bit_sign,
                    offset,
                    || bit_sign.map(|x| F::from(x)),
                )?;
                let _: Vec<_> = (0..self.config.decomp.len())
                    .map(|index_col| {
                        region
                            .assign_advice(
                                || "eltwise_inter word_repr",
                                self.config.decomp[index_col],
                                offset,
                                || {
                                    word_repr.clone().map(|x| match index_col >= x.len() {
                                        false => F::from(x[index_col] as u64),
                                        true => F::from(0),
                                    })
                                },
                            )
                            .unwrap()
                    })
                    .collect();
                region.assign_advice(
                    || "eltwise_output",
                    self.config.output,
                    offset,
                    || {
                        value.value().map(|x| {
                            let x = *x;
                            if x < F::TWO_INV {
                                x
                            } else {
                                F::zero()
                            }
                        })
                    },
                )
            },
        )
    }
}

///Config for `NormalizeChip`
#[derive(Clone, Debug)]
pub struct NormalizeConfig<F: FieldExt, const BASE: usize, const K: usize> {
    bit_sign: Column<Advice>,
    decomp: Vec<Column<Advice>>,
    input: Column<Advice>,
    output: Column<Advice>,
    selector: Selector,
    _marker: PhantomData<F>,
}

///Chip for eltwise Division
/// - `BASE` is word size for decomposition
/// - division is by `BASE^K`
#[derive(Clone, Debug)]
pub struct NormalizeChip<F: FieldExt, const BASE: usize, const K: usize> {
    config: DecompConfig<F>,
}

impl<F: FieldExt, const BASE: usize, const K: usize> Chip<F> for NormalizeChip<F, BASE, K> {
    type Config = DecompConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: FieldExt, const BASE: usize, const K: usize> NormalizeChip<F, BASE, K> {
    const ADVICE_LEN: usize = 10;

    pub fn construct(config: DecompConfig<F>) -> Self {
        Self { config }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        input: Column<Advice>,
        mut eltwise_inter: Vec<Column<Advice>>,
        eltwise_output: Column<Advice>,
        range_table: DecompTable<F, DefaultDecomp>,
    ) -> DecompConfig<F> {
        let selector = meta.complex_selector();

        let bit_sign_col = eltwise_inter.remove(0);

        //range check with lookup table for all words items
        for item in eltwise_inter.clone() {
            meta.lookup("lookup", |meta| {
                let s_elt = meta.query_selector(selector);
                let word = meta.query_advice(item, Rotation::cur());
                vec![(s_elt * word, range_table.range_check_table)]
            });
        }

        //check sum of word decomp = abs(item)
        meta.create_gate("Normalize Gate", |meta | {
            let base: u64 = BASE.try_into().unwrap();
            assert_eq!(
                Self::ADVICE_LEN, eltwise_inter.len(),
                "Must pass in sufficient advice columns for eltwise intermediate operations: passed in {}, need {}", 
                Self::ADVICE_LEN, eltwise_inter.len()
            );
            let input = meta.query_advice(input, Rotation::cur());
            let bit_sign = meta.query_advice(bit_sign_col, Rotation::cur());
            let iter = eltwise_inter.iter();
            let base = F::from(base);
            let s_elt = meta.query_selector(selector);
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

            let mut expr: Vec<Expression<F>> = Vec::new();
            let constant_1 = Constant(F::from(1));
            let output = meta.query_advice(eltwise_output, Rotation::cur());
            expr.push(
                s_elt.clone()
                    * (bit_sign.clone() * (input.clone() - word_sum.clone())
                        + (constant_1.clone() - bit_sign.clone()) * (input + word_sum)),
            );
            expr.push(s_elt * ((bit_sign.clone()*(output.clone() - trunc_sum.clone()))+((constant_1 - bit_sign)*(output + trunc_sum))));
            expr
        });
        DecompConfig {
            bit_sign: bit_sign_col,
            input,
            decomp: eltwise_inter,
            output: eltwise_output,
            selector,
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt, const BASE: usize, const K: usize> EltwiseInstructions<F>
    for NormalizeChip<F, BASE, K>
{
    fn construct(config: DecompConfig<F>) -> Self {
        Self { config }
    }

    fn apply_elt(
        &self,
        mut layouter: impl Layouter<F>,
        input: AssignedCell<F, F>,
    ) -> Result<AssignedCell<F, F>, PlonkError> {
        let base: u128 = BASE.try_into().unwrap();
        layouter.assign_region(
            || "apply Normalize",
            |mut region| {
                let offset = 0;
                let value = input.copy_advice(
                    || "eltwise input",
                    &mut region,
                    self.config.input,
                    offset,
                )?;
                self.config.selector.enable(&mut region, offset)?;
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
                let word_repr: Value<Vec<u16>> = output_abs.and_then(|mut x| {
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
                    self.config.bit_sign,
                    offset,
                    || bit_sign.map(|x| F::from(x)),
                )?;
                let _: Vec<_> = (0..self.config.decomp.len())
                    .map(|index_col| {
                        region
                            .assign_advice(
                                || "eltwise_inter word_repr",
                                self.config.decomp[index_col],
                                offset,
                                || {
                                    word_repr.clone().map(|x| match index_col >= x.len() {
                                        false => F::from(x[index_col] as u64),
                                        true => F::from(0),
                                    })
                                },
                            )
                            .unwrap()
                    })
                    .collect();
                region.assign_advice(
                    || "eltwise_output",
                    self.config.output,
                    offset,
                    || {
                        value.value().map(|x| {
                            let x = *x;
                            if x < F::TWO_INV {
                                F::from_u128(
                                    x.get_lower_128()
                                        / u128::try_from(BASE.pow(u32::try_from(K).unwrap()))
                                            .unwrap(),
                                )
                            } else {
                                F::from_u128(
                                    x.neg().get_lower_128()
                                        / u128::try_from(BASE.pow(u32::try_from(K).unwrap()))
                                            .unwrap(),
                                )
                                .neg()
                            }
                        })
                    },
                )
            },
        )
    }
}

///Config for `NormalizeReluChip`
#[derive(Clone, Debug)]
pub struct NormalizeReluConfig<F: FieldExt, const BASE: usize, const K: usize> {
    bit_sign: Column<Advice>,
    decomp: Vec<Column<Advice>>,
    input: Column<Advice>,
    output: Column<Advice>,
    selector: Selector,
    _marker: PhantomData<F>,
}

///Chip for eltwise Relu+Division
/// - `BASE` is word size for decomposition
/// - division is by `BASE^K`
#[derive(Clone, Debug)]
pub struct NormalizeReluChip<F: FieldExt, const BASE: usize, const K: usize> {
    config: DecompConfig<F>,
}

impl<F: FieldExt, const BASE: usize, const K: usize> Chip<F> for NormalizeReluChip<F, BASE, K> {
    type Config = DecompConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: FieldExt, const BASE: usize, const K: usize> NormalizeReluChip<F, BASE, K> {
    const ADVICE_LEN: usize = 10;

    pub fn construct(config: DecompConfig<F>) -> Self {
        Self { config }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        input: Column<Advice>,
        mut eltwise_inter: Vec<Column<Advice>>,
        eltwise_output: Column<Advice>,
        range_table: DecompTable<F, DefaultDecomp>,
    ) -> DecompConfig<F> {
        let selector = meta.complex_selector();

        let bit_sign_col = eltwise_inter.remove(0);

        //range check with lookup table for all words items
        for item in eltwise_inter.clone() {
            meta.lookup("lookup", |meta| {
                let s_elt = meta.query_selector(selector);
                let word = meta.query_advice(item, Rotation::cur());
                vec![(s_elt * word, range_table.range_check_table)]
            });
        }

        //check sum of word decomp = abs(item)
        meta.create_gate("Relu+Normalize Gate", |meta | {
            let base: u64 = BASE.try_into().unwrap();
            assert_eq!(
                Self::ADVICE_LEN, eltwise_inter.len(),
                "Must pass in sufficient advice columns for eltwise intermediate operations: passed in {}, need {}", 
                Self::ADVICE_LEN, eltwise_inter.len()
            );
            let input = meta.query_advice(input, Rotation::cur());
            let bit_sign = meta.query_advice(bit_sign_col, Rotation::cur());
            let iter = eltwise_inter.iter();
            let base = F::from(base);
            let s_elt = meta.query_selector(selector);
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

            let mut expr: Vec<Expression<F>> = Vec::new();
            let constant_1 = Constant(F::from(1));
            let output = meta.query_advice(eltwise_output, Rotation::cur());
            expr.push(
                s_elt.clone()
                    * (bit_sign.clone() * (input.clone() - word_sum.clone())
                        + (constant_1.clone() - bit_sign.clone()) * (input + word_sum)),
            );
            expr.push(s_elt * ((bit_sign.clone()*(output.clone() - trunc_sum))+((constant_1 - bit_sign)*output)));
            expr
        });
        DecompConfig {
            bit_sign: bit_sign_col,
            input,
            decomp: eltwise_inter,
            output: eltwise_output,
            selector,
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt, const BASE: usize, const K: usize> EltwiseInstructions<F>
    for NormalizeReluChip<F, BASE, K>
{
    fn construct(config: DecompConfig<F>) -> Self {
        Self { config }
    }

    fn apply_elt(
        &self,
        mut layouter: impl Layouter<F>,
        input: AssignedCell<F, F>,
    ) -> Result<AssignedCell<F, F>, PlonkError> {
        let base: u128 = BASE.try_into().unwrap();
        layouter.assign_region(
            || "apply ReLu+Normalize",
            |mut region| {
                let offset = 0;
                let value = input.copy_advice(
                    || "eltwise input",
                    &mut region,
                    self.config.input,
                    offset,
                )?;
                self.config.selector.enable(&mut region, offset)?;
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
                let word_repr: Value<Vec<u16>> = output_abs.and_then(|mut x| {
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
                    self.config.bit_sign,
                    offset,
                    || bit_sign.map(|x| F::from(x)),
                )?;
                let _: Vec<_> = (0..self.config.decomp.len())
                    .map(|index_col| {
                        region
                            .assign_advice(
                                || "eltwise_inter word_repr",
                                self.config.decomp[index_col],
                                offset,
                                || {
                                    word_repr.clone().map(|x| match index_col >= x.len() {
                                        false => F::from(x[index_col] as u64),
                                        true => F::from(0),
                                    })
                                },
                            )
                            .unwrap()
                    })
                    .collect();
                region.assign_advice(
                    || "eltwise_output",
                    self.config.output,
                    offset,
                    || {
                        value.value().map(|x| {
                            let x = *x;
                            if x < F::TWO_INV {
                                F::from_u128(
                                    x.get_lower_128()
                                        / u128::try_from(BASE.pow(u32::try_from(K).unwrap()))
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
    }
}
