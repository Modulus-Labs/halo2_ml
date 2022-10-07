use std::{fmt::Debug, marker::PhantomData};

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Region, Value},
    plonk::{
        Advice, Assigned, Column, ConstraintSystem, Error as PlonkError, Expression,
        Expression::Constant, Selector,
    },
    poly::Rotation,
};
use icecream::ic;

use crate::nn_chip::{felt_to_i128, Number};

use super::lookup_ops::DecompTable;

pub trait EltwiseOp<F: FieldExt, const BASE: usize>: Sized + Clone + Debug {
    const ADVICE_LEN: usize;
    fn configure(
        meta: &mut ConstraintSystem<F>,
        input: Column<Advice>,
        eltwise_inter: Vec<Column<Advice>>,
        eltwise_output: Column<Advice>,
        range_table: DecompTable<F, BASE>,
    ) -> Self;

    fn layout(
        &self,
        region: &mut Region<F>,
        mat_output: Vec<AssignedCell<Assigned<F>, F>>,
        offset: usize,
    ) -> Result<Vec<Number<F>>, PlonkError>;
}

///Eltwise Op for RELU
#[derive(Clone, Debug)]
pub struct ReluConfig<F: FieldExt, const BASE: usize> {
    bit_sign: Column<Advice>,
    decomp: Vec<Column<Advice>>,
    output: Column<Advice>,
    pub range_table: DecompTable<F, BASE>,
    selector: Selector,
    _marker: PhantomData<F>,
}

impl<F: FieldExt, const BASE: usize> EltwiseOp<F, BASE> for ReluConfig<F, BASE> {
    const ADVICE_LEN: usize = 21;
    fn configure(
        meta: &mut ConstraintSystem<F>,
        input: Column<Advice>,
        mut eltwise_inter: Vec<Column<Advice>>,
        eltwise_output: Column<Advice>,
        range_table: DecompTable<F, BASE>,
    ) -> ReluConfig<F, BASE> {
        let selector = meta.complex_selector();

        let bit_sign_col = eltwise_inter.remove(0);

        //range check with lookup table for all words items
        for item in eltwise_inter.clone() {
            meta.lookup(|meta| {
                let s_elt = meta.query_selector(selector);
                let word = meta.query_advice(item, Rotation::cur());
                vec![(s_elt * word, range_table.range_check_table)]
            });
        }

        //check sum of word decomp = abs(item)
        meta.create_gate("eltwise op", |meta | {
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
            let octal_sum = iter
                .clone()
                .enumerate()
                .map(|(index, column)| {
                    let b = meta.query_advice(*column, Rotation::cur());
                    let true_base = (0..index).fold(F::from(1), |expr, _input| expr * base);
                    b * true_base
                })
                .reduce(|accum, item| accum + item)
                .unwrap();

            // let mut expr: Vec<Expression<F>> = iter
            //     .map(|oct| {
            //         let b = meta.query_advice(*oct, Rotation::cur());
            //         s_elt.clone()
            //             * (b.clone())
            //             * (Constant(F::from(1)) - b.clone())
            //             * (Constant(F::from(1)) - b.clone())
            //             * (Constant(F::from(2)) - b.clone())
            //             * (Constant(F::from(3)) - b.clone())
            //             * (Constant(F::from(4)) - b.clone())
            //             * (Constant(F::from(5)) - b.clone())
            //             * (Constant(F::from(6)) - b.clone())
            //             * (Constant(F::from(7)) - b)
            //     })
            //     .collect();
            let mut expr: Vec<Expression<F>> = Vec::new();
            let constant_1 = Constant(F::from(1));
            let output = meta.query_advice(eltwise_output, Rotation::cur());
            expr.push(
                s_elt.clone()
                    * (bit_sign.clone() * (input.clone() - octal_sum.clone())
                        + (constant_1.clone() - bit_sign.clone()) * (input + octal_sum.clone())),
            );
            expr.push(s_elt * ((bit_sign.clone()*(output.clone() - octal_sum))+((constant_1 - bit_sign)*output)));
            expr
        });
        ReluConfig {
            bit_sign: bit_sign_col,
            decomp: eltwise_inter,
            output: eltwise_output,
            range_table,
            selector,
            _marker: PhantomData,
        }
    }

    fn layout(
        &self,
        region: &mut Region<F>,
        mat_output: Vec<AssignedCell<Assigned<F>, F>>,
        offset: usize,
    ) -> Result<Vec<Number<F>>, PlonkError> {
        let base: u128 = BASE.try_into().unwrap();
        Ok::<_, PlonkError>(
            mat_output
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    self.selector.enable(region, offset + index)?;
                    let output_i32 = value.value().map(|f| felt_to_i128(f.evaluate()));
                    let bit_sign = value.value().map(|x| {
                        match x.evaluate() < F::TWO_INV {
                            false => 0,
                            true => 1,
                    }});

                    // let word_repr: Value<Vec<u32>> = output_i32.map(|x| {
                    //     let str = format!("{:o}", x.abs());
                    //     str.chars()
                    //         .map(|char| char.to_digit(8).unwrap())
                    //         .rev()
                    //         .collect()
                    // });

                    let output_abs = value.value().map(|x| {
                        let x = x.evaluate();
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
                            x = x / base;

                            result.push(m as u16);
                            if x == 0 {
                                break;
                            }
                        }

                        Value::known(result)
                    });
                    region.assign_advice(
                        || "eltwise_inter bit_sign",
                        self.bit_sign,
                        offset + index,
                        || bit_sign.map(|x| F::from(x)),
                    )?;
                    let _: Vec<_> = (0..self.decomp.len())
                        .map(|index_col| {
                            region
                                .assign_advice(
                                    || "eltwise_inter word_repr",
                                    self.decomp[index_col],
                                    offset + index,
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
                    region
                        .assign_advice(
                            || "eltwise_output",
                            self.output,
                            offset + index,
                            || value.value().map(|x| {
                                let x = x.evaluate();
                                if x < F::TWO_INV {
                                    x
                                } else {
                                    F::zero()
                                }        
                            }),
                        )
                        .map(Number)
                })
                .collect(),
        )?
    }
}

///Eltwise Op for Normalization (RELU+TRUNC_DIV)
#[derive(Clone, Debug)]
pub struct NormalizeConfig<F: FieldExt, const BASE: usize> {
    bit_sign: Column<Advice>,
    decomp: Vec<Column<Advice>>,
    output: Column<Advice>,
    pub range_table: DecompTable<F, BASE>,
    selector: Selector,
    _marker: PhantomData<F>,
}

impl<F: FieldExt, const BASE: usize> EltwiseOp<F, BASE> for NormalizeConfig<F, BASE> {
    const ADVICE_LEN: usize = 21;
    fn configure(
        meta: &mut ConstraintSystem<F>,
        input: Column<Advice>,
        mut eltwise_inter: Vec<Column<Advice>>,
        eltwise_output: Column<Advice>,
        range_table: DecompTable<F, BASE>,
    ) -> NormalizeConfig<F, BASE> {
        let selector = meta.complex_selector();

        let bit_sign_col = eltwise_inter.remove(0);

        //range check with lookup table for all words items
        for item in eltwise_inter.clone() {
            meta.lookup(|meta| {
                let s_elt = meta.query_selector(selector);
                let word = meta.query_advice(item, Rotation::cur());
                vec![(s_elt * word, range_table.range_check_table)]
            });
        }

        //check sum of word decomp = abs(item)
        meta.create_gate("eltwise op", |meta | {
            let base: u64 = BASE.try_into().unwrap();
            assert_eq!(
                Self::ADVICE_LEN, eltwise_inter.len(),
                "Must pass in sufficient advice columns for eltwise intermediate operations: need {}, passed in {}", 
                Self::ADVICE_LEN, eltwise_inter.len()
            );
            let input = meta.query_advice(input, Rotation::cur());
            let bit_sign = meta.query_advice(bit_sign_col, Rotation::cur());
            let iter = eltwise_inter.iter();
            let base = F::from(base);
            let s_elt = meta.query_selector(selector);
            let octal_sum = iter
                .clone()
                .enumerate()
                .map(|(index, column)| {
                    let b = meta.query_advice(*column, Rotation::cur());
                    let true_base = (0..index).fold(base, |expr, _input| expr * base);
                    b * true_base
                })
                .reduce(|accum, item| accum + item)
                .unwrap();

            // let mut expr: Vec<Expression<F>> = iter
            //     .map(|oct| {
            //         let b = meta.query_advice(*oct, Rotation::cur());
            //         s_elt.clone()
            //             * (b.clone())
            //             * (Constant(F::from(1)) - b.clone())
            //             * (Constant(F::from(1)) - b.clone())
            //             * (Constant(F::from(2)) - b.clone())
            //             * (Constant(F::from(3)) - b.clone())
            //             * (Constant(F::from(4)) - b.clone())
            //             * (Constant(F::from(5)) - b.clone())
            //             * (Constant(F::from(6)) - b.clone())
            //             * (Constant(F::from(7)) - b)
            //     })
            //     .collect();
            let mut expr: Vec<Expression<F>> = Vec::new();
            let constant_1 = Constant(F::from(1));
            let output = meta.query_advice(eltwise_output, Rotation::cur());
            expr.push(
                s_elt.clone()
                    * (bit_sign.clone() * (input.clone() - octal_sum.clone())
                        + (constant_1 - bit_sign) * (input + octal_sum.clone())),
            );
            expr.push(s_elt * (output - octal_sum));
            expr
        });
        NormalizeConfig {
            bit_sign: bit_sign_col,
            decomp: eltwise_inter,
            output: eltwise_output,
            range_table,
            selector,
            _marker: PhantomData,
        }
    }

    fn layout(
        &self,
        region: &mut Region<F>,
        mat_output: Vec<AssignedCell<Assigned<F>, F>>,
        offset: usize,
    ) -> Result<Vec<Number<F>>, PlonkError> {
        let base: i128 = BASE.try_into().unwrap();
        ic!(mat_output);
        Ok::<_, PlonkError>(
            mat_output
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    self.selector.enable(region, offset + index)?;
                    let output_i32 = value.value().map(|f| felt_to_i128(f.evaluate()));
                    let bit_sign = output_i32.map(|x| match x >= 0 {
                        false => 0,
                        true => 1,
                    });
                    println!("eh!");
                    ic!(bit_sign);
                    ic!(output_i32);
                    // let word_repr: Value<Vec<u32>> = output_i32.map(|x| {
                    //     let str = format!("{:o}", x.abs());
                    //     str.chars()
                    //         .map(|char| char.to_digit(8).unwrap())
                    //         .rev()
                    //         .collect()
                    // });
                    let word_repr: Value<Vec<u16>> = output_i32.and_then(|mut x| {
                        let mut result = vec![];

                        loop {
                            let m = x % base;
                            x = x / base;

                            result.push(m as u16);
                            if x == 0 {
                                break;
                            }
                        }

                        Value::known(result)
                    });
                    region.assign_advice(
                        || "eltwise_inter bit_sign",
                        self.bit_sign,
                        offset + index,
                        || bit_sign.map(|x| F::from(x)),
                    )?;
                    let _: Vec<_> = (0..self.decomp.len())
                        .map(|index| {
                            region
                                .assign_advice(
                                    || "eltwise_inter word_repr",
                                    self.decomp[index],
                                    offset + index,
                                    || {
                                        word_repr.clone().map(|x| match index > x.len() {
                                            false => F::from(x[index] as u64),
                                            true => F::from(0),
                                        })
                                    },
                                )
                                .unwrap()
                        })
                        .collect();
                    region
                        .assign_advice(
                            || "eltwise_output",
                            self.output,
                            offset + index,
                            || output_i32.map(|x| F::from(x.unsigned_abs() as u64)),
                        )
                        .map(Number)
                })
                .collect(),
        )?
    }
}