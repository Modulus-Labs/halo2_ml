use std::{fmt::Debug, marker::PhantomData};

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter, Value},
    plonk::{
        Advice, Column, ConstraintSystem, Error as PlonkError, Expression,
        Expression::Constant, Selector,
    },
    poly::Rotation,
};

use super::lookup_ops::DecompTable;

pub trait EltwiseInstructions<F: FieldExt>: Clone + Debug {
    ///apply the eltwise operation to an `AssignedCell`
    fn apply_elt(
        &self,
        layouter: impl Layouter<F>,
        input: AssignedCell<F, F>,
    ) -> Result<AssignedCell<F, F>, PlonkError>;
}

///Eltwise Op for RELU
#[derive(Clone, Debug)]
pub struct ReluConfig<F: FieldExt, const BASE: usize> {
    bit_sign: Column<Advice>,
    decomp: Vec<Column<Advice>>,
    input: Column<Advice>,
    output: Column<Advice>,
    selector: Selector,
    _marker: PhantomData<F>,
}

#[derive(Clone, Debug)]
pub struct ReluChip<F: FieldExt, const BASE: usize> {
    config: ReluConfig<F, BASE>,
}

impl<F: FieldExt, const BASE: usize> Chip<F> for ReluChip<F, BASE> {
    type Config = ReluConfig<F, BASE>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: FieldExt, const BASE: usize> ReluChip<F, BASE> {
    const ADVICE_LEN: usize = 11;

    pub fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self { config }
    }

    pub fn configure(
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
            input,
            decomp: eltwise_inter,
            output: eltwise_output,
            selector,
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt, const BASE: usize> EltwiseInstructions<F> for ReluChip<F, BASE> {
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
