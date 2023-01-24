use std::marker::PhantomData;

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, Error as PlonkError, Expression, Fixed, Selector},
    poly::Rotation,
};

use ndarray::{stack, Array, Array1, Array2, Array3, Axis};

use crate::nn_ops::{lookup_ops::DecompTable, ColumnAllocator, DecompConfig, NNLayer};

#[derive(Clone, Debug)]
pub struct Normalize2dConfig<F: FieldExt> {
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
pub struct Normalize2dChip<F: FieldExt> {
    config: Normalize2dConfig<F>,
}

impl<F: FieldExt> Chip<F> for Normalize2dChip<F> {
    type Config = Normalize2dConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

pub struct Normalize2DChipConfig<F: FieldExt, Decomp: DecompConfig> {
    pub input_height: usize,
    pub input_width: usize,
    pub input_depth: usize,
    pub range_table: DecompTable<F, Decomp>,
}

impl<F: FieldExt> NNLayer<F> for Normalize2dChip<F> {
    type ConfigParams = Normalize2DChipConfig<F, Self::DecompConfig>;

    type LayerInput = Array3<AssignedCell<F, F>>;

    type LayerOutput = Array3<AssignedCell<F, F>>;

    fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self { config }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        config: Self::ConfigParams,
        advice_allocator: &mut ColumnAllocator<Advice>,
        _: &mut ColumnAllocator<Fixed>, // inputs: Array1<Column<Advice>>,
                                        // outputs: Array1<Column<Advice>>,
                                        // eltwise_inter: Array2<Column<Advice>>,
                                        // range_table: DecompTable<F, { Self::DecompConfig::BASE }>,
    ) -> <Self as Chip<F>>::Config {
        let selector = meta.complex_selector();
        let input_width = config.input_width;
        let range_table = config.range_table;

        let advice = advice_allocator.take(
            meta,
            config.input_width * (2 + Self::DecompConfig::ADVICE_LEN + 1),
        );
        let inputs = Array1::from_vec(advice[0..input_width].to_vec());
        let outputs = Array1::from_vec(advice[input_width..input_width * 2].to_vec());

        let bit_signs = Array1::from_vec(advice[input_width * 2..input_width * 3].to_vec());

        let eltwise_inter = advice[input_width * 3..advice.len()].to_vec();

        let eltwise_inter =
            Array::from_shape_vec((input_width, Self::DecompConfig::ADVICE_LEN), eltwise_inter)
                .unwrap();

        for &item in eltwise_inter.iter() {
            meta.lookup("lookup", |meta| {
                let s_elt = meta.query_selector(selector);
                let word = meta.query_advice(item, Rotation::cur());
                vec![(s_elt * word, range_table.range_check_table)]
            });
        }

        meta.create_gate("Eltwise 2d", |meta| -> Vec<Expression<F>> {
            let sel = meta.query_selector(selector);

            //iterate over all elements to the input
            let expressions = eltwise_inter
                .axis_iter(Axis(0))
                .zip(inputs.iter())
                .zip(outputs.iter())
                .zip(bit_signs.iter())
                .fold(
                    vec![],
                    |mut expressions, (((eltwise_inter, &input), &output), &bit_sign)| {
                        let base: u64 = Self::DecompConfig::BASE.try_into().unwrap();

                        let input = meta.query_advice(input, Rotation::cur());
                        let output = meta.query_advice(output, Rotation::cur());
                        let bit_sign = meta.query_advice(bit_sign, Rotation::cur());
                        let iter = eltwise_inter.iter();
                        let base = F::from(base);
                        let word_sum = iter
                            .clone()
                            .enumerate()
                            .map(|(index, column)| {
                                let b = meta.query_advice(*column, Rotation::cur());
                                let true_base =
                                    (0..index).fold(F::from(1), |expr, _input| expr * base);
                                b * true_base
                            })
                            .reduce(|accum, item| accum + item)
                            .unwrap();

                        let trunc_sum = iter
                            .clone()
                            .skip(Self::DecompConfig::K)
                            .enumerate()
                            .map(|(index, column)| {
                                let b = meta.query_advice(*column, Rotation::cur());
                                let true_base =
                                    (0..index).fold(F::from(1), |expr, _input| expr * base);
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
                        expressions.push(
                            sel.clone()
                                * ((bit_sign.clone() * (output.clone() - trunc_sum.clone()))
                                    + ((constant_1 - bit_sign) * (output + trunc_sum))),
                        );

                        expressions
                    },
                );

            expressions
        });

        Normalize2dConfig {
            inputs,
            outputs,
            eltwise_inter,
            bit_signs,
            selector,
            _marker: PhantomData,
        }
    }

    fn add_layer(
        &self,
        layouter: &mut impl Layouter<F>,
        inputs: Array3<AssignedCell<F, F>>,
        _: (),
    ) -> Result<Array3<AssignedCell<F, F>>, PlonkError> {
        let base: u128 = Self::DecompConfig::BASE.try_into().unwrap();
        let config = &self.config;

        let output: Result<Vec<_>, _> = inputs.axis_iter(Self::DEPTH_AXIS).map(|inputs| {
        layouter.assign_region(
            || "apply 2d normalize",
            |mut region| {
                let outputs = inputs
                    .axis_iter(Axis(1))
                    .enumerate()
                    .map(|(offset, inputs)| {
                        self.config.selector.enable(&mut region, offset)?;
                        let outputs = inputs
                            .iter()
                            .zip(config.inputs.iter())
                            .zip(config.outputs.iter())
                            .zip(config.eltwise_inter.axis_iter(Axis(0)))
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
                                    let _: Vec<_> = (0..eltwise_inter.len())
                                        .map(|index_col| {
                                            region
                                                .assign_advice(
                                                    || "eltwise_inter word_repr",
                                                    eltwise_inter[index_col],
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
                                                                Self::DecompConfig::BASE.pow(u32::try_from(Self::DecompConfig::K).unwrap()),
                                                            )
                                                            .unwrap(),
                                                    )
                                                } else {
                                                    F::from_u128(
                                                        x.neg().get_lower_128()
                                                            / u128::try_from(
                                                                Self::DecompConfig::BASE.pow(u32::try_from(Self::DecompConfig::K).unwrap()),
                                                            )
                                                            .unwrap(),
                                                    )
                                                    .neg()
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
                    Axis(1),
                    outputs
                        .iter()
                        .map(|item| item.view())
                        .collect::<Vec<_>>()
                        .as_slice(),
                )
                .unwrap())
            },
        )}).collect();

        Ok(stack(
            Axis(0),
            output?
                .iter()
                .map(|item| item.view())
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .unwrap())
    }
}

#[cfg(test)]
mod tests {
    use crate::nn_ops::{
        lookup_ops::DecompTable, matrix_ops::non_linear::norm_2d::Normalize2DChipConfig,
        ColumnAllocator, DefaultDecomp, NNLayer,
    };

    use super::{Normalize2dChip, Normalize2dConfig};
    use halo2_curves::bn256::Bn256;
    use halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        halo2curves::bn256::Fr,
        plonk::{
            create_proof, keygen_pk, keygen_vk, Advice, Circuit, Column, ConstraintSystem,
            Error as PlonkError, Fixed, Instance,
        },
        poly::{
            commitment::ParamsProver,
            kzg::{commitment::ParamsKZG, multiopen::ProverSHPLONK},
        },
        transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
    };
    use ndarray::{stack, Array, Array1, Array2, Array3, Axis, Zip};
    use rand::rngs::OsRng;

    #[derive(Clone, Debug)]
    struct Norm2DTestConfig<F: FieldExt> {
        input: Array2<Column<Instance>>,
        input_advice: Array2<Column<Advice>>,
        output: Array2<Column<Instance>>,
        norm_chip: Normalize2dConfig<F>,
        range_table: DecompTable<F, DefaultDecomp>,
    }

    struct Norm2DTestCircuit<F: FieldExt> {
        pub input: Array3<Value<F>>,
    }

    const INPUT_WIDTH: usize = 8;
    const INPUT_HEIGHT: usize = 8;

    const DEPTH: usize = 4;

    impl<F: FieldExt> Circuit<F> for Norm2DTestCircuit<F> {
        type Config = Norm2DTestConfig<F>;

        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                input: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH, INPUT_HEIGHT), || {
                    Value::unknown()
                }),
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let range_table = DecompTable::configure(meta);

            let config = Normalize2DChipConfig {
                input_height: INPUT_HEIGHT,
                input_width: INPUT_WIDTH,
                input_depth: DEPTH,
                range_table: range_table.clone(),
            };

            let mut advice_allocator = ColumnAllocator::<Advice>::new(meta, 1);
            let mut fixed_allocator = ColumnAllocator::<Fixed>::new(meta, 0);

            let norm_chip = Normalize2dChip::configure(
                meta,
                config,
                &mut advice_allocator,
                &mut fixed_allocator,
            );

            Norm2DTestConfig {
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
                norm_chip,
                range_table,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), PlonkError> {
            let norm_chip: Normalize2dChip<F> = Normalize2dChip::construct(config.norm_chip);

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

            let output = norm_chip.add_layer(&mut layouter, inputs, ())?;
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
    ///test that a simple 8x8 normalization works
    fn test_simple_norm() -> Result<(), PlonkError> {
        let circuit = Norm2DTestCircuit {
            input: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH, INPUT_HEIGHT), || {
                Value::known(Fr::from(1_048_576))
            }),
        };

        let mut input_instance = vec![vec![Fr::from(1_048_576); INPUT_HEIGHT]; INPUT_WIDTH * DEPTH];
        let mut output_instance = vec![vec![Fr::one(); INPUT_HEIGHT]; INPUT_WIDTH * DEPTH];

        input_instance.append(&mut output_instance);

        MockProver::run(11, &circuit, input_instance)
            .unwrap()
            .assert_satisfied();

        // let params: ParamsKZG<Bn256> = ParamsProver::new(11);

        // let vk = keygen_vk(&params, &circuit).unwrap();

        // let pk = keygen_pk(&params, vk, &circuit).unwrap();

        // let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

        // //let now = Instant::now();

        // println!("starting proof!");

        // let instances: Vec<_> = input_instance.iter().map(|x| {
        //     x.as_slice()
        // }).collect();

        // create_proof::<_, ProverSHPLONK<Bn256>, _, _, _, _>(
        //     &params,
        //     &pk,
        //     &[circuit],
        //     &[instances.as_slice()],
        //     OsRng,
        //     &mut transcript,
        // )?;

        Ok(())
    }
}
