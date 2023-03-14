use std::marker::PhantomData;

use halo2_base::halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, Error as PlonkError, Expression, Fixed, Selector},
    poly::Rotation,
};
use itertools::Itertools;
use ndarray::{concatenate, stack, Array, Array1, Array2, Array3, Array4, Axis, Zip};

use crate::{nn_ops::{ColumnAllocator, NNLayer}, felt_to_i64};

#[derive(Clone, Debug)]
pub struct FcConfig<F: FieldExt> {
    //pub in_width: usize,
    //pub in_height: usize,
    //pub in_depth: usize,
    pub inputs: Array1<Column<Advice>>,
    pub outputs: Column<Advice>,
    pub final_outputs: Column<Advice>,
    pub weights: Array1<Column<Fixed>>,
    pub bias: Column<Fixed>,
    pub fc_selector: Selector,
    pub out_selector: Selector,
    pub folding_factor: usize,
    _marker: PhantomData<F>,
}

///Chip for 2-D Convolution
///
/// Order for ndarrays is Channel-in, Width, Height, Channel-out
pub struct FcChip<F: FieldExt> {
    config: FcConfig<F>,
}

impl<'a, F: FieldExt> Chip<F> for FcChip<F> {
    type Config = FcConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

pub struct FcChipConfig {
    pub weights_height: usize,
    pub weights_width: usize,
    pub folding_factor: usize,
}

#[derive(Clone, Debug)]
pub struct FcChipParams<F: FieldExt> {
    pub weights: Array2<Value<F>>,
    pub biases: Array1<Value<F>>,
}

impl<'a, F: FieldExt> NNLayer<F> for FcChip<F> {
    type LayerInput = Array1<AssignedCell<F, F>>;

    type ConfigParams = FcChipConfig;
    type LayerParams = FcChipParams<F>;

    type LayerOutput = Array1<AssignedCell<F, F>>;

    fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self { config }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        config_params: FcChipConfig,
        advice_allocator: &mut ColumnAllocator<Advice>,
        fixed_allocator: &mut ColumnAllocator<Fixed>,
    ) -> <Self as Chip<F>>::Config {
        let width = config_params.weights_width;
        let input_len = width / config_params.folding_factor;
        let output_len = 2;
        let advice_len = input_len + output_len;
        let fixed_len = (width / config_params.folding_factor) + 1;
        let advice = advice_allocator.take(meta, advice_len);
        let fixed = fixed_allocator.take(meta, fixed_len);

        let inputs = Array1::from_vec(advice[0..input_len].to_vec());
        let output = advice[advice.len()-2];
        let output_final = advice[advice.len()-1];

        let weights = Array::from_shape_vec(input_len, fixed[0..fixed.len()-1].to_vec()).unwrap();
        let bias = fixed[fixed.len()-1];
        let selector = meta.selector();

        meta.create_gate("Fc Gate", |meta| {
            let sel = meta.query_selector(selector);
            let output = meta.query_advice(output, Rotation::cur());
            let inputs: Vec<(Expression<F>, Expression<F>)> = inputs.iter().zip(weights.iter()).map(|(&input_col, &weight_col)|{
                    (
                        meta.query_advice(input_col, Rotation::cur()),
                        meta.query_fixed(weight_col, Rotation::cur()),
                    )
                })
                .collect();

            let constraint = inputs
            .into_iter()
            .fold(Expression::Constant(F::zero()), |accum, (input, weight)| {
                accum + (input * weight)
            });
            vec![sel * (constraint - output)]
        });

        let final_selector = meta.selector();
        meta.create_gate("Fc Unfolding", |meta| {
            let sel = meta.query_selector(final_selector);
            let sum = (0..config_params.folding_factor).map(|index| {
                meta.query_advice(output, Rotation(-i32::try_from(index).unwrap()))
            }).reduce(|accum, item| accum + item).unwrap();
            let bias = meta.query_fixed(bias, Rotation::cur());

            let final_output = meta.query_advice(output_final, Rotation::cur());
            vec![sel * (sum + bias - final_output)]
        });
        
        FcConfig {
            inputs,
            outputs: output,
            final_outputs: output_final,
            weights,
            bias,
            fc_selector: selector,
            out_selector: final_selector,
            folding_factor: config_params.folding_factor,
            _marker: PhantomData,
        }
    }

    fn add_layer(
        &self,
        layouter: &mut impl Layouter<F>,
        inputs: Array1<AssignedCell<F, F>>,
        layer_params: FcChipParams<F>,
    ) -> Result<Array1<AssignedCell<F, F>>, PlonkError> {
        let dims = layer_params.weights.dim();
        let new_width = dims.0/self.config.folding_factor;
        let config = &self.config;
        let FcChipParams { weights, biases } = layer_params;
        layouter.assign_region(|| "synthesize FC", |mut region | {
            weights.axis_iter(Axis(1)).zip(biases.iter()).enumerate().map(|(row, (weights, &bias))| {
                let offset = row*config.folding_factor;

                //assign weights
                let calc_output = weights.axis_chunks_iter(Axis(0), new_width).zip(inputs.axis_chunks_iter(Axis(0), new_width)).enumerate().map(|(row, (weights, inputs))| {
                    //println!("weights are {:?}", weights.map(|x| x.map(|x| felt_to_i64(x))));
                    //println!("inputs are {:?}", inputs.map(|x| x.value().map(|&x| felt_to_i64(x))));
                    for (&weight, &column) in weights.iter().zip(config.weights.iter()) {
                        config.fc_selector.enable(&mut region, offset + row)?;
                        region.assign_fixed(|| "Assign Weight", column, offset + row, || weight)?;
                    }

                    for (input, &column) in inputs.iter().zip(config.inputs.iter()) {
                        input.copy_advice(|| "Assign Input", &mut region, column, offset + row)?;
                    }

                    let calc_output = inputs.iter().zip(weights.iter()).fold(Value::known(F::zero()), |accum, (input, &weight)| {
                        accum + (weight * input.value())
                    });

                    region.assign_advice(|| "Assign intermediate output", config.outputs, offset + row, || calc_output)?;
                    //println!("calc_output for half-row is {:?}", calc_output.map(|x| felt_to_i64(x)));
                    Ok::<_, PlonkError>(calc_output)
                }).reduce(|accum, item| Ok(accum? + item?)).unwrap()? + bias;

                //println!("calc_output for final output {:?}", calc_output.map(|x| felt_to_i64(x)));

                config.out_selector.enable(&mut region, offset + config.folding_factor - 1)?;

                //assign bias
                region.assign_fixed(|| "Assign Bias", config.bias, offset + config.folding_factor - 1, || bias)?;

                region.assign_advice(|| "Assign Final Output", config.final_outputs, offset + config.folding_factor - 1, || calc_output)

            }).collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        felt_from_i64,
        nn_ops::{ColumnAllocator, DefaultDecomp, NNLayer}, felt_to_i64,
    };

    use halo2_base::halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        halo2curves::bn256::Fr,
        plonk::{Advice, Circuit, Column, ConstraintSystem, Error as PlonkError, Fixed, Instance},
    };
    use ndarray::{stack, Array, Array2, Array3, Array4, Axis, Zip, Array1};

    use super::{FcConfig, FcChipConfig, FcChip};

    #[derive(Clone, Debug)]
    struct FcTestConfig<F: FieldExt> {
        input: Column<Instance>,
        input_advice: Column<Advice>,
        output: Column<Instance>,
        conv_chip: FcConfig<F>,
    }

    struct FcTestCircuit<F: FieldExt> {
        pub weights: Array2<Value<F>>,
        pub biases: Array1<Value<F>>,
        pub input: Array1<Value<F>>,
    }

    const WIDTH: usize = 4;
    const HEIGHT: usize = 4;

    impl<F: FieldExt> Circuit<F> for FcTestCircuit<F> {
        type Config = FcTestConfig<F>;

        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                weights: Array::from_shape_simple_fn(
                    (WIDTH, HEIGHT),
                    || Value::unknown(),
                ),
                biases: Array::from_shape_simple_fn(HEIGHT, || Value::unknown()),
                input: Array::from_shape_simple_fn(WIDTH, || {
                    Value::unknown()
                }),
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let mut advice_allocator = ColumnAllocator::<Advice>::new(meta, 0);

            let mut fixed_allocator = ColumnAllocator::<Fixed>::new(meta, 0);

            let config_params = FcChipConfig {
                weights_height: HEIGHT,
                weights_width: WIDTH,
                folding_factor: 4,
            };

            let conv_chip = FcChip::configure(
                meta,
                config_params,
                &mut advice_allocator,
                &mut fixed_allocator,
            );

            FcTestConfig {
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
            let conv_chip = FcChip::construct(config.conv_chip);

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
                    Ok(Array::from_shape_vec(WIDTH, inputs?).unwrap())
                },
            )?;

            println!("assigned inputs are {:?}", inputs.map(|x| x.value().map(|x| felt_to_i64(*x))));

            let layer_params = super::FcChipParams { weights: self.weights.clone(), biases: self.biases.clone() };

            let output = conv_chip.add_layer(&mut layouter, inputs, layer_params)?;
            println!("output is {:?}", output.map(|x| x.value().map(|x| felt_to_i64(*x))));

            for (row, output) in output.iter().enumerate() {
                layouter.constrain_instance(output.cell(), config.output, row)?;
            }

            Ok(())
        }
    }

    #[test]
    ///test that a simple 8x8x4 w/ 3x3x4 conv works; input and kernal are all 1
    fn test_simple_conv() -> Result<(), PlonkError> {
        let weights = Array::from_shape_fn(
            (WIDTH, HEIGHT),
            |(index, row)| Value::known(Fr::from((index*row) as u64))
        );
        let input = Array::from_shape_fn(
            WIDTH,
            |index| Value::known(Fr::from(index as u64))
        );
        let biases = Array::from_shape_fn(HEIGHT, |index| Value::known(Fr::from(index as u64)));
        // let output = Array::from_shape_simple_fn(
        //     HEIGHT,
        //     || Fr::from(16)
        // );

        let output = vec![Fr::from(0), Fr::from(15), Fr::from(30), Fr::from(45)];

        let circuit = FcTestCircuit { weights, input, biases };

        let instances = vec![
            Array::from_shape_fn(
                WIDTH,
                |index| Fr::from(index as u64)
            ).to_vec(),
            output,
        ];

        MockProver::run(9, &circuit, instances)
            .unwrap()
            .assert_satisfied();

        Ok(())
    }
}
