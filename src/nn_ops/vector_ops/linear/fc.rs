use std::{fmt::Debug, marker::PhantomData};

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter, Value},
    plonk::{
        Advice, Fixed, Assigned, Column, ConstraintSystem, Error as PlonkError, Expression, Instance,
        Selector,
    },
    poly::Rotation,
};

use crate::nn_ops::{vector_ops::non_linear::eltwise_ops::{DecompConfig, EltwiseInstructions}, NNLayer, ColumnAllocator};

#[derive(Default, Clone, Debug)]
pub struct FcParams<F: FieldExt> {
    pub weights: Vec<Value<F>>,
    pub biases: Vec<Value<F>>,
}

#[derive(Clone, Debug)]
pub struct FcConfig<F: FieldExt> {
    pub width: usize,
    pub height: usize,
    pub weights: Vec<Column<Fixed>>,
    pub bias: Column<Fixed>,
    pub inputs: Vec<Column<Advice>>,
    pub output: Column<Advice>,
    pub eltwise: DecompConfig<F>,
    // pub eltwise_inter: Vec<Column<Advice>>,
    // pub eltwise_output: Column<Advice>,
    pub nn_selector: Selector,
    // pub elt_selector: Selector,
    _marker: PhantomData<F>,
}

#[derive(Clone, Debug)]
///Chip to prove NeuralNet operations
pub struct FcChip<F: FieldExt, Elt: EltwiseInstructions<F>> {
    config: FcConfig<F>,
    _marker: PhantomData<(F, Elt)>,
}

impl<F: FieldExt, Elt: EltwiseInstructions<F>> Chip<F> for FcChip<F, Elt> {
    type Config = FcConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

#[derive(Clone, Debug)]
pub struct FcChipConfig<F: FieldExt> {
    pub weights_height: usize,
    pub weights_width: usize,
    pub elt_config: DecompConfig<F>,
}

impl<F: FieldExt, Elt: EltwiseInstructions<F>> NNLayer<F> for FcChip<F, Elt> {

    type ConfigParams = FcChipConfig<F>;

    type LayerInput = Vec<AssignedCell<F, F>>;

    type LayerParams = FcParams<F>;

    type LayerOutput = Vec<AssignedCell<F, F>>;

    fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        config: FcChipConfig<F>,
        advice_allocator: &mut ColumnAllocator<Advice>,
        fixed_allocator: &mut ColumnAllocator<Fixed>
    ) -> <Self as Chip<F>>::Config {
        let FcChipConfig { weights_height: height, weights_width: width, elt_config } = config;

        let advice = advice_allocator.take(meta, width + 1);
        let fixed = fixed_allocator.take(meta, width + 1);

        let inputs = advice[0..width].to_vec();
        let output = advice[width];

        let weights = fixed[0..width].to_vec();
        let bias = fixed[width];

        let nn_selector = meta.selector();
        meta.create_gate("FC", |meta| {
            let q = meta.query_selector(nn_selector);
            // We put the negation of the claimed output in the constraint tensor.
            // let constraints: Vec<Expression<F>> = (0..out_dim)
            //     .map(|i| -meta.query_advice(output, Rotation(i as i32)))
            //     .collect();

            let output = -meta.query_advice(output, Rotation::cur());

            let inputs: Vec<(Expression<F>, Expression<F>)> = (0..width)
                .map(|index| {
                    (
                        meta.query_advice(inputs[index], Rotation::cur()),
                        meta.query_fixed(weights[index], Rotation::cur()),
                    )
                })
                .collect();

            let bias = meta.query_fixed(bias, Rotation::cur());

            // Now we compute the linear expression,  and add it to constraints
            // let constraints: Vec<Expression<F>> = constraints
            //     .iter()
            //     .enumerate()
            //     .map(|item| {
            //         let i = item.0;
            //         let mut c = item.1.clone();
            //         for j in 0..WIDTH {
            //             c = c + meta.query_advice(weights[j], Rotation(i as i32))
            //                 * inputs[j].clone();
            //         }
            //         // add the bias
            //         q.clone() * (c + meta.query_advice(bias, Rotation(i as i32)))
            //     })
            //     .collect();
            let constraint = inputs
                .into_iter()
                .fold(Expression::Constant(F::zero()), |accum, (input, weight)| {
                    accum + (input * weight)
                });
            vec![q * (constraint + bias + output)]
        });
        //}
        // let eltwise_config =
        //     Elt::configure(meta, output, eltwise_inter, eltwise_output, range_table);
        FcConfig {
            width,
            height,
            inputs,
            weights,
            bias,
            output,
            eltwise: elt_config,
            nn_selector,
            _marker: PhantomData,
        }
    }

    fn add_layer(
        &self,
        layouter: &mut impl Layouter<F>,
        inputs: Vec<AssignedCell<F, F>>,
        params: FcParams<F>,
    ) -> Result<Vec<AssignedCell<F, F>>, PlonkError> {
        let config = &self.config;

        let layer = params;

        let mat_output = layouter.assign_region(
            || "NN Layer",
            |mut region| {
                // let layer_outputs: Result<Vec<_>, PlonkError> =
                let offset = 0;

                //assign parameters (weights+biases)
                let (biases, weights): (Vec<_>, Vec<_>) = ({
                    let thing: (Result<Vec<_>, _>, Result<Vec<_>, _>) = (
                        layer
                            .biases
                            .iter()
                            .enumerate()
                            .map(|(index, &bias)| {
                                region.assign_fixed(
                                    || "assigning biases".to_string(),
                                    config.bias,
                                    offset + index,
                                    || bias,
                                )
                            })
                            .collect(),
                        layer
                            .weights
                            .iter()
                            .enumerate()
                            .map(|(iii, weight)| {
                                region.assign_fixed(
                                    || "assigning weights".to_string(),
                                    // row indices
                                    config.weights[iii % config.width],
                                    // columns indices
                                    offset + (iii / config.width),
                                    || *weight,
                                )
                            })
                            .collect(),
                    );
                    Ok::<_, PlonkError>((thing.0.unwrap(), thing.1.unwrap()))
                })
                .unwrap();

                //calculate output and assign it to layer output
                let mat_output: Vec<AssignedCell<F, F>> = {
                    let out_dim = config.height;

                    // calculate value of output
                    let output: Vec<Value<F>> = (0..out_dim)
                        .map(|i| {
                            let mut o: Value<F> = Value::known(F::zero());
                            for (j, x) in inputs.iter().enumerate() {
                                o = o + layer.weights[j + (i * config.width)]
                                    * x.value();
                            }
                            o + layer.biases[i]
                        })
                        .collect();

                    let output: Result<Vec<AssignedCell<F, F>>, _> = output
                        .iter()
                        .enumerate()
                        .map(|(i, o)| {
                            config.nn_selector.enable(&mut region, offset + i).unwrap();
                            for (j, x) in inputs.iter().enumerate() {
                                x.copy_advice(
                                    || "input",
                                    &mut region,
                                    config.inputs[j],
                                    offset + i,
                                )?;
                            }
                            region.assign_advice(
                                || "o".to_string(),
                                config.output,
                                offset + i,
                                || *o,
                            )
                        })
                        .collect();
                    output.unwrap()
                };

                Ok(mat_output)
            },
        )?;

        let elt_chip: Elt = Elt::construct(config.eltwise.clone());

        mat_output
            .into_iter()
            .map(|out| elt_chip.apply_elt(layouter.namespace(|| "elt"), out))
            .collect()
    }
}
