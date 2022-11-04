use std::{fmt::Debug, hash::Hash, marker::PhantomData};

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter, Value},
    plonk::{
        Advice, Assigned, Column, ConstraintSystem, Error as PlonkError, Expression, Instance,
        Selector,
    },
    poly::Rotation,
};

use crate::nn_ops::eltwise_ops::{DecompConfig, EltwiseInstructions};

//TODO: move somehwere more appropriate
#[derive(Default, Clone, Debug)]
pub struct LayerParams<F: FieldExt> {
    pub weights: Vec<F>,
    pub biases: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct ForwardLayerConfig<F: FieldExt> {
    pub width: usize,
    pub height: usize,
    pub weights: Vec<Column<Advice>>,
    pub bias: Column<Advice>,
    pub inputs: Vec<Column<Advice>>,
    pub output: Column<Advice>,
    pub eltwise: DecompConfig<F>,
    // pub eltwise_inter: Vec<Column<Advice>>,
    // pub eltwise_output: Column<Advice>,
    pub nn_selector: Selector,
    // pub elt_selector: Selector,
    _marker: PhantomData<F>,
}

///Instructions that allow NN Chip to be used
pub trait NNLayerInstructions<F: FieldExt> {
    ///Loads inputs from constant
    fn load_input_constant(
        &self,
        layouter: impl Layouter<F>,
        input: &[F],
    ) -> Result<Vec<AssignedCell<F, F>>, PlonkError>;

    ///Loads inputs from advice
    fn load_input_advice(
        &self,
        layouter: impl Layouter<F>,
        input: Vec<Value<F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, PlonkError>;

    ///Loads inputs from instance
    fn load_input_instance(
        &self,
        layouter: impl Layouter<F>,
        instance: Column<Instance>,
        row: usize,
        len: usize,
    ) -> Result<Vec<AssignedCell<F, F>>, PlonkError>;

    ///Adds layers to the model, including constants for weights and biases
    fn add_layers(
        &self,
        layouter: impl Layouter<F>,
        input: Vec<AssignedCell<F, F>>,
        layers: &LayerParams<F>,
    ) -> Result<Vec<AssignedCell<F, F>>, PlonkError>;
}

#[derive(Clone, Debug)]
///Chip to prove NeuralNet operations
pub struct ForwardLayerChip<F: FieldExt, Elt: EltwiseInstructions<F>> {
    config: ForwardLayerConfig<F>,
    _marker: PhantomData<(F, Elt)>,
}

impl<F: FieldExt, Elt: EltwiseInstructions<F>> Chip<F> for ForwardLayerChip<F, Elt> {
    type Config = ForwardLayerConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: FieldExt, Elt: EltwiseInstructions<F>> ForwardLayerChip<F, Elt> {
    pub fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        width: usize,
        height: usize,
        //layer: ForwardLayerConfig<F, BASE, Elt>,
        inputs: &[Column<Advice>],
        weights: &[Column<Advice>],
        bias: Column<Advice>,
        output: Column<Advice>,
        elt_config: DecompConfig<F>,
    ) -> <Self as Chip<F>>::Config {
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
                        meta.query_advice(weights[index], Rotation::cur()),
                    )
                })
                .collect();

            let bias = meta.query_advice(bias, Rotation::cur());

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
        ForwardLayerConfig {
            width,
            height,
            inputs: inputs.to_vec(),
            weights: weights.to_vec(),
            bias,
            output,
            eltwise: elt_config,
            nn_selector,
            _marker: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Number<F: FieldExt>(pub AssignedCell<F, F>);

impl<F: FieldExt, Elt: EltwiseInstructions<F>> NNLayerInstructions<F> for ForwardLayerChip<F, Elt> {
    fn load_input_constant(
        &self,
        mut layouter: impl Layouter<F>,
        input: &[F],
    ) -> Result<Vec<AssignedCell<F, F>>, PlonkError> {
        let config = self.config();

        layouter.assign_region(
            || "load constants to NN",
            |mut region| {
                input
                    .iter()
                    .enumerate()
                    .map(|(i, item)| {
                        region.assign_advice_from_constant(
                            || "NN input from constant",
                            config.inputs[i],
                            0,
                            *item,
                        )
                    })
                    .collect()
            },
        )
    }

    fn load_input_advice(
        &self,
        mut layouter: impl Layouter<F>,
        input: Vec<Value<F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, PlonkError> {
        let config = self.config();

        layouter.assign_region(
            || "load constants to NN",
            |mut region| {
                input
                    .iter()
                    .enumerate()
                    .map(|(i, item)| {
                        region.assign_advice(
                            || "NN input from advice",
                            config.inputs[i],
                            0,
                            || *item,
                        )
                    })
                    .collect()
            },
        )
    }

    fn load_input_instance(
        &self,
        mut layouter: impl Layouter<F>,
        instance: Column<Instance>,
        starting_row: usize,
        len: usize,
    ) -> Result<Vec<AssignedCell<F, F>>, PlonkError> {
        let config = self.config();

        layouter.assign_region(
            || "load constants to NN",
            |mut region| {
                (0..len)
                    .map(|iii| {
                        region.assign_advice_from_instance(
                            || "NN input from instance",
                            instance,
                            starting_row + iii,
                            config.inputs[iii],
                            0,
                        )
                    })
                    .collect()
            },
        )
    }

    fn add_layers(
        &self,
        mut layouter: impl Layouter<F>,
        input: Vec<AssignedCell<F, F>>,
        layer: &LayerParams<F>,
    ) -> Result<Vec<AssignedCell<F, F>>, PlonkError> {
        let config = &self.config;

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
                                region.assign_advice(
                                    || "assigning biases".to_string(),
                                    config.bias,
                                    offset + index,
                                    || Value::known(bias),
                                )
                            })
                            .collect(),
                        layer
                            .weights
                            .iter()
                            .enumerate()
                            .map(|(iii, weight)| {
                                region.assign_advice(
                                    || "assigning weights".to_string(),
                                    // row indices
                                    config.weights[iii % config.width],
                                    // columns indices
                                    offset + (iii / config.width),
                                    || Value::known(*weight),
                                )
                            })
                            .collect(),
                    );
                    Ok::<_, PlonkError>((thing.0.unwrap(), thing.1.unwrap()))
                })
                .unwrap();

                //calculate output and assign it to layer output
                let mat_output: Vec<AssignedCell<Assigned<F>, F>> = {
                    let out_dim = config.height;

                    // calculate value of output
                    let output: Vec<Value<Assigned<F>>> = (0..out_dim)
                        .map(|i| {
                            let mut o: Value<Assigned<F>> = Value::known(F::zero().into());
                            for (j, x) in input.iter().enumerate() {
                                o = o + weights[j + (i * config.width)].value_field()
                                    * x.value_field();
                            }
                            o + biases[i].value_field()
                        })
                        .collect();

                    let output: Result<Vec<AssignedCell<Assigned<F>, F>>, _> = output
                        .iter()
                        .enumerate()
                        .map(|(i, o)| {
                            config.nn_selector.enable(&mut region, offset + i).unwrap();
                            for (j, x) in input.iter().enumerate() {
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
                    output?
                };

                Ok(mat_output)
            },
        )?;

        let elt_chip: Elt = Elt::construct(config.eltwise.clone());

        mat_output
            .into_iter()
            .map(|out| elt_chip.apply_elt(layouter.namespace(|| "elt"), out.evaluate()))
            .collect()
    }
}
