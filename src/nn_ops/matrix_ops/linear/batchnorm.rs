use std::marker::PhantomData;

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter, Value},
    plonk::{Advice, ConstraintSystem, Error as PlonkError, Fixed},
};
use ndarray::{Array1, Array3};

use crate::{
    nn_ops::matrix_ops::linear::dist_addmultadd::DistributedAddMulAddChipParams,
    nn_ops::matrix_ops::linear::dist_addmultadd::{
        DistributedAddMulAddChip, DistributedAddMulAddConfig,
    },
    nn_ops::{
        matrix_ops::non_linear::norm_2d::{Normalize2dChip, Normalize2dConfig},
        ColumnAllocator, InputSizeConfig, NNLayer,
    },
};

#[derive(Clone, Debug)]
pub struct BatchnormConfig<F: FieldExt> {
    add_mult_add_chip: DistributedAddMulAddConfig<F>,
    norm_2d_chip: Normalize2dConfig<F>,
    _marker: PhantomData<F>,
}

///Chip for 2-D Convolution (width, height, channel-in, channel-out)
///
/// Order for ndarrays is Channel-in, Width, Height, Channel-out
pub struct BatchnormChip<F: FieldExt> {
    config: BatchnormConfig<F>,
}

impl<F: FieldExt> Chip<F> for BatchnormChip<F> {
    type Config = BatchnormConfig<F>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

pub struct BatchnormChipConfig<F: FieldExt> {
    input_height: usize,
    input_width: usize,
    input_depth: usize,
    norm_2d_chip: Normalize2dConfig<F>,
}

pub struct BatchnormChipParams<F: FieldExt> {
    scalars: Array1<(Value<F>, Value<F>, Value<F>)>,
}

impl<F: FieldExt> NNLayer<F> for BatchnormChip<F> {
    type ConfigParams = BatchnormChipConfig<F>;

    type LayerInput = Array3<AssignedCell<F, F>>;

    type LayerParams = BatchnormChipParams<F>;

    type LayerOutput = Array3<AssignedCell<F, F>>;

    fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self { config }
    }

    fn configure(
        meta: &mut ConstraintSystem<F>,
        config: BatchnormChipConfig<F>,
        advice_allocator: &mut ColumnAllocator<Advice>,
        fixed_allocator: &mut ColumnAllocator<Fixed>,
    ) -> <Self as Chip<F>>::Config {
        let dist_config = InputSizeConfig {
            input_height: config.input_height,
            input_width: config.input_width,
            input_depth: config.input_depth,
        };

        let add_mult_add_chip = DistributedAddMulAddChip::configure(
            meta,
            dist_config,
            advice_allocator,
            fixed_allocator,
        );

        BatchnormConfig {
            add_mult_add_chip,
            norm_2d_chip: config.norm_2d_chip,
            _marker: PhantomData,
        }
    }

    fn add_layer(
        &self,
        layouter: &mut impl Layouter<F>,
        inputs: Array3<AssignedCell<F, F>>,
        params: BatchnormChipParams<F>,
    ) -> Result<Array3<AssignedCell<F, F>>, PlonkError> {
        let BatchnormChipParams { scalars } = params;
        let layouter = &mut layouter.namespace(|| "Batchnorm");
        let config = &self.config;

        let params = DistributedAddMulAddChipParams { scalars };

        let un_normed_output =
            DistributedAddMulAddChip::construct(config.add_mult_add_chip.clone())
                .add_layer(layouter, inputs, params)?;
        // Ok(stack(
        //     Self::DEPTH_AXIS,
        //     un_normed_output
        //         .axis_iter(Self::DEPTH_AXIS)
        //         .map(|input_mat| {
        //             Normalize2dChip::construct(config.norm_2d_chip.clone())
        //                 .add_layer(layouter, input_mat.to_owned())
        //         })
        //         .collect::<Result<Vec<_>, _>>()?
        //         .iter()
        //         .map(|x| x.view())
        //         .collect::<Vec<_>>()
        //         .as_slice(),
        // )
        // .unwrap())

        Normalize2dChip::construct(config.norm_2d_chip.clone()).add_layer(
            layouter,
            un_normed_output,
            (),
        )
    }
}

// #[cfg(test)]
// mod tests {
//     use super::{BatchnormChip, BatchnormConfig};
//     use halo2_proofs::{
//         arithmetic::FieldExt,
//         circuit::{AssignedCell, Chip, Layouter, SimpleFloorPlanner, Value},
//         dev::MockProver,
//         halo2curves::bn256::Fr,
//         plonk::{
//             Advice, Assigned, Assignment, Circuit, Column, ConstraintSystem, Error as PlonkError,
//             Expression, Instance, Selector,
//         },
//         poly::Rotation,
//     };
//     use ndarray::{array, stack, Array, Array2, Array3, Array4, ArrayBase, Axis, Zip};

//     #[derive(Clone, Debug)]
//     struct BatchnormTestConfig<F: FieldExt> {
//         input: Array2<Column<Instance>>,
//         input_advice: Array2<Column<Advice>>,
//         output: Array2<Column<Instance>>,
//         conv_chip: Conv3DLayerConfig<F>,
//     }

//     struct BatchnormTestCircuit<F: FieldExt> {
//         pub kernal: Array4<Value<F>>,
//         pub input: Array3<Value<F>>,
//     }

//     const INPUT_WIDTH: usize = 16;
//     const INPUT_HEIGHT: usize = 16;

//     const KERNAL_WIDTH: usize = 3;
//     const KERNAL_HEIGHT: usize = 3;

//     const DEPTH: usize = 4;

//     const PADDING_WIDTH: usize = 1;
//     const PADDING_HEIGHT: usize = 1;

//     impl<F: FieldExt> Circuit<F> for BatchnormTestCircuit<F> {
//         type Config = BatchnormTestConfig<F>;

//         type FloorPlanner = SimpleFloorPlanner;

//         fn without_witnesses(&self) -> Self {
//             Self {
//                 kernal: Array::from_shape_simple_fn(
//                     (DEPTH, KERNAL_WIDTH, KERNAL_HEIGHT, DEPTH),
//                     || Value::unknown(),
//                 ),
//                 input: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH, INPUT_HEIGHT), || {
//                     Value::unknown()
//                 }),
//             }
//         }

//         fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
//             let inputs =
//                 Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH + PADDING_WIDTH * 2), || {
//                     let col = meta.advice_column();
//                     meta.enable_equality(col);
//                     col
//                 });

//             let kernals = Array::from_shape_simple_fn((DEPTH, KERNAL_WIDTH), || {
//                 let col = meta.advice_column();
//                 meta.enable_equality(col);
//                 col
//             });

//             let output_width = INPUT_WIDTH + PADDING_WIDTH * 2 - KERNAL_WIDTH + 1;

//             let outputs = Array::from_shape_simple_fn(output_width, || {
//                 let col = meta.advice_column();
//                 meta.enable_equality(col);
//                 col
//             });

//             let conv_chip = BatchnormChip::configure(
//                 meta,
//                 inputs,
//                 kernals,
//                 outputs,
//                 KERNAL_HEIGHT,
//                 KERNAL_WIDTH,
//                 PADDING_WIDTH,
//                 PADDING_HEIGHT,
//             );

//             BatchnormTestConfig {
//                 input: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH), || {
//                     let col = meta.instance_column();
//                     meta.enable_equality(col);
//                     col
//                 }),
//                 output: Array::from_shape_simple_fn((DEPTH, output_width), || {
//                     let col = meta.instance_column();
//                     meta.enable_equality(col);
//                     col
//                 }),
//                 input_advice: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH), || {
//                     let col = meta.advice_column();
//                     meta.enable_equality(col);
//                     col
//                 }),
//                 conv_chip,
//             }
//         }

//         fn synthesize(
//             &self,
//             config: Self::Config,
//             mut layouter: impl Layouter<F>,
//         ) -> Result<(), PlonkError> {
//             let conv_chip = BatchnormChip::construct(config.conv_chip);

//             let inputs = layouter.assign_region(
//                 || "inputs",
//                 |mut region| {
//                     let input = config.input.view();
//                     let input_advice = config.input_advice.view();
//                     let result = stack(
//                         Axis(2),
//                         &self
//                             .input
//                             .axis_iter(Axis(2))
//                             .enumerate()
//                             .map(|(row, slice)| {
//                                 Zip::from(slice.view())
//                                     .and(input)
//                                     .and(input_advice)
//                                     .map_collect(|input, instance, column| {
//                                         region
//                                             .assign_advice_from_instance(
//                                                 || "assign input",
//                                                 *instance,
//                                                 row,
//                                                 *column,
//                                                 row,
//                                             )
//                                             .unwrap()
//                                     })
//                             })
//                             .collect::<Vec<_>>()
//                             .iter()
//                             .map(|x| x.view())
//                             .collect::<Vec<_>>(),
//                     )
//                     .unwrap();
//                     Ok(result)
//                 },
//             )?;

//             let output = conv_chip.add_layer(&mut layouter, &inputs, &self.kernal)?;
//             let input = config.output.view();
//             for (row, slice) in output.axis_iter(Axis(2)).enumerate() {
//                 Zip::from(slice.view())
//                     .and(input)
//                     .for_each(|input, column| {
//                         layouter
//                             .constrain_instance(input.cell(), *column, row)
//                             .unwrap();
//                     })
//             }
//             Ok(())
//         }
//     }

//     #[test]
//     ///test that a simple 16x16x4 w/ 3x3x4 conv works; input and kernal are all 1
//     fn test_simple_conv() -> Result<(), PlonkError> {
//         let circuit = BatchnormTestcircuit {
//             kernal: Array::from_shape_simple_fn(
//                 (DEPTH, KERNAL_WIDTH, KERNAL_HEIGHT, DEPTH),
//                 || Value::known(Fr::one()),
//             ),
//             input: Array::from_shape_simple_fn((DEPTH, INPUT_WIDTH, INPUT_HEIGHT), || {
//                 Value::known(Fr::one())
//             }),
//         };

//         let output_width = INPUT_WIDTH + PADDING_WIDTH * 2 - KERNAL_WIDTH + 1;
//         let output_height = INPUT_HEIGHT + PADDING_HEIGHT * 2 - KERNAL_HEIGHT + 1;

//         let mut input_instance = vec![vec![Fr::one(); INPUT_HEIGHT]; INPUT_WIDTH * DEPTH];
//         //let mut output_instance = vec![vec![Fr::one(); output_height]; output_width*DEPTH];
//         let edge: Vec<_> = vec![
//             16, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 16,
//         ]
//         .iter()
//         .map(|x| Fr::from(*x))
//         .collect();
//         let row: Vec<_> = vec![
//             24, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 24,
//         ]
//         .iter()
//         .map(|&x| Fr::from(x))
//         .collect();
//         let layer = vec![
//             edge.clone(),
//             row.clone(),
//             row.clone(),
//             row.clone(),
//             row.clone(),
//             row.clone(),
//             row.clone(),
//             row.clone(),
//             row.clone(),
//             row.clone(),
//             row.clone(),
//             row.clone(),
//             row.clone(),
//             row.clone(),
//             row,
//             edge,
//         ];
//         let mut output_instance: Vec<_> =
//             vec![layer.clone(), layer.clone(), layer.clone(), layer.clone()]
//                 .into_iter()
//                 .flatten()
//                 .collect();
//         input_instance.append(&mut output_instance);

//         MockProver::run(7, &circuit, input_instance)
//             .unwrap()
//             .assert_satisfied();

//         Ok(())
//     }
// }
