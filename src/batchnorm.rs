use std::marker::PhantomData;

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Chip, Layouter, Value},
    plonk::{
        Advice, Assigned, Column, ConstraintSystem, Error as PlonkError, Expression, Instance,
        Selector,
    },
    poly::Rotation,
};
use ndarray::{
    concatenate, stack, Array, Array1, Array2, Array3, Array4, ArrayBase, Axis, Dim, Zip,
};

use crate::{
    dist_addmultadd::{DistrubutedAddMulAddChip, DistrubutedAddMulAddConfig},
    norm_2d::{Normalize2dChip, Normalize2dConfig},
};

#[derive(Clone, Debug)]
pub struct BatchnormConfig<F: FieldExt> {
    add_mult_add_chip: DistrubutedAddMulAddConfig<F>,
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

impl<F: FieldExt> BatchnormChip<F> {
    const DEPTH_AXIS: Axis = Axis(0);
    const COLUMN_AXIS: Axis = Axis(1);
    const ROW_AXIS: Axis = Axis(2);

    pub fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self { config }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        inputs: Array2<Column<Advice>>,
        outputs: Array2<Column<Advice>>,
        scalars: Array1<(Column<Advice>, Column<Advice>, Column<Advice>)>,
        norm_2d_chip: Normalize2dConfig<F>,
    ) -> <Self as Chip<F>>::Config {
        let add_mult_add_chip = DistrubutedAddMulAddChip::configure(meta, inputs, outputs, scalars);

        BatchnormConfig {
            add_mult_add_chip,
            norm_2d_chip,
            _marker: PhantomData,
        }
    }

    pub fn add_layer(
        &self,
        layouter: &mut impl Layouter<F>,
        inputs: &Array3<AssignedCell<F, F>>,
        scalars: &Array1<(Value<F>, Value<F>, Value<F>)>,
    ) -> Result<Array3<AssignedCell<F, F>>, PlonkError> {
        let layouter = &mut layouter.namespace(|| "Batchnorm");
        let config = &self.config;
        let un_normed_output =
            DistrubutedAddMulAddChip::construct(config.add_mult_add_chip.clone())
                .add_layer(layouter, inputs, scalars)?;
        Ok(stack(
            Self::DEPTH_AXIS,
            un_normed_output
                .axis_iter(Self::DEPTH_AXIS)
                .map(|input_mat| {
                    Normalize2dChip::<F, 1024, 2>::construct(config.norm_2d_chip.clone())
                        .add_layer(layouter, &input_mat.to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?
                .iter()
                .map(|x| x.view())
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .unwrap())
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
