use std::fmt::Debug;

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Chip, Layouter},
    plonk::{Advice, Column, ColumnType, ConstraintSystem, Error as PlonkError, Fixed},
};
use ndarray::Axis;

pub mod lookup_ops;
pub mod matrix_ops;
pub mod vector_ops;

pub trait DecompConfig: Debug + Clone {
    const ADVICE_LEN: usize;
    const BASE: usize;
    const K: usize;
    const SCALING_FACTOR: u64;
}

#[derive(Debug, Clone)]
pub struct DefaultDecomp {}

impl DecompConfig for DefaultDecomp {
    const ADVICE_LEN: usize = 10;
    const BASE: usize = 1024;
    const K: usize = 2;
    const SCALING_FACTOR: u64 = 1_048_576;
}

///Convience wrapper for keeping track of columns
///
/// Not particularly efficient; Efficiency will depend on order of requests
pub struct ColumnAllocator<C: ColumnType> {
    advice: Vec<Column<C>>,
    last_index: usize,
}

impl ColumnAllocator<Fixed> {
    pub fn new<F: FieldExt>(meta: &mut ConstraintSystem<F>, max_advice_count: usize) -> Self {
        let advice = (0..max_advice_count)
            .map(|_| {
                let col = meta.fixed_column();
                meta.enable_equality(col);
                col
            })
            .collect();

        ColumnAllocator {
            advice,
            last_index: 0,
        }
    }

    pub fn take<F: FieldExt>(
        &mut self,
        meta: &mut ConstraintSystem<F>,
        n: usize,
    ) -> &[Column<Fixed>] {
        if n > self.advice.len() {
            self.advice.extend((0..n - self.advice.len()).map(|_| {
                let col = meta.fixed_column();
                meta.enable_equality(col);
                col
            }));
            &self.advice[0..n]
        } else {
            if self.last_index + n < self.advice.len() {
                let slice = &self.advice[self.last_index..self.last_index + n];
                self.last_index += n;
                slice
            } else {
                let slice = &self.advice[0..n];
                self.last_index = n;
                slice
            }
        }
    }
}

impl ColumnAllocator<Advice> {
    pub fn new<F: FieldExt>(meta: &mut ConstraintSystem<F>, max_advice_count: usize) -> Self {
        let advice = (0..max_advice_count)
            .map(|_| {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            })
            .collect();

        ColumnAllocator {
            advice,
            last_index: 0,
        }
    }

    pub fn take<F: FieldExt>(
        &mut self,
        meta: &mut ConstraintSystem<F>,
        n: usize,
    ) -> &[Column<Advice>] {
        if n > self.advice.len() {
            self.advice.extend((0..n - self.advice.len()).map(|_| {
                let col = meta.advice_column();
                meta.enable_equality(col);
                col
            }));
            &self.advice[0..n]
        } else {
            if self.last_index + n < self.advice.len() {
                let slice = &self.advice[self.last_index..self.last_index + n];
                self.last_index += n;
                slice
            } else {
                let slice = &self.advice[0..n];
                self.last_index = n;
                slice
            }
        }
    }
}

pub struct InputSizeConfig {
    input_height: usize,
    input_width: usize,
    input_depth: usize,
}

pub trait NNLayer<F: FieldExt>: Chip<F> {
    const DEPTH_AXIS: Axis = Axis(0);
    const COLUMN_AXIS: Axis = Axis(1);
    const ROW_AXIS: Axis = Axis(2);
    const C_OUT_AXIS: Axis = Axis(3);

    type DecompConfig = DefaultDecomp;

    type ConfigParams = InputSizeConfig;
    type LayerParams = ();
    type LayerInput;
    type LayerOutput;

    fn construct(config: <Self as Chip<F>>::Config) -> Self;

    fn configure(
        meta: &mut ConstraintSystem<F>,
        config_params: Self::ConfigParams,
        advice_allocator: &mut ColumnAllocator<Advice>,
        fixed_allocator: &mut ColumnAllocator<Fixed>,
    ) -> <Self as Chip<F>>::Config;

    fn add_layer(
        &self,
        layouter: &mut impl Layouter<F>,
        input: Self::LayerInput,
        layer_params: Self::LayerParams,
    ) -> Result<Self::LayerOutput, PlonkError>;
}
