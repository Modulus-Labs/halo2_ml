use std::{fmt::Debug, marker::PhantomData};

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Error as PlonkError, TableColumn},
};

#[derive(Debug, Clone)]
pub struct DecompTable<F: FieldExt, const SIZE: usize> {
    pub range_check_table: TableColumn,
    _marker: PhantomData<F>,
}

impl<F: FieldExt, const SIZE: usize> DecompTable<F, SIZE> {
    pub fn configure(meta: &mut ConstraintSystem<F>) -> DecompTable<F, SIZE> {
        DecompTable {
            range_check_table: meta.lookup_table_column(),
            _marker: PhantomData,
        }
    }

    pub fn layout(&self, mut layouter: impl Layouter<F>) -> Result<(), PlonkError> {
        layouter.assign_table(
            || "eltwise decomp table",
            |mut table| {
                for offset in 0..SIZE {
                    let value: u64 = offset.try_into().unwrap();
                    table.assign_cell(
                        || format!("decomp_table row {offset}"),
                        self.range_check_table,
                        offset,
                        || Value::known(F::from(value)),
                    )?;
                }
                Ok(())
            },
        )
    }
}
