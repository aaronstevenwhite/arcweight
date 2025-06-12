//! Difference algorithm

use crate::algorithms::ComposeFilter;
use crate::arc::Arc;
use crate::fst::{Fst, MutableFst};
use crate::semiring::Semiring;
use crate::{Error, Result};
use core::hash::Hash;

/// Difference filter for composition
pub struct DifferenceFilter;

impl<W: Semiring> ComposeFilter<W> for DifferenceFilter {
    type FilterState = ();

    fn start() -> Self::FilterState {}

    fn filter_arc(
        &self,
        arc1: &Arc<W>,
        arc2: &Arc<W>,
        _fs: &Self::FilterState,
    ) -> Option<(Arc<W>, Self::FilterState)> {
        // match when labels are different
        if arc1.olabel != arc2.ilabel && arc1.olabel != 0 && arc2.ilabel != 0 {
            Some((
                Arc::new(arc1.ilabel, arc1.olabel, arc1.weight.clone(), 0),
                (),
            ))
        } else {
            None
        }
    }
}

/// Difference of two FSTs (fst1 - fst2)
///
/// # Errors
///
/// Returns an error if:
/// - Either input FST is invalid or corrupted
/// - Memory allocation fails during computation
/// - The difference algorithm encounters incompatible FST structures
/// - The operation is not yet fully implemented
pub fn difference<W, F1, F2, M>(_fst1: &F1, _fst2: &F2) -> Result<M>
where
    W: Semiring + Hash,
    F1: Fst<W>,
    F2: Fst<W>,
    M: MutableFst<W> + Default,
{
    // for now, just return error
    Err(Error::Algorithm("Difference not fully implemented".into()))
}
