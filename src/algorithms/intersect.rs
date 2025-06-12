//! Intersection algorithm

use crate::algorithms::compose_default;
use crate::fst::{Fst, MutableFst};
use crate::semiring::Semiring;
use crate::Result;

/// Intersect two acceptors
/// 
/// # Errors
/// 
/// Returns an error if:
/// - Either input FST is invalid or has no start state
/// - Memory allocation fails during computation
/// - The intersection operation encounters incompatible FST structures
pub fn intersect<W, F1, F2, M>(fst1: &F1, fst2: &F2) -> Result<M>
where
    W: Semiring,
    F1: Fst<W>,
    F2: Fst<W>,
    M: MutableFst<W> + Default,
{
    // intersection is composition for acceptors
    compose_default(fst1, fst2)
}
