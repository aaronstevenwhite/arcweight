//! Minimization algorithm

use crate::algorithms::{connect, determinize, reverse};
use crate::fst::{Fst, MutableFst};
use crate::semiring::DivisibleSemiring;
use crate::Result;
use core::hash::Hash;

/// Minimize a deterministic FST
/// 
/// # Errors
/// 
/// Returns an error if:
/// - The input FST is invalid or corrupted
/// - Memory allocation fails during computation
/// - The semiring does not support required operations
/// - Any intermediate algorithm step (reverse, determinize, connect) fails
pub fn minimize<W, F, M>(fst: &F) -> Result<M>
where
    W: DivisibleSemiring + Hash + Eq + Ord,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    // brzozowski's algorithm:
    // 1. reverse
    // 2. determinize
    // 3. reverse
    // 4. determinize

    let rev1: M = reverse(fst)?;
    let det1: M = determinize(&rev1)?;
    let rev2: M = reverse(&det1)?;
    let det2: M = determinize(&rev2)?;

    // ensure connected
    connect(&det2)
}
