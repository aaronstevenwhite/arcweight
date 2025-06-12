//! Minimization algorithm

use crate::algorithms::{connect, determinize, reverse};
use crate::fst::{Fst, MutableFst};
use crate::semiring::DivisibleSemiring;
use crate::Result;
use core::hash::Hash;

/// Minimize a deterministic FST
///
/// Uses Brzozowski's algorithm: reverse → determinize → reverse → determinize → connect.
/// The input FST should be deterministic for best results.
///
/// # Examples
///
/// ```
/// use arcweight::prelude::*;
///
/// // Create an FST with redundant states
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
/// let s3 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s2, TropicalWeight::one());
/// fst.set_final(s3, TropicalWeight::one());
///
/// // Two paths to equivalent final states
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
/// fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));
/// fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s3));
///
/// // Minimize
/// let minimized: VectorFst<TropicalWeight> = minimize(&fst).unwrap();
///
/// // Should produce a valid minimized FST
/// assert!(minimized.num_states() > 0);
/// ```
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
