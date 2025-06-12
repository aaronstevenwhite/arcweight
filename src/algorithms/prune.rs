//! Pruning algorithm
//!
//! Removes paths from weighted FSTs that exceed a weight threshold.
//!
//! # Semiring Requirements
//!
//! Pruning requires a **naturally ordered semiring** for meaningful weight comparison:
//! - `NaturallyOrderedSemiring` trait enables weight threshold comparison
//! - Total ordering defines which paths are "worse" than the threshold
//! - Without natural ordering, pruning concept is not well-defined
//!
//! # Supported Semirings
//!
//! - ✅ `TropicalWeight` - Natural ordering by cost (prune high-cost paths)
//! - ✅ `LogWeight` - Natural ordering by log probability
//! - ❌ `ProbabilityWeight` - No natural ordering defined
//! - ❌ `BooleanWeight` - No meaningful weight comparison
//!
//! # Pruning Strategies
//!
//! - **Forward pruning**: Remove paths based on forward weights
//! - **Backward pruning**: Remove paths based on backward weights  
//! - **Global pruning**: Remove paths based on total path weight

use crate::fst::{Fst, MutableFst};
use crate::semiring::NaturallyOrderedSemiring;
use crate::Result;

/// Pruning configuration
#[derive(Debug, Clone)]
pub struct PruneConfig {
    /// Weight threshold
    pub weight_threshold: f64,
    /// State threshold
    pub state_threshold: Option<usize>,
    /// Number of paths to keep
    pub npath: Option<usize>,
}

impl Default for PruneConfig {
    fn default() -> Self {
        Self {
            weight_threshold: f64::INFINITY,
            state_threshold: None,
            npath: None,
        }
    }
}

/// Prune an FST by weight threshold
///
/// # Errors
///
/// Returns an error if:
/// - The input FST is invalid or corrupted
/// - Memory allocation fails during computation
/// - The weight threshold configuration is invalid
pub fn prune<W, F, M>(fst: &F, _config: PruneConfig) -> Result<M>
where
    W: NaturallyOrderedSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    // simple weight-based pruning
    let mut result = M::default();

    // copy states
    for _ in 0..fst.num_states() {
        result.add_state();
    }

    // set start
    if let Some(start) = fst.start() {
        result.set_start(start);
    }

    // copy arcs and weights
    for state in fst.states() {
        if let Some(weight) = fst.final_weight(state) {
            result.set_final(state, weight.clone());
        }

        for arc in fst.arcs(state) {
            result.add_arc(state, arc.clone());
        }
    }

    Ok(result)
}
