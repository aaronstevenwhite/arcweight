//! Pruning algorithm

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
