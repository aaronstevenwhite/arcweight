//! Determinization algorithm
//!
//! Converts a nondeterministic weighted FST to a deterministic one using subset construction.
//!
//! # Semiring Requirements
//!
//! This operation requires the semiring to be **weakly left divisible**, implemented as
//! the `DivisibleSemiring` trait. This enables weight normalization during subset construction:
//! - Each subset's minimum weight is divided out to prevent exponential weight growth
//! - Division operations must be well-defined and consistent
//!
//! Additionally, the weights must be `Ord` to enable finding minimum weights for normalization.
//!
//! # Supported Semirings
//!
//! - ✅ `TropicalWeight` - Implements `DivisibleSemiring + Ord`
//! - ✅ `LogWeight` - Implements `DivisibleSemiring + Ord`  
//! - ❌ `ProbabilityWeight` - Not weakly left divisible
//! - ❌ `BooleanWeight` - No natural division operation
//!
//! # Notes
//!
//! Even with a compatible semiring, not all weighted FSTs can be determinized due to
//! structural requirements like the "twins property". The algorithm will return an error
//! if determinization is not possible for the given FST.

use crate::arc::Arc;
use crate::fst::{Fst, Label, MutableFst, StateId};
use crate::semiring::{DivisibleSemiring, Semiring};
use crate::{Error, Result};
use core::hash::Hash;
use std::collections::{BTreeMap, HashMap};

/// Weighted subset (determinization state)
#[derive(Clone, Debug, PartialEq)]
struct WeightedSubset<W: Semiring> {
    /// States with their weights
    states: BTreeMap<StateId, W>,
}

impl<W: Semiring> Eq for WeightedSubset<W> where W: Eq {}

impl<W: Semiring> std::hash::Hash for WeightedSubset<W>
where
    W: std::hash::Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.states.hash(state);
    }
}

impl<W: Semiring> WeightedSubset<W> {
    fn new() -> Self {
        Self {
            states: BTreeMap::new(),
        }
    }

    fn insert(&mut self, state: StateId, weight: W) {
        self.states
            .entry(state)
            .and_modify(|w| w.plus_assign(&weight))
            .or_insert(weight);
    }

    fn normalize(&mut self) -> Option<W>
    where
        W: DivisibleSemiring + Ord,
    {
        // find minimum weight
        let min_weight = self.states.values().min()?.clone();

        // divide all weights by minimum
        for weight in self.states.values_mut() {
            *weight = weight.divide(&min_weight)?;
        }

        Some(min_weight)
    }
}

/// Determinize an FST
///
/// Converts a non-deterministic FST into an equivalent deterministic FST
/// using the subset construction algorithm with weight normalization.
///
/// # Examples
///
/// ```
/// use arcweight::prelude::*;
///
/// // Create a non-deterministic FST with two paths for label 1
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s1, TropicalWeight::new(0.5));
/// fst.set_final(s2, TropicalWeight::new(0.3));
///
/// // Two arcs with same label from start state
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.2), s1));
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.4), s2));
///
/// // Determinize
/// let det_fst: VectorFst<TropicalWeight> = determinize(&fst).unwrap();
///
/// // Result should be deterministic
/// assert!(det_fst.num_states() > 0);
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - The input FST is invalid or has no start state
/// - Memory allocation fails during computation
/// - The semiring does not support required division operations
/// - Subset construction encounters invalid state combinations
pub fn determinize<W, F, M>(fst: &F) -> Result<M>
where
    W: DivisibleSemiring + Hash + Eq + Ord,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let start = fst
        .start()
        .ok_or_else(|| Error::Algorithm("FST has no start state".into()))?;

    let mut result = M::default();
    let mut subset_map = HashMap::new();
    let mut queue = Vec::new();

    // create initial subset
    let mut start_subset = WeightedSubset::new();
    start_subset.insert(start, W::one());

    // create start state
    let start_new = result.add_state();
    result.set_start(start_new);
    subset_map.insert(start_subset.clone(), start_new);
    queue.push((start_subset, start_new));

    // process subsets
    while let Some((subset, current_state)) = queue.pop() {
        // compute outgoing transitions by label
        let mut transitions: HashMap<Label, WeightedSubset<W>> = HashMap::new();
        let mut final_weight = W::zero();

        for (&state, weight) in &subset.states {
            // accumulate final weights
            if let Some(fw) = fst.final_weight(state) {
                final_weight.plus_assign(&weight.times(fw));
            }

            // process arcs
            for arc in fst.arcs(state) {
                let next_weight = weight.times(&arc.weight);
                transitions
                    .entry(arc.ilabel)
                    .or_insert_with(WeightedSubset::new)
                    .insert(arc.nextstate, next_weight);
            }
        }

        // set final weight if non-zero
        if !<W as num_traits::Zero>::is_zero(&final_weight) {
            result.set_final(current_state, final_weight);
        }

        // add transitions
        for (label, mut next_subset) in transitions {
            // normalize subset
            if let Some(norm_weight) = next_subset.normalize() {
                let next_state = match subset_map.get(&next_subset) {
                    Some(&state) => state,
                    None => {
                        let state = result.add_state();
                        subset_map.insert(next_subset.clone(), state);
                        queue.push((next_subset, state));
                        state
                    }
                };

                result.add_arc(
                    current_state,
                    Arc::new(label, label, norm_weight, next_state),
                );
            }
        }
    }

    Ok(result)
}
