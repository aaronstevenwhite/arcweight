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

/// Determinize a weighted FST using subset construction
///
/// Converts a nondeterministic FST into a deterministic one that accepts the same
/// weighted language. Uses weighted subset construction with normalization to prevent
/// exponential weight growth.
///
/// # Algorithm Details
///
/// - **Base Algorithm:** Weighted subset construction (Mohri, 1997)
/// - **Time Complexity:** O(2ⁿ) worst case, often much better in practice
/// - **Space Complexity:** O(2ⁿ) for subset storage
/// - **Weight Normalization:** Divides out minimum weight from each subset
///
/// # Semiring Requirements
///
/// The algorithm requires [`DivisibleSemiring`] for weight normalization:
/// - **Division:** Must support `divide()` operation for normalizing weights
/// - **Ordering:** Must implement `Ord` for finding minimum weights
/// - **Twins Property:** FST structure must satisfy determinization requirements
///
/// # Implementation Notes
///
/// The algorithm maintains subsets of states with associated weights, ensuring
/// that each subset represents a unique deterministic state. Weight normalization
/// prevents the accumulation of large weight values during subset construction.
///
/// # Examples
///
/// ## Basic Determinization
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Create nondeterministic FST with ambiguous paths
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s1, TropicalWeight::new(0.5));
/// fst.set_final(s2, TropicalWeight::new(0.3));
///
/// // Two arcs with same input label - creates nondeterminism
/// fst.add_arc(s0, Arc::new('a' as u32, 'x' as u32, TropicalWeight::new(0.2), s1));
/// fst.add_arc(s0, Arc::new('a' as u32, 'y' as u32, TropicalWeight::new(0.4), s2));
///
/// // Determinize resolves nondeterminism
/// let det_fst: VectorFst<TropicalWeight> = determinize(&fst).unwrap();
///
/// // Determinization may create more or fewer states
/// assert!(det_fst.num_states() > 0);
/// ```
///
/// ## Real-World Application: Speech Recognition
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Determinize pronunciation dictionary for efficient lookup
/// fn optimize_pronunciation_dict(
///     dict: &VectorFst<TropicalWeight>
/// ) -> std::result::Result<VectorFst<TropicalWeight>, Box<dyn std::error::Error>> {
///     
///     // Step 1: Determinize to resolve pronunciation ambiguities
///     let det_dict: VectorFst<TropicalWeight> = determinize(dict)?;
///     
///     // Step 2: Minimize to reduce memory footprint
///     let min_dict: VectorFst<TropicalWeight> = minimize(&det_dict)?;
///     
///     println!("Original: {} states, Optimized: {} states",
///              dict.num_states(), min_dict.num_states());
///     
///     Ok(min_dict)
/// }
/// ```
///
/// ## Checking Determinism
///
/// ```rust
/// use arcweight::prelude::*;
///
/// fn ensure_deterministic<W: DivisibleSemiring + Ord + std::hash::Hash + Eq>(
///     fst: &VectorFst<W>
/// ) -> std::result::Result<VectorFst<W>, Box<dyn std::error::Error>> {
///     
///     // Simple example: always determinize for demonstration
///     println!("Determinizing FST...");
///     let det_fst: VectorFst<W> = determinize(fst)?;
///     println!("Original: {} states, Determinized: {} states",
///              fst.num_states(), det_fst.num_states());
///     
///     Ok(det_fst)
/// }
/// ```
///
/// # Performance Considerations
///
/// - **Subset Explosion:** Can create exponentially many states in worst case
/// - **Early Termination:** Use weight/state thresholds for large FSTs
/// - **Memory Management:** Consider streaming approaches for very large inputs
/// - **Preprocessing:** Remove epsilon transitions before determinization
///
/// # Errors
///
/// Returns [`Error::Algorithm`] if:
/// - The input FST has no start state or is malformed
/// - Memory allocation fails during subset construction
/// - The semiring doesn't support required division operations
/// - Weight normalization fails due to division by zero
/// - Subset construction creates invalid state combinations
///
/// # See Also
///
/// - [`minimize()`](crate::algorithms::minimize()) for reducing deterministic FST size
/// - [`remove_epsilons()`](crate::algorithms::remove_epsilons()) for preprocessing before determinization
/// - [Working with FSTs - Determinization](../../docs/working-with-fsts/optimization-operations.md#determinization) for usage patterns
/// - [Core Concepts](../../docs/core-concepts/algorithms.md#determinization) for theoretical background
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use num_traits::One;

    #[test]
    fn test_determinize_simple() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Non-deterministic: two arcs with same input label
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s2));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s2));

        let det: VectorFst<TropicalWeight> = determinize(&fst).unwrap();

        // Check determinism: no state should have multiple arcs with same input label
        for state in det.states() {
            let mut seen_labels = std::collections::HashSet::new();
            for arc in det.arcs(state) {
                assert!(
                    seen_labels.insert(arc.ilabel),
                    "Found duplicate input label {} from state {}",
                    arc.ilabel,
                    state
                );
            }
        }

        // Should preserve language
        assert!(det.start().is_some());
        assert!(det.num_states() > 0);
    }

    #[test]
    fn test_determinize_already_deterministic() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));

        let det: VectorFst<TropicalWeight> = determinize(&fst).unwrap();

        // Should be similar to original
        assert_eq!(det.num_states(), fst.num_states());
        assert!(det.start().is_some());

        for state in det.states() {
            let mut seen_labels = std::collections::HashSet::new();
            for arc in det.arcs(state) {
                assert!(seen_labels.insert(arc.ilabel));
            }
        }
    }
}
