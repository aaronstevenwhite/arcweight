//! Arc sum algorithm for combining duplicate arcs
//!
//! ## Overview
//!
//! Combines duplicate arcs by summing their weights. For arcs with identical
//! (ilabel, olabel, nextstate) tuples from the same source state, replaces them
//! with a single arc whose weight is the sum (⊕) of the original weights.
//!
//! ## Algorithm
//!
//! For each state:
//! 1. Collect all arcs from the state
//! 2. Group arcs by (ilabel, olabel, nextstate) using HashMap
//! 3. For each group: sum weights using semiring addition (⊕)
//! 4. Replace all arcs with one arc per group with summed weight
//!
//! ## Complexity
//!
//! - **Time:** O(|V| + |E| log |E|)
//!   - Iterate over all states: O(|V|)
//!   - For each state with k arcs: O(k log k) for sorting/grouping
//!   - Total arcs processed: O(|E| log |E|)
//!
//! - **Space:** O(|E|) - temporary arc storage
//!
//! ## Use Cases
//!
//! - Simplifying FSTs with redundant transitions
//! - Normalizing FST representation
//! - Preprocessing for determinization
//! - Combining multiple weighted paths
//!
//! ## Examples
//!
//! ```
//! use arcweight::prelude::*;
//!
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! fst.set_start(s0);
//!
//! // Add duplicate arcs with different weights
//! fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(1.0), s1));
//! fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(2.0), s1));
//!
//! arc_sum(&mut fst)?;
//!
//! // Now only one arc with summed weight (min for tropical)
//! assert_eq!(fst.num_arcs(s0), 1);
//! # Ok::<(), arcweight::Error>(())
//! ```

use crate::arc::Arc;
use crate::fst::{Label, MutableFst, StateId};
use crate::semiring::Semiring;
use crate::Result;
use std::collections::HashMap;

/// Arc key for grouping duplicate arcs
type ArcKey = (Label, Label, StateId);

/// Combines duplicate arcs by summing their weights.
///
/// For arcs with identical (ilabel, olabel, nextstate) tuples from the
/// same source state, replaces them with a single arc whose weight is
/// the sum (⊕) of the original weights in the semiring.
///
/// This operation is useful for:
/// - Simplifying FSTs with redundant transitions
/// - Normalizing FST representation
/// - Preprocessing for determinization
///
/// # Algorithm
///
/// For each state:
/// 1. Group arcs by (ilabel, olabel, nextstate)
/// 2. For each group: sum weights using semiring addition (⊕)
/// 3. Replace all arcs in group with single arc with summed weight
///
/// # Complexity
///
/// - **Time:** O(|V| + |E| log |E|)
///   - Iterate states: O(|V|)
///   - Group arcs per state: O(k log k) where k = arcs from state
///   - Total: O(|E| log |E|) across all arcs
///
/// - **Space:** O(|E|) - temporary arc storage
///
/// # Examples
///
/// ## Tropical Semiring (Min Combination)
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// fst.set_start(s0);
///
/// // Add duplicate arcs - tropical takes minimum
/// fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(1.0), s1));
/// fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(2.0), s1));
///
/// arc_sum(&mut fst)?;
///
/// // Result: one arc with weight 1.0 (min)
/// assert_eq!(fst.num_arcs(s0), 1);
/// let arc = fst.arcs(s0).next().unwrap();
/// assert_eq!(arc.weight, TropicalWeight::new(1.0));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Probability Semiring (Sum Combination)
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<ProbabilityWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// fst.set_start(s0);
///
/// // Add duplicate arcs - probability sums them
/// fst.add_arc(s0, Arc::new(1, 2, ProbabilityWeight::new(0.3), s1));
/// fst.add_arc(s0, Arc::new(1, 2, ProbabilityWeight::new(0.4), s1));
///
/// arc_sum(&mut fst)?;
///
/// // Result: one arc with weight 0.7 (sum)
/// assert_eq!(fst.num_arcs(s0), 1);
/// let arc = fst.arcs(s0).next().unwrap();
/// assert_eq!(arc.weight, ProbabilityWeight::new(0.7));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Partial Duplicates
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
/// fst.set_start(s0);
///
/// // Some duplicate, some unique
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1)); // Duplicate
/// fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(3.0), s2)); // Unique
///
/// arc_sum(&mut fst)?;
///
/// // Result: 2 arcs (one combined, one unique)
/// assert_eq!(fst.num_arcs(s0), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## No Duplicates (No Change)
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
/// fst.set_start(s0);
///
/// // All unique arcs
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
/// fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s2));
///
/// arc_sum(&mut fst)?;
///
/// // No change
/// assert_eq!(fst.num_arcs(s0), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn arc_sum<W, F>(fst: &mut F) -> Result<()>
where
    W: Semiring + Clone,
    F: MutableFst<W>,
{
    let num_states = fst.num_states();

    for state in 0..num_states as StateId {
        // Collect all arcs from this state
        let arcs: Vec<Arc<W>> = fst.arcs(state).collect();

        if arcs.is_empty() {
            continue;
        }

        // Group arcs by (ilabel, olabel, nextstate)
        let mut arc_groups: HashMap<ArcKey, W> = HashMap::new();

        for arc in arcs {
            let key = (arc.ilabel, arc.olabel, arc.nextstate);
            arc_groups
                .entry(key)
                .and_modify(|w| *w = w.plus(&arc.weight))
                .or_insert(arc.weight);
        }

        // Clear existing arcs
        fst.delete_arcs(state);

        // Add back one arc per group with summed weight
        for ((ilabel, olabel, nextstate), weight) in arc_groups {
            fst.add_arc(state, Arc::new(ilabel, olabel, weight, nextstate));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_exact_duplicates() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        // Two exact duplicates (same labels, different weights)
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(2.0), s1));

        arc_sum(&mut fst).unwrap();

        assert_eq!(fst.num_arcs(s0), 1);
        let arc = fst.arcs(s0).next().unwrap();
        assert_eq!(arc.ilabel, 1);
        assert_eq!(arc.olabel, 2);
        assert_eq!(arc.weight, TropicalWeight::new(1.0)); // min(1.0, 2.0)
    }

    #[test]
    fn test_multiple_duplicates() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        // Three duplicates
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(3.0), s1));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));

        arc_sum(&mut fst).unwrap();

        assert_eq!(fst.num_arcs(s0), 1);
        let arc = fst.arcs(s0).next().unwrap();
        assert_eq!(arc.weight, TropicalWeight::new(1.0)); // min(3, 1, 2)
    }

    #[test]
    fn test_no_duplicates() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s2));

        arc_sum(&mut fst).unwrap();

        // Should be unchanged
        assert_eq!(fst.num_arcs(s0), 2);
    }

    #[test]
    fn test_partial_duplicates() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1)); // Dup
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(3.0), s2)); // Unique

        arc_sum(&mut fst).unwrap();

        assert_eq!(fst.num_arcs(s0), 2);
    }

    #[test]
    fn test_different_weights() {
        let mut fst = VectorFst::<ProbabilityWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        // Probability semiring sums
        fst.add_arc(s0, Arc::new(1, 1, ProbabilityWeight::new(0.3), s1));
        fst.add_arc(s0, Arc::new(1, 1, ProbabilityWeight::new(0.4), s1));

        arc_sum(&mut fst).unwrap();

        assert_eq!(fst.num_arcs(s0), 1);
        let arc = fst.arcs(s0).next().unwrap();
        assert_eq!(arc.weight, ProbabilityWeight::new(0.7));
    }

    #[test]
    fn test_empty_fst() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        arc_sum(&mut fst).unwrap();
        assert_eq!(fst.num_states(), 0);
    }

    #[test]
    fn test_tropical_semiring() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(5.0), s1));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(3.0), s1));

        arc_sum(&mut fst).unwrap();

        let arc = fst.arcs(s0).next().unwrap();
        assert_eq!(arc.weight, TropicalWeight::new(3.0)); // min
    }

    #[test]
    fn test_log_semiring() {
        let mut fst = VectorFst::<LogWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        fst.add_arc(s0, Arc::new(1, 1, LogWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(1, 1, LogWeight::new(2.0), s1));

        arc_sum(&mut fst).unwrap();

        assert_eq!(fst.num_arcs(s0), 1);
        // Log semiring does log-add-exp
    }

    #[test]
    fn test_multiple_states_with_duplicates() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);

        // Duplicates from s0
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));

        // Duplicates from s1
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(3.0), s2));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(4.0), s2));

        arc_sum(&mut fst).unwrap();

        assert_eq!(fst.num_arcs(s0), 1);
        assert_eq!(fst.num_arcs(s1), 1);
    }

    #[test]
    fn test_same_labels_different_nextstates() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);

        // Same labels but different next states - NOT duplicates
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s2));

        arc_sum(&mut fst).unwrap();

        // Should remain separate
        assert_eq!(fst.num_arcs(s0), 2);
    }
}
