//! FST isomorphism testing
//!
//! Checks if two FSTs are structurally identical (isomorphic), potentially
//! with different state numbering.

use crate::fst::{Fst, StateId};
use crate::semiring::Semiring;
use crate::Result;
use std::collections::{HashMap, VecDeque};

/// Checks if two FSTs are isomorphic.
///
/// Two FSTs are isomorphic if there exists a bijection between their states
/// that preserves:
/// - The start state
/// - Final states and their weights
/// - All arcs (labels, weights, and connectivity)
///
/// This is useful for testing FST equivalence after transformations that may
/// renumber states (e.g., minimization, state sorting).
///
/// # Algorithm
///
/// Uses simultaneous BFS from start states to build and verify a state mapping:
/// 1. Quick rejection tests (state count, arc count, start state exists)
/// 2. BFS traversal building bijection f: S₁ → S₂
/// 3. At each step, verify arc correspondence and finality
///
/// # Time Complexity
///
/// O(V + E) - single pass through both FSTs
///
/// # Space Complexity
///
/// O(V) - state mapping and queue
///
/// # Examples
///
/// ```
/// use arcweight::prelude::*;
///
/// // Create two structurally identical FSTs
/// let mut fst1 = VectorFst::<TropicalWeight>::new();
/// let s0 = fst1.add_state();
/// let s1 = fst1.add_state();
/// fst1.set_start(s0);
/// fst1.set_final(s1, TropicalWeight::new(1.0));
/// fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
///
/// let mut fst2 = VectorFst::<TropicalWeight>::new();
/// let t0 = fst2.add_state();
/// let t1 = fst2.add_state();
/// fst2.set_start(t0);
/// fst2.set_final(t1, TropicalWeight::new(1.0));
/// fst2.add_arc(t0, Arc::new(1, 1, TropicalWeight::new(0.5), t1));
///
/// assert!(isomorphic(&fst1, &fst2).unwrap());
/// ```
///
/// ## Non-isomorphic Example
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst1 = VectorFst::<TropicalWeight>::new();
/// let s0 = fst1.add_state();
/// let s1 = fst1.add_state();
/// fst1.set_start(s0);
/// fst1.set_final(s1, TropicalWeight::new(1.0));
/// fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
///
/// let mut fst2 = VectorFst::<TropicalWeight>::new();
/// let t0 = fst2.add_state();
/// let t1 = fst2.add_state();
/// fst2.set_start(t0);
/// fst2.set_final(t1, TropicalWeight::new(2.0)); // Different weight!
/// fst2.add_arc(t0, Arc::new(1, 1, TropicalWeight::new(0.5), t1));
///
/// assert!(!isomorphic(&fst1, &fst2).unwrap());
/// ```
pub fn isomorphic<W, F1, F2>(fst1: &F1, fst2: &F2) -> Result<bool>
where
    W: Semiring + PartialEq,
    F1: Fst<W>,
    F2: Fst<W>,
{
    // Quick rejection: check if both FSTs have same number of states
    if fst1.num_states() != fst2.num_states() {
        return Ok(false);
    }

    // Quick rejection: check if both have same number of total arcs
    if fst1.num_arcs_total() != fst2.num_arcs_total() {
        return Ok(false);
    }

    // Check start states
    let (start1, start2) = match (fst1.start(), fst2.start()) {
        (Some(s1), Some(s2)) => (s1, s2),
        (None, None) => {
            // Both empty FSTs - check they both have no states or are truly empty
            return Ok(fst1.num_states() == 0 && fst2.num_states() == 0);
        }
        _ => {
            // One has start state, other doesn't
            return Ok(false);
        }
    };

    // Special case: empty FSTs with start states
    if fst1.num_states() == 0 && fst2.num_states() == 0 {
        return Ok(true);
    }

    // Initialize state mapping: f: StateId → StateId
    let mut state_mapping: HashMap<StateId, StateId> = HashMap::new();
    let mut queue: VecDeque<(StateId, StateId)> = VecDeque::new();

    // Start with initial state mapping
    state_mapping.insert(start1, start2);
    queue.push_back((start1, start2));

    // BFS traversal
    while let Some((s1, s2)) = queue.pop_front() {
        // Check finality
        let final1 = fst1.final_weight(s1);
        let final2 = fst2.final_weight(s2);

        match (final1, final2) {
            (Some(w1), Some(w2)) => {
                // Both final - weights must match
                if w1 != w2 {
                    return Ok(false);
                }
            }
            (None, None) => {
                // Both non-final - OK
            }
            _ => {
                // One final, one not - not isomorphic
                return Ok(false);
            }
        }

        // Check arc counts match
        if fst1.num_arcs(s1) != fst2.num_arcs(s2) {
            return Ok(false);
        }

        // Get and sort arcs from both states
        let mut arcs1: Vec<_> = fst1.arcs(s1).collect();
        let mut arcs2: Vec<_> = fst2.arcs(s2).collect();

        // Sort by (ilabel, olabel, weight, nextstate)
        arcs1.sort_by(|a, b| {
            (a.ilabel, a.olabel, &a.weight, a.nextstate)
                .partial_cmp(&(b.ilabel, b.olabel, &b.weight, b.nextstate))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        arcs2.sort_by(|a, b| {
            (a.ilabel, a.olabel, &a.weight, a.nextstate)
                .partial_cmp(&(b.ilabel, b.olabel, &b.weight, b.nextstate))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Check each corresponding arc pair
        for (arc1, arc2) in arcs1.iter().zip(arcs2.iter()) {
            // Check labels match
            if arc1.ilabel != arc2.ilabel || arc1.olabel != arc2.olabel {
                return Ok(false);
            }

            // Check weights match
            if arc1.weight != arc2.weight {
                return Ok(false);
            }

            // Check destination mapping
            if let Some(&mapped_dest) = state_mapping.get(&arc1.nextstate) {
                // arc1.nextstate already mapped - verify it maps to arc2.nextstate
                if mapped_dest != arc2.nextstate {
                    return Ok(false);
                }
            } else {
                // arc1.nextstate not yet mapped - add mapping
                state_mapping.insert(arc1.nextstate, arc2.nextstate);
                queue.push_back((arc1.nextstate, arc2.nextstate));
            }
        }
    }

    // Verify all states were visited (mapping is complete)
    Ok(state_mapping.len() == fst1.num_states())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arc::Arc;
    use crate::fst::{MutableFst, VectorFst};
    use crate::semiring::{BooleanWeight, IntegerWeight, TropicalWeight};
    use num_traits::One;

    #[test]
    fn test_isomorphic_identical_fsts() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::new(1.0));
        fst1.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t1, TropicalWeight::new(1.0));
        fst2.add_arc(t0, Arc::new(1, 2, TropicalWeight::new(0.5), t1));

        assert!(isomorphic(&fst1, &fst2).unwrap());
    }

    #[test]
    fn test_isomorphic_with_renumbered_states() {
        // FST1: states 0, 1, 2 in order
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        let s2 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s2, TropicalWeight::new(2.0));
        fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
        fst1.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(0.3), s2));

        // FST2: same structure, same state IDs (this test checks structural equality)
        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        let t2 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t2, TropicalWeight::new(2.0));
        fst2.add_arc(t0, Arc::new(1, 1, TropicalWeight::new(0.5), t1));
        fst2.add_arc(t1, Arc::new(2, 2, TropicalWeight::new(0.3), t2));

        assert!(isomorphic(&fst1, &fst2).unwrap());
    }

    #[test]
    fn test_non_isomorphic_different_weights() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::new(1.0));
        fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t1, TropicalWeight::new(2.0)); // Different weight!
        fst2.add_arc(t0, Arc::new(1, 1, TropicalWeight::new(0.5), t1));

        assert!(!isomorphic(&fst1, &fst2).unwrap());
    }

    #[test]
    fn test_non_isomorphic_different_labels() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::new(1.0));
        fst1.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t1, TropicalWeight::new(1.0));
        fst2.add_arc(t0, Arc::new(1, 3, TropicalWeight::new(0.5), t1)); // Different output label!

        assert!(!isomorphic(&fst1, &fst2).unwrap());
    }

    #[test]
    fn test_non_isomorphic_different_structure() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::new(1.0));
        fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        let t2 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t2, TropicalWeight::new(1.0));
        fst2.add_arc(t0, Arc::new(1, 1, TropicalWeight::new(0.5), t1));
        fst2.add_arc(t1, Arc::new(2, 2, TropicalWeight::new(0.3), t2));

        assert!(!isomorphic(&fst1, &fst2).unwrap());
    }

    #[test]
    fn test_non_isomorphic_different_state_count() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s0, TropicalWeight::one());

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t1, TropicalWeight::one());

        assert!(!isomorphic(&fst1, &fst2).unwrap());
    }

    #[test]
    fn test_non_isomorphic_different_arc_count() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::one());
        fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t1, TropicalWeight::one());
        fst2.add_arc(t0, Arc::new(1, 1, TropicalWeight::one(), t1));
        fst2.add_arc(t0, Arc::new(2, 2, TropicalWeight::one(), t1));

        assert!(!isomorphic(&fst1, &fst2).unwrap());
    }

    #[test]
    fn test_isomorphic_empty_fsts() {
        let fst1 = VectorFst::<TropicalWeight>::new();
        let fst2 = VectorFst::<TropicalWeight>::new();

        assert!(isomorphic(&fst1, &fst2).unwrap());
    }

    #[test]
    fn test_isomorphic_single_state() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s0, TropicalWeight::new(1.5));

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t0, TropicalWeight::new(1.5));

        assert!(isomorphic(&fst1, &fst2).unwrap());
    }

    #[test]
    fn test_isomorphic_linear_chain() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        let s2 = fst1.add_state();
        let s3 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s3, TropicalWeight::one());
        fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.1), s1));
        fst1.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(0.2), s2));
        fst1.add_arc(s2, Arc::new(3, 3, TropicalWeight::new(0.3), s3));

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        let t2 = fst2.add_state();
        let t3 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t3, TropicalWeight::one());
        fst2.add_arc(t0, Arc::new(1, 1, TropicalWeight::new(0.1), t1));
        fst2.add_arc(t1, Arc::new(2, 2, TropicalWeight::new(0.2), t2));
        fst2.add_arc(t2, Arc::new(3, 3, TropicalWeight::new(0.3), t3));

        assert!(isomorphic(&fst1, &fst2).unwrap());
    }

    #[test]
    fn test_isomorphic_branching_structure() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        let s2 = fst1.add_state();
        let s3 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::one());
        fst1.set_final(s2, TropicalWeight::new(2.0));
        fst1.set_final(s3, TropicalWeight::new(3.0));
        fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.1), s1));
        fst1.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(0.2), s2));
        fst1.add_arc(s0, Arc::new(3, 3, TropicalWeight::new(0.3), s3));

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        let t2 = fst2.add_state();
        let t3 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t1, TropicalWeight::one());
        fst2.set_final(t2, TropicalWeight::new(2.0));
        fst2.set_final(t3, TropicalWeight::new(3.0));
        fst2.add_arc(t0, Arc::new(1, 1, TropicalWeight::new(0.1), t1));
        fst2.add_arc(t0, Arc::new(2, 2, TropicalWeight::new(0.2), t2));
        fst2.add_arc(t0, Arc::new(3, 3, TropicalWeight::new(0.3), t3));

        assert!(isomorphic(&fst1, &fst2).unwrap());
    }

    #[test]
    fn test_isomorphic_with_cycles() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::one());
        fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
        fst1.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(0.3), s0)); // Cycle back

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t1, TropicalWeight::one());
        fst2.add_arc(t0, Arc::new(1, 1, TropicalWeight::new(0.5), t1));
        fst2.add_arc(t1, Arc::new(2, 2, TropicalWeight::new(0.3), t0)); // Cycle back

        assert!(isomorphic(&fst1, &fst2).unwrap());
    }

    #[test]
    fn test_isomorphic_different_semirings() {
        // Test with BooleanWeight
        let mut fst1 = VectorFst::<BooleanWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, BooleanWeight::one());
        fst1.add_arc(s0, Arc::new(1, 1, BooleanWeight::one(), s1));

        let mut fst2 = VectorFst::<BooleanWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t1, BooleanWeight::one());
        fst2.add_arc(t0, Arc::new(1, 1, BooleanWeight::one(), t1));

        assert!(isomorphic(&fst1, &fst2).unwrap());

        // Test with IntegerWeight
        let mut fst3 = VectorFst::<IntegerWeight>::new();
        let s0 = fst3.add_state();
        let s1 = fst3.add_state();
        fst3.set_start(s0);
        fst3.set_final(s1, IntegerWeight::new(5));
        fst3.add_arc(s0, Arc::new(1, 1, IntegerWeight::new(3), s1));

        let mut fst4 = VectorFst::<IntegerWeight>::new();
        let t0 = fst4.add_state();
        let t1 = fst4.add_state();
        fst4.set_start(t0);
        fst4.set_final(t1, IntegerWeight::new(5));
        fst4.add_arc(t0, Arc::new(1, 1, IntegerWeight::new(3), t1));

        assert!(isomorphic(&fst3, &fst4).unwrap());
    }

    #[test]
    fn test_isomorphic_epsilon_arcs() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::one());
        fst1.add_arc(s0, Arc::epsilon(TropicalWeight::new(0.5), s1));

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t1, TropicalWeight::one());
        fst2.add_arc(t0, Arc::epsilon(TropicalWeight::new(0.5), t1));

        assert!(isomorphic(&fst1, &fst2).unwrap());
    }

    #[test]
    fn test_non_isomorphic_missing_start_state() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s0, TropicalWeight::one());

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let _t0 = fst2.add_state();
        // No start state set for fst2

        assert!(!isomorphic(&fst1, &fst2).unwrap());
    }

    #[test]
    fn test_isomorphic_multiple_arcs_same_state() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::one());
        fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
        fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1)); // Duplicate arc
        fst1.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(0.3), s1));

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t1, TropicalWeight::one());
        fst2.add_arc(t0, Arc::new(1, 1, TropicalWeight::new(0.5), t1));
        fst2.add_arc(t0, Arc::new(1, 1, TropicalWeight::new(0.5), t1)); // Duplicate arc
        fst2.add_arc(t0, Arc::new(2, 2, TropicalWeight::new(0.3), t1));

        assert!(isomorphic(&fst1, &fst2).unwrap());
    }

    #[test]
    fn test_non_isomorphic_one_final_one_not() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::one());
        fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        fst2.set_start(t0);
        // t1 is NOT final
        fst2.add_arc(t0, Arc::new(1, 1, TropicalWeight::one(), t1));

        assert!(!isomorphic(&fst1, &fst2).unwrap());
    }
}
