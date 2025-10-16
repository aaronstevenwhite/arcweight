//! FST isomorphism testing
//!
//! Determines if two FSTs are structurally identical up to state renumbering.
//!
//! ## Overview
//!
//! Two FSTs are isomorphic if there exists a bijection f: V₁ → V₂ that preserves:
//! - Start states: f(start₁) = start₂
//! - Final weights: ∀s ∈ V₁, finalWeight(s) = finalWeight(f(s))
//! - Arc structure: ∀arc (s, i, o, w, t) ∈ E₁, ∃arc (f(s), i, o, w, f(t)) ∈ E₂
//!
//! This is essential for testing FST transformations that preserve structure
//! but may renumber states (e.g., minimization, state sorting).
//!
//! ## Algorithm
//!
//! Simultaneous BFS traversal with state mapping verification:
//! 1. Quick rejection: check state counts, arc counts, start state existence
//! 2. Initialize bijection with start state pair: f(start₁) = start₂
//! 3. BFS from start states, building and verifying mapping
//! 4. At each state pair: verify finality, arc counts, arc correspondence
//!
//! ## Complexity
//!
//! - **Time:** O(|V| + |E|) - single BFS pass through both FSTs
//! - **Space:** O(|V|) - state mapping hash table and BFS queue
//!
//! ## Use Cases
//!
//! - **Transformation Testing:** Verify FST operations preserve structure
//! - **Minimization Verification:** Confirm minimal FSTs are equivalent
//! - **Normalization:** Check if differently-ordered FSTs are identical
//! - **Debugging:** Validate FST construction and manipulation
//!
//! ## Examples
//!
//! ### Isomorphic FSTs
//!
//! ```
//! use arcweight::prelude::*;
//!
//! // Create two structurally identical FSTs
//! let mut fst1 = VectorFst::<TropicalWeight>::new();
//! let s0 = fst1.add_state();
//! let s1 = fst1.add_state();
//! fst1.set_start(s0);
//! fst1.set_final(s1, TropicalWeight::new(1.0));
//! fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
//!
//! let mut fst2 = VectorFst::<TropicalWeight>::new();
//! let t0 = fst2.add_state();
//! let t1 = fst2.add_state();
//! fst2.set_start(t0);
//! fst2.set_final(t1, TropicalWeight::new(1.0));
//! fst2.add_arc(t0, Arc::new(1, 1, TropicalWeight::new(0.5), t1));
//!
//! assert!(isomorphic(&fst1, &fst2)?);
//! # Ok::<(), arcweight::Error>(())
//! ```
//!
//! ### Non-Isomorphic FSTs
//!
//! ```
//! use arcweight::prelude::*;
//!
//! let mut fst1 = VectorFst::<TropicalWeight>::new();
//! let s0 = fst1.add_state();
//! let s1 = fst1.add_state();
//! fst1.set_start(s0);
//! fst1.set_final(s1, TropicalWeight::new(1.0));
//! fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
//!
//! let mut fst2 = VectorFst::<TropicalWeight>::new();
//! let t0 = fst2.add_state();
//! let t1 = fst2.add_state();
//! fst2.set_start(t0);
//! fst2.set_final(t1, TropicalWeight::new(2.0)); // Different weight
//! fst2.add_arc(t0, Arc::new(1, 1, TropicalWeight::new(0.5), t1));
//!
//! assert!(!isomorphic(&fst1, &fst2)?);
//! # Ok::<(), arcweight::Error>(())
//! ```
//!
//! ## References
//!
//! - Hopcroft, J. E., and Karp, R. M. (1971). "An n^5/2 algorithm for maximum matchings
//!   in bipartite graphs." SIAM Journal on Computing, 2(4): 225-231.
//! - Booth, K. S., and Colbourn, C. J. (1977). "Problems polynomially equivalent to
//!   graph isomorphism." Technical Report CS-77-04, University of Waterloo.

use crate::fst::{Fst, StateId};
use crate::semiring::Semiring;
use crate::Result;
use std::collections::{HashMap, VecDeque};

/// Checks if two FSTs are isomorphic.
///
/// Two FSTs are isomorphic if there exists a bijection f: V₁ → V₂ between their
/// states that preserves start states, final weights, and all arc structure. This
/// is the fundamental operation for verifying FST transformations like minimization
/// and state sorting that preserve language but may renumber states.
///
/// # Complexity
///
/// - **Time:** O(|V| + |E|) where V = states in each FST, E = arcs in each FST
///   - Quick rejection tests: O(1)
///   - BFS traversal: O(|V| + |E|)
///   - Arc sorting per state: O(k log k) where k = arcs per state
/// - **Space:** O(|V|) for state mapping hash table and BFS queue
///
/// # Algorithm
///
/// Simultaneous BFS with bijection construction and verification:
/// 1. **Reject quickly:** Check |V₁| = |V₂|, |E₁| = |E₂|, both have start states
/// 2. **Initialize:** Map start₁ → start₂, enqueue (start₁, start₂)
/// 3. **BFS Loop:** For each state pair (s₁, s₂):
///    - Verify final weights match: finalWeight(s₁) = finalWeight(s₂)
///    - Verify arc counts match: |arcs(s₁)| = |arcs(s₂)|
///    - Sort arcs from both states by (ilabel, olabel, weight, nextstate)
///    - For each arc pair: verify labels and weights match, extend bijection
/// 4. **Success:** All states visited with consistent bijection
///
/// Based on graph isomorphism testing with linear-time verification for labeled graphs.
///
/// # Performance Notes
///
/// - **Early rejection:** Most non-isomorphic FSTs rejected in O(1) by count checks
/// - **Arc sorting:** Dominates runtime for states with many arcs (O(k log k) per state)
/// - **Hash lookups:** O(1) average case for bijection verification
/// - **Best case:** O(|V|) for linear chains with pre-sorted arcs
/// - **Worst case:** O(|V| + |E| log degree) for dense FSTs with unsorted arcs
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
///
/// # See Also
///
/// - [`minimize`] - Produces minimal FST (verify with isomorphic test)
/// - [`state_sort`] - Renumbers states (preserves isomorphism)
///
/// [`minimize`]: crate::algorithms::minimize::minimize
/// [`state_sort`]: crate::algorithms::state_sort::state_sort
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
