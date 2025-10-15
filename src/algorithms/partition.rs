//! Partition FST states into equivalence classes
//!
//! Computes a partition of FST states based on bisimulation equivalence,
//! grouping states that cannot be distinguished by any sequence of transitions.
//!
//! ## Overview
//!
//! State partitioning identifies equivalence classes of states based on their
//! observable behavior. Two states are equivalent if they have the same final
//! weight and identical transition structure to equivalent states.
//!
//! ## Algorithm
//!
//! Uses iterative partition refinement (Hopcroft-style):
//! 1. Initialize partition based on final weights
//! 2. Refine partition based on arc signatures
//! 3. Continue until partition stabilizes (no further refinement)
//!
//! ## Complexity
//!
//! - **Time:** O(|E| log |V|) - iterative refinement with logarithmic depth
//! - **Space:** O(|V|) - partition mapping and signature storage
//!
//! ## Theoretical Background
//!
//! Partition refinement is based on bisimulation equivalence from process algebra.
//! States s and t are bisimilar if:
//! - They have the same final weight
//! - For every arc from s, there exists a matching arc from t to an equivalent state
//! - Vice versa
//!
//! This is the foundation of FST minimization algorithms.
//!
//! ## Use Cases
//!
//! - **Minimization:** Core subroutine for state minimization
//! - **Equivalence Testing:** Identify redundant states
//! - **Analysis:** Understand FST structure and symmetries
//! - **Optimization:** Preprocessing for other algorithms
//!
//! ## Examples
//!
//! ### Simple Partition
//!
//! ```
//! use arcweight::prelude::*;
//!
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! let s2 = fst.add_state();
//! fst.set_start(s0);
//! fst.set_final(s1, TropicalWeight::one());
//! fst.set_final(s2, TropicalWeight::one());
//! fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
//! fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s2));
//!
//! let classes = partition(&fst)?;
//!
//! // s1 and s2 are equivalent (both final with same weight, no outgoing arcs)
//! assert_eq!(classes[s1], classes[s2]);
//! assert_ne!(classes[s0], classes[s1]);
//! # Ok::<(), arcweight::Error>(())
//! ```
//!
//! ### Minimal FST
//!
//! ```
//! use arcweight::prelude::*;
//!
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! fst.set_start(s0);
//! fst.set_final(s1, TropicalWeight::one());
//! fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
//!
//! let classes = partition(&fst)?;
//!
//! // Already minimal: each state in own equivalence class
//! assert_eq!(classes.len(), 2);
//! assert_ne!(classes[s0], classes[s1]);
//! # Ok::<(), arcweight::Error>(())
//! ```

use crate::fst::{Fst, Label, StateId};
use crate::semiring::Semiring;
use crate::Result;
use std::collections::{HashMap, HashSet};

/// Partitions FST states into equivalence classes.
///
/// Uses iterative partition refinement based on bisimulation equivalence.
/// Returns a vector mapping each state ID to its equivalence class ID.
///
/// # Algorithm
///
/// 1. **Initialize:** Partition by final weights
/// 2. **Refine:** Split classes based on arc signatures
/// 3. **Iterate:** Continue until no further refinement possible
///
/// Two states are in the same equivalence class if they:
/// - Have the same final weight (or both non-final)
/// - Have identical arc signatures (same labels/weights to same classes)
///
/// # Time Complexity
///
/// O(|E| log |V|) - logarithmic refinement depth
///
/// # Space Complexity
///
/// O(|V| + |E|) - partition storage and arc signatures
///
/// # Examples
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
/// fst.set_start(s0);
/// fst.set_final(s1, TropicalWeight::one());
/// fst.set_final(s2, TropicalWeight::new(2.0));
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
/// fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s2));
///
/// let classes = partition(&fst)?;
///
/// // s0 is non-final, s1 and s2 are final with different weights
/// assert_eq!(classes.len(), 3);
/// assert_ne!(classes[s0], classes[s1]);
/// assert_ne!(classes[s1], classes[s2]);
/// # Ok::<(), arcweight::Error>(())
/// ```
///
/// # Returns
///
/// Vector of length `fst.num_states()` where `result[state]` is the
/// equivalence class ID for that state. Class IDs are consecutive
/// integers starting from 0.
///
/// # Errors
///
/// Returns error if FST structure is invalid.
pub fn partition<W, F>(fst: &F) -> Result<Vec<StateId>>
where
    W: Semiring + Eq + std::hash::Hash,
    F: Fst<W>,
{
    let n = fst.num_states();

    if n == 0 {
        return Ok(Vec::new());
    }

    // Initialize partition based on final weights
    let mut class_map = initialize_partition(fst);
    let mut changed = true;

    // Iteratively refine partition until stable
    while changed {
        changed = false;
        let old_class_map = class_map.clone();

        // Build reverse mapping: class -> states
        let mut classes: HashMap<StateId, Vec<StateId>> = HashMap::new();
        for (state_idx, class_id) in old_class_map.iter().enumerate() {
            let state = state_idx as StateId;
            classes.entry(*class_id).or_default().push(state);
        }

        // Try to split each class
        for states_in_class in classes.values() {
            if states_in_class.len() <= 1 {
                continue; // Can't split singleton classes
            }

            // Compute signatures for all states in this class
            let signatures: HashMap<StateId, Signature> = states_in_class
                .iter()
                .map(|&state| (state, compute_signature(fst, state, &old_class_map)))
                .collect();

            // Group states by signature
            let mut sig_groups: HashMap<Signature, Vec<StateId>> = HashMap::new();
            for (&state, sig) in &signatures {
                sig_groups.entry(sig.clone()).or_default().push(state);
            }

            // If multiple signature groups, we need to split this class
            if sig_groups.len() > 1 {
                changed = true;

                // Assign new class IDs to split groups
                let max_class = *class_map.iter().max().unwrap_or(&0);
                let mut new_class_id = max_class + 1;

                for (idx, group) in sig_groups.values().enumerate() {
                    let target_class = if idx == 0 {
                        // Keep first group in original class
                        old_class_map[group[0] as usize]
                    } else {
                        // Assign new class to other groups
                        let class_id = new_class_id;
                        new_class_id += 1;
                        class_id
                    };

                    for &state in group {
                        class_map[state as usize] = target_class;
                    }
                }
            }
        }
    }

    // Renumber classes to be consecutive starting from 0
    renumber_classes(&mut class_map);

    Ok(class_map)
}

/// Initialize partition based on final weights
fn initialize_partition<W, F>(fst: &F) -> Vec<StateId>
where
    W: Semiring + Eq + std::hash::Hash,
    F: Fst<W>,
{
    let n = fst.num_states();
    let mut class_map = vec![0; n];

    // Group states by final weight
    let mut weight_to_class: HashMap<Option<W>, StateId> = HashMap::new();
    let mut next_class = 0;

    for (state_idx, class_entry) in class_map.iter_mut().enumerate().take(n) {
        let state = state_idx as StateId;
        let final_weight = fst.final_weight(state).cloned();

        let class_id = weight_to_class.entry(final_weight).or_insert_with(|| {
            let id = next_class;
            next_class += 1;
            id
        });

        *class_entry = *class_id;
    }

    class_map
}

/// Signature type for state equivalence
type Signature = Vec<(Label, Label, String, StateId)>; // (ilabel, olabel, weight_str, dest_class)

/// Compute signature of a state based on its arcs
fn compute_signature<W, F>(fst: &F, state: StateId, class_map: &[StateId]) -> Signature
where
    W: Semiring,
    F: Fst<W>,
{
    let mut sig: Signature = fst
        .arcs(state)
        .map(|arc| {
            (
                arc.ilabel,
                arc.olabel,
                format!("{:?}", arc.weight), // Use Debug formatting for weight
                class_map[arc.nextstate as usize],
            )
        })
        .collect();

    // Sort for canonical representation
    sig.sort();
    sig
}

/// Renumber classes to be consecutive starting from 0
fn renumber_classes(class_map: &mut [StateId]) {
    let unique_classes: HashSet<StateId> = class_map.iter().copied().collect();
    let mut sorted_classes: Vec<StateId> = unique_classes.into_iter().collect();
    sorted_classes.sort();

    let renumbering: HashMap<StateId, StateId> = sorted_classes
        .into_iter()
        .enumerate()
        .map(|(new_id, old_id)| (old_id, new_id as StateId))
        .collect();

    for class_id in class_map.iter_mut() {
        *class_id = renumbering[class_id];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_partition_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();
        let classes = partition(&fst).unwrap();
        assert_eq!(classes.len(), 0);
    }

    #[test]
    fn test_partition_single_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::one());

        let classes = partition(&fst).unwrap();
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0], 0);
    }

    #[test]
    fn test_partition_two_equivalent_states() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.set_final(s2, TropicalWeight::one());

        // s0 has arcs to s1 and s2, which are equivalent
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s2));

        let classes = partition(&fst).unwrap();

        // s1 and s2 should be in same class (both final, no outgoing arcs)
        assert_eq!(classes[s1 as usize], classes[s2 as usize]);
        // s0 should be in different class (non-final)
        assert_ne!(classes[s0 as usize], classes[s1 as usize]);
    }

    #[test]
    fn test_partition_distinct_states() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::new(1.0));
        fst.set_final(s2, TropicalWeight::new(2.0));

        let classes = partition(&fst).unwrap();

        // All states have different properties
        assert_eq!(classes.len(), 3);
        assert_ne!(classes[s0 as usize], classes[s1 as usize]);
        assert_ne!(classes[s1 as usize], classes[s2 as usize]);
        assert_ne!(classes[s0 as usize], classes[s2 as usize]);
    }

    #[test]
    fn test_partition_by_arc_structure() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());
        fst.set_final(s3, TropicalWeight::one());

        // s1 goes to s2, s0 goes to s3
        // s1 and s0 are both non-final but have arcs to equivalent states
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s2));
        fst.add_arc(s1, Arc::new(1, 1, TropicalWeight::one(), s3));

        let classes = partition(&fst).unwrap();

        // s2 and s3 are equivalent (both final with no arcs)
        assert_eq!(classes[s2 as usize], classes[s3 as usize]);
        // s0 and s1 are equivalent (same arc structure to equivalent states)
        assert_eq!(classes[s0 as usize], classes[s1 as usize]);
    }

    #[test]
    fn test_partition_different_arc_labels() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());
        fst.set_final(s3, TropicalWeight::one());

        // s0 and s1 have different arc labels
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s2));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s3));

        let classes = partition(&fst).unwrap();

        // s2 and s3 are equivalent
        assert_eq!(classes[s2 as usize], classes[s3 as usize]);
        // s0 and s1 are NOT equivalent (different labels)
        assert_ne!(classes[s0 as usize], classes[s1 as usize]);
    }

    #[test]
    fn test_partition_minimal_fst() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));

        let classes = partition(&fst).unwrap();

        // Already minimal
        assert_eq!(classes.len(), 2);
        assert_ne!(classes[s0 as usize], classes[s1 as usize]);
    }

    #[test]
    fn test_partition_self_loop() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::one());
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s0));

        let classes = partition(&fst).unwrap();

        // s0 has self-loop, s1 doesn't -> different classes
        assert_ne!(classes[s0 as usize], classes[s1 as usize]);
    }

    #[test]
    fn test_partition_complex_equivalence() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();
        let s4 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());
        fst.set_final(s4, TropicalWeight::one());

        // Build symmetric structure
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s3));
        fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::one(), s2));
        fst.add_arc(s3, Arc::new(3, 3, TropicalWeight::one(), s4));

        let classes = partition(&fst).unwrap();

        // s2 and s4 are equivalent (both final, no arcs)
        assert_eq!(classes[s2 as usize], classes[s4 as usize]);
        // s1 and s3 are equivalent (same arc structure to equivalent states)
        assert_eq!(classes[s1 as usize], classes[s3 as usize]);
    }

    #[test]
    fn test_partition_with_boolean_weight() {
        let mut fst = VectorFst::<BooleanWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, BooleanWeight::one());

        let classes = partition(&fst).unwrap();

        assert_eq!(classes.len(), 2);
    }

    #[test]
    fn test_partition_renumbering() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::new(1.0));
        fst.set_final(s1, TropicalWeight::new(2.0));
        fst.set_final(s2, TropicalWeight::new(3.0));

        let classes = partition(&fst).unwrap();

        // Classes should be numbered 0, 1, 2
        let mut sorted_classes = classes.clone();
        sorted_classes.sort();
        sorted_classes.dedup();
        assert_eq!(sorted_classes, vec![0, 1, 2]);
    }

    #[test]
    fn test_partition_no_final_states() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));

        let classes = partition(&fst).unwrap();

        // Both non-final but different arc structure
        assert_eq!(classes.len(), 2);
    }

    #[test]
    fn test_partition_multiple_arcs_same_dest() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // s0 and s1 both have two arcs to s2
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s2));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s2));
        fst.add_arc(s1, Arc::new(1, 1, TropicalWeight::one(), s2));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));

        let classes = partition(&fst).unwrap();

        // s0 and s1 should be equivalent
        assert_eq!(classes[s0 as usize], classes[s1 as usize]);
    }
}
