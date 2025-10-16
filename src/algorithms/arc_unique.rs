//! Arc unique algorithm for removing exact duplicate arcs
//!
//! ## Overview
//!
//! Removes exact duplicate arcs, keeping only the first occurrence. An arc is
//! considered a duplicate if it has identical (ilabel, olabel, weight, nextstate)
//! to another arc from the same source state.
//!
//! This is simpler than arc_sum - it doesn't combine weights, just removes
//! exact duplicates.
//!
//! ## Algorithm
//!
//! For each state:
//! 1. Use HashSet to track seen (ilabel, olabel, weight, nextstate) tuples
//! 2. Keep only first occurrence of each unique arc
//! 3. Discard exact duplicates
//!
//! ## Complexity
//!
//! - **Time:** O(|V| + |E| log |E|)
//!   - Iterate over all states: O(|V|)
//!   - For each state with k arcs: O(k log k) for hashing/deduplication
//!   - Total arcs processed: O(|E| log |E|)
//!
//! - **Space:** O(|E|) - HashSet for duplicate detection
//!
//! ## Use Cases
//!
//! - Removing accidentally duplicated arcs
//! - Cleaning FST representations
//! - Normalizing FST structure before serialization
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
//! // Add exact duplicate arcs
//! let arc = Arc::new(1, 2, TropicalWeight::new(1.0), s1);
//! fst.add_arc(s0, arc.clone());
//! fst.add_arc(s0, arc.clone());
//!
//! arc_unique(&mut fst)?;
//!
//! // Now only one arc remains
//! assert_eq!(fst.num_arcs(s0), 1);
//! # Ok::<(), arcweight::Error>(())
//! ```
//!
//! ## References
//!
//! - Mohri, M. (2009). "Weighted Automata Algorithms." Handbook of Weighted
//!   Automata, Springer, pp. 213-254.
//! - Allauzen, C., Riley, M., Schalkwyk, J., Skut, W., and Mohri, M. (2007).
//!   "OpenFst: A General and Efficient Weighted Finite-State Transducer Library."
//!   Implementation and Application of Automata, LNCS 4783, pp. 11-23.

use crate::arc::Arc;
use crate::fst::{MutableFst, StateId};
use crate::semiring::Semiring;
use crate::Result;
use std::collections::HashSet;

/// Removes exact duplicate arcs, keeping only the first occurrence.
///
/// An arc is considered a duplicate if it has identical
/// (ilabel, olabel, weight, nextstate) to another arc from the same
/// source state. This is simpler than arc_sum - it doesn't combine
/// weights, just removes exact duplicates.
///
/// # Algorithm
///
/// For each state:
/// 1. Use HashSet to track seen (ilabel, olabel, weight, nextstate) tuples
/// 2. Keep only first occurrence of each unique arc
/// 3. Discard exact duplicates
///
/// # Complexity
///
/// - **Time:** O(|V| + |E| log |E|)
///   - Iterate states: O(|V|)
///   - Hash/deduplicate arcs per state: O(k log k) where k = arcs from state
///   - Total: O(|E| log |E|) across all arcs
///
/// - **Space:** O(|E|) - HashSet for duplicate detection
///
/// # Examples
///
/// ## Exact Duplicates
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// fst.set_start(s0);
///
/// // Add exact duplicate arcs
/// let arc = Arc::new(1, 2, TropicalWeight::new(1.0), s1);
/// fst.add_arc(s0, arc.clone());
/// fst.add_arc(s0, arc.clone());
/// fst.add_arc(s0, arc.clone());
///
/// arc_unique(&mut fst)?;
///
/// // Only one arc remains
/// assert_eq!(fst.num_arcs(s0), 1);
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
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
/// fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s2));
///
/// arc_unique(&mut fst)?;
///
/// // No change
/// assert_eq!(fst.num_arcs(s0), 2);
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
/// let arc1 = Arc::new(1, 1, TropicalWeight::new(1.0), s1);
/// fst.add_arc(s0, arc1.clone());
/// fst.add_arc(s0, arc1.clone()); // Duplicate
/// fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s2)); // Unique
///
/// arc_unique(&mut fst)?;
///
/// // One duplicate removed
/// assert_eq!(fst.num_arcs(s0), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Same Labels Different Weights (Not Duplicates)
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// fst.set_start(s0);
///
/// // Same labels but different weights - NOT duplicates
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));
///
/// arc_unique(&mut fst)?;
///
/// // Both kept (different weights)
/// assert_eq!(fst.num_arcs(s0), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Empty FST
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// arc_unique(&mut fst)?;
/// assert_eq!(fst.num_states(), 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Performance Notes
///
/// - **No duplicates:** O(|E|) with minimal overhead for unique arcs
/// - **Many duplicates:** Reduces arc count and improves FST performance
/// - **Hash operations:** O(1) average case for duplicate detection
/// - **Memory:** HashSet storage proportional to unique arcs per state
/// - **Best practice:** Apply after operations that may accidentally duplicate arcs
///
/// # See Also
///
/// - [`arc_sum`] - Combines duplicate arcs by summing weights
/// - [`arc_sort`] - Sorts arcs for easier duplicate detection
///
/// [`arc_sum`]: crate::algorithms::arc_sum::arc_sum
/// [`arc_sort`]: crate::algorithms::arc_sort::arc_sort
pub fn arc_unique<W, F>(fst: &mut F) -> Result<()>
where
    W: Semiring + Clone + Eq + std::hash::Hash,
    F: MutableFst<W>,
{
    let num_states = fst.num_states();

    for state in 0..num_states as StateId {
        // Collect all arcs from this state
        let arcs: Vec<Arc<W>> = fst.arcs(state).collect();

        if arcs.is_empty() {
            continue;
        }

        // Use HashSet to track unique arcs
        let mut seen = HashSet::new();
        let mut unique_arcs = Vec::new();

        for arc in arcs {
            // Create a hashable key from the arc
            let key = (
                arc.ilabel,
                arc.olabel,
                arc.weight.clone(),
                arc.nextstate,
            );

            if seen.insert(key) {
                // First occurrence - keep it
                unique_arcs.push(arc);
            }
            // Otherwise it's a duplicate - skip it
        }

        // Clear existing arcs and add back only unique ones
        fst.delete_arcs(state);
        for arc in unique_arcs {
            fst.add_arc(state, arc);
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

        let arc = Arc::new(1, 2, TropicalWeight::new(1.0), s1);
        fst.add_arc(s0, arc.clone());
        fst.add_arc(s0, arc.clone());

        arc_unique(&mut fst).unwrap();

        assert_eq!(fst.num_arcs(s0), 1);
        let result_arc = fst.arcs(s0).next().unwrap();
        assert_eq!(result_arc.ilabel, 1);
        assert_eq!(result_arc.olabel, 2);
        assert_eq!(result_arc.weight, TropicalWeight::new(1.0));
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

        arc_unique(&mut fst).unwrap();

        assert_eq!(fst.num_arcs(s0), 2);
    }

    #[test]
    fn test_partial_duplicates() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);

        let arc = Arc::new(1, 1, TropicalWeight::new(1.0), s1);
        fst.add_arc(s0, arc.clone());
        fst.add_arc(s0, arc.clone()); // Dup
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s2)); // Unique

        arc_unique(&mut fst).unwrap();

        assert_eq!(fst.num_arcs(s0), 2);
    }

    #[test]
    fn test_same_labels_different_weights() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        // Same labels but different weights - NOT duplicates
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));

        arc_unique(&mut fst).unwrap();

        assert_eq!(fst.num_arcs(s0), 2); // Both kept
    }

    #[test]
    fn test_empty_fst() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        arc_unique(&mut fst).unwrap();
        assert_eq!(fst.num_states(), 0);
    }

    #[test]
    fn test_multiple_exact_copies() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        let arc = Arc::new(1, 1, TropicalWeight::new(1.0), s1);
        fst.add_arc(s0, arc.clone());
        fst.add_arc(s0, arc.clone());
        fst.add_arc(s0, arc.clone());
        fst.add_arc(s0, arc.clone());

        arc_unique(&mut fst).unwrap();

        assert_eq!(fst.num_arcs(s0), 1);
    }

    #[test]
    fn test_different_semirings_tropical() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        let arc = Arc::new(1, 1, TropicalWeight::new(3.0), s1);
        fst.add_arc(s0, arc.clone());
        fst.add_arc(s0, arc.clone());

        arc_unique(&mut fst).unwrap();

        assert_eq!(fst.num_arcs(s0), 1);
    }

    #[test]
    fn test_different_semirings_log() {
        let mut fst = VectorFst::<LogWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        let arc = Arc::new(1, 1, LogWeight::new(2.0), s1);
        fst.add_arc(s0, arc.clone());
        fst.add_arc(s0, arc.clone());

        arc_unique(&mut fst).unwrap();

        assert_eq!(fst.num_arcs(s0), 1);
    }

    #[test]
    fn test_different_semirings_boolean() {
        let mut fst = VectorFst::<BooleanWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        let arc = Arc::new(1, 1, BooleanWeight::one(), s1);
        fst.add_arc(s0, arc.clone());
        fst.add_arc(s0, arc.clone());

        arc_unique(&mut fst).unwrap();

        assert_eq!(fst.num_arcs(s0), 1);
    }

    #[test]
    fn test_multiple_states() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);

        let arc1 = Arc::new(1, 1, TropicalWeight::new(1.0), s1);
        fst.add_arc(s0, arc1.clone());
        fst.add_arc(s0, arc1.clone());

        let arc2 = Arc::new(2, 2, TropicalWeight::new(2.0), s2);
        fst.add_arc(s1, arc2.clone());
        fst.add_arc(s1, arc2.clone());

        arc_unique(&mut fst).unwrap();

        assert_eq!(fst.num_arcs(s0), 1);
        assert_eq!(fst.num_arcs(s1), 1);
    }
}
