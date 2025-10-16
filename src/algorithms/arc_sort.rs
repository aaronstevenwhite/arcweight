//! Arc sorting algorithm for FST normalization
//!
//! ## Overview
//!
//! Sorts arcs within each state according to specified ordering criteria.
//! Arc sorting improves cache locality during traversal, enables binary search
//! for arc matching in composition algorithms, and normalizes FST representation
//! for comparison operations.
//!
//! This operation modifies the FST in-place and does not change the language
//! or weights - it only reorders the arcs within each state.
//!
//! ## Algorithm
//!
//! For each state:
//! 1. Collect all outgoing arcs
//! 2. Sort arcs using specified comparator (by input, output, or both)
//! 3. Replace arcs with sorted order
//!
//! ## Complexity
//!
//! - **Time:** O(|V| + |E| log |E|)
//!   - Iterate over all states: O(|V|)
//!   - For each state with k arcs: O(k log k) sorting
//!   - Total: Σ(k_i log k_i) ≤ O(|E| log |E|) across all arcs
//!
//! - **Space:** O(|E|) - temporary storage for arcs during sorting
//!
//! ## Use Cases
//!
//! - **Cache Optimization:** Improves memory locality during arc traversal
//! - **Composition Efficiency:** Enables binary search for arc matching
//! - **FST Normalization:** Standardizes representation for comparison/testing
//! - **Algorithm Prerequisites:** Some algorithms benefit from sorted arcs
//!
//! ## Examples
//!
//! ### Sort by Input Label
//!
//! ```
//! use arcweight::prelude::*;
//!
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! fst.set_start(s0);
//!
//! // Add arcs in arbitrary order
//! fst.add_arc(s0, Arc::new(3, 1, TropicalWeight::one(), s1));
//! fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::one(), s1));
//! fst.add_arc(s0, Arc::new(2, 3, TropicalWeight::one(), s1));
//!
//! arc_sort(&mut fst, ArcSortType::ByInput)?;
//!
//! // Arcs now sorted by input label: 1, 2, 3
//! let labels: Vec<_> = fst.arcs(s0).map(|a| a.ilabel).collect();
//! assert_eq!(labels, vec![1, 2, 3]);
//! # Ok::<(), arcweight::Error>(())
//! ```
//!
//! ### Sort by Output Label
//!
//! ```
//! use arcweight::prelude::*;
//!
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! fst.set_start(s0);
//!
//! fst.add_arc(s0, Arc::new(1, 3, TropicalWeight::one(), s1));
//! fst.add_arc(s0, Arc::new(2, 1, TropicalWeight::one(), s1));
//! fst.add_arc(s0, Arc::new(3, 2, TropicalWeight::one(), s1));
//!
//! arc_sort(&mut fst, ArcSortType::ByOutput)?;
//!
//! // Arcs now sorted by output label: 1, 2, 3
//! let labels: Vec<_> = fst.arcs(s0).map(|a| a.olabel).collect();
//! assert_eq!(labels, vec![1, 2, 3]);
//! # Ok::<(), arcweight::Error>(())
//! ```
//!
//! ## References
//!
//! - Mohri, M., Pereira, F., and Riley, M. (2008). "Speech Recognition with Weighted
//!   Finite-State Transducers." Springer Handbook of Speech Processing, pp. 559-584.
//! - Allauzen, C., Riley, M., Schalkwyk, J., Skut, W., and Mohri, M. (2007).
//!   "OpenFst: A General and Efficient Weighted Finite-State Transducer Library."
//!   Implementation and Application of Automata, LNCS 4783, pp. 11-23.

use crate::arc::Arc;
use crate::fst::{MutableFst, StateId};
use crate::semiring::Semiring;
use crate::Result;

/// Ordering criteria for arc sorting
///
/// Specifies how arcs should be sorted within each state. The ordering
/// determines which arc fields are used as primary and secondary sort keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArcSortType {
    /// Sort by input label only
    ///
    /// Arcs are ordered by their input label (ilabel) in ascending order.
    /// This is useful for composition where the left FST is typically
    /// scanned by input label.
    ///
    /// # Example
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let mut fst = VectorFst::<TropicalWeight>::new();
    /// let s0 = fst.add_state();
    /// let s1 = fst.add_state();
    /// fst.set_start(s0);
    ///
    /// fst.add_arc(s0, Arc::new(3, 10, TropicalWeight::one(), s1));
    /// fst.add_arc(s0, Arc::new(1, 20, TropicalWeight::one(), s1));
    /// fst.add_arc(s0, Arc::new(2, 30, TropicalWeight::one(), s1));
    ///
    /// arc_sort(&mut fst, ArcSortType::ByInput)?;
    ///
    /// let labels: Vec<_> = fst.arcs(s0).map(|a| a.ilabel).collect();
    /// assert_eq!(labels, vec![1, 2, 3]);
    /// # Ok::<(), arcweight::Error>(())
    /// ```
    ByInput,

    /// Sort by output label only
    ///
    /// Arcs are ordered by their output label (olabel) in ascending order.
    /// This is useful when scanning arcs by output label, such as in the
    /// right FST during composition.
    ///
    /// # Example
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let mut fst = VectorFst::<TropicalWeight>::new();
    /// let s0 = fst.add_state();
    /// let s1 = fst.add_state();
    /// fst.set_start(s0);
    ///
    /// fst.add_arc(s0, Arc::new(10, 3, TropicalWeight::one(), s1));
    /// fst.add_arc(s0, Arc::new(20, 1, TropicalWeight::one(), s1));
    /// fst.add_arc(s0, Arc::new(30, 2, TropicalWeight::one(), s1));
    ///
    /// arc_sort(&mut fst, ArcSortType::ByOutput)?;
    ///
    /// let labels: Vec<_> = fst.arcs(s0).map(|a| a.olabel).collect();
    /// assert_eq!(labels, vec![1, 2, 3]);
    /// # Ok::<(), arcweight::Error>(())
    /// ```
    ByOutput,

    /// Sort by input label first, then by output label
    ///
    /// Arcs are ordered first by input label, with ties broken by output
    /// label. This provides a total ordering for arcs and is useful for
    /// FST normalization and comparison.
    ///
    /// # Example
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let mut fst = VectorFst::<TropicalWeight>::new();
    /// let s0 = fst.add_state();
    /// let s1 = fst.add_state();
    /// fst.set_start(s0);
    ///
    /// fst.add_arc(s0, Arc::new(1, 3, TropicalWeight::one(), s1));
    /// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
    /// fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s1));
    ///
    /// arc_sort(&mut fst, ArcSortType::ByInputOutput)?;
    ///
    /// let pairs: Vec<_> = fst.arcs(s0).map(|a| (a.ilabel, a.olabel)).collect();
    /// assert_eq!(pairs, vec![(1, 1), (1, 3), (2, 2)]);
    /// # Ok::<(), arcweight::Error>(())
    /// ```
    ByInputOutput,

    /// Sort by output label first, then by input label
    ///
    /// Arcs are ordered first by output label, with ties broken by input
    /// label. This is useful when the primary access pattern is by output
    /// label but a secondary ordering by input is desired.
    ///
    /// # Example
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let mut fst = VectorFst::<TropicalWeight>::new();
    /// let s0 = fst.add_state();
    /// let s1 = fst.add_state();
    /// fst.set_start(s0);
    ///
    /// fst.add_arc(s0, Arc::new(3, 1, TropicalWeight::one(), s1));
    /// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
    /// fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s1));
    ///
    /// arc_sort(&mut fst, ArcSortType::ByOutputInput)?;
    ///
    /// let pairs: Vec<_> = fst.arcs(s0).map(|a| (a.ilabel, a.olabel)).collect();
    /// assert_eq!(pairs, vec![(1, 1), (3, 1), (2, 2)]);
    /// # Ok::<(), arcweight::Error>(())
    /// ```
    ByOutputInput,
}

/// Sorts arcs within each state according to the specified ordering.
///
/// Arc sorting can improve cache locality and enable binary search
/// for arc matching in composition algorithms. This operation modifies
/// the FST in-place and does not change the language or weights.
///
/// # Algorithm
///
/// For each state:
/// 1. Collect all outgoing arcs into a temporary vector
/// 2. Sort arcs using the specified comparator
/// 3. Delete all arcs from the state
/// 4. Add back arcs in sorted order
///
/// # Complexity
///
/// - **Time:** O(|V| + |E| log |E|) where V = states, E = arcs
///   - Iterate over all states: O(|V|)
///   - For each state with k arcs: O(k log k) sorting
///   - Total: Σᵢ kᵢ log kᵢ ≤ O(|E| log |E|) across all states
/// - **Space:** O(max_degree) where max_degree = maximum arcs from any state
///   - Worst case: O(|E|) if single state has all arcs
///
/// # Algorithm
///
/// In-place arc reordering via standard sorting:
/// 1. Iterate through all states in the FST
/// 2. For each state:
///    - Collect all outgoing arcs into temporary vector
///    - Sort using stable sort with specified comparator (ByInput, ByOutput, etc.)
///    - Delete all arcs from state
///    - Re-add arcs in sorted order
/// 3. Result: arcs within each state are ordered, FST structure preserved
///
/// Uses Rust's stable sort (TimSort) which has O(n log n) worst case,
/// O(n) best case on already-sorted data.
///
/// # Performance Notes
///
/// - **Already sorted arcs:** O(|V| + |E|) with stable sort optimization
/// - **Cache benefits:** Sorted arcs improve sequential access patterns
/// - **Composition:** Pre-sorting by ilabel (left FST) or olabel (right FST) enables binary search
/// - **Binary search:** O(log k) arc lookup vs O(k) linear scan per state
/// - **Best practice:** Sort once after construction, before composition operations
///
/// # Examples
///
/// ## Sort by Input Label
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// fst.set_start(s0);
///
/// // Add arcs in arbitrary order
/// fst.add_arc(s0, Arc::new(3, 1, TropicalWeight::one(), s1));
/// fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::one(), s1));
/// fst.add_arc(s0, Arc::new(2, 3, TropicalWeight::one(), s1));
///
/// arc_sort(&mut fst, ArcSortType::ByInput)?;
///
/// // Arcs now sorted by input label: 1, 2, 3
/// let labels: Vec<_> = fst.arcs(s0).map(|a| a.ilabel).collect();
/// assert_eq!(labels, vec![1, 2, 3]);
/// # Ok::<(), arcweight::Error>(())
/// ```
///
/// ## Sort by Output Label
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// fst.set_start(s0);
///
/// fst.add_arc(s0, Arc::new(1, 3, TropicalWeight::one(), s1));
/// fst.add_arc(s0, Arc::new(2, 1, TropicalWeight::one(), s1));
/// fst.add_arc(s0, Arc::new(3, 2, TropicalWeight::one(), s1));
///
/// arc_sort(&mut fst, ArcSortType::ByOutput)?;
///
/// // Arcs now sorted by output label: 1, 2, 3
/// let labels: Vec<_> = fst.arcs(s0).map(|a| a.olabel).collect();
/// assert_eq!(labels, vec![1, 2, 3]);
/// # Ok::<(), arcweight::Error>(())
/// ```
///
/// ## Sort by Input then Output
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// fst.set_start(s0);
///
/// fst.add_arc(s0, Arc::new(1, 3, TropicalWeight::one(), s1));
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
/// fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s1));
///
/// arc_sort(&mut fst, ArcSortType::ByInputOutput)?;
///
/// // Arcs sorted by input first, then output: (1,1), (1,3), (2,2)
/// let pairs: Vec<_> = fst.arcs(s0).map(|a| (a.ilabel, a.olabel)).collect();
/// assert_eq!(pairs, vec![(1, 1), (1, 3), (2, 2)]);
/// # Ok::<(), arcweight::Error>(())
/// ```
///
/// ## Multiple States
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
/// // Add arcs from s0
/// fst.add_arc(s0, Arc::new(3, 1, TropicalWeight::one(), s1));
/// fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::one(), s1));
///
/// // Add arcs from s1
/// fst.add_arc(s1, Arc::new(5, 3, TropicalWeight::one(), s2));
/// fst.add_arc(s1, Arc::new(4, 4, TropicalWeight::one(), s2));
///
/// arc_sort(&mut fst, ArcSortType::ByInput)?;
///
/// // Both states sorted independently
/// let s0_labels: Vec<_> = fst.arcs(s0).map(|a| a.ilabel).collect();
/// let s1_labels: Vec<_> = fst.arcs(s1).map(|a| a.ilabel).collect();
/// assert_eq!(s0_labels, vec![1, 3]);
/// assert_eq!(s1_labels, vec![4, 5]);
/// # Ok::<(), arcweight::Error>(())
/// ```
///
/// # See Also
///
/// - [`ArcSortType`] - Enumeration of available sorting strategies
/// - [`compose`] - Benefits from pre-sorted arcs for efficient label matching
///
/// [`compose`]: crate::algorithms::compose::compose
pub fn arc_sort<W, F>(fst: &mut F, sort_type: ArcSortType) -> Result<()>
where
    W: Semiring + Clone,
    F: MutableFst<W>,
{
    let num_states = fst.num_states();

    // Iterate over all states: O(|V|)
    for state in 0..num_states as StateId {
        // Collect all arcs from this state: O(k) where k = arcs from state
        let mut arcs: Vec<Arc<W>> = fst.arcs(state).collect();

        // Skip states with no arcs
        if arcs.is_empty() {
            continue;
        }

        // Sort arcs according to specified criteria: O(k log k)
        // Uses stable sort to maintain relative order for equal keys
        match sort_type {
            ArcSortType::ByInput => {
                arcs.sort_by_key(|arc| arc.ilabel);
            }
            ArcSortType::ByOutput => {
                arcs.sort_by_key(|arc| arc.olabel);
            }
            ArcSortType::ByInputOutput => {
                arcs.sort_by_key(|arc| (arc.ilabel, arc.olabel));
            }
            ArcSortType::ByOutputInput => {
                arcs.sort_by_key(|arc| (arc.olabel, arc.ilabel));
            }
        }

        // Replace arcs with sorted order: O(k) for delete + O(k) for adds
        fst.delete_arcs(state);
        for arc in arcs {
            fst.add_arc(state, arc);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    // Basic sorting tests

    #[test]
    fn test_sort_by_input() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        // Add arcs in reverse order
        fst.add_arc(s0, Arc::new(3, 10, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(1, 20, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(2, 30, TropicalWeight::one(), s1));

        arc_sort(&mut fst, ArcSortType::ByInput).unwrap();

        let labels: Vec<_> = fst.arcs(s0).map(|a| a.ilabel).collect();
        assert_eq!(labels, vec![1, 2, 3]);
    }

    #[test]
    fn test_sort_by_output() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        fst.add_arc(s0, Arc::new(10, 3, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(20, 1, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(30, 2, TropicalWeight::one(), s1));

        arc_sort(&mut fst, ArcSortType::ByOutput).unwrap();

        let labels: Vec<_> = fst.arcs(s0).map(|a| a.olabel).collect();
        assert_eq!(labels, vec![1, 2, 3]);
    }

    #[test]
    fn test_sort_by_input_output() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        // Same input, different outputs
        fst.add_arc(s0, Arc::new(1, 3, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s1));

        arc_sort(&mut fst, ArcSortType::ByInputOutput).unwrap();

        let pairs: Vec<_> = fst.arcs(s0).map(|a| (a.ilabel, a.olabel)).collect();
        assert_eq!(pairs, vec![(1, 1), (1, 3), (2, 2)]);
    }

    #[test]
    fn test_sort_by_output_input() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        // Same output, different inputs
        fst.add_arc(s0, Arc::new(3, 1, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s1));

        arc_sort(&mut fst, ArcSortType::ByOutputInput).unwrap();

        let pairs: Vec<_> = fst.arcs(s0).map(|a| (a.ilabel, a.olabel)).collect();
        assert_eq!(pairs, vec![(1, 1), (3, 1), (2, 2)]);
    }

    // Edge cases

    #[test]
    fn test_empty_fst() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        arc_sort(&mut fst, ArcSortType::ByInput).unwrap();
        assert_eq!(fst.num_states(), 0);
    }

    #[test]
    fn test_single_state_no_arcs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);

        arc_sort(&mut fst, ArcSortType::ByInput).unwrap();
        assert_eq!(fst.num_arcs(s0), 0);
    }

    #[test]
    fn test_single_arc() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::one(), s1));

        arc_sort(&mut fst, ArcSortType::ByInput).unwrap();

        assert_eq!(fst.num_arcs(s0), 1);
        let arc = fst.arcs(s0).next().unwrap();
        assert_eq!(arc.ilabel, 1);
        assert_eq!(arc.olabel, 2);
    }

    #[test]
    fn test_already_sorted() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::one(), s1));

        arc_sort(&mut fst, ArcSortType::ByInput).unwrap();

        let labels: Vec<_> = fst.arcs(s0).map(|a| a.ilabel).collect();
        assert_eq!(labels, vec![1, 2, 3]);
    }

    #[test]
    fn test_reverse_sorted() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        fst.add_arc(s0, Arc::new(3, 1, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(1, 3, TropicalWeight::one(), s1));

        arc_sort(&mut fst, ArcSortType::ByInput).unwrap();

        let labels: Vec<_> = fst.arcs(s0).map(|a| a.ilabel).collect();
        assert_eq!(labels, vec![1, 2, 3]);
    }

    // Multiple states

    #[test]
    fn test_multiple_states_with_arcs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);

        fst.add_arc(s0, Arc::new(3, 1, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::one(), s1));

        fst.add_arc(s1, Arc::new(5, 3, TropicalWeight::one(), s2));
        fst.add_arc(s1, Arc::new(4, 4, TropicalWeight::one(), s2));

        arc_sort(&mut fst, ArcSortType::ByInput).unwrap();

        let s0_labels: Vec<_> = fst.arcs(s0).map(|a| a.ilabel).collect();
        let s1_labels: Vec<_> = fst.arcs(s1).map(|a| a.ilabel).collect();

        assert_eq!(s0_labels, vec![1, 3]);
        assert_eq!(s1_labels, vec![4, 5]);
    }

    #[test]
    fn test_different_out_degrees() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);

        // s0 has 5 arcs
        for i in (1..=5).rev() {
            fst.add_arc(s0, Arc::new(i, i, TropicalWeight::one(), s1));
        }

        // s1 has 2 arcs
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));
        fst.add_arc(s1, Arc::new(1, 1, TropicalWeight::one(), s2));

        arc_sort(&mut fst, ArcSortType::ByInput).unwrap();

        let s0_labels: Vec<_> = fst.arcs(s0).map(|a| a.ilabel).collect();
        let s1_labels: Vec<_> = fst.arcs(s1).map(|a| a.ilabel).collect();

        assert_eq!(s0_labels, vec![1, 2, 3, 4, 5]);
        assert_eq!(s1_labels, vec![1, 2]);
    }

    // Stability tests

    #[test]
    fn test_stable_sort() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);

        // Same input label, different nextstates - should maintain relative order
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::one(), s2));

        arc_sort(&mut fst, ArcSortType::ByInput).unwrap();

        let arcs: Vec<_> = fst.arcs(s0).collect();
        assert_eq!(arcs[0].olabel, 1);
        assert_eq!(arcs[0].nextstate, s1);
        assert_eq!(arcs[1].olabel, 2);
        assert_eq!(arcs[1].nextstate, s2);
    }

    #[test]
    fn test_duplicate_labels() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        // Multiple arcs with same labels
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(3.0), s1));

        arc_sort(&mut fst, ArcSortType::ByInput).unwrap();

        // All arcs should still exist
        assert_eq!(fst.num_arcs(s0), 3);
        let labels: Vec<_> = fst.arcs(s0).map(|a| a.ilabel).collect();
        assert_eq!(labels, vec![1, 1, 1]);
    }

    // Different semirings

    #[test]
    fn test_tropical_semiring() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        fst.add_arc(s0, Arc::new(3, 1, TropicalWeight::new(5.0), s1));
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(3.0), s1));
        fst.add_arc(s0, Arc::new(2, 3, TropicalWeight::new(4.0), s1));

        arc_sort(&mut fst, ArcSortType::ByInput).unwrap();

        let labels: Vec<_> = fst.arcs(s0).map(|a| a.ilabel).collect();
        assert_eq!(labels, vec![1, 2, 3]);
    }

    #[test]
    fn test_log_semiring() {
        let mut fst = VectorFst::<LogWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        fst.add_arc(s0, Arc::new(3, 1, LogWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(1, 2, LogWeight::new(2.0), s1));
        fst.add_arc(s0, Arc::new(2, 3, LogWeight::new(3.0), s1));

        arc_sort(&mut fst, ArcSortType::ByOutput).unwrap();

        let labels: Vec<_> = fst.arcs(s0).map(|a| a.olabel).collect();
        assert_eq!(labels, vec![1, 2, 3]);
    }

    #[test]
    fn test_boolean_semiring() {
        let mut fst = VectorFst::<BooleanWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        fst.add_arc(s0, Arc::new(3, 1, BooleanWeight::one(), s1));
        fst.add_arc(s0, Arc::new(1, 2, BooleanWeight::one(), s1));

        arc_sort(&mut fst, ArcSortType::ByInput).unwrap();

        let labels: Vec<_> = fst.arcs(s0).map(|a| a.ilabel).collect();
        assert_eq!(labels, vec![1, 3]);
    }

    #[test]
    fn test_probability_semiring() {
        let mut fst = VectorFst::<ProbabilityWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        fst.add_arc(s0, Arc::new(2, 1, ProbabilityWeight::new(0.5), s1));
        fst.add_arc(s0, Arc::new(1, 2, ProbabilityWeight::new(0.3), s1));

        arc_sort(&mut fst, ArcSortType::ByInputOutput).unwrap();

        let pairs: Vec<_> = fst.arcs(s0).map(|a| (a.ilabel, a.olabel)).collect();
        assert_eq!(pairs, vec![(1, 2), (2, 1)]);
    }

    // Verification tests

    #[test]
    fn test_language_preserved() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(3.0), s1));
        fst.add_arc(s1, Arc::new(4, 4, TropicalWeight::new(4.0), s2));

        // Count arcs before sorting
        let arcs_before_count = fst.num_arcs(s0);

        arc_sort(&mut fst, ArcSortType::ByInput).unwrap();

        // Check same arcs exist after sorting (just reordered)
        let arcs_after_count = fst.num_arcs(s0);
        assert_eq!(arcs_after_count, arcs_before_count);

        // Verify all original paths still exist and are sorted
        let labels_after: Vec<_> = fst.arcs(s0).map(|a| a.ilabel).collect();
        assert_eq!(labels_after, vec![1, 2, 3]); // Sorted by input
    }

    #[test]
    fn test_weights_preserved() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        fst.add_arc(s0, Arc::new(3, 1, TropicalWeight::new(5.0), s1));
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(3.0), s1));
        fst.add_arc(s0, Arc::new(2, 3, TropicalWeight::new(4.0), s1));

        arc_sort(&mut fst, ArcSortType::ByInput).unwrap();

        let arcs: Vec<_> = fst.arcs(s0).collect();
        // After sorting by input: label 1 has weight 3.0, label 2 has 4.0, label 3 has 5.0
        assert_eq!(arcs[0].ilabel, 1);
        assert_eq!(arcs[0].weight, TropicalWeight::new(3.0));
        assert_eq!(arcs[1].ilabel, 2);
        assert_eq!(arcs[1].weight, TropicalWeight::new(4.0));
        assert_eq!(arcs[2].ilabel, 3);
        assert_eq!(arcs[2].weight, TropicalWeight::new(5.0));
    }

    // Large FST tests

    #[test]
    fn test_many_arcs_per_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        // Add 100 arcs in reverse order
        for i in (1..=100).rev() {
            fst.add_arc(s0, Arc::new(i, i, TropicalWeight::one(), s1));
        }

        arc_sort(&mut fst, ArcSortType::ByInput).unwrap();

        let labels: Vec<_> = fst.arcs(s0).map(|a| a.ilabel).collect();
        let expected: Vec<_> = (1..=100).collect();
        assert_eq!(labels, expected);
    }
}
