//! State sorting for FST normalization and optimization
//!
//! Renumbers states according to specified traversal order while preserving
//! FST structure, labels, and weights.
//!
//! ## Overview
//!
//! State sorting creates a new FST with states renumbered according to a traversal
//! order (breadth-first, depth-first, or topological) while preserving all structural
//! properties. This improves cache locality during traversal and enables certain
//! optimizations.
//!
//! ## Algorithm
//!
//! 1. Traverse FST using specified order (BFS/DFS/topological)
//! 2. Build state renumbering map: old_id → new_id
//! 3. Create new FST with renumbered states
//! 4. Copy arcs with updated state IDs
//!
//! ## Complexity
//!
//! - **Time:** O(|V| + |E|) - single traversal plus copying
//! - **Space:** O(|V|) - state mapping and queue/stack
//!
//! ## Use Cases
//!
//! - **Cache Optimization:** Improve memory access patterns during FST traversal
//! - **Canonical Ordering:** Standardize state ordering for FST comparison
//! - **Algorithm Preprocessing:** Optimize state order for specific algorithms
//! - **Debugging:** Simplify FST visualization with logical state ordering
//!
//! ## Examples
//!
//! ### Breadth-First Ordering
//!
//! ```
//! use arcweight::prelude::*;
//!
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! let s2 = fst.add_state();
//! fst.set_start(s0);
//! fst.set_final(s2, TropicalWeight::one());
//! fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
//! fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(0.3), s2));
//! fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::new(0.2), s2));
//!
//! let sorted = state_sort(&fst, StateSortType::BreadthFirst)?;
//!
//! // Start state is now 0, BFS order for rest
//! assert_eq!(sorted.start(), Some(0));
//! assert_eq!(sorted.num_states(), fst.num_states());
//! # Ok::<(), arcweight::Error>(())
//! ```
//!
//! ### Topological Ordering (Acyclic FST)
//!
//! ```
//! use arcweight::prelude::*;
//!
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! let s2 = fst.add_state();
//! fst.set_start(s0);
//! fst.set_final(s2, TropicalWeight::one());
//! fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
//! fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(0.3), s2));
//!
//! let sorted = state_sort(&fst, StateSortType::Topological)?;
//!
//! // States ordered topologically
//! assert_eq!(sorted.start(), Some(0));
//! # Ok::<(), arcweight::Error>(())
//! ```
//!
//! ## References
//!
//! - Moore, E. F. (1959). "The shortest path through a maze." Proceedings of
//!   the International Symposium on the Theory of Switching.
//! - Kahn, A. B. (1962). "Topological sorting of large networks." Communications
//!   of the ACM, 5(11): 558-562.
//! - Tarjan, R. E. (1972). "Depth-first search and linear graph algorithms."
//!   SIAM Journal on Computing, 1(2): 146-160.

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst, StateId, VectorFst};
use crate::semiring::Semiring;
use crate::{Error, Result};
use std::collections::{HashMap, HashSet, VecDeque};

/// State sorting strategies
///
/// Specifies how states should be renumbered in the output FST.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateSortType {
    /// Breadth-first order from start state
    ///
    /// States are visited in BFS order, ensuring parent states have
    /// lower IDs than their children. Useful for cache locality.
    ///
    /// # Example
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let mut fst = VectorFst::<TropicalWeight>::new();
    /// let s0 = fst.add_state();
    /// let s1 = fst.add_state();
    /// let s2 = fst.add_state();
    /// fst.set_start(s0);
    /// fst.set_final(s2, TropicalWeight::one());
    /// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
    /// fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s2));
    ///
    /// let sorted = state_sort(&fst, StateSortType::BreadthFirst)?;
    /// assert_eq!(sorted.start(), Some(0));
    /// # Ok::<(), arcweight::Error>(())
    /// ```
    BreadthFirst,

    /// Depth-first order from start state
    ///
    /// States are visited in DFS order. Useful for certain graph algorithms
    /// and can improve cache performance for linear paths.
    ///
    /// # Example
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let mut fst = VectorFst::<TropicalWeight>::new();
    /// let s0 = fst.add_state();
    /// let s1 = fst.add_state();
    /// let s2 = fst.add_state();
    /// fst.set_start(s0);
    /// fst.set_final(s2, TropicalWeight::one());
    /// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
    /// fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));
    ///
    /// let sorted = state_sort(&fst, StateSortType::DepthFirst)?;
    /// assert_eq!(sorted.start(), Some(0));
    /// # Ok::<(), arcweight::Error>(())
    /// ```
    DepthFirst,

    /// Topological order (requires acyclic FST)
    ///
    /// States are ordered such that for every arc (s, a, w, t), state s
    /// comes before state t. Only valid for acyclic FSTs.
    ///
    /// # Example
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let mut fst = VectorFst::<TropicalWeight>::new();
    /// let s0 = fst.add_state();
    /// let s1 = fst.add_state();
    /// let s2 = fst.add_state();
    /// fst.set_start(s0);
    /// fst.set_final(s2, TropicalWeight::one());
    /// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
    /// fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));
    ///
    /// let sorted = state_sort(&fst, StateSortType::Topological)?;
    /// assert_eq!(sorted.start(), Some(0));
    /// # Ok::<(), arcweight::Error>(())
    /// ```
    Topological,
}

/// Sorts FST states by renumbering them according to specified order.
///
/// Creates a new FST with states renumbered to follow the traversal order
/// while preserving all arcs, labels, weights, and finality. The start state
/// always becomes state 0.
///
/// # Complexity
///
/// - **Time:** O(|V| + |E|) where V = states, E = arcs
///   - Traversal (BFS/DFS/Topological): O(|V| + |E|)
///   - State renumbering and copying: O(|V| + |E|)
/// - **Space:** O(|V|) for state mapping and traversal queue/stack
///
/// # Algorithm
///
/// Generic state renumbering with traversal strategies:
/// 1. **Breadth-First:** Use queue for level-order traversal from start state
/// 2. **Depth-First:** Use recursive DFS to visit states depth-first
/// 3. **Topological:** Use DFS with cycle detection for DAG ordering
/// 4. Build renumbering map: old_id → new_id based on visit order
/// 5. Create output FST with states and arcs in new order
///
/// Based on Moore (1959) "The shortest path through a maze" for BFS,
/// Tarjan (1972) "Depth-first search and linear graph algorithms" for DFS,
/// and Kahn (1962) "Topological sorting of large networks" for topological ordering.
///
/// # Performance Notes
///
/// - **BFS:** Best for cache locality; parent states precede children
/// - **DFS:** Efficient for deep paths; good for linear chains
/// - **Topological:** Enables forward-arc-only algorithms on DAGs
/// - **Memory:** BFS uses queue (O(width)), DFS uses stack (O(depth))
/// - **Preprocessing:** Consider [`connect`] first to remove unreachable states
///
/// [`connect`]: crate::algorithms::connect::connect
///
/// # Errors
///
/// Returns error if:
/// - FST has no start state
/// - Topological sort requested on cyclic FST
///
/// # Examples
///
/// ## Breadth-First Ordering
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
/// fst.set_start(s0);
/// fst.set_final(s2, TropicalWeight::one());
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
/// fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(0.3), s2));
/// fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::new(0.2), s2));
///
/// let sorted = state_sort(&fst, StateSortType::BreadthFirst)?;
///
/// // Start state is now 0, BFS order for rest
/// assert_eq!(sorted.start(), Some(0));
/// assert_eq!(sorted.num_states(), fst.num_states());
/// # Ok::<(), arcweight::Error>(())
/// ```
///
/// ## Topological Ordering (Acyclic FST)
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
/// fst.set_start(s0);
/// fst.set_final(s2, TropicalWeight::one());
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
/// fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(0.3), s2));
///
/// let sorted = state_sort(&fst, StateSortType::Topological)?;
///
/// // States ordered topologically
/// assert_eq!(sorted.start(), Some(0));
/// # Ok::<(), arcweight::Error>(())
/// ```
///
/// # See Also
///
/// - [`connect`] - Removes unreachable states before sorting
/// - [`StateSortType`] - Enumeration of available sorting strategies
///
/// [`connect`]: crate::algorithms::connect::connect
pub fn state_sort<W, F>(fst: &F, sort_type: StateSortType) -> Result<VectorFst<W>>
where
    W: Semiring + Clone,
    F: Fst<W>,
{
    // Empty FST case
    if fst.num_states() == 0 {
        return Ok(VectorFst::new());
    }

    // Compute state ordering based on sort type
    let order = match sort_type {
        StateSortType::BreadthFirst => compute_bfs_order(fst)?,
        StateSortType::DepthFirst => compute_dfs_order(fst)?,
        StateSortType::Topological => compute_topological_order(fst)?,
    };

    // Build state renumbering map
    let mut state_map = HashMap::new();
    for (new_id, &old_id) in order.iter().enumerate() {
        state_map.insert(old_id, new_id as StateId);
    }

    // Create result FST with renumbered states
    let mut result = VectorFst::new();

    // Add all states in new order
    for _ in &order {
        result.add_state();
    }

    // Set start state (always becomes 0)
    if let Some(start) = fst.start() {
        if let Some(&new_start) = state_map.get(&start) {
            result.set_start(new_start);
        }
    }

    // Copy states with renumbered IDs
    for &old_state in &order {
        if let Some(&new_state) = state_map.get(&old_state) {
            // Copy final weight
            if let Some(weight) = fst.final_weight(old_state) {
                result.set_final(new_state, weight.clone());
            }

            // Copy arcs with renumbered destinations
            for arc in fst.arcs(old_state) {
                if let Some(&new_nextstate) = state_map.get(&arc.nextstate) {
                    result.add_arc(
                        new_state,
                        Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), new_nextstate),
                    );
                }
            }
        }
    }

    Ok(result)
}

/// Compute breadth-first ordering of states
fn compute_bfs_order<W: Semiring, F: Fst<W>>(fst: &F) -> Result<Vec<StateId>> {
    let start = fst
        .start()
        .ok_or_else(|| Error::Algorithm("FST has no start state".into()))?;

    let mut order = Vec::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    queue.push_back(start);
    visited.insert(start);

    while let Some(state) = queue.pop_front() {
        order.push(state);

        for arc in fst.arcs(state) {
            if !visited.contains(&arc.nextstate) {
                visited.insert(arc.nextstate);
                queue.push_back(arc.nextstate);
            }
        }
    }

    // Add any unreachable states at the end
    for state in fst.states() {
        if !visited.contains(&state) {
            order.push(state);
        }
    }

    Ok(order)
}

/// Compute depth-first ordering of states
fn compute_dfs_order<W: Semiring, F: Fst<W>>(fst: &F) -> Result<Vec<StateId>> {
    let start = fst
        .start()
        .ok_or_else(|| Error::Algorithm("FST has no start state".into()))?;

    let mut order = Vec::new();
    let mut visited = HashSet::new();

    fn dfs<W: Semiring, F: Fst<W>>(
        fst: &F,
        state: StateId,
        visited: &mut HashSet<StateId>,
        order: &mut Vec<StateId>,
    ) {
        visited.insert(state);
        order.push(state);

        for arc in fst.arcs(state) {
            if !visited.contains(&arc.nextstate) {
                dfs(fst, arc.nextstate, visited, order);
            }
        }
    }

    dfs(fst, start, &mut visited, &mut order);

    // Add any unreachable states at the end
    for state in fst.states() {
        if !visited.contains(&state) {
            dfs(fst, state, &mut visited, &mut order);
        }
    }

    Ok(order)
}

/// Compute topological ordering of states (for acyclic FSTs)
fn compute_topological_order<W: Semiring, F: Fst<W>>(fst: &F) -> Result<Vec<StateId>> {
    let mut visited = HashSet::new();
    let mut finished = HashSet::new();
    let mut order = Vec::new();

    fn dfs<W: Semiring, F: Fst<W>>(
        fst: &F,
        state: StateId,
        visited: &mut HashSet<StateId>,
        finished: &mut HashSet<StateId>,
        order: &mut Vec<StateId>,
    ) -> Result<()> {
        visited.insert(state);

        for arc in fst.arcs(state) {
            if !visited.contains(&arc.nextstate) {
                dfs(fst, arc.nextstate, visited, finished, order)?;
            } else if !finished.contains(&arc.nextstate) {
                return Err(Error::Algorithm(
                    "FST has cycles, cannot perform topological sort".into(),
                ));
            }
        }

        finished.insert(state);
        order.push(state);
        Ok(())
    }

    // Start DFS from start state if it exists
    if let Some(start) = fst.start() {
        if !visited.contains(&start) {
            dfs(fst, start, &mut visited, &mut finished, &mut order)?;
        }
    }

    // Visit any remaining unvisited states
    for state in fst.states() {
        if !visited.contains(&state) {
            dfs(fst, state, &mut visited, &mut finished, &mut order)?;
        }
    }

    order.reverse();
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use num_traits::One;

    #[test]
    fn test_state_sort_breadth_first_simple() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(0.3), s2));
        fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::new(0.2), s2));

        let sorted = state_sort(&fst, StateSortType::BreadthFirst).unwrap();

        assert_eq!(sorted.start(), Some(0));
        assert_eq!(sorted.num_states(), fst.num_states());

        // Check that at least one state is final
        let has_final = sorted.states().any(|s| sorted.is_final(s));
        assert!(has_final);
    }

    #[test]
    fn test_state_sort_depth_first_simple() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(0.3), s2));

        let sorted = state_sort(&fst, StateSortType::DepthFirst).unwrap();

        assert_eq!(sorted.start(), Some(0));
        assert_eq!(sorted.num_states(), fst.num_states());
    }

    #[test]
    fn test_state_sort_topological_simple() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(0.3), s2));

        let sorted = state_sort(&fst, StateSortType::Topological).unwrap();

        assert_eq!(sorted.start(), Some(0));
        assert_eq!(sorted.num_states(), fst.num_states());

        // Verify topological order: all arcs go forward
        for state in sorted.states() {
            for arc in sorted.arcs(state) {
                assert!(
                    state < arc.nextstate,
                    "Arc from {} to {} violates topological order",
                    state,
                    arc.nextstate
                );
            }
        }
    }

    #[test]
    fn test_state_sort_preserves_structure() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::new(2.5));

        fst.add_arc(s0, Arc::new(1, 10, TropicalWeight::new(0.5), s1));
        fst.add_arc(s1, Arc::new(2, 20, TropicalWeight::new(0.3), s2));

        let sorted = state_sort(&fst, StateSortType::BreadthFirst).unwrap();

        // Check structure is preserved
        assert_eq!(sorted.num_states(), fst.num_states());

        let orig_arc_count: usize = fst.states().map(|s| fst.num_arcs(s)).sum();
        let sorted_arc_count: usize = sorted.states().map(|s| sorted.num_arcs(s)).sum();
        assert_eq!(orig_arc_count, sorted_arc_count);
    }

    #[test]
    fn test_state_sort_breadth_first_complex() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let states: Vec<_> = (0..6).map(|_| fst.add_state()).collect();

        fst.set_start(states[0]);
        fst.set_final(states[5], TropicalWeight::one());

        // Create complex structure
        fst.add_arc(states[0], Arc::new(1, 1, TropicalWeight::one(), states[1]));
        fst.add_arc(states[0], Arc::new(2, 2, TropicalWeight::one(), states[2]));
        fst.add_arc(states[1], Arc::new(3, 3, TropicalWeight::one(), states[3]));
        fst.add_arc(states[2], Arc::new(4, 4, TropicalWeight::one(), states[4]));
        fst.add_arc(states[3], Arc::new(5, 5, TropicalWeight::one(), states[5]));
        fst.add_arc(states[4], Arc::new(6, 6, TropicalWeight::one(), states[5]));

        let sorted = state_sort(&fst, StateSortType::BreadthFirst).unwrap();

        assert_eq!(sorted.start(), Some(0));
        assert_eq!(sorted.num_states(), 6);
    }

    #[test]
    fn test_state_sort_depth_first_complex() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let states: Vec<_> = (0..5).map(|_| fst.add_state()).collect();

        fst.set_start(states[0]);
        fst.set_final(states[4], TropicalWeight::one());

        // Create branching structure
        fst.add_arc(states[0], Arc::new(1, 1, TropicalWeight::one(), states[1]));
        fst.add_arc(states[1], Arc::new(2, 2, TropicalWeight::one(), states[2]));
        fst.add_arc(states[2], Arc::new(3, 3, TropicalWeight::one(), states[3]));
        fst.add_arc(states[3], Arc::new(4, 4, TropicalWeight::one(), states[4]));
        fst.add_arc(states[0], Arc::new(5, 5, TropicalWeight::one(), states[4]));

        let sorted = state_sort(&fst, StateSortType::DepthFirst).unwrap();

        assert_eq!(sorted.start(), Some(0));
        assert_eq!(sorted.num_states(), 5);
    }

    #[test]
    fn test_state_sort_single_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::new(1.5));

        for sort_type in [
            StateSortType::BreadthFirst,
            StateSortType::DepthFirst,
            StateSortType::Topological,
        ] {
            let sorted = state_sort(&fst, sort_type).unwrap();

            assert_eq!(sorted.num_states(), 1);
            assert_eq!(sorted.start(), Some(0));
            assert!(sorted.is_final(0));
            assert_eq!(sorted.final_weight(0), Some(&TropicalWeight::new(1.5)));
        }
    }

    #[test]
    fn test_state_sort_linear_chain() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let states: Vec<_> = (0..5).map(|_| fst.add_state()).collect();

        fst.set_start(states[0]);
        fst.set_final(states[4], TropicalWeight::one());

        for i in 0..4 {
            fst.add_arc(
                states[i],
                Arc::new(
                    (i + 1) as u32,
                    (i + 1) as u32,
                    TropicalWeight::one(),
                    states[i + 1],
                ),
            );
        }

        for sort_type in [
            StateSortType::BreadthFirst,
            StateSortType::DepthFirst,
            StateSortType::Topological,
        ] {
            let sorted = state_sort(&fst, sort_type).unwrap();

            assert_eq!(sorted.num_states(), 5);
            assert_eq!(sorted.start(), Some(0));

            // For topological sort, verify ordering
            if sort_type == StateSortType::Topological {
                for state in sorted.states() {
                    for arc in sorted.arcs(state) {
                        assert!(state < arc.nextstate);
                    }
                }
            }
        }
    }

    #[test]
    fn test_state_sort_branching_fst() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s3, TropicalWeight::one());

        // Diamond pattern
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s2));
        fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::one(), s3));
        fst.add_arc(s2, Arc::new(4, 4, TropicalWeight::one(), s3));

        for sort_type in [
            StateSortType::BreadthFirst,
            StateSortType::DepthFirst,
            StateSortType::Topological,
        ] {
            let sorted = state_sort(&fst, sort_type).unwrap();

            assert_eq!(sorted.num_states(), 4);
            assert_eq!(sorted.start(), Some(0));
        }
    }

    #[test]
    fn test_state_sort_with_cycles_bfs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Create cycle
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));
        fst.add_arc(s2, Arc::new(3, 3, TropicalWeight::one(), s0));

        let sorted = state_sort(&fst, StateSortType::BreadthFirst).unwrap();

        assert_eq!(sorted.num_states(), 3);
        assert_eq!(sorted.start(), Some(0));
    }

    #[test]
    fn test_state_sort_with_cycles_dfs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Create cycle
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));
        fst.add_arc(s2, Arc::new(3, 3, TropicalWeight::one(), s1));

        let sorted = state_sort(&fst, StateSortType::DepthFirst).unwrap();

        assert_eq!(sorted.num_states(), 3);
        assert_eq!(sorted.start(), Some(0));
    }

    #[test]
    fn test_state_sort_topological_acyclic_only() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Acyclic: 0 -> 1 -> 2, 0 -> 2
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));
        fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::one(), s2));

        let sorted = state_sort(&fst, StateSortType::Topological).unwrap();

        // Verify topological property
        for state in sorted.states() {
            for arc in sorted.arcs(state) {
                assert!(state < arc.nextstate);
            }
        }
    }

    #[test]
    fn test_state_sort_topological_rejects_cycles() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        // Create cycle
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s0));

        let result = state_sort(&fst, StateSortType::Topological);
        assert!(result.is_err());
    }

    #[test]
    fn test_state_sort_different_semirings() {
        // Test with LogWeight
        let mut fst = VectorFst::<LogWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, LogWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, LogWeight::new(0.5), s1));

        let sorted = state_sort(&fst, StateSortType::BreadthFirst).unwrap();
        assert_eq!(sorted.num_states(), 2);

        // Test with IntegerWeight
        let mut fst2 = VectorFst::<IntegerWeight>::new();
        let s0 = fst2.add_state();
        let s1 = fst2.add_state();

        fst2.set_start(s0);
        fst2.set_final(s1, IntegerWeight::one());
        fst2.add_arc(s0, Arc::new(1, 1, IntegerWeight::new(42), s1));

        let sorted2 = state_sort(&fst2, StateSortType::DepthFirst).unwrap();
        assert_eq!(sorted2.num_states(), 2);
    }

    #[test]
    fn test_state_sort_preserves_weights() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::new(3.5));

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.2), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(2.3), s2));

        let sorted = state_sort(&fst, StateSortType::BreadthFirst).unwrap();

        // Verify weights are preserved
        let orig_weights: f32 = fst
            .states()
            .flat_map(|s| fst.arcs(s))
            .map(|arc| *arc.weight.value())
            .sum();
        let sorted_weights: f32 = sorted
            .states()
            .flat_map(|s| sorted.arcs(s))
            .map(|arc| *arc.weight.value())
            .sum();

        assert!((orig_weights - sorted_weights).abs() < 1e-6);
    }

    #[test]
    fn test_state_sort_error_no_start_state() {
        let fst = VectorFst::<TropicalWeight>::new();

        let result = state_sort(&fst, StateSortType::BreadthFirst);
        assert!(result.is_ok()); // Empty FST is OK

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        fst2.add_state();
        // No start state set

        let result2 = state_sort(&fst2, StateSortType::BreadthFirst);
        assert!(result2.is_err());
    }

    #[test]
    fn test_state_sort_with_epsilon_arcs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Add epsilon arc (label 0)
        fst.add_arc(s0, Arc::new(0, 0, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(1, 1, TropicalWeight::one(), s2));

        for sort_type in [
            StateSortType::BreadthFirst,
            StateSortType::DepthFirst,
            StateSortType::Topological,
        ] {
            let sorted = state_sort(&fst, sort_type).unwrap();

            assert_eq!(sorted.num_states(), 3);
            assert_eq!(sorted.start(), Some(0));

            // Verify epsilon arc is preserved
            let has_epsilon = sorted
                .states()
                .any(|s| sorted.arcs(s).any(|arc| arc.ilabel == 0 && arc.olabel == 0));
            assert!(has_epsilon);
        }
    }

    #[test]
    fn test_state_sort_diamond_structure() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s3, TropicalWeight::one());

        // Diamond: 0 -> {1, 2} -> 3
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s2));
        fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::one(), s3));
        fst.add_arc(s2, Arc::new(4, 4, TropicalWeight::one(), s3));

        let bfs = state_sort(&fst, StateSortType::BreadthFirst).unwrap();
        let dfs = state_sort(&fst, StateSortType::DepthFirst).unwrap();
        let topo = state_sort(&fst, StateSortType::Topological).unwrap();

        // All should have same structure
        assert_eq!(bfs.num_states(), 4);
        assert_eq!(dfs.num_states(), 4);
        assert_eq!(topo.num_states(), 4);

        // Topological should maintain forward arcs
        for state in topo.states() {
            for arc in topo.arcs(state) {
                assert!(state < arc.nextstate);
            }
        }
    }

    #[test]
    fn test_state_sort_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();

        for sort_type in [
            StateSortType::BreadthFirst,
            StateSortType::DepthFirst,
            StateSortType::Topological,
        ] {
            let sorted = state_sort(&fst, sort_type).unwrap();
            assert_eq!(sorted.num_states(), 0);
            assert!(sorted.is_empty());
        }
    }
}
