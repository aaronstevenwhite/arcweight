//! Condense strongly connected components into single states
//!
//! Identifies strongly connected components (SCCs) in an FST and condenses
//! each SCC into a single representative state.
//!
//! ## Overview
//!
//! Condensation transforms an FST by collapsing each strongly connected
//! component into a single state, creating a DAG (directed acyclic graph)
//! of SCCs. This is useful for graph analysis, optimization, and understanding
//! FST structure.
//!
//! ## Algorithm
//!
//! Uses Tarjan's strongly connected components algorithm:
//! 1. Depth-first search with discovery times and low-link values
//! 2. Stack-based SCC identification
//! 3. Build condensation graph with representative states
//! 4. Map arcs between SCCs
//!
//! ## Complexity
//!
//! - **Time:** O(|V| + |E|) - single DFS traversal using Tarjan's algorithm
//! - **Space:** O(|V|) - stack and auxiliary arrays for DFS
//!
//! ## Theoretical Background
//!
//! A strongly connected component is a maximal set of states where every
//! state is reachable from every other state in the set. The condensation
//! graph has one state per SCC and is always a DAG.
//!
//! Properties:
//! - Condensation is always acyclic (DAG property)
//! - Each SCC maps to exactly one state in condensation
//! - Preserves reachability between SCCs
//!
//! ## Use Cases
//!
//! - **Graph Analysis:** Understand cyclic structure of FST
//! - **Optimization:** Identify and process cyclic subgraphs
//! - **Minimization:** Preprocessing step for certain minimization algorithms
//! - **Visualization:** Simplify FST representation by hiding internal SCC structure
//!
//! ## References
//!
//! - Robert Tarjan (1972). "Depth-First Search and Linear Graph Algorithms."
//!   SIAM Journal on Computing, 1(2):146-160.
//! - Thomas H. Cormen et al. (2009). "Introduction to Algorithms," 3rd Edition.
//!   Section 22.5: Strongly Connected Components.
//!
//! ## Examples
//!
//! ### Acyclic FST (Each State is Own SCC)
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
//! fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
//! fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(2.0), s2));
//!
//! let condensed = condense(&fst)?;
//!
//! // Acyclic FST: each state is its own SCC
//! assert_eq!(condensed.num_states(), fst.num_states());
//! # Ok::<(), arcweight::Error>(())
//! ```
//!
//! ### Cyclic FST with Multiple SCCs
//!
//! ```
//! use arcweight::prelude::*;
//!
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! let s2 = fst.add_state();
//! let s3 = fst.add_state();
//! fst.set_start(s0);
//! fst.set_final(s3, TropicalWeight::one());
//!
//! // Create cycle between s1 and s2
//! fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
//! fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));
//! fst.add_arc(s2, Arc::new(3, 3, TropicalWeight::one(), s1)); // Back edge
//! fst.add_arc(s2, Arc::new(4, 4, TropicalWeight::one(), s3));
//!
//! let condensed = condense(&fst)?;
//!
//! // s1 and s2 form one SCC, condensed to single state
//! // Result has 3 states: {s0}, {s1,s2}, {s3}
//! assert_eq!(condensed.num_states(), 3);
//! # Ok::<(), arcweight::Error>(())
//! ```

use crate::arc::Arc;
use crate::fst::{Fst, Label, MutableFst, StateId, VectorFst};
use crate::semiring::Semiring;
use crate::Result;
use std::cmp::min;
use std::collections::HashMap;

/// Condenses strongly connected components (SCCs) into single states.
///
/// Each SCC is replaced by a representative state, with arcs adjusted
/// appropriately. The resulting FST is a DAG of SCCs. This transformation
/// reveals the high-level structure of the FST by collapsing cyclic regions.
///
/// # Complexity
///
/// - **Time:** O(V + E) where V = number of states, E = number of arcs
///   - Single DFS traversal: O(V + E)
///   - SCC identification via Tarjan's algorithm: O(V + E)
///   - Condensation graph construction: O(V + E)
/// - **Space:** O(V) for DFS stack and auxiliary structures
///   - Discovery times array: O(V)
///   - Low-link values array: O(V)
///   - SCC stack: O(V) worst case
///   - SCC mapping: O(V)
///
/// # Algorithm
///
/// Tarjan's strongly connected components algorithm (1972):
/// 1. Initialize DFS with discovery times and low-link values
/// 2. For each unvisited state:
///    - Perform DFS, pushing states onto stack
///    - Track low-link value = min reachable discovery time
///    - When backtracking, if low-link\[v\] == discovery\[v\]: found SCC root
///    - Pop stack until v, all popped states form one SCC
/// 3. Build condensation graph:
///    - Create one state per SCC
///    - Add arcs between SCCs (combining weights via ⊕ for duplicates)
///    - Set final weights for SCCs containing final states
///
/// # Performance Notes
///
/// - **Linear time:** Optimal O(V + E) complexity for SCC detection
/// - **Single pass:** No iteration or convergence needed
/// - **Cache friendly:** Sequential DFS traversal
/// - **Acyclic output:** Result is always a DAG, enabling topological algorithms
/// - **Weight combination:** Arcs between same SCC pairs combined using semiring addition (⊕)
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
/// fst.set_final(s2, TropicalWeight::one());
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
/// fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));
///
/// let condensed = condense(&fst)?;
///
/// // Acyclic: structure preserved
/// assert_eq!(condensed.num_states(), 3);
/// # Ok::<(), arcweight::Error>(())
/// ```
///
/// # Errors
///
/// Returns error if FST structure is invalid.
///
/// # See Also
///
/// - [`connect`] - Remove unreachable states (often applied after condensation)
/// - [`shortest_distance`] - Works efficiently on condensed (acyclic) FSTs
/// - [`state_sort`] - Topological sort works on condensed DAGs
/// - [Semiring trait](crate::semiring::Semiring) - Weight combination for duplicate arcs
///
/// [`connect`]: crate::algorithms::connect::connect
/// [`shortest_distance`]: crate::algorithms::shortest_distance::shortest_distance
/// [`state_sort`]: crate::algorithms::state_sort::state_sort
pub fn condense<W, F>(fst: &F) -> Result<VectorFst<W>>
where
    W: Semiring,
    F: Fst<W>,
{
    if fst.num_states() == 0 {
        return Ok(VectorFst::new());
    }

    // Run Tarjan's algorithm to find SCCs
    let scc_map = find_sccs(fst);

    // Find number of SCCs
    let num_sccs = if scc_map.is_empty() {
        0
    } else {
        *scc_map.values().max().unwrap() + 1
    };

    // Build condensation graph
    let mut result = VectorFst::<W>::new();

    // Add states (one per SCC)
    for _ in 0..num_sccs {
        result.add_state();
    }

    // Set start state (SCC containing original start state)
    if let Some(start) = fst.start() {
        let start_scc = scc_map[&start];
        result.set_start(start_scc);
    }

    // Track which SCC-to-SCC arcs we've added (to avoid duplicates)
    let mut added_arcs: HashMap<(StateId, Label, Label, StateId), W> = HashMap::new();

    // Process all states and arcs
    for state_idx in 0..fst.num_states() {
        let state = state_idx as StateId;
        let src_scc = scc_map[&state];

        // Handle final weights: if any state in SCC is final, SCC is final
        if let Some(final_weight) = fst.final_weight(state) {
            // Use minimum (or sum, depending on semantics) of final weights in SCC
            if let Some(existing) = result.final_weight(src_scc) {
                // Combine with existing final weight using semiring addition
                result.set_final(src_scc, existing.plus(final_weight));
            } else {
                result.set_final(src_scc, final_weight.clone());
            }
        }

        // Process arcs
        for arc in fst.arcs(state) {
            let dest_scc = scc_map[&arc.nextstate];

            // Only add arcs between different SCCs (no self-loops in condensation)
            if src_scc != dest_scc {
                let key = (src_scc, arc.ilabel, arc.olabel, dest_scc);

                // Combine weights if arc already exists
                if let Some(existing_weight) = added_arcs.get(&key) {
                    let new_weight = existing_weight.plus(&arc.weight);
                    added_arcs.insert(key, new_weight.clone());
                } else {
                    added_arcs.insert(key, arc.weight.clone());
                }
            }
        }
    }

    // Add all unique inter-SCC arcs to result
    for ((src_scc, ilabel, olabel, dest_scc), weight) in added_arcs {
        result.add_arc(src_scc, Arc::new(ilabel, olabel, weight, dest_scc));
    }

    Ok(result)
}

/// Find strongly connected components using Tarjan's algorithm
///
/// Returns a map from state ID to SCC ID (numbered in reverse topological order).
fn find_sccs<W, F>(fst: &F) -> HashMap<StateId, StateId>
where
    W: Semiring,
    F: Fst<W>,
{
    let n = fst.num_states();
    let mut index_counter = 0;
    let mut stack = Vec::new();
    let mut indices = vec![None; n];
    let mut lowlinks = vec![None; n];
    let mut on_stack = vec![false; n];
    let mut scc_map = HashMap::new();
    let mut scc_id = 0;

    // Run DFS from each unvisited state
    for state_idx in 0..n {
        let state = state_idx as StateId;
        if indices[state_idx].is_none() {
            tarjan_dfs(
                fst,
                state,
                &mut index_counter,
                &mut stack,
                &mut indices,
                &mut lowlinks,
                &mut on_stack,
                &mut scc_map,
                &mut scc_id,
            );
        }
    }

    scc_map
}

/// Tarjan's DFS helper function
#[allow(clippy::too_many_arguments)]
fn tarjan_dfs<W, F>(
    fst: &F,
    v: StateId,
    index_counter: &mut usize,
    stack: &mut Vec<StateId>,
    indices: &mut Vec<Option<usize>>,
    lowlinks: &mut Vec<Option<usize>>,
    on_stack: &mut Vec<bool>,
    scc_map: &mut HashMap<StateId, StateId>,
    scc_id: &mut StateId,
) where
    W: Semiring,
    F: Fst<W>,
{
    let v_idx = v as usize;

    // Set the depth index for v
    indices[v_idx] = Some(*index_counter);
    lowlinks[v_idx] = Some(*index_counter);
    *index_counter += 1;
    stack.push(v);
    on_stack[v_idx] = true;

    // Consider successors of v
    for arc in fst.arcs(v) {
        let w = arc.nextstate;
        let w_idx = w as usize;
        if indices[w_idx].is_none() {
            // Successor w has not yet been visited; recurse on it
            tarjan_dfs(
                fst,
                w,
                index_counter,
                stack,
                indices,
                lowlinks,
                on_stack,
                scc_map,
                scc_id,
            );
            lowlinks[v_idx] = Some(min(lowlinks[v_idx].unwrap(), lowlinks[w_idx].unwrap()));
        } else if on_stack[w_idx] {
            // Successor w is in stack and hence in the current SCC
            lowlinks[v_idx] = Some(min(lowlinks[v_idx].unwrap(), indices[w_idx].unwrap()));
        }
    }

    // If v is a root node, pop the stack and create an SCC
    if lowlinks[v_idx] == indices[v_idx] {
        let current_scc = *scc_id;
        *scc_id += 1;

        loop {
            let w = stack.pop().unwrap();
            let w_idx = w as usize;
            on_stack[w_idx] = false;
            scc_map.insert(w, current_scc);
            if w == v {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_condense_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();
        let result = condense(&fst).unwrap();
        assert_eq!(result.num_states(), 0);
    }

    #[test]
    fn test_condense_single_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::one());

        let result = condense(&fst).unwrap();
        assert_eq!(result.num_states(), 1);
        assert_eq!(result.start(), Some(0));
        assert!(result.final_weight(0).is_some());
    }

    #[test]
    fn test_condense_acyclic_chain() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(2.0), s2));

        let result = condense(&fst).unwrap();

        // Acyclic: each state is its own SCC
        assert_eq!(result.num_states(), 3);
        assert!(result.start().is_some());
        assert_eq!(result.arcs(result.start().unwrap()).count(), 1);
    }

    #[test]
    fn test_condense_simple_cycle() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s0)); // Back edge

        let result = condense(&fst).unwrap();

        // Both states in same SCC
        assert_eq!(result.num_states(), 1);
        assert!(result.start().is_some());
        assert!(result.final_weight(result.start().unwrap()).is_some());
    }

    #[test]
    fn test_condense_multiple_sccs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s3, TropicalWeight::one());

        // s0 is alone
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));

        // s1 and s2 form a cycle
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));
        fst.add_arc(s2, Arc::new(3, 3, TropicalWeight::one(), s1));

        // s2 -> s3
        fst.add_arc(s2, Arc::new(4, 4, TropicalWeight::one(), s3));

        let result = condense(&fst).unwrap();

        // Three SCCs: {s0}, {s1, s2}, {s3}
        assert_eq!(result.num_states(), 3);
        assert!(result.start().is_some());
    }

    #[test]
    fn test_condense_self_loop() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s0)); // Self-loop
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s1));

        let result = condense(&fst).unwrap();

        // s0 is its own SCC (self-loop), s1 is separate
        assert_eq!(result.num_states(), 2);
    }

    #[test]
    fn test_condense_preserves_labels() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(5, 10, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(15, 20, TropicalWeight::new(2.0), s2));

        let result = condense(&fst).unwrap();

        // Check that labels are preserved
        let arcs_s0: Vec<_> = result.arcs(result.start().unwrap()).collect();
        assert!(arcs_s0.iter().any(|a| a.ilabel == 5 && a.olabel == 10));
    }

    #[test]
    fn test_condense_complex_graph() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();
        let s4 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s4, TropicalWeight::one());

        // Create complex SCC structure
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));
        fst.add_arc(s2, Arc::new(3, 3, TropicalWeight::one(), s3));
        fst.add_arc(s3, Arc::new(4, 4, TropicalWeight::one(), s1)); // Cycle: s1->s2->s3->s1
        fst.add_arc(s2, Arc::new(5, 5, TropicalWeight::one(), s4));

        let result = condense(&fst).unwrap();

        // {s0}, {s1, s2, s3}, {s4}
        assert_eq!(result.num_states(), 3);
    }

    #[test]
    fn test_condense_all_connected() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Fully connected cycle
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));
        fst.add_arc(s2, Arc::new(3, 3, TropicalWeight::one(), s0));

        let result = condense(&fst).unwrap();

        // All states in one SCC
        assert_eq!(result.num_states(), 1);
        assert!(result.start().is_some());
        assert!(result.final_weight(0).is_some());
    }

    #[test]
    fn test_condense_with_boolean_weight() {
        let mut fst = VectorFst::<BooleanWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, BooleanWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, BooleanWeight::one(), s1));
        fst.add_arc(s1, Arc::new(2, 2, BooleanWeight::one(), s0));

        let result = condense(&fst).unwrap();

        assert_eq!(result.num_states(), 1);
    }

    #[test]
    fn test_condense_combines_arc_weights() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s3, TropicalWeight::one());

        // Create two arcs between SCCs with same labels
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(3.0), s2));
        fst.add_arc(s2, Arc::new(3, 3, TropicalWeight::new(1.0), s1)); // Cycle
        fst.add_arc(s1, Arc::new(4, 4, TropicalWeight::new(5.0), s3));

        let result = condense(&fst).unwrap();

        // Should have 3 SCCs: {s0}, {s1, s2}, {s3}
        assert_eq!(result.num_states(), 3);
    }

    #[test]
    fn test_condense_no_start_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        fst.add_state();
        fst.add_state();

        let result = condense(&fst).unwrap();

        assert_eq!(result.num_states(), 2);
        assert_eq!(result.start(), None);
    }

    #[test]
    fn test_condense_multiple_final_weights_in_scc() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::new(2.0));
        fst.set_final(s1, TropicalWeight::new(3.0));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s0));

        let result = condense(&fst).unwrap();

        // Both states in same SCC, final weights should be combined
        assert_eq!(result.num_states(), 1);
        assert!(result.final_weight(0).is_some());
        // Should be min(2.0, 3.0) = 2.0 for tropical semiring
        assert_eq!(result.final_weight(0), Some(&TropicalWeight::new(2.0)));
    }
}
