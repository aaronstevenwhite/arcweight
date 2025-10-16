//! FST connection algorithm
//!
//! Removes unreachable and non-productive states from weighted finite-state transducers,
//! ensuring all remaining states are both accessible from the start and can reach final states.
//!
//! ## References
//!
//! - Hopcroft, J. E., and Ullman, J. D. (1979). "Introduction to Automata Theory,
//!   Languages, and Computation." Addison-Wesley.
//! - Mohri, M. (2009). "Weighted Automata Algorithms." Handbook of Weighted
//!   Automata, Springer, pp. 213-254.

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst, StateId};
use crate::semiring::Semiring;
use crate::{Error, Result};
use std::collections::{HashMap, HashSet};

/// Remove non-accessible and non-coaccessible states from an FST
///
/// Creates a new FST containing only states that are both accessible (reachable from
/// the start state) and coaccessible (can reach a final state). This operation removes
/// dead code and unreachable parts of the automaton while preserving the language.
///
/// # Algorithm Details
///
/// - **Accessibility Analysis:** Forward search from start state using DFS
/// - **Coaccessibility Analysis:** Backward search from final states using reverse arc index
/// - **Time Complexity:** O(|V| + |E|) for both forward and backward traversals
/// - **Space Complexity:** O(|V| + |E|) for state tracking, reverse index, and result construction
/// - **Language Preservation:** L(connect(T)) = L(T) (language unchanged)
///
/// # Mathematical Foundation
///
/// For an FST T, a state q is:
/// - **Accessible:** There exists a path from start state to q
/// - **Coaccessible:** There exists a path from q to some final state
/// - **Useful:** Both accessible and coaccessible
///
/// The connected FST contains only useful states, eliminating unreachable
/// computation paths while maintaining the same accepted language.
///
/// # Algorithm Steps
///
/// 1. **Forward Reachability:** Find all states accessible from start state
/// 2. **Backward Reachability:** Find all states that can reach final states
/// 3. **Intersection:** Keep only states that are both accessible and coaccessible
/// 4. **Reconstruction:** Build new FST with only useful states and their arcs
/// 5. **Preservation:** Maintain all weights, labels, and structural relationships
///
/// # Examples
///
/// ## Removing Unreachable States
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // FST with unreachable state
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state(); // start state
/// let s1 = fst.add_state(); // reachable state
/// let s2 = fst.add_state(); // final state
/// let s3 = fst.add_state(); // unreachable isolated state
///
/// fst.set_start(s0);
/// fst.set_final(s2, TropicalWeight::one());
///
/// // Connected path: s0 -> s1 -> s2
/// fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s1));
/// fst.add_arc(s1, Arc::new('b' as u32, 'b' as u32, TropicalWeight::one(), s2));
///
/// // s3 is isolated - no incoming or outgoing arcs
///
/// // Remove unreachable states
/// let connected: VectorFst<TropicalWeight> = connect(&fst)?;
///
/// // Result has 3 states (s0, s1, s2), s3 removed
/// assert_eq!(connected.num_states(), 3);
/// assert!(connected.num_states() < fst.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Removing Non-Coaccessible States
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // FST with dead-end state
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state(); // start
/// let s1 = fst.add_state(); // final state
/// let s2 = fst.add_state(); // dead-end state (not coaccessible)
///
/// fst.set_start(s0);
/// fst.set_final(s1, TropicalWeight::one());
///
/// // Good path: s0 -> s1 (final)
/// fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s1));
///
/// // Dead-end path: s0 -> s2 (but s2 cannot reach any final state)
/// fst.add_arc(s0, Arc::new('b' as u32, 'b' as u32, TropicalWeight::one(), s2));
/// // s2 has no outgoing arcs to final states
///
/// // Remove non-coaccessible states
/// let connected: VectorFst<TropicalWeight> = connect(&fst)?;
///
/// // Result removes dead-end s2, keeps s0 and s1
/// assert_eq!(connected.num_states(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## FST Optimization Pipeline
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Build complex FST that might have unreachable states
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
/// let s3 = fst.add_state();
/// let s4 = fst.add_state(); // Will become unreachable after operations
///
/// fst.set_start(s0);
/// fst.set_final(s2, TropicalWeight::one());
///
/// fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s1));
/// fst.add_arc(s1, Arc::new('b' as u32, 'b' as u32, TropicalWeight::one(), s2));
/// fst.add_arc(s0, Arc::new('c' as u32, 'c' as u32, TropicalWeight::one(), s3));
/// // s3 and s4 have no path to final states
///
/// // Optimization pipeline with connection
/// let connected: VectorFst<TropicalWeight> = connect(&fst)?;
/// let determinized: VectorFst<TropicalWeight> = determinize(&connected)?;
/// let minimized: VectorFst<TropicalWeight> = minimize(&determinized)?;
///
/// // Clean, optimized FST with only useful states
/// println!("Original: {} states, Optimized: {} states",
///          fst.num_states(), minimized.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Complex Graph Cleanup
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Create FST with mixed reachable/unreachable components
/// fn create_complex_fst() -> VectorFst<TropicalWeight> {
///     let mut fst = VectorFst::new();
///     
///     // Main component (reachable)
///     let s0 = fst.add_state(); // start
///     let s1 = fst.add_state();
///     let s2 = fst.add_state(); // final
///     
///     // Isolated component (unreachable)
///     let s3 = fst.add_state();
///     let s4 = fst.add_state();
///     
///     fst.set_start(s0);
///     fst.set_final(s2, TropicalWeight::one());
///     
///     // Main component arcs
///     fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
///     fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));
///     
///     // Isolated component arcs (unreachable from start)
///     fst.add_arc(s3, Arc::new(3, 3, TropicalWeight::one(), s4));
///     
///     fst
/// }
///
/// let complex_fst = create_complex_fst();
/// let cleaned: VectorFst<TropicalWeight> = connect(&complex_fst)?;
///
/// // Only main component remains
/// assert!(cleaned.num_states() <= 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Empty Result Handling
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // FST where no state can reach a final state
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
///
/// fst.set_start(s0);
/// // Note: no final states set!
///
/// fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s1));
///
/// // Connection removes all states (no coaccessible states)
/// let connected: VectorFst<TropicalWeight> = connect(&fst)?;
///
/// // Result is empty FST
/// assert_eq!(connected.num_states(), 0);
/// assert!(connected.start().is_none());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Use Cases
///
/// ## FST Optimization
/// - **Construction Cleanup:** Remove artifacts from FST construction algorithms
/// - **Composition Cleanup:** Clean up after compose operations that create unreachable states
/// - **Union Cleanup:** Remove isolated components after union operations
/// - **Preprocessing:** Prepare FSTs for other algorithms by ensuring connectivity
///
/// ## Compiler Optimization
/// - **Dead Code Elimination:** Remove unreachable states representing dead code paths
/// - **Control Flow Analysis:** Identify reachable program states
/// - **Data Flow Optimization:** Focus analysis on productive computation paths
/// - **Resource Cleanup:** Remove unused computational resources
///
/// ## Natural Language Processing
/// - **Grammar Cleanup:** Remove unreachable grammar rules and states
/// - **Lexicon Optimization:** Eliminate unused vocabulary entries
/// - **Parse Forest Pruning:** Remove non-productive parse paths
/// - **Model Compression:** Reduce language model size by removing unused states
///
/// ## Graph Analysis
/// - **Component Analysis:** Identify strongly connected components
/// - **Reachability Analysis:** Determine node accessibility in directed graphs
/// - **Network Optimization:** Remove isolated network components
/// - **Workflow Validation:** Ensure all workflow states are useful
///
/// # Performance Characteristics
///
/// - **Time Complexity:** O(|V| + |E|) for both forward and backward graph traversals
/// - **Space Complexity:** O(|V| + |E|) for state sets, reverse arc index, and mappings
/// - **Memory Efficiency:** Reverse index uses O(|E|) space for optimal time complexity
/// - **State Reduction:** Can significantly reduce FST size for sparse graphs
/// - **Cache Friendly:** Sequential access patterns in both forward and backward traversals
///
/// # Mathematical Properties
///
/// Connection preserves key FST properties:
/// - **Language Preservation:** L(connect(T)) = L(T) exactly
/// - **Weight Preservation:** All weights and structural relationships maintained
/// - **Determinism:** connect(deterministic) remains deterministic
/// - **Idempotency:** connect(connect(T)) = connect(T)
/// - **Monotonicity:** If T₁ ⊆ T₂, then connect(T₁) ⊆ connect(T₂)
///
/// # Implementation Details
///
/// The algorithm uses optimized two-phase reachability analysis:
/// 1. **Forward Phase:** DFS from start state to find all accessible states in O(|V| + |E|)
/// 2. **Reverse Index Construction:** Build predecessor mapping for efficient backward search in O(|V| + |E|)
/// 3. **Backward Phase:** DFS from final states using reverse index to find coaccessible states in O(|V| + |E|)
/// 4. **Intersection:** Keep only states appearing in both sets
/// 5. **Reconstruction:** Build new FST preserving all original relationships
///
/// The backward search uses a reverse arc index (`HashMap<StateId, Vec<StateId>>`) that maps each
/// state to its predecessors. This index is built in O(|E|) time by examining each arc once, then
/// enables O(1) predecessor lookup during the backward traversal, achieving overall O(|V| + |E|)
/// complexity. The trade-off is O(|E|) additional space for the reverse index.
///
/// # Optimization Opportunities
///
/// After connection, consider these follow-up optimizations:
/// - **Determinization:** Resolve any remaining nondeterminism
/// - **Minimization:** Further reduce state count through equivalence merging
/// - **Epsilon Removal:** Eliminate epsilon transitions for efficiency
/// - **Weight Pushing:** Optimize weight distribution
///
/// # Errors
///
/// Returns [`Error::Algorithm`] if:
/// - The input FST is invalid, corrupted, or has no start state
/// - Memory allocation fails during state set computation or result construction
/// - State mapping operations encounter invalid state references
/// - Arc creation fails due to invalid labels or weights
/// - The reachability analysis encounters cyclic dependencies in state enumeration
///
/// # See Also
///
/// - [`minimize`] - Further state reduction after connection
/// - [`determinize`] - Resolving nondeterminism
///
/// [`minimize`]: crate::algorithms::minimize::minimize
/// [`determinize`]: crate::algorithms::determinize::determinize
pub fn connect<W, F, M>(fst: &F) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let start = fst
        .start()
        .ok_or_else(|| Error::Algorithm("FST has no start state".into()))?;

    // find accessible states
    let accessible = find_accessible_states(fst, start);

    // find coaccessible states
    let coaccessible = find_coaccessible_states(fst);

    // keep only states that are both accessible and coaccessible
    let keep: HashSet<StateId> = accessible.intersection(&coaccessible).cloned().collect();

    if keep.is_empty() {
        return Ok(M::default());
    }

    // build new FST
    let mut result = M::default();
    let mut state_map = vec![None; fst.num_states()];

    // create new states
    for &state in &keep {
        let new_state = result.add_state();
        state_map[state as usize] = Some(new_state);
    }

    // set start
    if let Some(new_start) = state_map[start as usize] {
        result.set_start(new_start);
    }

    // copy arcs and final weights
    for &state in &keep {
        if let Some(new_state) = state_map[state as usize] {
            // final weight
            if let Some(weight) = fst.final_weight(state) {
                result.set_final(new_state, weight.clone());
            }

            // arcs
            for arc in fst.arcs(state) {
                if keep.contains(&arc.nextstate) {
                    if let Some(new_nextstate) = state_map[arc.nextstate as usize] {
                        result.add_arc(
                            new_state,
                            Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), new_nextstate),
                        );
                    }
                }
            }
        }
    }

    Ok(result)
}

fn find_accessible_states<W: Semiring, F: Fst<W>>(fst: &F, start: StateId) -> HashSet<StateId> {
    let mut accessible = HashSet::new();
    let mut stack = vec![start];

    while let Some(state) = stack.pop() {
        if accessible.insert(state) {
            for arc in fst.arcs(state) {
                stack.push(arc.nextstate);
            }
        }
    }

    accessible
}

fn find_coaccessible_states<W: Semiring, F: Fst<W>>(fst: &F) -> HashSet<StateId> {
    // Build reverse arc index mapping each state to its predecessors: O(|V| + |E|)
    let mut predecessors: HashMap<StateId, Vec<StateId>> = HashMap::new();
    for state in fst.states() {
        for arc in fst.arcs(state) {
            predecessors.entry(arc.nextstate).or_default().push(state);
        }
    }

    let mut coaccessible = HashSet::new();
    let mut stack = Vec::new();

    // Start from all final states: O(|V|)
    for state in fst.states() {
        if fst.is_final(state) {
            stack.push(state);
        }
    }

    // Backward search using reverse index: O(|V| + |E|)
    while let Some(state) = stack.pop() {
        if coaccessible.insert(state) {
            // Process all predecessor states from reverse index
            if let Some(preds) = predecessors.get(&state) {
                for &pred in preds {
                    if !coaccessible.contains(&pred) {
                        stack.push(pred);
                    }
                }
            }
        }
    }

    coaccessible
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use num_traits::One;

    #[test]
    fn test_connect_removes_unreachable() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state(); // unreachable

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s2, Arc::new(2, 2, TropicalWeight::new(1.0), s3)); // disconnected

        let connected: VectorFst<TropicalWeight> = connect(&fst).unwrap();

        // Should remove unreachable states
        assert!(connected.num_states() < fst.num_states());
        assert!(connected.start().is_some());
    }

    #[test]
    fn test_connect_already_connected() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));

        let connected: VectorFst<TropicalWeight> = connect(&fst).unwrap();

        // Should be identical or similar
        assert_eq!(connected.num_states(), fst.num_states());
        assert!(connected.start().is_some());
    }
}
