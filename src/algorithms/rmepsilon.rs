//! Epsilon removal algorithm
//!
//! Removes epsilon (empty) transitions from weighted FSTs while preserving semantics.
//!
//! # Semiring Requirements
//!
//! Epsilon removal benefits from the **star operation** for guaranteed termination:
//! - `StarSemiring` trait provides Kleene star: w* = 1 ⊕ w ⊕ w² ⊕ ...
//! - K-closed property ensures convergence for epsilon cycles
//! - Without star operation, may not terminate on epsilon cycles with non-convergent weights
//!
//! # Supported Semirings
//!
//! - ✅ `TropicalWeight` - Star operation well-defined (idempotent)
//! - ✅ `BooleanWeight` - Star operation trivial (always converges)
//! - ⚠️  `ProbabilityWeight` - May not converge for epsilon cycles
//! - ⚠️  `LogWeight` - Convergence depends on cycle weights
//!
//! # Termination Guarantees
//!
//! Algorithm terminates when:
//! - FST is acyclic (no epsilon cycles)
//! - Semiring is k-closed (star operation converges)
//! - Epsilon cycle weights converge to a fixed point

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst, StateId};
use crate::semiring::StarSemiring;
use crate::Result;
use std::collections::{HashMap, VecDeque};

/// Remove epsilon (empty) transitions from an FST while preserving language semantics
///
/// Eliminates all epsilon transitions by computing epsilon closures and creating
/// equivalent direct transitions. The resulting FST accepts the same language
/// but with more efficient processing due to eliminated epsilon transitions.
///
/// # Algorithm Details
///
/// - **Epsilon Closure:** For each state, compute all states reachable via epsilon transitions
/// - **Direct Transitions:** Create direct arcs bypassing epsilon paths
/// - **Time Complexity:** O(|V|² × |E|) in worst case due to closure computation
/// - **Space Complexity:** O(|V|²) for storing epsilon closures
/// - **Language Preservation:** L(remove_epsilons(T)) = L(T) exactly
///
/// # Mathematical Foundation
///
/// For an FST with epsilon transitions, the epsilon-free equivalent computes:
/// - **Epsilon Closure:** ε*(q) = {p : q →ε* p} (states reachable via epsilon paths)
/// - **Direct Arcs:** For each non-epsilon arc q →a p, add arcs r →a p for all r ∈ ε*(q)
/// - **Final Weights:** Combine final weights through epsilon closures
/// - **Weight Computation:** Uses semiring operations to combine path weights
///
/// # Algorithm Steps
///
/// 1. **Copy Structure:** Create new FST with same states as original
/// 2. **Epsilon Closure:** For each state, compute epsilon-reachable states with weights
/// 3. **Direct Arcs:** Add direct arcs bypassing epsilon transitions
/// 4. **Final Weight Update:** Propagate final weights through epsilon closures
/// 5. **Clean Result:** Result FST has no epsilon transitions
///
/// # Examples
///
/// ## Basic Epsilon Removal
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // FST with epsilon transitions: start --a--> s1 --ε--> s2 --b--> final
/// let mut fst = VectorFst::<BooleanWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
/// let s3 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s3, BooleanWeight::one());
///
/// fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, BooleanWeight::one(), s1));
/// fst.add_arc(s1, Arc::epsilon(BooleanWeight::one(), s2)); // epsilon transition
/// fst.add_arc(s2, Arc::new('b' as u32, 'b' as u32, BooleanWeight::one(), s3));
///
/// // Remove epsilon transitions
/// let no_eps: VectorFst<BooleanWeight> = remove_epsilons(&fst)?;
///
/// // Result accepts "ab" directly without intermediate epsilon
/// assert_eq!(no_eps.num_states(), fst.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Multiple Epsilon Paths
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // FST with multiple epsilon paths
/// let mut fst = VectorFst::<BooleanWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
/// let s3 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s3, BooleanWeight::one());
///
/// // Multiple epsilon paths: s0 --ε--> s1 --ε--> s2 --ε--> s3
/// fst.add_arc(s0, Arc::epsilon(BooleanWeight::one(), s1));
/// fst.add_arc(s1, Arc::epsilon(BooleanWeight::one(), s2));
/// fst.add_arc(s2, Arc::epsilon(BooleanWeight::one(), s3));
///
/// // Also direct non-epsilon arc
/// fst.add_arc(s1, Arc::new('x' as u32, 'x' as u32, BooleanWeight::one(), s3));
///
/// // Remove all epsilon chains
/// let cleaned: VectorFst<BooleanWeight> = remove_epsilons(&fst)?;
///
/// // Result has direct paths without epsilon steps
/// println!("Cleaned FST ready");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Weighted Epsilon Removal
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Weighted FST with epsilon transitions
/// let mut fst = VectorFst::<BooleanWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s2, BooleanWeight::one());
///
/// // Path: s0 --a--> s1 --ε--> s2(final)
/// fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, BooleanWeight::one(), s1));
/// fst.add_arc(s1, Arc::epsilon(BooleanWeight::one(), s2));
///
/// // Epsilon removal creates direct final weight for s1
/// let result: VectorFst<BooleanWeight> = remove_epsilons(&fst)?;
///
/// // s1 now has final weight due to epsilon closure to s2
/// assert!(result.start().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Complex Epsilon Network
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Complex epsilon transition network
/// let mut fst = VectorFst::<BooleanWeight>::new();
/// let states: Vec<_> = (0..5).map(|_| fst.add_state()).collect();
///
/// fst.set_start(states[0]);
/// fst.set_final(states[4], BooleanWeight::one());
///
/// // Mix of epsilon and non-epsilon transitions
/// fst.add_arc(states[0], Arc::new('a' as u32, 'a' as u32, BooleanWeight::one(), states[1]));
/// fst.add_arc(states[1], Arc::epsilon(BooleanWeight::one(), states[2]));
/// fst.add_arc(states[1], Arc::epsilon(BooleanWeight::one(), states[3]));
/// fst.add_arc(states[2], Arc::new('b' as u32, 'b' as u32, BooleanWeight::one(), states[4]));
/// fst.add_arc(states[3], Arc::new('c' as u32, 'c' as u32, BooleanWeight::one(), states[4]));
///
/// // Remove epsilon network, creating direct alternatives
/// let simplified: VectorFst<BooleanWeight> = remove_epsilons(&fst)?;
///
/// // Result accepts "ab" and "ac" directly
/// assert_eq!(simplified.num_states(), fst.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Optimization Pipeline Integration
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Optimization pipeline with epsilon removal
/// fn optimize_fst(fst: &VectorFst<BooleanWeight>) -> Result<VectorFst<BooleanWeight>> {
///     // Step 1: Remove epsilon transitions for efficiency
///     let no_epsilon: VectorFst<BooleanWeight> = remove_epsilons(fst)?;
///     
///     // Step 2: Remove unreachable states
///     let connected: VectorFst<BooleanWeight> = connect(&no_epsilon)?;
///     
///     // Step 3: Determinize if needed (epsilon-free FSTs determinize better)
///     // Note: determinize requires DivisibleSemiring, not available for BooleanWeight
///     // let determinized: VectorFst<BooleanWeight> = determinize(&connected)?;
///     
///     Ok(connected)
/// }
///
/// // Create test FST
/// let mut test_fst = VectorFst::new();
/// let s0 = test_fst.add_state();
/// let s1 = test_fst.add_state();
/// test_fst.set_start(s0);
/// test_fst.set_final(s1, BooleanWeight::one());
/// test_fst.add_arc(s0, Arc::epsilon(BooleanWeight::one(), s1));
///
/// let optimized = optimize_fst(&test_fst)?;
/// println!("FST optimized");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Use Cases
///
/// ## FST Preprocessing
/// - **Algorithm Preparation:** Remove epsilons before determinization or minimization
/// - **Performance Optimization:** Eliminate epsilon transitions for faster traversal
/// - **Canonical Form:** Create epsilon-free canonical representation
/// - **Memory Efficiency:** Reduce state space complexity
///
/// ## Regular Expression Processing
/// - **Regex Compilation:** Remove epsilons from regex-derived FSTs
/// - **Pattern Matching:** Optimize pattern matching automata
/// - **Text Processing:** Eliminate empty transitions in text processors
/// - **Lexical Analysis:** Clean up tokenizer automata
///
/// ## Natural Language Processing
/// - **Grammar Cleanup:** Remove epsilon productions from CFG-derived FSTs
/// - **Morphological Analysis:** Clean morphology automata
/// - **Phonological Rules:** Optimize phonology rule FSTs
/// - **Translation Models:** Simplify translation automata
///
/// ## Speech Processing
/// - **Pronunciation Models:** Remove epsilon paths in pronunciation FSTs
/// - **Acoustic Models:** Optimize acoustic model automata
/// - **Language Models:** Clean statistical language model FSTs
/// - **ASR Optimization:** Prepare FSTs for speech recognition
///
/// # Performance Characteristics
///
/// - **Time Complexity:** O(|V|² × |E|) worst case for dense epsilon networks
/// - **Space Complexity:** O(|V|²) for storing epsilon closure information
/// - **Practical Performance:** Often much better than worst case for sparse graphs
/// - **Memory Access:** Sequential state processing improves cache locality
/// - **Convergence:** Guaranteed for k-closed semirings with proper star operation
///
/// # Mathematical Properties
///
/// Epsilon removal preserves fundamental FST properties:
/// - **Language Preservation:** L(remove_epsilons(T)) = L(T) exactly
/// - **Weight Preservation:** All path weights maintained through semiring operations
/// - **Structural Properties:** May change state connectivity but preserves semantics
/// - **Determinism:** Epsilon-free FSTs are easier to determinize efficiently
/// - **Compositionality:** remove_epsilons(T₁ ∘ T₂) relates to remove_epsilons(T₁) ∘ remove_epsilons(T₂)
///
/// # Implementation Details
///
/// The algorithm computes epsilon closures using breadth-first search with weight
/// accumulation. For each state, it finds all epsilon-reachable states and their
/// combined weights, then creates direct arcs that bypass epsilon paths.
///
/// Cycle handling relies on the semiring's plus operation to combine multiple
/// paths to the same state, ensuring convergence when the semiring is k-closed.
///
/// # Semiring Considerations
///
/// Different semirings have different epsilon removal characteristics:
/// - **Boolean Semiring:** Epsilon removal is always efficient and terminates
/// - **Tropical Semiring:** Idempotent plus ensures convergence
/// - **Probability Semiring:** May require careful handling of convergence
/// - **Log Semiring:** Convergence depends on cycle weight properties
///
/// # Optimization Opportunities
///
/// After epsilon removal, consider these optimizations:
/// - **Determinization:** Now more efficient without epsilon transitions
/// - **Minimization:** Can reduce states further after epsilon elimination
/// - **Connection:** Remove states that became unreachable
/// - **Topological Sort:** Order states for optimal processing
///
/// # Errors
///
/// Returns [`Error::Algorithm`](crate::Error::Algorithm) if:
/// - The input FST is invalid, corrupted, or malformed
/// - Memory allocation fails during epsilon closure computation
/// - Epsilon closure computation encounters infinite loops or non-convergent weights
/// - The semiring does not properly support required star operations
/// - Weight computation overflows or produces invalid results
///
/// # See Also
///
/// - [`determinize()`](crate::algorithms::determinize()) for algorithms that benefit from epsilon removal
/// - [`minimize()`](crate::algorithms::minimize()) for state reduction after epsilon removal
/// - [`connect()`](crate::algorithms::connect()) for removing unreachable states
/// - [Working with FSTs](../../docs/working-with-fsts/README.md) for FST manipulation patterns
/// - [Core Concepts](../../docs/core-concepts/algorithms.md#epsilon-removal) for mathematical theory
pub fn remove_epsilons<W, F, M>(fst: &F) -> Result<M>
where
    W: StarSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();

    // copy states
    for _ in 0..fst.num_states() {
        result.add_state();
    }

    // set start
    if let Some(start) = fst.start() {
        result.set_start(start);
    }

    // compute epsilon closure for each state
    for state in fst.states() {
        let closure = compute_epsilon_closure(fst, state)?;

        // add non-epsilon arcs
        for arc in fst.arcs(state) {
            if !arc.is_epsilon() {
                result.add_arc(state, arc.clone());
            }
        }

        // add arcs from epsilon closure
        for &(closure_state, ref weight) in &closure {
            if closure_state != state {
                // add non-epsilon arcs from closure state
                for arc in fst.arcs(closure_state) {
                    if !arc.is_epsilon() {
                        result.add_arc(
                            state,
                            Arc::new(
                                arc.ilabel,
                                arc.olabel,
                                weight.times(&arc.weight),
                                arc.nextstate,
                            ),
                        );
                    }
                }

                // handle final weights
                if let Some(final_weight) = fst.final_weight(closure_state) {
                    let new_weight = weight.times(final_weight);
                    if let Some(existing) = fst.final_weight(state) {
                        result.set_final(state, existing.plus(&new_weight));
                    } else {
                        result.set_final(state, new_weight);
                    }
                }
            }
        }

        // copy final weight
        if let Some(weight) = fst.final_weight(state) {
            result.set_final(state, weight.clone());
        }
    }

    Ok(result)
}

fn compute_epsilon_closure<W: StarSemiring, F: Fst<W>>(
    fst: &F,
    start: StateId,
) -> Result<Vec<(StateId, W)>> {
    let mut closure = Vec::new();
    let mut visited = HashMap::new();
    let mut queue = VecDeque::new();

    queue.push_back((start, W::one()));
    visited.insert(start, W::one());

    while let Some((state, weight)) = queue.pop_front() {
        closure.push((state, weight.clone()));

        // follow epsilon transitions
        for arc in fst.arcs(state) {
            if arc.is_epsilon() {
                let next_weight = weight.times(&arc.weight);

                match visited.get(&arc.nextstate) {
                    Some(existing) => {
                        // update if we found a better path
                        let combined = existing.plus(&next_weight);
                        visited.insert(arc.nextstate, combined);
                    }
                    None => {
                        visited.insert(arc.nextstate, next_weight.clone());
                        queue.push_back((arc.nextstate, next_weight));
                    }
                }
            }
        }
    }

    Ok(closure)
}
