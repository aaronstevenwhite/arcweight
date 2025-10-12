//! Shortest distance algorithm for computing path weight sums
//!
//! ## Overview
//!
//! Computes the sum of weights of all successful paths from the start state to
//! each state in an FST. The "sum" is defined by the semiring's addition operation (⊕),
//! so for tropical semirings this computes minimum distances, while for probability
//! semirings it computes total probabilities.
//!
//! ## Algorithm
//!
//! Based on Mohri (2002) "Semiring frameworks and algorithms for shortest-distance problems."
//! The algorithm has two cases:
//!
//! **Acyclic FSTs:**
//! - Uses topological ordering for single-pass computation
//! - Processes states in dependency order
//! - Guaranteed O(|V| + |E|) time complexity
//!
//! **Cyclic FSTs (k-closed semirings):**
//! - Iterative relaxation until convergence
//! - Requires k-closed semiring property
//! - Converges when distances stabilize
//! - O(k × (|V| + |E|)) where k is iterations to converge
//!
//! ## Complexity
//!
//! - **Acyclic case:**
//!   - Time: O(|V| + |E|) - topological sort + single pass
//!   - Space: O(|V|) - distance vector
//!
//! - **Cyclic case:**
//!   - Time: O(k × (|V| + |E|)) - k iterations of relaxation
//!   - Space: O(|V|) - distance vector
//!
//! ## References
//!
//! - Mehryar Mohri (2002). "Semiring frameworks and algorithms for shortest-distance problems."
//!   Journal of Automata, Languages and Combinatorics.
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
//! fst.set_final(s1, TropicalWeight::new(0.5));
//! fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
//!
//! let distances = shortest_distance(&fst)?;
//! // distances[s1] contains min weight from start to s1
//! # Ok::<(), arcweight::Error>(())
//! ```

use crate::fst::{Fst, StateId};
use crate::semiring::Semiring;
use crate::{Error, Result};
use std::collections::HashSet;

/// Maximum iterations for cyclic FST convergence
const MAX_ITERATIONS: usize = 1000;

/// Computes the sum of weights of all successful paths in an FST.
///
/// This function computes the shortest distance from the start state to each
/// state in the FST, where "shortest" is defined according to the semiring's
/// addition operation (which typically represents a generalized "sum").
///
/// For **acyclic FSTs**, uses a single forward pass in topological order.
/// For **cyclic FSTs** with k-closed semirings, iterates until convergence.
///
/// # Algorithm
///
/// Based on Mohri (2002) "Semiring frameworks and algorithms for
/// shortest-distance problems."
///
/// **Acyclic case:**
/// 1. Compute topological ordering of states
/// 2. Initialize distance[start] = One, others = Zero
/// 3. Process states in topological order
/// 4. For each outgoing arc: distance[dest] ⊕= distance[src] ⊗ arc.weight
///
/// **Cyclic case (k-closed semiring):**
/// 1. Initialize distance[start] = One, others = Zero
/// 2. Iterate until convergence (or max iterations):
///    - For each state, for each arc: distance[dest] ⊕= distance[src] ⊗ arc.weight
/// 3. Check for convergence when distances stabilize
///
/// # Complexity
///
/// - **Acyclic:** O(|V| + |E|) - single topological pass
///   - O(|V| + |E|) for topological sort
///   - O(|V| + |E|) for distance computation
///   - Total: O(|V| + |E|)
///
/// - **Cyclic:** O(k × (|V| + |E|)) where k is iterations until convergence
///   - Each iteration: O(|V| + |E|) for relaxation
///   - Convergence check: O(|V|)
///   - Total: O(k × (|V| + |E|))
///
/// - **Space:** O(|V|) for distance vector
///
/// # Examples
///
/// ## Acyclic FST (Linear Chain)
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
/// fst.set_start(s0);
/// fst.set_final(s2, TropicalWeight::new(0.5));
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
/// fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(2.0), s2));
///
/// let distances = shortest_distance(&fst)?;
/// // distances[s0] = 0.0 (start)
/// // distances[s1] = 1.0 (min path from s0)
/// // distances[s2] = 3.0 (min path from s0: 1.0 + 2.0)
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Branching Paths
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// fst.set_start(s0);
/// fst.set_final(s1, TropicalWeight::one());
///
/// // Two paths with different weights - tropical takes min
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
/// fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s1));
///
/// let distances = shortest_distance(&fst)?;
/// // distances[s1] = 1.0 (minimum of two paths)
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Probability Semiring (Sum of Paths)
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<ProbabilityWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// fst.set_start(s0);
/// fst.set_final(s1, ProbabilityWeight::one());
///
/// // Two paths - probability sums them
/// fst.add_arc(s0, Arc::new(1, 1, ProbabilityWeight::new(0.3), s1));
/// fst.add_arc(s0, Arc::new(2, 2, ProbabilityWeight::new(0.4), s1));
///
/// let distances = shortest_distance(&fst)?;
/// // distances[s1] = 0.7 (sum: 0.3 + 0.4)
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns error if:
/// - FST has no start state
/// - Cyclic FST fails to converge within max iterations (may indicate non-k-closed semiring)
/// - Memory allocation fails
pub fn shortest_distance<W, F>(fst: &F) -> Result<Vec<W>>
where
    W: Semiring + Clone + PartialEq,
    F: Fst<W>,
{
    let start = fst
        .start()
        .ok_or_else(|| Error::Algorithm("FST has no start state".into()))?;

    let num_states = fst.num_states();

    // Try acyclic case first (more efficient)
    if let Ok(topo_order) = compute_topo_order(fst) {
        shortest_distance_acyclic(fst, start, &topo_order)
    } else {
        // Cyclic case - use iterative relaxation
        shortest_distance_cyclic(fst, start, num_states)
    }
}

/// Compute topological ordering using DFS
///
/// # Complexity
/// - Time: O(|V| + |E|)
/// - Space: O(|V|)
fn compute_topo_order<W: Semiring, F: Fst<W>>(fst: &F) -> Result<Vec<StateId>> {
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
                return Err(Error::Algorithm("FST has cycles".into()));
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

/// Compute shortest distance for acyclic FSTs using topological ordering
///
/// # Complexity
/// - Time: O(|V| + |E|)
///   - Process each state once: O(|V|)
///   - Process each arc once: O(|E|)
/// - Space: O(|V|)
fn shortest_distance_acyclic<W, F>(
    fst: &F,
    start: StateId,
    topo_order: &[StateId],
) -> Result<Vec<W>>
where
    W: Semiring + Clone,
    F: Fst<W>,
{
    let num_states = fst.num_states();
    let mut distance = vec![W::zero(); num_states];
    distance[start as usize] = W::one();

    // Process states in topological order
    for &state in topo_order {
        let dist = distance[state as usize].clone();

        // Skip if distance is zero (unreachable)
        if Semiring::is_zero(&dist) {
            continue;
        }

        // Relax all outgoing arcs
        for arc in fst.arcs(state) {
            let new_dist = dist.times(&arc.weight);
            distance[arc.nextstate as usize] =
                distance[arc.nextstate as usize].plus(&new_dist);
        }
    }

    Ok(distance)
}

/// Compute shortest distance for cyclic FSTs using iterative relaxation
///
/// # Complexity
/// - Time: O(k × (|V| + |E|)) where k = number of iterations
///   - Each iteration: O(|V| + |E|)
///   - Convergence check: O(|V|)
/// - Space: O(|V|)
fn shortest_distance_cyclic<W, F>(
    fst: &F,
    start: StateId,
    num_states: usize,
) -> Result<Vec<W>>
where
    W: Semiring + Clone + PartialEq,
    F: Fst<W>,
{
    let mut distance = vec![W::zero(); num_states];
    distance[start as usize] = W::one();

    // Iterative relaxation until convergence
    for iteration in 0..MAX_ITERATIONS {
        let mut changed = false;
        let old_distance = distance.clone();

        // Relax all arcs
        for state in 0..num_states as StateId {
            let dist = distance[state as usize].clone();

            if Semiring::is_zero(&dist) {
                continue;
            }

            for arc in fst.arcs(state) {
                let new_dist = dist.times(&arc.weight);
                let next_idx = arc.nextstate as usize;
                let updated = distance[next_idx].plus(&new_dist);

                if updated != distance[next_idx] {
                    distance[next_idx] = updated;
                    changed = true;
                }
            }
        }

        // Check convergence
        if !changed {
            return Ok(distance);
        }

        // Additional convergence check: compare with previous iteration
        if iteration > 0 && distances_converged(&distance, &old_distance) {
            return Ok(distance);
        }
    }

    Err(Error::Algorithm(
        format!(
            "Shortest distance failed to converge after {} iterations (FST may be cyclic with non-k-closed semiring)",
            MAX_ITERATIONS
        )
    ))
}

/// Check if distances have converged between iterations
fn distances_converged<W: Semiring + Clone + PartialEq>(
    current: &[W],
    previous: &[W],
) -> bool {
    current.iter().zip(previous.iter()).all(|(c, p)| c == p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use num_traits::One;

    #[test]
    fn test_acyclic_linear_chain() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::new(0.5));

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(2.0), s2));

        let distances = shortest_distance(&fst).unwrap();

        assert_eq!(distances[s0 as usize], TropicalWeight::one());
        assert_eq!(distances[s1 as usize], TropicalWeight::new(1.0));
        assert_eq!(distances[s2 as usize], TropicalWeight::new(3.0));
    }

    #[test]
    fn test_acyclic_branching() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        // Two paths - tropical takes minimum
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s1));

        let distances = shortest_distance(&fst).unwrap();

        assert_eq!(distances[s0 as usize], TropicalWeight::one());
        assert_eq!(distances[s1 as usize], TropicalWeight::new(1.0)); // min(1.0, 2.0)
    }

    #[test]
    fn test_cyclic_simple_loop() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        // Create cycle: s0 -> s1 -> s0 with increasing weight
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(10.0), s0)); // High weight - won't improve

        let distances = shortest_distance(&fst).unwrap();

        assert_eq!(distances[s0 as usize], TropicalWeight::one());
        assert_eq!(distances[s1 as usize], TropicalWeight::new(1.0));
    }

    #[test]
    fn test_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();
        let result = shortest_distance(&fst);
        assert!(result.is_err()); // No start state
    }

    #[test]
    fn test_no_start_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        fst.add_state();
        // Don't set start state

        let result = shortest_distance(&fst);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::new(2.0));

        let distances = shortest_distance(&fst).unwrap();

        assert_eq!(distances[s0 as usize], TropicalWeight::one());
        assert_eq!(distances.len(), 1);
    }

    #[test]
    fn test_tropical_semiring() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(3.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(5.0), s1));

        let distances = shortest_distance(&fst).unwrap();

        // Tropical plus is min
        assert_eq!(distances[s1 as usize], TropicalWeight::new(3.0));
    }

    #[test]
    fn test_log_semiring() {
        let mut fst = VectorFst::<LogWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        fst.add_arc(s0, Arc::new(1, 1, LogWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, LogWeight::new(2.0), s1));

        let distances = shortest_distance(&fst).unwrap();

        assert_eq!(distances[s0 as usize], LogWeight::one());
        // Log semiring does log-add-exp
        assert!(distances[s1 as usize].value() < &2.0);
    }

    #[test]
    fn test_boolean_semiring() {
        let mut fst = VectorFst::<BooleanWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);

        fst.add_arc(s0, Arc::new(1, 1, BooleanWeight::one(), s1));

        let distances = shortest_distance(&fst).unwrap();

        assert_eq!(distances[s0 as usize], BooleanWeight::one());
        assert_eq!(distances[s1 as usize], BooleanWeight::one());
    }

    #[test]
    fn test_with_epsilon_transitions() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);

        fst.add_arc(s0, Arc::epsilon(TropicalWeight::new(0.5), s1));
        fst.add_arc(s1, Arc::new(1, 1, TropicalWeight::new(1.0), s2));

        let distances = shortest_distance(&fst).unwrap();

        assert_eq!(distances[s0 as usize], TropicalWeight::one());
        assert_eq!(distances[s1 as usize], TropicalWeight::new(0.5));
        assert_eq!(distances[s2 as usize], TropicalWeight::new(1.5));
    }

    #[test]
    fn test_convergence_cyclic() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);

        // Self-loop
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(5.0), s0));

        let distances = shortest_distance(&fst).unwrap();

        // Should converge to distance = 0.0 (start state)
        assert_eq!(distances[s0 as usize], TropicalWeight::one());
    }

    #[test]
    fn test_multiple_paths_diamond() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();

        fst.set_start(s0);

        // Diamond: s0 -> {s1, s2} -> s3
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(3.0), s2));
        fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::new(2.0), s3));
        fst.add_arc(s2, Arc::new(4, 4, TropicalWeight::new(1.0), s3));

        let distances = shortest_distance(&fst).unwrap();

        assert_eq!(distances[s0 as usize], TropicalWeight::one());
        assert_eq!(distances[s1 as usize], TropicalWeight::new(1.0));
        assert_eq!(distances[s2 as usize], TropicalWeight::new(3.0));
        // s3: min(1.0 + 2.0, 3.0 + 1.0) = min(3.0, 4.0) = 3.0
        assert_eq!(distances[s3 as usize], TropicalWeight::new(3.0));
    }
}
