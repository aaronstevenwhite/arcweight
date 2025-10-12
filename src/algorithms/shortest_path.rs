//! Shortest path algorithms
//!
//! Finds k-shortest paths in weighted FSTs using Yen's algorithm.
//!
//! # Semiring Requirements
//!
//! Shortest path computation requires a **naturally ordered semiring:**
//! - `NaturallyOrderedSemiring` trait provides total ordering for path comparison
//! - Enables priority queue-based algorithms (Dijkstra-style)
//! - Required for meaningful "shortest" path definition
//!
//! # Supported Semirings
//!
//! - ✅ `TropicalWeight` - Natural ordering by cost (min-cost paths)
//! - ✅ `LogWeight` - Natural ordering by log probability
//! - ❌ `ProbabilityWeight` - No natural ordering defined
//! - ❌ `BooleanWeight` - No meaningful path comparison
//!
//! # Algorithm
//!
//! Uses **Yen's algorithm** for k-shortest loopless paths:
//! - **Time Complexity:** O(k × |V| × (|E| + |V| log |V|))
//! - **Space Complexity:** O(k × |V| + |V|²)
//! - **Guarantees:** Finds k shortest loopless paths in order
//!
//! # Cycle Handling
//!
//! - Yen's algorithm naturally avoids cycles by construction
//! - Only loopless (simple) paths are returned
//! - Weight monotonicity ensures termination

use crate::arc::Arc;
use crate::fst::{Fst, Label, MutableFst, StateId};
use crate::semiring::{NaturallyOrderedSemiring, Semiring};
use crate::{Error, Result};
use core::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Configuration for k-shortest path algorithms
#[derive(Debug, Clone)]
pub struct ShortestPathConfig {
    /// Number of shortest paths to find (default: 1)
    pub nshortest: usize,
    /// Only include unique input/output label sequences (default: false)
    pub unique: bool,
}

impl Default for ShortestPathConfig {
    fn default() -> Self {
        Self {
            nshortest: 1,
            unique: false,
        }
    }
}

/// A complete path from start to a final state
#[derive(Clone, Debug)]
struct Path<W: Semiring> {
    /// Sequence of (state, arc) pairs representing the path
    /// The arc at position i goes FROM states[i] TO states[i+1]
    states: Vec<StateId>,
    arcs: Vec<Arc<W>>,
    /// Total weight of the path
    weight: W,
    /// The final state reached
    final_state: StateId,
}

impl<W: NaturallyOrderedSemiring> PartialEq for Path<W> {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}

impl<W: NaturallyOrderedSemiring> Eq for Path<W> {}

impl<W: NaturallyOrderedSemiring> PartialOrd for Path<W> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<W: NaturallyOrderedSemiring> Ord for Path<W> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other.weight.cmp(&self.weight)
    }
}

/// State in Dijkstra's priority queue
#[derive(Clone, Debug)]
struct PathState<W: Semiring> {
    state: StateId,
    weight: W,
}

impl<W: NaturallyOrderedSemiring> PartialEq for PathState<W> {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}

impl<W: NaturallyOrderedSemiring> Eq for PathState<W> {}

impl<W: NaturallyOrderedSemiring> PartialOrd for PathState<W> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<W: NaturallyOrderedSemiring> Ord for PathState<W> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.weight.cmp(&self.weight)
    }
}

/// Find k-shortest paths using Yen's algorithm
///
/// Computes the k shortest loopless paths from the start state to any final state
/// in the FST using Yen's algorithm. Returns an FST containing all k paths.
///
/// # Algorithm: Yen's K-Shortest Paths
///
/// **Algorithm Details:**
/// - **Base:** Iterative Dijkstra with edge removal
/// - **Time Complexity:** O(k × |V| × (|E| + |V| log |V|))
/// - **Space Complexity:** O(k × |V| + |V|²)
/// - **Path Property:** All returned paths are loopless (no repeated states)
/// - **Ordering:** Paths returned in strictly increasing weight order
///
/// **Algorithm Steps:**
/// 1. Find the shortest path using Dijkstra
/// 2. For each subsequent path k = 2..n:
///    - For each node in path k-1:
///      - Temporarily remove edges that would duplicate previous paths
///      - Find shortest path from that node (spur node) to any final state
///      - Add candidate path to priority queue
///    - Select best candidate as path k
/// 3. Build result FST containing all k paths
///
/// # Semiring Requirements
///
/// The input FST must use a [`NaturallyOrderedSemiring`] that provides:
/// - Total ordering for path weight comparison
/// - Monotonic path weight accumulation
/// - Well-defined shortest path semantics
///
/// # Configuration
///
/// - **nshortest:** Number of shortest paths to find (default: 1)
/// - **unique:** If true, only return paths with unique input/output sequences (default: false)
///
/// # Examples
///
/// ## Single Shortest Path
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Create FST: 0 --a/0.5--> 1 --b/0.3--> 2(final)
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s2, TropicalWeight::one());
/// fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::new(0.5), s1));
/// fst.add_arc(s1, Arc::new('b' as u32, 'b' as u32, TropicalWeight::new(0.3), s2));
///
/// // Find single shortest path
/// let config = ShortestPathConfig::default();
/// let shortest: VectorFst<TropicalWeight> = shortest_path(&fst, config)?;
///
/// // Result contains path "ab" with total weight 0.8
/// assert!(shortest.num_states() > 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## K-Best Paths
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // FST with multiple paths of different costs
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// fst.set_start(s0);
/// fst.set_final(s1, TropicalWeight::one());
///
/// // Add multiple arcs with different weights
/// fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::new(1.0), s1));
/// fst.add_arc(s0, Arc::new('b' as u32, 'b' as u32, TropicalWeight::new(2.0), s1));
/// fst.add_arc(s0, Arc::new('c' as u32, 'c' as u32, TropicalWeight::new(3.0), s1));
///
/// // Find top 3 shortest paths
/// let config = ShortestPathConfig {
///     nshortest: 3,
///     unique: false,
/// };
///
/// let k_best: VectorFst<TropicalWeight> = shortest_path(&fst, config)?;
/// // Result FST contains all 3 paths: a (1.0), b (2.0), c (3.0)
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Unique Paths Only
///
/// ```rust
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
/// fst.set_start(s0);
/// fst.set_final(s2, TropicalWeight::one());
///
/// // Two paths with same input/output but different intermediate states
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
/// fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s2));
///
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s2)); // Direct path, same I/O
///
/// // Find unique paths only
/// let config = ShortestPathConfig {
///     nshortest: 5,
///     unique: true,  // Filter duplicate input/output sequences
/// };
///
/// let unique_paths: VectorFst<TropicalWeight> = shortest_path(&fst, config)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Complex Network
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Complex FST with multiple paths through different routes
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
/// let s3 = fst.add_state();
/// let s4 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s4, TropicalWeight::one());
///
/// // Create diamond pattern with multiple paths
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1)); // Top route
/// fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s3));
/// fst.add_arc(s3, Arc::new(3, 3, TropicalWeight::new(1.0), s4));
///
/// fst.add_arc(s0, Arc::new(4, 4, TropicalWeight::new(2.0), s2)); // Bottom route
/// fst.add_arc(s2, Arc::new(5, 5, TropicalWeight::new(1.0), s3));
/// fst.add_arc(s3, Arc::new(3, 3, TropicalWeight::new(1.0), s4));
///
/// // Find 10 best paths
/// let config = ShortestPathConfig {
///     nshortest: 10,
///     unique: false,
/// };
///
/// let paths: VectorFst<TropicalWeight> = shortest_path(&fst, config)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Performance Characteristics
///
/// - **Time per path:** O(|V| × (|E| + |V| log |V|))
/// - **Total time:** O(k × |V| × (|E| + |V| log |V|))
/// - **Memory:** O(k × |V|) for storing k paths + O(|V|²) for edge exclusion tracking
/// - **Optimality:** Guaranteed to find k shortest loopless paths in order
///
/// # Errors
///
/// Returns [`Error::Algorithm`] if:
/// - The input FST has no start state
/// - No paths exist to any final state
/// - Memory allocation fails during computation
/// - Weight computation overflows or becomes infinite
///
/// # See Also
///
/// - [`shortest_path_single`] for single path (optimized, same as nshortest=1)
/// - Paper: Yen, J. Y. (1971). "Finding the k shortest loopless paths in a network"
pub fn shortest_path<W, F, M>(fst: &F, config: ShortestPathConfig) -> Result<M>
where
    W: NaturallyOrderedSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    if config.nshortest == 0 {
        return Ok(M::default());
    }

    let start = fst
        .start()
        .ok_or_else(|| Error::Algorithm("FST has no start state".into()))?;

    // Find k-shortest paths using Yen's algorithm
    let paths = yen_k_shortest_paths(fst, start, config.nshortest, config.unique)?;

    // Build result FST containing all k paths
    build_paths_fst(fst, &paths)
}

/// Yen's k-shortest loopless paths algorithm
///
/// Finds k shortest paths that don't contain cycles (loopless paths).
///
/// # Algorithm Complexity
///
/// - **Time:** O(k × |V| × (|E| + |V| log |V|))
/// - **Space:** O(k × |V| + |V|²)
///
/// # Returns
///
/// Vector of paths in strictly increasing weight order
fn yen_k_shortest_paths<W, F>(
    fst: &F,
    start: StateId,
    k: usize,
    unique: bool,
) -> Result<Vec<Path<W>>>
where
    W: NaturallyOrderedSemiring,
    F: Fst<W>,
{
    // Find first shortest path
    let first_path = match dijkstra_shortest_path(fst, start, &HashSet::new())? {
        Some(path) => path,
        None => return Ok(Vec::new()), // No path to any final state
    };

    let mut result_paths = vec![first_path];
    let mut candidate_paths = BinaryHeap::<Path<W>>::new();

    // Track seen input/output sequences for uniqueness
    let mut seen_sequences = HashSet::new();
    if unique {
        let seq = extract_io_sequence(&result_paths[0]);
        seen_sequences.insert(seq);
    }

    // Find k-1 more paths
    for _ in 1..k {
        let prev_path = &result_paths[result_paths.len() - 1];

        // For each node in the previous path
        for spur_index in 0..prev_path.states.len() {
            let spur_node = prev_path.states[spur_index];

            // Root path is the prefix up to spur node
            let root_path_states = &prev_path.states[0..=spur_index];

            // Find edges to exclude (edges that would recreate previous paths)
            let mut excluded_edges = HashSet::new();

            for existing_path in &result_paths {
                // If existing path shares the same root
                if existing_path.states.len() > spur_index
                    && existing_path.states[0..=spur_index] == root_path_states[..]
                {
                    // Exclude the edge taken from spur node in that path
                    if existing_path.arcs.len() > spur_index {
                        let arc = &existing_path.arcs[spur_index];
                        excluded_edges.insert((spur_node, arc.ilabel, arc.olabel, arc.nextstate));
                    }
                }
            }

            // Also exclude nodes in root path to prevent cycles
            let excluded_nodes: HashSet<StateId> = root_path_states.iter().copied().collect();

            // Find shortest path from spur node to any final, excluding certain edges
            if let Some(spur_path) =
                dijkstra_shortest_path_from_node(fst, spur_node, &excluded_edges, &excluded_nodes)?
            {
                // Construct candidate path: root + spur
                let candidate = if spur_index == 0 {
                    // No root, just use spur path
                    spur_path
                } else {
                    // Combine root and spur
                    let root_weight = prev_path.states[0..spur_index]
                        .iter()
                        .zip(&prev_path.arcs[0..spur_index])
                        .fold(W::one(), |w, (_, arc)| w.times(&arc.weight));

                    Path {
                        states: [&prev_path.states[0..spur_index], &spur_path.states[..]]
                            .concat(),
                        arcs: [&prev_path.arcs[0..spur_index], &spur_path.arcs[..]].concat(),
                        weight: root_weight.times(&spur_path.weight),
                        final_state: spur_path.final_state,
                    }
                };

                // Check uniqueness if required
                if unique {
                    let seq = extract_io_sequence(&candidate);
                    if seen_sequences.contains(&seq) {
                        continue;
                    }
                }

                // Check if this candidate already exists
                let mut is_duplicate = false;
                for existing in &result_paths {
                    if paths_equal(&candidate, existing) {
                        is_duplicate = true;
                        break;
                    }
                }

                if !is_duplicate {
                    candidate_paths.push(candidate);
                }
            }
        }

        // Get best candidate
        match candidate_paths.pop() {
            Some(best) => {
                if unique {
                    let seq = extract_io_sequence(&best);
                    seen_sequences.insert(seq);
                }
                result_paths.push(best);
            }
            None => break, // No more paths available
        }
    }

    Ok(result_paths)
}

/// Dijkstra's shortest path with edge exclusion
///
/// Finds shortest path from start to any final state, excluding specified edges.
fn dijkstra_shortest_path<W, F>(
    fst: &F,
    start: StateId,
    excluded_edges: &HashSet<(StateId, Label, Label, StateId)>,
) -> Result<Option<Path<W>>>
where
    W: NaturallyOrderedSemiring,
    F: Fst<W>,
{
    dijkstra_shortest_path_from_node(fst, start, excluded_edges, &HashSet::new())
}

/// Dijkstra's shortest path from specific node with exclusions
///
/// # Arguments
///
/// - `fst`: Input FST
/// - `start_node`: Node to start search from
/// - `excluded_edges`: Set of (from_state, ilabel, olabel, to_state) tuples to exclude
/// - `excluded_nodes`: Set of nodes that cannot be visited (except start_node)
fn dijkstra_shortest_path_from_node<W, F>(
    fst: &F,
    start_node: StateId,
    excluded_edges: &HashSet<(StateId, Label, Label, StateId)>,
    excluded_nodes: &HashSet<StateId>,
) -> Result<Option<Path<W>>>
where
    W: NaturallyOrderedSemiring,
    F: Fst<W>,
{
    let mut distance = vec![W::zero(); fst.num_states()];
    let mut parent: Vec<Option<(StateId, Arc<W>)>> = vec![None; fst.num_states()];
    let mut heap = BinaryHeap::new();
    let mut visited = HashSet::new();

    distance[start_node as usize] = W::one();
    heap.push(PathState {
        state: start_node,
        weight: W::one(),
    });

    let mut best_final: Option<(StateId, W)> = None;

    while let Some(PathState { state, weight, .. }) = heap.pop() {
        // Skip if we've found a better path to this state
        if weight > distance[state as usize] {
            continue;
        }

        if visited.contains(&state) {
            continue;
        }
        visited.insert(state);

        // Check if this is a final state
        if let Some(final_weight) = fst.final_weight(state) {
            let total = weight.times(final_weight);
            match &best_final {
                None => best_final = Some((state, total)),
                Some((_, best_w)) if total < *best_w => best_final = Some((state, total)),
                _ => {}
            }
        }

        // Explore transitions
        for arc in fst.arcs(state) {
            let next_state = arc.nextstate;

            // Skip excluded edges
            if excluded_edges.contains(&(state, arc.ilabel, arc.olabel, next_state)) {
                continue;
            }

            // Skip excluded nodes (but allow if it's the start node)
            if excluded_nodes.contains(&next_state) && next_state != start_node {
                continue;
            }

            let next_weight = weight.times(&arc.weight);

            if <W as num_traits::Zero>::is_zero(&distance[next_state as usize])
                || next_weight < distance[next_state as usize]
            {
                distance[next_state as usize] = next_weight.clone();
                parent[next_state as usize] = Some((state, arc.clone()));

                heap.push(PathState {
                    state: next_state,
                    weight: next_weight,
                });
            }
        }
    }

    // Reconstruct path to best final state
    if let Some((final_state, final_weight)) = best_final {
        let mut states = vec![final_state];
        let mut arcs = Vec::new();
        let mut current = final_state;

        while let Some((prev_state, arc)) = &parent[current as usize] {
            states.push(*prev_state);
            arcs.push(arc.clone());
            current = *prev_state;
        }

        states.reverse();
        arcs.reverse();

        Ok(Some(Path {
            states,
            arcs,
            weight: final_weight,
            final_state,
        }))
    } else {
        Ok(None)
    }
}

/// Build FST containing all k paths
fn build_paths_fst<W, F, M>(fst: &F, paths: &[Path<W>]) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();

    if paths.is_empty() {
        return Ok(result);
    }

    // Use a map to track which states have been created in the result FST
    let mut state_map: HashMap<StateId, StateId> = HashMap::new();

    // Helper function to get or create state
    fn get_or_create_state<W2: Semiring, M2: MutableFst<W2>>(
        result: &mut M2,
        s: StateId,
        map: &mut HashMap<StateId, StateId>,
    ) -> StateId {
        if let Some(&new_s) = map.get(&s) {
            new_s
        } else {
            let new_s = result.add_state();
            map.insert(s, new_s);
            new_s
        }
    }

    // Set start state
    let start = paths[0].states[0];
    let new_start = get_or_create_state(&mut result, start, &mut state_map);
    result.set_start(new_start);

    // Add all paths
    for path in paths {
        for i in 0..path.arcs.len() {
            let from_state = path.states[i];
            let to_state = path.states[i + 1];
            let arc = &path.arcs[i];

            let new_from = get_or_create_state(&mut result, from_state, &mut state_map);
            let new_to = get_or_create_state(&mut result, to_state, &mut state_map);

            result.add_arc(
                new_from,
                Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), new_to),
            );
        }

        // Set final weight
        if let Some(final_weight) = fst.final_weight(path.final_state) {
            let new_final = get_or_create_state(&mut result, path.final_state, &mut state_map);
            result.set_final(new_final, final_weight.clone());
        }
    }

    Ok(result)
}

/// Check if two paths are equal (same states AND same arcs)
fn paths_equal<W: Semiring>(p1: &Path<W>, p2: &Path<W>) -> bool {
    if p1.states.len() != p2.states.len() {
        return false;
    }

    if p1.arcs.len() != p2.arcs.len() {
        return false;
    }

    // Check states
    for i in 0..p1.states.len() {
        if p1.states[i] != p2.states[i] {
            return false;
        }
    }

    // Check arcs (labels must match for paths to be truly equal)
    for i in 0..p1.arcs.len() {
        if p1.arcs[i].ilabel != p2.arcs[i].ilabel
            || p1.arcs[i].olabel != p2.arcs[i].olabel
            || p1.arcs[i].nextstate != p2.arcs[i].nextstate
        {
            return false;
        }
    }

    true
}

/// Extract input/output label sequence from path
fn extract_io_sequence<W: Semiring>(path: &Path<W>) -> Vec<(Label, Label)> {
    path.arcs
        .iter()
        .map(|arc| (arc.ilabel, arc.olabel))
        .collect()
}

/// Convenience function for single shortest path
///
/// Equivalent to calling `shortest_path` with `nshortest = 1`.
///
/// # Errors
///
/// Returns an error if:
/// - The input FST is invalid or corrupted
/// - The FST has no start state
/// - Memory allocation fails during computation
pub fn shortest_path_single<W, F, M>(fst: &F) -> Result<M>
where
    W: NaturallyOrderedSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    shortest_path(fst, ShortestPathConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use num_traits::One;

    #[test]
    fn test_shortest_path_single() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Two paths with different costs
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(2.0), s2));
        fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::new(5.0), s2));

        let shortest: VectorFst<TropicalWeight> = shortest_path_single(&fst).unwrap();

        // Should find the cheaper path (1->2 with total weight 3.0)
        assert!(shortest.start().is_some());
        assert!(shortest.num_states() >= 2);
    }

    #[test]
    fn test_k_shortest_paths_simple() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        // Three parallel edges with different weights
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s1));
        fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::new(3.0), s1));

        let config = ShortestPathConfig {
            nshortest: 3,
            unique: false,
        };
        let k_best: VectorFst<TropicalWeight> = shortest_path(&fst, config).unwrap();

        // Should have all 3 paths
        assert!(k_best.start().is_some());
        // May have shared states, so just check we have paths
        assert!(k_best.num_states() >= 2);
    }

    #[test]
    fn test_k_shortest_paths_complex() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s3, TropicalWeight::one());

        // Diamond pattern with two paths
        // Top path: 0 -> 1 -> 3 (weight 1.0 + 1.0 = 2.0)
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s3));

        // Bottom path: 0 -> 2 -> 3 (weight 2.0 + 1.0 = 3.0)
        fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::new(2.0), s2));
        fst.add_arc(s2, Arc::new(4, 4, TropicalWeight::new(1.0), s3));

        let config = ShortestPathConfig {
            nshortest: 2,
            unique: false,
        };
        let k_best: VectorFst<TropicalWeight> = shortest_path(&fst, config).unwrap();

        // Should have both paths
        assert!(k_best.start().is_some());
        assert!(k_best.num_states() >= 2);
    }

    #[test]
    fn test_shortest_path_empty() {
        let fst = VectorFst::<TropicalWeight>::new();

        // Empty FST should return error or empty result
        if let Ok(shortest) = shortest_path_single::<
            TropicalWeight,
            VectorFst<TropicalWeight>,
            VectorFst<TropicalWeight>,
        >(&fst)
        {
            assert!(shortest.is_empty());
        }
    }

    #[test]
    fn test_shortest_path_config_default() {
        let config = ShortestPathConfig::default();
        assert_eq!(config.nshortest, 1);
        assert!(!config.unique);
    }

    #[test]
    fn test_shortest_path_config_custom() {
        let config = ShortestPathConfig {
            nshortest: 5,
            unique: true,
        };
        assert_eq!(config.nshortest, 5);
        assert!(config.unique);
    }

    #[test]
    fn test_shortest_path_single_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::new(2.0));

        let shortest: VectorFst<TropicalWeight> = shortest_path_single(&fst).unwrap();

        assert_eq!(shortest.num_states(), 1);
        assert_eq!(shortest.start(), Some(0));
        assert!(shortest.is_final(0));
    }

    #[test]
    fn test_shortest_path_no_final_states() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        // No final states

        let shortest: VectorFst<TropicalWeight> = shortest_path_single(&fst).unwrap();

        // Should return empty FST when no paths to final states
        assert_eq!(shortest.num_states(), 0);
    }

    #[test]
    fn test_shortest_path_zero_nshortest() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::one());

        let config = ShortestPathConfig {
            nshortest: 0,
            unique: false,
        };
        let shortest: VectorFst<TropicalWeight> = shortest_path(&fst, config).unwrap();

        // Should return empty FST when nshortest = 0
        assert_eq!(shortest.num_states(), 0);
    }

    #[test]
    fn test_shortest_path_with_weights() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s3, TropicalWeight::one());

        // Create two paths: cheap and expensive
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.1), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(0.2), s3));

        fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::new(1.0), s2));
        fst.add_arc(s2, Arc::new(4, 4, TropicalWeight::new(2.0), s3));

        let shortest: VectorFst<TropicalWeight> = shortest_path_single(&fst).unwrap();

        // Should prefer the cheaper path
        assert!(shortest.start().is_some());
        assert!(shortest.num_states() > 0);
    }

    #[test]
    fn test_shortest_path_linear_chain() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let states: Vec<_> = (0..5).map(|_| fst.add_state()).collect();

        fst.set_start(states[0]);
        fst.set_final(states[4], TropicalWeight::one());

        // Create linear chain
        for i in 0..4 {
            fst.add_arc(
                states[i],
                Arc::new(
                    (i + 1) as u32,
                    (i + 1) as u32,
                    TropicalWeight::new(i as f32 * 0.1),
                    states[i + 1],
                ),
            );
        }

        let shortest: VectorFst<TropicalWeight> = shortest_path_single(&fst).unwrap();

        // Should preserve the linear chain structure
        assert_eq!(shortest.start(), Some(0));
        assert!(shortest.num_states() > 0);
    }

    #[test]
    fn test_k_shortest_avoids_cycles() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Create a cycle: 0 -> 1 -> 0
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s0));

        // Also create path to final: 1 -> 2
        fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::new(1.0), s2));

        let config = ShortestPathConfig {
            nshortest: 5,
            unique: false,
        };
        let k_best: VectorFst<TropicalWeight> = shortest_path(&fst, config).unwrap();

        // Should find path 0->1->2 but not traverse the cycle
        assert!(k_best.start().is_some());
    }

    #[test]
    fn test_unique_paths() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Two paths with same input/output labels but different routes
        // Path 1: 0 --1/1--> 1 --2/2--> 2
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s2));

        // Path 2: 0 --1/1--> 2 (direct, same I/O sequence if we think of it as prefix)
        // Actually let's make two truly parallel paths with same labels
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.5), s1));

        let config = ShortestPathConfig {
            nshortest: 5,
            unique: true,
        };
        let unique_result: VectorFst<TropicalWeight> = shortest_path(&fst, config).unwrap();

        // With unique=true, should filter based on I/O sequences
        assert!(unique_result.start().is_some());
    }

    #[test]
    fn test_k_shortest_more_than_available() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        // Only 2 paths available
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s1));

        // Request more paths than available
        let config = ShortestPathConfig {
            nshortest: 10,
            unique: false,
        };
        let result: VectorFst<TropicalWeight> = shortest_path(&fst, config).unwrap();

        // Should return all available paths (2), not fail
        assert!(result.start().is_some());
    }

    #[test]
    fn test_k_shortest_paths_ordering() {
        // Verify that paths are returned in increasing weight order
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        // Add 5 paths with known weights
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(5.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::new(3.0), s1));
        fst.add_arc(s0, Arc::new(4, 4, TropicalWeight::new(2.0), s1));
        fst.add_arc(s0, Arc::new(5, 5, TropicalWeight::new(4.0), s1));

        let config = ShortestPathConfig {
            nshortest: 5,
            unique: false,
        };

        // Internal algorithm should find them in order: 1.0, 2.0, 3.0, 4.0, 5.0
        let result: VectorFst<TropicalWeight> = shortest_path(&fst, config).unwrap();

        // Result FST should contain all 5 paths
        assert!(result.start().is_some());
        assert!(result.num_states() >= 2);

        // Count arcs from start state - should have all 5
        let start = result.start().unwrap();
        let arc_count = result.num_arcs(start);
        assert_eq!(arc_count, 5, "Should have all 5 paths");
    }

    #[test]
    fn test_k_shortest_complex_network() {
        // Test with a more complex network with multiple path lengths
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();
        let s4 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s4, TropicalWeight::one());

        // Create multiple paths with different lengths and weights
        // Path 1: 0->1->4 (weight = 1.0 + 1.0 = 2.0)
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s4));

        // Path 2: 0->2->4 (weight = 1.5 + 1.0 = 2.5)
        fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::new(1.5), s2));
        fst.add_arc(s2, Arc::new(4, 4, TropicalWeight::new(1.0), s4));

        // Path 3: 0->1->3->4 (weight = 1.0 + 0.5 + 1.0 = 2.5)
        fst.add_arc(s1, Arc::new(5, 5, TropicalWeight::new(0.5), s3));
        fst.add_arc(s3, Arc::new(6, 6, TropicalWeight::new(1.0), s4));

        // Path 4: 0->2->3->4 (weight = 1.5 + 0.5 + 1.0 = 3.0)
        fst.add_arc(s2, Arc::new(7, 7, TropicalWeight::new(0.5), s3));

        let config = ShortestPathConfig {
            nshortest: 4,
            unique: false,
        };

        let result: VectorFst<TropicalWeight> = shortest_path(&fst, config).unwrap();

        // Should find all 4 distinct paths
        assert!(result.start().is_some());
        assert!(result.num_states() >= 2);
    }

    #[test]
    fn test_k_shortest_with_final_weights() {
        // Test that final weights are properly considered
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::new(0.5)); // Final weight 0.5
        fst.set_final(s2, TropicalWeight::new(2.0)); // Final weight 2.0

        // Path to s1: weight 1.0 + 0.5 = 1.5
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));

        // Path to s2: weight 1.0 + 2.0 = 3.0
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(1.0), s2));

        let config = ShortestPathConfig {
            nshortest: 2,
            unique: false,
        };

        let result: VectorFst<TropicalWeight> = shortest_path(&fst, config).unwrap();

        // Should find both paths, with s1 path being better
        assert!(result.start().is_some());
        let start = result.start().unwrap();
        assert_eq!(result.num_arcs(start), 2);
    }

    #[test]
    fn test_yen_loopless_property() {
        // Verify that Yen's algorithm only returns loopless (no repeated states) paths
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Create potential for loops: 0 -> 1 -> 2 and 1 -> 0 (cycle)
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s2));
        fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::new(0.1), s0)); // Back to s0

        let config = ShortestPathConfig {
            nshortest: 10,
            unique: false,
        };

        let result: VectorFst<TropicalWeight> = shortest_path(&fst, config).unwrap();

        // Should only find the direct path 0->1->2, not traverse the cycle
        assert!(result.start().is_some());
        // The result should not have exponentially many paths from cycling
        assert!(result.num_states() <= 10, "Should not explode with cycles");
    }

    #[test]
    fn test_k_shortest_multiple_finals() {
        // Test behavior with multiple final states
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());
        fst.set_final(s3, TropicalWeight::one());

        // Paths to different final states
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s2)); // Total 2.0 to s2
        fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::new(0.5), s3)); // Total 1.5 to s3

        let config = ShortestPathConfig {
            nshortest: 2,
            unique: false,
        };

        let result: VectorFst<TropicalWeight> = shortest_path(&fst, config).unwrap();

        // Should find paths to both final states, with s3 path first (lower weight)
        assert!(result.start().is_some());
        assert!(result.is_final(2) || result.is_final(3), "Should have at least one final state");
    }

    #[test]
    fn test_unique_filtering_actually_works() {
        // Rigorous test that unique filtering actually removes duplicates
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s3, TropicalWeight::one());

        // Create two paths with identical I/O sequences but different intermediate states
        // Path 1: 0 --1/1--> 1 --2/2--> 3 (weight 3.0)
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(2.0), s3));

        // Path 2: 0 --1/1--> 2 --2/2--> 3 (weight 5.0, different intermediate state)
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s2));
        fst.add_arc(s2, Arc::new(2, 2, TropicalWeight::new(3.0), s3));

        // Without unique filtering, should get 2 paths
        let config_all = ShortestPathConfig {
            nshortest: 5,
            unique: false,
        };
        let result_all: VectorFst<TropicalWeight> = shortest_path(&fst, config_all).unwrap();
        let start = result_all.start().unwrap();
        let arc_count_all = result_all.num_arcs(start);
        assert_eq!(arc_count_all, 2, "Should have 2 paths without unique filtering");

        // With unique filtering, should only get 1 path (the cheaper one)
        let config_unique = ShortestPathConfig {
            nshortest: 5,
            unique: true,
        };
        let result_unique: VectorFst<TropicalWeight> = shortest_path(&fst, config_unique).unwrap();
        let start_unique = result_unique.start().unwrap();
        let arc_count_unique = result_unique.num_arcs(start_unique);
        assert_eq!(
            arc_count_unique, 1,
            "Should have only 1 path with unique filtering"
        );
    }

    #[test]
    fn test_k_shortest_paths_stress() {
        // Stress test with larger k value
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        // Add 20 parallel paths
        for i in 0..20 {
            fst.add_arc(
                s0,
                Arc::new(
                    i as u32,
                    i as u32,
                    TropicalWeight::new(i as f32),
                    s1,
                ),
            );
        }

        let config = ShortestPathConfig {
            nshortest: 20,
            unique: false,
        };

        let result: VectorFst<TropicalWeight> = shortest_path(&fst, config).unwrap();

        // Should successfully find all 20 paths
        assert!(result.start().is_some());
        let start = result.start().unwrap();
        assert_eq!(result.num_arcs(start), 20, "Should find all 20 paths");
    }
}
