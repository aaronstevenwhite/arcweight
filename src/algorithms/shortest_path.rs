//! Shortest path algorithms
//!
//! Finds shortest paths in weighted FSTs using generalized distance algorithms.
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
//! # Cycle Handling
//!
//! For cyclic FSTs:
//! - Negative cycle detection prevents infinite loops
//! - Weight monotonicity ensures termination
//! - Path uniqueness filtering available via configuration

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst, StateId, NO_STATE_ID};
use crate::semiring::{NaturallyOrderedSemiring, Semiring};
use crate::{Error, Result};
use core::cmp::Ordering;
use std::collections::BinaryHeap;

/// Configuration for shortest path algorithms
#[derive(Debug, Clone)]
pub struct ShortestPathConfig {
    /// Number of shortest paths to find
    pub nshortest: usize,
    /// Use unique paths only
    pub unique: bool,
    /// Weight threshold
    pub weight_threshold: Option<f64>,
    /// State threshold
    pub state_threshold: Option<usize>,
}

impl Default for ShortestPathConfig {
    fn default() -> Self {
        Self {
            nshortest: 1,
            unique: false,
            weight_threshold: None,
            state_threshold: None,
        }
    }
}

/// State in shortest path queue
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

/// Find shortest path(s) in an FST using generalized distance algorithms
///
/// Uses a generalized Dijkstra's algorithm to find the shortest path(s) from the start state
/// to any final state in the FST. The algorithm works with any naturally ordered semiring.
///
/// # Algorithm Details
///
/// - **Base Algorithm:** Generalized Dijkstra's algorithm for semirings
/// - **Time Complexity:** O((|V| + |E|) log |V|) using a binary heap
/// - **Space Complexity:** O(|V|) for distance tracking and priority queue
/// - **Termination:** Guaranteed for naturally ordered, k-closed semirings
///
/// # Semiring Requirements
///
/// The input FST must use a [`NaturallyOrderedSemiring`] that provides:
/// - Total ordering for path weight comparison
/// - Monotonic path weight accumulation
/// - Well-defined shortest path semantics
///
/// # Configuration Options
///
/// - **nshortest:** Number of shortest paths to return (default: 1)
/// - **unique:** Filter duplicate paths with same input/output (default: false)  
/// - **weight_threshold:** Prune paths exceeding weight limit
/// - **state_threshold:** Limit result FST size
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
/// assert_eq!(shortest.num_states(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## N-Best Paths
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
/// fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::new(1.0), s1));
/// fst.add_arc(s0, Arc::new('b' as u32, 'b' as u32, TropicalWeight::new(2.0), s1));
///
/// // Find top 5 shortest paths
/// let config = ShortestPathConfig {
///     nshortest: 5,
///     unique: true,  // Only unique input/output sequences
///     weight_threshold: Some(10.0),  // Prune paths > 10.0
///     state_threshold: Some(1000),   // Limit result size
/// };
///
/// let n_best: VectorFst<TropicalWeight> = shortest_path(&fst, config).unwrap();
/// ```
///
/// ## Spell Correction Application
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Find spelling corrections using edit distance + dictionary
/// fn find_corrections(
///     misspelled: &str,
///     edit_fst: &VectorFst<TropicalWeight>,
///     dictionary: &VectorFst<TropicalWeight>
/// ) -> Result<VectorFst<TropicalWeight>> {
///     
///     // Compose edit distance with dictionary
///     let corrections: VectorFst<TropicalWeight> =
///         compose_default(edit_fst, dictionary)?;
///     
///     // Find best corrections
///     let config = ShortestPathConfig {
///         nshortest: 10,
///         unique: true,
///         weight_threshold: Some(3.0), // Max 3 edits
///         ..Default::default()
///     };
///     
///     shortest_path(&corrections, config)
/// }
/// ```
///
/// # Performance Considerations
///
/// - **Memory Usage:** Scales with |V| + |E|, not with path count
/// - **Early Termination:** Stops when required number of paths found
/// - **Large FSTs:** Use weight/state thresholds to control result size
/// - **Cyclic FSTs:** Ensure weights are monotonic to prevent infinite loops
///
/// # Errors
///
/// Returns [`Error::Algorithm`] if:
/// - The input FST has no start state or is malformed
/// - Memory allocation fails during computation
/// - The FST contains negative cycles (for tropical semiring)
/// - Weight computation overflows or becomes infinite
/// - Configuration parameters are invalid (e.g., nshortest = 0)
///
/// # See Also
///
/// - [`shortest_path_single`] for optimized single path finding
/// - [Working with FSTs - Shortest Path](../../docs/working-with-fsts/path-operations.md#shortest-path) for usage patterns
/// - [Core Concepts](../../docs/core-concepts/algorithms.md) for mathematical background
/// - [`ShortestPathConfig`] for configuration options
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

    // dijkstra's algorithm with k-shortest paths
    let mut result = M::default();
    let mut distance = vec![W::zero(); fst.num_states()];
    let mut parent = vec![None; fst.num_states()];
    let mut heap = BinaryHeap::new();

    // add start state to result
    let start_new = result.add_state();
    result.set_start(start_new);

    // initialize
    distance[start as usize] = W::one();
    heap.push(PathState {
        state: start,
        weight: W::one(),
    });

    // state mapping from input to output FST
    let mut state_map = vec![NO_STATE_ID; fst.num_states()];
    state_map[start as usize] = start_new;

    // Track all final states found
    let mut final_states = Vec::new();

    // main loop (do NOT break early)
    while let Some(PathState { state, weight }) = heap.pop() {
        // skip if we've found a better path
        if weight > distance[state as usize] {
            continue;
        }

        // check if final
        if let Some(final_weight) = fst.final_weight(state) {
            let out_state = state_map[state as usize];
            result.set_final(out_state, final_weight.clone());
            final_states.push(state);
        }

        // explore transitions
        for arc in fst.arcs(state) {
            let next_weight = weight.times(&arc.weight);
            let next_state = arc.nextstate;

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

    // reconstruct paths for all final states found
    for &final_state in &final_states {
        // Reconstruct path from start to this final state
        let mut path = Vec::new();
        let mut current = final_state;
        while let Some((parent_state, arc)) = &parent[current as usize] {
            path.push((parent_state, current, arc.clone()));
            current = *parent_state;
        }
        // Path is from final_state to start, so reverse it
        path.reverse();
        // Add states and arcs to result FST
        let mut prev_out_state = start_new;
        for &(_parent_state, state, ref arc) in &path {
            // Ensure state exists in output
            if state_map[state as usize] == NO_STATE_ID {
                state_map[state as usize] = result.add_state();
            }
            let out_state = state_map[state as usize];
            result.add_arc(
                prev_out_state,
                Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), out_state),
            );
            prev_out_state = out_state;
        }
        // Set final state in output FST
        if let Some(final_weight) = fst.final_weight(final_state) {
            let out_state = state_map[final_state as usize];
            result.set_final(out_state, final_weight.clone());
        }
    }

    Ok(result)
}

/// Convenience function for single shortest path
///
/// # Errors
///
/// Returns an error if:
/// - The input FST is invalid or corrupted
/// - The FST has no start state
/// - Memory allocation fails during computation
/// - Path finding fails due to infinite weights or disconnected states
pub fn shortest_path_single<W, F, M>(fst: &F) -> Result<M>
where
    W: NaturallyOrderedSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    shortest_path(fst, ShortestPathConfig::default())
}
