//! Shortest path algorithms

use crate::fst::{Fst, MutableFst, StateId, NO_STATE_ID};
use crate::semiring::{Semiring, NaturallyOrderedSemiring};
use crate::arc::Arc;
use crate::{Result, Error};
use std::collections::BinaryHeap;
use core::cmp::Ordering;

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

/// Find shortest path(s) in an FST
pub fn shortest_path<W, F, M>(
    fst: &F,
    config: ShortestPathConfig,
) -> Result<M>
where
    W: NaturallyOrderedSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    if config.nshortest == 0 {
        return Ok(M::default());
    }
    
    let start = fst.start().ok_or_else(|| {
        Error::Algorithm("FST has no start state".into())
    })?;
    
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
            
            if <W as num_traits::Zero>::is_zero(&distance[next_state as usize]) || 
               next_weight < distance[next_state as usize] {
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
            result.add_arc(prev_out_state, Arc::new(
                arc.ilabel,
                arc.olabel,
                arc.weight.clone(),
                out_state
            ));
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
pub fn shortest_path_single<W, F, M>(fst: &F) -> Result<M>
where
    W: NaturallyOrderedSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    shortest_path(fst, ShortestPathConfig::default())
}