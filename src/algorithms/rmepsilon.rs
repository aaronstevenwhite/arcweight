//! Epsilon removal algorithm

use crate::fst::{Fst, MutableFst, StateId};
use crate::semiring::StarSemiring;
use crate::arc::Arc;
use crate::Result;
use std::collections::{HashMap, VecDeque};

/// Remove epsilon transitions
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
                        result.add_arc(state, Arc::new(
                            arc.ilabel,
                            arc.olabel,
                            weight.times(&arc.weight),
                            arc.nextstate,
                        ));
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