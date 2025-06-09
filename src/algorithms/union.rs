//! Union algorithm

use crate::fst::{Fst, MutableFst};
use crate::semiring::Semiring;
use crate::arc::Arc;
use crate::Result;

/// Union of two FSTs
pub fn union<W, F1, F2, M>(fst1: &F1, fst2: &F2) -> Result<M>
where
    W: Semiring,
    F1: Fst<W>,
    F2: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();
    
    // create new start state
    let new_start = result.add_state();
    result.set_start(new_start);
    
    // copy first FST
    let mut state_map1 = vec![None; fst1.num_states()];
    for state in fst1.states() {
        let new_state = result.add_state();
        state_map1[state as usize] = Some(new_state);
        
        if let Some(weight) = fst1.final_weight(state) {
            result.set_final(new_state, weight.clone());
        }
    }
    
    // connect new start to fst1's start
    if let Some(start1) = fst1.start() {
        if let Some(mapped_start1) = state_map1[start1 as usize] {
            result.add_arc(new_start, Arc::epsilon(W::one(), mapped_start1));
        }
    }
    
    // add arcs from fst1
    for state in fst1.states() {
        if let Some(new_state) = state_map1[state as usize] {
            for arc in fst1.arcs(state) {
                if let Some(new_nextstate) = state_map1[arc.nextstate as usize] {
                    result.add_arc(new_state, Arc::new(
                        arc.ilabel,
                        arc.olabel,
                        arc.weight.clone(),
                        new_nextstate,
                    ));
                }
            }
        }
    }
    
    // copy second FST
    let mut state_map2 = vec![None; fst2.num_states()];
    for state in fst2.states() {
        let new_state = result.add_state();
        state_map2[state as usize] = Some(new_state);
        
        if let Some(weight) = fst2.final_weight(state) {
            result.set_final(new_state, weight.clone());
        }
    }
    
    // connect new start to fst2's start
    if let Some(start2) = fst2.start() {
        if let Some(mapped_start2) = state_map2[start2 as usize] {
            result.add_arc(new_start, Arc::epsilon(W::one(), mapped_start2));
        }
    }
    
    // add arcs from fst2
    for state in fst2.states() {
        if let Some(new_state) = state_map2[state as usize] {
            for arc in fst2.arcs(state) {
                if let Some(new_nextstate) = state_map2[arc.nextstate as usize] {
                    result.add_arc(new_state, Arc::new(
                        arc.ilabel,
                        arc.olabel,
                        arc.weight.clone(),
                        new_nextstate,
                    ));
                }
            }
        }
    }
    
    Ok(result)
}