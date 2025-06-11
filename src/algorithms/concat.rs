//! Concatenation algorithm

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst};
use crate::semiring::Semiring;
use crate::Result;

/// Concatenate two FSTs
pub fn concat<W, F1, F2, M>(fst1: &F1, fst2: &F2) -> Result<M>
where
    W: Semiring,
    F1: Fst<W>,
    F2: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();

    // copy first FST
    let mut state_map1 = vec![None; fst1.num_states()];
    for state in fst1.states() {
        let new_state = result.add_state();
        state_map1[state as usize] = Some(new_state);

        if let Some(weight) = fst1.final_weight(state) {
            result.set_final(new_state, weight.clone());
        }
    }

    // set start state from fst1
    if let Some(start1) = fst1.start() {
        if let Some(new_start) = state_map1[start1 as usize] {
            result.set_start(new_start);
        }
    }

    // add arcs from fst1
    for state in fst1.states() {
        if let Some(new_state) = state_map1[state as usize] {
            for arc in fst1.arcs(state) {
                if let Some(new_nextstate) = state_map1[arc.nextstate as usize] {
                    result.add_arc(
                        new_state,
                        Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), new_nextstate),
                    );
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

    // add arcs from fst2
    for state in fst2.states() {
        if let Some(new_state) = state_map2[state as usize] {
            for arc in fst2.arcs(state) {
                if let Some(new_nextstate) = state_map2[arc.nextstate as usize] {
                    result.add_arc(
                        new_state,
                        Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), new_nextstate),
                    );
                }
            }
        }
    }

    // connect final states of fst1 to start of fst2
    if let Some(start2) = fst2.start() {
        if let Some(new_start2) = state_map2[start2 as usize] {
            for state in fst1.states() {
                if let Some(final_weight) = fst1.final_weight(state) {
                    if let Some(new_state) = state_map1[state as usize] {
                        // remove final weight
                        result.remove_final(new_state);
                        // add epsilon arc to start of fst2
                        result.add_arc(new_state, Arc::epsilon(final_weight.clone(), new_start2));
                    }
                }
            }
        }
    }

    Ok(result)
}
