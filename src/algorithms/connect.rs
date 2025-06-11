//! Connect algorithm - remove non-accessible/coaccessible states

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst, StateId};
use crate::semiring::Semiring;
use crate::{Error, Result};
use std::collections::HashSet;

/// Remove non-accessible and non-coaccessible states
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
    let mut coaccessible = HashSet::new();
    let mut stack = Vec::new();

    // start from all final states
    for state in fst.states() {
        if fst.is_final(state) {
            stack.push(state);
        }
    }

    // backward search
    while let Some(state) = stack.pop() {
        if coaccessible.insert(state) {
            // find states with arcs to this state
            for s in fst.states() {
                for arc in fst.arcs(s) {
                    if arc.nextstate == state && !coaccessible.contains(&s) {
                        stack.push(s);
                    }
                }
            }
        }
    }

    coaccessible
}
