//! Topological sort algorithm

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst, StateId};
use crate::semiring::Semiring;
use crate::{Error, Result};
use std::collections::HashSet;

/// Topologically sort an acyclic FST
/// 
/// # Errors
/// 
/// Returns an error if:
/// - The input FST is invalid or corrupted
/// - Memory allocation fails during computation
/// - The FST contains cycles (violates acyclic requirement)
/// - Topological ordering computation fails
pub fn topsort<W, F, M>(fst: &F) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    // compute topological order
    let order = compute_topological_order(fst)?;

    // create mapping from old to new state IDs
    let mut state_map = vec![None; fst.num_states()];
    for (new_id, &old_id) in order.iter().enumerate() {
        state_map[old_id as usize] = Some(new_id as StateId);
    }

    let mut result = M::default();

    // create states in topological order
    for _ in &order {
        result.add_state();
    }

    // set start
    if let Some(start) = fst.start() {
        if let Some(new_start) = state_map[start as usize] {
            result.set_start(new_start);
        }
    }

    // copy with remapped states
    for &old_state in &order {
        if let Some(new_state) = state_map[old_state as usize] {
            // final weight
            if let Some(weight) = fst.final_weight(old_state) {
                result.set_final(new_state, weight.clone());
            }

            // arcs
            for arc in fst.arcs(old_state) {
                if let Some(new_nextstate) = state_map[arc.nextstate as usize] {
                    result.add_arc(
                        new_state,
                        Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), new_nextstate),
                    );
                }
            }
        }
    }

    Ok(result)
}

fn compute_topological_order<W: Semiring, F: Fst<W>>(fst: &F) -> Result<Vec<StateId>> {
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

    // start DFS from start state if it exists
    if let Some(start) = fst.start() {
        if !visited.contains(&start) {
            dfs(fst, start, &mut visited, &mut finished, &mut order)?;
        }
    }

    // visit any remaining unvisited states
    for state in fst.states() {
        if !visited.contains(&state) {
            dfs(fst, state, &mut visited, &mut finished, &mut order)?;
        }
    }

    order.reverse();
    Ok(order)
}
