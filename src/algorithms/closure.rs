//! Closure algorithms

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst};
use crate::semiring::StarSemiring;
use crate::Result;

/// Kleene closure (star)
/// 
/// # Errors
/// 
/// Returns an error if:
/// - The input FST is invalid or corrupted
/// - Memory allocation fails during computation
/// - The semiring does not support star operations
pub fn closure<W, F, M>(fst: &F) -> Result<M>
where
    W: StarSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    closure_impl(fst, true)
}

/// Kleene plus
/// 
/// # Errors
/// 
/// Returns an error if:
/// - The input FST is invalid or corrupted
/// - Memory allocation fails during computation
/// - The semiring does not support star operations
pub fn closure_plus<W, F, M>(fst: &F) -> Result<M>
where
    W: StarSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    closure_impl(fst, false)
}

fn closure_impl<W, F, M>(fst: &F, allow_empty: bool) -> Result<M>
where
    W: StarSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();

    // copy original FST
    let mut state_map = vec![None; fst.num_states()];
    for state in fst.states() {
        let new_state = result.add_state();
        state_map[state as usize] = Some(new_state);
    }

    // create new start/final state
    let new_start = result.add_state();
    result.set_start(new_start);
    result.set_final(new_start, W::one());

    // connect to original start
    if let Some(orig_start) = fst.start() {
        if let Some(mapped_start) = state_map[orig_start as usize] {
            result.add_arc(new_start, Arc::epsilon(W::one(), mapped_start));

            if !allow_empty {
                // for plus, remove final weight from new start
                result.remove_final(new_start);
            }
        }
    }

    // copy arcs
    for state in fst.states() {
        if let Some(new_state) = state_map[state as usize] {
            for arc in fst.arcs(state) {
                if let Some(new_nextstate) = state_map[arc.nextstate as usize] {
                    result.add_arc(
                        new_state,
                        Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), new_nextstate),
                    );
                }
            }

            // connect final states back to new start
            if let Some(weight) = fst.final_weight(state) {
                result.add_arc(new_state, Arc::epsilon(weight.clone(), new_start));
            }
        }
    }

    Ok(result)
}
