//! Reverse algorithm

use crate::fst::{Fst, MutableFst};
use crate::semiring::Semiring;
use crate::arc::Arc;
use crate::Result;

/// Reverse an FST
pub fn reverse<W, F, M>(fst: &F) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();
    
    // create states
    for _ in 0..fst.num_states() {
        result.add_state();
    }
    
    // create new start state
    let new_start = result.add_state();
    result.set_start(new_start);
    
    // add reversed arcs
    for state in fst.states() {
        for arc in fst.arcs(state) {
            result.add_arc(arc.nextstate, Arc::new(
                arc.ilabel,
                arc.olabel,
                arc.weight.clone(),
                state,
            ));
        }
        
        // final states become arcs from new start
        if let Some(weight) = fst.final_weight(state) {
            result.add_arc(new_start, Arc::epsilon(weight.clone(), state));
        }
    }
    
    // original start becomes final
    if let Some(orig_start) = fst.start() {
        result.set_final(orig_start, W::one());
    }
    
    Ok(result)
}