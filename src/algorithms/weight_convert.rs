//! Weight conversion utilities

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst};
use crate::semiring::Semiring;
use crate::Result;

/// Convert between weight types
pub fn weight_convert<W1, W2, F, M, C>(fst: &F, converter: C) -> Result<M>
where
    W1: Semiring,
    W2: Semiring,
    F: Fst<W1>,
    M: MutableFst<W2> + Default,
    C: Fn(&W1) -> W2,
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

    // convert weights
    for state in fst.states() {
        // final weight
        if let Some(weight) = fst.final_weight(state) {
            result.set_final(state, converter(weight));
        }

        // arcs
        for arc in fst.arcs(state) {
            result.add_arc(
                state,
                Arc::new(
                    arc.ilabel,
                    arc.olabel,
                    converter(&arc.weight),
                    arc.nextstate,
                ),
            );
        }
    }

    Ok(result)
}
