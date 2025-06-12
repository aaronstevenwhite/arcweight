//! Projection algorithms

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst};
use crate::semiring::Semiring;
use crate::Result;

/// Project input labels
///
/// # Errors
///
/// Returns an error if:
/// - The input FST is invalid or corrupted
/// - Memory allocation fails during computation
/// - The projection operation encounters invalid state or arc data
pub fn project_input<W, F, M>(fst: &F) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    project_impl(fst, true)
}

/// Project output labels
///
/// # Errors
///
/// Returns an error if:
/// - The input FST is invalid or corrupted
/// - Memory allocation fails during computation
/// - The projection operation encounters invalid state or arc data
pub fn project_output<W, F, M>(fst: &F) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    project_impl(fst, false)
}

fn project_impl<W, F, M>(fst: &F, project_input: bool) -> Result<M>
where
    W: Semiring,
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

    // copy with projection
    for state in fst.states() {
        // final weights
        if let Some(weight) = fst.final_weight(state) {
            result.set_final(state, weight.clone());
        }

        // project arcs
        for arc in fst.arcs(state) {
            let label = if project_input {
                arc.ilabel
            } else {
                arc.olabel
            };
            result.add_arc(
                state,
                Arc::new(label, label, arc.weight.clone(), arc.nextstate),
            );
        }
    }

    Ok(result)
}
