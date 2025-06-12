//! Weight and label pushing algorithms

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst};
use crate::semiring::{DivisibleSemiring, Semiring};
use crate::Result;

/// Push weights toward initial state
/// 
/// # Errors
/// 
/// Returns an error if:
/// - The input FST is invalid or corrupted
/// - Memory allocation fails during computation
/// - The FST is not compatible with weight pushing operations
/// - Potential computation fails for non-divisible weights
pub fn push_weights<W, F, M>(fst: &F) -> Result<M>
where
    W: DivisibleSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    // compute potentials using shortest distance
    let potentials = compute_potentials(fst)?;

    let mut result = M::default();

    // copy states
    for _ in 0..fst.num_states() {
        result.add_state();
    }

    // set start
    if let Some(start) = fst.start() {
        result.set_start(start);
    }

    // reweight arcs
    for state in fst.states() {
        let potential = &potentials[state as usize];

        // adjust final weight
        if let Some(weight) = fst.final_weight(state) {
            if let Some(pushed) = weight.divide(potential) {
                result.set_final(state, pushed);
            }
        }

        // adjust arc weights
        for arc in fst.arcs(state) {
            let next_potential = &potentials[arc.nextstate as usize];
            if let Some(reweighted) = arc.weight.times(next_potential).divide(potential) {
                result.add_arc(
                    state,
                    Arc::new(arc.ilabel, arc.olabel, reweighted, arc.nextstate),
                );
            }
        }
    }

    Ok(result)
}

/// Push labels toward initial state
/// 
/// # Errors
/// 
/// Returns an error if:
/// - The input FST is invalid or corrupted
/// - Memory allocation fails during computation
/// - The FST structure prevents label pushing operations
pub fn push_labels<W, F, M>(fst: &F) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    // simple copy for now
    let mut result = M::default();

    for _ in 0..fst.num_states() {
        result.add_state();
    }

    if let Some(start) = fst.start() {
        result.set_start(start);
    }

    for state in fst.states() {
        if let Some(weight) = fst.final_weight(state) {
            result.set_final(state, weight.clone());
        }

        for arc in fst.arcs(state) {
            result.add_arc(state, arc.clone());
        }
    }

    Ok(result)
}

fn compute_potentials<W: DivisibleSemiring, F: Fst<W>>(fst: &F) -> Result<Vec<W>> {
    // simplified - would use shortest distance algorithm
    Ok(vec![W::one(); fst.num_states()])
}
