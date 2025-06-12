//! FST composition algorithm

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst};
use crate::semiring::Semiring;
use crate::{Error, Result};
use std::collections::HashMap;

/// Composition filter trait
pub trait ComposeFilter<W: Semiring> {
    /// Filter state type
    type FilterState: Clone + Default + Eq + std::hash::Hash;

    /// Start state
    fn start() -> Self::FilterState;

    /// Filter an arc pair
    fn filter_arc(
        &self,
        arc1: &Arc<W>,
        arc2: &Arc<W>,
        fs: &Self::FilterState,
    ) -> Option<(Arc<W>, Self::FilterState)>;
}

/// Default composition filter (match on labels)
pub struct DefaultComposeFilter;

impl<W: Semiring> ComposeFilter<W> for DefaultComposeFilter {
    type FilterState = ();

    fn start() -> Self::FilterState {}

    fn filter_arc(
        &self,
        arc1: &Arc<W>,
        arc2: &Arc<W>,
        _fs: &Self::FilterState,
    ) -> Option<(Arc<W>, Self::FilterState)> {
        if arc1.olabel == arc2.ilabel {
            Some((
                Arc::new(
                    arc1.ilabel,
                    arc2.olabel,
                    arc1.weight.times(&arc2.weight),
                    0, // nextstate set later
                ),
                (),
            ))
        } else {
            None
        }
    }
}

/// Compose two FSTs
/// 
/// Computes the composition of two FSTs using the specified composition filter.
/// The result FST accepts input sequences that are accepted by the first FST
/// and whose output is accepted by the second FST.
/// 
/// # Examples
/// 
/// ```
/// use arcweight::prelude::*;
/// 
/// // Create first FST: maps 1 -> 2
/// let mut fst1 = VectorFst::<TropicalWeight>::new();
/// let s0 = fst1.add_state();
/// let s1 = fst1.add_state();
/// fst1.set_start(s0);
/// fst1.set_final(s1, TropicalWeight::one());
/// fst1.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));
/// 
/// // Create second FST: maps 2 -> 3
/// let mut fst2 = VectorFst::<TropicalWeight>::new();
/// let s0 = fst2.add_state();
/// let s1 = fst2.add_state();
/// fst2.set_start(s0);
/// fst2.set_final(s1, TropicalWeight::one());
/// fst2.add_arc(s0, Arc::new(2, 3, TropicalWeight::new(0.3), s1));
/// 
/// // Compose: result maps 1 -> 3 with weight 0.8
/// let composed: VectorFst<TropicalWeight> = compose_default(&fst1, &fst2).unwrap();
/// 
/// assert!(composed.num_states() > 0);
/// ```
/// 
/// # Errors
/// 
/// Returns an error if:
/// - Either input FST is invalid or has no start state
/// - Memory allocation fails during computation
/// - The composition operation encounters incompatible state structures
pub fn compose<W, F1, F2, M, CF>(fst1: &F1, fst2: &F2, filter: CF) -> Result<M>
where
    W: Semiring,
    F1: Fst<W>,
    F2: Fst<W>,
    M: MutableFst<W> + Default,
    CF: ComposeFilter<W>,
{
    let start1 = fst1
        .start()
        .ok_or_else(|| Error::Algorithm("First FST has no start state".into()))?;
    let start2 = fst2
        .start()
        .ok_or_else(|| Error::Algorithm("Second FST has no start state".into()))?;

    let mut result = M::default();
    let mut state_map = HashMap::new();
    let mut queue = Vec::new();

    // create start state
    let start_state = result.add_state();
    result.set_start(start_state);
    state_map.insert((start1, start2, CF::start()), start_state);
    queue.push((start1, start2, CF::start(), start_state));

    // process states
    while let Some((s1, s2, fs, current)) = queue.pop() {
        // handle final states
        if let (Some(w1), Some(w2)) = (fst1.final_weight(s1), fst2.final_weight(s2)) {
            result.set_final(current, w1.times(w2));
        }

        // process arc pairs
        for arc1 in fst1.arcs(s1) {
            for arc2 in fst2.arcs(s2) {
                if let Some((mut arc, next_fs)) = filter.filter_arc(&arc1, &arc2, &fs) {
                    let next_key = (arc1.nextstate, arc2.nextstate, next_fs.clone());

                    let next_state = match state_map.get(&next_key) {
                        Some(&state) => state,
                        None => {
                            let state = result.add_state();
                            state_map.insert(next_key.clone(), state);
                            queue.push((arc1.nextstate, arc2.nextstate, next_fs, state));
                            state
                        }
                    };

                    arc.nextstate = next_state;
                    result.add_arc(current, arc);
                }
            }
        }
    }

    Ok(result)
}

/// Compose with default filter
/// 
/// # Errors
/// 
/// Returns an error if:
/// - Either input FST is invalid or has no start state
/// - Memory allocation fails during computation
/// - The composition operation encounters incompatible state structures
pub fn compose_default<W, F1, F2, M>(fst1: &F1, fst2: &F2) -> Result<M>
where
    W: Semiring,
    F1: Fst<W>,
    F2: Fst<W>,
    M: MutableFst<W> + Default,
{
    compose(fst1, fst2, DefaultComposeFilter)
}
