//! Synchronization algorithm

use crate::arc::Arc;
use crate::fst::{Fst, Label, MutableFst, StateId};
use crate::semiring::Semiring;
use crate::{Error, Result};
use std::collections::{HashMap, VecDeque};

/// Synchronize a transducer
///
/// # Errors
///
/// Returns an error if:
/// - The input FST is invalid or corrupted
/// - The FST has no start state
/// - Memory allocation fails during computation
/// - Label synchronization creates inconsistent buffer states
/// - The transducer cannot be properly synchronized
pub fn synchronize<W, F, M>(fst: &F) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();

    // state represents (original_state, input_buffer, output_buffer)
    type SyncState = (StateId, Vec<Label>, Vec<Label>);
    let mut state_map: HashMap<SyncState, StateId> = HashMap::new();
    let mut queue: VecDeque<SyncState> = VecDeque::new();

    // start state
    let start = fst
        .start()
        .ok_or_else(|| Error::Algorithm("FST has no start state".into()))?;

    let start_sync = (start, Vec::new(), Vec::new());
    let start_new = result.add_state();
    result.set_start(start_new);
    state_map.insert(start_sync.clone(), start_new);
    queue.push_back(start_sync);

    // process states
    while let Some((state, in_buffer, out_buffer)) = queue.pop_front() {
        let current_new = state_map[&(state, in_buffer.clone(), out_buffer.clone())];

        // process arcs
        for arc in fst.arcs(state) {
            let mut new_in_buffer = in_buffer.clone();
            let mut new_out_buffer = out_buffer.clone();

            // add labels to buffers
            if arc.ilabel != 0 {
                new_in_buffer.push(arc.ilabel);
            }
            if arc.olabel != 0 {
                new_out_buffer.push(arc.olabel);
            }

            // emit synchronized symbols
            while !new_in_buffer.is_empty() && !new_out_buffer.is_empty() {
                let ilabel = new_in_buffer.remove(0);
                let olabel = new_out_buffer.remove(0);

                let next_sync = (arc.nextstate, new_in_buffer.clone(), new_out_buffer.clone());

                let next_new = match state_map.get(&next_sync) {
                    Some(&s) => s,
                    None => {
                        let s = result.add_state();
                        state_map.insert(next_sync.clone(), s);
                        queue.push_back(next_sync);
                        s
                    }
                };

                result.add_arc(
                    current_new,
                    Arc::new(ilabel, olabel, arc.weight.clone(), next_new),
                );
            }
        }

        // handle final states
        if fst.is_final(state) && in_buffer.is_empty() && out_buffer.is_empty() {
            if let Some(weight) = fst.final_weight(state) {
                result.set_final(current_new, weight.clone());
            }
        }
    }

    Ok(result)
}
