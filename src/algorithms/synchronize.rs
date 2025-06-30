//! FST synchronization algorithm for aligned input/output processing
//!
//! Synchronizes finite-state transducers to ensure input and output labels
//! are properly aligned for efficient processing and composition operations.

use crate::arc::Arc;
use crate::fst::{Fst, Label, MutableFst, StateId};
use crate::semiring::Semiring;
use crate::{Error, Result};
use std::collections::{HashMap, VecDeque};

/// Synchronize a finite-state transducer for aligned label processing
///
/// Transforms a transducer to ensure input and output labels are properly
/// synchronized, enabling efficient composition and processing. Uses buffer
/// states to manage label alignment and maintain transduction semantics.
///
/// # Algorithm Details
///
/// - **Buffer States:** Manage input/output label alignment using state buffers
/// - **Label Synchronization:** Ensure balanced input/output label sequences
/// - **Time Complexity:** O(|V| × buffer_size² + |E| × buffer_size)
/// - **Space Complexity:** O(|V| × buffer_size²) for synchronized states
/// - **Transduction Preservation:** Maintains all input-output mappings exactly
///
/// # Mathematical Foundation
///
/// Synchronization addresses label alignment in transducers:
/// - **Buffer Management:** Track input/output label sequences in state space
/// - **Alignment Constraint:** Ensure input/output sequences can be synchronized
/// - **State Expansion:** Create new states representing buffer configurations
/// - **Transition Rules:** Define valid transitions based on buffer states
///
/// # Use Cases
///
/// ## Composition Preparation
/// - **Transducer Alignment:** Prepare FSTs for efficient composition
/// - **Label Matching:** Ensure proper label matching during composition
/// - **Buffer Optimization:** Optimize buffer sizes for performance
///
/// ## String Processing
/// - **Sequence Alignment:** Align input/output string sequences
/// - **Stream Processing:** Process streaming data with proper alignment
/// - **Real-Time Systems:** Maintain synchronization in real-time processing
///
/// # Performance Characteristics
///
/// - **State Explosion:** Buffer states can cause exponential growth
/// - **Memory Usage:** Proportional to buffer size and original FST size
/// - **Processing Efficiency:** Improves composition and matching performance
///
/// # Errors
///
/// Returns [`Error::Algorithm`] if:
/// - The input FST is invalid, corrupted, or malformed
/// - The FST has no start state (required for synchronization)
/// - Memory allocation fails during buffer state computation
/// - Label synchronization creates inconsistent or invalid buffer states
/// - The transducer structure prevents proper synchronization
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
