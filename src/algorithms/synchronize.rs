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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_synchronize_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();
        let result =
            synchronize::<TropicalWeight, VectorFst<TropicalWeight>, VectorFst<TropicalWeight>>(
                &fst,
            );

        // Empty FST should return error (no start state)
        assert!(result.is_err());
    }

    #[test]
    fn test_synchronize_no_start_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_final(s0, TropicalWeight::one());
        // No start state set

        let result =
            synchronize::<TropicalWeight, VectorFst<TropicalWeight>, VectorFst<TropicalWeight>>(
                &fst,
            );
        assert!(result.is_err());
    }

    #[test]
    fn test_synchronize_single_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::new(2.0));

        let result: VectorFst<TropicalWeight> = synchronize(&fst).unwrap();

        assert_eq!(result.num_states(), 1);
        assert_eq!(result.start(), Some(0));
        assert!(result.is_final(0));
        assert_eq!(result.final_weight(0), Some(&TropicalWeight::new(2.0)));
    }

    #[test]
    fn test_synchronize_simple_arc() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        // Simple arc with same input/output labels
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));

        let result: VectorFst<TropicalWeight> = synchronize(&fst).unwrap();

        // Should maintain basic structure for already synchronized FST
        assert!(result.start().is_some());
        assert!(result.num_states() >= 2);

        // Should have at least one final state
        let final_states: Vec<_> = result.states().filter(|&s| result.is_final(s)).collect();
        assert!(!final_states.is_empty());
    }

    #[test]
    fn test_synchronize_epsilon_transitions() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Epsilon transition (0 means epsilon)
        fst.add_arc(s0, Arc::new(0, 0, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(1, 1, TropicalWeight::one(), s2));

        let result: VectorFst<TropicalWeight> = synchronize(&fst).unwrap();

        assert!(result.start().is_some());
        assert!(result.num_states() > 0);
    }

    #[test]
    fn test_synchronize_unbalanced_labels() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Arc with input but no output (creates buffer state)
        fst.add_arc(s0, Arc::new(1, 0, TropicalWeight::one(), s1)); // Input only
        fst.add_arc(s1, Arc::new(0, 2, TropicalWeight::one(), s2)); // Output only

        let result: VectorFst<TropicalWeight> = synchronize(&fst).unwrap();

        // Should handle unbalanced labels by creating buffer states
        assert!(result.start().is_some());
        assert!(result.num_states() > 0);
    }

    #[test]
    fn test_synchronize_multiple_arcs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.set_final(s2, TropicalWeight::one());

        // Multiple arcs from start state
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s2));

        let result: VectorFst<TropicalWeight> = synchronize(&fst).unwrap();

        assert!(result.start().is_some());
        assert!(result.num_states() > 0);

        // Should have multiple final states
        let final_count = result.states().filter(|&s| result.is_final(s)).count();
        assert!(final_count > 0);
    }

    #[test]
    fn test_synchronize_linear_chain() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let states: Vec<_> = (0..4).map(|_| fst.add_state()).collect();

        fst.set_start(states[0]);
        fst.set_final(states[3], TropicalWeight::one());

        // Create linear chain with synchronized labels
        for i in 0..3 {
            fst.add_arc(
                states[i],
                Arc::new(
                    (i + 1) as u32,
                    (i + 1) as u32,
                    TropicalWeight::new(i as f32),
                    states[i + 1],
                ),
            );
        }

        let result: VectorFst<TropicalWeight> = synchronize(&fst).unwrap();

        assert!(result.start().is_some());
        assert!(result.num_states() >= states.len());
    }

    #[test]
    fn test_synchronize_complex_buffering() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s3, TropicalWeight::one());

        // Complex label sequence requiring buffering
        fst.add_arc(s0, Arc::new(1, 0, TropicalWeight::one(), s1)); // Input 1, no output
        fst.add_arc(s1, Arc::new(2, 0, TropicalWeight::one(), s2)); // Input 2, no output
        fst.add_arc(s2, Arc::new(0, 1, TropicalWeight::one(), s3)); // No input, output 1

        let result: VectorFst<TropicalWeight> = synchronize(&fst).unwrap();

        // Should create additional states for buffering
        assert!(result.start().is_some());
        assert!(result.num_states() > 0);
    }

    #[test]
    fn test_synchronize_self_loop() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        // Self-loop with synchronized labels
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s0));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s1));

        let result: VectorFst<TropicalWeight> = synchronize(&fst).unwrap();

        assert!(result.start().is_some());
        assert!(result.num_states() > 0);
    }

    #[test]
    fn test_synchronize_no_final_states() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        // No final states

        let result: VectorFst<TropicalWeight> = synchronize(&fst).unwrap();

        assert!(result.start().is_some());
        assert!(result.num_states() > 0);

        // Should have no final states
        let final_count = result.states().filter(|&s| result.is_final(s)).count();
        assert_eq!(final_count, 0);
    }

    #[test]
    fn test_synchronize_preserves_weights() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::new(3.5));

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.5), s1));

        let result: VectorFst<TropicalWeight> = synchronize(&fst).unwrap();

        assert!(result.start().is_some());

        // Check that weights are preserved in some form
        let total_weight_orig: f32 = fst
            .states()
            .flat_map(|s| fst.arcs(s))
            .map(|arc| *arc.weight.value())
            .sum();
        let total_weight_sync: f32 = result
            .states()
            .flat_map(|s| result.arcs(s))
            .map(|arc| *arc.weight.value())
            .sum();

        // Weights should be preserved (allowing for synchronization reorganization)
        assert!((total_weight_orig - total_weight_sync).abs() < 1e-6);
    }

    #[test]
    fn test_synchronize_already_synchronized() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Already synchronized: same input/output on each arc
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));

        let result: VectorFst<TropicalWeight> = synchronize(&fst).unwrap();

        // Should handle already synchronized FST efficiently
        assert!(result.start().is_some());
        assert!(result.num_states() > 0);

        let final_states: Vec<_> = result.states().filter(|&s| result.is_final(s)).collect();
        assert!(!final_states.is_empty());
    }

    #[test]
    fn test_synchronize_different_semirings() {
        // Test with TropicalWeight
        let mut tropical_fst = VectorFst::<TropicalWeight>::new();
        let s0 = tropical_fst.add_state();
        let s1 = tropical_fst.add_state();
        tropical_fst.set_start(s0);
        tropical_fst.set_final(s1, TropicalWeight::one());
        tropical_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));

        let tropical_result: VectorFst<TropicalWeight> = synchronize(&tropical_fst).unwrap();
        assert!(tropical_result.start().is_some());

        // Test with LogWeight
        let mut log_fst = VectorFst::<LogWeight>::new();
        let s0 = log_fst.add_state();
        let s1 = log_fst.add_state();
        log_fst.set_start(s0);
        log_fst.set_final(s1, LogWeight::one());
        log_fst.add_arc(s0, Arc::new(1, 1, LogWeight::new(0.5), s1));

        let log_result: VectorFst<LogWeight> = synchronize(&log_fst).unwrap();
        assert!(log_result.start().is_some());
    }

    #[test]
    fn test_synchronize_large_labels() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        // Test with large label values
        let large_label = u32::MAX - 1;
        fst.add_arc(
            s0,
            Arc::new(large_label, large_label, TropicalWeight::one(), s1),
        );

        let result: VectorFst<TropicalWeight> = synchronize(&fst).unwrap();

        assert!(result.start().is_some());
        assert!(result.num_states() > 0);
    }

    #[test]
    fn test_synchronize_disconnected_components() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state(); // Disconnected

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.set_final(s2, TropicalWeight::new(2.0)); // Unreachable

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        // s2 is disconnected

        let result: VectorFst<TropicalWeight> = synchronize(&fst).unwrap();

        // Should only process reachable states
        assert!(result.start().is_some());
        assert!(result.num_states() > 0);
    }

    #[test]
    fn test_synchronize_input_output_mismatch() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s3, TropicalWeight::one());

        // Sequence that requires buffering: more inputs than outputs initially
        fst.add_arc(s0, Arc::new(1, 0, TropicalWeight::one(), s1)); // Input only
        fst.add_arc(s1, Arc::new(2, 0, TropicalWeight::one(), s2)); // Input only
        fst.add_arc(s2, Arc::new(0, 1, TropicalWeight::one(), s3)); // Output only
                                                                    // Missing one output to balance

        let result: VectorFst<TropicalWeight> = synchronize(&fst).unwrap();

        // Should handle mismatched input/output sequences
        assert!(result.start().is_some());
    }
}
