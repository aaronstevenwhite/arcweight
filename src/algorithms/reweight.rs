//! Reweight FST using potential function
//!
//! Redistributes weights in an FST according to a potential function on states
//! while maintaining path equivalence.
//!
//! ## Overview
//!
//! Reweighting transforms arc and final weights using state potentials while
//! preserving the total weight of all paths. This is fundamental to weight
//! pushing algorithms and FST normalization.
//!
//! ## Algorithm
//!
//! Given potentials V[s] for each state s, reweighting applies:
//! - Arc weight: w'(e) = V[source(e)]⁻¹ ⊗ w(e) ⊗ V[dest(e)]
//! - Final weight: ρ'[s] = V[s]⁻¹ ⊗ ρ[s]
//!
//! This maintains path weights: any path from start to final state has the
//! same total weight before and after reweighting.
//!
//! ## Complexity
//!
//! - **Time:** O(|V| + |E|) - single pass over states and arcs
//! - **Space:** O(1) - operates in-place or with constant additional storage
//!
//! ## Theoretical Background
//!
//! Reweighting is based on the weight pushing framework of Mohri & Riley (2001).
//! The potential function enables redistributing weights while preserving path
//! costs, which is essential for:
//! - **Weight pushing:** Move weights toward initial or final states
//! - **Normalization:** Ensure sum of outgoing arc weights equals one
//! - **Optimization:** Improve numerical stability and cache behavior
//!
//! ## Use Cases
//!
//! - **Weight Pushing:** Implement push_weights() algorithm
//! - **Normalization:** Normalize probability distributions on arcs
//! - **Stochastic FSTs:** Ensure proper probability distributions
//! - **Numerical Stability:** Reduce accumulation of very large/small weights
//!
//! ## Examples
//!
//! ### Basic Reweighting
//!
//! ```
//! use arcweight::prelude::*;
//!
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! let s2 = fst.add_state();
//! fst.set_start(s0);
//! fst.set_final(s2, TropicalWeight::new(1.0));
//! fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));
//! fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(3.0), s2));
//!
//! // Define potentials (e.g., all zeros for identity reweighting)
//! let potentials = vec![
//!     TropicalWeight::new(0.0),  // s0
//!     TropicalWeight::new(0.0),  // s1
//!     TropicalWeight::new(0.0),  // s2
//! ];
//!
//! let reweighted = reweight(&fst, &potentials, ReweightType::ToInitial)?;
//!
//! // FST structure preserved, weights redistributed
//! assert_eq!(reweighted.num_states(), fst.num_states());
//! # Ok::<(), arcweight::Error>(())
//! ```
//!
//! ### Weight Pushing Example
//!
//! ```
//! use arcweight::prelude::*;
//!
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! fst.set_start(s0);
//! fst.set_final(s1, TropicalWeight::new(5.0));
//! fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(3.0), s1));
//!
//! // Push weights toward final states
//! let potentials = vec![
//!     TropicalWeight::new(3.0),  // Potential for s0
//!     TropicalWeight::new(0.0),  // Potential for s1
//! ];
//!
//! let pushed = reweight(&fst, &potentials, ReweightType::ToFinal)?;
//!
//! // Arc weight should be reduced, final weight increased
//! // maintaining total path weight
//! assert_eq!(pushed.num_states(), 2);
//! # Ok::<(), arcweight::Error>(())
//! ```

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst, StateId, VectorFst};
use crate::semiring::{DivisibleSemiring, Semiring};
use crate::{Error, Result};

/// Reweight direction type
///
/// Specifies the direction of reweighting, which affects how potentials
/// are applied to maintain path equivalence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReweightType {
    /// Reweight toward initial state
    ///
    /// Applies formula: w'(e) = V[source(e)]⁻¹ ⊗ w(e) ⊗ V[dest(e)]
    /// and ρ'[s] = V[s]⁻¹ ⊗ ρ[s]
    ///
    /// Useful for pushing weights backward from final states.
    ToInitial,

    /// Reweight toward final states
    ///
    /// Applies formula: w'(e) = V[dest(e)]⁻¹ ⊗ w(e) ⊗ V[source(e)]
    /// and ρ'[s] = V[s] ⊗ ρ[s]
    ///
    /// Useful for pushing weights forward from initial state.
    ToFinal,
}

/// Reweights an FST using a potential function on states.
///
/// Given potentials V[s] for each state s, reweights arcs and final
/// weights to maintain path equivalence while redistributing weight
/// according to the potential function. This is the core operation
/// for weight pushing algorithms.
///
/// # Algorithm
///
/// For `ReweightType::ToInitial`:
/// - Arc weight: w'(e) = V[source(e)]⁻¹ ⊗ w(e) ⊗ V[dest(e)]
/// - Final weight: ρ'[s] = V[s]⁻¹ ⊗ ρ[s]
///
/// For `ReweightType::ToFinal`:
/// - Arc weight: w'(e) = V[dest(e)]⁻¹ ⊗ w(e) ⊗ V[source(e)]
/// - Final weight: ρ'[s] = V[s] ⊗ ρ[s]
///
/// # Requirements
///
/// - Potentials vector must have length equal to `fst.num_states()`
/// - All potential values must be valid (non-zero for division)
/// - Semiring must support division (implement `DivisibleSemiring`)
///
/// # Time Complexity
///
/// O(|V| + |E|) - single pass over all states and arcs
///
/// # Space Complexity
///
/// O(|V| + |E|) - creates new FST with same structure
///
/// # Examples
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// fst.set_start(s0);
/// fst.set_final(s1, TropicalWeight::new(2.0));
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(3.0), s1));
///
/// // Identity potentials (zeros in tropical semiring)
/// let potentials = vec![TropicalWeight::one(); fst.num_states()];
///
/// let reweighted = reweight(&fst, &potentials, ReweightType::ToInitial)?;
///
/// assert_eq!(reweighted.num_states(), fst.num_states());
/// # Ok::<(), arcweight::Error>(())
/// ```
///
/// # Errors
///
/// Returns error if:
/// - Potentials vector length doesn't match number of states
/// - Division by potential fails (zero or undefined)
/// - FST structure is invalid
pub fn reweight<W, F>(
    fst: &F,
    potentials: &[W],
    reweight_type: ReweightType,
) -> Result<VectorFst<W>>
where
    W: Semiring + DivisibleSemiring,
    F: Fst<W>,
{
    // Validate input
    if potentials.len() != fst.num_states() {
        return Err(Error::InvalidOperation(format!(
            "Potentials length {} doesn't match FST states {}",
            potentials.len(),
            fst.num_states()
        )));
    }

    // Create output FST with same structure
    let mut result = VectorFst::<W>::new();

    // Add states
    for _ in 0..fst.num_states() {
        result.add_state();
    }

    // Set start state
    if let Some(start) = fst.start() {
        result.set_start(start);
    }

    // Process all states
    for state_idx in 0..fst.num_states() {
        let state = state_idx as StateId;

        // Reweight final weights
        if let Some(final_weight) = fst.final_weight(state) {
            let new_final_weight = match reweight_type {
                ReweightType::ToInitial => {
                    // ρ'[s] = V[s]⁻¹ ⊗ ρ[s]
                    final_weight
                        .divide(&potentials[state_idx])
                        .ok_or_else(|| {
                            Error::InvalidOperation(format!(
                                "Cannot divide by potential at state {}",
                                state
                            ))
                        })?
                }
                ReweightType::ToFinal => {
                    // ρ'[s] = V[s] ⊗ ρ[s]
                    potentials[state_idx].times(final_weight)
                }
            };
            result.set_final(state, new_final_weight);
        }

        // Reweight arcs
        for arc in fst.arcs(state) {
            let new_weight = match reweight_type {
                ReweightType::ToInitial => {
                    // w'(e) = V[source(e)]⁻¹ ⊗ w(e) ⊗ V[dest(e)]
                    let temp = arc
                        .weight
                        .divide(&potentials[state_idx])
                        .ok_or_else(|| {
                            Error::InvalidOperation(format!(
                                "Cannot divide by source potential at state {}",
                                state
                            ))
                        })?;
                    temp.times(&potentials[arc.nextstate as usize])
                }
                ReweightType::ToFinal => {
                    // w'(e) = V[dest(e)]⁻¹ ⊗ w(e) ⊗ V[source(e)]
                    let temp = arc
                        .weight
                        .divide(&potentials[arc.nextstate as usize])
                        .ok_or_else(|| {
                            Error::InvalidOperation(format!(
                                "Cannot divide by destination potential at state {}",
                                arc.nextstate
                            ))
                        })?;
                    temp.times(&potentials[state_idx])
                }
            };

            result.add_arc(
                state,
                Arc::new(arc.ilabel, arc.olabel, new_weight, arc.nextstate),
            );
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_reweight_identity_potentials() {
        // Identity potentials should leave FST unchanged
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::new(1.0));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));

        let potentials = vec![TropicalWeight::one(); fst.num_states()];

        let result = reweight(&fst, &potentials, ReweightType::ToInitial).unwrap();

        assert_eq!(result.num_states(), 2);
        assert_eq!(result.start(), Some(0));
        assert_eq!(result.final_weight(1), Some(&TropicalWeight::new(1.0)));

        let arcs: Vec<_> = result.arcs(0).collect();
        assert_eq!(arcs.len(), 1);
        assert_eq!(arcs[0].weight, TropicalWeight::new(2.0));
    }

    #[test]
    fn test_reweight_to_initial_simple() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::new(5.0));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(3.0), s1));

        let potentials = vec![TropicalWeight::new(0.0), TropicalWeight::new(2.0)];

        let result = reweight(&fst, &potentials, ReweightType::ToInitial).unwrap();

        assert_eq!(result.num_states(), 2);
        assert_eq!(result.start(), Some(0));

        // Check that structure is preserved
        let arcs: Vec<_> = result.arcs(0).collect();
        assert_eq!(arcs.len(), 1);
        assert_eq!(arcs[0].ilabel, 1);
        assert_eq!(arcs[0].olabel, 1);
        assert_eq!(arcs[0].nextstate, 1);
    }

    #[test]
    fn test_reweight_to_final_simple() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::new(5.0));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(3.0), s1));

        let potentials = vec![TropicalWeight::new(1.0), TropicalWeight::new(0.0)];

        let result = reweight(&fst, &potentials, ReweightType::ToFinal).unwrap();

        assert_eq!(result.num_states(), 2);
        assert_eq!(result.start(), Some(0));

        // Check structure preservation
        let arcs: Vec<_> = result.arcs(0).collect();
        assert_eq!(arcs.len(), 1);
    }

    #[test]
    fn test_reweight_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();
        let potentials = vec![];

        let result = reweight(&fst, &potentials, ReweightType::ToInitial).unwrap();

        assert_eq!(result.num_states(), 0);
        assert_eq!(result.start(), None);
    }

    #[test]
    fn test_reweight_single_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::new(3.0));

        let potentials = vec![TropicalWeight::new(1.0)];

        let result = reweight(&fst, &potentials, ReweightType::ToInitial).unwrap();

        assert_eq!(result.num_states(), 1);
        assert_eq!(result.start(), Some(0));
        assert!(result.final_weight(0).is_some());
    }

    #[test]
    fn test_reweight_multiple_arcs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::new(1.0));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(3.0), s2));
        fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::new(4.0), s2));

        let potentials = vec![
            TropicalWeight::new(0.0),
            TropicalWeight::new(1.0),
            TropicalWeight::new(2.0),
        ];

        let result = reweight(&fst, &potentials, ReweightType::ToInitial).unwrap();

        assert_eq!(result.num_states(), 3);
        let arcs_s0: Vec<_> = result.arcs(0).collect();
        assert_eq!(arcs_s0.len(), 2);
        let arcs_s1: Vec<_> = result.arcs(1).collect();
        assert_eq!(arcs_s1.len(), 1);
    }

    #[test]
    fn test_reweight_preserves_labels() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(5, 10, TropicalWeight::new(1.0), s1));

        let potentials = vec![TropicalWeight::new(0.5), TropicalWeight::new(0.3)];

        let result = reweight(&fst, &potentials, ReweightType::ToInitial).unwrap();

        let arcs: Vec<_> = result.arcs(0).collect();
        assert_eq!(arcs[0].ilabel, 5);
        assert_eq!(arcs[0].olabel, 10);
        assert_eq!(arcs[0].nextstate, 1);
    }

    #[test]
    fn test_reweight_invalid_potentials_length() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        fst.add_state();
        fst.add_state();

        let potentials = vec![TropicalWeight::new(1.0)]; // Wrong length

        let result = reweight(&fst, &potentials, ReweightType::ToInitial);
        assert!(result.is_err());
    }

    #[test]
    fn test_reweight_with_log_weight() {
        let mut fst = VectorFst::<LogWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, LogWeight::new(1.0));
        fst.add_arc(s0, Arc::new(1, 1, LogWeight::new(2.0), s1));

        let potentials = vec![LogWeight::one(); fst.num_states()];

        let result = reweight(&fst, &potentials, ReweightType::ToInitial).unwrap();

        assert_eq!(result.num_states(), 2);
        assert_eq!(result.start(), Some(0));
    }

    #[test]
    fn test_reweight_preserves_structure() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s3, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(2.0), s2));
        fst.add_arc(s2, Arc::new(3, 3, TropicalWeight::new(3.0), s3));

        let potentials = vec![
            TropicalWeight::new(0.0),
            TropicalWeight::new(0.5),
            TropicalWeight::new(1.0),
            TropicalWeight::new(1.5),
        ];

        let result = reweight(&fst, &potentials, ReweightType::ToFinal).unwrap();

        // Verify structure preserved
        assert_eq!(result.num_states(), 4);
        assert_eq!(result.arcs(0).count(), 1);
        assert_eq!(result.arcs(1).count(), 1);
        assert_eq!(result.arcs(2).count(), 1);
        assert_eq!(result.arcs(3).count(), 0);
    }

    #[test]
    fn test_reweight_no_final_states() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));

        let potentials = vec![TropicalWeight::one(); fst.num_states()];

        let result = reweight(&fst, &potentials, ReweightType::ToInitial).unwrap();

        assert_eq!(result.num_states(), 2);
        assert!(result.final_weight(0).is_none());
        assert!(result.final_weight(1).is_none());
    }

    #[test]
    fn test_reweight_cyclic_fst() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(2.0), s0)); // Cycle

        let potentials = vec![TropicalWeight::new(0.5), TropicalWeight::new(1.5)];

        let result = reweight(&fst, &potentials, ReweightType::ToInitial).unwrap();

        assert_eq!(result.num_states(), 2);
        assert_eq!(result.arcs(0).count(), 1);
        assert_eq!(result.arcs(1).count(), 1);
    }
}
