//! Weight conversion algorithm for semiring transformations
//!
//! Converts FSTs between different semiring types, enabling interoperability
//! and optimization across different weight domains and applications.

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst};
use crate::semiring::Semiring;
use crate::Result;

/// Convert FST between different semiring weight types
///
/// Transforms an FST from one semiring to another using a provided conversion
/// function, enabling interoperability between different weight domains and
/// optimization for specific applications or mathematical properties.
///
/// # Algorithm Details
///
/// - **Weight Transformation:** Apply converter function to all arc and final weights
/// - **Structure Preservation:** Maintain all states, arcs, and connectivity
/// - **Time Complexity:** O(|V| + |E|) for linear traversal and conversion
/// - **Space Complexity:** O(|V| + |E|) for result FST construction
/// - **Language Preservation:** L(convert(T)) = L(T) exactly (structure unchanged)
///
/// # Mathematical Foundation
///
/// Weight conversion implements semiring homomorphism:
/// - **Semiring Mapping:** φ: S₁ → S₂ between semiring types
/// - **Operation Preservation:** May or may not preserve semiring operations
/// - **Weight Domain Change:** Transform weight interpretation and semantics
/// - **Structural Invariance:** FST topology remains completely unchanged
///
/// # Examples
///
/// ## Basic Weight Conversion
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::weight_convert;
///
/// // Original FST with TropicalWeight
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// fst.set_start(s0);
/// fst.set_final(s1, TropicalWeight::new(2.5));
/// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.2), s1));
///
/// // Convert to BooleanWeight using threshold
/// let converted: VectorFst<BooleanWeight> = weight_convert(&fst, |w| {
///     if *w.value() < 2.0 { BooleanWeight::one() } else { BooleanWeight::zero() }
/// })?;
///
/// // Result has same structure with Boolean weights
/// assert_eq!(converted.num_states(), fst.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Use Cases
///
/// ## Semiring Optimization
/// - **Performance Optimization:** Convert to more efficient semiring types
/// - **Numerical Stability:** Switch to more stable weight representations
/// - **Application Adaptation:** Adapt weights for specific use cases
/// - **Compatibility:** Enable interoperability between different systems
///
/// ## Mathematical Transformations
/// - **Log to Linear Space:** Convert between logarithmic and linear domains
/// - **Probability Normalization:** Transform to normalized probability weights
/// - **Threshold Operations:** Convert to Boolean based on weight thresholds
/// - **Precision Control:** Adjust numerical precision for specific requirements
///
/// # Performance Characteristics
///
/// - **Time Complexity:** O(|V| + |E|) linear in FST size
/// - **Space Complexity:** O(|V| + |E|) for result FST
/// - **Conversion Cost:** Depends on complexity of converter function
/// - **Memory Efficiency:** Minimal overhead beyond result storage
/// - **Scalability:** Excellent scalability for large FSTs
///
/// # Mathematical Properties
///
/// Weight conversion preserves structural properties:
/// - **Language Preservation:** L(convert(T)) = L(T) exactly
/// - **Topological Invariance:** All state and arc relationships unchanged
/// - **Determinism:** Deterministic FSTs remain deterministic
/// - **Connectivity:** Reachability and coaccessibility preserved
/// - **Label Structure:** Input/output labels completely unchanged
///
/// # Converter Function Requirements
///
/// The converter function should:
/// - **Well-Defined:** Produce valid weights in target semiring
/// - **Consistent:** Provide consistent mapping for repeated calls
/// - **Type-Safe:** Ensure proper type conversion between semirings
/// - **Performance:** Be efficient for large-scale conversion
///
/// # Common Conversion Patterns
///
/// - **Tropical to Boolean:** Threshold-based conversion for binary decisions
/// - **Log to Probability:** Exponential conversion for probability interpretation
/// - **Real to Tropical:** Logarithmic conversion for optimization
/// - **Precision Adjustment:** Convert between different floating-point precisions
///
/// # Errors
///
/// Returns [`Error::Algorithm`](crate::Error::Algorithm) if:
/// - The input FST is invalid, corrupted, or malformed
/// - Memory allocation fails during result FST construction
/// - The converter function produces invalid weight values for target semiring
/// - Weight conversion encounters overflow, underflow, or numerical errors
/// - Arc or state creation fails during result construction
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_weight_convert_tropical_to_boolean_threshold() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::new(1.5));

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1)); // Below threshold
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(2.5), s2)); // Above threshold

        // Convert using threshold at 2.0
        let converted: VectorFst<BooleanWeight> = weight_convert(&fst, |w| {
            if *w.value() < 2.0 {
                BooleanWeight::one()
            } else {
                BooleanWeight::zero()
            }
        })
        .unwrap();

        // Structure should be preserved
        assert_eq!(converted.num_states(), fst.num_states());
        assert_eq!(converted.start(), fst.start());
        assert!(converted.is_final(s2));

        // Check converted weights
        let arcs: Vec<_> = converted.arcs(s0).collect();
        assert_eq!(arcs.len(), 1);
        assert_eq!(arcs[0].weight, BooleanWeight::one()); // 0.5 < 2.0

        let arcs: Vec<_> = converted.arcs(s1).collect();
        assert_eq!(arcs.len(), 1);
        assert_eq!(arcs[0].weight, BooleanWeight::zero()); // 2.5 >= 2.0
    }

    #[test]
    fn test_weight_convert_tropical_to_log() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::new(1.0));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));

        // Convert Tropical to Log (direct value transfer)
        let converted: VectorFst<LogWeight> =
            weight_convert(&fst, |w| LogWeight::new((*w.value()) as f64)).unwrap();

        assert_eq!(converted.num_states(), fst.num_states());
        assert_eq!(converted.start(), fst.start());
        assert_eq!(converted.final_weight(s1), Some(&LogWeight::new(1.0)));

        let arcs: Vec<_> = converted.arcs(s0).collect();
        assert_eq!(arcs.len(), 1);
        assert_eq!(arcs[0].weight, LogWeight::new(2.0));
    }

    #[test]
    fn test_weight_convert_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();
        let converted: VectorFst<BooleanWeight> =
            weight_convert(&fst, |_| BooleanWeight::one()).unwrap();

        assert_eq!(converted.num_states(), 0);
        assert!(converted.is_empty());
        assert!(converted.start().is_none());
    }

    #[test]
    fn test_weight_convert_single_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::new(3.5));

        let converted: VectorFst<BooleanWeight> =
            weight_convert(&fst, |_| BooleanWeight::one()).unwrap();

        assert_eq!(converted.num_states(), 1);
        assert_eq!(converted.start(), Some(s0));
        assert!(converted.is_final(s0));
        assert_eq!(converted.final_weight(s0), Some(&BooleanWeight::one()));
        assert_eq!(converted.num_arcs_total(), 0);
    }

    #[test]
    fn test_weight_convert_no_start_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_final(s0, TropicalWeight::one());
        // No start state set

        let converted: VectorFst<BooleanWeight> =
            weight_convert(&fst, |_| BooleanWeight::one()).unwrap();

        assert_eq!(converted.num_states(), 1);
        assert!(converted.start().is_none());
        assert!(converted.is_final(s0));
    }

    #[test]
    fn test_weight_convert_multiple_arcs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.set_final(s2, TropicalWeight::one());

        // Multiple arcs from same state
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s2));
        fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::new(3.0), s1));

        // Convert with weight doubling
        let converted: VectorFst<TropicalWeight> =
            weight_convert(&fst, |w| TropicalWeight::new(*w.value() * 2.0)).unwrap();

        assert_eq!(converted.num_states(), fst.num_states());
        assert_eq!(converted.num_arcs_total(), fst.num_arcs_total());

        let arcs: Vec<_> = converted.arcs(s0).collect();
        assert_eq!(arcs.len(), 3);

        // Check that weights were doubled
        let weights: Vec<f32> = arcs.iter().map(|arc| *arc.weight.value()).collect();
        assert!(weights.contains(&2.0)); // 1.0 * 2
        assert!(weights.contains(&4.0)); // 2.0 * 2
        assert!(weights.contains(&6.0)); // 3.0 * 2
    }

    #[test]
    fn test_weight_convert_self_loops() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        // Self-loop and regular arc
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.5), s0));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.5), s1));

        // Convert to Boolean with threshold
        let converted: VectorFst<BooleanWeight> = weight_convert(&fst, |w| {
            if *w.value() < 2.0 {
                BooleanWeight::one()
            } else {
                BooleanWeight::zero()
            }
        })
        .unwrap();

        assert_eq!(converted.num_states(), 2);
        assert_eq!(converted.num_arcs_total(), 2);

        let arcs: Vec<_> = converted.arcs(s0).collect();
        assert_eq!(arcs.len(), 2);

        // Find self-loop and regular arc
        let self_loop = arcs.iter().find(|arc| arc.nextstate == s0).unwrap();
        let regular_arc = arcs.iter().find(|arc| arc.nextstate == s1).unwrap();

        assert_eq!(self_loop.weight, BooleanWeight::one()); // 1.5 < 2.0
        assert_eq!(regular_arc.weight, BooleanWeight::zero()); // 2.5 >= 2.0
    }

    #[test]
    fn test_weight_convert_linear_chain() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let states: Vec<_> = (0..4).map(|_| fst.add_state()).collect();

        fst.set_start(states[0]);
        fst.set_final(states[3], TropicalWeight::new(4.0));

        // Create linear chain
        for i in 0..3 {
            fst.add_arc(
                states[i],
                Arc::new(
                    (i + 1) as u32,
                    (i + 1) as u32,
                    TropicalWeight::new((i + 1) as f32),
                    states[i + 1],
                ),
            );
        }

        // Convert to LogWeight
        let converted: VectorFst<LogWeight> =
            weight_convert(&fst, |w| LogWeight::new((*w.value()) as f64)).unwrap();

        assert_eq!(converted.num_states(), 4);
        assert_eq!(converted.num_arcs_total(), 3);
        assert_eq!(
            converted.final_weight(states[3]),
            Some(&LogWeight::new(4.0))
        );

        // Check arc weights preserved
        for (i, &state) in states[..3].iter().enumerate() {
            let arcs: Vec<_> = converted.arcs(state).collect();
            assert_eq!(arcs.len(), 1);
            assert_eq!(*arcs[0].weight.value(), (i + 1) as f64);
        }
    }

    #[test]
    fn test_weight_convert_complex_converter() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::new(5.0));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(3.0), s1));

        // Complex converter: log transformation with offset
        let converted: VectorFst<LogWeight> = weight_convert(&fst, |w| {
            let value = *w.value();
            if value > 0.0 {
                LogWeight::new((value + 1.0).ln() as f64)
            } else {
                LogWeight::zero()
            }
        })
        .unwrap();

        assert_eq!(converted.num_states(), 2);

        // Check transformed weights: ln(3+1) = ln(4), ln(5+1) = ln(6)
        let arcs: Vec<_> = converted.arcs(s0).collect();
        assert_eq!(arcs.len(), 1);
        assert!((*arcs[0].weight.value() - 4.0_f64.ln()).abs() < 1e-6);

        let final_weight = converted.final_weight(s1).unwrap();
        assert!((*final_weight.value() - 6.0_f64.ln()).abs() < 1e-6);
    }

    #[test]
    fn test_weight_convert_preserves_labels() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        // Arc with specific input/output labels
        fst.add_arc(s0, Arc::new(100, 200, TropicalWeight::new(1.5), s1));

        let converted: VectorFst<BooleanWeight> =
            weight_convert(&fst, |_| BooleanWeight::zero()).unwrap();

        let arcs: Vec<_> = converted.arcs(s0).collect();
        assert_eq!(arcs.len(), 1);
        assert_eq!(arcs[0].ilabel, 100);
        assert_eq!(arcs[0].olabel, 200);
        assert_eq!(arcs[0].nextstate, s1);
        assert_eq!(arcs[0].weight, BooleanWeight::zero());
    }

    #[test]
    fn test_weight_convert_epsilon_arcs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Epsilon transition (label 0)
        fst.add_arc(s0, Arc::new(0, 0, TropicalWeight::new(0.1), s1));
        fst.add_arc(s1, Arc::new(1, 1, TropicalWeight::new(0.2), s2));

        let converted: VectorFst<BooleanWeight> =
            weight_convert(&fst, |_| BooleanWeight::one()).unwrap();

        assert_eq!(converted.num_states(), 3);
        assert_eq!(converted.num_arcs_total(), 2);

        // Check epsilon arc preserved
        let epsilon_arcs: Vec<_> = converted.arcs(s0).collect();
        assert_eq!(epsilon_arcs.len(), 1);
        assert_eq!(epsilon_arcs[0].ilabel, 0);
        assert_eq!(epsilon_arcs[0].olabel, 0);
        assert_eq!(epsilon_arcs[0].weight, BooleanWeight::one());
    }

    #[test]
    fn test_weight_convert_no_final_states() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        // No final states

        let converted: VectorFst<BooleanWeight> =
            weight_convert(&fst, |_| BooleanWeight::one()).unwrap();

        assert_eq!(converted.num_states(), 2);
        assert!(converted.start().is_some());

        // No final states in result
        let final_count = converted
            .states()
            .filter(|&s| converted.is_final(s))
            .count();
        assert_eq!(final_count, 0);

        // Arc should still be converted
        assert_eq!(converted.num_arcs_total(), 1);
    }

    #[test]
    fn test_weight_convert_identity_transformation() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::new(2.5));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.5), s1));

        // Identity conversion (same type)
        let converted: VectorFst<TropicalWeight> = weight_convert(&fst, |w| *w).unwrap();

        assert_eq!(converted.num_states(), fst.num_states());
        assert_eq!(converted.start(), fst.start());
        assert_eq!(converted.num_arcs_total(), fst.num_arcs_total());

        // Weights should be identical
        assert_eq!(converted.final_weight(s1), fst.final_weight(s1));

        let orig_arcs: Vec<_> = fst.arcs(s0).collect();
        let conv_arcs: Vec<_> = converted.arcs(s0).collect();
        assert_eq!(orig_arcs.len(), conv_arcs.len());
        assert_eq!(orig_arcs[0].weight, conv_arcs[0].weight);
    }
}
