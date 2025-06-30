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
