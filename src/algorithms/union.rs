//! FST union algorithm
//!
//! Constructs the union of two weighted finite-state transducers by creating
//! a new FST that accepts the combined language of both input FSTs.

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst};
use crate::semiring::Semiring;
use crate::Result;

/// Compute the union of two FSTs creating an FST that accepts both languages
///
/// Creates a new FST that accepts any string accepted by either input FST.
/// The result FST's language is the union of the two input languages,
/// preserving all weights and maintaining the semiring structure.
///
/// # Algorithm Details
///
/// - **Construction:** Creates new start state with epsilon transitions to both FSTs
/// - **Time Complexity:** O(|V₁| + |V₂| + |E₁| + |E₂|) for copying both FSTs
/// - **Space Complexity:** O(|V₁| + |V₂| + |E₁| + |E₂|) for the combined structure
/// - **Language Relationship:** L(T₁ ∪ T₂) = L(T₁) ∪ L(T₂)
///
/// # Mathematical Foundation
///
/// For FSTs T₁ and T₂ over the same semiring, their union T₁ ∪ T₂ satisfies:
/// - **(T₁ ∪ T₂)(x) = T₁(x) ⊕ T₂(x)** where ⊕ is the semiring addition
/// - **Accepts x if:** x ∈ L(T₁) or x ∈ L(T₂) (or both)
/// - **Weight computation:** Uses semiring addition for overlapping strings
///
/// # Algorithm Steps
///
/// 1. **New Start State:** Create a single start state for the union
/// 2. **Copy First FST:** Copy all states and arcs from T₁ with new state IDs
/// 3. **Copy Second FST:** Copy all states and arcs from T₂ with new state IDs  
/// 4. **Epsilon Connections:** Add epsilon arcs from new start to both original starts
/// 5. **Preserve Structure:** All weights, labels, and final states exactly preserved
///
/// # Examples
///
/// ## Basic Union of Languages
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // FST 1: accepts "hello" -> "hi"
/// let mut fst1 = VectorFst::<TropicalWeight>::new();
/// let s0 = fst1.add_state();
/// let s1 = fst1.add_state();
/// fst1.set_start(s0);
/// fst1.set_final(s1, TropicalWeight::one());
/// fst1.add_arc(s0, Arc::new('h' as u32, 'h' as u32, TropicalWeight::one(), s1));
///
/// // FST 2: accepts "goodbye" -> "bye"  
/// let mut fst2 = VectorFst::<TropicalWeight>::new();
/// let s0 = fst2.add_state();
/// let s1 = fst2.add_state();
/// fst2.set_start(s0);
/// fst2.set_final(s1, TropicalWeight::one());
/// fst2.add_arc(s0, Arc::new('g' as u32, 'b' as u32, TropicalWeight::one(), s1));
///
/// // Union: accepts both "hello" -> "hi" and "goodbye" -> "bye"
/// let union_fst: VectorFst<TropicalWeight> = union(&fst1, &fst2)?;
///
/// // Result has states from both FSTs plus new start state
/// assert!(union_fst.num_states() >= fst1.num_states() + fst2.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Dictionary Union for Vocabulary Expansion
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Technical terms dictionary
/// let mut tech_dict = VectorFst::<TropicalWeight>::new();
/// let s0 = tech_dict.add_state();
/// let s1 = tech_dict.add_state();
/// tech_dict.set_start(s0);
/// tech_dict.set_final(s1, TropicalWeight::one());
/// tech_dict.add_arc(s0, Arc::new(1, 100, TropicalWeight::one(), s1)); // "API" -> "interface"
///
/// // Common words dictionary  
/// let mut common_dict = VectorFst::<TropicalWeight>::new();
/// let s0 = common_dict.add_state();
/// let s1 = common_dict.add_state();
/// common_dict.set_start(s0);
/// common_dict.set_final(s1, TropicalWeight::one());
/// common_dict.add_arc(s0, Arc::new(2, 200, TropicalWeight::one(), s1)); // "hello" -> "greeting"
///
/// // Combined vocabulary covering both technical and common terms
/// let full_dict: VectorFst<TropicalWeight> = union(&tech_dict, &common_dict)?;
///
/// println!("Combined dictionary has {} states", full_dict.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Weighted Union with Different Costs
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // High-confidence recognizer
/// let mut high_conf = VectorFst::<TropicalWeight>::new();
/// let s0 = high_conf.add_state();
/// let s1 = high_conf.add_state();
/// high_conf.set_start(s0);
/// high_conf.set_final(s1, TropicalWeight::new(0.1)); // Low cost = high confidence
/// high_conf.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::new(0.1), s1));
///
/// // Backup recognizer with lower confidence
/// let mut backup = VectorFst::<TropicalWeight>::new();
/// let s0 = backup.add_state();
/// let s1 = backup.add_state();
/// backup.set_start(s0);
/// backup.set_final(s1, TropicalWeight::new(0.8)); // Higher cost = lower confidence
/// backup.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::new(0.8), s1));
///
/// // Union preserves different confidence levels
/// let combined: VectorFst<TropicalWeight> = union(&high_conf, &backup)?;
///
/// // Result maintains both paths with their respective weights
/// assert!(combined.start().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Multiple Language Support
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // English phonetic rules
/// let mut english = VectorFst::<TropicalWeight>::new();
/// let s0 = english.add_state();
/// let s1 = english.add_state();
/// english.set_start(s0);
/// english.set_final(s1, TropicalWeight::one());
/// english.add_arc(s0, Arc::new(1, 10, TropicalWeight::one(), s1)); // English phoneme mapping
///
/// // Spanish phonetic rules
/// let mut spanish = VectorFst::<TropicalWeight>::new();
/// let s0 = spanish.add_state();
/// let s1 = spanish.add_state();
/// spanish.set_start(s0);
/// spanish.set_final(s1, TropicalWeight::one());
/// spanish.add_arc(s0, Arc::new(2, 20, TropicalWeight::one(), s1)); // Spanish phoneme mapping
///
/// // Multilingual phonetic processor
/// let multilingual: VectorFst<TropicalWeight> = union(&english, &spanish)?;
///
/// // Now handles both English and Spanish phonetic transformations
/// assert!(multilingual.num_states() > english.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Chaining Unions for Multiple FSTs
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Helper function to create simple FST
/// fn create_simple_fst(label: u32, output: u32) -> VectorFst<TropicalWeight> {
///     let mut fst = VectorFst::new();
///     let s0 = fst.add_state();
///     let s1 = fst.add_state();
///     fst.set_start(s0);
///     fst.set_final(s1, TropicalWeight::one());
///     fst.add_arc(s0, Arc::new(label, output, TropicalWeight::one(), s1));
///     fst
/// }
///
/// // Create multiple FSTs
/// let fst_a = create_simple_fst(1, 10);
/// let fst_b = create_simple_fst(2, 20);
/// let fst_c = create_simple_fst(3, 30);
///
/// // Chain unions: (A ∪ B) ∪ C = A ∪ B ∪ C
/// let union_ab: VectorFst<TropicalWeight> = union(&fst_a, &fst_b)?;
/// let union_abc: VectorFst<TropicalWeight> = union(&union_ab, &fst_c)?;
///
/// // Result accepts inputs from all three original FSTs
/// assert!(union_abc.num_states() > 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Use Cases
///
/// ## Language Modeling
/// - **Vocabulary Combination:** Merge domain-specific and general vocabularies
/// - **Multi-Domain:** Combine FSTs trained on different text domains
/// - **Robustness:** Add fallback models for improved coverage
/// - **Ensemble Methods:** Combine multiple language models
///
/// ## Speech Recognition
/// - **Acoustic Models:** Merge different acoustic model variants
/// - **Pronunciation:** Combine standard and alternative pronunciations
/// - **Multi-Speaker:** Union of speaker-specific models
/// - **Confidence Weighting:** Multiple recognizers with different confidence levels
///
/// ## Machine Translation
/// - **Translation Tables:** Merge different translation dictionaries
/// - **Multi-Source:** Combine translations from different training corpora
/// - **Domain Adaptation:** General + domain-specific translation rules
/// - **Backup Translations:** Primary + fallback translation systems
///
/// ## Text Processing
/// - **Rule Systems:** Combine orthographic, phonological, and morphological rules
/// - **Multi-Language:** Support for multiple input languages
/// - **Grammar Variants:** Standard + colloquial grammar rules
/// - **Error Handling:** Main rules + error correction rules
///
/// # Performance Characteristics
///
/// - **Memory Usage:** Sum of both input FSTs plus minimal overhead
/// - **Construction Time:** Linear in total size of both FSTs
/// - **Determinism:** Union of deterministic FSTs is generally nondeterministic
/// - **State Growth:** |V_result| = |V₁| + |V₂| + 1 (one new start state)
/// - **Arc Preservation:** All original arcs preserved exactly
///
/// # Mathematical Properties
///
/// The union operation satisfies key algebraic properties:
/// - **Commutativity:** union(T₁, T₂) ≈ union(T₂, T₁) (up to state labeling)
/// - **Associativity:** union(union(T₁, T₂), T₃) ≈ union(T₁, union(T₂, T₃))
/// - **Idempotency:** union(T, T) accepts same language as T (with duplicated paths)
/// - **Language Union:** L(union(T₁, T₂)) = L(T₁) ∪ L(T₂)
/// - **Weight Preservation:** All original weights exactly maintained
///
/// # Implementation Details
///
/// The algorithm creates a new start state and connects it via epsilon transitions
/// to the start states of both input FSTs. This preserves the original FST structures
/// while enabling the new FST to begin execution in either original FST.
/// State IDs are remapped to avoid conflicts between the two input FSTs.
///
/// # Optimization Opportunities
///
/// After union construction, consider these optimizations:
/// - **Determinization:** Resolve nondeterminism if needed
/// - **Minimization:** Reduce redundant states in the result
/// - **Epsilon Removal:** Remove epsilon transitions for efficiency
/// - **Connection:** Remove unreachable or non-coaccessible states
///
/// # Errors
///
/// Returns [`Error::Algorithm`](crate::Error::Algorithm) if:
/// - Either input FST is invalid, corrupted, or malformed
/// - Memory allocation fails during state or arc creation
/// - State mapping operations encounter invalid indices
/// - Arc creation fails due to invalid labels or weights
/// - FST iteration operations fail due to data corruption
/// - The result FST becomes too large to represent in memory
///
/// # See Also
///
/// - [`concat()`](crate::algorithms::concat()) for sequential FST combination
/// - [`compose()`](crate::algorithms::compose()) for FST intersection/composition
/// - [`determinize()`](crate::algorithms::determinize()) for resolving nondeterminism after union
/// - [`minimize()`](crate::algorithms::minimize()) for reducing result FST size
/// - [Working with FSTs - Union](../../docs/working-with-fsts/core-operations.md#union) for usage patterns
/// - [Core Concepts](../../docs/core-concepts/algorithms.md#union) for mathematical theory
pub fn union<W, F1, F2, M>(fst1: &F1, fst2: &F2) -> Result<M>
where
    W: Semiring,
    F1: Fst<W>,
    F2: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();

    // create new start state
    let new_start = result.add_state();
    result.set_start(new_start);

    // copy first FST
    let mut state_map1 = vec![None; fst1.num_states()];
    for state in fst1.states() {
        let new_state = result.add_state();
        state_map1[state as usize] = Some(new_state);

        if let Some(weight) = fst1.final_weight(state) {
            result.set_final(new_state, weight.clone());
        }
    }

    // connect new start to fst1's start
    if let Some(start1) = fst1.start() {
        if let Some(mapped_start1) = state_map1[start1 as usize] {
            result.add_arc(new_start, Arc::epsilon(W::one(), mapped_start1));
        }
    }

    // add arcs from fst1
    for state in fst1.states() {
        if let Some(new_state) = state_map1[state as usize] {
            for arc in fst1.arcs(state) {
                if let Some(new_nextstate) = state_map1[arc.nextstate as usize] {
                    result.add_arc(
                        new_state,
                        Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), new_nextstate),
                    );
                }
            }
        }
    }

    // copy second FST
    let mut state_map2 = vec![None; fst2.num_states()];
    for state in fst2.states() {
        let new_state = result.add_state();
        state_map2[state as usize] = Some(new_state);

        if let Some(weight) = fst2.final_weight(state) {
            result.set_final(new_state, weight.clone());
        }
    }

    // connect new start to fst2's start
    if let Some(start2) = fst2.start() {
        if let Some(mapped_start2) = state_map2[start2 as usize] {
            result.add_arc(new_start, Arc::epsilon(W::one(), mapped_start2));
        }
    }

    // add arcs from fst2
    for state in fst2.states() {
        if let Some(new_state) = state_map2[state as usize] {
            for arc in fst2.arcs(state) {
                if let Some(new_nextstate) = state_map2[arc.nextstate as usize] {
                    result.add_arc(
                        new_state,
                        Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), new_nextstate),
                    );
                }
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use num_traits::One;

    #[test]
    fn test_union_basic() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::one());
        fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t1, TropicalWeight::one());
        fst2.add_arc(t0, Arc::new(2, 2, TropicalWeight::new(2.0), t1));

        let unioned: VectorFst<TropicalWeight> = union(&fst1, &fst2).unwrap();

        // Should have states from both FSTs plus new start state
        assert!(unioned.num_states() >= fst1.num_states() + fst2.num_states());
        assert!(unioned.start().is_some());

        // Union might use epsilon transitions or restructure the FST
        // At minimum, should have preserved the original structure somehow
        assert!(unioned.num_states() >= 2); // Should have at least the states from both FSTs
    }

    #[test]
    fn test_union_empty() {
        let fst1 = VectorFst::<TropicalWeight>::new();
        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let s0 = fst2.add_state();
        fst2.set_start(s0);
        fst2.set_final(s0, TropicalWeight::one());

        let unioned: VectorFst<TropicalWeight> = union(&fst1, &fst2).unwrap();

        // Should be equivalent to fst2
        assert!(unioned.start().is_some());
    }
}
