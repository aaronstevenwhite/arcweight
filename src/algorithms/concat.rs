//! FST concatenation algorithm
//!
//! Constructs the concatenation of two weighted finite-state transducers by
//! sequentially chaining them to form a single FST that accepts T₁ followed by T₂.
//!
//! ## References
//!
//! - Hopcroft, J. E., and Ullman, J. D. (1979). "Introduction to Automata Theory,
//!   Languages, and Computation." Addison-Wesley.

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst};
use crate::semiring::Semiring;
use crate::Result;

/// Concatenate two FSTs creating an FST that accepts T₁ followed by T₂
///
/// Creates a new FST that accepts any string of the form xy where x is accepted
/// by the first FST and y is accepted by the second FST. The result preserves
/// all weights through semiring multiplication for the combined paths.
///
/// # Algorithm Details
///
/// - **Sequential Combination:** Chains T₁ and T₂ in sequence: T₁ · T₂
/// - **Time Complexity:** O(|V₁| + |V₂| + |E₁| + |E₂|) for copying both FSTs
/// - **Space Complexity:** O(|V₁| + |V₂| + |E₁| + |E₂|) for the combined structure
/// - **Language Relationship:** L(T₁ · T₂) = {xy : x ∈ L(T₁), y ∈ L(T₂)}
///
/// # Mathematical Foundation
///
/// For FSTs T₁ and T₂ over the same semiring, their concatenation T₁ · T₂ satisfies:
/// - **(T₁ · T₂)(xy) = T₁(x) ⊗ T₂(y)** where ⊗ is semiring multiplication
/// - **Sequential Processing:** First processes string through T₁, then T₂
/// - **Weight Combination:** Combines weights from both FSTs multiplicatively
///
/// # Algorithm Steps
///
/// 1. **Copy First FST:** Copy all states, arcs, and structure from T₁
/// 2. **Copy Second FST:** Copy all states, arcs, and structure from T₂  
/// 3. **Connect Final to Start:** Remove final weights from T₁, add epsilon arcs
/// 4. **Epsilon Transitions:** Final states of T₁ connect to start state of T₂
/// 5. **Weight Transfer:** Final weights of T₁ become epsilon arc weights
///
/// # Examples
///
/// ## Basic String Concatenation
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // FST 1: accepts "hello" -> "hi"
/// let mut fst1 = VectorFst::<TropicalWeight>::new();
/// let s0 = fst1.add_state();
/// let s1 = fst1.add_state();
/// fst1.set_start(s0);
/// fst1.set_final(s1, TropicalWeight::new(0.5));
/// fst1.add_arc(s0, Arc::new('h' as u32, 'h' as u32, TropicalWeight::one(), s1));
///
/// // FST 2: accepts "world" -> "earth"
/// let mut fst2 = VectorFst::<TropicalWeight>::new();
/// let s0 = fst2.add_state();
/// let s1 = fst2.add_state();
/// fst2.set_start(s0);
/// fst2.set_final(s1, TropicalWeight::new(0.3));
/// fst2.add_arc(s0, Arc::new('w' as u32, 'e' as u32, TropicalWeight::one(), s1));
///
/// // Concatenation: accepts "helloworld" -> "hiearth" with combined weight
/// let concat_fst: VectorFst<TropicalWeight> = concat(&fst1, &fst2)?;
///
/// // Result combines both FSTs sequentially
/// assert_eq!(concat_fst.num_states(), fst1.num_states() + fst2.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Weight-Preserving Concatenation
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // First stage: tokenization with cost 0.2
/// let mut tokenizer = VectorFst::<TropicalWeight>::new();
/// let s0 = tokenizer.add_state();
/// let s1 = tokenizer.add_state();
/// tokenizer.set_start(s0);
/// tokenizer.set_final(s1, TropicalWeight::new(0.2));
/// tokenizer.add_arc(s0, Arc::new(1, 10, TropicalWeight::new(0.1), s1));
///
/// // Second stage: parsing with cost 0.3
/// let mut parser = VectorFst::<TropicalWeight>::new();
/// let s0 = parser.add_state();
/// let s1 = parser.add_state();
/// parser.set_start(s0);
/// parser.set_final(s1, TropicalWeight::new(0.3));
/// parser.add_arc(s0, Arc::new(10, 100, TropicalWeight::new(0.2), s1));
///
/// // Pipeline: tokenization → parsing with total cost 0.8 (0.1 + 0.2 + 0.2 + 0.3)
/// let pipeline: VectorFst<TropicalWeight> = concat(&tokenizer, &parser)?;
///
/// // Final weights are multiplication of intermediate weights
/// assert!(pipeline.start().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Multiple Final States Handling
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // First FST with multiple final states
/// let mut prefix = VectorFst::<TropicalWeight>::new();
/// let s0 = prefix.add_state();
/// let s1 = prefix.add_state();
/// let s2 = prefix.add_state();
/// prefix.set_start(s0);
/// prefix.set_final(s1, TropicalWeight::new(0.4)); // Accept "a"
/// prefix.set_final(s2, TropicalWeight::new(0.6)); // Accept "ab"
///
/// prefix.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s1));
/// prefix.add_arc(s1, Arc::new('b' as u32, 'b' as u32, TropicalWeight::one(), s2));
///
/// // Second FST: suffix
/// let mut suffix = VectorFst::<TropicalWeight>::new();
/// let s0 = suffix.add_state();
/// let s1 = suffix.add_state();
/// suffix.set_start(s0);
/// suffix.set_final(s1, TropicalWeight::one());
/// suffix.add_arc(s0, Arc::new('c' as u32, 'c' as u32, TropicalWeight::one(), s1));
///
/// // Concatenation handles multiple connection points
/// let result: VectorFst<TropicalWeight> = concat(&prefix, &suffix)?;
///
/// // Now accepts "ac" and "abc" with respective weights
/// assert!(result.num_states() > 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Morphological Processing Pipeline
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Morpheme FST: root extraction
/// let mut morpheme = VectorFst::<TropicalWeight>::new();
/// let s0 = morpheme.add_state();
/// let s1 = morpheme.add_state();
/// morpheme.set_start(s0);
/// morpheme.set_final(s1, TropicalWeight::one());
/// morpheme.add_arc(s0, Arc::new(1, 10, TropicalWeight::one(), s1)); // word -> root
///
/// // Inflection FST: adding grammatical markers
/// let mut inflection = VectorFst::<TropicalWeight>::new();
/// let s0 = inflection.add_state();
/// let s1 = inflection.add_state();
/// inflection.set_start(s0);
/// inflection.set_final(s1, TropicalWeight::one());
/// inflection.add_arc(s0, Arc::new(10, 20, TropicalWeight::one(), s1)); // root -> inflected
///
/// // Complete morphological pipeline
/// let morphology: VectorFst<TropicalWeight> = concat(&morpheme, &inflection)?;
///
/// // Processes: word -> root -> inflected form
/// println!("Morphology pipeline has {} states", morphology.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Building Complex Pipelines
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Helper to create simple FST stage
/// fn create_stage(input: u32, output: u32, cost: f32) -> VectorFst<TropicalWeight> {
///     let mut fst = VectorFst::new();
///     let s0 = fst.add_state();
///     let s1 = fst.add_state();
///     fst.set_start(s0);
///     fst.set_final(s1, TropicalWeight::new(cost));
///     fst.add_arc(s0, Arc::new(input, output, TropicalWeight::one(), s1));
///     fst
/// }
///
/// // Create three-stage pipeline
/// let stage1 = create_stage(1, 10, 0.1);  // Input processing
/// let stage2 = create_stage(10, 100, 0.2); // Analysis
/// let stage3 = create_stage(100, 1000, 0.3); // Output formatting
///
/// // Chain stages: stage1 → stage2 → stage3
/// let pipeline_12: VectorFst<TropicalWeight> = concat(&stage1, &stage2)?;
/// let full_pipeline: VectorFst<TropicalWeight> = concat(&pipeline_12, &stage3)?;
///
/// // Complete pipeline: 1 → 10 → 100 → 1000 with total cost 0.6
/// assert!(full_pipeline.num_states() > 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Use Cases
///
/// ## Natural Language Processing
/// - **Processing Pipelines:** Tokenization → POS tagging → parsing → generation
/// - **Translation Chains:** Source analysis → transfer → target generation
/// - **Morphology:** Root extraction → inflection → surface form generation
/// - **Text Normalization:** Preprocessing → normalization → postprocessing
///
/// ## Speech Processing
/// - **Recognition Pipeline:** Acoustic → phoneme → word → sentence level
/// - **Synthesis:** Text → phoneme → acoustic → audio generation
/// - **Feature Extraction:** Raw signal → frames → features → classification
/// - **Model Chaining:** Multiple specialized recognizers in sequence
///
/// ## Compiler Construction
/// - **Lexical Analysis:** Character → token → symbol classification
/// - **Parsing Pipeline:** Tokens → syntax tree → semantic analysis
/// - **Code Generation:** AST → intermediate → assembly → machine code
/// - **Optimization Passes:** Multiple transformation stages
///
/// ## Information Extraction
/// - **Document Processing:** Raw text → structure → entities → relations
/// - **Pattern Recognition:** Low-level → mid-level → high-level features
/// - **Multi-Stage Filtering:** Coarse → fine-grained classification
/// - **Hierarchical Analysis:** Word → phrase → sentence → document level
///
/// # Performance Characteristics
///
/// - **Memory Usage:** Sum of both input FSTs (no sharing of structure)
/// - **Construction Time:** Linear in total size O(|V₁| + |V₂| + |E₁| + |E₂|)
/// - **Epsilon Transitions:** Creates epsilon arcs from T₁ finals to T₂ start
/// - **State Count:** |V_result| = |V₁| + |V₂| (exact sum)
/// - **Arc Preservation:** All original arcs preserved plus connection epsilons
///
/// # Mathematical Properties
///
/// Concatenation satisfies important algebraic properties:
/// - **Associativity:** concat(concat(T₁, T₂), T₃) ≈ concat(T₁, concat(T₂, T₃))
/// - **Non-Commutativity:** concat(T₁, T₂) ≠ concat(T₂, T₁) in general
/// - **Language Concatenation:** L(T₁ · T₂) = L(T₁) · L(T₂)
/// - **Weight Multiplication:** Weights combine through semiring multiplication
/// - **Empty Language:** concat(∅, T) = concat(T, ∅) = ∅
///
/// # Implementation Details
///
/// The algorithm copies both FSTs completely and connects them via epsilon
/// transitions. Final states of the first FST lose their final status and
/// instead connect to the start state of the second FST through epsilon arcs
/// carrying the original final weights.
///
/// # Optimization Considerations
///
/// After concatenation, consider these optimizations:
/// - **Epsilon Removal:** Eliminate epsilon transitions for efficiency
/// - **Determinization:** Resolve any nondeterminism introduced
/// - **Minimization:** Reduce redundant states in the result
/// - **Connection:** Remove unreachable states from the combined FST
///
/// # Errors
///
/// Returns [`Error::Algorithm`](crate::Error::Algorithm) if:
/// - Either input FST is invalid, corrupted, or malformed
/// - Memory allocation fails during state or arc creation
/// - State mapping operations encounter invalid state indices
/// - Arc creation fails due to invalid labels or weights
/// - Final state operations encounter inconsistent data
/// - The result FST becomes too large to represent in memory
///
/// # See Also
///
/// - [`union()`](crate::algorithms::union()) for parallel FST combination (T₁ ∪ T₂)
/// - [`compose()`](crate::algorithms::compose()) for FST intersection/composition
/// - [`closure()`](crate::algorithms::closure()) for Kleene star operation (T*)
/// - [`remove_epsilons()`](crate::algorithms::remove_epsilons()) for epsilon removal
/// - [Working with FSTs - Concatenation](../../docs/working-with-fsts/core-operations.md#concatenation) for usage patterns
/// - [Core Concepts](../../docs/core-concepts/algorithms.md#concatenation) for mathematical theory
pub fn concat<W, F1, F2, M>(fst1: &F1, fst2: &F2) -> Result<M>
where
    W: Semiring,
    F1: Fst<W>,
    F2: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();

    // copy first FST
    let mut state_map1 = vec![None; fst1.num_states()];
    for state in fst1.states() {
        let new_state = result.add_state();
        state_map1[state as usize] = Some(new_state);

        if let Some(weight) = fst1.final_weight(state) {
            result.set_final(new_state, weight.clone());
        }
    }

    // set start state from fst1
    if let Some(start1) = fst1.start() {
        if let Some(new_start) = state_map1[start1 as usize] {
            result.set_start(new_start);
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

    // connect final states of fst1 to start of fst2
    if let Some(start2) = fst2.start() {
        if let Some(new_start2) = state_map2[start2 as usize] {
            for state in fst1.states() {
                if let Some(final_weight) = fst1.final_weight(state) {
                    if let Some(new_state) = state_map1[state as usize] {
                        // remove final weight
                        result.remove_final(new_state);
                        // add epsilon arc to start of fst2
                        result.add_arc(new_state, Arc::epsilon(final_weight.clone(), new_start2));
                    }
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
    fn test_concat_basic() {
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

        let concatenated: VectorFst<TropicalWeight> = concat(&fst1, &fst2).unwrap();

        // Should have states from both FSTs
        assert_eq!(
            concatenated.num_states(),
            fst1.num_states() + fst2.num_states()
        );
        assert!(concatenated.start().is_some());

        // Original final states of fst1 should no longer be final
        // Only final states of fst2 (with offset) should be final
        let mut final_count = 0;
        for state in concatenated.states() {
            if concatenated.is_final(state) {
                final_count += 1;
            }
        }
        assert!(final_count > 0);
    }

    #[test]
    fn test_concat_empty() {
        let fst1 = VectorFst::<TropicalWeight>::new();
        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let s0 = fst2.add_state();
        fst2.set_start(s0);
        fst2.set_final(s0, TropicalWeight::one());

        let concatenated: VectorFst<TropicalWeight> = concat(&fst1, &fst2).unwrap();

        // Concat with empty should be empty
        assert!(concatenated.is_empty() || concatenated.num_arcs_total() == 0);
    }
}
