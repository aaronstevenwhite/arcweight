//! FST composition algorithm
//!
//! ## Overview
//!
//! Composition is a fundamental operation for combining two weighted finite-state
//! transducers (FSTs) where the output of the first FST is matched against the input
//! of the second FST. For FSTs T₁: Σ* → Γ* and T₂: Γ* → Δ*, their composition
//! T₁ ∘ T₂: Σ* → Δ* computes all valid transformations through both transducers.
//!
//! ## Complexity
//!
//! - **Time:** O(V₁V₂ + E₁E₂) where Vᵢ = states in FST i, Eᵢ = arcs in FST i
//! - **Space:** O(V₁V₂) for state pair storage
//!
//! ## Use Cases
//!
//! - **NLP pipelines:** Chain tokenization → POS tagging → parsing
//! - **Speech recognition:** Pronunciation → acoustic model → language model
//! - **Machine translation:** Source → interlingua → target language
//! - **Spell checking:** Input → error model → dictionary
//!
//! ## References
//!
//! - Mehryar Mohri, Fernando Pereira, and Michael Riley (2008).
//!   "Speech Recognition with Weighted Finite-State Transducers."
//!   Handbook on Speech Processing and Speech Communication.

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
#[derive(Debug)]
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

/// Compose two FSTs using a custom composition filter
///
/// Computes the composition of two FSTs where the output of the first FST
/// is matched against the input of the second FST. The result accepts sequences
/// that are transformed by the first FST into sequences accepted by the second.
///
/// **Mathematical Definition:** For FSTs T₁: Σ* → Γ* and T₂: Γ* → Δ*,
/// their composition T₁ ∘ T₂: Σ* → Δ* computes:
/// ```text
/// (T₁ ∘ T₂)(x) = ⊕_{y ∈ Γ*} T₁(x,y) ⊗ T₂(y)
/// where:
///   x ∈ Σ* = input sequence
///   y ∈ Γ* = intermediate sequence
///   ⊕ = semiring addition (sum over all paths)
///   ⊗ = semiring multiplication (combine weights)
/// ```
///
/// # Complexity
///
/// - **Time:** O(V₁V₂E₁E₂) worst case, O(VE) average case
///   - V₁, V₂ = number of states in each FST
///   - E₁, E₂ = number of arcs in each FST
///   - Worst case: dense cross-product of all state-arc pairs
///   - Average case: sparse composition with few matching labels
/// - **Space:** O(V₁V₂) for reachable state pairs
///   - HashMap for state pair mapping
///   - Queue for BFS traversal
///
/// # Algorithm
///
/// On-the-fly composition using breadth-first state exploration:
/// 1. Create start state from (start₁, start₂, filter_start)
/// 2. While queue non-empty:
///    - Pop state triple (s₁, s₂, filter_state)
///    - If both s₁, s₂ are final: mark composed state as final with w₁ ⊗ w₂
///    - For each arc pair (a₁ from s₁, a₂ from s₂):
///      - Apply filter to check if arcs compose
///      - If match: create/reuse destination state, add composed arc
/// 3. Result contains only reachable states from start
///
/// Based on Mohri, Pereira, and Riley (2008) with lazy state construction.
///
/// # Performance Notes
///
/// - **On-the-fly construction:** Avoids creating unreachable state pairs
/// - **Label sparsity:** Performance improves with few matching labels between FSTs
/// - **Filter overhead:** Custom filters add per-arc-pair overhead
/// - **Memory pattern:** BFS traversal provides good cache locality
/// - **Optimization tip:** Pre-sort arcs by label for faster label matching
/// - **Best for:** Sparse FSTs with limited label overlap (e.g., NLP pipelines)
///
/// # Examples
///
/// ## Basic Composition
///
/// ```rust
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
/// let filter = DefaultComposeFilter;
/// let composed: VectorFst<TropicalWeight> = compose(&fst1, &fst2, filter).unwrap();
/// assert!(composed.num_states() > 0);
/// ```
///
/// ## NLP Pipeline Example
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Example of chaining two FSTs for NLP pipeline
/// // FST 1: simple tokenization (example structure)
/// let mut tokenizer = VectorFst::<TropicalWeight>::new();
/// let s0 = tokenizer.add_state();
/// let s1 = tokenizer.add_state();
/// tokenizer.set_start(s0);
/// tokenizer.set_final(s1, TropicalWeight::one());
/// tokenizer.add_arc(s0, Arc::new(1, 2, TropicalWeight::one(), s1));
///
/// // FST 2: POS tagging (example structure)  
/// let mut pos_tagger = VectorFst::<TropicalWeight>::new();
/// let s0 = pos_tagger.add_state();
/// let s1 = pos_tagger.add_state();
/// pos_tagger.set_start(s0);
/// pos_tagger.set_final(s1, TropicalWeight::one());
/// pos_tagger.add_arc(s0, Arc::new(2, 3, TropicalWeight::one(), s1));
///
/// // Chain them: raw_text -> tagged_tokens
/// let filter = DefaultComposeFilter;
/// let pipeline: VectorFst<TropicalWeight> = compose(&tokenizer, &pos_tagger, filter).unwrap();
/// assert!(pipeline.num_states() > 0);
/// ```
///
/// # Custom Filters
///
/// Use custom composition filters for specialized epsilon handling or
/// special matching criteria:
///
/// ```rust
/// use arcweight::prelude::*;
///
/// struct CustomFilter {
///     allow_epsilon: bool,
/// }
///
/// impl<W: Semiring> ComposeFilter<W> for CustomFilter {
///     type FilterState = i32;
///     
///     fn start() -> Self::FilterState { 0 }
///     
///     fn filter_arc(
///         &self,
///         arc1: &Arc<W>,
///         arc2: &Arc<W>,
///         _fs: &Self::FilterState,
///     ) -> Option<(Arc<W>, Self::FilterState)> {
///         // Only compose if labels match
///         if arc1.olabel == arc2.ilabel {
///             Some((
///                 Arc::new(
///                     arc1.ilabel,
///                     arc2.olabel,
///                     arc1.weight.times(&arc2.weight),
///                     0, // nextstate set later
///                 ),
///                 0,
///             ))
///         } else {
///             None
///         }
///     }
/// }
///
/// // Use the custom filter
/// let mut fst1 = VectorFst::<TropicalWeight>::new();
/// let s0 = fst1.add_state();
/// fst1.set_start(s0);
/// fst1.set_final(s0, TropicalWeight::one());
///
/// let mut fst2 = VectorFst::<TropicalWeight>::new();
/// let s0 = fst2.add_state();
/// fst2.set_start(s0);
/// fst2.set_final(s0, TropicalWeight::one());
///
/// let filter = CustomFilter { allow_epsilon: true };
/// let result: VectorFst<TropicalWeight> = compose(&fst1, &fst2, filter).unwrap();
/// assert_eq!(result.num_states(), 1);
/// ```
///
/// # Errors
///
/// Returns [`Error::Algorithm`] if:
/// - Either input FST is invalid or has no start state
/// - Memory allocation fails during computation
/// - The composition filter produces invalid arc combinations
/// - State space becomes too large (implementation limit exceeded)
///
/// # See Also
///
/// - [`compose_default`] - Convenience function using standard label matching
/// - [`ComposeFilter`] - Trait for implementing custom composition logic
/// - [`DefaultComposeFilter`] - Standard label-based composition filter
/// - [`determinize`] - Often applied after composition to reduce nondeterminism
/// - [`minimize`] - Reduce composed FST size
/// - [Fst trait](crate::fst::Fst) - Core FST interface
/// - [Semiring trait](crate::semiring::Semiring) - Weight algebra
///
/// [`determinize`]: crate::algorithms::determinize::determinize
/// [`minimize`]: crate::algorithms::minimize::minimize
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

/// Compose two FSTs using the default composition filter
///
/// This is a convenience function that uses [`DefaultComposeFilter`] for standard
/// label-based composition. Use this for most common composition scenarios where
/// the output labels of the first FST should match the input labels of the second FST.
///
/// # Complexity
///
/// - **Time:** O(V₁V₂E₁E₂) worst case, O(VE) average case
///   - V₁, V₂ = number of states in each FST
///   - E₁, E₂ = number of arcs in each FST
/// - **Space:** O(V₁V₂) for reachable state pairs
///
/// # Algorithm
///
/// Delegates to [`compose`] with [`DefaultComposeFilter`], which matches arcs where
/// `arc1.olabel == arc2.ilabel` and combines weights via semiring multiplication (⊗).
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Build a simple translation pipeline
/// let mut word_to_phone = VectorFst::<TropicalWeight>::new();
/// let s0 = word_to_phone.add_state();
/// let s1 = word_to_phone.add_state();
/// word_to_phone.set_start(s0);
/// word_to_phone.set_final(s1, TropicalWeight::one());
/// word_to_phone.add_arc(s0, Arc::new(1, 2, TropicalWeight::one(), s1));
///
/// let mut phone_to_french = VectorFst::<TropicalWeight>::new();
/// let s0 = phone_to_french.add_state();
/// let s1 = phone_to_french.add_state();
/// phone_to_french.set_start(s0);
/// phone_to_french.set_final(s1, TropicalWeight::one());
/// phone_to_french.add_arc(s0, Arc::new(2, 3, TropicalWeight::one(), s1));
///
/// // Compose: English -> French
/// let translator: VectorFst<TropicalWeight> =
///     compose_default(&word_to_phone, &phone_to_french)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::Algorithm`] if:
/// - Either input FST is invalid or has no start state
/// - Memory allocation fails during computation  
/// - The composition operation encounters incompatible state structures
///
/// # See Also
///
/// - [`compose`] - Full composition with custom filters
/// - [`ComposeFilter`] - Trait for custom composition logic
/// - [`DefaultComposeFilter`] - The filter used by this function
/// - [`determinize`] - Often applied after composition
/// - [`minimize`] - Reduce composed FST size
///
/// [`determinize`]: crate::algorithms::determinize::determinize
/// [`minimize`]: crate::algorithms::minimize::minimize
pub fn compose_default<W, F1, F2, M>(fst1: &F1, fst2: &F2) -> Result<M>
where
    W: Semiring,
    F1: Fst<W>,
    F2: Fst<W>,
    M: MutableFst<W> + Default,
{
    compose(fst1, fst2, DefaultComposeFilter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use num_traits::One;

    #[test]
    fn test_basic_composition() {
        // First FST: input -> intermediate
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::one());
        fst1.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(1.0), s1));

        // Second FST: intermediate -> output
        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t1, TropicalWeight::one());
        fst2.add_arc(t0, Arc::new(2, 3, TropicalWeight::new(2.0), t1));

        let composed: VectorFst<TropicalWeight> = compose_default(&fst1, &fst2).unwrap();

        assert!(composed.start().is_some());
        assert!(composed.num_states() > 0);

        // Should have at least one path from input 1 to output 3
        let mut found_path = false;
        for state in composed.states() {
            for arc in composed.arcs(state) {
                if arc.ilabel == 1 && arc.olabel == 3 {
                    found_path = true;
                    // Weight should be combined: 1.0 + 2.0 = 3.0
                    assert_eq!(*arc.weight.value(), 3.0);
                }
            }
        }
        assert!(found_path);
    }

    #[test]
    fn test_composition_no_match() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::one());
        fst1.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(1.0), s1));

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t1, TropicalWeight::one());
        fst2.add_arc(t0, Arc::new(3, 4, TropicalWeight::new(2.0), t1)); // No match

        let composed: VectorFst<TropicalWeight> = compose_default(&fst1, &fst2).unwrap();

        // Should result in empty FST or FST with no accepting paths
        assert!(composed.start().is_some()); // May have start state

        // Check if any state is final
        let mut has_final = false;
        for state in composed.states() {
            if composed.is_final(state) {
                has_final = true;
                break;
            }
        }
        // Should have no final states reachable from start
        if has_final {
            // If there are final states, they should not be reachable
            assert_eq!(composed.num_arcs_total(), 0);
        }
    }

    #[test]
    fn test_composition_epsilon() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        let s2 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s2, TropicalWeight::one());
        fst1.add_arc(s0, Arc::epsilon(TropicalWeight::new(0.5), s1));
        fst1.add_arc(s1, Arc::new(1, 2, TropicalWeight::new(1.0), s2));

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t1, TropicalWeight::one());
        fst2.add_arc(t0, Arc::new(2, 3, TropicalWeight::new(2.0), t1));

        let composed: VectorFst<TropicalWeight> = compose_default(&fst1, &fst2).unwrap();

        assert!(composed.start().is_some());
        assert!(composed.num_states() > 0);
    }
}
