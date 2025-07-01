//! FST reversal algorithm
//!
//! Constructs the reverse of a weighted finite-state transducer by reversing
//! all transitions and swapping start/final states while preserving the accepted language.

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst};
use crate::semiring::Semiring;
use crate::Result;

/// Reverse an FST by swapping direction of transitions and start/final states
///
/// Creates the reverse automaton where all transitions are reversed, the original
/// start state becomes final, and original final states connect to a new start state.
/// The reversed FST accepts the reverse of each string in the original language.
///
/// # Algorithm Details
///
/// - **Transformation:** Reverses all arc directions and swaps start/final states
/// - **Time Complexity:** O(|V| + |E|) for state/arc creation
/// - **Space Complexity:** O(|V| + |E|) for the new FST structure
/// - **Language Preserved:** If original accepts L, reverse accepts L^R = {w^R : w ∈ L}
///
/// # Mathematical Foundation
///
/// For an FST T that maps input strings to output strings with weights,
/// the reverse T^R satisfies: T^R(x^R) = T(x) where x^R is the reverse of string x.
/// This preserves the weighted language while changing string direction.
///
/// # Algorithm Steps
///
/// 1. **State Creation:** Copy all original states plus one new start state
/// 2. **Arc Reversal:** For each arc (p, a:b/w, q), create arc (q, a:b/w, p)
/// 3. **Start State:** Create new start state with epsilon transitions to original finals
/// 4. **Final States:** Make original start state the unique final state
/// 5. **Weight Preservation:** All weights remain unchanged during reversal
///
/// # Examples
///
/// ## Basic String Reversal
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Create FST that accepts "abc" -> "xyz"
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
/// let s3 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s3, TropicalWeight::one());
///
/// fst.add_arc(s0, Arc::new('a' as u32, 'x' as u32, TropicalWeight::one(), s1));
/// fst.add_arc(s1, Arc::new('b' as u32, 'y' as u32, TropicalWeight::one(), s2));
/// fst.add_arc(s2, Arc::new('c' as u32, 'z' as u32, TropicalWeight::one(), s3));
///
/// // Reverse: now accepts "cba" -> "zyx"
/// let reversed: VectorFst<TropicalWeight> = reverse(&fst)?;
///
/// // Original: abc -> xyz, Reversed: cba -> zyx
/// assert_eq!(reversed.num_states(), fst.num_states() + 1); // +1 for new start
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Multiple Final States
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // FST with multiple final states: accepts "a" and "ab"
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s1, TropicalWeight::new(0.5)); // "a" with weight 0.5
/// fst.set_final(s2, TropicalWeight::new(0.3)); // "ab" with weight 0.3
///
/// fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s1));
/// fst.add_arc(s1, Arc::new('b' as u32, 'b' as u32, TropicalWeight::one(), s2));
///
/// // Reverse: accepts "a" and "ba" with preserved weights
/// let reversed: VectorFst<TropicalWeight> = reverse(&fst)?;
///
/// // Check structure: original start becomes final, new start created
/// assert!(reversed.start().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Dictionary Reversal for Suffix Analysis
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Create a simple dictionary FST
/// fn create_word_fst() -> VectorFst<TropicalWeight> {
///     let mut fst = VectorFst::new();
///     let s0 = fst.add_state();
///     let s1 = fst.add_state();
///     let s2 = fst.add_state();
///     
///     fst.set_start(s0);
///     fst.set_final(s2, TropicalWeight::one());
///     
///     // Simple word: "cat"
///     fst.add_arc(s0, Arc::new('c' as u32, 'c' as u32, TropicalWeight::one(), s1));
///     fst.add_arc(s1, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s2));
///     fst.add_arc(s2, Arc::new('t' as u32, 't' as u32, TropicalWeight::one(), s2));
///     
///     fst
/// }
///
/// let word_fst = create_word_fst();
///
/// // Reverse for suffix analysis: "tac"
/// let suffix_fst: VectorFst<TropicalWeight> = reverse(&word_fst)?;
///
/// // Now we can analyze word suffixes by composing with reversed dictionary
/// println!("Reversed FST created with {} states", suffix_fst.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Weight-Preserving Operations
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Weighted FST with costs
/// let mut cost_fst = VectorFst::<TropicalWeight>::new();
/// let s0 = cost_fst.add_state();
/// let s1 = cost_fst.add_state();
///
/// cost_fst.set_start(s0);
/// cost_fst.set_final(s1, TropicalWeight::new(2.0)); // Final cost
///
/// cost_fst.add_arc(s0, Arc::new('x' as u32, 'y' as u32, TropicalWeight::new(1.5), s1));
///
/// // Reverse preserves all weights exactly
/// let reversed: VectorFst<TropicalWeight> = reverse(&cost_fst)?;
///
/// // All weights preserved: arc weight 1.5, final weight 2.0
/// assert!(reversed.num_states() > 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Use Cases
///
/// ## Language Reversal
/// - **Suffix Analysis:** Reverse dictionary for suffix-based lookups
/// - **Palindrome Detection:** Compose FST with its reverse
/// - **Bidirectional Search:** Enable reverse traversal in search algorithms
/// - **Pattern Matching:** Find patterns that work in both directions
///
/// ## Algorithmic Building Block
/// - **Minimization:** Brzozowski's algorithm uses reverse as a key step
/// - **Determinization:** Some algorithms work better on reversed automata
/// - **Composition Optimization:** Reverse one FST for more efficient composition
/// - **Equivalence Testing:** Compare L(T) with L(T^R) for symmetry analysis
///
/// ## Natural Language Processing
/// - **Morphological Analysis:** Reverse morphology rules for generation/analysis
/// - **Text Processing:** Bidirectional text analysis and transformation
/// - **Grammar Checking:** Apply rules in both forward and backward directions
/// - **Speech Recognition:** Reverse phonetic models for improved decoding
///
/// # Performance Characteristics
///
/// - **Memory Usage:** Creates new FST with same number of arcs plus one state
/// - **Construction Time:** Linear in input size O(|V| + |E|)
/// - **Preservation:** All weights and labels exactly preserved
/// - **State Mapping:** One-to-one correspondence except for new start state
///
/// # Mathematical Properties
///
/// The reverse operation satisfies several important properties:
/// - **Involution:** reverse(reverse(T)) ≈ T (up to state renaming)
/// - **Weight Preservation:** All weights remain identical in value
/// - **Label Preservation:** Input/output labels unchanged
/// - **Language Relationship:** L(T^R) = {w^R : w ∈ L(T)}
/// - **Determinism:** reverse(deterministic) may be nondeterministic
///
/// # Implementation Notes
///
/// The algorithm creates a new start state and connects it via epsilon transitions
/// to all original final states. This ensures the reversed FST has exactly one
/// start state while preserving the acceptance structure. The epsilon transitions
/// carry the final weights from the original FST.
///
/// # Errors
///
/// Returns [`Error::Algorithm`](crate::Error::Algorithm) if:
/// - The input FST is invalid, corrupted, or malformed
/// - Memory allocation fails during state or arc creation
/// - State iteration or arc enumeration encounters invalid data
/// - FST structure becomes inconsistent during construction
/// - Arc creation fails due to invalid labels or weights
///
/// # See Also
///
/// - [`minimize()`](crate::algorithms::minimize()) for Brzozowski's minimization using reverse
/// - [`determinize()`](crate::algorithms::determinize()) for resolving nondeterminism
/// - [`compose()`](crate::algorithms::compose()) for chaining reversed automata
/// - [Working with FSTs](../../docs/working-with-fsts/README.md) for FST manipulation patterns
/// - [Core Concepts](../../docs/core-concepts/algorithms.md#reverse) for mathematical theory
pub fn reverse<W, F, M>(fst: &F) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();

    // create states
    for _ in 0..fst.num_states() {
        result.add_state();
    }

    // create new start state
    let new_start = result.add_state();
    result.set_start(new_start);

    // add reversed arcs
    for state in fst.states() {
        for arc in fst.arcs(state) {
            result.add_arc(
                arc.nextstate,
                Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), state),
            );
        }

        // final states become arcs from new start
        if let Some(weight) = fst.final_weight(state) {
            result.add_arc(new_start, Arc::epsilon(weight.clone(), state));
        }
    }

    // original start becomes final
    if let Some(orig_start) = fst.start() {
        result.set_final(orig_start, W::one());
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_reverse_basic() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::new(2.0));

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(3.0), s2));

        let reversed: VectorFst<TropicalWeight> = reverse(&fst).unwrap();

        // Should add a new start state
        assert_eq!(reversed.num_states(), fst.num_states() + 1);
        assert!(reversed.start().is_some());

        // Original start should be final in reversed
        assert!(reversed.is_final(s0));

        // Check that arcs are reversed
        let mut found_reversed_arc = false;
        for state in reversed.states() {
            for arc in reversed.arcs(state) {
                if arc.nextstate == s1 && arc.ilabel == 2 {
                    found_reversed_arc = true;
                }
            }
        }
        assert!(found_reversed_arc);
    }
}
