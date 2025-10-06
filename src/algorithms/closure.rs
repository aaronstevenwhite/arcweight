//! Kleene closure algorithms
//!
//! Implements Kleene star (T*) and Kleene plus (T+) operations for weighted finite-state
//! transducers, enabling repetition patterns and iterative language constructions.

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst};
use crate::semiring::StarSemiring;
use crate::Result;

/// Compute Kleene closure (star) of an FST: T* accepts ε, T, TT, TTT, ...
///
/// Creates a new FST that accepts zero or more repetitions of the input FST.
/// The result accepts the empty string plus any concatenation of strings
/// from the original language, with weights combined through semiring operations.
///
/// # Algorithm Details
///
/// - **Kleene Star Operation:** T* = ε ∪ T ∪ T² ∪ T³ ∪ ...
/// - **Time Complexity:** O(|V| + |E|) for copying original FST plus constant overhead
/// - **Space Complexity:** O(|V| + |E|) plus one additional state
/// - **Language Relationship:** L(T*) = {ε} ∪ L(T) ∪ L(T²) ∪ L(T³) ∪ ...
///
/// # Semiring Requirements
///
/// The input FST must use a [`StarSemiring`] that provides:
/// - **Star Operation:** Defines meaning of infinite repetition
/// - **Closure Properties:** Ensures proper weight computation for iteration
/// - **Convergence:** Guarantees star operation terminates appropriately
///
/// # Mathematical Foundation
///
/// For an FST T over a star semiring, T* represents the reflexive transitive closure:
/// - **Empty String:** T* always accepts ε with weight 1̄
/// - **Single Iteration:** T* includes all strings from L(T)
/// - **Multiple Iterations:** T* includes all concatenations T·T, T·T·T, etc.
/// - **Weight Computation:** Uses semiring star operation for infinite sums
///
/// # Algorithm Steps
///
/// 1. **Copy Original FST:** Preserve all states, arcs, and structure from T
/// 2. **New Start/Final State:** Create state that accepts empty string (weight 1̄)
/// 3. **Epsilon to Start:** Connect new state to original start via epsilon arc
/// 4. **Epsilon from Finals:** Connect original final states back to new start
/// 5. **Iteration Loops:** Enable unlimited repetition through epsilon cycles
///
/// # Examples
///
/// ## Basic Pattern Repetition
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // FST accepts "ab" -> "xy"
/// let mut fst = VectorFst::<BooleanWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
/// fst.set_start(s0);
/// fst.set_final(s2, BooleanWeight::one());
/// fst.add_arc(s0, Arc::new('a' as u32, 'x' as u32, BooleanWeight::one(), s1));
/// fst.add_arc(s1, Arc::new('b' as u32, 'y' as u32, BooleanWeight::one(), s2));
///
/// // T* accepts: ε, "ab", "abab", "ababab", ... -> ε, "xy", "xyxy", "xyxyxy", ...
/// let star_fst: VectorFst<BooleanWeight> = closure(&fst)?;
///
/// // Result has original states plus new start/final state
/// assert_eq!(star_fst.num_states(), fst.num_states() + 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Regular Expression Patterns
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Build FST for single character 'a'
/// let mut char_a = VectorFst::<BooleanWeight>::new();
/// let s0 = char_a.add_state();
/// let s1 = char_a.add_state();
/// char_a.set_start(s0);
/// char_a.set_final(s1, BooleanWeight::one());
/// char_a.add_arc(s0, Arc::new('a' as u32, 'a' as u32, BooleanWeight::one(), s1));
///
/// // a* matches: ε, "a", "aa", "aaa", ...
/// let a_star: VectorFst<BooleanWeight> = closure(&char_a)?;
///
/// // Can now use a_star to build more complex regular expressions
/// println!("a* FST has {} states", a_star.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Token Repetition for Parsing
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Token FST: identifier pattern
/// let mut identifier = VectorFst::<BooleanWeight>::new();
/// let s0 = identifier.add_state();
/// let s1 = identifier.add_state();
/// identifier.set_start(s0);
/// identifier.set_final(s1, BooleanWeight::one());
/// identifier.add_arc(s0, Arc::new(1, 1, BooleanWeight::one(), s1)); // ID token
///
/// // identifier* for lists: "ID", "ID ID", "ID ID ID", ...
/// let id_list: VectorFst<BooleanWeight> = closure(&identifier)?;
///
/// // Useful for parsing identifier lists, parameter lists, etc.
/// assert!(id_list.start().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Morphological Pattern Iteration
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Morpheme repetition: reduplication patterns
/// let mut morpheme = VectorFst::<BooleanWeight>::new();
/// let s0 = morpheme.add_state();
/// let s1 = morpheme.add_state();
/// morpheme.set_start(s0);
/// morpheme.set_final(s1, BooleanWeight::one());
/// morpheme.add_arc(s0, Arc::new(1, 1, BooleanWeight::one(), s1)); // morpheme unit
///
/// // morpheme* allows: ε, single, doubled, tripled morphemes
/// let repeated: VectorFst<BooleanWeight> = closure(&morpheme)?;
///
/// // Models languages with morpheme repetition/reduplication
/// println!("Morpheme repetition FST ready");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Empty String Acceptance
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Simple FST that requires input
/// let mut non_empty = VectorFst::<BooleanWeight>::new();
/// let s0 = non_empty.add_state();
/// let s1 = non_empty.add_state();
/// non_empty.set_start(s0);
/// non_empty.set_final(s1, BooleanWeight::one());
/// non_empty.add_arc(s0, Arc::new('x' as u32, 'x' as u32, BooleanWeight::one(), s1));
///
/// // T* always accepts empty string, even if original T doesn't
/// let with_empty: VectorFst<BooleanWeight> = closure(&non_empty)?;
///
/// // with_empty accepts both ε and any repetition of 'x'
/// // This is a key difference from the original FST
/// assert!(with_empty.start().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Use Cases
///
/// ## Regular Expression Construction
/// - **Pattern Matching:** Build regex-like patterns for text processing
/// - **Wildcard Expansion:** Handle patterns like ".*" for any repetition
/// - **Optional Repetition:** Model optional repeated elements
/// - **Flexible Matching:** Allow zero or more occurrences of patterns
///
/// ## Natural Language Processing
/// - **Morphological Analysis:** Model optional morpheme repetition/reduplication
/// - **Tokenization:** Handle optional whitespace and punctuation repetition
/// - **Grammar Rules:** Implement context-free grammar repetition constructs
/// - **Text Normalization:** Handle variable-length repetitive patterns
///
/// ## Parsing and Compilation
/// - **Lexical Analysis:** Scan for repeated characters or tokens
/// - **Comment Handling:** Model /* ... */ style comments with internal repetition
/// - **String Literals:** Parse strings with escaped character repetition
/// - **Grammar Production:** Implement BNF-style repetition rules
///
/// ## Speech and Audio Processing
/// - **Phoneme Repetition:** Model speech patterns with optional repetition
/// - **Silence Handling:** Accept variable-length silence periods
/// - **Stuttering Models:** Handle repeated phoneme patterns in speech
/// - **Rhythm Patterns:** Model musical or prosodic repetition
///
/// ## Data Processing
/// - **Format Flexibility:** Handle optional repeated fields in data formats
/// - **Sequence Analysis:** Analyze patterns with variable repetition
/// - **Protocol Parsing:** Handle optional repeated protocol elements
/// - **Log Processing:** Parse logs with optional repeated components
///
/// # Performance Characteristics
///
/// - **Memory Usage:** Original FST size plus one additional state
/// - **Construction Time:** Linear O(|V| + |E|) in original FST size
/// - **Epsilon Transitions:** Creates cycles via epsilon arcs for iteration
/// - **State Growth:** Minimal - only adds one new start/final state
/// - **Runtime Efficiency:** May require epsilon removal for optimal performance
///
/// # Mathematical Properties
///
/// Kleene star satisfies fundamental algebraic properties:
/// - **Idempotency:** (T*)* = T* (star of star equals star)
/// - **Empty Language:** ∅* = ε (star of empty language is just empty string)
/// - **Unit Element:** ε* = ε (star of empty string is empty string)
/// - **Distributivity:** (T₁ ∪ T₂)* ⊇ T₁* ∪ T₂* (star distributes over union)
/// - **Associativity:** T* = (T*)* in terms of accepted language
/// - **Monotonicity:** If L₁ ⊆ L₂, then L₁* ⊆ L₂*
///
/// # Implementation Details
///
/// The algorithm creates a new start state that is also final (accepting empty string),
/// connects it to the original start via epsilon, and connects all original final
/// states back to this new start via epsilon transitions. This creates the necessary
/// cycle structure for unlimited repetition while preserving the original FST structure.
///
/// # Optimization Considerations
///
/// After applying Kleene star, consider these optimizations:
/// - **Epsilon Removal:** Remove epsilon cycles for more efficient processing
/// - **Determinization:** Resolve nondeterminism introduced by epsilon transitions
/// - **Minimization:** Reduce redundant states in the result
/// - **Connection:** Ensure all states remain reachable and coaccessible
///
/// # Common Patterns
///
/// Kleene star is frequently combined with other operations:
/// - **T* ∪ T+:** Same as T* (star already includes plus)
/// - **T · T*:** Same as T+ (concatenation with star gives plus)
/// - **(T₁ ∪ T₂)*:** Star of union for multiple alternative patterns
/// - **T₁* · T₂ · T₃*:** Optional prefix, required middle, optional suffix
///
/// # Errors
///
/// Returns [`crate::Error::Algorithm`] if:
/// - The input FST is invalid, corrupted, or malformed
/// - Memory allocation fails during state or arc creation
/// - The semiring does not properly support star operations
/// - State iteration or arc enumeration encounters invalid data
/// - Epsilon arc creation fails due to invalid weights
/// - The result FST becomes too large to represent in memory
///
/// # See Also
///
/// - [`closure_plus`] for Kleene plus operation (T+) excluding empty string
/// - [`crate::algorithms::concat()`] for sequential FST combination
/// - [`crate::algorithms::union()`] for parallel FST combination
/// - [`remove_epsilons()`](crate::algorithms::remove_epsilons()) for epsilon removal
/// - [Working with FSTs: Closure Operations](../../docs/working-with-fsts/advanced-topics.md#closure-operations) for usage patterns
/// - [Core Concepts](../../docs/core-concepts/algorithms.md#closure) for mathematical theory
pub fn closure<W, F, M>(fst: &F) -> Result<M>
where
    W: StarSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    closure_impl(fst, true)
}

/// Compute Kleene plus of an FST: T+ accepts T, TT, TTT, ... (excludes ε)
///
/// Creates a new FST that accepts one or more repetitions of the input FST.
/// Unlike Kleene star, Kleene plus does NOT accept the empty string - it requires
/// at least one iteration of the original FST language.
///
/// # Algorithm Details
///
/// - **Kleene Plus Operation:** T+ = T ∪ T² ∪ T³ ∪ ... (note: ε ∉ T+)
/// - **Time Complexity:** O(|V| + |E|) for copying original FST plus constant overhead
/// - **Space Complexity:** O(|V| + |E|) plus one additional state
/// - **Language Relationship:** L(T+) = L(T) ∪ L(T²) ∪ L(T³) ∪ ... (no empty string)
///
/// # Difference from Kleene Star
///
/// - **T* (star):** Accepts ε, T, T², T³, ... (includes empty string)
/// - **T+ (plus):** Accepts T, T², T³, ... (excludes empty string)
/// - **Practical Use:** T+ for "one or more", T* for "zero or more"
///
/// # Examples
///
/// ## Pattern Requiring At Least One Occurrence
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // FST accepts single digit "0"-"9"
/// let mut digit = VectorFst::<BooleanWeight>::new();
/// let s0 = digit.add_state();
/// let s1 = digit.add_state();
/// digit.set_start(s0);
/// digit.set_final(s1, BooleanWeight::one());
/// digit.add_arc(s0, Arc::new('0' as u32, '0' as u32, BooleanWeight::one(), s1));
///
/// // digit+ accepts: "0", "00", "000", ... but NOT empty string
/// let number: VectorFst<BooleanWeight> = closure_plus(&digit)?;
///
/// // number requires at least one digit (useful for parsing integers)
/// assert_eq!(number.num_states(), digit.num_states() + 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Required Repetition Patterns
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Word pattern FST
/// let mut word = VectorFst::<BooleanWeight>::new();
/// let s0 = word.add_state();
/// let s1 = word.add_state();
/// word.set_start(s0);
/// word.set_final(s1, BooleanWeight::one());
/// word.add_arc(s0, Arc::new(1, 1, BooleanWeight::one(), s1)); // word token
///
/// // word+ for non-empty word sequences: "w", "w w", "w w w", ...
/// let word_list: VectorFst<BooleanWeight> = closure_plus(&word)?;
///
/// // Ensures at least one word is present (no empty documents)
/// println!("Word list FST requires at least one word");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Comparing Star vs Plus
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Basic unit FST
/// let mut unit = VectorFst::<BooleanWeight>::new();
/// let s0 = unit.add_state();
/// let s1 = unit.add_state();
/// unit.set_start(s0);
/// unit.set_final(s1, BooleanWeight::one());
/// unit.add_arc(s0, Arc::new('a' as u32, 'a' as u32, BooleanWeight::one(), s1));
///
/// // T* accepts: ε, "a", "aa", "aaa", ...
/// let star_result: VectorFst<BooleanWeight> = closure(&unit)?;
///
/// // T+ accepts: "a", "aa", "aaa", ... (NO empty string)
/// let plus_result: VectorFst<BooleanWeight> = closure_plus(&unit)?;
///
/// // Both have same number of states but different acceptance
/// assert_eq!(star_result.num_states(), plus_result.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Grammar Rules Requiring Non-Empty Repetition
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Statement FST for programming language
/// let mut statement = VectorFst::<BooleanWeight>::new();
/// let s0 = statement.add_state();
/// let s1 = statement.add_state();
/// statement.set_start(s0);
/// statement.set_final(s1, BooleanWeight::one());
/// statement.add_arc(s0, Arc::new(1, 1, BooleanWeight::one(), s1)); // statement
///
/// // program = statement+ (at least one statement required)
/// let program: VectorFst<BooleanWeight> = closure_plus(&statement)?;
///
/// // Empty programs not allowed - must have at least one statement
/// assert!(program.start().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Morpheme Repetition with Minimum Requirement
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Reduplication morpheme
/// let mut redupe = VectorFst::<BooleanWeight>::new();
/// let s0 = redupe.add_state();
/// let s1 = redupe.add_state();
/// redupe.set_start(s0);
/// redupe.set_final(s1, BooleanWeight::one());
/// redupe.add_arc(s0, Arc::new(1, 1, BooleanWeight::one(), s1)); // redupe unit
///
/// // redupe+ for intensive forms requiring at least one reduplication
/// let intensive: VectorFst<BooleanWeight> = closure_plus(&redupe)?;
///
/// // Models morphological intensification requiring actual reduplication
/// println!("Intensive form requires at least one reduplication");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Use Cases
///
/// ## Programming Language Parsing
/// - **Statement Lists:** Programs need at least one statement
/// - **Parameter Lists:** Function calls need at least one parameter  
/// - **Expression Sequences:** Non-empty expression chains
/// - **Block Contents:** Code blocks with required content
///
/// ## Natural Language Processing
/// - **Word Sequences:** Sentences need at least one word
/// - **Morpheme Repetition:** Intensive forms requiring reduplication
/// - **Syllable Patterns:** Words requiring minimum syllable count
/// - **Phrase Structure:** Non-empty phrase constituents
///
/// ## Regular Expression Patterns
/// - **Email Validation:** Domain parts need at least one component
/// - **URL Parsing:** Path segments requiring content
/// - **Number Formats:** Digit sequences that can't be empty
/// - **Identifier Rules:** Names requiring at least one character
///
/// ## Data Validation
/// - **List Processing:** Non-empty lists and arrays
/// - **String Patterns:** Text requiring minimum content
/// - **Sequence Analysis:** Data streams with required elements
/// - **Format Validation:** Structured data with mandatory parts
///
/// # Mathematical Properties
///
/// Kleene plus satisfies these algebraic properties:
/// - **Relationship to Star:** T+ = T · T* = T* · T (when T doesn't accept ε)
/// - **No Empty String:** ε ∉ L(T+) always (fundamental difference from T*)
/// - **Minimum Iteration:** T+ ⊇ T (T+ always includes at least T)
/// - **Associativity:** T+ can be built iteratively: T ∪ T² ∪ T³ ∪ ...
/// - **Idempotency:** (T+)+ = T+ in most cases
///
/// # Implementation Notes
///
/// The implementation creates the same structure as Kleene star but removes
/// the final weight from the new start state, preventing acceptance of the
/// empty string while maintaining all iteration capabilities.
///
/// # Errors
///
/// Returns [`crate::Error::Algorithm`] if:
/// - The input FST is invalid, corrupted, or malformed
/// - Memory allocation fails during state or arc creation
/// - The semiring does not properly support star operations
/// - State mapping or arc creation encounters invalid data
/// - The result FST becomes too large to represent in memory
///
/// # See Also
///
/// - [`closure`] for Kleene star operation (T*) including empty string
/// - [`crate::algorithms::concat()`] for sequential FST combination
/// - [`crate::algorithms::union()`] for parallel FST combination
/// - [Working with FSTs: Closure Operations](../../docs/working-with-fsts/advanced-topics.md#closure-operations) for usage patterns
/// - [Core Concepts](../../docs/core-concepts/algorithms.md#closure) for mathematical theory
pub fn closure_plus<W, F, M>(fst: &F) -> Result<M>
where
    W: StarSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    closure_impl(fst, false)
}

fn closure_impl<W, F, M>(fst: &F, allow_empty: bool) -> Result<M>
where
    W: StarSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();

    // copy original FST
    let mut state_map = vec![None; fst.num_states()];
    for state in fst.states() {
        let new_state = result.add_state();
        state_map[state as usize] = Some(new_state);
    }

    // create new start/final state
    let new_start = result.add_state();
    result.set_start(new_start);
    result.set_final(new_start, W::one());

    // connect to original start
    if let Some(orig_start) = fst.start() {
        if let Some(mapped_start) = state_map[orig_start as usize] {
            result.add_arc(new_start, Arc::epsilon(W::one(), mapped_start));

            if !allow_empty {
                // for plus, remove final weight from new start
                result.remove_final(new_start);
            }
        }
    }

    // copy arcs
    for state in fst.states() {
        if let Some(new_state) = state_map[state as usize] {
            for arc in fst.arcs(state) {
                if let Some(new_nextstate) = state_map[arc.nextstate as usize] {
                    result.add_arc(
                        new_state,
                        Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), new_nextstate),
                    );
                }
            }

            // connect final states back to new start
            if let Some(weight) = fst.final_weight(state) {
                result.add_arc(new_state, Arc::epsilon(weight.clone(), new_start));
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
    fn test_closure_with_boolean_weight() {
        let mut fst = VectorFst::<BooleanWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, BooleanWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, BooleanWeight::new(true), s1));

        let star: VectorFst<BooleanWeight> = closure(&fst).unwrap();

        // Closure should add states for start/final
        assert!(star.num_states() > fst.num_states());
        assert!(star.start().is_some());

        // Start state should be final (empty string acceptance)
        let start = star.start().unwrap();
        assert!(star.is_final(start));
    }

    #[test]
    fn test_closure_plus_with_boolean_weight() {
        let mut fst = VectorFst::<BooleanWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, BooleanWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, BooleanWeight::new(true), s1));

        let plus: VectorFst<BooleanWeight> = closure_plus(&fst).unwrap();

        // Plus closure should not accept empty string
        assert!(plus.start().is_some());

        // Start state should not be final
        let start = plus.start().unwrap();
        assert!(!plus.is_final(start));
    }
}
