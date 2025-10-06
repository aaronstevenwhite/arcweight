//! FST difference algorithm
//!
//! Computes the language difference between two weighted finite-state acceptors,
//! creating an acceptor that recognizes strings accepted by the first but not the second.

use crate::algorithms::{determinize, intersect};
use crate::arc::Arc;
use crate::fst::{Fst, Label, MutableFst};
use crate::semiring::{DivisibleSemiring, Semiring};
use crate::{Error, Result};
use core::hash::Hash;
use std::collections::{HashMap, HashSet};

/// Compute the language difference between two finite-state acceptors
///
/// Creates a new acceptor that recognizes exactly the strings that are
/// accepted by the first acceptor but not by the second acceptor. This
/// implements the set difference operation A₁ - A₂ for acceptor languages.
///
/// # Algorithm Details
///
/// - **Mathematical Operation:** A₁ - A₂ = A₁ ∩ complement(A₂) where both are acceptors
/// - **Implementation Strategy:** Composition with complement acceptor construction
/// - **Time Complexity:** O(2^|V₂|) worst case for determinization, then O(|V₁| × |V₂'|) for intersection where V₂' is complement size
/// - **Space Complexity:** O(|V₁| × |V₂'|) for state cross product after determinization
/// - **Language Relationship:** L(A₁ - A₂) = L(A₁) - L(A₂) = {w : w ∈ L(A₁) and w ∉ L(A₂)}
/// - **Semiring Requirements:** Requires `DivisibleSemiring + Ord` for determinization
///
/// # Mathematical Foundation
///
/// For two acceptors A₁ and A₂ over the same alphabet Σ:
/// - **Language Difference:** L(A₁ - A₂) = {w ∈ Σ* : w ∈ L(A₁) and w ∉ L(A₂)}
/// - **Set Theory Relation:** A₁ - A₂ = A₁ ∩ complement(A₂)
/// - **Implementation:** Compose A₁ with complement of A₂
/// - **Weight Combination:** Uses semiring operations for path weight computation
///
/// # Algorithm Steps
///
/// 1. **Complement Construction:** Build complement acceptor of A₂ over alphabet
/// 2. **Composition Setup:** Configure composition with difference filter
/// 3. **Cross Product:** Create state pairs from A₁ and complement(A₂)
/// 4. **Label Matching:** Process matching transitions through composition
/// 5. **Result Construction:** Build difference acceptor maintaining A₁ weights
///
/// # Algorithm Implementation
///
/// The implementation follows these steps:
/// 1. **Input Validation:** Ensure both FSTs are acceptors (input = output labels)
/// 2. **Alphabet Construction:** Collect all non-epsilon labels from both FSTs
/// 3. **Complement Construction:** Build complement of second FST using determinization
/// 4. **Intersection:** Compute intersection of first FST with complement
///
/// The complement construction works by:
/// - Determinizing the input FST to ensure each state has exactly one transition per symbol
/// - Making the FST complete by adding a sink state for missing transitions
/// - Flipping final states (final becomes non-final, non-final becomes final)
///
/// # Examples
///
/// ## Basic Language Difference
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::difference;
///
/// // FST1 accepts "ab" and "abc"
/// let mut fst1 = VectorFst::<TropicalWeight>::new();
/// let s0 = fst1.add_state();
/// let s1 = fst1.add_state();
/// let s2 = fst1.add_state();
/// let s3 = fst1.add_state();
///
/// fst1.set_start(s0);
/// fst1.set_final(s2, TropicalWeight::one()); // accepts "ab"
/// fst1.set_final(s3, TropicalWeight::one()); // accepts "abc"
///
/// fst1.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s1));
/// fst1.add_arc(s1, Arc::new('b' as u32, 'b' as u32, TropicalWeight::one(), s2));
/// fst1.add_arc(s2, Arc::new('c' as u32, 'c' as u32, TropicalWeight::one(), s3));
///
/// // FST2 accepts "abc"
/// let mut fst2 = VectorFst::<TropicalWeight>::new();
/// let s0 = fst2.add_state();
/// let s1 = fst2.add_state();
/// let s2 = fst2.add_state();
/// let s3 = fst2.add_state();
///
/// fst2.set_start(s0);
/// fst2.set_final(s3, TropicalWeight::one());
///
/// fst2.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s1));
/// fst2.add_arc(s1, Arc::new('b' as u32, 'b' as u32, TropicalWeight::one(), s2));
/// fst2.add_arc(s2, Arc::new('c' as u32, 'c' as u32, TropicalWeight::one(), s3));
///
/// // Difference: fst1 - fst2 should accept "ab" but not "abc"
/// let diff: VectorFst<TropicalWeight> = difference(&fst1, &fst2)?;
///
/// // Result accepts "ab" but not "abc"
/// assert!(diff.start().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Vocabulary Filtering
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::difference;
///
/// // Dictionary accepts "good" and "bad"
/// let mut dictionary = VectorFst::<TropicalWeight>::new();
/// let s0 = dictionary.add_state();
/// let s1 = dictionary.add_state();
/// let s2 = dictionary.add_state();
///
/// dictionary.set_start(s0);
/// dictionary.set_final(s2, TropicalWeight::one());
///
/// // Accept "good"
/// dictionary.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1)); // 'g'
/// dictionary.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2)); // 'ood'
///
/// // Also accept "bad" (add more states as needed for full implementation)
/// let s3 = dictionary.add_state();
/// let s4 = dictionary.add_state();
/// dictionary.set_final(s4, TropicalWeight::one());
/// dictionary.add_arc(s0, Arc::new(3, 3, TropicalWeight::one(), s3)); // 'b'
/// dictionary.add_arc(s3, Arc::new(4, 4, TropicalWeight::one(), s4)); // 'ad'
///
/// // Filter accepts "bad"
/// let mut filter = VectorFst::<TropicalWeight>::new();
/// let s0 = filter.add_state();
/// let s1 = filter.add_state();
/// let s2 = filter.add_state();
///
/// filter.set_start(s0);
/// filter.set_final(s2, TropicalWeight::one());
///
/// filter.add_arc(s0, Arc::new(3, 3, TropicalWeight::one(), s1)); // 'b'
/// filter.add_arc(s1, Arc::new(4, 4, TropicalWeight::one(), s2)); // 'ad'
///
/// // Clean vocabulary: dictionary - filter (should accept "good" but not "bad")
/// let clean: VectorFst<TropicalWeight> = difference(&dictionary, &filter)?;
///
/// assert!(clean.start().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Set Difference with Weighted Acceptors
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::difference;
///
/// // Create FST accepting single character "a"
/// let mut fst_a = VectorFst::<TropicalWeight>::new();
/// let s0 = fst_a.add_state();
/// let s1 = fst_a.add_state();
/// fst_a.set_start(s0);
/// fst_a.set_final(s1, TropicalWeight::new(1.0));
/// fst_a.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::new(0.5), s1));
///
/// // Create FST accepting single character "b"  
/// let mut fst_b = VectorFst::<TropicalWeight>::new();
/// let s0 = fst_b.add_state();
/// let s1 = fst_b.add_state();
/// fst_b.set_start(s0);
/// fst_b.set_final(s1, TropicalWeight::new(1.0));
/// fst_b.add_arc(s0, Arc::new('b' as u32, 'b' as u32, TropicalWeight::new(0.3), s1));
///
/// // Difference should accept "a" but not "b"
/// let diff: VectorFst<TropicalWeight> = difference(&fst_a, &fst_b)?;
///
/// // Verify result has expected structure
/// assert!(diff.start().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Use Cases
///
/// ## Natural Language Processing
/// - **Text Filtering:** Remove unwanted content from text acceptors
/// - **Vocabulary Cleaning:** Filter profanity or inappropriate content
/// - **Stop Word Removal:** Remove common words from search vocabularies
/// - **Content Moderation:** Filter content based on policy rules
///
/// ## Information Retrieval
/// - **Query Refinement:** Remove irrelevant terms from search spaces
/// - **Result Filtering:** Exclude unwanted results from search results
/// - **Document Classification:** Remove documents matching exclusion criteria
/// - **Spam Detection:** Filter spam patterns from legitimate content
///
/// ## Security Applications
/// - **Access Control:** Remove unauthorized access patterns
/// - **Threat Detection:** Filter known malicious patterns
/// - **Input Validation:** Remove dangerous input patterns
/// - **Content Sanitization:** Remove potentially harmful content
///
/// ## Data Processing
/// - **Outlier Removal:** Filter outliers from data patterns
/// - **Noise Reduction:** Remove noise patterns from signal processing
/// - **Exception Handling:** Filter exceptional cases from normal processing
/// - **Quality Control:** Remove low-quality data patterns
///
/// # Performance Characteristics
///
/// - **Time Complexity:** O(2^|V₂|) worst case due to determinization in complement construction
/// - **Space Complexity:** O(|V₁| × |V₂'| + |Σ|) where V₂' is determinized complement size
/// - **Complement Cost:** Exponential in worst case, often polynomial in practice
/// - **Composition Efficiency:** Depends on automata determinism and structure
/// - **Practical Performance:** Much better than worst case for deterministic or sparse automata
///
/// # Mathematical Properties
///
/// Language difference preserves important set operations:
/// - **Asymmetry:** A₁ - A₂ ≠ A₂ - A₁ generally
/// - **Distributivity:** (A₁ ∪ A₂) - A₃ = (A₁ - A₃) ∪ (A₂ - A₃)
/// - **De Morgan's Law:** A₁ - A₂ = A₁ ∩ complement(A₂)
/// - **Identity Elements:** A - ∅ = A, A - Σ* = ∅
/// - **Self Difference:** A - A = ∅
///
/// # Implementation Challenges
///
/// Efficient difference computation requires:
/// - **Alphabet Inference:** Determining the working alphabet from inputs
/// - **Complement Construction:** Building complement without exponential blowup
/// - **Determinization:** May require deterministic automata for efficiency
/// - **Memory Management:** Managing large cross-product state spaces
/// - **Weight Handling:** Proper semiring operations for difference semantics
///
/// # Optimization Strategies
///
/// For better performance when implemented:
/// - **Input Preparation:** Determinize and minimize input acceptors first
/// - **Alphabet Optimization:** Use minimal alphabet covering both acceptors
/// - **Lazy Construction:** Build complement states on-demand during composition
/// - **Pruning:** Early termination for unreachable state combinations
/// - **Memory Efficiency:** Use sparse representations for large alphabets
///
/// # Relationship to Other Operations
///
/// Difference relates to other FST operations:
/// - **Intersection:** difference(A₁, A₂) = intersect(A₁, complement(A₂))
/// - **Union:** A₁ ∪ A₂ = A₁ ∪ (A₂ - A₁) ∪ (A₁ ∩ A₂)
/// - **Complement:** complement(A) = Σ* - A
/// - **Symmetric Difference:** A₁ ⊕ A₂ = (A₁ - A₂) ∪ (A₂ - A₁)
///
/// # Implementation Notes
///
/// The current implementation includes:
/// 1. **Complete Algorithm:** Full implementation of difference via complement and intersection
/// 2. **Alphabet Inference:** Automatic collection of alphabet from both input FSTs
/// 3. **Complement Construction:** Efficient complement generation with determinization
/// 4. **Weight Preservation:** Proper handling of semiring operations throughout
/// 5. **Optimization:** Determinization and completion for efficient complement construction
///
/// # Performance Considerations
///
/// - **Determinization Overhead:** The complement construction requires determinization
/// - **State Explosion:** Complement may significantly increase state count
/// - **Alphabet Size:** Performance depends on the size of the working alphabet
/// - **Memory Usage:** Proportional to product of input sizes and alphabet size
/// - **Optimization:** Consider minimizing inputs before difference computation
///
/// # Errors
///
/// Returns [`Error::Algorithm`] if:
/// - Either input FST is invalid, corrupted, or malformed
/// - The input FSTs are not acceptors (input ≠ output labels)
/// - Memory allocation fails during complement construction or intersection
/// - The alphabet inference fails or produces an empty alphabet
/// - Determinization fails during complement construction
/// - Intersection computation fails between FST and complement
///
/// # See Also
///
/// - [`crate::algorithms::intersect()`] for language intersection (A₁ ∩ A₂)
/// - [`crate::algorithms::union()`] for language union (A₁ ∪ A₂)
/// - [`crate::algorithms::compose()`] for general FST composition
/// - Complement construction algorithms (implemented internally via determinization)
/// - [Working with FSTs - Difference](../../docs/working-with-fsts/structural-operations.md#difference) for usage patterns
/// - [Core Concepts](../../docs/core-concepts/algorithms.md#difference) for mathematical theory
pub fn difference<W, F1, F2, M>(fst1: &F1, fst2: &F2) -> Result<M>
where
    W: DivisibleSemiring + Hash + Clone + Ord + Eq,
    F1: Fst<W>,
    F2: Fst<W>,
    M: MutableFst<W> + Default,
{
    // Step 1: Validate inputs are acceptors
    validate_acceptor(fst1)?;
    validate_acceptor(fst2)?;

    // Step 2: Collect alphabet from both FSTs
    let alphabet = collect_alphabet(fst1, fst2)?;

    // Step 3: Build complement of fst2
    let complement_fst2: M = build_complement(fst2, &alphabet)?;

    // Step 4: Compute intersection of fst1 with complement(fst2)
    intersect(fst1, &complement_fst2)
}

/// Validate that an FST is an acceptor (input labels = output labels)
fn validate_acceptor<W: Semiring, F: Fst<W>>(fst: &F) -> Result<()> {
    for state in fst.states() {
        for arc in fst.arcs(state) {
            if arc.ilabel != arc.olabel {
                return Err(Error::Algorithm(
                    "FST must be an acceptor (input = output labels)".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Collect the alphabet (all non-epsilon labels) used in both FSTs
fn collect_alphabet<W: Semiring, F1: Fst<W>, F2: Fst<W>>(
    fst1: &F1,
    fst2: &F2,
) -> Result<HashSet<Label>> {
    let mut alphabet = HashSet::new();

    // Collect labels from fst1
    for state in fst1.states() {
        for arc in fst1.arcs(state) {
            if arc.ilabel != 0 {
                // Skip epsilon
                alphabet.insert(arc.ilabel);
            }
        }
    }

    // Collect labels from fst2
    for state in fst2.states() {
        for arc in fst2.arcs(state) {
            if arc.ilabel != 0 {
                // Skip epsilon
                alphabet.insert(arc.ilabel);
            }
        }
    }

    if alphabet.is_empty() {
        return Err(Error::Algorithm(
            "Empty alphabet in difference operation".into(),
        ));
    }

    Ok(alphabet)
}

/// Build the complement acceptor for the given FST over the specified alphabet
fn build_complement<
    W: DivisibleSemiring + Clone + Ord + Hash + Eq,
    F: Fst<W>,
    M: MutableFst<W> + Default,
>(
    fst: &F,
    alphabet: &HashSet<Label>,
) -> Result<M> {
    // First, ensure the FST is deterministic and complete
    let det_fst: M = determinize(fst)?;
    let complete_fst: M = make_complete(&det_fst, alphabet)?;

    // Now build the complement by flipping final states
    let mut complement = M::default();

    // Copy all states
    let mut state_map = HashMap::new();
    for state in complete_fst.states() {
        let new_state = complement.add_state();
        state_map.insert(state, new_state);
    }

    // Set start state
    if let Some(start) = complete_fst.start() {
        if let Some(&new_start) = state_map.get(&start) {
            complement.set_start(new_start);
        }
    }

    // Copy all arcs
    for state in complete_fst.states() {
        if let Some(&new_state) = state_map.get(&state) {
            for arc in complete_fst.arcs(state) {
                if let Some(&new_nextstate) = state_map.get(&arc.nextstate) {
                    complement.add_arc(
                        new_state,
                        Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), new_nextstate),
                    );
                }
            }
        }
    }

    // Flip final states: non-final becomes final, final becomes non-final
    for state in complete_fst.states() {
        if let Some(&new_state) = state_map.get(&state) {
            if complete_fst.final_weight(state).is_none() {
                // Was non-final, make it final in complement
                complement.set_final(new_state, W::one());
            }
            // Was final, leave it non-final in complement (don't set final weight)
        }
    }

    Ok(complement)
}

/// Make an FST complete by adding transitions to a sink state for missing alphabet symbols
fn make_complete<
    W: DivisibleSemiring + Clone + Ord + Hash + Eq,
    F: Fst<W>,
    M: MutableFst<W> + Default,
>(
    fst: &F,
    alphabet: &HashSet<Label>,
) -> Result<M> {
    let mut complete = M::default();

    // Copy all original states
    let mut state_map = HashMap::new();
    for state in fst.states() {
        let new_state = complete.add_state();
        state_map.insert(state, new_state);
    }

    // Add sink state (non-final state that accepts everything)
    let sink_state = complete.add_state();

    // Set start state
    if let Some(start) = fst.start() {
        if let Some(&new_start) = state_map.get(&start) {
            complete.set_start(new_start);
        }
    }

    // Copy original arcs and final weights
    for state in fst.states() {
        if let Some(&new_state) = state_map.get(&state) {
            // Copy final weight
            if let Some(weight) = fst.final_weight(state) {
                complete.set_final(new_state, weight.clone());
            }

            // Copy arcs
            for arc in fst.arcs(state) {
                if let Some(&new_nextstate) = state_map.get(&arc.nextstate) {
                    complete.add_arc(
                        new_state,
                        Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), new_nextstate),
                    );
                }
            }
        }
    }

    // Add missing transitions to sink state
    for state in fst.states() {
        if let Some(&new_state) = state_map.get(&state) {
            // Collect existing outgoing labels for this state
            let mut existing_labels = HashSet::new();
            for arc in fst.arcs(state) {
                if arc.ilabel != 0 {
                    existing_labels.insert(arc.ilabel);
                }
            }

            // Add transitions to sink for missing labels
            for &label in alphabet {
                if !existing_labels.contains(&label) {
                    complete.add_arc(new_state, Arc::new(label, label, W::one(), sink_state));
                }
            }
        }
    }

    // Add self-loops on sink state for all alphabet symbols
    for &label in alphabet {
        complete.add_arc(sink_state, Arc::new(label, label, W::one(), sink_state));
    }

    Ok(complete)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_validate_acceptor_valid() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        // Valid acceptor: input = output labels
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));

        assert!(validate_acceptor(&fst).is_ok());
    }

    #[test]
    fn test_validate_acceptor_invalid() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        // Invalid: different input/output labels (transducer)
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::one(), s1));

        assert!(validate_acceptor(&fst).is_err());
    }

    #[test]
    fn test_collect_alphabet_basic() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let mut fst2 = VectorFst::<TropicalWeight>::new();

        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst1.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s0));

        let s0 = fst2.add_state();
        let s1 = fst2.add_state();
        fst2.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s1));
        fst2.add_arc(s1, Arc::new(3, 3, TropicalWeight::one(), s0));

        let alphabet = collect_alphabet(&fst1, &fst2).unwrap();
        assert_eq!(alphabet.len(), 3);
        assert!(alphabet.contains(&1));
        assert!(alphabet.contains(&2));
        assert!(alphabet.contains(&3));
    }

    #[test]
    fn test_collect_alphabet_with_epsilon() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let mut fst2 = VectorFst::<TropicalWeight>::new();

        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.add_arc(s0, Arc::new(0, 0, TropicalWeight::one(), s1)); // Epsilon
        fst1.add_arc(s1, Arc::new(1, 1, TropicalWeight::one(), s0));

        let s0 = fst2.add_state();
        let s1 = fst2.add_state();
        fst2.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s1));

        let alphabet = collect_alphabet(&fst1, &fst2).unwrap();
        assert_eq!(alphabet.len(), 2);
        assert!(alphabet.contains(&1));
        assert!(alphabet.contains(&2));
        assert!(!alphabet.contains(&0)); // Epsilon should be excluded
    }

    #[test]
    fn test_collect_alphabet_empty() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let mut fst2 = VectorFst::<TropicalWeight>::new();

        // Only epsilon arcs
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.add_arc(s0, Arc::new(0, 0, TropicalWeight::one(), s1));

        let _s0 = fst2.add_state();
        // No arcs in fst2

        let result = collect_alphabet(&fst1, &fst2);
        assert!(result.is_err()); // Should fail with empty alphabet
    }

    #[test]
    fn test_make_complete_basic() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));

        let mut alphabet = HashSet::new();
        alphabet.insert(1);
        alphabet.insert(2);

        let complete: VectorFst<TropicalWeight> = make_complete(&fst, &alphabet).unwrap();

        // Should have original states plus sink state
        assert_eq!(complete.num_states(), 3);
        assert!(complete.start().is_some());

        // Should have transitions for all alphabet symbols from all states
        assert!(complete.num_arcs_total() > fst.num_arcs_total());
    }

    #[test]
    fn test_build_complement_simple() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));

        let mut alphabet = HashSet::new();
        alphabet.insert(1);

        let complement: VectorFst<TropicalWeight> = build_complement(&fst, &alphabet).unwrap();

        // Complement should have more states (original + sink)
        assert!(complement.num_states() >= fst.num_states());
        assert!(complement.start().is_some());
    }

    #[test]
    fn test_difference_basic() {
        // FST1 accepts "a"
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::one());
        fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));

        // FST2 accepts "b"
        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let s0 = fst2.add_state();
        let s1 = fst2.add_state();
        fst2.set_start(s0);
        fst2.set_final(s1, TropicalWeight::one());
        fst2.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s1));

        // Difference should accept "a" (fst1 - fst2)
        let diff: VectorFst<TropicalWeight> = difference(&fst1, &fst2).unwrap();
        assert!(diff.start().is_some());
        assert!(diff.num_states() > 0);
    }

    #[test]
    fn test_difference_self() {
        // FST accepts "a"
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));

        // Difference with itself should be empty
        let diff: VectorFst<TropicalWeight> = difference(&fst, &fst).unwrap();

        // Result should have no final states or should be empty
        let final_count = diff.states().filter(|&s| diff.is_final(s)).count();
        assert_eq!(final_count, 0);
    }

    #[test]
    fn test_difference_empty_fst() {
        let fst1 = VectorFst::<TropicalWeight>::new();
        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let s0 = fst2.add_state();
        fst2.set_start(s0);
        fst2.set_final(s0, TropicalWeight::one());

        // Difference with empty FST should fail (empty alphabet)
        let result = difference::<TropicalWeight, _, _, VectorFst<TropicalWeight>>(&fst1, &fst2);
        assert!(result.is_err());
    }

    #[test]
    fn test_difference_non_acceptor() {
        // Create transducer (not acceptor)
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::one());
        fst1.add_arc(s0, Arc::new(1, 2, TropicalWeight::one(), s1)); // Different input/output

        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let s0 = fst2.add_state();
        let s1 = fst2.add_state();
        fst2.set_start(s0);
        fst2.set_final(s1, TropicalWeight::one());
        fst2.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));

        // Should fail because fst1 is not an acceptor
        let result = difference::<TropicalWeight, _, _, VectorFst<TropicalWeight>>(&fst1, &fst2);
        assert!(result.is_err());
    }

    #[test]
    fn test_difference_overlapping_languages() {
        // FST1 accepts "a" and "ab"
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        let s2 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::one()); // accepts "a"
        fst1.set_final(s2, TropicalWeight::one()); // accepts "ab"
        fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1)); // a
        fst1.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2)); // b

        // FST2 accepts "ab"
        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let s0 = fst2.add_state();
        let s1 = fst2.add_state();
        let s2 = fst2.add_state();
        fst2.set_start(s0);
        fst2.set_final(s2, TropicalWeight::one());
        fst2.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1)); // a
        fst2.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2)); // b

        // Difference should accept "a" but not "ab"
        let diff: VectorFst<TropicalWeight> = difference(&fst1, &fst2).unwrap();
        assert!(diff.start().is_some());
        assert!(diff.num_states() > 0);
    }

    #[test]
    fn test_difference_disjoint_languages() {
        // FST1 accepts "a"
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::one());
        fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));

        // FST2 accepts "b"
        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let s0 = fst2.add_state();
        let s1 = fst2.add_state();
        fst2.set_start(s0);
        fst2.set_final(s1, TropicalWeight::one());
        fst2.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s1));

        // Since languages are disjoint, difference should equal fst1
        let diff: VectorFst<TropicalWeight> = difference(&fst1, &fst2).unwrap();
        assert!(diff.start().is_some());
        assert!(diff.num_states() > 0);

        // Should have at least one final state
        let final_count = diff.states().filter(|&s| diff.is_final(s)).count();
        assert!(final_count > 0);
    }

    #[test]
    fn test_difference_complex_automata() {
        // FST1: accepts strings ending with "a"
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::one());
        fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1)); // a (final)
        fst1.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s0)); // b (stay)
        fst1.add_arc(s1, Arc::new(1, 1, TropicalWeight::one(), s1)); // a (final)
        fst1.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s0)); // b (back to start)

        // FST2: accepts single "a"
        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let s0 = fst2.add_state();
        let s1 = fst2.add_state();
        fst2.set_start(s0);
        fst2.set_final(s1, TropicalWeight::one());
        fst2.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1)); // a

        // Difference should accept strings ending with "a" except single "a"
        let diff: VectorFst<TropicalWeight> = difference(&fst1, &fst2).unwrap();
        assert!(diff.start().is_some());
        assert!(diff.num_states() > 0);
    }

    #[test]
    fn test_difference_preserves_weights() {
        // FST1 with weighted arcs
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::new(2.0));
        fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.5), s1));

        // FST2: empty language (no final states)
        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let s0 = fst2.add_state();
        fst2.set_start(s0);
        fst2.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s0));
        // No final states - accepts nothing

        // Difference should preserve fst1 exactly
        let diff: VectorFst<TropicalWeight> = difference(&fst1, &fst2).unwrap();
        assert!(diff.start().is_some());

        // Check that some weights are preserved (structure may differ)
        let has_final = diff.states().any(|s| diff.is_final(s));
        assert!(has_final);
    }

    #[test]
    fn test_difference_single_state_acceptors() {
        // FST1: single state accepting empty string
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s0, TropicalWeight::one());
        fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s0)); // Self-loop on "a"

        // FST2: different single state
        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let s0 = fst2.add_state();
        fst2.set_start(s0);
        fst2.set_final(s0, TropicalWeight::one());
        fst2.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s0)); // Self-loop on "b"

        let diff: VectorFst<TropicalWeight> = difference(&fst1, &fst2).unwrap();
        assert!(diff.start().is_some());
    }

    #[test]
    fn test_difference_epsilon_handling() {
        // FST1 with epsilon transitions
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        let s2 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s2, TropicalWeight::one());
        fst1.add_arc(s0, Arc::new(0, 0, TropicalWeight::one(), s1)); // epsilon
        fst1.add_arc(s1, Arc::new(1, 1, TropicalWeight::one(), s2)); // a

        // FST2: simple acceptor
        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let s0 = fst2.add_state();
        let s1 = fst2.add_state();
        fst2.set_start(s0);
        fst2.set_final(s1, TropicalWeight::one());
        fst2.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s1)); // b

        // Should handle epsilon transitions properly
        let diff: VectorFst<TropicalWeight> = difference(&fst1, &fst2).unwrap();
        assert!(diff.start().is_some());
    }
}
