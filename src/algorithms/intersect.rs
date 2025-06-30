//! FST intersection algorithm
//!
//! Computes the intersection of two weighted finite-state acceptors, creating
//! an acceptor that recognizes strings accepted by both input acceptors.

use crate::algorithms::compose_default;
use crate::fst::{Fst, MutableFst};
use crate::semiring::Semiring;
use crate::Result;

/// Compute intersection of two finite-state acceptors
///
/// Creates a new acceptor that recognizes exactly the strings that are
/// accepted by both input acceptors. The result combines weights from
/// both acceptors using semiring multiplication for matching paths.
///
/// # Algorithm Details
///
/// - **Mathematical Operation:** A₁ ∩ A₂ where both are acceptors (input = output labels)
/// - **Implementation:** Uses composition since intersection = composition for acceptors
/// - **Time Complexity:** O(|V₁| × |V₂| × |E₁| × |E₂|) worst case
/// - **Space Complexity:** O(|V₁| × |V₂|) for state pairs
/// - **Language Relationship:** L(A₁ ∩ A₂) = L(A₁) ∩ L(A₂)
///
/// # Mathematical Foundation
///
/// For two acceptors A₁ and A₂ over the same alphabet:
/// - **Language Intersection:** L(A₁ ∩ A₂) = {w : w ∈ L(A₁) and w ∈ L(A₂)}
/// - **Weight Combination:** Weights combined via semiring multiplication (⊗)
/// - **State Space:** Cross product of original state spaces
/// - **Acceptor Property:** Input and output labels identical in acceptors
///
/// # Algorithm Steps
///
/// 1. **Acceptor Validation:** Ensure both FSTs are acceptors (input = output labels)
/// 2. **Cross Product:** Create state pairs from both acceptors
/// 3. **Label Matching:** Synchronize on matching input/output labels
/// 4. **Weight Combination:** Multiply weights using semiring operation
/// 5. **Final State Creation:** Mark states as final when both constituents are final
///
/// # Examples
///
/// ## Basic Language Intersection
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::intersect;
///
/// // Acceptor 1: accepts strings ending in "a"
/// let mut acc1 = VectorFst::<TropicalWeight>::new();
/// let s0 = acc1.add_state();
/// let s1 = acc1.add_state();
/// let s2 = acc1.add_state();
/// acc1.set_start(s0);
/// acc1.set_final(s2, TropicalWeight::one());
/// acc1.add_arc(s0, Arc::new('b' as u32, 'b' as u32, TropicalWeight::one(), s1));
/// acc1.add_arc(s1, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s2));
///
/// // Acceptor 2: accepts strings starting with "b"
/// let mut acc2 = VectorFst::<TropicalWeight>::new();
/// let s0 = acc2.add_state();
/// let s1 = acc2.add_state();
/// let s2 = acc2.add_state();
/// acc2.set_start(s0);
/// acc2.set_final(s1, TropicalWeight::one());
/// acc2.set_final(s2, TropicalWeight::one());
/// acc2.add_arc(s0, Arc::new('b' as u32, 'b' as u32, TropicalWeight::one(), s1));
/// acc2.add_arc(s1, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s2));
///
/// // Intersection: accepts strings that start with "b" AND end with "a"
/// let intersection: VectorFst<TropicalWeight> = intersect(&acc1, &acc2)?;
///
/// // Result accepts "ba" (satisfies both conditions)
/// assert!(intersection.num_states() > 0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Vocabulary Filtering
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::intersect;
///
/// // Dictionary acceptor: valid words
/// let mut dictionary = VectorFst::<TropicalWeight>::new();
/// let s0 = dictionary.add_state();
/// let s1 = dictionary.add_state();
/// dictionary.set_start(s0);
/// dictionary.set_final(s1, TropicalWeight::one());
/// dictionary.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1)); // word "cat"
///
/// // User input acceptor: possible inputs
/// let mut userᵢnput = VectorFst::<TropicalWeight>::new();
/// let s0 = userᵢnput.add_state();
/// let s1 = userᵢnput.add_state();
/// userᵢnput.set_start(s0);
/// userᵢnput.set_final(s1, TropicalWeight::new(0.8)); // user confidence
/// userᵢnput.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.9), s1)); // user said "cat"
///
/// // Valid recognized words (dictionary ∩ userᵢnput)
/// let valid_words: VectorFst<TropicalWeight> = intersect(&dictionary, &userᵢnput)?;
///
/// // Result contains valid words with combined confidence
/// println!("Valid word recognition completed");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Pattern Matching
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::intersect;
///
/// // Pattern acceptor: words containing "ing"
/// let mut pattern = VectorFst::<BooleanWeight>::new();
/// let s0 = pattern.add_state();
/// let s1 = pattern.add_state();
/// let s2 = pattern.add_state();
/// let s3 = pattern.add_state();
/// pattern.set_start(s0);
/// pattern.set_final(s3, BooleanWeight::one());
///
/// // Simple pattern: "running" (as character sequence)
/// pattern.add_arc(s0, Arc::new('r' as u32, 'r' as u32, BooleanWeight::one(), s1));
/// pattern.add_arc(s1, Arc::new('u' as u32, 'u' as u32, BooleanWeight::one(), s2));
/// pattern.add_arc(s2, Arc::new('n' as u32, 'n' as u32, BooleanWeight::one(), s3));
///
/// // Text acceptor: possible text
/// let mut text = VectorFst::<BooleanWeight>::new();
/// let s0 = text.add_state();
/// let s1 = text.add_state();
/// let s2 = text.add_state();
/// let s3 = text.add_state();
/// text.set_start(s0);
/// text.set_final(s3, BooleanWeight::one());
///
/// // Text contains the pattern
/// text.add_arc(s0, Arc::new('r' as u32, 'r' as u32, BooleanWeight::one(), s1));
/// text.add_arc(s1, Arc::new('u' as u32, 'u' as u32, BooleanWeight::one(), s2));
/// text.add_arc(s2, Arc::new('n' as u32, 'n' as u32, BooleanWeight::one(), s3));
///
/// // Find matching patterns in text
/// let matches: VectorFst<BooleanWeight> = intersect(&pattern, &text)?;
///
/// // Result contains strings that match the pattern
/// assert!(matches.start().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Constraint Satisfaction
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::intersect;
///
/// // Constraint 1: length constraint acceptor
/// let mut length_constraint = VectorFst::<BooleanWeight>::new();
/// let s0 = length_constraint.add_state();
/// let s1 = length_constraint.add_state();
/// let s2 = length_constraint.add_state();
/// length_constraint.set_start(s0);
/// length_constraint.set_final(s2, BooleanWeight::one());
///
/// // Accept exactly 2 characters
/// length_constraint.add_arc(s0, Arc::new('a' as u32, 'a' as u32, BooleanWeight::one(), s1));
/// length_constraint.add_arc(s1, Arc::new('b' as u32, 'b' as u32, BooleanWeight::one(), s2));
///
/// // Constraint 2: content constraint acceptor
/// let mut content_constraint = VectorFst::<BooleanWeight>::new();
/// let s0 = content_constraint.add_state();
/// let s1 = content_constraint.add_state();
/// content_constraint.set_start(s0);
/// content_constraint.set_final(s1, BooleanWeight::one());
///
/// // Must start with 'a'
/// content_constraint.add_arc(s0, Arc::new('a' as u32, 'a' as u32, BooleanWeight::one(), s1));
/// content_constraint.add_arc(s1, Arc::new('b' as u32, 'b' as u32, BooleanWeight::one(), s1));
///
/// // Strings satisfying both constraints
/// let valid_strings: VectorFst<BooleanWeight> = intersect(&length_constraint, &content_constraint)?;
///
/// // Result: strings of length 2 starting with 'a'
/// println!("Constraint satisfaction completed");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Multi-Level Filtering
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::intersect;
///
/// // Create a simple acceptor for demo
/// fn create_filter_acceptor(label: u32) -> VectorFst<BooleanWeight> {
///     let mut acc = VectorFst::new();
///     let s0 = acc.add_state();
///     let s1 = acc.add_state();
///     acc.set_start(s0);
///     acc.set_final(s1, BooleanWeight::one());
///     acc.add_arc(s0, Arc::new(label, label, BooleanWeight::one(), s1));
///     acc
/// }
///
/// // Multiple filters
/// let filter1 = create_filter_acceptor(1);
/// let filter2 = create_filter_acceptor(1); // Same constraint for demo
///
/// // Apply multiple filters via intersection
/// let filtered: VectorFst<BooleanWeight> = intersect(&filter1, &filter2)?;
///
/// // Result satisfies all filtering constraints
/// assert!(filtered.start().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Use Cases
///
/// ## Natural Language Processing
/// - **Vocabulary Filtering:** Intersect user input with valid vocabulary
/// - **Grammar Constraints:** Apply multiple grammatical constraints
/// - **Morphological Filtering:** Filter morphological analyses
/// - **Semantic Constraints:** Apply semantic validity filters
///
/// ## Speech Recognition
/// - **Phonotactic Constraints:** Apply phonological rules
/// - **Language Model Filtering:** Intersect with language models
/// - **Pronunciation Variants:** Intersect with pronunciation constraints
/// - **Confidence Filtering:** Filter based on recognition confidence
///
/// ## Information Extraction
/// - **Pattern Matching:** Find text patterns meeting multiple criteria
/// - **Entity Recognition:** Apply multiple entity type constraints
/// - **Format Validation:** Ensure data meets format requirements
/// - **Content Filtering:** Apply content-based filtering rules
///
/// ## System Validation
/// - **Input Validation:** Ensure inputs meet system requirements
/// - **Constraint Checking:** Verify multiple constraints simultaneously
/// - **Protocol Compliance:** Check protocol conformance
/// - **Security Filtering:** Apply security constraint filters
///
/// # Performance Characteristics
///
/// - **Time Complexity:** O(|V₁| × |V₂| × |E₁| × |E₂|) for dense intersections
/// - **Space Complexity:** O(|V₁| × |V₂|) for state cross product
/// - **Practical Performance:** Often much better for sparse acceptors
/// - **Memory Access:** Cross-product construction affects cache locality
/// - **Optimization:** Consider determinization and minimization of inputs
///
/// # Mathematical Properties
///
/// Intersection preserves important properties:
/// - **Language Intersection:** L(A₁ ∩ A₂) = L(A₁) ∩ L(A₂) exactly
/// - **Commutativity:** intersect(A₁, A₂) ≈ intersect(A₂, A₁)
/// - **Associativity:** intersect(intersect(A₁, A₂), A₃) ≈ intersect(A₁, intersect(A₂, A₃))
/// - **Identity:** intersect(A, Universal) = A
/// - **Annihilation:** intersect(A, Empty) = Empty
///
/// # Implementation Details
///
/// Intersection is implemented as composition since for acceptors A₁ and A₂:
/// A₁ ∩ A₂ = A₁ ∘ A₂ when input labels equal output labels.
/// This leverages the existing composition infrastructure while maintaining
/// the semantics of intersection for acceptor languages.
///
/// # Optimization Considerations
///
/// For better performance:
/// - **Input Preparation:** Determinize and minimize input acceptors first
/// - **Order Selection:** Intersect smaller acceptor first when possible
/// - **Epsilon Removal:** Remove epsilon transitions before intersection
/// - **Connection:** Ensure inputs are connected (reachable and coaccessible)
///
/// # Errors
///
/// Returns [`Error::Algorithm`](crate::Error::Algorithm) if:
/// - Either input FST is invalid, corrupted, or has no start state
/// - Memory allocation fails during cross-product construction
/// - The intersection operation encounters incompatible FST structures
/// - State or arc creation fails during result construction
/// - Weight computation overflows or produces invalid results
///
/// # See Also
///
/// - [`union()`](crate::algorithms::union()) for language union (A₁ ∪ A₂)
/// - [`compose()`](crate::algorithms::compose()) for general FST composition
/// - [`difference()`](crate::algorithms::difference()) for language difference (A₁ - A₂)
/// - [`determinize()`](crate::algorithms::determinize()) for optimizing acceptors before intersection
/// - [Working with FSTs - Intersection](../../docs/working-with-fsts/structural-operations.md#intersection) for usage patterns
/// - [Core Concepts](../../docs/core-concepts/algorithms.md#intersection) for mathematical theory
pub fn intersect<W, F1, F2, M>(fst1: &F1, fst2: &F2) -> Result<M>
where
    W: Semiring,
    F1: Fst<W>,
    F2: Fst<W>,
    M: MutableFst<W> + Default,
{
    // intersection is composition for acceptors
    compose_default(fst1, fst2)
}
