//! FST projection algorithms
//!
//! Extracts input or output labels from weighted finite-state transducers,
//! converting transducers into acceptors that recognize single label sequences.
//!
//! ## References
//!
//! - Mohri, M., Pereira, F., and Riley, M. (2008). "Speech Recognition with Weighted
//!   Finite-State Transducers." Springer Handbook of Speech Processing, pp. 559-584.

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst};
use crate::semiring::Semiring;
use crate::Result;

/// Project FST onto its input labels, creating an acceptor that recognizes input sequences
///
/// Converts a finite-state transducer (FST) into a finite-state acceptor (FSA)
/// by extracting the input labels and setting both input and output labels to
/// the same values. The result accepts exactly the input language of the original FST.
///
/// # Algorithm Details
///
/// - **Label Extraction:** Extract input labels from all arcs
/// - **Label Duplication:** Set both input and output labels to the same value
/// - **Time Complexity:** O(|V| + |E|) for linear traversal and copying
/// - **Space Complexity:** O(|V| + |E|) for the result FST
/// - **Language Relationship:** L(project_input(T)) = Domain(T)
///
/// # Mathematical Foundation
///
/// For an FST T that maps strings x to strings y with weights w,
/// the input projection extracts the domain:
/// - **Domain Extraction:** Domain(T) = {x : ∃y,w such that T(x,y) = w}
/// - **Acceptor Creation:** Result recognizes input strings regardless of output
/// - **Weight Preservation:** All path weights maintained exactly
///
/// # Algorithm Steps
///
/// 1. **Structure Copy:** Copy all states and state connectivity from original FST
/// 2. **Start/Final Copy:** Preserve start state and all final weights
/// 3. **Arc Projection:** For each arc (s, i:o/w, t), create arc (s, i:i/w, t)
/// 4. **Label Unification:** Both input and output labels become the same
/// 5. **Weight Preservation:** All arc and final weights remain unchanged
///
/// # Examples
///
/// ## Basic Input Projection
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // FST that maps "hello" -> "hi" and "world" -> "earth"
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s2, TropicalWeight::one());
///
/// // "hello" -> "hi"
/// fst.add_arc(s0, Arc::new('h' as u32, 'h' as u32, TropicalWeight::one(), s1));
/// fst.add_arc(s1, Arc::new('i' as u32, 'e' as u32, TropicalWeight::one(), s2));
///
/// // Project to input: result accepts "hi" (input sequence)
/// let input_acceptor: VectorFst<TropicalWeight> = project_input(&fst)?;
///
/// // Result is an acceptor for the input language
/// assert_eq!(input_acceptor.num_states(), fst.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Translation System Projection
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Translation FST: English -> French
/// let mut translator = VectorFst::<TropicalWeight>::new();
/// let s0 = translator.add_state();
/// let s1 = translator.add_state();
/// let s2 = translator.add_state();
///
/// translator.set_start(s0);
/// translator.set_final(s2, TropicalWeight::one());
///
/// // "cat" -> "chat" (1:3, 2:4 represent word IDs)
/// translator.add_arc(s0, Arc::new(1, 3, TropicalWeight::new(0.8), s1));
/// translator.add_arc(s1, Arc::new(2, 4, TropicalWeight::new(0.9), s2));
///
/// // Extract English vocabulary (input projection)
/// let english_vocab: VectorFst<TropicalWeight> = project_input(&translator)?;
///
/// // Result accepts English word sequences regardless of translation
/// println!("English vocabulary extractor created");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Morphological Analysis Projection
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Morphological analyzer: surface -> analysis
/// let mut morph = VectorFst::<TropicalWeight>::new();
/// let s0 = morph.add_state();
/// let s1 = morph.add_state();
/// let s2 = morph.add_state();
///
/// morph.set_start(s0);
/// morph.set_final(s2, TropicalWeight::one());
///
/// // "running" -> "run+VERB+PRESENT"
/// morph.add_arc(s0, Arc::new('r' as u32, 'r' as u32, TropicalWeight::one(), s1));
/// morph.add_arc(s1, Arc::new('u' as u32, '+' as u32, TropicalWeight::one(), s2));
///
/// // Extract surface forms (input projection)
/// let surface_forms: VectorFst<TropicalWeight> = project_input(&morph)?;
///
/// // Result recognizes surface word forms only
/// assert!(surface_forms.start().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Multi-Level Processing
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Speech recognition: phoneme -> word
/// let mut speech = VectorFst::<TropicalWeight>::new();
/// let s0 = speech.add_state();
/// let s1 = speech.add_state();
///
/// speech.set_start(s0);
/// speech.set_final(s1, TropicalWeight::one());
///
/// // Phoneme sequence -> word
/// speech.add_arc(s0, Arc::new(1, 100, TropicalWeight::new(0.7), s1)); // /k/ -> "cat"
///
/// // Extract phoneme acceptor (input projection)
/// let phoneme_acceptor: VectorFst<TropicalWeight> = project_input(&speech)?;
///
/// // Result accepts phoneme sequences independent of word recognition
/// println!("Phoneme acceptor extracted");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Weighted Path Extraction
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Weighted transducer with costs
/// let mut weighted_fst = VectorFst::<TropicalWeight>::new();
/// let s0 = weighted_fst.add_state();
/// let s1 = weighted_fst.add_state();
///
/// weighted_fst.set_start(s0);
/// weighted_fst.set_final(s1, TropicalWeight::new(0.5));
///
/// weighted_fst.add_arc(s0, Arc::new('a' as u32, 'x' as u32, TropicalWeight::new(1.2), s1));
///
/// // Project with weight preservation
/// let weighted_acceptor: VectorFst<TropicalWeight> = project_input(&weighted_fst)?;
///
/// // Input acceptor maintains all path costs
/// assert_eq!(weighted_acceptor.num_states(), weighted_fst.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Use Cases
///
/// ## Language Analysis
/// - **Source Language Extraction:** Extract source vocabulary from translation models
/// - **Input Validation:** Create acceptors to validate system inputs
/// - **Morphological Analysis:** Extract surface forms from analysis transducers
/// - **Speech Recognition:** Extract phoneme sequences from recognition models
///
/// ## System Design
/// - **Input Space Analysis:** Understand what inputs a system can handle
/// - **Vocabulary Coverage:** Determine input vocabulary requirements
/// - **Interface Design:** Design input interfaces based on system capabilities
/// - **Constraint Definition:** Define input constraints from system models
///
/// ## Preprocessing
/// - **Filter Creation:** Create input filters for preprocessing pipelines
/// - **Validation Sets:** Build input validation from known good inputs
/// - **Test Generation:** Generate test inputs from system specifications
/// - **Data Preparation:** Prepare input data for further processing
///
/// # Performance Characteristics
///
/// - **Time Complexity:** O(|V| + |E|) for linear state and arc processing
/// - **Space Complexity:** O(|V| + |E|) matching the original FST size
/// - **Memory Efficiency:** Simple copying with minimal overhead
/// - **Cache Friendly:** Sequential access pattern improves performance
/// - **Parallelizable:** State processing can be parallelized easily
///
/// # Mathematical Properties
///
/// Input projection preserves essential FST properties:
/// - **Language Preservation:** Domain exactly preserved in acceptor form
/// - **Weight Preservation:** All path weights maintained identically
/// - **Structural Properties:** State connectivity and reachability preserved
/// - **Determinism:** Deterministic FSTs produce deterministic acceptors
/// - **Compositionality:** project_input(T₁ ∘ T₂) relates to domain analysis
///
/// # Implementation Details
///
/// The algorithm performs a simple structural copy with label transformation.
/// For each arc (s, i:o/w, t) in the original FST, it creates (s, i:i/w, t)
/// in the result. This preserves all structural and weight information while
/// creating an acceptor that recognizes the input language.
///
/// # Optimization Opportunities
///
/// After input projection, consider these optimizations:
/// - **Determinization:** Convert to deterministic acceptor if needed
/// - **Minimization:** Reduce state count through equivalence merging
/// - **Connection:** Remove unreachable states from the result
/// - **Epsilon Removal:** Eliminate epsilon transitions for efficiency
///
/// # Errors
///
/// Returns [`Error::Algorithm`](crate::Error::Algorithm) if:
/// - The input FST is invalid, corrupted, or malformed
/// - Memory allocation fails during computation or result construction
/// - The projection operation encounters invalid state or arc data
/// - State or arc enumeration fails due to data corruption
/// - Weight operations fail during arc creation
///
/// # See Also
///
/// - [`project_output`] for extracting output language from FSTs
/// - [`compose()`](crate::algorithms::compose()) for combining projections with other FSTs
/// - [`union()`](crate::algorithms::union()) for combining multiple projections
/// - [Working with FSTs - Projection](../../docs/working-with-fsts/structural-operations.md#projection) for usage patterns
/// - [Core Concepts](../../docs/core-concepts/algorithms.md#projection) for mathematical theory
pub fn project_input<W, F, M>(fst: &F) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    project_impl(fst, true)
}

/// Project FST onto its output labels, creating an acceptor that recognizes output sequences
///
/// Converts a finite-state transducer (FST) into a finite-state acceptor (FSA)
/// by extracting the output labels and setting both input and output labels to
/// the same values. The result accepts exactly the output language of the original FST.
///
/// # Algorithm Details
///
/// - **Label Extraction:** Extract output labels from all arcs
/// - **Label Duplication:** Set both input and output labels to the same value
/// - **Time Complexity:** O(|V| + |E|) for linear traversal and copying
/// - **Space Complexity:** O(|V| + |E|) for the result FST
/// - **Language Relationship:** L(project_output(T)) = Range(T)
///
/// # Mathematical Foundation
///
/// For an FST T that maps strings x to strings y with weights w,
/// the output projection extracts the range:
/// - **Range Extraction:** Range(T) = {y : ∃x,w such that T(x,y) = w}
/// - **Acceptor Creation:** Result recognizes output strings regardless of input
/// - **Weight Preservation:** All path weights maintained exactly
///
/// # Examples
///
/// ## Basic Output Projection
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // FST that maps "hello" -> "hi" and "world" -> "earth"
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s2, TropicalWeight::one());
///
/// // "hello" -> "hi": input 'h' -> output 'h', input 'i' -> output 'e'
/// fst.add_arc(s0, Arc::new('h' as u32, 'h' as u32, TropicalWeight::one(), s1));
/// fst.add_arc(s1, Arc::new('i' as u32, 'e' as u32, TropicalWeight::one(), s2));
///
/// // Project to output: result accepts "he" (output sequence)
/// let output_acceptor: VectorFst<TropicalWeight> = project_output(&fst)?;
///
/// // Result is an acceptor for the output language
/// assert_eq!(output_acceptor.num_states(), fst.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Translation Target Extraction
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Translation FST: English -> French
/// let mut translator = VectorFst::<TropicalWeight>::new();
/// let s0 = translator.add_state();
/// let s1 = translator.add_state();
/// let s2 = translator.add_state();
///
/// translator.set_start(s0);
/// translator.set_final(s2, TropicalWeight::one());
///
/// // "cat" -> "chat" (1:3, 2:4 represent word IDs)
/// translator.add_arc(s0, Arc::new(1, 3, TropicalWeight::new(0.8), s1));
/// translator.add_arc(s1, Arc::new(2, 4, TropicalWeight::new(0.9), s2));
///
/// // Extract French vocabulary (output projection)
/// let french_vocab: VectorFst<TropicalWeight> = project_output(&translator)?;
///
/// // Result accepts French word sequences regardless of English input
/// println!("French vocabulary extractor created");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Generated Text Recognition
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Text generation FST: concept -> surface text
/// let mut generator = VectorFst::<TropicalWeight>::new();
/// let s0 = generator.add_state();
/// let s1 = generator.add_state();
///
/// generator.set_start(s0);
/// generator.set_final(s1, TropicalWeight::one());
///
/// // Concept -> Text: "GREETING" -> "hello"
/// generator.add_arc(s0, Arc::new(1, 'h' as u32, TropicalWeight::one(), s1));
///
/// // Extract generated text acceptor (output projection)
/// let text_acceptor: VectorFst<TropicalWeight> = project_output(&generator)?;
///
/// // Result recognizes generated text sequences
/// assert!(text_acceptor.start().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Use Cases
///
/// ## Language Analysis
/// - **Target Language Extraction:** Extract target vocabulary from translation models
/// - **Generated Text Analysis:** Analyze possible outputs from generation systems
/// - **Morphological Generation:** Extract generated surface forms
/// - **Speech Synthesis:** Extract possible acoustic outputs
///
/// ## System Validation
/// - **Output Space Analysis:** Understand what outputs a system can produce
/// - **Vocabulary Coverage:** Determine output vocabulary coverage
/// - **Quality Assessment:** Analyze generated content possibilities
/// - **Constraint Verification:** Ensure outputs meet requirements
///
/// ## Preprocessing
/// - **Cascade Preparation:** Prepare acceptors for further composition
/// - **Filter Creation:** Create filters based on desired outputs
/// - **Validation Sets:** Build validation acceptors from known good outputs
/// - **Testing Infrastructure:** Create test cases from system capabilities
///
/// # Errors
///
/// Returns [`Error::Algorithm`](crate::Error::Algorithm) if:
/// - The input FST is invalid, corrupted, or malformed
/// - Memory allocation fails during computation or result construction
/// - The projection operation encounters invalid state or arc data
/// - State or arc enumeration fails due to data corruption
/// - Weight operations fail during arc creation
///
/// # See Also
///
/// - [`project_input`] for extracting input language from FSTs
/// - [`compose()`](crate::algorithms::compose()) for combining projections with other FSTs
/// - [`union()`](crate::algorithms::union()) for combining multiple projections
/// - [Working with FSTs - Projection](../../docs/working-with-fsts/structural-operations.md#projection) for usage patterns
/// - [Core Concepts](../../docs/core-concepts/algorithms.md#projection) for mathematical theory
pub fn project_output<W, F, M>(fst: &F) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    project_impl(fst, false)
}

fn project_impl<W, F, M>(fst: &F, project_input: bool) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
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

    // copy with projection
    for state in fst.states() {
        // final weights
        if let Some(weight) = fst.final_weight(state) {
            result.set_final(state, weight.clone());
        }

        // project arcs
        for arc in fst.arcs(state) {
            let label = if project_input {
                arc.ilabel
            } else {
                arc.olabel
            };
            result.add_arc(
                state,
                Arc::new(label, label, arc.weight.clone(), arc.nextstate),
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
    fn test_project_input_simple() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Create FST with different input/output labels
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));
        fst.add_arc(s1, Arc::new(3, 4, TropicalWeight::new(1.0), s2));

        let projected: VectorFst<TropicalWeight> = project_input(&fst).unwrap();

        // Check structure is preserved
        assert_eq!(projected.num_states(), fst.num_states());
        assert_eq!(projected.start(), fst.start());
        assert!(projected.is_final(s2));

        // Check arc projection (input labels should be copied to both input and output)
        let arcs_s0: Vec<_> = projected.arcs(s0).collect();
        assert_eq!(arcs_s0.len(), 1);
        assert_eq!(arcs_s0[0].ilabel, 1); // Original input label
        assert_eq!(arcs_s0[0].olabel, 1); // Should now match input label
        assert_eq!(arcs_s0[0].weight, TropicalWeight::new(0.5));

        let arcs_s1: Vec<_> = projected.arcs(s1).collect();
        assert_eq!(arcs_s1.len(), 1);
        assert_eq!(arcs_s1[0].ilabel, 3); // Original input label
        assert_eq!(arcs_s1[0].olabel, 3); // Should now match input label
    }

    #[test]
    fn test_project_output_simple() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::new(2.0));

        // Create FST with different input/output labels
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));
        fst.add_arc(s1, Arc::new(3, 4, TropicalWeight::new(1.0), s2));

        let projected: VectorFst<TropicalWeight> = project_output(&fst).unwrap();

        // Check structure is preserved
        assert_eq!(projected.num_states(), fst.num_states());
        assert_eq!(projected.start(), fst.start());
        assert!(projected.is_final(s2));
        assert_eq!(projected.final_weight(s2), Some(&TropicalWeight::new(2.0)));

        // Check arc projection (output labels should be copied to both input and output)
        let arcs_s0: Vec<_> = projected.arcs(s0).collect();
        assert_eq!(arcs_s0.len(), 1);
        assert_eq!(arcs_s0[0].ilabel, 2); // Should now match original output label
        assert_eq!(arcs_s0[0].olabel, 2); // Original output label
        assert_eq!(arcs_s0[0].weight, TropicalWeight::new(0.5));

        let arcs_s1: Vec<_> = projected.arcs(s1).collect();
        assert_eq!(arcs_s1.len(), 1);
        assert_eq!(arcs_s1[0].ilabel, 4); // Should now match original output label
        assert_eq!(arcs_s1[0].olabel, 4); // Original output label
    }

    #[test]
    fn test_project_epsilon_labels() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        // Add arc with epsilon input and regular output
        fst.add_arc(s0, Arc::new(0, 5, TropicalWeight::one(), s1)); // 0 = epsilon

        let input_proj: VectorFst<TropicalWeight> = project_input(&fst).unwrap();
        let output_proj: VectorFst<TropicalWeight> = project_output(&fst).unwrap();

        // Input projection should have epsilon:epsilon
        let input_arcs: Vec<_> = input_proj.arcs(s0).collect();
        assert_eq!(input_arcs[0].ilabel, 0);
        assert_eq!(input_arcs[0].olabel, 0);

        // Output projection should have 5:5
        let output_arcs: Vec<_> = output_proj.arcs(s0).collect();
        assert_eq!(output_arcs[0].ilabel, 5);
        assert_eq!(output_arcs[0].olabel, 5);
    }

    #[test]
    fn test_project_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();

        let input_proj: VectorFst<TropicalWeight> = project_input(&fst).unwrap();
        let output_proj: VectorFst<TropicalWeight> = project_output(&fst).unwrap();

        assert_eq!(input_proj.num_states(), 0);
        assert_eq!(output_proj.num_states(), 0);
        assert!(input_proj.is_empty());
        assert!(output_proj.is_empty());
    }

    #[test]
    fn test_project_single_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::new(3.0));

        let input_proj: VectorFst<TropicalWeight> = project_input(&fst).unwrap();
        let output_proj: VectorFst<TropicalWeight> = project_output(&fst).unwrap();

        assert_eq!(input_proj.num_states(), 1);
        assert_eq!(output_proj.num_states(), 1);
        assert_eq!(input_proj.start(), Some(s0));
        assert_eq!(output_proj.start(), Some(s0));
        assert_eq!(input_proj.final_weight(s0), Some(&TropicalWeight::new(3.0)));
        assert_eq!(
            output_proj.final_weight(s0),
            Some(&TropicalWeight::new(3.0))
        );
    }

    #[test]
    fn test_project_multiple_arcs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        // Add multiple arcs with different input/output labels
        fst.add_arc(s0, Arc::new(1, 10, TropicalWeight::new(0.1), s1));
        fst.add_arc(s0, Arc::new(2, 20, TropicalWeight::new(0.2), s1));
        fst.add_arc(s0, Arc::new(3, 30, TropicalWeight::new(0.3), s1));

        let input_proj: VectorFst<TropicalWeight> = project_input(&fst).unwrap();
        let output_proj: VectorFst<TropicalWeight> = project_output(&fst).unwrap();

        // Check input projection
        let input_arcs: Vec<_> = input_proj.arcs(s0).collect();
        assert_eq!(input_arcs.len(), 3);
        assert_eq!(input_arcs[0].ilabel, 1);
        assert_eq!(input_arcs[0].olabel, 1);
        assert_eq!(input_arcs[1].ilabel, 2);
        assert_eq!(input_arcs[1].olabel, 2);
        assert_eq!(input_arcs[2].ilabel, 3);
        assert_eq!(input_arcs[2].olabel, 3);

        // Check output projection
        let output_arcs: Vec<_> = output_proj.arcs(s0).collect();
        assert_eq!(output_arcs.len(), 3);
        assert_eq!(output_arcs[0].ilabel, 10);
        assert_eq!(output_arcs[0].olabel, 10);
        assert_eq!(output_arcs[1].ilabel, 20);
        assert_eq!(output_arcs[1].olabel, 20);
        assert_eq!(output_arcs[2].ilabel, 30);
        assert_eq!(output_arcs[2].olabel, 30);
    }
}
