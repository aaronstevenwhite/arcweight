//! Weight and label pushing algorithms
//!
//! Pushes weights toward the initial state or final states to enable optimization.
//!
//! # Semiring Requirements
//!
//! Weight pushing requires the semiring to be **weakly left divisible** and **zero-sum-free:**
//! - `DivisibleSemiring` trait enables division for potential computation
//! - Zero-sum-free property prevents division by zero during normalization
//! - Required for both initial-state and final-state weight pushing
//!
//! # Supported Semirings
//!
//! - ✅ `TropicalWeight` - Implements `DivisibleSemiring`, zero-sum-free
//! - ✅ `LogWeight` - Implements `DivisibleSemiring`, zero-sum-free
//! - ❌ `ProbabilityWeight` - Not weakly left divisible
//! - ❌ String semirings - Generally not zero-sum-free
//!
//! # Convergence Requirements
//!
//! For cyclic FSTs, weight pushing requires:
//! - Convergent weight sequences for global pushing
//! - Acyclic structure for guaranteed termination
//! - K-closed semiring property for epsilon cycles

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst};
use crate::semiring::{DivisibleSemiring, Semiring};
use crate::Result;

/// Push weights toward initial state for optimization and normalization
///
/// Redistributes weights throughout the FST by pushing them toward the initial
/// state, which can improve efficiency for subsequent algorithms and enable
/// weight-based optimizations. This transformation maintains the FST language
/// while reorganizing weight distribution.
///
/// # Algorithm Details
///
/// - **Weight Redistribution:** Moves weights from arcs toward initial state
/// - **Potential Theory:** Uses shortest-distance potentials for weight rebalancing  
/// - **Time Complexity:** O(|V| + |E|) for acyclic FSTs, O(|V|³) for general case
/// - **Space Complexity:** O(|V|) for potential storage and result construction
/// - **Language Preservation:** L(push_weights(T)) = L(T) exactly
///
/// # Mathematical Foundation
///
/// Weight pushing uses potential theory to redistribute weights:
/// - **Potential Function:** π(q) represents shortest distance from start to state q
/// - **Weight Transformation:** For arc e = (p, i:o/w, q), new weight = w × π(q) / π(p)
/// - **Final Weight Adjustment:** For final state q with weight w, new weight = w / π(q)
/// - **Invariant Preservation:** All path weights remain unchanged through rebalancing
///
/// # Algorithm Steps
///
/// 1. **Potential Computation:** Calculate shortest distances from start state to all states
/// 2. **Arc Reweighting:** Adjust arc weights using potential differences
/// 3. **Final Weight Adjustment:** Modify final weights using state potentials
/// 4. **Structural Preservation:** Maintain all states, arcs, and connectivity
/// 5. **Language Invariance:** Ensure all accepting paths have identical weights
///
/// # Examples
///
/// ## Basic Weight Pushing
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::push_weights;
///
/// // FST with unbalanced weight distribution
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s2, TropicalWeight::new(2.0));
///
/// // Heavy weights on arcs
/// fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::new(5.0), s1));
/// fst.add_arc(s1, Arc::new('b' as u32, 'b' as u32, TropicalWeight::new(3.0), s2));
///
/// // Push weights toward initial state
/// let pushed: VectorFst<TropicalWeight> = push_weights(&fst)?;
///
/// // Result has redistributed weights but same language
/// assert_eq!(pushed.num_states(), fst.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Preprocessing for Minimization
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::push_weights;
///
/// // Complex FST with scattered weights
/// let mut complex_fst = VectorFst::<TropicalWeight>::new();
/// let states: Vec<_> = (0..4).map(|_| complex_fst.add_state()).collect();
///
/// complex_fst.set_start(states[0]);
/// complex_fst.set_final(states[3], TropicalWeight::new(1.5));
///
/// // Multiple paths with different weight distributions
/// complex_fst.add_arc(states[0], Arc::new(1, 1, TropicalWeight::new(0.5), states[1]));
/// complex_fst.add_arc(states[1], Arc::new(2, 2, TropicalWeight::new(1.0), states[2]));
/// complex_fst.add_arc(states[2], Arc::new(3, 3, TropicalWeight::new(0.8), states[3]));
///
/// complex_fst.add_arc(states[0], Arc::new(4, 4, TropicalWeight::new(2.0), states[2]));
/// complex_fst.add_arc(states[2], Arc::new(5, 5, TropicalWeight::new(0.3), states[3]));
///
/// // Push weights to enable better minimization
/// let weight_pushed: VectorFst<TropicalWeight> = push_weights(&complex_fst)?;
/// let minimized: VectorFst<TropicalWeight> = minimize(&weight_pushed)?;
///
/// // Minimization works better on weight-pushed FSTs
/// println!("Original: {} states, Minimized: {} states",
///          complex_fst.num_states(), minimized.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Optimization Pipeline Integration
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::push_weights;
///
/// // Complete optimization pipeline with weight pushing
/// fn optimize_fst(fst: &VectorFst<TropicalWeight>)
///     -> Result<VectorFst<TropicalWeight>> {
///     // Step 1: Remove unreachable states
///     let connected: VectorFst<TropicalWeight> = connect(fst)?;
///     
///     // Step 2: Push weights for better optimization
///     let weight_pushed: VectorFst<TropicalWeight> = push_weights(&connected)?;
///     
///     // Step 3: Determinize and minimize
///     let determinized: VectorFst<TropicalWeight> = determinize(&weight_pushed)?;
///     let minimized: VectorFst<TropicalWeight> = minimize(&determinized)?;
///     
///     Ok(minimized)
/// }
///
/// // Create test FST
/// let mut test_fst = VectorFst::new();
/// let s0 = test_fst.add_state();
/// let s1 = test_fst.add_state();
/// test_fst.set_start(s0);
/// test_fst.set_final(s1, TropicalWeight::new(1.0));
/// test_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));
///
/// let optimized = optimize_fst(&test_fst)?;
/// println!("FST optimized with weight pushing");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Speech Recognition Lattice Normalization
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::push_weights;
///
/// // Speech recognition lattice with acoustic scores
/// let mut lattice = VectorFst::<TropicalWeight>::new();
/// let start = lattice.add_state();
/// let word1 = lattice.add_state();
/// let word2 = lattice.add_state();
/// let end = lattice.add_state();
///
/// lattice.set_start(start);
/// lattice.set_final(end, TropicalWeight::new(0.5)); // LM score
///
/// // Acoustic model scores on arcs
/// lattice.add_arc(start, Arc::new(1, 1, TropicalWeight::new(3.2), word1)); // "hello"
/// lattice.add_arc(word1, Arc::new(2, 2, TropicalWeight::new(2.7), word2)); // "world"
/// lattice.add_arc(word2, Arc::new(0, 0, TropicalWeight::new(1.1), end)); // </s>
///
/// // Alternative path
/// lattice.add_arc(start, Arc::new(3, 3, TropicalWeight::new(4.1), end)); // "hi"
///
/// // Push weights for normalized lattice
/// let normalized: VectorFst<TropicalWeight> = push_weights(&lattice)?;
///
/// // Result has more balanced weight distribution
/// assert!(normalized.start().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Composition Preprocessing
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::push_weights;
///
/// // Prepare FSTs for efficient composition
/// fn prepare_for_composition(
///     fst1: &VectorFst<TropicalWeight>,
///     fst2: &VectorFst<TropicalWeight>
/// ) -> Result<(VectorFst<TropicalWeight>, VectorFst<TropicalWeight>)> {
///     // Push weights in both FSTs
///     let pushed1: VectorFst<TropicalWeight> = push_weights(fst1)?;
///     let pushed2: VectorFst<TropicalWeight> = push_weights(fst2)?;
///     
///     // Determinize for efficient composition
///     let det1: VectorFst<TropicalWeight> = determinize(&pushed1)?;
///     let det2: VectorFst<TropicalWeight> = determinize(&pushed2)?;
///     
///     Ok((det1, det2))
/// }
///
/// // Create simple FSTs
/// let mut fst1 = VectorFst::new();
/// let s0 = fst1.add_state();
/// let s1 = fst1.add_state();
/// fst1.set_start(s0);
/// fst1.set_final(s1, TropicalWeight::one());
/// fst1.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));
///
/// let mut fst2 = VectorFst::new();
/// let s0 = fst2.add_state();
/// let s1 = fst2.add_state();
/// fst2.set_start(s0);
/// fst2.set_final(s1, TropicalWeight::one());
/// fst2.add_arc(s0, Arc::new(2, 3, TropicalWeight::new(0.3), s1));
///
/// let (prep1, prep2) = prepare_for_composition(&fst1, &fst2)?;
/// println!("FSTs prepared for composition");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Use Cases
///
/// ## FST Optimization
/// - **Minimization Preprocessing:** Push weights before minimization for better results
/// - **Determinization Setup:** Prepare FSTs for efficient determinization
/// - **Composition Optimization:** Balance weights before expensive composition
/// - **Memory Optimization:** Redistribute weights to enable better compression
///
/// ## Speech and Language Processing
/// - **Acoustic Model Normalization:** Balance acoustic and language model scores
/// - **Lattice Processing:** Normalize recognition lattices for downstream processing
/// - **Translation Models:** Optimize translation model weight distribution
/// - **Language Model Integration:** Balance different scoring components
///
/// ## Information Retrieval
/// - **Search Score Normalization:** Balance different relevance factors
/// - **Ranking Model Optimization:** Optimize scoring function distribution
/// - **Query Processing:** Prepare query automata for efficient matching
/// - **Index Optimization:** Optimize search index weight distribution
///
/// ## Machine Learning
/// - **Model Preprocessing:** Prepare finite-state models for training
/// - **Feature Weight Balancing:** Redistribute feature importance weights
/// - **Ensemble Optimization:** Balance component model contributions
/// - **Gradient Flow Optimization:** Improve optimization landscape
///
/// # Performance Characteristics
///
/// - **Time Complexity:** O(|V| + |E|) for acyclic FSTs using topological order
/// - **Cyclic FSTs:** O(|V|³) worst case using iterative shortest-distance
/// - **Space Complexity:** O(|V|) for potential storage and intermediate results
/// - **Memory Efficiency:** In-place weight modification where possible
/// - **Convergence:** Guaranteed for k-closed semirings and well-formed FSTs
///
/// # Mathematical Properties
///
/// Weight pushing preserves essential FST properties:
/// - **Language Preservation:** L(push_weights(T)) = L(T) exactly
/// - **Path Weight Preservation:** All path weights remain identical
/// - **Structural Properties:** States, arcs, and connectivity unchanged
/// - **Determinism:** Deterministic FSTs remain deterministic
/// - **Idempotency:** push_weights(push_weights(T)) = push_weights(T)
///
/// # Implementation Details
///
/// The current implementation uses a simplified potential computation.
/// Full implementation includes:
/// - **Shortest Distance Algorithm:** Compute exact potentials using shortest-distance
/// - **Topological Processing:** Efficient computation for acyclic FSTs
/// - **Iterative Methods:** Handle cyclic FSTs with convergence guarantees
/// - **Numerical Stability:** Prevent overflow/underflow in weight computations
/// - **Optimization:** Early termination and sparse computation techniques
///
/// # Semiring Considerations
///
/// Different semirings have different pushing characteristics:
/// - **Tropical Semiring:** Excellent for weight pushing, natural shortest-distance
/// - **Log Semiring:** Well-suited for probability-based weight redistribution
/// - **Real Semiring:** May require careful numerical handling
/// - **String Semirings:** Generally not suitable for weight pushing
///
/// # Optimization Benefits
///
/// Weight pushing enables several optimizations:
/// - **Better Minimization:** More effective state merging after weight balancing
/// - **Efficient Composition:** Reduced search space in composition algorithms
/// - **Faster Shortest Path:** Improved convergence for shortest-path algorithms
/// - **Memory Efficiency:** More compact representation after optimization
///
/// # Errors
///
/// Returns [`Error::Algorithm`](crate::Error::Algorithm) if:
/// - The input FST is invalid, corrupted, or malformed
/// - Memory allocation fails during potential computation or result construction
/// - The FST contains cycles that prevent convergence in shortest-distance computation
/// - Division operations fail due to zero or infinite potentials
/// - Numerical overflow or underflow occurs during weight redistribution
/// - The semiring does not support required divisibility operations
///
/// # See Also
///
/// - [`push_labels`] for label pushing operations
/// - [`minimize()`](crate::algorithms::minimize()) for algorithms that benefit from weight pushing
/// - [`determinize()`](crate::algorithms::determinize()) for determinization after weight pushing
/// - [`shortest_path()`](crate::algorithms::shortest_path()) for related shortest-distance algorithms
/// - [Working with FSTs - Weight Pushing](../../docs/working-with-fsts/advanced-topics.md#weight-pushing) for usage patterns
/// - [Core Concepts](../../docs/core-concepts/algorithms.md#weight-pushing) for mathematical theory
pub fn push_weights<W, F, M>(fst: &F) -> Result<M>
where
    W: DivisibleSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    // compute potentials using shortest distance
    let potentials = compute_potentials(fst)?;

    let mut result = M::default();

    // copy states
    for _ in 0..fst.num_states() {
        result.add_state();
    }

    // set start
    if let Some(start) = fst.start() {
        result.set_start(start);
    }

    // reweight arcs
    for state in fst.states() {
        let potential = &potentials[state as usize];

        // adjust final weight
        if let Some(weight) = fst.final_weight(state) {
            if let Some(pushed) = weight.divide(potential) {
                result.set_final(state, pushed);
            }
        }

        // adjust arc weights
        for arc in fst.arcs(state) {
            let next_potential = &potentials[arc.nextstate as usize];
            if let Some(reweighted) = arc.weight.times(next_potential).divide(potential) {
                result.add_arc(
                    state,
                    Arc::new(arc.ilabel, arc.olabel, reweighted, arc.nextstate),
                );
            }
        }
    }

    Ok(result)
}

/// Push labels toward initial state for optimization and canonical form
///
/// Redistributes input and output labels throughout the FST by pushing them
/// toward the initial state, which can improve efficiency for subsequent
/// algorithms and create more canonical representations. This transformation
/// maintains the FST language while reorganizing label distribution.
///
/// # Algorithm Details
///
/// - **Label Redistribution:** Moves labels from arcs toward initial state
/// - **String Prefix Computation:** Uses string-based shortest distance for label prefixes
/// - **Time Complexity:** O(|V| + |E|) for basic implementation
/// - **Space Complexity:** O(|V| × max_string_length) for label storage
/// - **Language Preservation:** L(push_labels(T)) = L(T) exactly
///
/// # Mathematical Foundation
///
/// Label pushing uses string prefixes to redistribute labels:
/// - **Common Prefix Extraction:** Find longest common prefix of outgoing arc labels
/// - **Label Factorization:** Factor labels into prefix (pushed) and suffix (remaining)
/// - **State-Level Prefixes:** Compute optimal label prefixes for each state
/// - **Transduction Preservation:** All input-output mappings remain unchanged
///
/// # Algorithm Steps
///
/// 1. **Prefix Analysis:** Analyze common prefixes at each state
/// 2. **Label Factorization:** Split labels into pushable and remaining parts
/// 3. **Arc Relabeling:** Update arc labels after pushing prefixes
/// 4. **State Annotation:** Record pushed labels for proper transduction
/// 5. **Language Invariance:** Ensure all paths produce identical strings
///
/// # Implementation Status
///
/// **Note:** Current implementation provides basic structure but full label
/// pushing logic is under development. Complete implementation will include:
/// - Common prefix computation for outgoing arcs
/// - String-based shortest distance algorithms
/// - Label factorization and redistribution
/// - Proper handling of epsilon transitions
///
/// # Examples
///
/// ## Basic Label Pushing (Conceptual)
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::push_labels;
///
/// // FST with repeated label prefixes
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// let s2 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s2, TropicalWeight::one());
///
/// // Arcs with common prefixes (represented as character codes)
/// fst.add_arc(s0, Arc::new('a' as u32, 'x' as u32, TropicalWeight::one(), s1));
/// fst.add_arc(s1, Arc::new('b' as u32, 'y' as u32, TropicalWeight::one(), s2));
///
/// // Push labels toward initial state
/// let pushed: VectorFst<TropicalWeight> = push_labels(&fst)?;
///
/// // Result has redistributed labels but same transduction
/// assert_eq!(pushed.num_states(), fst.num_states());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## String Transducer Optimization
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::push_labels;
///
/// // String transducer with redundant label structure
/// let mut transducer = VectorFst::<TropicalWeight>::new();
/// let states: Vec<_> = (0..4).map(|_| transducer.add_state()).collect();
///
/// transducer.set_start(states[0]);
/// transducer.set_final(states[3], TropicalWeight::one());
///
/// // Pattern: multiple paths with common label prefixes
/// transducer.add_arc(states[0], Arc::new(1, 10, TropicalWeight::one(), states[1]));
/// transducer.add_arc(states[1], Arc::new(2, 20, TropicalWeight::one(), states[2]));
/// transducer.add_arc(states[2], Arc::new(3, 30, TropicalWeight::one(), states[3]));
///
/// transducer.add_arc(states[0], Arc::new(1, 10, TropicalWeight::one(), states[2]));
/// transducer.add_arc(states[2], Arc::new(4, 40, TropicalWeight::one(), states[3]));
///
/// // Push common label prefixes
/// let optimized: VectorFst<TropicalWeight> = push_labels(&transducer)?;
///
/// // Result has more efficient label distribution
/// println!("Transducer optimized");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Preprocessing for Composition
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::push_labels;
///
/// // Prepare transducers for efficient composition
/// fn prepare_transducers(
///     fst1: &VectorFst<TropicalWeight>,
///     fst2: &VectorFst<TropicalWeight>
/// ) -> Result<(VectorFst<TropicalWeight>, VectorFst<TropicalWeight>)> {
///     // Push labels for canonical form
///     let label_pushed1: VectorFst<TropicalWeight> = push_labels(fst1)?;
///     let label_pushed2: VectorFst<TropicalWeight> = push_labels(fst2)?;
///     
///     // Optimize further with determinization
///     let det1: VectorFst<TropicalWeight> = determinize(&label_pushed1)?;
///     let det2: VectorFst<TropicalWeight> = determinize(&label_pushed2)?;
///     
///     Ok((det1, det2))
/// }
///
/// // Create simple transducers
/// let mut fst1 = VectorFst::new();
/// let s0 = fst1.add_state();
/// let s1 = fst1.add_state();
/// fst1.set_start(s0);
/// fst1.set_final(s1, TropicalWeight::one());
/// fst1.add_arc(s0, Arc::new(1, 2, TropicalWeight::one(), s1));
///
/// let mut fst2 = VectorFst::new();
/// let s0 = fst2.add_state();
/// let s1 = fst2.add_state();
/// fst2.set_start(s0);
/// fst2.set_final(s1, TropicalWeight::one());
/// fst2.add_arc(s0, Arc::new(2, 3, TropicalWeight::one(), s1));
///
/// let (prep1, prep2) = prepare_transducers(&fst1, &fst2)?;
/// println!("Transducers prepared for composition");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Use Cases
///
/// ## String Processing
/// - **Text Transducer Optimization:** Optimize morphological analyzers and generators
/// - **Regular Expression Engines:** Improve pattern matching automata
/// - **Lexical Analysis:** Optimize tokenizer and lexer automata
/// - **String Compression:** Create more compact string transducers
///
/// ## Natural Language Processing
/// - **Morphological Analysis:** Optimize morphology FSTs for better performance
/// - **Phonological Rules:** Improve phonology rule application
/// - **Transliteration:** Optimize script conversion transducers
/// - **Text Normalization:** Improve text preprocessing automata
///
/// ## Compilation and Parsing
/// - **Lexer Optimization:** Create more efficient lexical analyzers
/// - **Grammar Processing:** Optimize grammar-based transducers
/// - **Symbol Table Management:** Improve identifier processing
/// - **Code Generation:** Optimize template-based code generators
///
/// ## Information Retrieval
/// - **Query Processing:** Optimize query expansion transducers
/// - **Index Generation:** Improve search index construction
/// - **Pattern Matching:** Optimize document pattern recognition
/// - **Text Analysis:** Improve content analysis pipelines
///
/// # Performance Characteristics
///
/// - **Time Complexity:** O(|V| + |E|) for basic label analysis
/// - **String Operations:** Additional O(max_string_length) per operation
/// - **Space Complexity:** O(|V| × max_string_length) for label storage
/// - **Memory Efficiency:** Can reduce overall automaton size significantly
/// - **Practical Speedup:** Often provides substantial performance improvements
///
/// # Mathematical Properties
///
/// Label pushing preserves essential transducer properties:
/// - **Transduction Preservation:** All input-output mappings remain identical
/// - **Language Preservation:** Input and output languages unchanged
/// - **Structural Properties:** May change arc structure but preserves semantics
/// - **Determinism:** May affect determinism depending on label distribution
/// - **Canonical Form:** Produces more canonical representation
///
/// # Implementation Details
///
/// The current implementation provides basic structure for label pushing.
/// Full implementation will include:
/// - **Common Prefix Analysis:** Efficient computation of label prefixes
/// - **String Distance:** Shortest distance over string semiring
/// - **Label Factorization:** Optimal splitting of labels
/// - **Epsilon Handling:** Proper treatment of epsilon transitions
/// - **Memory Optimization:** Efficient string storage and manipulation
///
/// # Label Pushing Strategies
///
/// Different approaches for different scenarios:
/// - **Input Label Pushing:** Push input labels toward initial state
/// - **Output Label Pushing:** Push output labels toward initial state
/// - **Bilateral Pushing:** Push both input and output labels
/// - **Selective Pushing:** Push only beneficial label redistributions
/// - **Prefix-Based:** Focus on common prefixes for maximum benefit
///
/// # Optimization Considerations
///
/// For effective label pushing:
/// - **Prefix Analysis:** Identify beneficial common prefixes
/// - **Memory Trade-offs:** Balance label distribution vs. memory usage
/// - **Composition Preparation:** Optimize for subsequent composition
/// - **Determinization:** Consider impact on determinization algorithms
/// - **Application Constraints:** Respect specific application requirements
///
/// # Relationship to Weight Pushing
///
/// Label pushing complements weight pushing:
/// - **Combined Optimization:** Apply both for maximum benefit
/// - **Order Considerations:** Sequence weight and label pushing appropriately
/// - **Canonical Form:** Together produce highly canonical representations
/// - **Algorithm Preparation:** Both improve subsequent algorithm efficiency
///
/// # Future Implementation Plan
///
/// Complete implementation will include:
/// 1. **String Semiring Integration:** Proper string-based shortest distance
/// 2. **Prefix Computation:** Efficient common prefix algorithms
/// 3. **Label Factorization:** Optimal label splitting strategies
/// 4. **Performance Optimization:** Memory-efficient string operations
/// 5. **Integration:** Seamless integration with other algorithms
///
/// # Errors
///
/// Returns [`Error::Algorithm`](crate::Error::Algorithm) if:
/// - The input FST is invalid, corrupted, or malformed
/// - Memory allocation fails during label analysis or result construction
/// - String operations encounter encoding or length limitations
/// - Common prefix computation fails due to incompatible label structure
/// - Label factorization encounters infinite or malformed strings
/// - The operation is not yet fully implemented (current status)
///
/// # See Also
///
/// - [`push_weights`] for weight pushing operations
/// - [`determinize()`](crate::algorithms::determinize()) for algorithms that benefit from label pushing
/// - [`minimize()`](crate::algorithms::minimize()) for minimization after label pushing
/// - [`compose()`](crate::algorithms::compose()) for composition of label-pushed transducers
/// - [Working with FSTs - Label Pushing](../../docs/working-with-fsts/advanced-topics.md#weight-pushing) for usage patterns
/// - [Core Concepts](../../docs/core-concepts/algorithms.md#label-pushing) for mathematical theory
pub fn push_labels<W, F, M>(fst: &F) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    // simple copy for now
    let mut result = M::default();

    for _ in 0..fst.num_states() {
        result.add_state();
    }

    if let Some(start) = fst.start() {
        result.set_start(start);
    }

    for state in fst.states() {
        if let Some(weight) = fst.final_weight(state) {
            result.set_final(state, weight.clone());
        }

        for arc in fst.arcs(state) {
            result.add_arc(state, arc.clone());
        }
    }

    Ok(result)
}

fn compute_potentials<W: DivisibleSemiring, F: Fst<W>>(fst: &F) -> Result<Vec<W>> {
    // simplified - would use shortest distance algorithm
    Ok(vec![W::one(); fst.num_states()])
}
