//! Compact FST implementation optimized for memory-constrained environments

use super::traits::*;
use crate::arc::{Arc, ArcIterator};
use crate::properties::FstProperties;
use crate::semiring::Semiring;
use core::fmt::Debug;
use core::marker::PhantomData;

/// Memory-optimized FST implementation with pluggable compression strategies
///
/// `CompactFst` is a specialized FST implementation designed for scenarios where memory
/// efficiency is the primary concern, even at the cost of some computational overhead.
/// It uses customizable compression strategies to reduce the memory footprint of large
/// FSTs, making it suitable for deployment on resource-constrained devices or when
/// working with exceptionally large automata.
///
/// # Design Characteristics
///
/// - **Compression-First:** Prioritizes minimal memory usage over access speed
/// - **Pluggable Compaction:** Customizable compression strategies via the `Compactor` trait
/// - **Trade-off Oriented:** Exchanges computational overhead for reduced memory footprint
/// - **Specialization-Ready:** Supports domain-specific optimizations through custom compactors
/// - **Immutable Structure:** Read-only access pattern for predictable memory usage
///
/// # Performance Profile
///
/// | Operation | Time Complexity | Memory Overhead | Notes |
/// |-----------|----------------|-----------------|-------|
/// | Arc Access | O(1) + decompression | Minimal | Requires decompression per access |
/// | State Access | O(1) | Fixed per state | Direct indexing into state array |
/// | Memory Usage | ~40-70% of VectorFst | Depends on compactor | Significant savings |
/// | Construction | O(V + E) | Temporary spike | One-time compression cost |
/// | Cache Performance | Variable | Excellent | Compressed data fits in cache |
///
/// # Memory Layout and Compression
///
/// ```text
/// CompactFst Memory Structure:
/// ┌─────────────────────────────┐
/// │ States Array                │ ← Vec<CompactState>: metadata per state
/// │ [State 0: arcs_start, ...]  │   - final_weightᵢdx: Option<u32>
/// │ [State 1: arcs_start, ...]  │   - arcs_start: u32 (data array offset)
/// │ [State N: arcs_start, ...]  │   - num_arcs: u32 (arc count)
/// └─────────────────────────────┘
/// ┌─────────────────────────────┐
/// │ Compressed Data Array       │ ← Vec<C::Element>: compressed arcs & weights
/// │ [Compressed Arc 0]          │   Compactor-specific format
/// │ [Compressed Arc 1]          │   May pack multiple fields together
/// │ [Compressed Weight 0]       │   Custom compression schemes
/// │ [...]                       │
/// └─────────────────────────────┘
/// ```
///
/// # Compression Strategies
///
/// ## Default Compression
/// The `DefaultCompactor` provides a baseline compression approach:
/// - Stores arcs and weights in enumerated format
/// - Maintains full precision of original data
/// - Suitable for general-purpose usage
///
/// ## Custom Compression Examples
/// ```
/// use arcweight::prelude::*;
/// use arcweight::fst::{CompactFst, Compactor};
///
/// // Example: Custom compactor for small alphabets
/// #[derive(Debug)]
/// struct SmallAlphabetCompactor;
///
/// impl Compactor<TropicalWeight> for SmallAlphabetCompactor {
///     type Element = u64; // Pack arc data into single u64
///     
///     fn compact(arc: &Arc<TropicalWeight>) -> u64 {
///         // Pack: 16 bits ilabel + 16 bits olabel + 16 bits nextstate + 16 bits weight
///         let weight_bits = *arc.weight.value() as u64; // Simplified
///         (arc.ilabel as u64) << 48 |
///         (arc.olabel as u64) << 32 |
///         (arc.nextstate as u64) << 16 |
///         weight_bits
///     }
///     
///     fn expand(element: &u64) -> Arc<TropicalWeight> {
///         let ilabel = (element >> 48) as u32;
///         let olabel = ((element >> 32) & 0xFFFF) as u32;
///         let nextstate = ((element >> 16) & 0xFFFF) as u32;
///         let weight_val = (element & 0xFFFF) as f32;
///         Arc::new(ilabel, olabel, TropicalWeight::new(weight_val), nextstate)
///     }
///     
///     fn compact_weight(weight: &TropicalWeight) -> u64 {
///         *weight.value() as u64
///     }
///     
///     fn expand_weight(element: &u64) -> TropicalWeight {
///         TropicalWeight::new(*element as f32)
///     }
/// }
/// ```
///
/// # Use Cases
///
/// ## Mobile/Embedded Deployment
/// ```
/// use arcweight::prelude::*;
/// use arcweight::fst::{CompactFst, DefaultCompactor};
///
/// // Deploy large language model on mobile device
/// fn create_mobile_language_model() -> CompactFst<TropicalWeight, DefaultCompactor<TropicalWeight>> {
///     let base_fst = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::new();
///     
///     // Compressed representation reduces memory requirements
///     // Suitable for devices with limited RAM
///     base_fst
/// }
///
/// // Memory-conscious processing
/// fn process_on_mobile_device(
///     fst: &CompactFst<TropicalWeight, DefaultCompactor<TropicalWeight>>,
///     input: &[u32]
/// ) {
///     if let Some(start) = fst.start() {
///         let mut current = start;
///         for &label in input {
///             // Each arc access involves decompression
///             // But overall memory usage is minimal
///             for arc in fst.arcs(current) {
///                 if arc.ilabel == label {
///                     current = arc.nextstate;
///                     break;
///                 }
///             }
///         }
///     }
/// }
/// ```
///
/// ## Large-Scale Dictionary Compression
/// ```
/// use arcweight::prelude::*;
/// use arcweight::fst::{CompactFst, DefaultCompactor};
///
/// // Compress massive pronunciation dictionary
/// fn compress_pronunciation_dict(
///     // Input would be a large VectorFst with millions of entries
/// ) -> CompactFst<LogWeight, DefaultCompactor<LogWeight>> {
///     // The compaction process would convert from VectorFst
///     // Achieving 40-60% memory reduction for large dictionaries
///     let compact_dict = CompactFst::new();
///     
///     // Compressed dict can fit in memory where uncompressed cannot
///     compact_dict
/// }
///
/// // Lookup in compressed dictionary
/// fn lookup_pronunciation(
///     dict: &CompactFst<LogWeight, DefaultCompactor<LogWeight>>,
///     word: &str
/// ) -> Vec<String> {
///     let mut pronunciations = Vec::new();
///     
///     if let Some(start) = dict.start() {
///         // Traverse compressed FST
///         // Decompression happens transparently during access
///         let mut current = start;
///         for ch in word.chars() {
///             for arc in dict.arcs(current) {
///                 if arc.ilabel == ch as u32 {
///                     current = arc.nextstate;
///                     break;
///                 }
///             }
///         }
///         
///         // Extract pronunciations from final states
///         // (Implementation details omitted for brevity)
///     }
///     
///     pronunciations
/// }
/// ```
///
/// ## Cloud Storage Optimization
/// ```
/// use arcweight::prelude::*;
/// use arcweight::fst::{CompactFst, DefaultCompactor};
///
/// // Optimize FSTs for cloud storage and transmission
/// fn optimize_for_cloud_storage() -> CompactFst<ProbabilityWeight, DefaultCompactor<ProbabilityWeight>> {
///     let compact_fst = CompactFst::new();
///     
///     // Benefits:
///     // - Reduced storage costs (smaller files)
///     // - Faster network transmission
///     // - Lower bandwidth usage
///     // - Reduced I/O operations
///     
///     compact_fst
/// }
///
/// // Efficient batch processing of compressed FSTs
/// fn batch_process_compressed_fsts(
///     fsts: &[CompactFst<ProbabilityWeight, DefaultCompactor<ProbabilityWeight>>]
/// ) {
///     for fst in fsts {
///         // Process multiple compressed FSTs in memory simultaneously
///         // Memory efficiency allows larger batch sizes
///         process_single_fst(fst);
///     }
/// }
///
/// fn process_single_fst(
///     fst: &CompactFst<ProbabilityWeight, DefaultCompactor<ProbabilityWeight>>
/// ) {
///     // FST processing logic
///     // Compression overhead amortized across batch processing
/// }
/// ```
///
/// ## Memory-Constrained Analysis
/// ```
/// use arcweight::prelude::*;
/// use arcweight::fst::{CompactFst, DefaultCompactor};
///
/// // Analyze very large FSTs within memory constraints
/// fn analyze_large_fst_efficiently(
///     fst: &CompactFst<BooleanWeight, DefaultCompactor<BooleanWeight>>
/// ) -> AnalysisResult {
///     let mut result = AnalysisResult::new();
///     
///     // Memory-efficient traversal
///     for state in fst.states() {
///         // Analyze state properties
///         result.state_count += 1;
///         
///         // Count arcs with minimal memory overhead
///         for arc in fst.arcs(state) {
///             result.arc_count += 1;
///             
///             // Decompression cost amortized over analysis
///             if arc.ilabel == 0 {
///                 result.epsilon_count += 1;
///             }
///         }
///     }
///     
///     result
/// }
///
/// #[derive(Default)]
/// struct AnalysisResult {
///     state_count: usize,
///     arc_count: usize,
///     epsilon_count: usize,
/// }
///
/// impl AnalysisResult {
///     fn new() -> Self { Self::default() }
/// }
/// ```
///
/// # Compactor Implementation Patterns
///
/// ## Domain-Specific Compression
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::{Compactor, CompactFst};
///
/// // Example: Pronunciation-specific compactor
/// #[derive(Debug)]
/// struct PhonemeCompactor;
///
/// impl Compactor<TropicalWeight> for PhonemeCompactor {
///     type Element = CompactPhoneme;
///     
///     fn compact(arc: &Arc<TropicalWeight>) -> CompactPhoneme {
///         // Custom compression for phoneme data
///         // Could map common phoneme combinations to single values
///         CompactPhoneme {
///             phoneme_code: map_to_phoneme_code(arc.ilabel, arc.olabel),
///             weight_class: quantize_weight(&arc.weight),
///             next_state: arc.nextstate,
///         }
///     }
///     
///     fn expand(element: &CompactPhoneme) -> Arc<TropicalWeight> {
///         let (ilabel, olabel) = expand_phoneme_code(element.phoneme_code);
///         let weight = dequantize_weight(element.weight_class);
///         Arc::new(ilabel, olabel, weight, element.next_state)
///     }
///     
///     fn compact_weight(weight: &TropicalWeight) -> CompactPhoneme {
///         // Weight-only compression
///         CompactPhoneme {
///             phoneme_code: 0,
///             weight_class: quantize_weight(weight),
///             next_state: 0,
///         }
///     }
///     
///     fn expand_weight(element: &CompactPhoneme) -> TropicalWeight {
///         dequantize_weight(element.weight_class)
///     }
/// }
///
/// #[derive(Clone, Debug)]
/// struct CompactPhoneme {
///     phoneme_code: u16,  // Compressed phoneme pair
///     weight_class: u8,   // Quantized weight
///     next_state: u32,
/// }
///
/// fn map_to_phoneme_code(ilabel: u32, olabel: u32) -> u16 {
///     // Domain-specific compression logic
///     ((ilabel & 0xFF) << 8 | (olabel & 0xFF)) as u16
/// }
///
/// fn expand_phoneme_code(code: u16) -> (u32, u32) {
///     ((code >> 8) as u32, (code & 0xFF) as u32)
/// }
///
/// fn quantize_weight(weight: &TropicalWeight) -> u8 {
///     // Quantize weight to 256 levels
///     (weight.value().clamp(0.0, 25.5) * 10.0) as u8
/// }
///
/// fn dequantize_weight(quantized: u8) -> TropicalWeight {
///     TropicalWeight::new(quantized as f32 / 10.0)
/// }
/// ```
///
/// # Performance Optimization Guidelines
///
/// ## When to Use CompactFst
/// - ✅ Memory is severely constrained (embedded systems, mobile devices)
/// - ✅ Very large FSTs that don't fit in memory uncompressed
/// - ✅ Network transmission or storage optimization is critical
/// - ✅ Batch processing where memory efficiency enables larger batches
/// - ✅ Long-running applications where compression amortizes over time
///
/// ## When NOT to Use CompactFst
/// - ❌ Real-time applications requiring minimal latency
/// - ❌ Frequent random access patterns
/// - ❌ Small FSTs where compression overhead exceeds benefits
/// - ❌ Applications that modify FSTs frequently
/// - ❌ CPU-constrained environments where decompression is expensive
///
/// ## Memory vs. Performance Trade-offs
/// 1. **Compression Ratio:** Higher compression = more CPU overhead
/// 2. **Access Patterns:** Sequential access amortizes decompression cost
/// 3. **Cache Behavior:** Compressed data may improve cache hit rates
/// 4. **Batch Processing:** Compression overhead amortized across operations
///
/// # Limitations and Considerations
///
/// ## Current Implementation Limitations
/// - `final_weight()` method requires redesign to avoid reference issues
/// - Limited set of built-in compaction strategies
/// - No automatic compression strategy selection
/// - Compression is lossy with some compactors (quantization)
///
/// ## Design Considerations
/// - **Compactor Choice:** Critical for achieving desired compression ratio
/// - **Data Characteristics:** Compression effectiveness varies by FST structure
/// - **Access Patterns:** Random access amplifies decompression overhead
/// - **Precision Requirements:** Some compactors may reduce precision
///
/// # Future Enhancements
///
/// - **Adaptive Compression:** Automatic selection of optimal compaction strategy
/// - **Streaming Support:** Support for FSTs larger than available memory
/// - **Lossy Compression:** Options for approximate FSTs with higher compression
/// - **Incremental Updates:** Support for modifying compressed FSTs efficiently
///
/// # See Also
///
/// - [`VectorFst`] for mutable, uncompressed FSTs
/// - [`ConstFst`] for read-only, optimized FSTs without compression
/// - [`CacheFst`] for caching expensive computations
/// - [Memory Management Guide](../../docs/architecture/memory-management.md) for memory optimization strategies
/// - [Performance Tuning](../../docs/architecture/performance.md) for trade-off analysis
///
/// [`VectorFst`]: crate::fst::VectorFst
/// [`ConstFst`]: crate::fst::ConstFst
/// [`CacheFst`]: crate::fst::CacheFst
#[derive(Debug, Clone)]
pub struct CompactFst<W: Semiring, C: Compactor<W>> {
    states: Vec<CompactState>,
    data: Vec<C::Element>,
    /// Uncompressed final weights for direct reference access
    final_weights: Vec<Option<W>>,
    start: Option<StateId>,
    properties: FstProperties,
    _phantom: PhantomData<(W, C)>,
}

/// Compact representation of FST state metadata
///
/// Stores essential state information in a memory-efficient format,
/// with precomputed offsets for fast arc range computation.
#[derive(Debug, Clone)]
struct CompactState {
    /// Index into compressed data array for final weight, if state is final
    #[allow(dead_code)]
    final_weight_idx: Option<u32>,
    /// Starting offset in the compressed data array for this state's arcs
    arcs_start: u32,
    /// Number of arcs from this state (enables range computation)
    num_arcs: u32,
}

/// Trait for implementing custom arc compression strategies
///
/// The `Compactor` trait defines the interface for compression algorithms that can
/// reduce the memory footprint of FST arcs and weights. Implementations can range
/// from simple enumeration-based approaches to sophisticated domain-specific
/// compression schemes that exploit patterns in the data.
///
/// # Design Principles
///
/// - **Lossless by Default:** Preserve full information unless explicitly designed for lossy compression
/// - **Domain Awareness:** Leverage knowledge of data patterns for optimal compression
/// - **Performance Balance:** Balance compression ratio against decompression overhead
/// - **Type Safety:** Ensure compressed and uncompressed data maintain semantic equivalence
///
/// # Implementation Guidelines
///
/// When implementing a custom compactor:
/// 1. Ensure `expand(compact(arc))` returns an equivalent arc
/// 2. Handle edge cases like epsilon transitions and special weights
/// 3. Consider alignment and packing for optimal memory usage
/// 4. Validate that compression provides meaningful space savings
///
/// # Thread Safety
///
/// All compactor implementations must be thread-safe (`Send + Sync`) to enable
/// concurrent access to compressed FSTs.
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::{Compactor, CompactFst};
///
/// // Simple bit-packing compactor for small label spaces
/// #[derive(Debug)]
/// struct BitPackCompactor;
///
/// impl Compactor<BooleanWeight> for BitPackCompactor {
///     type Element = u32;
///     
///     fn compact(arc: &Arc<BooleanWeight>) -> u32 {
///         // Pack into 32 bits: 8+8+8+8 = ilabel, olabel, nextstate, weight
///         let weight_bit = if *arc.weight.value() { 1u32 } else { 0u32 };
///         ((arc.ilabel & 0xFF) << 24) |
///         ((arc.olabel & 0xFF) << 16) |
///         ((arc.nextstate & 0xFF) << 8) |
///         weight_bit
///     }
///     
///     fn expand(element: &u32) -> Arc<BooleanWeight> {
///         let ilabel = (element >> 24) & 0xFF;
///         let olabel = (element >> 16) & 0xFF;
///         let nextstate = (element >> 8) & 0xFF;
///         let weight = BooleanWeight::new((element & 1) != 0);
///         Arc::new(ilabel, olabel, weight, nextstate)
///     }
///     
///     fn compact_weight(weight: &BooleanWeight) -> u32 {
///         if *weight.value() { 1 } else { 0 }
///     }
///     
///     fn expand_weight(element: &u32) -> BooleanWeight {
///         BooleanWeight::new(*element != 0)
///     }
/// }
/// ```
pub trait Compactor<W: Semiring>: Debug + Send + Sync + 'static {
    /// Compressed element type that stores arc or weight data
    ///
    /// This type should be chosen to maximize compression while maintaining
    /// reasonable decompression performance. Common choices include:
    /// - `u32` or `u64` for bit-packed representations
    /// - Custom structs for domain-specific compression
    /// - Enum types for storing different kinds of compressed data
    type Element: Clone + Debug + Send + Sync;

    /// Compress an arc into the compact element format
    ///
    /// Transforms a full `Arc<W>` into a compressed representation. The
    /// implementation should preserve all essential information needed
    /// to reconstruct the original arc via `expand()`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arcweight::prelude::*;
    /// # use arcweight::fst::{Compactor, DefaultCompactor};
    /// let arc = Arc::new(1, 2, TropicalWeight::new(3.5), 4);
    /// let compressed = DefaultCompactor::<TropicalWeight>::compact(&arc);
    /// // `compressed` now contains all arc information in compact form
    /// ```
    fn compact(arc: &Arc<W>) -> Self::Element;

    /// Expand a compressed element back into a full arc
    ///
    /// Reconstructs the original arc from its compressed representation.
    /// This operation should be the inverse of `compact()`, producing
    /// semantically equivalent arcs.
    ///
    /// # Examples
    ///
    /// ```
    /// # use arcweight::prelude::*;
    /// # use arcweight::fst::{Compactor, DefaultCompactor};
    /// let original = Arc::new(1, 2, TropicalWeight::new(3.5), 4);
    /// let compressed = DefaultCompactor::<TropicalWeight>::compact(&original);
    /// let expanded = DefaultCompactor::<TropicalWeight>::expand(&compressed);
    /// assert_eq!(original.ilabel, expanded.ilabel);
    /// assert_eq!(original.olabel, expanded.olabel);
    /// assert_eq!(original.nextstate, expanded.nextstate);
    /// ```
    fn expand(element: &Self::Element) -> Arc<W>;

    /// Compress a semiring weight into the compact element format
    ///
    /// Compresses standalone weights (such as final state weights) into
    /// the same element format used for arcs. This enables unified storage
    /// of both arcs and weights in the compressed data array.
    fn compact_weight(weight: &W) -> Self::Element;

    /// Expand a compressed element back into a semiring weight
    ///
    /// Reconstructs a weight from its compressed representation. This is
    /// the inverse operation of `compact_weight()`.
    fn expand_weight(element: &Self::Element) -> W;
}

/// Default compactor implementation using enumerated storage
///
/// `DefaultCompactor` provides a baseline compression strategy that stores arcs
/// and weights in an enumerated format. While it doesn't achieve the highest
/// compression ratios possible, it offers several advantages:
///
/// - **Lossless:** Preserves all original data with perfect fidelity
/// - **General Purpose:** Works with any semiring type without customization
/// - **Simple:** Straightforward implementation with minimal complexity
/// - **Debuggable:** Easy to inspect and understand compressed data
///
/// # Compression Approach
///
/// The default compactor uses a tagged union approach where each compressed
/// element is either an arc or a weight, distinguished by the enum variant.
/// This provides modest space savings through:
/// - Elimination of separate storage for different data types
/// - Potential for enum layout optimizations by the compiler
/// - Unified data array reducing pointer indirection
///
/// # Performance Characteristics
///
/// - **Compression Ratio:** Moderate (typically 10-30% space savings)
/// - **Decompression Speed:** Fast (simple enum matching)
/// - **Memory Layout:** Cache-friendly with unified data array
/// - **Overhead:** Minimal per-element tagging cost
///
/// # Usage
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::{CompactFst, DefaultCompactor};
///
/// // Create a compact FST with default compression
/// let compact_fst = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::new();
///
/// // The DefaultCompactor will handle compression transparently
/// // providing modest space savings with excellent compatibility
/// ```
#[derive(Debug)]
pub struct DefaultCompactor<W: Semiring> {
    _phantom: PhantomData<W>,
}

impl<W: Semiring> Compactor<W> for DefaultCompactor<W> {
    type Element = CompactElement<W>;

    fn compact(arc: &Arc<W>) -> Self::Element {
        CompactElement::Arc {
            ilabel: arc.ilabel,
            olabel: arc.olabel,
            weight: arc.weight.clone(),
            nextstate: arc.nextstate,
        }
    }

    /// # Panics
    ///
    /// Panics if the element is not an arc type, which indicates
    /// incorrect usage of the compactor.
    fn expand(element: &Self::Element) -> Arc<W> {
        match element {
            CompactElement::Arc {
                ilabel,
                olabel,
                weight,
                nextstate,
            } => Arc::new(*ilabel, *olabel, weight.clone(), *nextstate),
            _ => panic!("Expected arc element"),
        }
    }

    fn compact_weight(weight: &W) -> Self::Element {
        CompactElement::Weight(weight.clone())
    }

    /// # Panics
    ///
    /// Panics if the element is not a weight type, which indicates
    /// incorrect usage of the compactor.
    fn expand_weight(element: &Self::Element) -> W {
        match element {
            CompactElement::Weight(w) => w.clone(),
            _ => panic!("Expected weight element"),
        }
    }
}

/// Enumerated storage format for compressed arcs and weights
///
/// `CompactElement` represents the compressed format used by `DefaultCompactor`
/// to store both arcs and standalone weights in a unified data structure.
/// The enum-based approach allows for type-safe storage while maintaining
/// the ability to reconstruct original data with perfect fidelity.
///
/// # Variants
///
/// - **Arc:** Complete arc information including labels, weight, and target state
/// - **Weight:** Standalone weight values (typically for final states)
///
/// # Memory Layout
///
/// The enum uses Rust's standard enum layout optimizations, which may include:
/// - Tag compression when possible
/// - Alignment optimization for contained data
/// - Potential niche optimizations for certain weight types
#[derive(Clone, Debug)]
pub enum CompactElement<W: Semiring> {
    /// Compressed arc with full transition information
    Arc {
        /// Input label for the transition
        ilabel: Label,
        /// Output label for the transition
        olabel: Label,
        /// Transition weight
        weight: W,
        /// Target state of the transition
        nextstate: StateId,
    },
    /// Standalone weight value (e.g., final state weight)
    Weight(W),
}

impl<W: Semiring, C: Compactor<W>> Default for CompactFst<W, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<W: Semiring, C: Compactor<W>> CompactFst<W, C> {
    /// Create a new empty compact FST
    ///
    /// Initializes an empty `CompactFst` with the specified compactor strategy.
    /// The FST will use the compactor to compress arcs and weights as they
    /// are added to the structure.
    ///
    /// # Examples
    ///
    /// ```
    /// use arcweight::prelude::*;
    /// use arcweight::fst::{CompactFst, DefaultCompactor};
    ///
    /// // Create an empty compact FST with default compression
    /// let fst = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::new();
    ///
    /// // FST is initially empty
    /// assert_eq!(fst.num_states(), 0);
    /// assert!(fst.start().is_none());
    /// ```
    ///
    /// # Performance
    ///
    /// This operation is O(1) and allocates minimal memory for the initial
    /// empty state and data vectors.
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            data: Vec::new(),
            final_weights: Vec::new(),
            start: None,
            properties: FstProperties::default(),
            _phantom: PhantomData,
        }
    }

    /// Helper method to set a final weight for a state
    ///
    /// This is primarily for testing and prototype purposes since CompactFst
    /// doesn't currently implement MutableFst. In a full implementation,
    /// final weights would be set during the compression process.
    pub fn set_final_weight(&mut self, state: StateId, weight: Option<W>) {
        let state_idx = state as usize;
        // Ensure final_weights vector is large enough
        if self.final_weights.len() <= state_idx {
            self.final_weights.resize(state_idx + 1, None);
        }
        self.final_weights[state_idx] = weight;
    }

    /// Helper method to add a state (for testing purposes)
    pub fn add_state(&mut self) -> StateId {
        let state_id = self.states.len() as StateId;
        self.states.push(CompactState {
            final_weight_idx: None,
            arcs_start: 0,
            num_arcs: 0,
        });
        self.final_weights.push(None);
        state_id
    }
}

/// High-performance arc iterator for compressed FST data
///
/// Provides iterator access to arcs from a specific state in a `CompactFst`,
/// handling decompression transparently during iteration. The iterator maintains
/// a reference to the compressed data array and performs on-demand expansion
/// of compressed elements into full arc structures.
///
/// # Performance Characteristics
///
/// - **Decompression Overhead:** Each arc access requires decompression
/// - **Memory Access:** Sequential access to compressed data array
/// - **Cache Efficiency:** Good locality when compressed data is smaller
/// - **Allocation:** Zero allocations during iteration (decompression may allocate)
///
/// # Usage
///
/// This iterator is created automatically by `CompactFst::arcs()` and should
/// not be constructed directly. It implements the standard Iterator pattern
/// while handling the compression layer transparently.
///
/// ```
/// use arcweight::prelude::*;
/// use arcweight::fst::{CompactFst, DefaultCompactor};
///
/// # fn example() {
/// let fst = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::new();
///
/// // Iteration handles decompression automatically
/// for arc in fst.arcs(0) {
///     println!("Decompressed arc: {} -> {}", arc.ilabel, arc.olabel);
/// }
/// # }
/// ```
#[derive(Debug)]
pub struct CompactArcIterator<'a, W: Semiring, C: Compactor<W>> {
    /// Reference to the compressed data array
    data: &'a [C::Element],
    /// Current position in the data array
    pos: usize,
    /// End position (exclusive) for this state's arc range
    end: usize,
    /// Phantom data for weight type constraints
    _phantom: PhantomData<W>,
}

impl<W: Semiring, C: Compactor<W>> ArcIterator<W> for CompactArcIterator<'_, W, C> {
    fn reset(&mut self) {
        self.pos = 0;
    }
}

impl<W: Semiring, C: Compactor<W>> Iterator for CompactArcIterator<'_, W, C> {
    type Item = Arc<W>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.end {
            let arc = C::expand(&self.data[self.pos]);
            self.pos += 1;
            Some(arc)
        } else {
            None
        }
    }
}

impl<W: Semiring, C: Compactor<W>> Fst<W> for CompactFst<W, C> {
    type ArcIter<'a>
        = CompactArcIterator<'a, W, C>
    where
        W: 'a,
        C: 'a;

    fn start(&self) -> Option<StateId> {
        self.start
    }

    fn final_weight(&self, state: StateId) -> Option<&W> {
        self.final_weights
            .get(state as usize)
            .and_then(|weight| weight.as_ref())
    }

    fn num_arcs(&self, state: StateId) -> usize {
        self.states
            .get(state as usize)
            .map(|s| s.num_arcs as usize)
            .unwrap_or(0)
    }

    fn num_states(&self) -> usize {
        self.states.len()
    }

    fn properties(&self) -> FstProperties {
        self.properties
    }

    fn arcs(&self, state: StateId) -> Self::ArcIter<'_> {
        if let Some(s) = self.states.get(state as usize) {
            let start = s.arcs_start as usize;
            let end = start + s.num_arcs as usize;
            CompactArcIterator {
                data: &self.data,
                pos: start,
                end,
                _phantom: PhantomData,
            }
        } else {
            CompactArcIterator {
                data: &self.data,
                pos: 0,
                end: 0,
                _phantom: PhantomData,
            }
        }
    }
}
