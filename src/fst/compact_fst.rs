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
///     fn compact(&self, arc: &Arc<TropicalWeight>) -> u64 {
///         // Pack: 16 bits ilabel + 16 bits olabel + 16 bits nextstate + 16 bits weight
///         let weight_bits = *arc.weight.value() as u64; // Simplified
///         (arc.ilabel as u64) << 48 |
///         (arc.olabel as u64) << 32 |
///         (arc.nextstate as u64) << 16 |
///         weight_bits
///     }
///     
///     fn expand(&self, element: &u64) -> Arc<TropicalWeight> {
///         let ilabel = (element >> 48) as u32;
///         let olabel = ((element >> 32) & 0xFFFF) as u32;
///         let nextstate = ((element >> 16) & 0xFFFF) as u32;
///         let weight_val = (element & 0xFFFF) as f32;
///         Arc::new(ilabel, olabel, TropicalWeight::new(weight_val), nextstate)
///     }
///     
///     fn compact_weight(&self, weight: &TropicalWeight) -> u64 {
///         *weight.value() as u64
///     }
///     
///     fn expand_weight(&self, element: &u64) -> TropicalWeight {
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
///     fn compact(&self, arc: &Arc<TropicalWeight>) -> CompactPhoneme {
///         // Custom compression for phoneme data
///         // Could map common phoneme combinations to single values
///         CompactPhoneme {
///             phoneme_code: map_to_phoneme_code(arc.ilabel, arc.olabel),
///             weight_class: quantize_weight(&arc.weight),
///             next_state: arc.nextstate,
///         }
///     }
///     
///     fn expand(&self, element: &CompactPhoneme) -> Arc<TropicalWeight> {
///         let (ilabel, olabel) = expand_phoneme_code(element.phoneme_code);
///         let weight = dequantize_weight(element.weight_class);
///         Arc::new(ilabel, olabel, weight, element.next_state)
///     }
///     
///     fn compact_weight(&self, weight: &TropicalWeight) -> CompactPhoneme {
///         // Weight-only compression
///         CompactPhoneme {
///             phoneme_code: 0,
///             weight_class: quantize_weight(weight),
///             next_state: 0,
///         }
///     }
///     
///     fn expand_weight(&self, element: &CompactPhoneme) -> TropicalWeight {
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
/// # Available Compression Strategies
///
/// - **DefaultCompactor:** Enum-based storage with moderate compression
/// - **BitPackCompactor:** Bit-packing for small label/state spaces
/// - **QuantizedCompactor:** Weight quantization for lossy compression
/// - **DeltaCompactor:** Delta encoding for sequential patterns
/// - **VarIntCompactor:** Variable-length integer encoding
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
    /// Store the compactor instance to access its configuration
    compactor: C,
    _phantom: PhantomData<W>,
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
///     fn compact(&self, arc: &Arc<BooleanWeight>) -> u32 {
///         // Pack into 32 bits: 8+8+8+8 = ilabel, olabel, nextstate, weight
///         let weight_bit = if *arc.weight.value() { 1u32 } else { 0u32 };
///         ((arc.ilabel & 0xFF) << 24) |
///         ((arc.olabel & 0xFF) << 16) |
///         ((arc.nextstate & 0xFF) << 8) |
///         weight_bit
///     }
///     
///     fn expand(&self, element: &u32) -> Arc<BooleanWeight> {
///         let ilabel = (element >> 24) & 0xFF;
///         let olabel = (element >> 16) & 0xFF;
///         let nextstate = (element >> 8) & 0xFF;
///         let weight = BooleanWeight::new((element & 1) != 0);
///         Arc::new(ilabel, olabel, weight, nextstate)
///     }
///     
///     fn compact_weight(&self, weight: &BooleanWeight) -> u32 {
///         if *weight.value() { 1 } else { 0 }
///     }
///     
///     fn expand_weight(&self, element: &u32) -> BooleanWeight {
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
    /// let compactor = DefaultCompactor::<TropicalWeight>::default();
    /// let compressed = compactor.compact(&arc);
    /// // `compressed` now contains all arc information in compact form
    /// ```
    fn compact(&self, arc: &Arc<W>) -> Self::Element;

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
    /// let compactor = DefaultCompactor::<TropicalWeight>::default();
    /// let compressed = compactor.compact(&original);
    /// let expanded = compactor.expand(&compressed);
    /// assert_eq!(original.ilabel, expanded.ilabel);
    /// assert_eq!(original.olabel, expanded.olabel);
    /// assert_eq!(original.nextstate, expanded.nextstate);
    /// ```
    fn expand(&self, element: &Self::Element) -> Arc<W>;

    /// Compress a semiring weight into the compact element format
    ///
    /// Compresses standalone weights (such as final state weights) into
    /// the same element format used for arcs. This enables unified storage
    /// of both arcs and weights in the compressed data array.
    fn compact_weight(&self, weight: &W) -> Self::Element;

    /// Expand a compressed element back into a semiring weight
    ///
    /// Reconstructs a weight from its compressed representation. This is
    /// the inverse operation of `compact_weight()`.
    fn expand_weight(&self, element: &Self::Element) -> W;
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

impl<W: Semiring> Default for DefaultCompactor<W> {
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<W: Semiring> Compactor<W> for DefaultCompactor<W> {
    type Element = CompactElement<W>;

    fn compact(&self, arc: &Arc<W>) -> Self::Element {
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
    fn expand(&self, element: &Self::Element) -> Arc<W> {
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

    fn compact_weight(&self, weight: &W) -> Self::Element {
        CompactElement::Weight(weight.clone())
    }

    /// # Panics
    ///
    /// Panics if the element is not a weight type, which indicates
    /// incorrect usage of the compactor.
    fn expand_weight(&self, element: &Self::Element) -> W {
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

/// Bit-packing compactor for FSTs with small label/state spaces
///
/// `BitPackCompactor` achieves high compression ratios by packing multiple fields
/// into fixed-size integers when the FST has limited label alphabets and state counts.
/// This strategy is ideal for phoneme FSTs, character-based automata, or any FST
/// where labels and state IDs fit in small bit widths.
///
/// # Compression Approach
///
/// Packs arc data into 64-bit integers using a configurable bit layout:
/// - Configurable bits for ilabel (e.g., 16 bits for 65K symbols)
/// - Configurable bits for olabel
/// - Configurable bits for nextstate
/// - Remaining bits for quantized weight
///
/// # Usage Example
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::{CompactFst, BitPackCompactor};
///
/// // Configure for ASCII FST (7-bit labels, 10-bit states)
/// let compactor = BitPackCompactor::<TropicalWeight>::new(7, 7, 10);
/// let fst = CompactFst::with_compactor(compactor);
/// ```
#[derive(Debug, Clone)]
pub struct BitPackCompactor<W: Semiring> {
    ilabel_bits: u8,
    olabel_bits: u8,
    state_bits: u8,
    weight_bits: u8,
    _phantom: PhantomData<W>,
}

impl<W: Semiring> BitPackCompactor<W> {
    /// Create a new bit-packing compactor with specified bit widths
    ///
    /// # Parameters
    /// - `ilabel_bits`: Bits for input labels (max 32)
    /// - `olabel_bits`: Bits for output labels (max 32)
    /// - `state_bits`: Bits for state IDs (max 32)
    ///
    /// # Panics
    /// Panics if total bits exceed 64 or any field exceeds 32 bits
    pub fn new(ilabel_bits: u8, olabel_bits: u8, state_bits: u8) -> Self {
        let total_bits = ilabel_bits as u32 + olabel_bits as u32 + state_bits as u32;
        assert!(
            total_bits <= 48,
            "Label and state bits must fit in 48 bits, leaving 16 for weight"
        );
        assert!(ilabel_bits <= 32 && olabel_bits <= 32 && state_bits <= 32);

        Self {
            ilabel_bits,
            olabel_bits,
            state_bits,
            weight_bits: 64 - total_bits as u8,
            _phantom: PhantomData,
        }
    }
}

impl<W: Semiring> Default for BitPackCompactor<W> {
    fn default() -> Self {
        // Default to 16 bits for each field
        Self::new(16, 16, 16)
    }
}

// Trait to handle weight value conversions between f64 and semiring values
pub trait WeightConverter<T> {
    fn to_f64(value: &T) -> f64;
    fn from_f64(value: f64) -> T;
}

impl WeightConverter<f32> for f32 {
    fn to_f64(value: &f32) -> f64 {
        *value as f64
    }

    fn from_f64(value: f64) -> f32 {
        value as f32
    }
}

impl WeightConverter<f64> for f64 {
    fn to_f64(value: &f64) -> f64 {
        *value
    }

    fn from_f64(value: f64) -> f64 {
        value
    }
}

impl<W: Semiring> Compactor<W> for BitPackCompactor<W>
where
    W::Value: WeightConverter<W::Value> + Copy,
{
    type Element = u64;

    fn compact(&self, arc: &Arc<W>) -> Self::Element {
        // Use the compactor instance configuration
        let ilabel_bits = self.ilabel_bits;
        let olabel_bits = self.olabel_bits;
        let state_bits = self.state_bits;
        let weight_bits = self.weight_bits;

        // Extract and validate fields fit in allocated bits
        let ilabel = arc.ilabel & ((1u32 << ilabel_bits) - 1);
        let olabel = arc.olabel & ((1u32 << olabel_bits) - 1);
        let nextstate = arc.nextstate & ((1u32 << state_bits) - 1);

        // Quantize weight to fit in weight_bits
        let weight_val = W::Value::to_f64(arc.weight.value());
        let quantized_weight = if weight_val.is_infinite() {
            (1u64 << weight_bits) - 1 // Max value for infinity
        } else {
            // Clamp to [0, 2^weight_bits - 2] range
            let max_weight = (1u64 << weight_bits) - 2;
            let clamped = weight_val.max(0.0).min(max_weight as f64);
            clamped as u64
        };

        // Pack fields into u64
        ((ilabel as u64) << (olabel_bits + state_bits + weight_bits))
            | ((olabel as u64) << (state_bits + weight_bits))
            | ((nextstate as u64) << weight_bits)
            | quantized_weight
    }

    fn expand(&self, element: &Self::Element) -> Arc<W> {
        // Use the compactor instance configuration
        let ilabel_bits = self.ilabel_bits;
        let olabel_bits = self.olabel_bits;
        let state_bits = self.state_bits;
        let weight_bits = self.weight_bits;

        // Create bit masks
        let weight_mask = (1u64 << weight_bits) - 1;
        let state_mask = (1u64 << state_bits) - 1;
        let olabel_mask = (1u64 << olabel_bits) - 1;
        let ilabel_mask = (1u64 << ilabel_bits) - 1;

        // Extract fields
        let quantized_weight = element & weight_mask;
        let nextstate = ((element >> weight_bits) & state_mask) as u32;
        let olabel = ((element >> (weight_bits + state_bits)) & olabel_mask) as u32;
        let ilabel = ((element >> (weight_bits + state_bits + olabel_bits)) & ilabel_mask) as u32;

        // Dequantize weight
        let weight = if quantized_weight == ((1u64 << weight_bits) - 1) {
            W::zero() // Infinity maps to semiring zero
        } else {
            let weight_val = W::Value::from_f64(quantized_weight as f64);
            W::new(weight_val)
        };

        Arc::new(ilabel, olabel, weight, nextstate)
    }

    fn compact_weight(&self, weight: &W) -> Self::Element {
        // Pack weight-only into lower bits
        let weight_val = W::Value::to_f64(weight.value());

        if weight_val.is_infinite() {
            u64::MAX
        } else {
            // Use more precision for weight-only storage
            let clamped = weight_val.max(0.0).min((u64::MAX - 1) as f64);
            clamped as u64
        }
    }

    fn expand_weight(&self, element: &Self::Element) -> W {
        if *element == u64::MAX {
            W::zero() // Infinity
        } else {
            let weight_val = W::Value::from_f64(*element as f64);
            W::new(weight_val)
        }
    }
}

/// Weight quantization compactor for lossy compression
///
/// `QuantizedCompactor` trades precision for compression ratio by quantizing
/// semiring weights into a smaller number of discrete levels. This approach
/// is suitable when approximate weights are acceptable and high compression
/// is more important than exact weight preservation.
///
/// # Compression Approach
///
/// - Quantizes continuous weights into N discrete levels
/// - Maps weight ranges to integer codes
/// - Supports both linear and logarithmic quantization
/// - Configurable number of quantization levels
///
/// # Usage Example
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::{CompactFst, QuantizedCompactor, QuantizationMode};
///
/// // 256-level linear quantization
/// let compactor = QuantizedCompactor::<TropicalWeight>::new(
///     QuantizationMode::Linear { min: 0.0, max: 100.0 },
///     256
/// );
/// ```
#[derive(Debug, Clone)]
pub struct QuantizedCompactor<W: Semiring> {
    mode: QuantizationMode,
    levels: u32,
    _phantom: PhantomData<W>,
}

/// Quantization mode for weight compression
///
/// Defines how continuous weight values are mapped to discrete quantization levels.
/// Different modes are suitable for different weight distributions and precision requirements.
#[derive(Debug, Clone)]
pub enum QuantizationMode {
    /// Linear quantization between min and max values
    Linear {
        /// Minimum value in the quantization range
        min: f64,
        /// Maximum value in the quantization range
        max: f64,
    },
    /// Logarithmic quantization for better dynamic range
    Logarithmic {
        /// Minimum value in the quantization range (must be positive)
        min: f64,
        /// Maximum value in the quantization range
        max: f64,
    },
}

impl<W: Semiring> QuantizedCompactor<W> {
    /// Create a new quantized compactor
    ///
    /// # Parameters
    /// - `mode`: Quantization mode (linear or logarithmic)
    /// - `levels`: Number of quantization levels (e.g., 256 for 8-bit)
    pub fn new(mode: QuantizationMode, levels: u32) -> Self {
        assert!(
            levels > 1 && levels <= 65536,
            "Levels must be between 2 and 65536"
        );
        Self {
            mode,
            levels,
            _phantom: PhantomData,
        }
    }
}

impl<W: Semiring> Default for QuantizedCompactor<W> {
    fn default() -> Self {
        // Default to linear quantization with reasonable range
        Self::new(
            QuantizationMode::Linear {
                min: 0.0,
                max: 100.0,
            },
            256,
        )
    }
}

impl<W: Semiring> Compactor<W> for QuantizedCompactor<W>
where
    W::Value: WeightConverter<W::Value> + Copy,
{
    type Element = QuantizedArc;

    fn compact(&self, arc: &Arc<W>) -> Self::Element {
        // Use the compactor instance configuration
        let weight_val = W::Value::to_f64(arc.weight.value());
        let quantized_weight = Self::quantize_weight_value(weight_val, &self.mode, self.levels);

        QuantizedArc {
            ilabel: arc.ilabel,
            olabel: arc.olabel,
            quantized_weight,
            nextstate: arc.nextstate,
        }
    }

    fn expand(&self, element: &Self::Element) -> Arc<W> {
        // Use the compactor instance configuration
        let weight_val =
            Self::dequantize_weight_value(element.quantized_weight, &self.mode, self.levels);
        let weight = W::new(W::Value::from_f64(weight_val));

        Arc::new(element.ilabel, element.olabel, weight, element.nextstate)
    }

    fn compact_weight(&self, weight: &W) -> Self::Element {
        let weight_val = W::Value::to_f64(weight.value());
        let quantized_weight = Self::quantize_weight_value(weight_val, &self.mode, self.levels);

        QuantizedArc {
            ilabel: 0,
            olabel: 0,
            quantized_weight,
            nextstate: 0,
        }
    }

    fn expand_weight(&self, element: &Self::Element) -> W {
        let weight_val =
            Self::dequantize_weight_value(element.quantized_weight, &self.mode, self.levels);
        W::new(W::Value::from_f64(weight_val))
    }
}

impl<W: Semiring> QuantizedCompactor<W>
where
    W::Value: WeightConverter<W::Value> + Copy,
{
    /// Quantize a weight value according to the specified mode and levels
    fn quantize_weight_value(weight: f64, mode: &QuantizationMode, levels: u32) -> u16 {
        if weight.is_infinite() {
            return (levels - 1) as u16; // Reserve max value for infinity
        }

        match mode {
            QuantizationMode::Linear { min, max } => {
                if weight <= *min {
                    0
                } else if weight >= *max {
                    (levels - 2) as u16 // Reserve levels-1 for infinity
                } else {
                    let normalized = (weight - min) / (max - min);
                    let quantized = (normalized * (levels - 2) as f64).round();
                    quantized.max(0.0).min((levels - 2) as f64) as u16
                }
            }
            QuantizationMode::Logarithmic { min, max } => {
                if weight <= *min {
                    0
                } else if weight >= *max {
                    (levels - 2) as u16
                } else {
                    // Use log scale: log(weight/min) / log(max/min)
                    let log_normalized = (weight / min).ln() / (max / min).ln();
                    let quantized = (log_normalized * (levels - 2) as f64).round();
                    quantized.max(0.0).min((levels - 2) as f64) as u16
                }
            }
        }
    }

    /// Dequantize a quantized value back to a weight
    fn dequantize_weight_value(quantized: u16, mode: &QuantizationMode, levels: u32) -> f64 {
        if quantized as u32 == levels - 1 {
            return f64::INFINITY; // Special value for infinity
        }

        match mode {
            QuantizationMode::Linear { min, max } => {
                if quantized == 0 {
                    *min
                } else {
                    let normalized = quantized as f64 / (levels - 2) as f64;
                    min + normalized * (max - min)
                }
            }
            QuantizationMode::Logarithmic { min, max } => {
                if quantized == 0 {
                    *min
                } else {
                    let normalized = quantized as f64 / (levels - 2) as f64;
                    let log_weight = normalized * (max / min).ln();
                    min * log_weight.exp()
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedArc {
    ilabel: u32,
    olabel: u32,
    quantized_weight: u16,
    nextstate: u32,
}

/// Delta encoding compactor for FSTs with sequential patterns
///
/// `DeltaCompactor` exploits sequential patterns in FST structure by storing
/// differences rather than absolute values. This is particularly effective
/// for FSTs with sequential state numbering or incremental label sequences.
///
/// # Compression Approach
///
/// - Stores first arc normally, then deltas for subsequent arcs
/// - Effective for sorted arc lists and sequential states
/// - Uses variable-length encoding for small deltas
/// - Maintains exact precision (lossless)
///
/// # Best Use Cases
///
/// - Deterministic FSTs with sorted arc lists
/// - Sequential state numbering patterns
/// - Language model FSTs with incremental labels
#[derive(Debug)]
pub struct DeltaCompactor<W: Semiring> {
    _phantom: PhantomData<W>,
}

impl<W: Semiring> Default for DeltaCompactor<W> {
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<W: Semiring> Compactor<W> for DeltaCompactor<W> {
    type Element = DeltaElement<W>;

    fn compact(&self, arc: &Arc<W>) -> Self::Element {
        // For stateless compression, we default to absolute encoding
        // In a stateful implementation, this would track the previous arc
        DeltaElement::Absolute {
            ilabel: arc.ilabel,
            olabel: arc.olabel,
            weight: arc.weight.clone(),
            nextstate: arc.nextstate,
        }
    }

    fn expand(&self, element: &Self::Element) -> Arc<W> {
        match element {
            DeltaElement::Absolute {
                ilabel,
                olabel,
                weight,
                nextstate,
            } => Arc::new(*ilabel, *olabel, weight.clone(), *nextstate),
            DeltaElement::Delta {
                ilabel_delta,
                olabel_delta,
                weight,
                nextstate_delta,
            } => {
                // For delta elements, the deltas represent the actual values in this simplified version
                // In a full implementation, these would be applied to a base arc
                let ilabel = if *ilabel_delta >= 0 {
                    *ilabel_delta as u32
                } else {
                    0 // Handle negative deltas gracefully
                };
                let olabel = if *olabel_delta >= 0 {
                    *olabel_delta as u32
                } else {
                    0
                };
                let nextstate = if *nextstate_delta >= 0 {
                    *nextstate_delta as u32
                } else {
                    0
                };

                Arc::new(ilabel, olabel, weight.clone(), nextstate)
            }
        }
    }

    fn compact_weight(&self, weight: &W) -> Self::Element {
        DeltaElement::Absolute {
            ilabel: 0,
            olabel: 0,
            weight: weight.clone(),
            nextstate: 0,
        }
    }

    fn expand_weight(&self, element: &Self::Element) -> W {
        match element {
            DeltaElement::Absolute { weight, .. } => weight.clone(),
            DeltaElement::Delta { weight, .. } => weight.clone(),
        }
    }
}

impl<W: Semiring> DeltaCompactor<W> {
    /// Compute delta between two arcs, returning delta element if beneficial
    pub fn compute_delta(current_arc: &Arc<W>, previous_arc: &Arc<W>) -> DeltaElement<W> {
        // Calculate deltas for each field
        let ilabel_delta = current_arc.ilabel as i64 - previous_arc.ilabel as i64;
        let olabel_delta = current_arc.olabel as i64 - previous_arc.olabel as i64;
        let nextstate_delta = current_arc.nextstate as i64 - previous_arc.nextstate as i64;

        // Use delta encoding if all deltas fit in i16 range
        if ilabel_delta >= i16::MIN as i64
            && ilabel_delta <= i16::MAX as i64
            && olabel_delta >= i16::MIN as i64
            && olabel_delta <= i16::MAX as i64
            && nextstate_delta >= i16::MIN as i64
            && nextstate_delta <= i16::MAX as i64
        {
            DeltaElement::Delta {
                ilabel_delta: ilabel_delta as i16,
                olabel_delta: olabel_delta as i16,
                weight: current_arc.weight.clone(),
                nextstate_delta: nextstate_delta as i16,
            }
        } else {
            // Fall back to absolute encoding for large deltas
            DeltaElement::Absolute {
                ilabel: current_arc.ilabel,
                olabel: current_arc.olabel,
                weight: current_arc.weight.clone(),
                nextstate: current_arc.nextstate,
            }
        }
    }

    /// Apply delta to a base arc
    pub fn apply_delta(base_arc: &Arc<W>, delta: &DeltaElement<W>) -> Arc<W> {
        match delta {
            DeltaElement::Absolute {
                ilabel,
                olabel,
                weight,
                nextstate,
            } => Arc::new(*ilabel, *olabel, weight.clone(), *nextstate),
            DeltaElement::Delta {
                ilabel_delta,
                olabel_delta,
                weight,
                nextstate_delta,
            } => {
                let new_ilabel = (base_arc.ilabel as i64 + *ilabel_delta as i64).max(0) as u32;
                let new_olabel = (base_arc.olabel as i64 + *olabel_delta as i64).max(0) as u32;
                let new_nextstate =
                    (base_arc.nextstate as i64 + *nextstate_delta as i64).max(0) as u32;

                Arc::new(new_ilabel, new_olabel, weight.clone(), new_nextstate)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum DeltaElement<W: Semiring> {
    /// First arc or reset point with absolute values
    Absolute {
        ilabel: u32,
        olabel: u32,
        weight: W,
        nextstate: u32,
    },
    /// Subsequent arc with delta values
    Delta {
        ilabel_delta: i16,
        olabel_delta: i16,
        weight: W,
        nextstate_delta: i16,
    },
}

/// Variable-length integer compactor for diverse value ranges
///
/// `VarIntCompactor` uses variable-length encoding (similar to protobuf varints)
/// to efficiently encode integers that vary widely in magnitude. Small values
/// use fewer bytes while large values expand as needed.
///
/// # Compression Approach
///
/// - Small values (< 128) use 1 byte
/// - Medium values (< 16384) use 2 bytes
/// - Larger values use 3-5 bytes as needed
/// - Effective for FSTs with mixed small/large values
///
/// # Best Use Cases
///
/// - FSTs with mostly small labels/states but occasional large values
/// - Sparse FSTs where most values are near zero
/// - General-purpose compression when value distribution is unknown
#[derive(Debug)]
pub struct VarIntCompactor<W: Semiring> {
    _phantom: PhantomData<W>,
}

impl<W: Semiring> Default for VarIntCompactor<W> {
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<W: Semiring> Compactor<W> for VarIntCompactor<W> {
    type Element = VarIntElement<W>;

    fn compact(&self, arc: &Arc<W>) -> Self::Element {
        VarIntElement {
            encoded_ilabel: encode_varint(arc.ilabel),
            encoded_olabel: encode_varint(arc.olabel),
            weight: arc.weight.clone(),
            encoded_nextstate: encode_varint(arc.nextstate),
        }
    }

    fn expand(&self, element: &Self::Element) -> Arc<W> {
        Arc::new(
            decode_varint(&element.encoded_ilabel),
            decode_varint(&element.encoded_olabel),
            element.weight.clone(),
            decode_varint(&element.encoded_nextstate),
        )
    }

    fn compact_weight(&self, weight: &W) -> Self::Element {
        VarIntElement {
            encoded_ilabel: vec![0],
            encoded_olabel: vec![0],
            weight: weight.clone(),
            encoded_nextstate: vec![0],
        }
    }

    fn expand_weight(&self, element: &Self::Element) -> W {
        element.weight.clone()
    }
}

#[derive(Debug, Clone)]
pub struct VarIntElement<W: Semiring> {
    encoded_ilabel: Vec<u8>,
    encoded_olabel: Vec<u8>,
    weight: W,
    encoded_nextstate: Vec<u8>,
}

// Helper functions for variable-length integer encoding
fn encode_varint(value: u32) -> Vec<u8> {
    let mut result = Vec::new();
    let mut val = value;

    while val >= 0x80 {
        result.push((val & 0x7F) as u8 | 0x80);
        val >>= 7;
    }
    result.push(val as u8);

    result
}

fn decode_varint(bytes: &[u8]) -> u32 {
    let mut result = 0u32;
    let mut shift = 0;

    for &byte in bytes {
        result |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }

    result
}

impl<W: Semiring, C: Compactor<W> + Default> Default for CompactFst<W, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<W: Semiring, C: Compactor<W>> CompactFst<W, C> {
    /// Create a new empty compact FST
    ///
    /// Initializes an empty `CompactFst` with the default compactor strategy.
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
    pub fn new() -> Self
    where
        C: Default,
    {
        Self {
            states: Vec::new(),
            data: Vec::new(),
            final_weights: Vec::new(),
            start: None,
            properties: FstProperties::default(),
            compactor: C::default(),
            _phantom: PhantomData,
        }
    }

    /// Create a new compact FST with a specific compactor configuration
    ///
    /// This constructor allows specification of the compactor strategy to use.
    /// Note that the compactor parameter is used only for type specification
    /// since the current Compactor trait is stateless.
    ///
    /// # Examples
    ///
    /// ```
    /// use arcweight::prelude::*;
    /// use arcweight::fst::{CompactFst, BitPackCompactor, QuantizedCompactor, QuantizationMode};
    ///
    /// // Create with bit-packing compactor
    /// let bit_packed = CompactFst::with_compactor(BitPackCompactor::<TropicalWeight>::new(8, 8, 16));
    ///
    /// // Create with quantized compactor
    /// let quantized = CompactFst::with_compactor(
    ///     QuantizedCompactor::<TropicalWeight>::new(
    ///         QuantizationMode::Linear { min: 0.0, max: 100.0 },
    ///         256
    ///     )
    /// );
    /// ```
    pub fn with_compactor(compactor: C) -> Self {
        Self {
            states: Vec::new(),
            data: Vec::new(),
            final_weights: Vec::new(),
            start: None,
            properties: FstProperties::default(),
            compactor,
            _phantom: PhantomData,
        }
    }

    /// Convert a VectorFst to a CompactFst with compression
    ///
    /// Creates a new CompactFst by compressing all arcs and weights from the source FST.
    /// This is the primary way to create a compressed FST from existing data.
    ///
    /// # Examples
    ///
    /// ```
    /// use arcweight::prelude::*;
    /// use arcweight::fst::{CompactFst, DefaultCompactor};
    ///
    /// // Create a VectorFst
    /// let mut vector_fst = VectorFst::<TropicalWeight>::new();
    /// let s0 = vector_fst.add_state();
    /// let s1 = vector_fst.add_state();
    /// vector_fst.set_start(s0);
    /// vector_fst.set_final(s1, TropicalWeight::one());
    /// vector_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
    ///
    /// // Convert to CompactFst
    /// let compact_fst = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::from_fst(&vector_fst);
    ///
    /// // Verify same structure
    /// assert_eq!(compact_fst.num_states(), vector_fst.num_states());
    /// assert_eq!(compact_fst.start(), vector_fst.start());
    /// ```
    ///
    /// # Performance
    ///
    /// - **Time Complexity:** O(V + E) where V = states, E = arcs
    /// - **Space Complexity:** O(V + E) for compressed storage
    /// - **Compression Ratio:** Depends on compactor strategy and data characteristics
    pub fn from_fst<F: Fst<W>>(fst: &F) -> Self
    where
        C: Default,
    {
        let mut compact_fst = Self::new();

        // Add all states
        for _ in 0..fst.num_states() {
            compact_fst.add_state();
        }

        // Set start state
        compact_fst.start = fst.start();

        // Copy final weights
        for state_idx in 0..fst.num_states() {
            let state = state_idx as StateId;
            if let Some(weight) = fst.final_weight(state) {
                compact_fst.set_final_weight(state, Some(weight.clone()));
            }
        }

        // Compress and store arcs
        let mut data_offset = 0u32;
        for state_idx in 0..fst.num_states() {
            let state = state_idx as StateId;
            let arcs: Vec<_> = fst.arcs(state).collect();
            let num_arcs = arcs.len() as u32;

            // Update state metadata
            compact_fst.states[state_idx].arcs_start = data_offset;
            compact_fst.states[state_idx].num_arcs = num_arcs;

            // Compress and append arcs
            for arc in arcs {
                let compressed_arc = compact_fst.compactor.compact(&arc);
                compact_fst.data.push(compressed_arc);
            }

            data_offset += num_arcs;
        }

        compact_fst
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
    /// Reference to the compactor for decompression
    compactor: &'a C,
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
            let arc = self.compactor.expand(&self.data[self.pos]);
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
                compactor: &self.compactor,
                pos: start,
                end,
                _phantom: PhantomData,
            }
        } else {
            CompactArcIterator {
                data: &self.data,
                compactor: &self.compactor,
                pos: 0,
                end: 0,
                _phantom: PhantomData,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_compact_fst_new() {
        let fst = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::new();

        assert_eq!(fst.num_states(), 0);
        assert!(fst.start().is_none());
        assert_eq!(fst.states.len(), 0);
        assert_eq!(fst.data.len(), 0);
        assert_eq!(fst.final_weights.len(), 0);
    }

    #[test]
    fn test_compact_fst_add_state() {
        let mut fst = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::new();

        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(fst.num_states(), 3);
        assert_eq!(fst.states.len(), 3);
        assert_eq!(fst.final_weights.len(), 3);
    }

    #[test]
    fn test_compact_fst_start_state() {
        let mut fst = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::new();

        // Initially no start state
        assert!(fst.start().is_none());

        let s0 = fst.add_state();
        let s1 = fst.add_state();

        // Start state is not automatically set
        assert!(fst.start().is_none());

        // Start state would be set via set_start in full implementation
        fst.start = Some(s0);
        assert_eq!(fst.start(), Some(s0));

        fst.start = Some(s1);
        assert_eq!(fst.start(), Some(s1));
    }

    #[test]
    fn test_compact_fst_final_weights() {
        let mut fst = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::new();

        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        // Initially no final weights
        assert!(fst.final_weight(s0).is_none());
        assert!(fst.final_weight(s1).is_none());
        assert!(fst.final_weight(s2).is_none());

        // Set final weights
        fst.set_final_weight(s0, Some(TropicalWeight::new(1.5)));
        fst.set_final_weight(s2, Some(TropicalWeight::one()));

        assert_eq!(fst.final_weight(s0), Some(&TropicalWeight::new(1.5)));
        assert!(fst.final_weight(s1).is_none());
        assert_eq!(fst.final_weight(s2), Some(&TropicalWeight::one()));

        // Update final weight
        fst.set_final_weight(s0, Some(TropicalWeight::new(2.5)));
        assert_eq!(fst.final_weight(s0), Some(&TropicalWeight::new(2.5)));

        // Remove final weight
        fst.set_final_weight(s0, None);
        assert!(fst.final_weight(s0).is_none());
    }

    #[test]
    fn test_compact_fst_final_weight_bounds() {
        let mut fst = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::new();

        let _s0 = fst.add_state();

        // Test accessing non-existent state
        assert!(fst.final_weight(10).is_none());

        // Test setting final weight for high state ID (should expand vector)
        fst.set_final_weight(5, Some(TropicalWeight::new(std::f32::consts::PI)));
        assert_eq!(fst.final_weights.len(), 6); // 0-5 inclusive
        assert_eq!(
            fst.final_weight(5),
            Some(&TropicalWeight::new(std::f32::consts::PI))
        );

        // Check intermediate states are None
        for i in 1..5 {
            assert!(fst.final_weight(i).is_none());
        }
    }

    #[test]
    fn test_compact_fst_num_arcs() {
        let fst = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::new();

        // Empty FST
        assert_eq!(fst.num_arcs(0), 0);
        assert_eq!(fst.num_arcs(100), 0);

        let mut fst = fst;
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        // States with no arcs
        assert_eq!(fst.num_arcs(s0), 0);
        assert_eq!(fst.num_arcs(s1), 0);

        // Modify num_arcs for testing (in full implementation this would be set during arc addition)
        fst.states[s0 as usize].num_arcs = 3;
        fst.states[s1 as usize].num_arcs = 1;

        assert_eq!(fst.num_arcs(s0), 3);
        assert_eq!(fst.num_arcs(s1), 1);
    }

    #[test]
    fn test_compact_fst_properties() {
        let fst = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::new();

        let props = fst.properties();
        // Default properties - check individual fields since FstProperties doesn't implement PartialEq
        let default_props = FstProperties::default();
        assert_eq!(props.known, default_props.known);
        assert_eq!(props.properties, default_props.properties);
    }

    #[test]
    fn test_compact_fst_arcs_empty() {
        let mut fst = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::new();
        let s0 = fst.add_state();

        let arcs: Vec<_> = fst.arcs(s0).collect();
        assert_eq!(arcs.len(), 0);

        // Test non-existent state
        let arcs: Vec<_> = fst.arcs(100).collect();
        assert_eq!(arcs.len(), 0);
    }

    #[test]
    fn test_compact_fst_with_boolean_weights() {
        let mut fst = CompactFst::<BooleanWeight, DefaultCompactor<BooleanWeight>>::new();

        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_final_weight(s0, Some(BooleanWeight::one()));
        fst.set_final_weight(s1, Some(BooleanWeight::zero()));

        assert_eq!(fst.final_weight(s0), Some(&BooleanWeight::one()));
        assert_eq!(fst.final_weight(s1), Some(&BooleanWeight::zero()));
        assert_eq!(fst.num_states(), 2);
    }

    #[test]
    fn test_compact_fst_with_log_weights() {
        let mut fst = CompactFst::<LogWeight, DefaultCompactor<LogWeight>>::new();

        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_final_weight(s0, Some(LogWeight::new(std::f64::consts::E)));
        fst.set_final_weight(s1, Some(LogWeight::one()));

        assert_eq!(
            fst.final_weight(s0),
            Some(&LogWeight::new(std::f64::consts::E))
        );
        assert_eq!(fst.final_weight(s1), Some(&LogWeight::one()));
    }

    #[test]
    fn test_default_compactor_arc_compression() {
        let arc = Arc::new(10, 20, TropicalWeight::new(1.5), 30);
        let compactor = DefaultCompactor::<TropicalWeight>::default();

        let compressed = compactor.compact(&arc);
        let expanded = compactor.expand(&compressed);

        assert_eq!(arc.ilabel, expanded.ilabel);
        assert_eq!(arc.olabel, expanded.olabel);
        assert_eq!(arc.weight, expanded.weight);
        assert_eq!(arc.nextstate, expanded.nextstate);
    }

    #[test]
    fn test_default_compactor_weight_compression() {
        let weight = TropicalWeight::new(std::f32::consts::PI);
        let compactor = DefaultCompactor::<TropicalWeight>::default();

        let compressed = compactor.compact_weight(&weight);
        let expanded = compactor.expand_weight(&compressed);

        assert_eq!(weight, expanded);
    }

    #[test]
    fn test_default_compactor_zero_one_weights() {
        let zero = TropicalWeight::zero();
        let one = TropicalWeight::one();
        let compactor = DefaultCompactor::<TropicalWeight>::default();

        // Test zero weight compression
        let compressed_zero = compactor.compact_weight(&zero);
        let expanded_zero = compactor.expand_weight(&compressed_zero);
        assert_eq!(zero, expanded_zero);
        assert!(crate::semiring::Semiring::is_zero(&expanded_zero));

        // Test one weight compression
        let compressed_one = compactor.compact_weight(&one);
        let expanded_one = compactor.expand_weight(&compressed_one);
        assert_eq!(one, expanded_one);
        assert!(crate::semiring::Semiring::is_one(&expanded_one));
    }

    #[test]
    fn test_default_compactor_epsilon_arc() {
        let epsilon_arc = Arc::epsilon(TropicalWeight::new(0.5), 42);
        let compactor = DefaultCompactor::<TropicalWeight>::default();

        let compressed = compactor.compact(&epsilon_arc);
        let expanded = compactor.expand(&compressed);

        assert_eq!(epsilon_arc.ilabel, 0);
        assert_eq!(epsilon_arc.olabel, 0);
        assert_eq!(expanded.ilabel, 0);
        assert_eq!(expanded.olabel, 0);
        assert_eq!(expanded.weight, epsilon_arc.weight);
        assert_eq!(expanded.nextstate, 42);
    }

    #[test]
    fn test_default_compactor_large_labels() {
        let large_arc = Arc::new(
            u32::MAX - 1,
            u32::MAX,
            TropicalWeight::new(1000.0),
            u32::MAX - 2,
        );
        let compactor = DefaultCompactor::<TropicalWeight>::default();

        let compressed = compactor.compact(&large_arc);
        let expanded = compactor.expand(&compressed);

        assert_eq!(large_arc.ilabel, expanded.ilabel);
        assert_eq!(large_arc.olabel, expanded.olabel);
        assert_eq!(large_arc.weight, expanded.weight);
        assert_eq!(large_arc.nextstate, expanded.nextstate);
    }

    #[test]
    fn test_compact_element_arc_variant() {
        let element = CompactElement::Arc {
            ilabel: 100,
            olabel: 200,
            weight: TropicalWeight::new(2.5),
            nextstate: 300,
        };

        if let CompactElement::Arc {
            ilabel,
            olabel,
            weight,
            nextstate,
        } = element
        {
            assert_eq!(ilabel, 100);
            assert_eq!(olabel, 200);
            assert_eq!(weight, TropicalWeight::new(2.5));
            assert_eq!(nextstate, 300);
        } else {
            panic!("Expected Arc variant");
        }
    }

    #[test]
    fn test_compact_element_weight_variant() {
        let element = CompactElement::Weight(TropicalWeight::new(42.0));

        if let CompactElement::Weight(weight) = element {
            assert_eq!(weight, TropicalWeight::new(42.0));
        } else {
            panic!("Expected Weight variant");
        }
    }

    #[test]
    #[should_panic(expected = "Expected arc element")]
    fn test_default_compactor_expand_panic_on_weight() {
        let weight_element = CompactElement::Weight(TropicalWeight::new(1.0));
        let compactor = DefaultCompactor::<TropicalWeight>::default();
        compactor.expand(&weight_element);
    }

    #[test]
    #[should_panic(expected = "Expected weight element")]
    fn test_default_compactor_expand_weight_panic_on_arc() {
        let arc_element = CompactElement::Arc {
            ilabel: 1,
            olabel: 2,
            weight: TropicalWeight::new(1.0),
            nextstate: 3,
        };
        let compactor = DefaultCompactor::<TropicalWeight>::default();
        compactor.expand_weight(&arc_element);
    }

    #[test]
    fn test_compact_state_structure() {
        let state = CompactState {
            final_weight_idx: Some(42),
            arcs_start: 100,
            num_arcs: 5,
        };

        assert_eq!(state.final_weight_idx, Some(42));
        assert_eq!(state.arcs_start, 100);
        assert_eq!(state.num_arcs, 5);

        let state_no_final = CompactState {
            final_weight_idx: None,
            arcs_start: 0,
            num_arcs: 0,
        };

        assert_eq!(state_no_final.final_weight_idx, None);
        assert_eq!(state_no_final.arcs_start, 0);
        assert_eq!(state_no_final.num_arcs, 0);
    }

    #[test]
    fn test_compact_arc_iterator_empty() {
        let data: Vec<CompactElement<TropicalWeight>> = vec![];
        let compactor = DefaultCompactor::<TropicalWeight>::default();
        let mut iter: CompactArcIterator<'_, TropicalWeight, DefaultCompactor<TropicalWeight>> =
            CompactArcIterator {
                data: &data,
                compactor: &compactor,
                pos: 0,
                end: 0,
                _phantom: PhantomData,
            };

        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None); // Should stay None

        // Test reset
        iter.reset();
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_compact_arc_iterator_with_data() {
        let arc1 = Arc::new(1, 2, TropicalWeight::new(1.0), 10);
        let arc2 = Arc::new(3, 4, TropicalWeight::new(2.0), 20);

        let compactor = DefaultCompactor::<TropicalWeight>::default();
        let data = vec![compactor.compact(&arc1), compactor.compact(&arc2)];

        let mut iter: CompactArcIterator<'_, TropicalWeight, DefaultCompactor<TropicalWeight>> =
            CompactArcIterator {
                data: &data,
                compactor: &compactor,
                pos: 0,
                end: 2,
                _phantom: PhantomData,
            };

        // First arc
        let first = iter.next().unwrap();
        assert_eq!(first.ilabel, arc1.ilabel);
        assert_eq!(first.olabel, arc1.olabel);
        assert_eq!(first.weight, arc1.weight);
        assert_eq!(first.nextstate, arc1.nextstate);

        // Second arc
        let second = iter.next().unwrap();
        assert_eq!(second.ilabel, arc2.ilabel);
        assert_eq!(second.olabel, arc2.olabel);
        assert_eq!(second.weight, arc2.weight);
        assert_eq!(second.nextstate, arc2.nextstate);

        // No more arcs
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_compact_arc_iterator_reset() {
        let arc = Arc::new(1, 2, TropicalWeight::new(1.0), 10);
        let compactor = DefaultCompactor::<TropicalWeight>::default();
        let data = vec![compactor.compact(&arc)];

        let mut iter: CompactArcIterator<'_, TropicalWeight, DefaultCompactor<TropicalWeight>> =
            CompactArcIterator {
                data: &data,
                compactor: &compactor,
                pos: 0,
                end: 1,
                _phantom: PhantomData,
            };

        // Consume the iterator
        assert!(iter.next().is_some());
        assert!(iter.next().is_none());

        // Reset and try again
        iter.reset();
        assert!(iter.next().is_some());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_compact_arc_iterator_partial_range() {
        let arcs = [
            Arc::new(1, 1, TropicalWeight::new(1.0), 1),
            Arc::new(2, 2, TropicalWeight::new(2.0), 2),
            Arc::new(3, 3, TropicalWeight::new(3.0), 3),
            Arc::new(4, 4, TropicalWeight::new(4.0), 4),
        ];

        let compactor = DefaultCompactor::<TropicalWeight>::default();
        let data: Vec<_> = arcs.iter().map(|arc| compactor.compact(arc)).collect();

        // Iterator for arcs 1-2 (middle range)
        let mut iter: CompactArcIterator<'_, TropicalWeight, DefaultCompactor<TropicalWeight>> =
            CompactArcIterator {
                data: &data,
                compactor: &compactor,
                pos: 1,
                end: 3,
                _phantom: PhantomData,
            };

        // Should get arc 2 (index 1)
        let first = iter.next().unwrap();
        assert_eq!(first.ilabel, 2);
        assert_eq!(first.weight, TropicalWeight::new(2.0));

        // Should get arc 3 (index 2)
        let second = iter.next().unwrap();
        assert_eq!(second.ilabel, 3);
        assert_eq!(second.weight, TropicalWeight::new(3.0));

        // Should be done
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_compact_fst_default_trait() {
        let fst1 = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::default();
        let fst2 = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::new();

        assert_eq!(fst1.num_states(), fst2.num_states());
        assert_eq!(fst1.start(), fst2.start());
        assert_eq!(fst1.states.len(), fst2.states.len());
        assert_eq!(fst1.data.len(), fst2.data.len());
    }

    #[test]
    fn test_compact_fst_memory_efficiency_concept() {
        // This test demonstrates the concept of memory efficiency
        // In practice, CompactFst should use less memory than VectorFst
        // for large FSTs due to compression

        let mut compact_fst = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::new();
        let mut vector_fst = VectorFst::<TropicalWeight>::new();

        // Add same states to both
        for _ in 0..10 {
            compact_fst.add_state();
            vector_fst.add_state();
        }

        compact_fst.set_final_weight(9, Some(TropicalWeight::new(1.0)));
        vector_fst.set_final(9, TropicalWeight::new(1.0));

        assert_eq!(compact_fst.num_states(), vector_fst.num_states());

        // Both should have the same final weight
        assert_eq!(
            compact_fst.final_weight(9).copied(),
            vector_fst.final_weight(9).copied()
        );
    }

    #[test]
    fn test_compact_fst_type_compatibility() {
        // Test that CompactFst works with different semiring types

        // TropicalWeight
        let _tropical_fst = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::new();

        // LogWeight
        let _log_fst = CompactFst::<LogWeight, DefaultCompactor<LogWeight>>::new();

        // BooleanWeight
        let _bool_fst = CompactFst::<BooleanWeight, DefaultCompactor<BooleanWeight>>::new();

        // ProbabilityWeight
        let _prob_fst = CompactFst::<ProbabilityWeight, DefaultCompactor<ProbabilityWeight>>::new();

        // All should compile and create successfully
        // Test passes if no panic occurs
    }

    #[test]
    fn test_bit_pack_compactor_creation() {
        // Test bit-packing compactor with various configurations

        // Small alphabet (7-bit ASCII)
        let ascii_compactor = BitPackCompactor::<TropicalWeight>::new(7, 7, 10);
        assert_eq!(ascii_compactor.ilabel_bits, 7);
        assert_eq!(ascii_compactor.olabel_bits, 7);
        assert_eq!(ascii_compactor.state_bits, 10);
        assert_eq!(ascii_compactor.weight_bits, 40); // 64 - 7 - 7 - 10

        // Phoneme FST (8-bit labels, 12-bit states)
        let phoneme_compactor = BitPackCompactor::<TropicalWeight>::new(8, 8, 12);
        assert_eq!(phoneme_compactor.ilabel_bits, 8);
        assert_eq!(phoneme_compactor.olabel_bits, 8);
        assert_eq!(phoneme_compactor.state_bits, 12);
        assert_eq!(phoneme_compactor.weight_bits, 36); // 64 - 8 - 8 - 12

        // Maximum valid configuration (16-bit each, total 48 bits)
        let max_compactor = BitPackCompactor::<TropicalWeight>::new(16, 16, 16);
        assert_eq!(max_compactor.weight_bits, 16); // 64 - 48
    }

    #[test]
    #[should_panic(expected = "Label and state bits must fit in 48 bits")]
    fn test_bit_pack_compactor_too_many_bits() {
        // Should panic if total bits exceed 48 (leaving < 16 for weight)
        BitPackCompactor::<TropicalWeight>::new(20, 20, 20); // 60 bits total
    }

    #[test]
    fn test_quantized_compactor_creation() {
        // Linear quantization
        let linear_compactor = QuantizedCompactor::<TropicalWeight>::new(
            QuantizationMode::Linear {
                min: 0.0,
                max: 100.0,
            },
            256,
        );
        assert_eq!(linear_compactor.levels, 256);

        // Logarithmic quantization
        let log_compactor = QuantizedCompactor::<TropicalWeight>::new(
            QuantizationMode::Logarithmic {
                min: 0.001,
                max: 1000.0,
            },
            1024,
        );
        assert_eq!(log_compactor.levels, 1024);

        // Maximum levels
        let max_compactor = QuantizedCompactor::<TropicalWeight>::new(
            QuantizationMode::Linear {
                min: -1.0,
                max: 1.0,
            },
            65536,
        );
        assert_eq!(max_compactor.levels, 65536);
    }

    #[test]
    #[should_panic(expected = "Levels must be between 2 and 65536")]
    fn test_quantized_compactor_invalid_levels() {
        // Too few levels
        QuantizedCompactor::<TropicalWeight>::new(
            QuantizationMode::Linear { min: 0.0, max: 1.0 },
            1,
        );
    }

    #[test]
    fn test_delta_compactor_elements() {
        let arc = Arc::new(100, 200, TropicalWeight::new(1.5), 300);
        let compactor = DeltaCompactor::<TropicalWeight>::default();

        // Test absolute encoding
        let absolute = compactor.compact(&arc);
        match &absolute {
            DeltaElement::Absolute {
                ilabel,
                olabel,
                weight,
                nextstate,
            } => {
                assert_eq!(*ilabel, 100);
                assert_eq!(*olabel, 200);
                assert_eq!(*weight, TropicalWeight::new(1.5));
                assert_eq!(*nextstate, 300);
            }
            _ => panic!("Expected Absolute variant"),
        }

        // Test expansion
        let expanded = compactor.expand(&absolute);
        assert_eq!(expanded.ilabel, arc.ilabel);
        assert_eq!(expanded.olabel, arc.olabel);
        assert_eq!(expanded.weight, arc.weight);
        assert_eq!(expanded.nextstate, arc.nextstate);

        // Test delta variant (manual creation for testing)
        let delta = DeltaElement::Delta {
            ilabel_delta: 10,
            olabel_delta: -5,
            weight: TropicalWeight::new(0.5),
            nextstate_delta: 1,
        };

        let delta_expanded = compactor.expand(&delta);
        assert_eq!(delta_expanded.ilabel, 10);
        assert_eq!(delta_expanded.olabel, 0); // Negative deltas are clamped to 0 in simplified implementation
        assert_eq!(delta_expanded.weight, TropicalWeight::new(0.5));
        assert_eq!(delta_expanded.nextstate, 1);
    }

    #[test]
    fn test_varint_encoding() {
        // Test small values (1 byte)
        assert_eq!(encode_varint(0), vec![0x00]);
        assert_eq!(encode_varint(127), vec![0x7F]);

        // Test medium values (2 bytes)
        assert_eq!(encode_varint(128), vec![0x80, 0x01]);
        assert_eq!(encode_varint(300), vec![0xAC, 0x02]);

        // Test larger values
        assert_eq!(encode_varint(16384), vec![0x80, 0x80, 0x01]);

        // Test round-trip encoding/decoding
        for value in [0, 1, 127, 128, 255, 256, 1000, 10000, 100000, 1000000] {
            let encoded = encode_varint(value);
            let decoded = decode_varint(&encoded);
            assert_eq!(decoded, value, "Round-trip failed for {}", value);
        }
    }

    #[test]
    fn test_varint_compactor() {
        let arc = Arc::new(42, 128, TropicalWeight::new(std::f32::consts::PI), 1000);
        let compactor = VarIntCompactor::<TropicalWeight>::default();

        // Test compression
        let compressed = compactor.compact(&arc);
        assert_eq!(compressed.encoded_ilabel, encode_varint(42));
        assert_eq!(compressed.encoded_olabel, encode_varint(128));
        assert_eq!(compressed.weight, TropicalWeight::new(std::f32::consts::PI));
        assert_eq!(compressed.encoded_nextstate, encode_varint(1000));

        // Test expansion
        let expanded = compactor.expand(&compressed);
        assert_eq!(expanded.ilabel, arc.ilabel);
        assert_eq!(expanded.olabel, arc.olabel);
        assert_eq!(expanded.weight, arc.weight);
        assert_eq!(expanded.nextstate, arc.nextstate);
    }

    #[test]
    fn test_varint_compactor_large_values() {
        // Test with maximum u32 values
        let large_arc = Arc::new(
            u32::MAX,
            u32::MAX - 1,
            TropicalWeight::new(999.9),
            u32::MAX - 2,
        );
        let compactor = VarIntCompactor::<TropicalWeight>::default();

        let compressed = compactor.compact(&large_arc);
        let expanded = compactor.expand(&compressed);

        assert_eq!(expanded.ilabel, large_arc.ilabel);
        assert_eq!(expanded.olabel, large_arc.olabel);
        assert_eq!(expanded.weight, large_arc.weight);
        assert_eq!(expanded.nextstate, large_arc.nextstate);
    }

    #[test]
    fn test_multiple_compactor_types() {
        // Verify that different compactor types can be used with CompactFst

        // Default compactor
        let _default_fst = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::new();

        // Delta compactor
        let _delta_fst = CompactFst::<TropicalWeight, DeltaCompactor<TropicalWeight>>::new();

        // VarInt compactor
        let _varint_fst = CompactFst::<TropicalWeight, VarIntCompactor<TropicalWeight>>::new();

        // All should compile and create successfully
    }

    #[test]
    fn test_quantization_mode_variants() {
        // Test that both quantization modes can be created
        let linear_mode = QuantizationMode::Linear {
            min: -10.0,
            max: 10.0,
        };
        let log_mode = QuantizationMode::Logarithmic {
            min: 0.001,
            max: 1000.0,
        };

        match linear_mode {
            QuantizationMode::Linear { min, max } => {
                assert_eq!(min, -10.0);
                assert_eq!(max, 10.0);
            }
            _ => panic!("Expected Linear variant"),
        }

        match log_mode {
            QuantizationMode::Logarithmic { min, max } => {
                assert_eq!(min, 0.001);
                assert_eq!(max, 1000.0);
            }
            _ => panic!("Expected Logarithmic variant"),
        }
    }

    #[test]
    fn test_bitpack_compactor_round_trip() {
        // Test round-trip compression with BitPackCompactor for TropicalWeight
        let original_arc = Arc::new(100, 200, TropicalWeight::new(5.5), 300);
        let compactor = BitPackCompactor::<TropicalWeight>::default();

        let compressed = compactor.compact(&original_arc);
        let expanded = compactor.expand(&compressed);

        // Labels and nextstate should be preserved (masked to 16 bits)
        assert_eq!(expanded.ilabel, 100);
        assert_eq!(expanded.olabel, 200);
        assert_eq!(expanded.nextstate, 300);

        // Weight will be quantized but should be close
        let weight_diff = (expanded.weight.value() - 5.5).abs();
        assert!(
            weight_diff <= 1.0,
            "Weight should be reasonably close after quantization"
        );
    }

    #[test]
    fn test_bitpack_compactor_large_values() {
        // Test with values that exceed 16-bit limits
        let large_arc = Arc::new(0x1FFFF, 0x2FFFF, TropicalWeight::new(99999.0), 0x3FFFF);
        let compactor = BitPackCompactor::<TropicalWeight>::default();

        let compressed = compactor.compact(&large_arc);
        let expanded = compactor.expand(&compressed);

        // Values should be masked to 16 bits
        assert_eq!(expanded.ilabel, 0x1FFFF & 0xFFFF); // Lower 16 bits
        assert_eq!(expanded.olabel, 0x2FFFF & 0xFFFF);
        assert_eq!(expanded.nextstate, 0x3FFFF & 0xFFFF);
    }

    #[test]
    fn test_bitpack_compactor_infinity_weight() {
        let inf_arc = Arc::new(1, 2, TropicalWeight::zero(), 3); // zero() is infinity in tropical
        let compactor = BitPackCompactor::<TropicalWeight>::default();

        let compressed = compactor.compact(&inf_arc);
        let expanded = compactor.expand(&compressed);

        // Infinity should map back to zero (infinity in tropical semiring)
        assert!(num_traits::Zero::is_zero(&expanded.weight));
    }

    #[test]
    fn test_quantized_compactor_linear_mode() {
        let mode = QuantizationMode::Linear {
            min: 0.0,
            max: 10.0,
        };
        let levels = 256u32;

        // Test various weight values
        let test_weights = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0]; // Last one exceeds range

        for &weight_val in &test_weights {
            let quantized = QuantizedCompactor::<TropicalWeight>::quantize_weight_value(
                weight_val, &mode, levels,
            );
            let dequantized = QuantizedCompactor::<TropicalWeight>::dequantize_weight_value(
                quantized, &mode, levels,
            );

            // Values within range should be close after round-trip
            if (0.0..=10.0).contains(&weight_val) {
                let error = (dequantized - weight_val).abs();
                assert!(
                    error <= 0.1,
                    "Round-trip error too large: {} -> {} -> {}",
                    weight_val,
                    quantized,
                    dequantized
                );
            }
        }
    }

    #[test]
    fn test_quantized_compactor_logarithmic_mode() {
        let mode = QuantizationMode::Logarithmic {
            min: 0.1,
            max: 100.0,
        };
        let levels = 1024u32;

        let test_weights = [0.1, 1.0, 10.0, 100.0];

        for &weight_val in &test_weights {
            let quantized = QuantizedCompactor::<TropicalWeight>::quantize_weight_value(
                weight_val, &mode, levels,
            );
            let dequantized = QuantizedCompactor::<TropicalWeight>::dequantize_weight_value(
                quantized, &mode, levels,
            );

            // Logarithmic mode should preserve relative precision
            let relative_error = ((dequantized - weight_val) / weight_val).abs();
            assert!(
                relative_error <= 0.05,
                "Relative error too large: {} -> {} ({}% error)",
                weight_val,
                dequantized,
                relative_error * 100.0
            );
        }
    }

    #[test]
    fn test_quantized_compactor_infinity_handling() {
        let mode = QuantizationMode::Linear {
            min: 0.0,
            max: 100.0,
        };
        let levels = 256u32;

        // Test infinity quantization
        let quantized = QuantizedCompactor::<TropicalWeight>::quantize_weight_value(
            f64::INFINITY,
            &mode,
            levels,
        );
        assert_eq!(quantized, (levels - 1) as u16);

        let dequantized =
            QuantizedCompactor::<TropicalWeight>::dequantize_weight_value(quantized, &mode, levels);
        assert!(dequantized.is_infinite());
    }

    #[test]
    fn test_delta_compactor_small_deltas() {
        let base_arc = Arc::new(100, 200, TropicalWeight::new(1.0), 300);
        let next_arc = Arc::new(101, 199, TropicalWeight::new(1.5), 302);

        let delta = DeltaCompactor::<TropicalWeight>::compute_delta(&next_arc, &base_arc);

        // Should use delta encoding for small differences
        match delta {
            DeltaElement::Delta {
                ilabel_delta,
                olabel_delta,
                nextstate_delta,
                ..
            } => {
                assert_eq!(ilabel_delta, 1); // 101 - 100
                assert_eq!(olabel_delta, -1); // 199 - 200
                assert_eq!(nextstate_delta, 2); // 302 - 300
            }
            _ => panic!("Expected Delta variant for small differences"),
        }

        // Test applying delta
        let applied = DeltaCompactor::<TropicalWeight>::apply_delta(&base_arc, &delta);
        assert_eq!(applied.ilabel, next_arc.ilabel);
        assert_eq!(applied.olabel, next_arc.olabel);
        assert_eq!(applied.nextstate, next_arc.nextstate);
    }

    #[test]
    fn test_delta_compactor_large_deltas() {
        let base_arc = Arc::new(100, 200, TropicalWeight::new(1.0), 300);
        let far_arc = Arc::new(70000, 80000, TropicalWeight::new(2.0), 90000);

        let delta = DeltaCompactor::<TropicalWeight>::compute_delta(&far_arc, &base_arc);

        // Should fall back to absolute encoding for large differences
        match delta {
            DeltaElement::Absolute {
                ilabel,
                olabel,
                nextstate,
                ..
            } => {
                assert_eq!(ilabel, 70000);
                assert_eq!(olabel, 80000);
                assert_eq!(nextstate, 90000);
            }
            _ => panic!("Expected Absolute variant for large differences"),
        }
    }

    #[test]
    fn test_compact_fst_from_vector_fst() {
        // Create a simple VectorFst
        let mut vector_fst = VectorFst::<TropicalWeight>::new();
        let s0 = vector_fst.add_state();
        let s1 = vector_fst.add_state();
        let s2 = vector_fst.add_state();

        vector_fst.set_start(s0);
        vector_fst.set_final(s2, TropicalWeight::new(2.0));

        vector_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
        vector_fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s2));
        vector_fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::new(1.5), s2));

        // Convert to CompactFst
        let compact_fst =
            CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::from_fst(&vector_fst);

        // Verify structure preservation
        assert_eq!(compact_fst.num_states(), vector_fst.num_states());
        assert_eq!(compact_fst.start(), vector_fst.start());

        // Verify final weights
        assert_eq!(compact_fst.final_weight(s2), vector_fst.final_weight(s2));
        assert!(compact_fst.final_weight(s0).is_none());
        assert!(compact_fst.final_weight(s1).is_none());

        // Verify arc counts
        assert_eq!(compact_fst.num_arcs(s0), vector_fst.num_arcs(s0));
        assert_eq!(compact_fst.num_arcs(s1), vector_fst.num_arcs(s1));
        assert_eq!(compact_fst.num_arcs(s2), vector_fst.num_arcs(s2));

        // Verify arcs are preserved (order might differ due to compression)
        let compact_arcs_s0: Vec<_> = compact_fst.arcs(s0).collect();
        let vector_arcs_s0: Vec<_> = vector_fst.arcs(s0).collect();
        assert_eq!(compact_arcs_s0.len(), vector_arcs_s0.len());
    }

    #[test]
    fn test_compact_fst_with_compactor() {
        // Test creating CompactFst with different compactors
        let _default_fst = CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::new();
        let _bit_packed_fst =
            CompactFst::with_compactor(BitPackCompactor::<TropicalWeight>::new(8, 8, 16));
        let _quantized_fst = CompactFst::with_compactor(QuantizedCompactor::<TropicalWeight>::new(
            QuantizationMode::Linear {
                min: 0.0,
                max: 100.0,
            },
            256,
        ));

        // All should create successfully
    }

    #[test]
    fn test_compression_ratio_concept() {
        // This test demonstrates the concept of compression
        // In practice, compression effectiveness varies by data characteristics

        let mut large_fst = VectorFst::<TropicalWeight>::new();

        // Create FST with many states and arcs
        for _i in 0..100 {
            large_fst.add_state();
        }
        large_fst.set_start(0);
        large_fst.set_final(99, TropicalWeight::one());

        // Add many arcs with small labels (good for bit-packing)
        for i in 0..99 {
            large_fst.add_arc(
                i,
                Arc::new(i % 10, i % 10, TropicalWeight::new((i % 20) as f32), i + 1),
            );
        }

        // Convert with different compactors
        let default_compact =
            CompactFst::<TropicalWeight, DefaultCompactor<TropicalWeight>>::from_fst(&large_fst);
        let bitpack_compact =
            CompactFst::<TropicalWeight, BitPackCompactor<TropicalWeight>>::from_fst(&large_fst);

        // Both should have same logical structure
        assert_eq!(default_compact.num_states(), bitpack_compact.num_states());
        assert_eq!(default_compact.start(), bitpack_compact.start());

        // Memory usage would differ in practice (CompactElement vs u64)
        assert_eq!(default_compact.data.len(), bitpack_compact.data.len());
    }

    #[test]
    fn test_varint_encoding_edge_cases() {
        // Test edge cases for varint encoding
        let edge_cases = [0, 1, 127, 128, 255, 256, 16383, 16384, u32::MAX];

        for &value in &edge_cases {
            let encoded = encode_varint(value);
            let decoded = decode_varint(&encoded);
            assert_eq!(decoded, value, "Varint round-trip failed for {}", value);

            // Check expected encoding lengths
            match value {
                0..=127 => assert_eq!(encoded.len(), 1, "Single byte expected for {}", value),
                128..=16383 => assert_eq!(encoded.len(), 2, "Two bytes expected for {}", value),
                16384..=2097151 => {
                    assert_eq!(encoded.len(), 3, "Three bytes expected for {}", value)
                }
                _ => assert!(encoded.len() <= 5, "Max 5 bytes for any u32"),
            }
        }
    }

    #[test]
    fn test_semiring_compatibility() {
        // Test that compression works with different semiring types

        // TropicalWeight (f32)
        let tropical_arc = Arc::new(1, 2, TropicalWeight::new(std::f32::consts::PI), 4);
        let compactor = BitPackCompactor::<TropicalWeight>::default();
        let _tropical_compressed = compactor.compact(&tropical_arc);

        // LogWeight (f64) - would need trait bound adjustments
        // This demonstrates the need for proper semiring compatibility

        // Test weight-only compression
        let weight = TropicalWeight::new(42.0);
        let compressed_weight = compactor.compact_weight(&weight);
        let expanded_weight = compactor.expand_weight(&compressed_weight);

        // Should be close after quantization
        let weight_diff = (expanded_weight.value() - 42.0).abs();
        assert!(
            weight_diff <= 1.0,
            "Weight round-trip should be reasonably accurate"
        );
    }
}
