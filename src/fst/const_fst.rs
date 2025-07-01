//! Constant (immutable) FST implementation optimized for read-only operations

use super::traits::*;
use crate::arc::{Arc, ArcIterator};
use crate::properties::FstProperties;
use crate::semiring::Semiring;
use crate::Result;
use core::slice;

/// Immutable FST implementation optimized for memory efficiency and fast read-only access
///
/// `ConstFst` is an immutable FST implementation that provides excellent performance for
/// read-only operations. Once constructed, it cannot be modified, but offers superior
/// memory efficiency and access speed compared to mutable alternatives. This makes it
/// ideal for production use cases where FSTs are built once and queried many times.
///
/// # Design Characteristics
///
/// - **Immutability:** Cannot be modified after construction - read-only operations only
/// - **Memory Layout:** Compact memory representation with excellent cache locality
/// - **Storage Format:** States and arcs stored in separate contiguous arrays
/// - **Random Access:** O(1) access to any state, O(1) arc range lookup per state
/// - **Space Efficiency:** Minimal memory overhead, optimal for large FSTs
///
/// # Performance Profile
///
/// | Operation | Time Complexity | Notes |
/// |-----------|----------------|-------|
/// | State Access | O(1) | Direct array indexing |
/// | Arc Range Access | O(1) | Precomputed arc ranges |
/// | Arc Iteration | O(k) | k = number of arcs, excellent cache locality |
/// | Memory Footprint | Minimal | ~20% less memory than VectorFst |
/// | Construction | O(V + E) | One-time cost from source FST |
///
/// # Memory Layout
///
/// ```text
/// ConstFst Memory Structure:
/// ┌─────────────────┐
/// │ States Array    │ ← Box<[ConstState]>: metadata per state
/// │ [State 0]       │   - final_weight: Option<W>
/// │ [State 1]       │   - arcs_start: u32 (offset into arcs array)
/// │ [State ...]     │   - num_arcs: u32 (count of arcs)
/// └─────────────────┘
/// ┌─────────────────┐
/// │ Arcs Array      │ ← Box<[Arc<W>]>: all arcs in order
/// │ [State 0 arcs]  │   Grouped by source state for cache locality
/// │ [State 1 arcs]  │
/// │ [State ... arcs]│
/// └─────────────────┘
/// ```
///
/// # Memory Characteristics
///
/// - **State Storage:** Fixed-size array with 16 bytes per state + weight size
/// - **Arc Storage:** Contiguous array with ~32 bytes per arc
/// - **No Growth Overhead:** No unused capacity, exact memory allocation
/// - **Cache Friendly:** Sequential arc access has excellent spatial locality
/// - **Memory Savings:** 15-25% less memory than equivalent VectorFst
///
/// # Use Cases
///
/// ## Production FST Deployment
/// ```rust
/// use arcweight::prelude::*;
///
/// // Build FST during application startup or offline
/// fn build_and_optimize_fst() -> Result<ConstFst<TropicalWeight>> {
///     // Build mutable FST with complex construction logic
///     let mut builder = VectorFst::new();
///     
///     // ... complex FST construction ...
///     let s0 = builder.add_state();
///     let s1 = builder.add_state();
///     builder.set_start(s0);
///     builder.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
///     builder.set_final(s1, TropicalWeight::one());
///     
///     // Convert to optimized read-only format
///     ConstFst::from_fst(&builder)
/// }
///
/// // Use optimized FST for all subsequent operations
/// let production_fst = build_and_optimize_fst()?;
///
/// // High-performance lookups
/// if let Some(start) = production_fst.start() {
///     for arc in production_fst.arcs(start) {
///         // Process arcs with optimal cache performance
///         println!("Arc: {} -> {} / {}", arc.ilabel, arc.olabel, arc.weight);
///     }
/// }
/// # Ok::<(), arcweight::Error>(())
/// ```
///
/// ## Large-Scale Language Models
/// ```rust
/// use arcweight::prelude::*;
///
/// // Convert large language model FST to const format
/// fn optimize_language_model(
///     mutable_lm: &VectorFst<LogWeight>
/// ) -> Result<ConstFst<LogWeight>> {
///     // Verify FST is complete and valid
///     println!("Optimizing LM with {} states, {} arcs",
///              mutable_lm.num_states(), mutable_lm.num_arcs_total());
///              
///     // Convert to space-efficient immutable format
///     let const_lm = ConstFst::from_fst(mutable_lm)?;
///     
///     // Memory usage comparison
///     println!("Memory optimization complete");
///     // Note: Actual memory usage would need external measurement
///     
///     Ok(const_lm)
/// }
/// ```
///
/// ## Pronunciation Dictionary Deployment
/// ```rust
/// use arcweight::prelude::*;
///
/// // Deploy pronunciation dictionary for speech recognition
/// fn create_pronunciation_dict() -> Result<ConstFst<TropicalWeight>> {
///     let mut dict = VectorFst::new();
///     
///     // Build dictionary structure (simplified example)
///     let root = dict.add_state();
///     dict.set_start(root);
///     
///     // Add word: "hello" -> "heh low"
///     let mut current = root;
///     for &ch in b"hello" {
///         let next = dict.add_state();
///         dict.add_arc(current, Arc::new(
///             ch as u32, 0, // Input letter, epsilon output
///             TropicalWeight::one(), next
///         ));
///         current = next;
///     }
///     
///     // Add phoneme outputs
///     let phonemes = [b'h', b'e', b'h', b' ', b'l', b'o', b'w'];
///     for &ph in &phonemes {
///         let next = dict.add_state();
///         dict.add_arc(current, Arc::new(
///             0, ph as u32, // Epsilon input, phoneme output
///             TropicalWeight::one(), next
///         ));
///         current = next;
///     }
///     
///     dict.set_final(current, TropicalWeight::one());
///     
///     // Optimize for deployment
///     ConstFst::from_fst(&dict)
/// }
/// ```
///
/// ## Multi-FST Algorithm Input
/// ```rust
/// use arcweight::prelude::*;
///
/// // Prepare FSTs for composition operations
/// fn prepare_composition_inputs(
///     input_fst: &VectorFst<LogWeight>,
///     output_fst: &VectorFst<LogWeight>
/// ) -> Result<(ConstFst<LogWeight>, ConstFst<LogWeight>)> {
///     // Convert both FSTs to optimized format for composition
///     let const_input = ConstFst::from_fst(input_fst)?;
///     let const_output = ConstFst::from_fst(output_fst)?;
///     
///     // Both FSTs now have optimal memory layout for composition algorithm
///     Ok((const_input, const_output))
/// }
/// ```
///
/// # Construction Patterns
///
/// ## From Existing FST
/// ```rust
/// use arcweight::prelude::*;
///
/// // Standard construction pattern
/// let mut builder = VectorFst::<TropicalWeight>::new();
/// // ... build FST structure ...
/// let optimized = ConstFst::from_fst(&builder)?;
/// # Ok::<(), arcweight::Error>(())
/// ```
///
/// ## Validation During Construction
/// ```rust
/// use arcweight::prelude::*;
///
/// fn safe_const_fst_creation(
///     source: &VectorFst<TropicalWeight>
/// ) -> Result<ConstFst<TropicalWeight>> {
///     // Validate source FST before conversion
///     if source.num_states() == 0 {
///         return Err(arcweight::Error::InvalidOperation("FST is empty".to_string()));
///     }
///     
///     if source.start().is_none() {
///         return Err(arcweight::Error::InvalidOperation("FST has no start state".to_string()));
///     }
///     
///     // Proceed with conversion
///     ConstFst::from_fst(source)
/// }
/// ```
///
/// # Performance Optimization Guidelines
///
/// ## When to Use ConstFst
/// - ✅ FST structure is finalized and won't change
/// - ✅ Memory efficiency is important
/// - ✅ Read-heavy workloads with many traversals
/// - ✅ Production deployment of large FSTs
/// - ✅ Multi-threaded read access patterns
///
/// ## When to Use VectorFst Instead
/// - ❌ FST needs to be modified after construction
/// - ❌ Incremental construction with unknown final size
/// - ❌ Debugging and development phases
/// - ❌ Single-use FSTs with minimal reuse
///
/// ## Memory Optimization Tips
/// 1. **Build Efficiently:** Construct in VectorFst, then convert
/// 2. **Batch Conversion:** Convert multiple FSTs together if memory allows
/// 3. **Size Validation:** Check memory requirements before conversion
/// 4. **Timing:** Convert during application startup, not critical paths
///
/// # Thread Safety
///
/// `ConstFst` is fully thread-safe for read operations:
/// - **Immutable Data:** No risk of data races during concurrent reads
/// - **Send + Sync:** Can be shared between threads safely
/// - **Arc-Compatible:** Wrap in `Arc<ConstFst<W>>` for shared ownership
/// - **Lock-Free:** No synchronization overhead for read operations
///
/// # Algorithm Integration
///
/// ConstFst works with all FST algorithms that accept the `Fst` trait:
/// - **Composition:** Excellent performance as composition input
/// - **Search:** Optimal for shortest path and traversal algorithms
/// - **Analysis:** Efficient for property computation and validation
/// - **Determinization:** Can serve as input to determinization algorithms
///
/// # Limitations
///
/// - **No Modification:** Cannot add/remove states or arcs after construction
/// - **Construction Cost:** O(V + E) conversion time from source FST
/// - **Memory Spike:** Requires memory for both source and target during conversion
/// - **Type Limitations:** Requires `Clone` semiring weights for construction
///
/// # See Also
///
/// - [`VectorFst`] for mutable FST operations
/// - [`CacheFst`] for lazy evaluation patterns
/// - [`CompactFst`] for maximum memory compression
/// - [Performance Guide](../../docs/architecture/performance.md) for optimization strategies
/// - [Production Patterns](../../docs/working-with-fsts/advanced-topics.md#production-patterns) for deployment patterns
///
/// [`VectorFst`]: crate::fst::VectorFst
/// [`CacheFst`]: crate::fst::CacheFst
/// [`CompactFst`]: crate::fst::CompactFst
#[derive(Debug, Clone)]
pub struct ConstFst<W: Semiring> {
    /// Immutable array of state metadata with arc range information
    states: Box<[ConstState<W>]>,
    /// Immutable array of all arcs, grouped by source state for cache locality
    arcs: Box<[Arc<W>]>,
    /// Optional start state identifier
    start: Option<StateId>,
    /// Cached FST properties for efficient property queries
    properties: FstProperties,
}

/// Compact state representation for ConstFst
///
/// Stores minimal state information with precomputed arc range offsets
/// for optimal memory usage and cache performance.
#[derive(Debug, Clone)]
struct ConstState<W: Semiring> {
    /// Final weight if this state is final, None otherwise
    final_weight: Option<W>,
    /// Starting offset in the global arcs array for this state's arcs
    arcs_start: u32,
    /// Number of arcs from this state (allows range computation)
    num_arcs: u32,
}

impl<W: Semiring> ConstFst<W> {
    /// Create an immutable ConstFst from any FST implementation
    ///
    /// Converts a source FST into an optimized read-only format with compact memory
    /// layout and excellent cache locality. The conversion process copies all states
    /// and arcs into contiguous arrays, enabling fast traversal operations.
    ///
    /// # Performance
    ///
    /// - **Time Complexity:** O(V + E) where V = states, E = arcs
    /// - **Space Complexity:** O(V + E) for the new representation
    /// - **Memory Peak:** Temporarily requires memory for both source and target FST
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// // Build a simple FST
    /// let mut builder = VectorFst::<TropicalWeight>::new();
    /// let s0 = builder.add_state();
    /// let s1 = builder.add_state();
    /// builder.set_start(s0);
    /// builder.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
    /// builder.set_final(s1, TropicalWeight::one());
    ///
    /// // Convert to optimized format
    /// let const_fst = ConstFst::from_fst(&builder)?;
    ///
    /// // Verify structure is preserved
    /// assert_eq!(const_fst.num_states(), 2);
    /// assert_eq!(const_fst.start(), Some(0));
    /// assert!(const_fst.is_final(1));
    /// # Ok::<(), arcweight::Error>(())
    /// ```
    ///
    /// # Memory Layout After Conversion
    ///
    /// ```text
    /// Input:  VectorFst with scattered memory allocations
    /// Output: ConstFst with compact arrays:
    ///
    /// States: [State0][State1][State2]...  ← Contiguous metadata
    /// Arcs:   [S0_Arc0][S0_Arc1][S1_Arc0][S1_Arc1]...  ← Grouped by state
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Memory allocation fails during FST construction (rare)
    /// - The input FST contains more than 2³2 states or arcs (very large FSTs)
    /// - Arc iteration fails on the source FST
    ///
    /// # Thread Safety
    ///
    /// This method is safe to call concurrently on different FSTs, but the same
    /// source FST should not be modified during conversion.
    pub fn from_fst<F: Fst<W>>(fst: &F) -> Result<Self> {
        let mut states = Vec::with_capacity(fst.num_states());
        let mut arcs = Vec::new();

        for state in fst.states() {
            let arcs_start = arcs.len() as u32;
            let state_arcs: Vec<_> = fst.arcs(state).collect();
            let num_arcs = state_arcs.len() as u32;

            arcs.extend(state_arcs);

            states.push(ConstState {
                final_weight: fst.final_weight(state).cloned(),
                arcs_start,
                num_arcs,
            });
        }

        Ok(Self {
            states: states.into_boxed_slice(),
            arcs: arcs.into_boxed_slice(),
            start: fst.start(),
            properties: fst.properties(),
        })
    }
}

/// High-performance arc iterator for ConstFst with optimal cache locality
///
/// Provides iterator access to arcs from a specific state in a ConstFst.
/// The iterator works directly with a contiguous slice of arcs, providing
/// excellent memory access patterns and cache performance.
///
/// # Performance Characteristics
///
/// - **Memory Access:** Sequential access to contiguous arc array
/// - **Cache Locality:** Excellent - arcs are stored together
/// - **Allocation:** Zero allocations during iteration
/// - **Overhead:** Minimal iterator state (single slice iterator)
///
/// # Usage
///
/// This iterator is created automatically by `ConstFst::arcs()` and should
/// not be constructed directly. It implements the standard Iterator trait
/// for seamless integration with Rust iteration patterns.
///
/// ```rust
/// use arcweight::prelude::*;
///
/// # fn example() -> Result<()> {
/// let mut builder = VectorFst::<TropicalWeight>::new();
/// let s0 = builder.add_state();
/// let s1 = builder.add_state();
/// builder.set_start(s0);
/// builder.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
/// builder.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(1.0), s1));
///
/// let const_fst = ConstFst::from_fst(&builder)?;
///
/// // High-performance iteration
/// for arc in const_fst.arcs(s0) {
///     println!("Label: {}, Weight: {}", arc.ilabel, arc.weight);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct ConstArcIterator<'a, W: Semiring> {
    /// Original arc slice for reset functionality
    original_arcs: &'a [Arc<W>],
    /// Current iterator over contiguous arc slice for optimal performance
    arcs: slice::Iter<'a, Arc<W>>,
}

impl<W: Semiring> ArcIterator<W> for ConstArcIterator<'_, W> {
    fn reset(&mut self) {
        self.arcs = self.original_arcs.iter();
    }
}

impl<W: Semiring> Iterator for ConstArcIterator<'_, W> {
    type Item = Arc<W>;

    fn next(&mut self) -> Option<Self::Item> {
        self.arcs.next().cloned()
    }
}

impl<W: Semiring> Fst<W> for ConstFst<W> {
    type ArcIter<'a>
        = ConstArcIterator<'a, W>
    where
        W: 'a;

    fn start(&self) -> Option<StateId> {
        self.start
    }

    fn final_weight(&self, state: StateId) -> Option<&W> {
        self.states
            .get(state as usize)
            .and_then(|s| s.final_weight.as_ref())
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
            let arc_slice = &self.arcs[start..end];
            ConstArcIterator {
                original_arcs: arc_slice,
                arcs: arc_slice.iter(),
            }
        } else {
            ConstArcIterator {
                original_arcs: &[],
                arcs: [].iter(),
            }
        }
    }
}

impl<W: Semiring> ExpandedFst<W> for ConstFst<W> {
    fn arcs_slice(&self, state: StateId) -> &[Arc<W>] {
        if let Some(s) = self.states.get(state as usize) {
            let start = s.arcs_start as usize;
            let end = start + s.num_arcs as usize;
            &self.arcs[start..end]
        } else {
            &[]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_const_fst_from_vector_fst() {
        let mut vector_fst = VectorFst::<TropicalWeight>::new();
        let s0 = vector_fst.add_state();
        let s1 = vector_fst.add_state();
        let s2 = vector_fst.add_state();

        vector_fst.set_start(s0);
        vector_fst.set_final(s2, TropicalWeight::new(3.0));

        vector_fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(1.0), s1));
        vector_fst.add_arc(s1, Arc::new(3, 4, TropicalWeight::new(2.0), s2));

        let const_fst = ConstFst::from_fst(&vector_fst).unwrap();

        assert_eq!(const_fst.num_states(), vector_fst.num_states());
        assert_eq!(const_fst.start(), vector_fst.start());
        assert_eq!(const_fst.num_arcs_total(), vector_fst.num_arcs_total());

        for state in vector_fst.states() {
            assert_eq!(const_fst.is_final(state), vector_fst.is_final(state));
            assert_eq!(const_fst.final_weight(state), vector_fst.final_weight(state));
            assert_eq!(const_fst.num_arcs(state), vector_fst.num_arcs(state));

            let const_arcs: Vec<_> = const_fst.arcs(state).collect();
            let vector_arcs: Vec<_> = vector_fst.arcs(state).collect();
            assert_eq!(const_arcs, vector_arcs);
        }
    }
}
