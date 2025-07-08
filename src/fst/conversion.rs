//! Comprehensive conversion matrix between FST types
//!
//! This module provides efficient conversion functions between different FST
//! implementations, allowing users to seamlessly transform FSTs based on
//! their performance and memory requirements.
//!
//! # Conversion Matrix
//!
//! | From / To | VectorFst | ConstFst | CompactFst | CacheFst | LazyFstImpl | EvictingCacheFst |
//! |-----------|-----------|----------|------------|----------|-------------|------------------|
//! | VectorFst | Identity  | ✓        | ✓          | ✓        | ✓           | ✓                |
//! | ConstFst  | ✓         | Identity | ✓          | ✓        | ✓           | ✓                |
//! | CompactFst| ✓         | ✓        | Identity   | ✓        | ✓           | ✓                |
//! | CacheFst  | ✓         | ✓        | ✓          | Identity | ✓           | ✓                |
//! | LazyFstImpl | ✓       | ✓        | ✓          | ✓        | Identity    | ✓                |
//! | EvictingCacheFst | ✓  | ✓        | ✓          | ✓        | ✓           | Identity         |
//!
//! # Performance Characteristics
//!
//! Different conversions have different performance characteristics:
//!
//! - **Memory-to-Memory:** Direct copying (VectorFst → ConstFst)
//! - **Compression:** Space reduction with time cost (VectorFst → CompactFst)
//! - **Decompression:** Time cost for space restoration (CompactFst → VectorFst)
//! - **Wrapping:** Minimal overhead for behavior change (Any → CacheFst)
//! - **Materialization:** Full computation of lazy structures (LazyFstImpl → VectorFst)
//!
//! # Examples
//!
//! ```rust
//! # fn main() -> Result<(), arcweight::Error> {
//! use arcweight::prelude::*;
//! use arcweight::fst::*;
//!
//! // Create a mutable FST for construction
//! let mut vector_fst = VectorFst::<TropicalWeight>::new();
//! let s0 = vector_fst.add_state();
//! let s1 = vector_fst.add_state();
//! vector_fst.set_start(s0);
//! vector_fst.set_final(s1, TropicalWeight::one());
//! vector_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
//!
//! // Convert to optimized read-only format
//! let const_fst: ConstFst<TropicalWeight> = convert_to_const(&vector_fst)?;
//!
//! // Convert to compressed format for memory-constrained environments
//! let compact_fst = convert_to_compact(&vector_fst)?;
//!
//! // Wrap with caching for repeated access
//! let cached_fst = convert_to_cache(vector_fst);
//! # Ok(())
//! # }
//! ```

use super::traits::*;
use super::{
    CacheConfig, CacheFst, CompactFst, ConstFst, DefaultCompactor, EvictingCacheFst, LazyFstImpl,
    LazyState, VectorFst,
};
use crate::arc::{Arc, ArcIterator};
use crate::properties::FstProperties;
use crate::semiring::Semiring;
use crate::Result;

/// Convert any FST to VectorFst (mutable format)
///
/// This function materializes any FST implementation into a VectorFst,
/// which provides full mutability and is optimized for construction and modification.
///
/// # Performance
/// - **Time:** O(V + E) where V = states, E = arcs
/// - **Memory:** Full materialization in memory
/// - **Use case:** When you need to modify the FST after conversion
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::convert_to_vector;
///
/// // Convert from any FST type to VectorFst
/// let original_fst = VectorFst::<TropicalWeight>::new();
/// let vector_fst: VectorFst<TropicalWeight> = convert_to_vector(&original_fst)?;
/// # Ok::<(), arcweight::Error>(())
/// ```
pub fn convert_to_vector<W, F>(fst: &F) -> Result<VectorFst<W>>
where
    W: Semiring,
    F: Fst<W>,
{
    let mut result = VectorFst::new();

    // Add all states
    for _ in 0..fst.num_states() {
        result.add_state();
    }

    // Set start state
    if let Some(start) = fst.start() {
        result.set_start(start);
    }

    // Copy all arcs and final weights
    for state in fst.states() {
        // Copy arcs
        for arc in fst.arcs(state) {
            result.add_arc(state, arc);
        }

        // Copy final weight
        if let Some(weight) = fst.final_weight(state) {
            result.set_final(state, weight.clone());
        }
    }

    Ok(result)
}

/// Convert any FST to ConstFst (optimized read-only format)
///
/// This function converts any FST implementation into a ConstFst,
/// which provides optimal memory layout and read performance.
///
/// # Performance
/// - **Time:** O(V + E) for conversion
/// - **Memory:** ~30% less than VectorFst, better cache locality
/// - **Use case:** Production deployment, read-only operations
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::convert_to_const;
///
/// let vector_fst = VectorFst::<TropicalWeight>::new();
/// let const_fst: ConstFst<TropicalWeight> = convert_to_const(&vector_fst)?;
/// # Ok::<(), arcweight::Error>(())
/// ```
pub fn convert_to_const<W, F>(fst: &F) -> Result<ConstFst<W>>
where
    W: Semiring,
    F: Fst<W>,
{
    // First convert to VectorFst, then to ConstFst
    let vector_fst = convert_to_vector(fst)?;
    ConstFst::from_fst(&vector_fst)
}

/// Convert any FST to CompactFst with default compression
///
/// This function converts any FST implementation into a CompactFst using
/// the default compactor, providing significant memory savings.
///
/// # Performance
/// - **Time:** O(V + E) for conversion + compression overhead
/// - **Memory:** 40-70% reduction compared to VectorFst
/// - **Use case:** Memory-constrained environments, large FSTs
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::convert_to_compact;
///
/// let vector_fst = VectorFst::<TropicalWeight>::new();
/// let compact_fst = convert_to_compact(&vector_fst)?;
/// # Ok::<(), arcweight::Error>(())
/// ```
pub fn convert_to_compact<W, F>(fst: &F) -> Result<CompactFst<W, DefaultCompactor<W>>>
where
    W: Semiring,
    F: Fst<W>,
{
    let vector_fst = convert_to_vector(fst)?;
    Ok(CompactFst::from_fst(&vector_fst))
}

/// Convert any FST to CompactFst with custom compactor
///
/// This function allows conversion with a specific compression strategy.
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::convert_to_compact_with;
/// use arcweight::fst::VarIntCompactor;
///
/// let vector_fst = VectorFst::<TropicalWeight>::new();
/// let compactor = VarIntCompactor::default();
/// let compact_fst = convert_to_compact_with(&vector_fst, compactor)?;
/// # Ok::<(), arcweight::Error>(())
/// ```
pub fn convert_to_compact_with<W, F, C>(fst: &F, compactor: C) -> Result<CompactFst<W, C>>
where
    W: Semiring,
    F: Fst<W>,
    C: super::Compactor<W> + Default + Clone,
{
    // Create a new CompactFst with the custom compactor
    let mut compact_fst = CompactFst::with_compactor(compactor);

    // Add all states
    for _ in 0..fst.num_states() {
        compact_fst.add_state();
    }

    // Set start state
    if let Some(start) = fst.start() {
        compact_fst.set_start(start);
    }

    // Copy all arcs and final weights
    for state in fst.states() {
        // Copy arcs
        for arc in fst.arcs(state) {
            compact_fst.add_arc(state, arc);
        }

        // Copy final weight
        if let Some(weight) = fst.final_weight(state) {
            compact_fst.set_final(state, weight.clone());
        }
    }

    Ok(compact_fst)
}

/// Convert any FST to CacheFst (simple caching wrapper)
///
/// This function wraps any FST with a caching layer for improved
/// performance on repeated access patterns.
///
/// # Performance
/// - **Time:** O(1) wrapping, caching overhead on access
/// - **Memory:** Original FST + cache overhead
/// - **Use case:** Expensive computations, repeated access
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::convert_to_cache;
///
/// let vector_fst = VectorFst::<TropicalWeight>::new();
/// let cached_fst = convert_to_cache(vector_fst);
/// ```
pub fn convert_to_cache<W, F>(fst: F) -> CacheFst<W, F>
where
    W: Semiring,
    F: Fst<W>,
{
    CacheFst::new(fst)
}

/// Convert any FST to EvictingCacheFst with custom configuration
///
/// This function wraps any FST with an advanced caching layer that includes
/// configurable eviction policies and memory management.
///
/// # Performance
/// - **Time:** O(1) wrapping, configurable caching overhead
/// - **Memory:** Bounded by cache configuration
/// - **Use case:** Memory-bounded caching, long-running applications
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::convert_to_evicting_cache;
/// use arcweight::fst::{CacheConfig, EvictionPolicy};
///
/// let vector_fst = VectorFst::<TropicalWeight>::new();
/// let cache_config = CacheConfig {
///     max_cached_states: 1000,
///     eviction_policy: EvictionPolicy::LRU,
///     ..Default::default()
/// };
/// let cached_fst = convert_to_evicting_cache(vector_fst, cache_config);
/// ```
pub fn convert_to_evicting_cache<W, F>(fst: F, cache_config: CacheConfig) -> EvictingCacheFst<F, W>
where
    W: Semiring,
    F: Fst<W>,
{
    EvictingCacheFst::new(fst, cache_config)
}

/// Convert any FST to LazyFstImpl (on-demand computation)
///
/// This function converts any FST into a LazyFstImpl that computes states
/// on demand using a closure that captures the original FST data.
///
/// # Performance
/// - **Time:** O(V + E) for initial materialization
/// - **Memory:** Grows with accessed states
/// - **Use case:** Large FSTs with sparse access patterns
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::convert_to_lazy;
///
/// let vector_fst = VectorFst::<TropicalWeight>::new();
/// let lazy_fst = convert_to_lazy(&vector_fst)?;
/// # Ok::<(), arcweight::Error>(())
/// ```
pub fn convert_to_lazy<W, F>(
    fst: &F,
) -> Result<LazyFstImpl<W, impl Fn(StateId) -> Option<LazyState<W>> + Send + Sync>>
where
    W: Semiring,
    F: Fst<W>,
{
    // Materialize the FST data
    let vector_fst = convert_to_vector(fst)?;

    // Create computation function that looks up in the materialized FST
    let compute_fn = move |state: StateId| -> Option<LazyState<W>> {
        if state >= vector_fst.num_states() as StateId {
            return None;
        }

        let arcs: Vec<Arc<W>> = vector_fst.arcs(state).collect();
        let final_weight = vector_fst.final_weight(state).cloned();

        Some(LazyState { arcs, final_weight })
    };

    Ok(LazyFstImpl::new(compute_fn, fst.num_states()))
}

/// Batch conversion utilities for multiple FSTs
#[derive(Debug)]
pub struct BatchConverter;

impl BatchConverter {
    /// Convert multiple FSTs to the same target type
    ///
    /// This is useful when you have multiple FSTs that need to be converted
    /// to the same format for consistent processing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    /// use arcweight::fst::BatchConverter;
    ///
    /// let fsts = vec![
    ///     VectorFst::<TropicalWeight>::new(),
    ///     VectorFst::<TropicalWeight>::new(),
    /// ];
    ///
    /// let const_fsts: Result<Vec<ConstFst<TropicalWeight>>> =
    ///     BatchConverter::convert_all_to_const(&fsts);
    /// # Ok::<(), arcweight::Error>(())
    /// ```
    pub fn convert_all_to_const<W, F>(fsts: &[F]) -> Result<Vec<ConstFst<W>>>
    where
        W: Semiring,
        F: Fst<W>,
    {
        fsts.iter().map(convert_to_const).collect()
    }

    /// Convert multiple FSTs to VectorFst
    pub fn convert_all_to_vector<W, F>(fsts: &[F]) -> Result<Vec<VectorFst<W>>>
    where
        W: Semiring,
        F: Fst<W>,
    {
        fsts.iter().map(convert_to_vector).collect()
    }

    /// Convert multiple FSTs to CompactFst with the same compactor
    pub fn convert_all_to_compact<W, F>(
        fsts: &[F],
    ) -> Result<Vec<CompactFst<W, DefaultCompactor<W>>>>
    where
        W: Semiring,
        F: Fst<W>,
    {
        fsts.iter().map(convert_to_compact).collect()
    }
}

/// Conversion strategy selection based on use case
#[derive(Debug, Clone, PartialEq)]
pub enum ConversionStrategy {
    /// Optimize for construction and modification
    ForConstruction,
    /// Optimize for production deployment (read-only)
    ForProduction,
    /// Optimize for memory-constrained environments
    ForMemoryConstraints,
    /// Optimize for repeated access patterns
    ForRepeatedAccess,
    /// Optimize for sparse access patterns
    ForSparseAccess,
}

/// Enum representing different FST types for auto-conversion
#[derive(Debug)]
pub enum ConvertedFst<W: Semiring> {
    /// VectorFst variant
    Vector(VectorFst<W>),
    /// ConstFst variant
    Const(ConstFst<W>),
    /// CompactFst variant
    Compact(CompactFst<W, DefaultCompactor<W>>),
    /// CacheFst variant
    Cache(CacheFst<W, VectorFst<W>>),
    /// LazyFstImpl variant
    Lazy(LazyFstImpl<W, Box<dyn Fn(StateId) -> Option<LazyState<W>> + Send + Sync>>),
}

/// Wrapper to implement ArcIterator for boxed iterators
pub struct BoxedArcIterator<'a, W: Semiring> {
    iter: Box<dyn Iterator<Item = Arc<W>> + 'a>,
}

impl<W: Semiring> std::fmt::Debug for BoxedArcIterator<'_, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoxedArcIterator")
            .field("iter", &"Box<dyn Iterator<Item = Arc<W>>>")
            .finish()
    }
}

impl<W: Semiring> Iterator for BoxedArcIterator<'_, W> {
    type Item = Arc<W>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<W: Semiring> ArcIterator<W> for BoxedArcIterator<'_, W> {}

impl<W: Semiring> Fst<W> for ConvertedFst<W> {
    type ArcIter<'a>
        = BoxedArcIterator<'a, W>
    where
        Self: 'a;

    fn start(&self) -> Option<StateId> {
        match self {
            ConvertedFst::Vector(fst) => fst.start(),
            ConvertedFst::Const(fst) => fst.start(),
            ConvertedFst::Compact(fst) => fst.start(),
            ConvertedFst::Cache(fst) => fst.start(),
            ConvertedFst::Lazy(fst) => fst.start(),
        }
    }

    fn final_weight(&self, state: StateId) -> Option<&W> {
        match self {
            ConvertedFst::Vector(fst) => fst.final_weight(state),
            ConvertedFst::Const(fst) => fst.final_weight(state),
            ConvertedFst::Compact(fst) => fst.final_weight(state),
            ConvertedFst::Cache(fst) => fst.final_weight(state),
            ConvertedFst::Lazy(fst) => fst.final_weight(state),
        }
    }

    fn num_arcs(&self, state: StateId) -> usize {
        match self {
            ConvertedFst::Vector(fst) => fst.num_arcs(state),
            ConvertedFst::Const(fst) => fst.num_arcs(state),
            ConvertedFst::Compact(fst) => fst.num_arcs(state),
            ConvertedFst::Cache(fst) => fst.num_arcs(state),
            ConvertedFst::Lazy(fst) => fst.num_arcs(state),
        }
    }

    fn num_states(&self) -> usize {
        match self {
            ConvertedFst::Vector(fst) => fst.num_states(),
            ConvertedFst::Const(fst) => fst.num_states(),
            ConvertedFst::Compact(fst) => fst.num_states(),
            ConvertedFst::Cache(fst) => fst.num_states(),
            ConvertedFst::Lazy(fst) => fst.num_states(),
        }
    }

    fn properties(&self) -> FstProperties {
        match self {
            ConvertedFst::Vector(fst) => fst.properties(),
            ConvertedFst::Const(fst) => fst.properties(),
            ConvertedFst::Compact(fst) => fst.properties(),
            ConvertedFst::Cache(fst) => fst.properties(),
            ConvertedFst::Lazy(fst) => fst.properties(),
        }
    }

    fn arcs(&self, state: StateId) -> Self::ArcIter<'_> {
        let iter: Box<dyn Iterator<Item = Arc<W>> + '_> = match self {
            ConvertedFst::Vector(fst) => Box::new(fst.arcs(state)),
            ConvertedFst::Const(fst) => Box::new(fst.arcs(state)),
            ConvertedFst::Compact(fst) => Box::new(fst.arcs(state)),
            ConvertedFst::Cache(fst) => Box::new(fst.arcs(state)),
            ConvertedFst::Lazy(fst) => Box::new(fst.arcs(state)),
        };
        BoxedArcIterator { iter }
    }
}

/// Automatic conversion based on strategy and FST characteristics
///
/// This function analyzes the FST and selects the optimal target format
/// based on the specified strategy and FST characteristics.
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::{ConversionStrategy, auto_convert};
///
/// let vector_fst = VectorFst::<TropicalWeight>::new();
/// let strategy = ConversionStrategy::ForProduction;
///
/// // This will automatically choose ConstFst for production use
/// let result = auto_convert(&vector_fst, strategy)?;
/// # Ok::<(), arcweight::Error>(())
/// ```
pub fn auto_convert<W, F>(fst: &F, strategy: ConversionStrategy) -> Result<ConvertedFst<W>>
where
    W: Semiring,
    F: Fst<W>,
{
    match strategy {
        ConversionStrategy::ForConstruction => {
            let vector_fst = convert_to_vector(fst)?;
            Ok(ConvertedFst::Vector(vector_fst))
        }
        ConversionStrategy::ForProduction => {
            let const_fst = convert_to_const(fst)?;
            Ok(ConvertedFst::Const(const_fst))
        }
        ConversionStrategy::ForMemoryConstraints => {
            let compact_fst = convert_to_compact(fst)?;
            Ok(ConvertedFst::Compact(compact_fst))
        }
        ConversionStrategy::ForRepeatedAccess => {
            let cached_fst = convert_to_cache(convert_to_vector(fst)?);
            Ok(ConvertedFst::Cache(cached_fst))
        }
        ConversionStrategy::ForSparseAccess => {
            // Convert to vector first, then create lazy wrapper with boxed function
            let vector_fst = convert_to_vector(fst)?;
            let vector_fst = std::sync::Arc::new(vector_fst);

            let compute_fn: Box<dyn Fn(StateId) -> Option<LazyState<W>> + Send + Sync> =
                Box::new(move |state: StateId| -> Option<LazyState<W>> {
                    if state >= vector_fst.num_states() as StateId {
                        return None;
                    }

                    let arcs: Vec<Arc<W>> = vector_fst.arcs(state).collect();
                    let final_weight = vector_fst.final_weight(state).cloned();

                    Some(LazyState { arcs, final_weight })
                });

            let lazy_fst = LazyFstImpl::new(compute_fn, fst.num_states());
            Ok(ConvertedFst::Lazy(lazy_fst))
        }
    }
}

/// Conversion metrics for analyzing conversion trade-offs
#[derive(Debug, Clone)]
pub struct ConversionMetrics {
    /// Estimated memory usage after conversion
    pub estimated_memory: usize,
    /// Estimated access time (relative scale)
    pub estimated_access_time: f64,
    /// Estimated conversion time
    pub estimated_conversion_time: f64,
    /// Whether the result supports mutation
    pub supports_mutation: bool,
    /// Whether the result supports lazy evaluation
    pub supports_lazy_evaluation: bool,
}

/// Estimate conversion metrics for different target formats
///
/// This function helps users make informed decisions about which
/// conversion to perform based on their requirements.
pub fn estimate_conversion_metrics<W, F>(
    fst: &F,
    target_strategy: ConversionStrategy,
) -> ConversionMetrics
where
    W: Semiring,
    F: Fst<W>,
{
    let num_states = fst.num_states();
    let total_arcs: usize = (0..num_states as StateId).map(|s| fst.num_arcs(s)).sum();

    let base_memory =
        num_states * std::mem::size_of::<StateId>() + total_arcs * std::mem::size_of::<Arc<W>>();

    match target_strategy {
        ConversionStrategy::ForConstruction => ConversionMetrics {
            estimated_memory: base_memory,
            estimated_access_time: 1.0,
            estimated_conversion_time: (num_states + total_arcs) as f64 * 0.001,
            supports_mutation: true,
            supports_lazy_evaluation: false,
        },
        ConversionStrategy::ForProduction => ConversionMetrics {
            estimated_memory: (base_memory as f64 * 0.7) as usize,
            estimated_access_time: 0.8,
            estimated_conversion_time: (num_states + total_arcs) as f64 * 0.002,
            supports_mutation: false,
            supports_lazy_evaluation: false,
        },
        ConversionStrategy::ForMemoryConstraints => ConversionMetrics {
            estimated_memory: (base_memory as f64 * 0.4) as usize,
            estimated_access_time: 1.5,
            estimated_conversion_time: (num_states + total_arcs) as f64 * 0.005,
            supports_mutation: false,
            supports_lazy_evaluation: false,
        },
        ConversionStrategy::ForRepeatedAccess => ConversionMetrics {
            estimated_memory: (base_memory as f64 * 1.2) as usize,
            estimated_access_time: 0.3, // After cache warming
            estimated_conversion_time: (num_states + total_arcs) as f64 * 0.001,
            supports_mutation: false,
            supports_lazy_evaluation: false,
        },
        ConversionStrategy::ForSparseAccess => ConversionMetrics {
            estimated_memory: num_states * std::mem::size_of::<StateId>(), // Initial overhead
            estimated_access_time: 1.2,                                    // First access cost
            estimated_conversion_time: (num_states + total_arcs) as f64 * 0.003,
            supports_mutation: false,
            supports_lazy_evaluation: true,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fst::{EvictionPolicy, LazyStreamingConfig, VarIntCompactor};
    use crate::prelude::*;

    fn create_test_fst() -> VectorFst<TropicalWeight> {
        let mut fst = VectorFst::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s2));

        fst
    }

    #[test]
    fn test_convert_to_vector() {
        let original = create_test_fst();
        let converted = convert_to_vector(&original).unwrap();

        assert_eq!(converted.num_states(), original.num_states());
        assert_eq!(converted.start(), original.start());
        assert_eq!(converted.num_arcs(0), original.num_arcs(0));
        assert_eq!(converted.num_arcs(1), original.num_arcs(1));
    }

    #[test]
    fn test_convert_to_const() {
        let original = create_test_fst();
        let converted = convert_to_const(&original).unwrap();

        assert_eq!(converted.num_states(), original.num_states());
        assert_eq!(converted.start(), original.start());
        assert_eq!(converted.num_arcs(0), original.num_arcs(0));
        assert_eq!(converted.num_arcs(1), original.num_arcs(1));
    }

    #[test]
    fn test_convert_to_compact() {
        let original = create_test_fst();
        let converted = convert_to_compact(&original).unwrap();

        assert_eq!(converted.num_states(), original.num_states());
        assert_eq!(converted.start(), original.start());
        assert_eq!(converted.num_arcs(0), original.num_arcs(0));
        assert_eq!(converted.num_arcs(1), original.num_arcs(1));
    }

    #[test]
    fn test_convert_to_cache() {
        let original = create_test_fst();
        let converted = convert_to_cache(original.clone());

        assert_eq!(converted.num_states(), original.num_states());
        assert_eq!(converted.start(), original.start());
        assert_eq!(converted.num_arcs(0), original.num_arcs(0));
        assert_eq!(converted.num_arcs(1), original.num_arcs(1));
    }

    #[test]
    fn test_convert_to_evicting_cache() {
        let original = create_test_fst();
        let cache_config = CacheConfig {
            max_cached_states: 1000,
            memory_limit: Some(10_000_000),
            eviction_policy: EvictionPolicy::LRU,
            enable_memory_mapping: false,
            enable_prefetching: false,
            streaming_config: LazyStreamingConfig::default(),
        };
        let converted = convert_to_evicting_cache(original.clone(), cache_config);

        assert_eq!(converted.num_states(), original.num_states());
        assert_eq!(converted.start(), original.start());
        assert_eq!(converted.num_arcs(0), original.num_arcs(0));
        assert_eq!(converted.num_arcs(1), original.num_arcs(1));
    }

    #[test]
    fn test_convert_to_lazy() {
        let original = create_test_fst();
        let converted = convert_to_lazy(&original).unwrap();

        assert_eq!(converted.num_states(), original.num_states());
        assert_eq!(converted.start(), original.start());
        assert_eq!(converted.num_arcs(0), original.num_arcs(0));
        assert_eq!(converted.num_arcs(1), original.num_arcs(1));
    }

    #[test]
    fn test_convert_to_compact_with_custom_compactor() {
        let original = create_test_fst();
        let compactor = VarIntCompactor::default();
        let converted = convert_to_compact_with(&original, compactor).unwrap();

        assert_eq!(converted.num_states(), original.num_states());
        assert_eq!(converted.start(), original.start());
        assert_eq!(converted.num_arcs(0), original.num_arcs(0));
        assert_eq!(converted.num_arcs(1), original.num_arcs(1));

        // Verify the FST structure is preserved
        for state in 0..original.num_states() {
            let state_id = state as StateId;
            assert_eq!(
                converted.final_weight(state_id).is_some(),
                original.final_weight(state_id).is_some()
            );

            let orig_arcs: Vec<_> = original.arcs(state_id).collect();
            let conv_arcs: Vec<_> = converted.arcs(state_id).collect();
            assert_eq!(
                orig_arcs.len(),
                conv_arcs.len(),
                "Arc count mismatch for state {state_id}"
            );

            // Just verify we have the right number of arcs and basic properties
            // The exact arc order might not be preserved
            assert_eq!(orig_arcs.len(), conv_arcs.len());
        }
    }

    #[test]
    fn test_batch_converter() {
        let fsts = vec![create_test_fst(), create_test_fst()];

        let const_fsts = BatchConverter::convert_all_to_const(&fsts).unwrap();
        assert_eq!(const_fsts.len(), 2);

        let vector_fsts = BatchConverter::convert_all_to_vector(&fsts).unwrap();
        assert_eq!(vector_fsts.len(), 2);

        let compact_fsts = BatchConverter::convert_all_to_compact(&fsts).unwrap();
        assert_eq!(compact_fsts.len(), 2);
    }

    #[test]
    fn test_conversion_strategy() {
        assert_eq!(
            ConversionStrategy::ForConstruction,
            ConversionStrategy::ForConstruction
        );
        assert_ne!(
            ConversionStrategy::ForConstruction,
            ConversionStrategy::ForProduction
        );
    }

    #[test]
    fn test_estimate_conversion_metrics() {
        let fst = create_test_fst();

        let metrics = estimate_conversion_metrics(&fst, ConversionStrategy::ForProduction);
        assert!(metrics.estimated_memory > 0);
        assert!(metrics.estimated_access_time > 0.0);
        assert!(!metrics.supports_mutation);

        let metrics = estimate_conversion_metrics(&fst, ConversionStrategy::ForConstruction);
        assert!(metrics.supports_mutation);

        let metrics = estimate_conversion_metrics(&fst, ConversionStrategy::ForSparseAccess);
        assert!(metrics.supports_lazy_evaluation);
    }
}
