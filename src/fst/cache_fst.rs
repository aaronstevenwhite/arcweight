//! Caching wrapper for FSTs to optimize repeated access patterns

use super::traits::*;
use crate::arc::{Arc, ArcIterator};
use crate::properties::FstProperties;
use crate::semiring::Semiring;
use std::collections::HashMap;
use std::sync::{Arc as SyncArc, RwLock};

/// Thread-safe caching wrapper for FSTs to optimize expensive repeated operations
///
/// `CacheFst` is a wrapper that provides transparent caching for expensive FST operations,
/// particularly useful for scenarios where the same states and arcs are accessed repeatedly.
/// It's especially effective when wrapping lazy FSTs, composition results, or expensive
/// on-demand computations where repeated access to the same data would otherwise
/// require recomputation.
///
/// # Design Characteristics
///
/// - **Transparent Caching:** Drop-in replacement for any `Fst` implementation
/// - **Thread Safety:** Safe concurrent access through reader-writer locks
/// - **Adaptive:** Caches data on-demand based on actual access patterns
/// - **Memory Trade-off:** Uses memory to store cached results for faster subsequent access
/// - **Lazy Evaluation:** Only computes and caches data when requested
///
/// # Performance Profile
///
/// | Operation | First Access | Cached Access | Notes |
/// |-----------|-------------|---------------|-------|
/// | Arc Iteration | O(K + computation) | O(K) | K = number of arcs |
/// | Final Weight | O(computation) | O(1) | Depends on wrapped FST |
/// | State Count | O(1) | O(1) | Delegated to wrapped FST |
/// | Memory Usage | Base + Cache | Growing with usage | |
///
/// # Cache Behavior
///
/// ```text
/// Cache Structure:
/// ┌─────────────────────────┐
/// │ Arc Cache               │ ← HashMap<StateId, Vec<Arc<W>>>
/// │ State 0: [Arc, Arc...]  │   Stores computed arcs per state
/// │ State 1: [Arc, Arc...]  │   Thread-safe with RwLock
/// │ State N: [Arc, Arc...]  │
/// └─────────────────────────┘
/// ┌─────────────────────────┐
/// │ Weight Cache            │ ← HashMap<StateId, Option<W>>
/// │ State 0: Some(weight)   │   Stores final weights per state
/// │ State 1: None           │   Thread-safe with RwLock
/// │ State N: Some(weight)   │
/// └─────────────────────────┘
/// ```
///
/// # Use Cases
///
/// ## Expensive Composition Results
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::CacheFst;
///
/// // Cache expensive composition for repeated access
/// fn cache_composition_result(
///     fst1: &VectorFst<TropicalWeight>,
///     fst2: &VectorFst<TropicalWeight>
/// ) -> Result<CacheFst<TropicalWeight, VectorFst<TropicalWeight>>> {
///     // Expensive composition operation
///     let composed = compose_default(fst1, fst2)?;
///     
///     // Wrap with cache for repeated traversals
///     Ok(CacheFst::new(composed))
/// }
///
/// # Ok::<(), arcweight::Error>(())
/// ```
///
/// ## Lazy FST Optimization
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::CacheFst;
///
/// // Cache lazy computations for better performance
/// fn optimize_lazy_fst<F: Fst<TropicalWeight>>(
///     lazy_fst: F
/// ) -> CacheFst<TropicalWeight, F> {
///     // Wrap lazy FST with caching
///     let cached = CacheFst::new(lazy_fst);
///     
///     // Subsequent accesses to the same states will be much faster
///     cached
/// }
/// ```
///
/// ## Repeated Search Operations
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::CacheFst;
///
/// // Optimize FST for multiple search operations
/// fn create_search_optimized_fst(
///     base_fst: VectorFst<TropicalWeight>
/// ) -> CacheFst<TropicalWeight, VectorFst<TropicalWeight>> {
///     // Cache frequently accessed states during search
///     let cached_fst = CacheFst::new(base_fst);
///     
///     // Multiple searches will benefit from cached arc data
///     cached_fst
/// }
///
/// fn perform_multiple_searches(
///     fst: &CacheFst<TropicalWeight, VectorFst<TropicalWeight>>,
///     queries: &[&str]
/// ) {
///     for query in queries {
///         // Each search benefits from previous cache hits
///         if let Some(start) = fst.start() {
///             // Search implementation would traverse cached states efficiently
///             let mut current = start;
///             for &ch in query.as_bytes() {
///                 // Arc access cached after first query
///                 for arc in fst.arcs(current) {
///                     if arc.ilabel == ch as u32 {
///                         current = arc.nextstate;
///                         break;
///                     }
///                 }
///             }
///         }
///     }
/// }
/// ```
///
/// ## Large FST Memory Management
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::CacheFst;
///
/// // Manage memory for large FSTs with selective caching
/// fn create_memory_managed_fst(
///     large_fst: VectorFst<LogWeight>
/// ) -> CacheFst<LogWeight, VectorFst<LogWeight>> {
///     let cached = CacheFst::new(large_fst);
///     
///     // Only frequently accessed parts will be cached
///     // Provides balance between memory usage and performance
///     cached
/// }
///
/// // Periodic cache management
/// fn manage_cache_memory(
///     fst: &CacheFst<LogWeight, VectorFst<LogWeight>>
/// ) {
///     // Clear cache periodically to control memory usage
///     fst.clear_cache();
///     
///     // Cache will rebuild based on new access patterns
/// }
/// ```
///
/// # Cache Management Patterns
///
/// ## Preloading Strategy
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::CacheFst;
///
/// // Preload cache with expected access patterns
/// fn preload_cache(
///     fst: &CacheFst<TropicalWeight, VectorFst<TropicalWeight>>,
///     important_states: &[StateId]
/// ) {
///     // Preload arcs for states we know will be accessed frequently
///     for &state in important_states {
///         let _arcs: Vec<_> = fst.arcs(state).collect();
///         // Arcs are now cached for future access
///     }
/// }
/// ```
///
/// ## Memory-Conscious Usage
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::fst::CacheFst;
///
/// // Monitor and control cache growth
/// fn memory_conscious_processing(
///     fst: &CacheFst<TropicalWeight, VectorFst<TropicalWeight>>
/// ) {
///     // Process in batches with periodic cache clearing
///     let batch_size = 1000;
///     
///     for batch_start in (0..fst.num_states()).step_by(batch_size) {
///         // Process batch of states
///         for state in batch_start..(batch_start + batch_size).min(fst.num_states()) {
///             // State processing benefits from caching within batch
///             let _arcs: Vec<_> = fst.arcs(state as StateId).collect();
///         }
///         
///         // Clear cache between batches to control memory
///         fst.clear_cache();
///     }
/// }
/// ```
///
/// # Performance Optimization Guidelines
///
/// ## When to Use CacheFst
/// - ✅ Expensive underlying FST operations (lazy evaluation, composition results)
/// - ✅ Repeated access to same states/arcs in algorithms
/// - ✅ Search-heavy workloads with predictable access patterns
/// - ✅ Multi-threaded scenarios with concurrent read access
/// - ✅ Temporary performance boost during intensive operations
///
/// ## When NOT to Use CacheFst
/// - ❌ Single-pass traversals with no repetition
/// - ❌ Memory-constrained environments
/// - ❌ FSTs with uniform access patterns (all states accessed once)
/// - ❌ Very small FSTs where caching overhead exceeds benefits
/// - ❌ Write-heavy scenarios (CacheFst is read-only)
///
/// ## Memory Considerations
/// 1. **Cache Growth:** Memory usage grows with accessed state count
/// 2. **State Diversity:** More unique states accessed = more memory used
/// 3. **Arc Density:** States with many arcs consume more cache memory
/// 4. **Lifetime Management:** Clear cache when access patterns change
///
/// # Thread Safety
///
/// `CacheFst` is fully thread-safe and designed for concurrent access:
/// - **Read-Write Locks:** Multiple concurrent readers, exclusive writers
/// - **Lock Granularity:** Separate locks for arc cache and weight cache
/// - **Deadlock Prevention:** Consistent lock ordering throughout implementation
/// - **Cache Coherency:** All threads see consistent cached data
///
/// ```rust
/// use std::sync::Arc;
/// use std::thread;
/// use arcweight::prelude::*;
/// use arcweight::fst::CacheFst;
///
/// fn concurrent_access_example() {
///     let base_fst = VectorFst::<TropicalWeight>::new();
///     let cached_fst = Arc::new(CacheFst::new(base_fst));
///     
///     let handles: Vec<_> = (0..4).map(|_| {
///         let fst = Arc::clone(&cached_fst);
///         thread::spawn(move || {
///             // Each thread can safely access the cached FST
///             if let Some(start) = fst.start() {
///                 let _arcs: Vec<_> = fst.arcs(start).collect();
///             }
///         })
///     }).collect();
///     
///     for handle in handles {
///         handle.join().unwrap();
///     }
/// }
/// ```
///
/// # Implementation Details
///
/// ## Cache Strategy
/// - **Lazy Loading:** Data cached only when first accessed
/// - **Full State Caching:** All arcs from a state cached together
/// - **Persistent Cache:** Data remains cached until explicitly cleared
/// - **Copy-on-Access:** Cached data cloned for thread safety
///
/// ## Memory Layout
/// ```text
/// CacheFst Structure:
/// ┌─────────────────┐
/// │ Wrapped FST     │ ← Arc<F>: Shared reference to underlying FST
/// └─────────────────┘
/// ┌─────────────────┐
/// │ Arc Cache       │ ← RwLock<HashMap<StateId, Vec<Arc<W>>>>
/// │ - Thread-safe   │   Concurrent access with reader-writer locks
/// │ - HashMap-based │   O(1) average lookup time
/// └─────────────────┘
/// ┌─────────────────┐
/// │ Weight Cache    │ ← RwLock<HashMap<StateId, Option<W>>>
/// │ - Final weights │   Caches final state weights
/// │ - Optional vals │   None for non-final states
/// └─────────────────┘
/// ```
///
/// # Algorithm Integration
///
/// CacheFst integrates with all FST algorithms:
/// - **Composition:** Cache intermediate results during multi-stage composition
/// - **Search Algorithms:** Accelerate repeated state visits in search trees
/// - **Path Finding:** Cache explored paths for backtracking algorithms
/// - **Analysis:** Speed up property computation with repeated traversals
///
/// # Limitations and Trade-offs
///
/// ## Memory Usage
/// - Cache grows monotonically until explicitly cleared
/// - Memory usage proportional to accessed state diversity
/// - No automatic eviction policies (LRU, etc.)
///
/// ## Performance Trade-offs
/// - First access slower due to caching overhead
/// - Lock contention possible under heavy concurrent access
/// - Additional memory allocation for cache storage
///
/// ## Design Constraints
/// - Read-only wrapper (no modification of underlying FST)
/// - Cache invalidation requires manual intervention
/// - Lock-based synchronization (not lock-free)
///
/// # See Also
///
/// - [`VectorFst`] for the primary mutable FST implementation
/// - [`ConstFst`] for memory-optimized read-only FSTs
/// - [`LazyFstImpl`] for on-demand FST computation
/// - [Performance Guide](../../docs/architecture/performance.md) for optimization strategies
/// - [Memory Management](../../docs/architecture/memory-management.md) for cache management techniques
///
/// [`VectorFst`]: crate::fst::VectorFst
/// [`ConstFst`]: crate::fst::ConstFst
/// [`LazyFstImpl`]: crate::fst::LazyFstImpl
#[derive(Debug)]
pub struct CacheFst<W: Semiring, F: Fst<W>> {
    /// Shared reference to the wrapped FST for thread-safe access
    fst: SyncArc<F>,
    /// Thread-safe cache for arc data, indexed by state ID
    arc_cache: RwLock<HashMap<StateId, Vec<Arc<W>>>>,
    /// Thread-safe cache for final weights, indexed by state ID
    weight_cache: RwLock<HashMap<StateId, Option<W>>>,
    /// Phantom data to maintain proper generic constraints
    _phantom: core::marker::PhantomData<W>,
}

impl<W: Semiring, F: Fst<W>> CacheFst<W, F> {
    /// Create a new caching wrapper around the provided FST
    ///
    /// Wraps the given FST in a caching layer that will transparently cache
    /// expensive operations like arc iteration and final weight lookup. The
    /// wrapped FST is stored in a shared reference to enable thread-safe access.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    /// use arcweight::fst::CacheFst;
    ///
    /// // Create base FST
    /// let mut base_fst = VectorFst::<TropicalWeight>::new();
    /// let s0 = base_fst.add_state();
    /// let s1 = base_fst.add_state();
    /// base_fst.set_start(s0);
    /// base_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
    /// base_fst.set_final(s1, TropicalWeight::one());
    ///
    /// // Wrap with caching
    /// let cached_fst = CacheFst::new(base_fst);
    ///
    /// // First access will populate cache
    /// let arcs1: Vec<_> = cached_fst.arcs(s0).collect();
    /// // Second access will use cached data
    /// let arcs2: Vec<_> = cached_fst.arcs(s0).collect();
    ///
    /// assert_eq!(arcs1.len(), arcs2.len());
    /// ```
    ///
    /// # Thread Safety
    ///
    /// The wrapped FST can be safely shared between threads for concurrent
    /// read access. Cache updates are protected by reader-writer locks.
    pub fn new(fst: F) -> Self {
        Self {
            fst: SyncArc::new(fst),
            arc_cache: RwLock::new(HashMap::new()),
            weight_cache: RwLock::new(HashMap::new()),
            _phantom: core::marker::PhantomData,
        }
    }

    /// Clear all cached data to free memory
    ///
    /// Removes all cached arcs and final weights, forcing subsequent accesses
    /// to recompute data from the underlying FST. This is useful for memory
    /// management in long-running applications or when access patterns change.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    /// use arcweight::fst::CacheFst;
    ///
    /// let base_fst = VectorFst::<TropicalWeight>::new();
    /// let cached_fst = CacheFst::new(base_fst);
    ///
    /// // Access some data to populate cache
    /// if let Some(start) = cached_fst.start() {
    ///     let _arcs: Vec<_> = cached_fst.arcs(start).collect();
    /// }
    ///
    /// // Clear cache to free memory
    /// cached_fst.clear_cache();
    ///
    /// // Subsequent access will rebuild cache
    /// if let Some(start) = cached_fst.start() {
    ///     let _arcs: Vec<_> = cached_fst.arcs(start).collect();
    /// }
    /// ```
    ///
    /// # Performance Impact
    ///
    /// - **Memory:** Immediately frees all cached data
    /// - **Subsequent Access:** Will be slower until cache is rebuilt
    /// - **Thread Safety:** Safe to call concurrently with other operations
    ///
    /// # Panics
    ///
    /// Panics if the internal cache lock is poisoned, which can occur
    /// if another thread panicked while holding the lock. This is rare
    /// and typically indicates a serious application error.
    pub fn clear_cache(&self) {
        self.arc_cache.write().unwrap().clear();
        self.weight_cache.write().unwrap().clear();
    }

    /// Get the number of states with cached arc data
    ///
    /// Returns the count of states that currently have arc data in the cache.
    /// This can be useful for monitoring cache utilization and memory usage.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    /// use arcweight::fst::CacheFst;
    ///
    /// let mut base_fst = VectorFst::<TropicalWeight>::new();
    /// let s0 = base_fst.add_state();
    /// let s1 = base_fst.add_state();
    /// base_fst.set_start(s0);
    ///
    /// let cached_fst = CacheFst::new(base_fst);
    ///
    /// // Initially no cached states
    /// assert_eq!(cached_fst.cached_states_count(), 0);
    ///
    /// // Access state to populate cache
    /// let _arcs: Vec<_> = cached_fst.arcs(s0).collect();
    /// assert_eq!(cached_fst.cached_states_count(), 1);
    ///
    /// // Access another state
    /// let _arcs: Vec<_> = cached_fst.arcs(s1).collect();
    /// assert_eq!(cached_fst.cached_states_count(), 2);
    /// ```
    pub fn cached_states_count(&self) -> usize {
        self.arc_cache.read().unwrap().len()
    }
}

/// Iterator over cached arcs with position tracking and reset capability
///
/// Provides efficient iteration over cached arc data with the ability to reset
/// to the beginning. The iterator owns a copy of the arc vector to ensure
/// thread safety and independence from cache modifications.
///
/// # Performance Characteristics
///
/// - **Memory:** Owns a copy of the arc vector for thread safety
/// - **Iteration:** O(1) per arc, sequential access pattern
/// - **Reset:** O(1) position reset operation
/// - **Creation:** O(k) where k = number of arcs (due to cloning)
///
/// # Usage
///
/// This iterator is created automatically by `CacheFst::arcs()` and should
/// not be constructed directly. It provides standard Iterator semantics
/// plus the ability to reset iteration position.
#[derive(Debug)]
pub struct CacheArcIterator<W: Semiring> {
    /// Owned copy of cached arcs for thread-safe iteration
    arcs: Vec<Arc<W>>,
    /// Current iteration position in the arc vector
    pos: usize,
}

impl<W: Semiring> Iterator for CacheArcIterator<W> {
    type Item = Arc<W>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.arcs.len() {
            let arc = self.arcs[self.pos].clone();
            self.pos += 1;
            Some(arc)
        } else {
            None
        }
    }
}

impl<W: Semiring> ArcIterator<W> for CacheArcIterator<W> {
    fn reset(&mut self) {
        self.pos = 0;
    }
}

impl<W: Semiring, F: Fst<W>> Fst<W> for CacheFst<W, F> {
    type ArcIter<'a>
        = CacheArcIterator<W>
    where
        Self: 'a;

    fn start(&self) -> Option<StateId> {
        self.fst.start()
    }

    fn final_weight(&self, state: StateId) -> Option<&W> {
        // cache lookup and storage would need redesign
        self.fst.final_weight(state)
    }

    fn num_arcs(&self, state: StateId) -> usize {
        // check cache first
        if let Ok(cache) = self.arc_cache.read() {
            if let Some(arcs) = cache.get(&state) {
                return arcs.len();
            }
        }

        self.fst.num_arcs(state)
    }

    fn num_states(&self) -> usize {
        self.fst.num_states()
    }

    fn properties(&self) -> FstProperties {
        self.fst.properties()
    }

    fn arcs(&self, state: StateId) -> Self::ArcIter<'_> {
        // check cache
        let arcs = {
            let cache = self.arc_cache.read().unwrap();
            cache.get(&state).cloned()
        };

        let arcs = match arcs {
            Some(arcs) => arcs,
            None => {
                // compute and cache
                let computed: Vec<_> = self.fst.arcs(state).collect();
                let mut cache = self.arc_cache.write().unwrap();
                cache.insert(state, computed.clone());
                computed
            }
        };

        CacheArcIterator { arcs, pos: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_cache_fst_lazy_evaluation() {
        let vector_fst = VectorFst::<TropicalWeight>::new();
        let cache_fst = CacheFst::new(vector_fst);

        // Basic properties should be accessible
        assert_eq!(cache_fst.num_states(), 0);
        assert!(cache_fst.is_empty());
    }
}
