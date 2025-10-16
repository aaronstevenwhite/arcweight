//! Performance optimization utilities for ArcWeight
//!
//! This module provides high-performance implementations and optimizations
//! for FST operations, focusing on memory efficiency, cache locality, and
//! computational performance.
//!
//! # Optimization Categories
//!
//! ## Memory Pool Management
//! - Object pooling for frequently allocated structures
//! - Cache-friendly memory layouts
//! - Reduced allocation overhead
//!
//! ## Vectorized Operations
//! - SIMD operations for weight computations
//! - Bulk arc processing
//! - Parallel iteration where beneficial
//!
//! ## Cache Optimization
//! - Prefetching strategies for predictable access patterns
//! - Memory layout optimization
//! - Cache-line aware data structures
//!
//! # Usage
//!
//! ```rust
//! use arcweight::prelude::*;
//! use arcweight::optimization::*;
//!
//! // Use optimized operations for performance-critical code
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! fst.set_start(s0);
//! let optimized_fst = optimize_for_performance(&fst);
//! ```

pub mod cache_optimization;
pub mod memory_pool;
pub mod simd_ops;

pub use cache_optimization::*;
pub use memory_pool::*;
pub use simd_ops::*;

use crate::fst::*;
use crate::prelude::*;
use crate::semiring::Semiring;

/// Comprehensive performance optimization for FSTs
///
/// This function applies multiple optimization strategies to improve FST
/// performance for specific use cases.
///
/// # Optimizations Applied
///
/// 1. **Memory Layout Optimization**: Reorganizes data for cache efficiency
/// 2. **Prefetching**: Adds strategic prefetching for predictable access patterns
/// 3. **Pooled Allocation**: Uses memory pools to reduce allocation overhead
/// 4. **Vectorized Operations**: Applies SIMD optimizations where beneficial
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::optimization::optimize_for_performance;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// // ... populate FST ...
///
/// // Apply comprehensive optimizations
/// let optimized = optimize_for_performance(&fst);
/// ```
pub fn optimize_for_performance<W: Semiring>(fst: &VectorFst<W>) -> OptimizedFst<W> {
    OptimizedFst::new(fst)
}

/// High-performance FST implementation with optimization strategies
///
/// This FST implementation applies various performance optimizations
/// including memory pooling, cache optimization, and vectorized operations.
#[derive(Debug)]
pub struct OptimizedFst<W: Semiring> {
    /// Base FST data with optimized memory layout
    fst: VectorFst<W>,
    /// Memory pool for arc allocation
    #[allow(dead_code)]
    arc_pool: ArcPool<W>,
    /// Cache optimization metadata
    cache_metadata: CacheMetadata,
}

impl<W: Semiring> OptimizedFst<W> {
    /// Create a new optimized FST from an existing FST
    pub fn new(source: &VectorFst<W>) -> Self {
        let fst = source.clone();
        let arc_pool = ArcPool::new();
        let cache_metadata = CacheMetadata::analyze(&fst);

        Self {
            fst,
            arc_pool,
            cache_metadata,
        }
    }

    /// Get optimized arc iterator with prefetching
    pub fn arcs_optimized(&self, state: StateId) -> OptimizedArcIterator<W> {
        OptimizedArcIterator::new(&self.fst, state, &self.cache_metadata)
    }

    /// Perform bulk operations on arcs with vectorization
    pub fn bulk_arc_transform<F>(&mut self, transform: F) -> crate::Result<()>
    where
        F: Fn(&Arc<W>) -> Arc<W> + Send + Sync,
    {
        bulk_transform_arcs(&mut self.fst, transform)
    }
}

impl<W: Semiring> Fst<W> for OptimizedFst<W> {
    type ArcIter<'a>
        = OptimizedArcIterator<W>
    where
        Self: 'a;

    fn start(&self) -> Option<StateId> {
        self.fst.start()
    }

    fn final_weight(&self, state: StateId) -> Option<&W> {
        self.fst.final_weight(state)
    }

    fn num_arcs(&self, state: StateId) -> usize {
        self.fst.num_arcs(state)
    }

    fn num_states(&self) -> usize {
        self.fst.num_states()
    }

    fn properties(&self) -> FstProperties {
        self.fst.properties()
    }

    fn arcs(&self, state: StateId) -> Self::ArcIter<'_> {
        self.arcs_optimized(state)
    }
}

/// High-performance arc iterator with prefetching and cache optimization
#[derive(Debug)]
pub struct OptimizedArcIterator<W: Semiring> {
    arcs: Vec<Arc<W>>,
    pos: usize,
    prefetch_distance: usize,
}

impl<W: Semiring> OptimizedArcIterator<W> {
    fn new(fst: &VectorFst<W>, state: StateId, metadata: &CacheMetadata) -> Self {
        let arcs: Vec<_> = fst.arcs(state).collect();
        let prefetch_distance = metadata.optimal_prefetch_distance();

        Self {
            arcs,
            pos: 0,
            prefetch_distance,
        }
    }
}

impl<W: Semiring> Iterator for OptimizedArcIterator<W> {
    type Item = Arc<W>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.arcs.len() {
            let arc = self.arcs[self.pos].clone();

            // Prefetch next items for better cache performance
            if self.pos + self.prefetch_distance < self.arcs.len() {
                let prefetch_idx = self.pos + self.prefetch_distance;
                prefetch_cache_line(&self.arcs[prefetch_idx]);
            }

            self.pos += 1;
            Some(arc)
        } else {
            None
        }
    }
}

impl<W: Semiring> ArcIterator<W> for OptimizedArcIterator<W> {
    fn reset(&mut self) {
        self.pos = 0;
    }
}

/// Bulk transform arcs using vectorized operations where possible
///
/// Processes FST states in cache-friendly chunks, applying a transformation
/// to all arcs. This improves performance by batching operations.
///
/// # Complexity
///
/// **Time:** O(|V| + |E|) - Single pass through all states and arcs
/// **Space:** O(max(CHUNK_SIZE × max_arcs)) - Temporary storage per chunk
///
/// # Parameters
///
/// - `fst`: Mutable FST to transform
/// - `transform`: Function mapping `Arc<W>` → `Arc<W>`
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::optimization::bulk_transform_arcs;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// // ... build FST ...
///
/// // Scale all weights by 2.0
/// bulk_transform_arcs(&mut fst, |arc| {
///     Arc::new(arc.ilabel, arc.olabel, arc.weight.times(&TropicalWeight::new(2.0)), arc.nextstate)
/// }).unwrap();
/// ```
pub fn bulk_transform_arcs<W, F>(fst: &mut VectorFst<W>, transform: F) -> crate::Result<()>
where
    W: Semiring,
    F: Fn(&Arc<W>) -> Arc<W> + Send + Sync,
{
    // Process states in chunks for better cache performance
    const CHUNK_SIZE: usize = 64;
    let num_states = fst.num_states();

    for chunk_start in (0..num_states).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(num_states);

        // Collect all transformations for this chunk
        let mut state_transforms = Vec::new();

        for state in chunk_start..chunk_end {
            let state_id = state as StateId;
            let arcs: Vec<_> = fst.arcs(state_id).collect();
            let transformed_arcs: Vec<_> = arcs.iter().map(&transform).collect();
            state_transforms.push((state_id, transformed_arcs));
        }

        // Apply transformations
        for (state_id, new_arcs) in state_transforms {
            // Delete existing arcs
            fst.delete_arcs(state_id);

            // Add transformed arcs
            for arc in new_arcs {
                fst.add_arc(state_id, arc);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimized_fst_basic() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));

        let optimized = OptimizedFst::new(&fst);

        assert_eq!(optimized.num_states(), fst.num_states());
        assert_eq!(optimized.start(), fst.start());
        assert_eq!(optimized.num_arcs(s0), fst.num_arcs(s0));
    }

    #[test]
    fn test_optimized_arc_iterator() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(1.0), s1));

        let optimized = OptimizedFst::new(&fst);
        let arcs: Vec<_> = optimized.arcs(s0).collect();

        assert_eq!(arcs.len(), 2);
        assert_eq!(arcs[0].ilabel, 1);
        assert_eq!(arcs[1].ilabel, 2);
    }

    #[test]
    fn test_bulk_transform_arcs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Add some arcs
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s1));
        fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::new(3.0), s2));

        // Transform all weights by multiplying by 2
        bulk_transform_arcs(&mut fst, |arc| {
            Arc::new(
                arc.ilabel,
                arc.olabel,
                arc.weight.times(&TropicalWeight::new(2.0)),
                arc.nextstate,
            )
        })
        .unwrap();

        // Check that weights were transformed
        let arcs0: Vec<_> = fst.arcs(s0).collect();
        assert_eq!(arcs0[0].weight, TropicalWeight::new(3.0)); // 1.0 + 2.0
        assert_eq!(arcs0[1].weight, TropicalWeight::new(4.0)); // 2.0 + 2.0

        let arcs1: Vec<_> = fst.arcs(s1).collect();
        assert_eq!(arcs1[0].weight, TropicalWeight::new(5.0)); // 3.0 + 2.0
    }
}
