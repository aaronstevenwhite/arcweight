//! Memory pool implementations for high-performance FST operations
//!
//! This module provides object pooling and memory management optimizations
//! to reduce allocation overhead and improve cache locality for FST operations.

use crate::arc::Arc;
use crate::semiring::Semiring;
use std::collections::VecDeque;
use std::sync::{Arc as SyncArc, Mutex};

/// High-performance memory pool for arc allocation
///
/// Reduces allocation overhead by reusing arc objects, particularly beneficial
/// for algorithms that create and destroy many arcs.
///
/// # Performance Benefits
///
/// - **Reduced Allocations**: Reuses existing arc objects
/// - **Cache Locality**: Keeps frequently used objects in cache
/// - **Predictable Performance**: Eliminates allocation spikes
/// - **Lower Fragmentation**: Reduces memory fragmentation
///
/// # Usage
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::optimization::ArcPool;
///
/// let pool = ArcPool::<TropicalWeight>::new();
///
/// // Get arc from pool (or create new if pool empty)
/// let arc = pool.get_arc(1, 1, TropicalWeight::new(0.5), 0);
///
/// // Use arc...
///
/// // Return to pool for reuse
/// pool.return_arc(arc);
/// ```
#[derive(Debug)]
pub struct ArcPool<W: Semiring> {
    /// Pool of available arcs for reuse
    pool: Mutex<VecDeque<Arc<W>>>,
    /// Maximum number of arcs to keep in pool
    max_pool_size: usize,
    /// Statistics for pool performance monitoring
    stats: Mutex<PoolStats>,
}

/// Statistics for memory pool performance monitoring
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total number of arcs requested from pool
    pub requests: usize,
    /// Number of requests satisfied from pool (hits)
    pub hits: usize,
    /// Number of requests requiring new allocation (misses)
    pub misses: usize,
    /// Current number of arcs in pool
    pub pool_size: usize,
    /// Maximum pool size reached
    pub max_pool_size_reached: usize,
}

impl<W: Semiring> ArcPool<W> {
    /// Create a new arc pool with default configuration
    ///
    /// # Configuration
    /// - **Max Pool Size**: 1000 arcs
    /// - **Initial Capacity**: 100 arcs
    pub fn new() -> Self {
        Self::with_capacity(1000)
    }

    /// Create a new arc pool with specified maximum capacity
    ///
    /// # Parameters
    /// - `max_size`: Maximum number of arcs to keep in pool
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::optimization::ArcPool;
    /// use arcweight::prelude::TropicalWeight;
    ///
    /// // Create pool with capacity for 500 arcs
    /// let pool = ArcPool::<TropicalWeight>::with_capacity(500);
    /// ```
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            pool: Mutex::new(VecDeque::with_capacity(max_size.min(100))),
            max_pool_size: max_size,
            stats: Mutex::new(PoolStats::default()),
        }
    }

    /// Get an arc from the pool or create a new one
    ///
    /// This method first attempts to reuse an arc from the pool. If the pool
    /// is empty, it creates a new arc. The returned arc has the specified
    /// labels, weight, and next state.
    ///
    /// # Parameters
    /// - `ilabel`: Input label for the arc
    /// - `olabel`: Output label for the arc
    /// - `weight`: Weight for the arc
    /// - `nextstate`: Next state for the arc
    ///
    /// # Performance
    /// - **Pool Hit**: O(1) - reuses existing arc
    /// - **Pool Miss**: O(1) + allocation cost
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    /// use arcweight::optimization::ArcPool;
    ///
    /// let pool = ArcPool::new();
    /// let arc = pool.get_arc(1, 1, TropicalWeight::new(0.5), 2);
    ///
    /// assert_eq!(arc.ilabel, 1);
    /// assert_eq!(arc.olabel, 1);
    /// assert_eq!(arc.nextstate, 2);
    /// ```
    pub fn get_arc(&self, ilabel: u32, olabel: u32, weight: W, nextstate: u32) -> Arc<W> {
        let mut stats = self.stats.lock().unwrap();
        stats.requests += 1;

        let mut pool = self.pool.lock().unwrap();
        stats.pool_size = pool.len();

        if let Some(mut arc) = pool.pop_front() {
            // Reuse existing arc
            arc.ilabel = ilabel;
            arc.olabel = olabel;
            arc.weight = weight;
            arc.nextstate = nextstate;

            stats.hits += 1;
            arc
        } else {
            // Create new arc
            stats.misses += 1;
            Arc::new(ilabel, olabel, weight, nextstate)
        }
    }

    /// Return an arc to the pool for reuse
    ///
    /// This method returns an arc to the pool so it can be reused by future
    /// `get_arc` calls. If the pool is full, the arc is discarded.
    ///
    /// # Parameters
    /// - `arc`: Arc to return to the pool
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    /// use arcweight::optimization::ArcPool;
    ///
    /// let pool = ArcPool::new();
    /// let arc = pool.get_arc(1, 1, TropicalWeight::new(0.5), 2);
    ///
    /// // Use arc...
    ///
    /// // Return to pool for reuse
    /// pool.return_arc(arc);
    /// ```
    pub fn return_arc(&self, arc: Arc<W>) {
        let mut pool = self.pool.lock().unwrap();

        if pool.len() < self.max_pool_size {
            pool.push_back(arc);

            let mut stats = self.stats.lock().unwrap();
            stats.pool_size = pool.len();
            stats.max_pool_size_reached = stats.max_pool_size_reached.max(pool.len());
        }
        // If pool is full, arc is dropped and deallocated
    }

    /// Get current pool statistics
    ///
    /// Returns performance statistics for the pool, useful for monitoring
    /// and tuning pool performance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    /// use arcweight::optimization::ArcPool;
    ///
    /// let pool = ArcPool::<TropicalWeight>::new();
    /// let stats = pool.stats();
    ///
    /// println!("Hit rate: {:.2}%", stats.hit_rate() * 100.0);
    /// println!("Pool utilization: {}", stats.pool_size);
    /// ```
    pub fn stats(&self) -> PoolStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear all arcs from the pool
    ///
    /// This method removes all arcs from the pool, freeing their memory.
    /// Useful for memory management in long-running applications.
    pub fn clear(&self) {
        let mut pool = self.pool.lock().unwrap();
        pool.clear();

        let mut stats = self.stats.lock().unwrap();
        stats.pool_size = 0;
    }

    /// Pre-allocate arcs in the pool
    ///
    /// This method pre-fills the pool with arcs to avoid allocation overhead
    /// during initial operations. The pre-allocated arcs have default values
    /// and will be overwritten when retrieved.
    ///
    /// # Parameters
    /// - `count`: Number of arcs to pre-allocate
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    /// use arcweight::optimization::ArcPool;
    ///
    /// let pool = ArcPool::<TropicalWeight>::new();
    /// pool.preallocate(100);  // Pre-allocate 100 arcs
    ///
    /// // First 100 get_arc calls will not require allocation
    /// ```
    pub fn preallocate(&self, count: usize) {
        let mut pool = self.pool.lock().unwrap();
        let to_allocate = count.min(self.max_pool_size - pool.len());

        for _ in 0..to_allocate {
            pool.push_back(Arc::new(0, 0, W::zero(), 0));
        }

        let mut stats = self.stats.lock().unwrap();
        stats.pool_size = pool.len();
        stats.max_pool_size_reached = stats.max_pool_size_reached.max(pool.len());
    }
}

impl<W: Semiring> Default for ArcPool<W> {
    fn default() -> Self {
        Self::new()
    }
}

impl PoolStats {
    /// Calculate hit rate as a percentage (0.0 to 1.0)
    ///
    /// Returns the percentage of requests that were satisfied from the pool
    /// rather than requiring new allocation.
    pub fn hit_rate(&self) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            self.hits as f64 / self.requests as f64
        }
    }

    /// Calculate miss rate as a percentage (0.0 to 1.0)
    ///
    /// Returns the percentage of requests that required new allocation.
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }

    /// Check if pool performance is good
    ///
    /// Returns true if hit rate is above 80% and pool is being utilized effectively.
    pub fn is_performing_well(&self) -> bool {
        self.hit_rate() > 0.8 && self.requests > 10
    }
}

/// Shared arc pool for use across multiple threads
///
/// This type provides a thread-safe arc pool that can be shared across
/// multiple threads for coordinated memory management.
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::optimization::{ArcPool, SharedArcPool};
/// use std::thread;
///
/// let pool = SharedArcPool::new(ArcPool::<TropicalWeight>::new());
/// let pool_clone = pool.clone();
///
/// let handle = thread::spawn(move || {
///     let arc = pool_clone.get_arc(1, 1, TropicalWeight::new(0.5), 2);
///     // Use arc...
///     pool_clone.return_arc(arc);
/// });
///
/// handle.join().unwrap();
/// ```
pub type SharedArcPool<W> = SyncArc<ArcPool<W>>;

/// Batch arc allocator for high-performance bulk operations
///
/// This allocator is optimized for scenarios where many arcs need to be
/// created at once, such as during FST construction or transformation.
///
/// # Performance Benefits
///
/// - **Bulk Allocation**: Allocates arcs in batches to reduce overhead
/// - **Contiguous Memory**: Arcs allocated in contiguous blocks for cache efficiency
/// - **Reduced Fragmentation**: Fewer, larger allocations reduce fragmentation
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::optimization::BatchArcAllocator;
///
/// let allocator = BatchArcAllocator::<TropicalWeight>::new();
///
/// // Allocate 1000 arcs at once
/// let arcs = allocator.allocate_batch(Some(1000));
///
/// // Use arcs...
///
/// // Return batch to allocator
/// allocator.return_batch(arcs);
/// ```
#[derive(Debug)]
pub struct BatchArcAllocator<W: Semiring> {
    /// Pool of available arc batches
    batches: Mutex<Vec<Vec<Arc<W>>>>,
    /// Standard batch size
    batch_size: usize,
    /// Maximum number of batches to keep
    max_batches: usize,
}

impl<W: Semiring> BatchArcAllocator<W> {
    /// Create a new batch allocator with default configuration
    pub fn new() -> Self {
        Self::with_config(1000, 10)
    }

    /// Create a new batch allocator with custom configuration
    ///
    /// # Parameters
    /// - `batch_size`: Number of arcs per batch
    /// - `max_batches`: Maximum number of batches to keep in pool
    pub fn with_config(batch_size: usize, max_batches: usize) -> Self {
        Self {
            batches: Mutex::new(Vec::new()),
            batch_size,
            max_batches,
        }
    }

    /// Allocate a batch of arcs
    ///
    /// Returns a vector of arcs, either from the pool or newly allocated.
    /// The returned arcs have default values and should be initialized
    /// before use.
    ///
    /// # Parameters
    /// - `count`: Number of arcs to allocate (defaults to batch_size if not specified)
    pub fn allocate_batch(&self, count: Option<usize>) -> Vec<Arc<W>> {
        let size = count.unwrap_or(self.batch_size);
        let mut batches = self.batches.lock().unwrap();

        // Try to reuse existing batch of appropriate size
        if let Some(pos) = batches.iter().position(|batch| batch.len() >= size) {
            let mut batch = batches.remove(pos);
            batch.truncate(size);
            batch
        } else {
            // Create new batch
            (0..size).map(|_| Arc::new(0, 0, W::zero(), 0)).collect()
        }
    }

    /// Return a batch of arcs to the allocator
    ///
    /// # Parameters
    /// - `batch`: Vector of arcs to return
    pub fn return_batch(&self, batch: Vec<Arc<W>>) {
        let mut batches = self.batches.lock().unwrap();

        if batches.len() < self.max_batches {
            batches.push(batch);
        }
        // If pool is full, batch is dropped
    }

    /// Clear all batches from the allocator
    pub fn clear(&self) {
        let mut batches = self.batches.lock().unwrap();
        batches.clear();
    }
}

impl<W: Semiring> Default for BatchArcAllocator<W> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_arc_pool_basic() {
        let pool = ArcPool::<TropicalWeight>::new();

        // Get arc from pool
        let arc = pool.get_arc(1, 2, TropicalWeight::new(0.5), 3);
        assert_eq!(arc.ilabel, 1);
        assert_eq!(arc.olabel, 2);
        assert_eq!(arc.nextstate, 3);

        // Return arc to pool
        pool.return_arc(arc);

        // Get another arc (should reuse the returned one)
        let arc2 = pool.get_arc(4, 5, TropicalWeight::new(1.0), 6);
        assert_eq!(arc2.ilabel, 4);
        assert_eq!(arc2.olabel, 5);
        assert_eq!(arc2.nextstate, 6);
    }

    #[test]
    fn test_pool_stats() {
        let pool = ArcPool::<TropicalWeight>::new();

        // Initially no requests
        let stats = pool.stats();
        assert_eq!(stats.requests, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);

        // First request should be a miss
        let arc = pool.get_arc(1, 1, TropicalWeight::new(0.5), 2);
        let stats = pool.stats();
        assert_eq!(stats.requests, 1);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 1);

        // Return arc and get another should be a hit
        pool.return_arc(arc);
        let _arc2 = pool.get_arc(2, 2, TropicalWeight::new(1.0), 3);
        let stats = pool.stats();
        assert_eq!(stats.requests, 2);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[test]
    fn test_batch_allocator() {
        let allocator = BatchArcAllocator::<TropicalWeight>::new();

        // Allocate a batch
        let batch = allocator.allocate_batch(Some(100));
        assert_eq!(batch.len(), 100);

        // Return batch
        allocator.return_batch(batch);

        // Allocate another batch (should reuse)
        let batch2 = allocator.allocate_batch(Some(50));
        assert_eq!(batch2.len(), 50);
    }

    #[test]
    fn test_preallocate() {
        let pool = ArcPool::<TropicalWeight>::new();

        // Preallocate some arcs
        pool.preallocate(10);

        let stats = pool.stats();
        assert_eq!(stats.pool_size, 10);

        // Getting arcs should now be hits
        let _arc = pool.get_arc(1, 1, TropicalWeight::new(0.5), 2);
        let stats = pool.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
    }
}
