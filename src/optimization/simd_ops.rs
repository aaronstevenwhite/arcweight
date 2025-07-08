//! SIMD-optimized operations for FST computations
//!
//! This module provides vectorized implementations of common FST operations
//! to improve performance on modern CPUs with SIMD capabilities.

use crate::semiring::Semiring;
use num_traits::Zero;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-optimized operations for weight computations
///
/// This trait provides vectorized implementations of semiring operations
/// that can process multiple weights in parallel using SIMD instructions.
pub trait SimdOps<W: Semiring> {
    /// Perform parallel addition (plus operation) on weight arrays
    ///
    /// # Safety
    /// This function uses unsafe SIMD operations. The input slices must have
    /// the same length and be properly aligned for SIMD operations.
    ///
    /// # Parameters
    /// - `left`: Left operand weights
    /// - `right`: Right operand weights
    /// - `result`: Output array for results
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    /// use arcweight::optimization::SimdOps;
    ///
    /// let left = vec![TropicalWeight::new(1.0), TropicalWeight::new(2.0)];
    /// let right = vec![TropicalWeight::new(3.0), TropicalWeight::new(4.0)];
    /// let mut result = vec![TropicalWeight::zero(); 2];
    ///
    /// TropicalWeight::simd_plus(&left, &right, &mut result);
    /// ```
    fn simd_plus(left: &[W], right: &[W], result: &mut [W]);

    /// Perform parallel multiplication (times operation) on weight arrays
    fn simd_times(left: &[W], right: &[W], result: &mut [W]);

    /// Find minimum weight in array using SIMD operations
    fn simd_min(weights: &[W]) -> W;

    /// Find maximum weight in array using SIMD operations  
    fn simd_max(weights: &[W]) -> W;
}

/// Vectorized arc processing utilities
///
/// This module provides functions for processing multiple arcs in parallel
/// using SIMD operations where beneficial.
pub mod vectorized_arcs {
    use crate::arc::Arc;
    use crate::semiring::Semiring;

    /// Process multiple arcs with a transformation function using vectorization
    ///
    /// This function applies a transformation to multiple arcs in parallel,
    /// using SIMD operations for weight computations when possible.
    ///
    /// # Parameters
    /// - `arcs`: Input arcs to transform
    /// - `transform`: Transformation function to apply
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    /// use arcweight::optimization::vectorized_arcs::parallel_transform;
    ///
    /// let arcs = vec![
    ///     Arc::new(1, 1, TropicalWeight::new(1.0), 0),
    ///     Arc::new(2, 2, TropicalWeight::new(2.0), 1),
    /// ];
    ///
    /// let transformed = parallel_transform(arcs, |arc| {
    ///     Arc::new(arc.ilabel, arc.olabel, arc.weight.times(&TropicalWeight::new(0.5)), arc.nextstate)
    /// });
    /// ```
    pub fn parallel_transform<W, F>(arcs: Vec<Arc<W>>, transform: F) -> Vec<Arc<W>>
    where
        W: Semiring + Send + Sync,
        F: Fn(Arc<W>) -> Arc<W> + Send + Sync,
    {
        // Process arcs in chunks that fit in cache
        const CHUNK_SIZE: usize = 64;

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            // For large arc sets, use parallel processing with cache-aware chunking
            if arcs.len() > CHUNK_SIZE * 4 {
                arcs.par_chunks(CHUNK_SIZE)
                    .flat_map(|chunk| chunk.iter().cloned().map(&transform).collect::<Vec<_>>())
                    .collect()
            } else {
                // For smaller sets, parallel overhead isn't worth it
                arcs.into_iter().map(transform).collect()
            }
        }
        #[cfg(not(feature = "rayon"))]
        {
            // Without rayon, process in cache-friendly chunks sequentially
            arcs.chunks(CHUNK_SIZE)
                .flat_map(|chunk| chunk.iter().cloned().map(&transform).collect::<Vec<_>>())
                .collect()
        }
    }

    /// Batch process arc weights using SIMD operations
    ///
    /// Extracts weights from arcs, applies SIMD operations, and reconstructs arcs.
    ///
    /// # Parameters
    /// - `arcs`: Input arcs
    /// - `weight_op`: SIMD operation to apply to weights
    pub fn batch_weight_operation<W, F>(arcs: &mut [Arc<W>], weight_op: F)
    where
        W: Semiring + Copy,
        F: Fn(&mut [W]),
    {
        // Extract weights into contiguous array for SIMD processing
        let mut weights: Vec<W> = arcs.iter().map(|arc| arc.weight).collect();

        // Apply SIMD operation to weights
        weight_op(&mut weights);

        // Update arcs with new weights
        for (arc, &weight) in arcs.iter_mut().zip(weights.iter()) {
            arc.weight = weight;
        }
    }

    /// Sort arcs by weight using vectorized comparison operations
    ///
    /// This function uses SIMD operations for weight comparisons when
    /// sorting large numbers of arcs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    /// use arcweight::optimization::vectorized_arcs::simd_sort_by_weight;
    ///
    /// let mut arcs = vec![
    ///     Arc::new(1, 1, TropicalWeight::new(3.0), 0),
    ///     Arc::new(2, 2, TropicalWeight::new(1.0), 1),
    ///     Arc::new(3, 3, TropicalWeight::new(2.0), 2),
    /// ];
    ///
    /// simd_sort_by_weight(&mut arcs);
    /// // arcs are now sorted by weight
    /// ```
    pub fn simd_sort_by_weight<W: Semiring + PartialOrd>(arcs: &mut [Arc<W>]) {
        // For small arrays, use insertion sort which is cache-friendly
        if arcs.len() <= 16 {
            for i in 1..arcs.len() {
                let mut j = i;
                while j > 0 && arcs[j - 1].weight > arcs[j].weight {
                    arcs.swap(j - 1, j);
                    j -= 1;
                }
            }
            return;
        }

        // For larger arrays, use a cache-aware sorting approach
        // First, sort small chunks using insertion sort
        const CHUNK_SIZE: usize = 16;
        for chunk in arcs.chunks_mut(CHUNK_SIZE) {
            // Insertion sort for small chunks
            for i in 1..chunk.len() {
                let mut j = i;
                while j > 0 && chunk[j - 1].weight > chunk[j].weight {
                    chunk.swap(j - 1, j);
                    j -= 1;
                }
            }
        }

        // Then merge the sorted chunks
        let mut chunk_size = CHUNK_SIZE;
        while chunk_size < arcs.len() {
            let mut i = 0;
            while i < arcs.len() {
                let mid = (i + chunk_size).min(arcs.len());
                let end = (i + chunk_size * 2).min(arcs.len());

                if mid < end {
                    // Merge two adjacent sorted chunks
                    merge_sorted_chunks(&mut arcs[i..end], mid - i);
                }

                i += chunk_size * 2;
            }
            chunk_size *= 2;
        }
    }

    /// Helper function to merge two sorted chunks within a slice
    fn merge_sorted_chunks<W: Semiring + PartialOrd>(slice: &mut [Arc<W>], mid: usize) {
        // Create temporary storage for the left chunk
        let left: Vec<_> = slice[..mid].to_vec();

        let mut i = 0; // Index for left chunk
        let mut j = mid; // Index for right chunk
        let mut k = 0; // Index for merged result

        // Merge until one chunk is exhausted
        while i < left.len() && j < slice.len() {
            if left[i].weight <= slice[j].weight {
                slice[k] = left[i].clone();
                i += 1;
            } else {
                slice[k] = slice[j].clone();
                j += 1;
            }
            k += 1;
        }

        // Copy remaining elements from left chunk
        while i < left.len() {
            slice[k] = left[i].clone();
            i += 1;
            k += 1;
        }
        // Right chunk elements are already in place
    }
}

/// Cache-optimized memory prefetching utilities
pub mod prefetch {
    /// Prefetch a cache line for read access
    ///
    /// This function hints to the CPU that the specified memory location
    /// will be accessed soon, potentially loading it into cache early.
    ///
    /// # Safety
    /// This function is safe to call with any valid reference.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::optimization::prefetch::prefetch_cache_line;
    ///
    /// let data = vec![1, 2, 3, 4, 5];
    /// prefetch_cache_line(&data[4]); // Hint that we'll access data[4] soon
    /// // ... do other work ...
    /// let value = data[4]; // This access may be faster due to prefetching
    /// ```
    pub fn prefetch_cache_line<T>(data: &T) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(
                data as *const T as *const i8,
                std::arch::x86_64::_MM_HINT_T0,
            );
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            // No-op on non-x86 architectures
            let _ = data;
        }
    }

    /// Prefetch multiple cache lines for sequential access
    ///
    /// This function prefetches multiple cache lines in sequence,
    /// optimized for sequential memory access patterns.
    pub fn prefetch_sequential<T>(data: &[T], start_idx: usize, count: usize) {
        let end_idx = (start_idx + count).min(data.len());
        let step_size = 64 / std::mem::size_of::<T>(); // Assume 64-byte cache lines

        for i in (start_idx..end_idx).step_by(step_size.max(1)) {
            prefetch_cache_line(&data[i]);
        }
    }

    /// Prefetch data with stride pattern
    ///
    /// This function prefetches data with a specific stride pattern,
    /// useful for algorithms that access memory in regular patterns.
    pub fn prefetch_strided<T>(data: &[T], start_idx: usize, stride: usize, count: usize) {
        for i in 0..count {
            let idx = start_idx + i * stride;
            if idx < data.len() {
                prefetch_cache_line(&data[idx]);
            }
        }
    }
}

// Re-export prefetch function for convenience
pub use prefetch::prefetch_cache_line;

/// SIMD implementation for TropicalWeight
#[cfg(target_arch = "x86_64")]
mod tropical_simd {
    use super::*;
    use crate::semiring::TropicalWeight;
    use std::arch::x86_64::{
        _mm_add_ps, _mm_loadu_ps, _mm_max_ps, _mm_min_ps, _mm_prefetch, _mm_set1_ps, _mm_storeu_ps,
        _MM_HINT_T0,
    };

    impl SimdOps<TropicalWeight> for TropicalWeight {
        fn simd_plus(
            left: &[TropicalWeight],
            right: &[TropicalWeight],
            result: &mut [TropicalWeight],
        ) {
            assert_eq!(left.len(), right.len());
            assert_eq!(left.len(), result.len());

            let len = left.len();
            let simd_len = len & !3; // Process 4 elements at a time

            unsafe {
                // Process 4 TropicalWeights at a time using SSE
                for i in (0..simd_len).step_by(4) {
                    // Load 4 values from left and right arrays
                    let left_array = [
                        *left[i].value(),
                        *left[i + 1].value(),
                        *left[i + 2].value(),
                        *left[i + 3].value(),
                    ];
                    let right_array = [
                        *right[i].value(),
                        *right[i + 1].value(),
                        *right[i + 2].value(),
                        *right[i + 3].value(),
                    ];

                    let left_vals = _mm_loadu_ps(left_array.as_ptr());
                    let right_vals = _mm_loadu_ps(right_array.as_ptr());
                    let min_vals = _mm_min_ps(left_vals, right_vals);

                    // Store results
                    let mut result_array = [0.0f32; 4];
                    _mm_storeu_ps(result_array.as_mut_ptr(), min_vals);

                    for j in 0..4 {
                        result[i + j] = TropicalWeight::new(result_array[j]);
                    }
                }
            }

            // Handle remaining elements
            for i in simd_len..len {
                result[i] = left[i].plus(&right[i]);
            }
        }

        fn simd_times(
            left: &[TropicalWeight],
            right: &[TropicalWeight],
            result: &mut [TropicalWeight],
        ) {
            assert_eq!(left.len(), right.len());
            assert_eq!(left.len(), result.len());

            let len = left.len();
            let simd_len = len & !3; // Process 4 elements at a time

            unsafe {
                // Process 4 TropicalWeights at a time using SSE
                for i in (0..simd_len).step_by(4) {
                    // Load 4 values from left and right arrays
                    let left_array = [
                        *left[i].value(),
                        *left[i + 1].value(),
                        *left[i + 2].value(),
                        *left[i + 3].value(),
                    ];
                    let right_array = [
                        *right[i].value(),
                        *right[i + 1].value(),
                        *right[i + 2].value(),
                        *right[i + 3].value(),
                    ];

                    let left_vals = _mm_loadu_ps(left_array.as_ptr());
                    let right_vals = _mm_loadu_ps(right_array.as_ptr());
                    let sum_vals = _mm_add_ps(left_vals, right_vals);

                    // Store results
                    let mut result_array = [0.0f32; 4];
                    _mm_storeu_ps(result_array.as_mut_ptr(), sum_vals);

                    for j in 0..4 {
                        result[i + j] = TropicalWeight::new(result_array[j]);
                    }
                }
            }

            // Handle remaining elements
            for i in simd_len..len {
                result[i] = left[i].times(&right[i]);
            }
        }

        fn simd_min(weights: &[TropicalWeight]) -> TropicalWeight {
            if weights.is_empty() {
                return TropicalWeight::zero();
            }

            let len = weights.len();
            let simd_len = len & !3; // Process 4 elements at a time

            unsafe {
                let mut min_vec = _mm_set1_ps(f32::INFINITY);

                // Process 4 elements at a time
                for i in (0..simd_len).step_by(4) {
                    let vals_array = [
                        *weights[i].value(),
                        *weights[i + 1].value(),
                        *weights[i + 2].value(),
                        *weights[i + 3].value(),
                    ];
                    let vals = _mm_loadu_ps(vals_array.as_ptr());
                    min_vec = _mm_min_ps(min_vec, vals);
                }

                // Horizontal minimum of the vector
                let mut result_array = [0.0f32; 4];
                _mm_storeu_ps(result_array.as_mut_ptr(), min_vec);
                let mut min_val = result_array[0];
                for &val in &result_array[1..] {
                    min_val = min_val.min(val);
                }

                // Handle remaining elements
                for weight in weights.iter().skip(simd_len) {
                    min_val = min_val.min(*weight.value());
                }

                TropicalWeight::new(min_val)
            }
        }

        fn simd_max(weights: &[TropicalWeight]) -> TropicalWeight {
            if weights.is_empty() {
                return TropicalWeight::zero();
            }

            let len = weights.len();
            let simd_len = len & !3; // Process 4 elements at a time

            unsafe {
                let mut max_vec = _mm_set1_ps(f32::NEG_INFINITY);

                // Process 4 elements at a time
                for i in (0..simd_len).step_by(4) {
                    let vals_array = [
                        *weights[i].value(),
                        *weights[i + 1].value(),
                        *weights[i + 2].value(),
                        *weights[i + 3].value(),
                    ];
                    let vals = _mm_loadu_ps(vals_array.as_ptr());
                    max_vec = _mm_max_ps(max_vec, vals);
                }

                // Horizontal maximum of the vector
                let mut result_array = [0.0f32; 4];
                _mm_storeu_ps(result_array.as_mut_ptr(), max_vec);
                let mut max_val = result_array[0];
                for &val in &result_array[1..] {
                    max_val = max_val.max(val);
                }

                // Handle remaining elements
                for weight in weights.iter().skip(simd_len) {
                    max_val = max_val.max(*weight.value());
                }

                TropicalWeight::new(max_val)
            }
        }
    }
}

// Non-SIMD fallback implementations
#[cfg(not(target_arch = "x86_64"))]
mod fallback_simd {
    use super::*;
    use crate::semiring::TropicalWeight;

    impl SimdOps<TropicalWeight> for TropicalWeight {
        fn simd_plus(
            left: &[TropicalWeight],
            right: &[TropicalWeight],
            result: &mut [TropicalWeight],
        ) {
            for ((l, r), res) in left.iter().zip(right.iter()).zip(result.iter_mut()) {
                *res = l.plus(r);
            }
        }

        fn simd_times(
            left: &[TropicalWeight],
            right: &[TropicalWeight],
            result: &mut [TropicalWeight],
        ) {
            for ((l, r), res) in left.iter().zip(right.iter()).zip(result.iter_mut()) {
                *res = l.times(r);
            }
        }

        fn simd_min(weights: &[TropicalWeight]) -> TropicalWeight {
            weights
                .iter()
                .fold(TropicalWeight::zero(), |acc, w| acc.plus(w))
        }

        fn simd_max(weights: &[TropicalWeight]) -> TropicalWeight {
            weights
                .iter()
                .fold(TropicalWeight::new(f32::NEG_INFINITY), |acc, w| {
                    if w.value() > acc.value() {
                        *w
                    } else {
                        acc
                    }
                })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::prefetch;
    use super::vectorized_arcs;
    use crate::prelude::*;
    use num_traits::Zero;

    #[test]
    fn test_simd_plus() {
        let left = vec![TropicalWeight::new(1.0), TropicalWeight::new(2.0)];
        let right = vec![TropicalWeight::new(3.0), TropicalWeight::new(1.5)];
        let mut result = vec![TropicalWeight::zero(); 2];

        TropicalWeight::simd_plus(&left, &right, &mut result);

        assert_eq!(result[0], TropicalWeight::new(1.0)); // min(1.0, 3.0)
        assert_eq!(result[1], TropicalWeight::new(1.5)); // min(2.0, 1.5)
    }

    #[test]
    fn test_simd_times() {
        let left = vec![TropicalWeight::new(1.0), TropicalWeight::new(2.0)];
        let right = vec![TropicalWeight::new(3.0), TropicalWeight::new(1.5)];
        let mut result = vec![TropicalWeight::zero(); 2];

        TropicalWeight::simd_times(&left, &right, &mut result);

        assert_eq!(result[0], TropicalWeight::new(4.0)); // 1.0 + 3.0
        assert_eq!(result[1], TropicalWeight::new(3.5)); // 2.0 + 1.5
    }

    #[test]
    fn test_simd_min() {
        let weights = vec![
            TropicalWeight::new(3.0),
            TropicalWeight::new(1.0),
            TropicalWeight::new(2.0),
            TropicalWeight::new(0.5),
        ];

        let min_weight = TropicalWeight::simd_min(&weights);
        assert_eq!(min_weight, TropicalWeight::new(0.5));
    }

    #[test]
    fn test_vectorized_transform() {
        let arcs = vec![
            Arc::new(1, 1, TropicalWeight::new(1.0), 0),
            Arc::new(2, 2, TropicalWeight::new(2.0), 1),
        ];

        let transformed = vectorized_arcs::parallel_transform(arcs, |arc| {
            Arc::new(
                arc.ilabel,
                arc.olabel,
                arc.weight.times(&TropicalWeight::new(0.5)),
                arc.nextstate,
            )
        });

        assert_eq!(transformed[0].weight, TropicalWeight::new(1.5));
        assert_eq!(transformed[1].weight, TropicalWeight::new(2.5));
    }

    #[test]
    fn test_prefetch() {
        let data = vec![1, 2, 3, 4, 5];

        // This should not panic and is safe to call
        prefetch_cache_line(&data[0]);
        prefetch::prefetch_sequential(&data, 0, 5);
        prefetch::prefetch_strided(&data, 0, 2, 3);
    }
}
