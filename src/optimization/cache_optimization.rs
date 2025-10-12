//! Cache optimization utilities for improved FST performance
//!
//! This module provides cache-aware data structures and algorithms
//! to improve performance through better memory access patterns.

use crate::fst::*;
use crate::semiring::Semiring;
use std::collections::HashMap;

/// Cache performance metadata for FST optimization
///
/// This structure analyzes FST access patterns and provides optimization
/// hints for cache-friendly operations.
#[derive(Debug, Clone)]
pub struct CacheMetadata {
    /// Average number of arcs per state
    pub avg_arcs_per_state: f64,
    /// Distribution of arc counts across states
    pub arc_count_distribution: Vec<(usize, usize)>, // (arc_count, frequency)
    /// Estimated cache line utilization
    pub cache_line_utilization: f64,
    /// Recommended prefetch distance
    pub prefetch_distance: usize,
    /// Memory access pattern analysis
    pub access_pattern: AccessPattern,
}

/// Memory access pattern classification
#[derive(Debug, Clone, PartialEq)]
pub enum AccessPattern {
    /// Sequential access (predictable)
    Sequential,
    /// Random access (unpredictable)
    Random,
    /// Clustered access (some locality)
    Clustered,
    /// Sparse access (widely distributed)
    Sparse,
}

impl CacheMetadata {
    /// Analyze an FST to generate cache optimization metadata
    ///
    /// # Complexity
    ///
    /// **Time:** O(|V|) - Collects arc counts and computes statistics
    /// **Space:** O(|V|) - Stores arc count distribution
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    /// use arcweight::optimization::CacheMetadata;
    ///
    /// let mut fst = VectorFst::<TropicalWeight>::new();
    /// let s0 = fst.add_state();
    /// let s1 = fst.add_state();
    /// fst.set_start(s0);
    /// fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
    ///
    /// let metadata = CacheMetadata::analyze(&fst);
    /// println!("Average arcs per state: {}", metadata.avg_arcs_per_state);
    /// ```
    pub fn analyze<W: Semiring>(fst: &VectorFst<W>) -> Self {
        let num_states = fst.num_states();
        if num_states == 0 {
            return Self::default();
        }

        // Collect arc count statistics
        let mut arc_counts = Vec::new();
        let mut total_arcs = 0;

        for state in 0..num_states as StateId {
            let num_arcs = fst.num_arcs(state);
            arc_counts.push(num_arcs);
            total_arcs += num_arcs;
        }

        // Calculate average arcs per state
        let avg_arcs_per_state = total_arcs as f64 / num_states as f64;

        // Build arc count distribution
        let mut distribution_map = HashMap::new();
        for &count in &arc_counts {
            *distribution_map.entry(count).or_insert(0) += 1;
        }
        let mut arc_count_distribution: Vec<_> = distribution_map.into_iter().collect();
        arc_count_distribution.sort_by_key(|&(count, _)| count);

        // Estimate cache line utilization
        let cache_line_utilization = Self::estimate_cache_utilization(&arc_counts);

        // Determine optimal prefetch distance
        let prefetch_distance = Self::calculate_prefetch_distance(avg_arcs_per_state);

        // Classify access pattern
        let access_pattern = Self::classify_access_pattern(&arc_counts);

        Self {
            avg_arcs_per_state,
            arc_count_distribution,
            cache_line_utilization,
            prefetch_distance,
            access_pattern,
        }
    }

    /// Get optimal prefetch distance for this FST
    pub fn optimal_prefetch_distance(&self) -> usize {
        self.prefetch_distance
    }

    /// Check if FST is cache-friendly
    pub fn is_cache_friendly(&self) -> bool {
        self.cache_line_utilization > 0.7
            && matches!(
                self.access_pattern,
                AccessPattern::Sequential | AccessPattern::Clustered
            )
    }

    /// Get optimization recommendations
    pub fn recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        if self.cache_line_utilization < 0.5 {
            recommendations.push(OptimizationRecommendation::ImproveDataLayout);
        }

        if self.avg_arcs_per_state > 10.0 {
            recommendations.push(OptimizationRecommendation::UsePrefetching);
        }

        if matches!(
            self.access_pattern,
            AccessPattern::Random | AccessPattern::Sparse
        ) {
            recommendations.push(OptimizationRecommendation::UseBlocking);
        }

        if self.avg_arcs_per_state < 2.0 {
            recommendations.push(OptimizationRecommendation::ConsiderCompression);
        }

        recommendations
    }

    fn estimate_cache_utilization(arc_counts: &[usize]) -> f64 {
        // Estimate based on typical cache line size (64 bytes) and arc size
        const CACHE_LINE_SIZE: usize = 64;
        const ESTIMATED_ARC_SIZE: usize = 16; // Rough estimate
        const ARCS_PER_CACHE_LINE: usize = CACHE_LINE_SIZE / ESTIMATED_ARC_SIZE;

        let total_states = arc_counts.len();
        let well_utilized_states = arc_counts
            .iter()
            .filter(|&&count| count >= ARCS_PER_CACHE_LINE / 2)
            .count();

        well_utilized_states as f64 / total_states as f64
    }

    fn calculate_prefetch_distance(avg_arcs: f64) -> usize {
        // Heuristic: prefetch distance based on arc density
        if avg_arcs < 2.0 {
            1
        } else if avg_arcs < 5.0 {
            2
        } else if avg_arcs < 10.0 {
            3
        } else {
            4
        }
    }

    fn classify_access_pattern(arc_counts: &[usize]) -> AccessPattern {
        if arc_counts.is_empty() {
            return AccessPattern::Sequential;
        }

        // Calculate variance in arc counts
        let mean = arc_counts.iter().sum::<usize>() as f64 / arc_counts.len() as f64;
        let variance = arc_counts
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / arc_counts.len() as f64;
        let std_dev = variance.sqrt();

        // Classify based on distribution characteristics
        let coefficient_of_variation = std_dev / mean;

        if coefficient_of_variation < 0.3 {
            AccessPattern::Sequential
        } else if coefficient_of_variation < 0.7 {
            AccessPattern::Clustered
        } else if mean < 2.0 {
            AccessPattern::Sparse
        } else {
            AccessPattern::Random
        }
    }
}

impl Default for CacheMetadata {
    fn default() -> Self {
        Self {
            avg_arcs_per_state: 0.0,
            arc_count_distribution: Vec::new(),
            cache_line_utilization: 0.0,
            prefetch_distance: 1,
            access_pattern: AccessPattern::Sequential,
        }
    }
}

/// Optimization recommendations based on cache analysis
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationRecommendation {
    /// Improve data layout for better cache utilization
    ImproveDataLayout,
    /// Use prefetching for large arc counts
    UsePrefetching,
    /// Use cache blocking for random access patterns
    UseBlocking,
    /// Consider compression for sparse FSTs
    ConsiderCompression,
    /// Use memory pooling for frequent allocations
    UseMemoryPooling,
}

/// Cache-optimized data layout utilities
pub mod layout_optimization {
    use super::*;
    use crate::arc::Arc;
    use crate::semiring::Semiring;

    /// Reorder FST states for improved cache locality
    ///
    /// This function reorders states in an FST to improve cache performance
    /// by placing frequently accessed states closer together in memory.
    ///
    /// # Complexity
    ///
    /// **Time:** O(|V| + |E| log(max_degree)) - BFS + sorting neighbors by in-degree
    /// **Space:** O(|V| + |E|) - Adjacency lists and new FST
    ///
    /// # Correctness
    ///
    /// **Guarantee:** Creates isomorphic FST with L(T') = L(T)
    /// - All states and arcs preserved
    /// - State IDs remapped via old_to_new HashMap
    /// - Start state and final weights preserved
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    /// use arcweight::optimization::layout_optimization::reorder_for_locality;
    ///
    /// let mut fst = VectorFst::<TropicalWeight>::new();
    /// // ... populate FST ...
    ///
    /// let optimized_fst = reorder_for_locality(&fst);
    /// ```
    pub fn reorder_for_locality<W: Semiring>(fst: &VectorFst<W>) -> VectorFst<W> {
        use std::collections::{HashMap, VecDeque};

        let num_states = fst.num_states();
        if num_states == 0 {
            return fst.clone();
        }

        // Build adjacency information
        let mut adjacency: HashMap<StateId, Vec<StateId>> = HashMap::new();
        let mut in_degree: HashMap<StateId, usize> = HashMap::new();

        for state in 0..num_states as StateId {
            let neighbors: Vec<StateId> = fst.arcs(state).map(|arc| arc.nextstate).collect();

            // Count in-degrees
            for &next in &neighbors {
                *in_degree.entry(next).or_insert(0) += 1;
            }

            adjacency.insert(state, neighbors);
        }

        // BFS traversal from start state to determine new ordering
        let mut new_order = Vec::new();
        let mut visited = vec![false; num_states];
        let mut queue = VecDeque::new();

        // Start from the start state if available
        if let Some(start) = fst.start() {
            queue.push_back(start);
            visited[start as usize] = true;
        }

        // BFS to group connected states together
        while let Some(state) = queue.pop_front() {
            new_order.push(state);

            // Sort neighbors by in-degree to prioritize highly connected states
            let mut neighbors = adjacency.get(&state).cloned().unwrap_or_default();
            neighbors.sort_by_key(|&s| std::cmp::Reverse(in_degree.get(&s).copied().unwrap_or(0)));

            for next in neighbors {
                if !visited[next as usize] {
                    visited[next as usize] = true;
                    queue.push_back(next);
                }
            }
        }

        // Add any unvisited states (disconnected components)
        for state in 0..num_states as StateId {
            if !visited[state as usize] {
                new_order.push(state);
            }
        }

        // Create mapping from old state IDs to new state IDs
        let mut old_to_new: HashMap<StateId, StateId> = HashMap::new();
        for (new_id, &old_id) in new_order.iter().enumerate() {
            old_to_new.insert(old_id, new_id as StateId);
        }

        // Build new FST with reordered states
        let mut new_fst = VectorFst::new();

        // Add all states first
        for _ in 0..num_states {
            new_fst.add_state();
        }

        // Set start state
        if let Some(start) = fst.start() {
            new_fst.set_start(*old_to_new.get(&start).unwrap());
        }

        // Copy arcs and final weights with remapped state IDs
        for (old_state, &new_state) in &old_to_new {
            // Copy arcs with remapped target states
            for arc in fst.arcs(*old_state) {
                let new_arc = Arc::new(
                    arc.ilabel,
                    arc.olabel,
                    arc.weight.clone(),
                    *old_to_new.get(&arc.nextstate).unwrap(),
                );
                new_fst.add_arc(new_state, new_arc);
            }

            // Copy final weight
            if let Some(weight) = fst.final_weight(*old_state) {
                new_fst.set_final(new_state, weight.clone());
            }
        }

        new_fst
    }

    /// Optimize arc layout within states for cache efficiency
    ///
    /// This function reorders arcs within each state to improve cache
    /// performance during arc iteration.
    ///
    /// # Complexity
    ///
    /// **Time:** O(|V| + |E| log(max_degree)) where max_degree is max arcs from any state
    /// - For each state: collect arcs O(d), sort O(d log d), re-add O(d)
    /// - Total: Σ d log d across all states
    ///
    /// **Space:** O(max_degree) - Temporary arc storage per state
    ///
    /// # Correctness
    ///
    /// **Guarantee:** FST structure preserved, L(T) unchanged
    /// - Only reorders arcs within each state
    /// - All arc properties (labels, weights, targets) preserved
    ///
    /// # Sorting Strategy
    ///
    /// Arcs sorted by:
    /// 1. Target state cache group (states with similar IDs)
    /// 2. Access frequency (more frequent targets first)
    /// 3. Label (for deterministic ordering)
    pub fn optimize_arc_layout<W: Semiring>(fst: &mut VectorFst<W>) {
        use std::collections::HashMap;

        // Analyze arc access patterns
        let mut target_frequency: HashMap<StateId, usize> = HashMap::new();

        // Count how often each state is a target
        for state in 0..fst.num_states() as StateId {
            for arc in fst.arcs(state) {
                *target_frequency.entry(arc.nextstate).or_insert(0) += 1;
            }
        }

        // Process each state
        for state in 0..fst.num_states() as StateId {
            let mut arcs: Vec<_> = fst.arcs(state).collect();

            // Sort arcs by multiple criteria for optimal cache usage:
            // 1. Group by target state locality (states close in ID are likely close in memory)
            // 2. Within groups, sort by frequency of target access
            // 3. As a tiebreaker, sort by label for predictable access
            arcs.sort_by(|a, b| {
                // First, group arcs by target state ranges (cache line grouping)
                const CACHE_GROUP_SIZE: StateId = 8; // States that likely fit in same cache line
                let a_group = a.nextstate / CACHE_GROUP_SIZE;
                let b_group = b.nextstate / CACHE_GROUP_SIZE;

                match a_group.cmp(&b_group) {
                    std::cmp::Ordering::Equal => {
                        // Within same cache group, sort by access frequency
                        let a_freq = target_frequency.get(&a.nextstate).copied().unwrap_or(0);
                        let b_freq = target_frequency.get(&b.nextstate).copied().unwrap_or(0);

                        match b_freq.cmp(&a_freq) {
                            // Higher frequency first
                            std::cmp::Ordering::Equal => {
                                // Finally, sort by label for deterministic ordering
                                match a.ilabel.cmp(&b.ilabel) {
                                    std::cmp::Ordering::Equal => a.olabel.cmp(&b.olabel),
                                    other => other,
                                }
                            }
                            other => other,
                        }
                    }
                    other => other,
                }
            });

            // Delete existing arcs and add sorted ones
            fst.delete_arcs(state);
            for arc in arcs {
                fst.add_arc(state, arc);
            }
        }
    }

    /// Pack small structures to fit cache lines efficiently
    ///
    /// This function analyzes data structures and suggests packing
    /// strategies to minimize cache line waste.
    pub fn analyze_packing<T>(data: &[T]) -> PackingAnalysis {
        let size_of_t = std::mem::size_of::<T>();
        let cache_line_size = 64; // Typical cache line size
        let items_per_line = cache_line_size / size_of_t;
        let waste_per_line = cache_line_size % size_of_t;
        let total_lines = data.len().div_ceil(items_per_line);
        let total_waste = total_lines * waste_per_line;

        PackingAnalysis {
            items_per_cache_line: items_per_line,
            waste_bytes_per_line: waste_per_line,
            total_cache_lines: total_lines,
            total_waste_bytes: total_waste,
            efficiency: 1.0 - (total_waste as f64 / (total_lines * cache_line_size) as f64),
        }
    }
}

/// Analysis of data packing efficiency
#[derive(Debug, Clone)]
pub struct PackingAnalysis {
    /// Number of items that fit in one cache line
    pub items_per_cache_line: usize,
    /// Wasted bytes per cache line
    pub waste_bytes_per_line: usize,
    /// Total cache lines used
    pub total_cache_lines: usize,
    /// Total wasted bytes
    pub total_waste_bytes: usize,
    /// Packing efficiency (0.0 to 1.0)
    pub efficiency: f64,
}

/// Cache-aware iteration utilities
pub mod cache_aware_iteration {
    use super::*;
    use crate::arc::Arc;
    use crate::semiring::Semiring;

    /// Iterator that processes FST states in cache-friendly order
    ///
    /// This iterator reorders state traversal to improve cache locality
    /// by processing states that are likely to share cache lines together.
    #[derive(Debug)]
    pub struct CacheAwareStateIterator {
        states: Vec<StateId>,
        current: usize,
    }

    impl CacheAwareStateIterator {
        /// Create a new cache-aware state iterator
        ///
        /// # Parameters
        /// - `num_states`: Total number of states in the FST
        /// - `metadata`: Cache metadata for optimization hints
        pub fn new(num_states: usize, metadata: &CacheMetadata) -> Self {
            let mut states: Vec<StateId> = (0..num_states as StateId).collect();

            // Reorder states based on access pattern
            match metadata.access_pattern {
                AccessPattern::Sequential => {
                    // Keep sequential order
                }
                AccessPattern::Random => {
                    // Use cache-blocking strategy
                    Self::apply_cache_blocking(&mut states);
                }
                AccessPattern::Clustered => {
                    // Group related states together
                    Self::apply_clustering(&mut states);
                }
                AccessPattern::Sparse => {
                    // Use space-filling curve
                    Self::apply_space_filling_order(&mut states);
                }
            }

            Self { states, current: 0 }
        }

        fn apply_cache_blocking(states: &mut [StateId]) {
            // Reorder states in blocks that fit in cache
            const BLOCK_SIZE: usize = 64; // Cache-friendly block size

            for _chunk in states.chunks_mut(BLOCK_SIZE) {
                // Process each block sequentially
                // In a real implementation, we might sort by some criterion
            }
        }

        fn apply_clustering(states: &mut [StateId]) {
            // Group states that are likely to be accessed together
            // This is a simplified implementation
            states.sort();
        }

        fn apply_space_filling_order(states: &mut [StateId]) {
            // Use space-filling curve for sparse access patterns
            // This is a simplified implementation
            states.reverse();
        }
    }

    impl Iterator for CacheAwareStateIterator {
        type Item = StateId;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current < self.states.len() {
                let state = self.states[self.current];
                self.current += 1;
                Some(state)
            } else {
                None
            }
        }
    }

    /// Process FST with cache-aware iteration
    ///
    /// This function processes an FST using cache-optimized iteration
    /// patterns to improve performance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    /// use arcweight::optimization::cache_aware_iteration::process_cache_aware;
    ///
    /// let fst = VectorFst::<TropicalWeight>::new();
    /// let result = process_cache_aware(&fst, |state, arcs| {
    ///     // Process state and its arcs
    ///     arcs.len()
    /// });
    /// ```
    pub fn process_cache_aware<W, F, R>(fst: &VectorFst<W>, mut processor: F) -> Vec<R>
    where
        W: Semiring,
        F: FnMut(StateId, Vec<Arc<W>>) -> R,
    {
        let metadata = CacheMetadata::analyze(fst);
        let state_iter = CacheAwareStateIterator::new(fst.num_states(), &metadata);

        state_iter
            .map(|state| {
                let arcs: Vec<_> = fst.arcs(state).collect();
                processor(state, arcs)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_cache_metadata_analysis() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Add different numbers of arcs to create variation
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::new(1.5), s2));

        let metadata = CacheMetadata::analyze(&fst);

        assert!(metadata.avg_arcs_per_state > 0.0);
        assert!(!metadata.arc_count_distribution.is_empty());
        assert!(metadata.prefetch_distance > 0);
    }

    #[test]
    fn test_optimization_recommendations() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);

        let metadata = CacheMetadata::analyze(&fst);
        let recommendations = metadata.recommendations();

        // Should get some recommendations for a simple FST
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_access_pattern_classification() {
        // Test with uniform arc distribution (should be Sequential)
        let uniform_counts = vec![2, 2, 2, 2, 2];
        let pattern = CacheMetadata::classify_access_pattern(&uniform_counts);
        assert_eq!(pattern, AccessPattern::Sequential);

        // Test with sparse distribution
        let sparse_counts = vec![0, 1, 0, 1, 0];
        let pattern = CacheMetadata::classify_access_pattern(&sparse_counts);
        assert_eq!(pattern, AccessPattern::Sparse);
    }

    #[test]
    fn test_cache_aware_iteration() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));

        let results =
            cache_aware_iteration::process_cache_aware(&fst, |state, arcs| (state, arcs.len()));

        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|&(state, _)| state == s0));
        assert!(results.iter().any(|&(state, _)| state == s1));
    }

    #[test]
    fn test_packing_analysis() {
        let data = vec![1u32; 100];
        let analysis = layout_optimization::analyze_packing(&data);

        assert!(analysis.items_per_cache_line > 0);
        assert!(analysis.efficiency >= 0.0 && analysis.efficiency <= 1.0);
    }
}
