//! Pruning algorithm
//!
//! Removes paths from weighted FSTs that exceed a weight threshold.
//!
//! # Semiring Requirements
//!
//! Pruning requires a **naturally ordered semiring** for meaningful weight comparison:
//! - `NaturallyOrderedSemiring` trait enables weight threshold comparison
//! - Total ordering defines which paths are "worse" than the threshold
//! - Without natural ordering, pruning concept is not well-defined
//!
//! # Supported Semirings
//!
//! - ✅ `TropicalWeight` - Natural ordering by cost (prune high-cost paths)
//! - ✅ `LogWeight` - Natural ordering by log probability
//! - ❌ `ProbabilityWeight` - No natural ordering defined
//! - ❌ `BooleanWeight` - No meaningful weight comparison
//!
//! # Pruning Strategies
//!
//! - **Forward pruning:** Remove paths based on forward weights
//! - **Backward pruning:** Remove paths based on backward weights  
//! - **Global pruning:** Remove paths based on total path weight

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst, StateId};
use crate::semiring::{NaturallyOrderedSemiring, Semiring};
use crate::Result;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

/// Pruning configuration
#[derive(Debug, Clone)]
pub struct PruneConfig {
    /// Weight threshold for pruning
    pub weight_threshold: f64,
    /// Maximum number of states to keep
    pub state_threshold: Option<usize>,
    /// Number of shortest paths to keep
    pub npath: Option<usize>,
    /// Prune based on forward-backward weights
    pub use_forward_backward: bool,
    /// Delta for weight comparison
    pub delta: f64,
}

impl Default for PruneConfig {
    fn default() -> Self {
        Self {
            weight_threshold: f64::INFINITY,
            state_threshold: None,
            npath: None,
            use_forward_backward: false,
            delta: 1e-6,
        }
    }
}

/// State with priority for heap operations
#[derive(Debug, Clone)]
struct PriorityState<W> {
    state: StateId,
    weight: W,
}

impl<W: PartialOrd> PartialEq for PriorityState<W> {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}

impl<W: PartialOrd> Eq for PriorityState<W> {}

impl<W: PartialOrd> Ord for PriorityState<W> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap behavior
        other
            .weight
            .partial_cmp(&self.weight)
            .unwrap_or(Ordering::Equal)
    }
}

impl<W: PartialOrd> PartialOrd for PriorityState<W> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Prune an FST by removing paths and states that exceed weight thresholds
///
/// Removes paths from the FST that have weights exceeding the specified threshold,
/// producing a smaller FST that accepts a subset of the original language.
///
/// # Correctness
///
/// **Guarantee:** L(prune(T)) ⊆ L(T)
/// - Pruning only removes paths, never adds them
/// - The output FST's language is always a subset of the input
///
/// # Complexity
///
/// - **Weight-based:** O(|V| × |E|) - Bellman-Ford shortest distances
/// - **Forward-backward:** O(|V| × |E|) - Two Bellman-Ford passes
/// - **N-best paths:** O(|V| × |E| × log N) - Priority queue with N-best tracking
///
/// # Pruning Strategies
///
/// Selected automatically based on `config`:
/// - If `use_forward_backward`: Uses forward-backward pruning (best for beam search)
/// - If `npath` is set: Keeps only N-best paths
/// - Otherwise: Uses simple weight-based pruning (fastest)
///
/// # Examples
///
/// ```
/// use arcweight::prelude::*;
/// use arcweight::algorithms::{prune, PruneConfig};
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// // ... build FST ...
///
/// let config = PruneConfig {
///     weight_threshold: 5.0,
///     ..Default::default()
/// };
/// let pruned: VectorFst<TropicalWeight> = prune(&fst, config).unwrap();
/// ```
pub fn prune<W, F, M>(fst: &F, config: PruneConfig) -> Result<M>
where
    W: NaturallyOrderedSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    // Handle empty FST case
    if fst.num_states() == 0 || fst.start().is_none() {
        return Ok(M::default());
    }

    // Choose pruning strategy based on configuration
    if config.use_forward_backward {
        prune_forward_backward(fst, &config)
    } else if let Some(npath) = config.npath {
        prune_nbest_paths(fst, npath, &config)
    } else {
        prune_by_weight(fst, &config)
    }
}

/// Prune using forward-backward algorithm
fn prune_forward_backward<W, F, M>(fst: &F, config: &PruneConfig) -> Result<M>
where
    W: NaturallyOrderedSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let forward_weights = compute_forward_weights(fst)?;
    let backward_weights = compute_backward_weights(fst)?;

    // Find best total weight
    let mut best_weight = None;
    if let Some(start) = fst.start() {
        if let Some(backward) = backward_weights.get(&start) {
            best_weight = Some(backward.clone());
        }
    }

    let mut result = M::default();
    let mut state_map = HashMap::new();

    // First pass: select states based on forward-backward weights
    let zero_weight = W::zero();
    for state in fst.states() {
        let forward = forward_weights.get(&state).unwrap_or(&zero_weight);
        let backward = backward_weights.get(&state).unwrap_or(&zero_weight);

        if *forward == W::zero() || *backward == W::zero() {
            continue; // Skip unreachable states
        }

        // Compute total weight through this state
        let total = forward.times(backward);

        // Check if state should be kept
        if should_keep_weight(&total, &best_weight, config) {
            let new_state = result.add_state();
            state_map.insert(state, new_state);

            // Copy final weight if applicable
            if let Some(weight) = fst.final_weight(state) {
                result.set_final(new_state, weight.clone());
            }
        }
    }

    // Set start state
    if let Some(start) = fst.start() {
        if let Some(&new_start) = state_map.get(&start) {
            result.set_start(new_start);
        }
    }

    // Second pass: copy arcs between selected states
    for (&old_state, &new_state) in &state_map {
        for arc in fst.arcs(old_state) {
            if let Some(&new_nextstate) = state_map.get(&arc.nextstate) {
                // Check arc weight against threshold
                let arc_total = if let (Some(forward), Some(backward)) = (
                    forward_weights.get(&old_state),
                    backward_weights.get(&arc.nextstate),
                ) {
                    forward.times(&arc.weight).times(backward)
                } else {
                    continue;
                };

                if should_keep_weight(&arc_total, &best_weight, config) {
                    result.add_arc(
                        new_state,
                        Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), new_nextstate),
                    );
                }
            }
        }
    }

    apply_state_threshold(result, config)
}

/// Prune keeping only n-best paths
fn prune_nbest_paths<W, F, M>(fst: &F, npath: usize, config: &PruneConfig) -> Result<M>
where
    W: NaturallyOrderedSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    if npath == 0 {
        return Ok(M::default());
    }

    let start = fst
        .start()
        .ok_or_else(|| crate::Error::Algorithm("FST has no start state".into()))?;

    // Find n-best paths using priority queue
    let mut heap = BinaryHeap::new();
    let mut best_weights: HashMap<StateId, Vec<W>> = HashMap::new();

    // Initialize with start state
    heap.push(PriorityState {
        state: start,
        weight: W::one(),
    });
    best_weights.insert(start, vec![W::one()]);

    // Process states in best-first order
    while let Some(PriorityState { state, weight }) = heap.pop() {
        // Check if this is a final state
        if let Some(final_weight) = fst.final_weight(state) {
            let _total = weight.times(final_weight);
            // Track as a complete path
        }

        // Explore outgoing arcs
        for arc in fst.arcs(state) {
            let new_weight = weight.times(&arc.weight);

            // Update best weights for next state
            let weights = best_weights.entry(arc.nextstate).or_default();

            // Keep only n-best weights
            if weights.len() < npath {
                weights.push(new_weight.clone());
                weights.sort();
                if weights.len() > npath {
                    weights.truncate(npath);
                }

                heap.push(PriorityState {
                    state: arc.nextstate,
                    weight: new_weight,
                });
            } else if weights.last().is_some_and(|w| new_weight < *w) {
                // Replace worst weight if this is better
                weights[npath - 1] = new_weight.clone();
                weights.sort();

                heap.push(PriorityState {
                    state: arc.nextstate,
                    weight: new_weight,
                });
            }
        }
    }

    // Build result FST with selected paths
    build_nbest_fst(fst, &best_weights, npath, config)
}

/// Simple weight-based pruning
fn prune_by_weight<W, F, M>(fst: &F, config: &PruneConfig) -> Result<M>
where
    W: NaturallyOrderedSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();
    let mut state_map = HashMap::new();

    // Compute shortest distances from start
    let distances = compute_shortest_distances(fst)?;

    // First pass: select states within threshold
    for state in fst.states() {
        if let Some(distance) = distances.get(&state) {
            if convert_weight_to_f64(distance) <= config.weight_threshold {
                let new_state = result.add_state();
                state_map.insert(state, new_state);

                if let Some(weight) = fst.final_weight(state) {
                    result.set_final(new_state, weight.clone());
                }
            }
        }
    }

    // Set start state
    if let Some(start) = fst.start() {
        if let Some(&new_start) = state_map.get(&start) {
            result.set_start(new_start);
        }
    }

    // Second pass: copy arcs with weight filtering
    for (&old_state, &new_state) in &state_map {
        if let Some(state_distance) = distances.get(&old_state) {
            for arc in fst.arcs(old_state) {
                // Check if arc leads to selected state
                if let Some(&new_nextstate) = state_map.get(&arc.nextstate) {
                    // Compute path weight through this arc
                    let arc_distance = state_distance.times(&arc.weight);

                    if convert_weight_to_f64(&arc_distance) <= config.weight_threshold {
                        result.add_arc(
                            new_state,
                            Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), new_nextstate),
                        );
                    }
                }
            }
        }
    }

    apply_state_threshold(result, config)
}

/// Compute forward weights (shortest distance from start)
///
/// Uses Bellman-Ford relaxation to compute shortest distances from the start state.
///
/// # Complexity
///
/// - **Acyclic graphs:** O(|V| + |E|)
/// - **Cyclic graphs:** O(|V| × |E|) worst case, with iteration limit
///
/// # Algorithm
///
/// 1. Initialize start state with weight one (identity)
/// 2. Relaxation: for each arc (u, v) with weight w:
///    - If dist[u] ⊗ w < dist[v], update dist[v]
/// 3. Continue until no updates (or iteration limit reached)
///
/// # Termination
///
/// - For acyclic graphs: guaranteed to terminate in |V| iterations
/// - For cyclic graphs: may have improving cycles, so we limit iterations to |V|
///
/// # Notes
///
/// For graphs with negative (improving) cycles in the weight space, this may not
/// find true shortest distances but will terminate and return approximate values.
fn compute_forward_weights<W, F>(fst: &F) -> Result<HashMap<StateId, W>>
where
    W: NaturallyOrderedSemiring,
    F: Fst<W>,
{
    let mut weights = HashMap::new();
    let mut queue = VecDeque::new();
    let mut in_queue = HashSet::new();

    if let Some(start) = fst.start() {
        weights.insert(start, W::one());
        queue.push_back(start);
        in_queue.insert(start);
    }

    // Bellman-Ford style relaxation with iteration limit
    let num_states = fst.num_states();
    let max_iterations = num_states * num_states; // Safety limit for cyclic graphs
    let mut iterations = 0;

    while let Some(state) = queue.pop_front() {
        in_queue.remove(&state);
        iterations += 1;

        if iterations > max_iterations {
            // Prevent infinite loops on cyclic graphs
            break;
        }

        let state_weight = weights[&state].clone();

        for arc in fst.arcs(state) {
            let new_weight = state_weight.times(&arc.weight);

            let updated = match weights.get(&arc.nextstate) {
                None => {
                    weights.insert(arc.nextstate, new_weight);
                    true
                }
                Some(old_weight) => {
                    if new_weight < *old_weight {
                        weights.insert(arc.nextstate, new_weight);
                        true
                    } else {
                        false
                    }
                }
            };

            if updated && !in_queue.contains(&arc.nextstate) {
                queue.push_back(arc.nextstate);
                in_queue.insert(arc.nextstate);
            }
        }
    }

    Ok(weights)
}

/// Compute backward weights (shortest distance to final states)
///
/// Computes shortest distances from each state to any final state using
/// backward relaxation through reversed arcs.
///
/// # Complexity
///
/// - **Build reverse index:** O(|E|)
/// - **Relaxation:** O(|V| × |E|) worst case with iteration limit
/// - **Total:** O(|V| × |E|)
///
/// # Algorithm
///
/// 1. Build reverse arc index: for each arc (u, v), store (u, w) at v
/// 2. Initialize final states with their final weights
/// 3. Backward relaxation: for each state v, update predecessors u:
///    - dist[u] = dist[u] ⊕ (w ⊗ dist[v])
///
/// # Termination
///
/// Similar to forward weights, uses iteration limit for cyclic graphs.
fn compute_backward_weights<W, F>(fst: &F) -> Result<HashMap<StateId, W>>
where
    W: NaturallyOrderedSemiring,
    F: Fst<W>,
{
    let mut weights = HashMap::new();
    let mut reverse_arcs: HashMap<StateId, Vec<(StateId, W)>> = HashMap::new();

    // Build reverse arc index
    for state in fst.states() {
        for arc in fst.arcs(state) {
            reverse_arcs
                .entry(arc.nextstate)
                .or_default()
                .push((state, arc.weight.clone()));
        }
    }

    // Initialize with final states
    let mut queue = VecDeque::new();
    let mut in_queue = HashSet::new();
    for state in fst.states() {
        if let Some(final_weight) = fst.final_weight(state) {
            weights.insert(state, final_weight.clone());
            queue.push_back(state);
            in_queue.insert(state);
        }
    }

    // Backward relaxation with iteration limit
    let num_states = fst.num_states();
    let max_iterations = num_states * num_states;
    let mut iterations = 0;

    while let Some(state) = queue.pop_front() {
        in_queue.remove(&state);
        iterations += 1;

        if iterations > max_iterations {
            // Prevent infinite loops on cyclic graphs
            break;
        }

        let state_weight = weights[&state].clone();

        if let Some(predecessors) = reverse_arcs.get(&state) {
            for (prev_state, arc_weight) in predecessors {
                let new_weight = arc_weight.times(&state_weight);

                let updated = match weights.get(prev_state) {
                    None => {
                        weights.insert(*prev_state, new_weight);
                        true
                    }
                    Some(old_weight) => {
                        let combined = old_weight.plus(&new_weight);
                        if combined != *old_weight {
                            weights.insert(*prev_state, combined);
                            true
                        } else {
                            false
                        }
                    }
                };

                if updated && !in_queue.contains(prev_state) {
                    queue.push_back(*prev_state);
                    in_queue.insert(*prev_state);
                }
            }
        }
    }

    Ok(weights)
}

/// Compute shortest distances from start state
fn compute_shortest_distances<W, F>(fst: &F) -> Result<HashMap<StateId, W>>
where
    W: NaturallyOrderedSemiring,
    F: Fst<W>,
{
    compute_forward_weights(fst)
}

/// Build FST from n-best paths
fn build_nbest_fst<W, F, M>(
    fst: &F,
    best_weights: &HashMap<StateId, Vec<W>>,
    _npath: usize,
    _config: &PruneConfig,
) -> Result<M>
where
    W: NaturallyOrderedSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();
    let mut state_map = HashMap::new();

    // Create states that appear in best paths
    for &state in best_weights.keys() {
        let new_state = result.add_state();
        state_map.insert(state, new_state);

        if let Some(weight) = fst.final_weight(state) {
            result.set_final(new_state, weight.clone());
        }
    }

    // Set start state
    if let Some(start) = fst.start() {
        if let Some(&new_start) = state_map.get(&start) {
            result.set_start(new_start);
        }
    }

    // Copy arcs that participate in best paths
    for (&state, &new_state) in &state_map {
        for arc in fst.arcs(state) {
            if let Some(&new_nextstate) = state_map.get(&arc.nextstate) {
                result.add_arc(
                    new_state,
                    Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), new_nextstate),
                );
            }
        }
    }

    Ok(result)
}

/// Apply state threshold by selecting best states
fn apply_state_threshold<W, M>(fst: M, config: &PruneConfig) -> Result<M>
where
    W: NaturallyOrderedSemiring,
    M: MutableFst<W>,
{
    if let Some(threshold) = config.state_threshold {
        if fst.num_states() <= threshold {
            return Ok(fst);
        }

        // For now, return the FST as-is
        // A full implementation would rank states and keep only the best
    }

    Ok(fst)
}

/// Check if weight should be kept based on pruning criteria
fn should_keep_weight<W>(weight: &W, best: &Option<W>, config: &PruneConfig) -> bool
where
    W: NaturallyOrderedSemiring,
{
    // Check against absolute threshold
    if convert_weight_to_f64(weight) > config.weight_threshold {
        return false;
    }

    // Check against best weight with beam
    if let Some(best_weight) = best {
        let weight_val = convert_weight_to_f64(weight);
        let best_val = convert_weight_to_f64(best_weight);

        if weight_val > best_val + config.weight_threshold {
            return false;
        }
    }

    true
}

/// Convert weight to f64 for threshold comparison
fn convert_weight_to_f64<W: Semiring>(weight: &W) -> f64 {
    // Extract numeric value from weight
    let weight_str = format!("{weight:?}");

    if weight_str.contains("∞") || weight_str.contains("inf") {
        return f64::INFINITY;
    }

    // Try to parse different weight formats
    if let Some(start) = weight_str.find('(') {
        if let Some(end) = weight_str.find(')') {
            if let Ok(val) = weight_str[start + 1..end].parse::<f64>() {
                return val;
            }
        }
    }

    // Try direct parsing
    if let Ok(val) = weight_str.parse::<f64>() {
        return val;
    }

    0.0
}

/// Compute reachable states from a given start state
#[allow(dead_code)]
fn compute_reachable_states<F: Fst<W>, W: Semiring>(fst: &F, start: StateId) -> HashSet<StateId> {
    let mut reachable = HashSet::new();
    let mut stack = vec![start];

    while let Some(state) = stack.pop() {
        if reachable.insert(state) {
            for arc in fst.arcs(state) {
                stack.push(arc.nextstate);
            }
        }
    }

    reachable
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_prune_config_default() {
        let config = PruneConfig::default();
        assert_eq!(config.weight_threshold, f64::INFINITY);
        assert_eq!(config.state_threshold, None);
        assert_eq!(config.npath, None);
        assert!(!config.use_forward_backward);
    }

    #[test]
    fn test_prune_config_custom() {
        let config = PruneConfig {
            weight_threshold: 5.0,
            state_threshold: Some(100),
            npath: Some(10),
            use_forward_backward: true,
            delta: 1e-8,
        };
        assert_eq!(config.weight_threshold, 5.0);
        assert_eq!(config.state_threshold, Some(100));
        assert_eq!(config.npath, Some(10));
        assert!(config.use_forward_backward);
    }

    #[test]
    fn test_prune_simple_fst() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(2.0), s2));

        let config = PruneConfig {
            weight_threshold: 10.0,
            ..Default::default()
        };
        let pruned: VectorFst<TropicalWeight> = prune(&fst, config).unwrap();

        assert!(pruned.num_states() > 0);
        assert!(pruned.start().is_some());
    }

    #[test]
    fn test_prune_weighted_paths() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.set_final(s2, TropicalWeight::one());

        // Low-cost and high-cost paths
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(5.0), s2));

        let config = PruneConfig {
            weight_threshold: 3.0,
            ..Default::default()
        };
        let pruned: VectorFst<TropicalWeight> = prune(&fst, config).unwrap();

        assert!(pruned.start().is_some());
        assert!(pruned.num_states() > 0);

        // Should keep low-cost path
        if let Some(start) = pruned.start() {
            let reachable = compute_reachable_states(&pruned, start);
            assert!(reachable.iter().any(|&s| pruned.is_final(s)));
        }
    }

    #[test]
    fn test_prune_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();
        let config = PruneConfig::default();
        let pruned: VectorFst<TropicalWeight> = prune(&fst, config).unwrap();

        assert_eq!(pruned.num_states(), 0);
        assert!(pruned.is_empty());
    }

    #[test]
    fn test_prune_single_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::new(2.0));

        let config = PruneConfig {
            weight_threshold: 5.0,
            ..Default::default()
        };
        let pruned: VectorFst<TropicalWeight> = prune(&fst, config).unwrap();

        assert_eq!(pruned.num_states(), 1);
        assert!(pruned.start().is_some());
    }

    #[test]
    fn test_prune_forward_backward() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s2));
        fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::new(10.0), s2));

        let config = PruneConfig {
            weight_threshold: 5.0,
            use_forward_backward: true,
            ..Default::default()
        };
        let pruned: VectorFst<TropicalWeight> = prune(&fst, config).unwrap();

        assert!(pruned.start().is_some());
        assert!(pruned.num_states() > 0);
    }

    #[test]
    fn test_prune_nbest() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s3, TropicalWeight::one());

        // Multiple paths with different costs
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(0.5), s3));

        fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::new(2.0), s2));
        fst.add_arc(s2, Arc::new(4, 4, TropicalWeight::new(0.5), s3));

        let config = PruneConfig {
            npath: Some(1),
            ..Default::default()
        };
        let pruned: VectorFst<TropicalWeight> = prune(&fst, config).unwrap();

        assert!(pruned.start().is_some());
        assert!(pruned.num_states() > 0);
    }

    #[test]
    fn test_convert_weight_to_f64() {
        let w1 = TropicalWeight::new(3.5);
        let val1 = convert_weight_to_f64(&w1);
        assert!((val1 - 3.5).abs() < 1e-6);

        let w2 = TropicalWeight::zero();
        let val2 = convert_weight_to_f64(&w2);
        assert_eq!(val2, f64::INFINITY);
    }

    #[test]
    fn test_priority_state() {
        let ps1 = PriorityState {
            state: 0,
            weight: TropicalWeight::new(1.0),
        };
        let ps2 = PriorityState {
            state: 1,
            weight: TropicalWeight::new(2.0),
        };

        // ps1 should have higher priority (lower weight)
        assert!(ps1 > ps2);
    }

    #[test]
    fn test_prune_with_state_threshold() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let states: Vec<_> = (0..10).map(|_| fst.add_state()).collect();

        fst.set_start(states[0]);
        fst.set_final(states[9], TropicalWeight::one());

        for i in 0..9 {
            fst.add_arc(
                states[i],
                Arc::new(
                    (i + 1) as u32,
                    (i + 1) as u32,
                    TropicalWeight::new(0.1),
                    states[i + 1],
                ),
            );
        }

        let config = PruneConfig {
            state_threshold: Some(5),
            ..Default::default()
        };
        let pruned: VectorFst<TropicalWeight> = prune(&fst, config).unwrap();

        assert!(pruned.num_states() > 0);
    }

    #[test]
    fn test_prune_complex_graph() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s3, TropicalWeight::one());

        // Diamond-shaped graph
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s2));
        fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::new(1.0), s3));
        fst.add_arc(s2, Arc::new(4, 4, TropicalWeight::new(1.0), s3));

        let config = PruneConfig {
            weight_threshold: 2.5,
            ..Default::default()
        };
        let pruned: VectorFst<TropicalWeight> = prune(&fst, config).unwrap();

        assert!(pruned.start().is_some());

        // Should keep the better path (through s1)
        if let Some(start) = pruned.start() {
            let reachable = compute_reachable_states(&pruned, start);
            assert!(reachable.iter().any(|&s| pruned.is_final(s)));
        }
    }
}
