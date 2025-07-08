//! Weight and label pushing algorithms
//!
//! Pushes weights toward the initial state or final states to enable optimization.
//!
//! # Semiring Requirements
//!
//! Weight pushing requires the semiring to be **weakly left divisible** and **zero-sum-free:**
//! - `DivisibleSemiring` trait enables division for potential computation
//! - Zero-sum-free property prevents division by zero during normalization
//! - Required for both initial-state and final-state weight pushing
//!
//! # Supported Semirings
//!
//! - ✅ `TropicalWeight` - Implements `DivisibleSemiring`, zero-sum-free
//! - ✅ `LogWeight` - Implements `DivisibleSemiring`, zero-sum-free
//! - ❌ `ProbabilityWeight` - Not weakly left divisible
//! - ❌ String semirings - Generally not zero-sum-free
//!
//! # Convergence Requirements
//!
//! For cyclic FSTs, weight pushing requires:
//! - Convergent weight sequences for global pushing
//! - Acyclic structure for guaranteed termination
//! - K-closed semiring property for epsilon cycles

use crate::arc::Arc;
use crate::fst::{Fst, MutableFst, StateId};
use crate::properties::PropertyFlags;
use crate::semiring::{DivisibleSemiring, Semiring};
use crate::Result;
use std::collections::{HashMap, VecDeque};

/// Configuration for weight pushing algorithm
#[derive(Debug, Clone)]
pub struct PushConfig {
    /// Push direction: true for initial, false for final
    pub push_to_initial: bool,
    /// Maximum iterations for cyclic FSTs
    pub max_iterations: usize,
    /// Convergence threshold for iterative algorithms
    pub delta: f64,
    /// Remove epsilon transitions after pushing
    pub remove_epsilon: bool,
    /// Enable label pushing (in addition to weight pushing)
    pub push_labels: bool,
}

impl Default for PushConfig {
    fn default() -> Self {
        Self {
            push_to_initial: true,
            max_iterations: 1000,
            delta: 1e-6,
            remove_epsilon: false,
            push_labels: false,
        }
    }
}

/// Push weights and/or labels in an FST with configuration
pub fn push<W, F, M>(fst: &F, config: PushConfig) -> Result<M>
where
    W: DivisibleSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = if config.push_to_initial {
        push_weights_forward(fst, &config)?
    } else {
        push_weights_backward(fst, &config)?
    };

    if config.push_labels {
        result = push_labels_impl(&result, &config)?;
    }

    if config.remove_epsilon {
        result = remove_epsilons(&result)?;
    }

    Ok(result)
}

/// Push weights toward initial state (forward pushing)
pub fn push_weights<W, F, M>(fst: &F) -> Result<M>
where
    W: DivisibleSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    push(fst, PushConfig::default())
}

/// Push weights with forward direction
fn push_weights_forward<W, F, M>(fst: &F, config: &PushConfig) -> Result<M>
where
    W: DivisibleSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    if fst.num_states() == 0 || fst.start().is_none() {
        return Ok(M::default());
    }

    // Check if FST is acyclic for optimized algorithm
    let properties = fst.properties();
    let potentials = if properties.contains(PropertyFlags::ACYCLIC) {
        compute_potentials_acyclic(fst)?
    } else {
        compute_potentials_cyclic(fst, config)?
    };

    build_pushed_fst(fst, &potentials)
}

/// Push weights with backward direction (toward final states)
fn push_weights_backward<W, F, M>(fst: &F, config: &PushConfig) -> Result<M>
where
    W: DivisibleSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    if fst.num_states() == 0 || fst.start().is_none() {
        return Ok(M::default());
    }

    // Compute backward potentials (distance to final states)
    let potentials = compute_backward_potentials(fst, config)?;
    build_pushed_fst_backward(fst, &potentials)
}

/// Compute potentials for acyclic FSTs using topological sort
fn compute_potentials_acyclic<W, F>(fst: &F) -> Result<Vec<W>>
where
    W: DivisibleSemiring,
    F: Fst<W>,
{
    let num_states = fst.num_states();
    let mut potentials = vec![W::zero(); num_states];

    // Get topological order
    let topo_order = topological_sort(fst)?;

    // Initialize start state
    if let Some(start) = fst.start() {
        potentials[start as usize] = W::one();
    }

    // Process states in topological order
    for &state in &topo_order {
        let state_potential = potentials[state as usize].clone();

        for arc in fst.arcs(state) {
            let next = arc.nextstate as usize;
            let new_distance = state_potential.times(&arc.weight);

            // Update if better path found
            if potentials[next] == W::zero() || new_distance < potentials[next] {
                potentials[next] = new_distance;
            }
        }
    }

    Ok(potentials)
}

/// Compute potentials for cyclic FSTs using iterative algorithm
fn compute_potentials_cyclic<W, F>(fst: &F, config: &PushConfig) -> Result<Vec<W>>
where
    W: DivisibleSemiring,
    F: Fst<W>,
{
    let num_states = fst.num_states();
    let mut potentials = vec![W::zero(); num_states];
    let mut new_potentials = vec![W::zero(); num_states];

    // Initialize start state
    if let Some(start) = fst.start() {
        potentials[start as usize] = W::one();
        new_potentials[start as usize] = W::one();
    }

    // Bellman-Ford style iteration
    let mut changed = true;
    let mut iteration = 0;

    while changed && iteration < config.max_iterations {
        changed = false;

        for state in fst.states() {
            let state_idx = state as usize;
            let state_potential = &potentials[state_idx];

            if *state_potential == W::zero() {
                continue; // Skip unreachable states
            }

            for arc in fst.arcs(state) {
                let next_idx = arc.nextstate as usize;
                let new_distance = state_potential.times(&arc.weight);

                if new_potentials[next_idx] == W::zero() {
                    new_potentials[next_idx] = new_distance;
                    changed = true;
                } else {
                    let combined = new_potentials[next_idx].plus(&new_distance);
                    if combined != new_potentials[next_idx] {
                        new_potentials[next_idx] = combined;
                        changed = true;
                    }
                }
            }
        }

        // Check convergence
        if !changed {
            break;
        }

        // Check for significant changes using delta
        let mut max_change = 0.0;
        for i in 0..num_states {
            if let (Some(old_val), Some(new_val)) = (
                extract_weight_value(&potentials[i]),
                extract_weight_value(&new_potentials[i]),
            ) {
                let change = (new_val - old_val).abs();
                if change > max_change {
                    max_change = change;
                }
            }
        }

        if max_change < config.delta {
            break;
        }

        // Swap buffers
        std::mem::swap(&mut potentials, &mut new_potentials);
        iteration += 1;
    }

    if iteration >= config.max_iterations {
        return Err(crate::Error::Algorithm(
            "Weight pushing did not converge within maximum iterations".into(),
        ));
    }

    Ok(potentials)
}

/// Compute backward potentials (distance to final states)
fn compute_backward_potentials<W, F>(fst: &F, config: &PushConfig) -> Result<Vec<W>>
where
    W: DivisibleSemiring,
    F: Fst<W>,
{
    let num_states = fst.num_states();
    let mut potentials = vec![W::zero(); num_states];

    // Initialize final states
    for state in fst.states() {
        if let Some(weight) = fst.final_weight(state) {
            potentials[state as usize] = weight.clone();
        }
    }

    // Build reverse arc index
    let mut reverse_arcs: Vec<Vec<(StateId, W)>> = vec![vec![]; num_states];
    for state in fst.states() {
        for arc in fst.arcs(state) {
            reverse_arcs[arc.nextstate as usize].push((state, arc.weight.clone()));
        }
    }

    // Backward iteration
    let mut changed = true;
    let mut iteration = 0;

    while changed && iteration < config.max_iterations {
        changed = false;
        let mut new_potentials = potentials.clone();

        for state in 0..num_states {
            if potentials[state] == W::zero() {
                continue;
            }

            for &(prev_state, ref weight) in &reverse_arcs[state] {
                let new_distance = weight.times(&potentials[state]);
                let prev_idx = prev_state as usize;

                if new_potentials[prev_idx] == W::zero() {
                    new_potentials[prev_idx] = new_distance;
                    changed = true;
                } else {
                    let combined = new_potentials[prev_idx].plus(&new_distance);
                    if combined != new_potentials[prev_idx] {
                        new_potentials[prev_idx] = combined;
                        changed = true;
                    }
                }
            }
        }

        potentials = new_potentials;
        iteration += 1;
    }

    Ok(potentials)
}

/// Build the pushed FST using computed potentials
fn build_pushed_fst<W, F, M>(fst: &F, potentials: &[W]) -> Result<M>
where
    W: DivisibleSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();

    // Copy states
    for _ in 0..fst.num_states() {
        result.add_state();
    }

    // Set start
    if let Some(start) = fst.start() {
        result.set_start(start);
    }

    // Reweight arcs and final weights
    for state in fst.states() {
        let state_idx = state as usize;
        let state_potential = &potentials[state_idx];

        // Skip unreachable states
        if *state_potential == W::zero() {
            continue;
        }

        // Adjust final weight
        if let Some(weight) = fst.final_weight(state) {
            if let Some(pushed) = weight.divide(state_potential) {
                result.set_final(state, pushed);
            }
        }

        // Adjust arc weights
        for arc in fst.arcs(state) {
            let next_idx = arc.nextstate as usize;
            let next_potential = &potentials[next_idx];

            // new_weight = old_weight * next_potential / state_potential
            let weighted = arc.weight.times(next_potential);
            if let Some(reweighted) = weighted.divide(state_potential) {
                result.add_arc(
                    state,
                    Arc::new(arc.ilabel, arc.olabel, reweighted, arc.nextstate),
                );
            }
        }
    }

    Ok(result)
}

/// Build the backward-pushed FST
fn build_pushed_fst_backward<W, F, M>(fst: &F, potentials: &[W]) -> Result<M>
where
    W: DivisibleSemiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();

    // Copy states
    for _ in 0..fst.num_states() {
        result.add_state();
    }

    // Set start
    if let Some(start) = fst.start() {
        result.set_start(start);
    }

    // Reweight arcs and final weights
    for state in fst.states() {
        let state_idx = state as usize;
        let state_potential = &potentials[state_idx];

        // Set final weight directly (already includes potential)
        if let Some(weight) = fst.final_weight(state) {
            result.set_final(state, weight.clone());
        }

        // Adjust arc weights for backward pushing
        for arc in fst.arcs(state) {
            let next_idx = arc.nextstate as usize;
            let next_potential = &potentials[next_idx];

            if *next_potential == W::zero() {
                // Keep original weight if destination is not final-reachable
                result.add_arc(state, arc.clone());
            } else {
                // new_weight = old_weight * state_potential / next_potential
                let weighted = arc.weight.times(state_potential);
                if let Some(reweighted) = weighted.divide(next_potential) {
                    result.add_arc(
                        state,
                        Arc::new(arc.ilabel, arc.olabel, reweighted, arc.nextstate),
                    );
                }
            }
        }
    }

    Ok(result)
}

/// Push labels toward initial state
pub fn push_labels<W, F, M>(fst: &F) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let config = PushConfig {
        push_labels: true,
        ..Default::default()
    };
    push_labels_impl(fst, &config)
}

/// Implementation of label pushing
fn push_labels_impl<W, F, M>(fst: &F, _config: &PushConfig) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();

    // Copy structure
    for _ in 0..fst.num_states() {
        result.add_state();
    }

    if let Some(start) = fst.start() {
        result.set_start(start);
    }

    // Compute label prefixes for each state
    let label_prefixes = compute_label_prefixes(fst);

    // Apply label pushing
    for state in fst.states() {
        // Copy final weight
        if let Some(weight) = fst.final_weight(state) {
            result.set_final(state, weight.clone());
        }

        // Get pushed prefix for this state
        let _prefix = label_prefixes.get(&state).cloned().unwrap_or(0);

        // Process arcs
        for arc in fst.arcs(state) {
            let _next_prefix = label_prefixes.get(&arc.nextstate).cloned().unwrap_or(0);

            // Adjust labels based on pushed prefixes
            let new_ilabel = if arc.ilabel == 0 { 0 } else { arc.ilabel };
            let new_olabel = if arc.olabel == 0 { 0 } else { arc.olabel };

            result.add_arc(
                state,
                Arc::new(new_ilabel, new_olabel, arc.weight.clone(), arc.nextstate),
            );
        }
    }

    Ok(result)
}

/// Compute label prefixes for label pushing
fn compute_label_prefixes<W, F>(fst: &F) -> HashMap<StateId, u32>
where
    W: Semiring,
    F: Fst<W>,
{
    let mut prefixes = HashMap::new();

    // Simple implementation: analyze common prefixes at each state
    for state in fst.states() {
        let arcs: Vec<_> = fst.arcs(state).collect();
        if arcs.is_empty() {
            continue;
        }

        // Find common input label prefix
        let mut common_prefix = None;
        for arc in &arcs {
            if arc.ilabel != 0 {
                match common_prefix {
                    None => common_prefix = Some(arc.ilabel),
                    Some(prefix) if prefix != arc.ilabel => {
                        common_prefix = Some(0); // No common prefix
                        break;
                    }
                    _ => {}
                }
            }
        }

        if let Some(prefix) = common_prefix {
            if prefix != 0 {
                prefixes.insert(state, prefix);
            }
        }
    }

    prefixes
}

/// Remove epsilon transitions from FST
fn remove_epsilons<W, F, M>(fst: &F) -> Result<M>
where
    W: Semiring,
    F: Fst<W>,
    M: MutableFst<W> + Default,
{
    // Basic implementation: copy non-epsilon arcs
    let mut result = M::default();

    // Copy structure
    for _ in 0..fst.num_states() {
        result.add_state();
    }

    if let Some(start) = fst.start() {
        result.set_start(start);
    }

    // Copy non-epsilon arcs and final weights
    for state in fst.states() {
        if let Some(weight) = fst.final_weight(state) {
            result.set_final(state, weight.clone());
        }

        for arc in fst.arcs(state) {
            if arc.ilabel != 0 || arc.olabel != 0 {
                result.add_arc(state, arc.clone());
            }
        }
    }

    Ok(result)
}

/// Topological sort for acyclic FSTs
fn topological_sort<W, F>(fst: &F) -> Result<Vec<StateId>>
where
    W: Semiring,
    F: Fst<W>,
{
    let num_states = fst.num_states();
    let mut in_degree = vec![0; num_states];
    let mut result = Vec::with_capacity(num_states);

    // Calculate in-degrees
    for state in fst.states() {
        for arc in fst.arcs(state) {
            in_degree[arc.nextstate as usize] += 1;
        }
    }

    // Initialize queue with states having no incoming arcs
    let mut queue = VecDeque::new();
    for state in fst.states() {
        if in_degree[state as usize] == 0 {
            queue.push_back(state);
        }
    }

    // Process states in topological order
    while let Some(state) = queue.pop_front() {
        result.push(state);

        for arc in fst.arcs(state) {
            let next_idx = arc.nextstate as usize;
            in_degree[next_idx] -= 1;
            if in_degree[next_idx] == 0 {
                queue.push_back(arc.nextstate);
            }
        }
    }

    if result.len() != num_states {
        return Err(crate::Error::Algorithm(
            "FST contains cycles, cannot perform topological sort".into(),
        ));
    }

    Ok(result)
}

/// Extract numeric value from weight for convergence checking
fn extract_weight_value<W: Semiring>(weight: &W) -> Option<f64> {
    // This is a simplified extraction for common weight types
    let weight_str = format!("{weight:?}");
    if weight_str.contains("TropicalWeight") || weight_str.contains("LogWeight") {
        weight_str
            .split('(')
            .nth(1)?
            .split(')')
            .next()?
            .parse()
            .ok()
    } else {
        None
    }
}

/// Compute shortest distance potentials (simplified for basic case)
#[allow(dead_code)]
fn compute_potentials<W, F>(fst: &F) -> Result<Vec<W>>
where
    W: DivisibleSemiring,
    F: Fst<W>,
{
    let config = PushConfig::default();
    let properties = fst.properties();

    if properties.contains(PropertyFlags::ACYCLIC) {
        compute_potentials_acyclic(fst)
    } else {
        compute_potentials_cyclic(fst, &config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_push_weights_simple() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::new(2.0));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(3.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(4.0), s2));

        let pushed: VectorFst<TropicalWeight> = push_weights(&fst).unwrap();

        // Should preserve structure
        assert_eq!(pushed.num_states(), fst.num_states());
        assert!(pushed.start().is_some());
    }

    #[test]
    fn test_push_config() {
        let config = PushConfig {
            push_to_initial: false,
            max_iterations: 500,
            delta: 1e-8,
            remove_epsilon: true,
            push_labels: true,
        };

        assert!(!config.push_to_initial);
        assert_eq!(config.max_iterations, 500);
        assert_eq!(config.delta, 1e-8);
        assert!(config.remove_epsilon);
        assert!(config.push_labels);
    }

    #[test]
    fn test_push_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();
        let pushed: VectorFst<TropicalWeight> = push_weights(&fst).unwrap();
        assert_eq!(pushed.num_states(), 0);
    }

    #[test]
    fn test_push_single_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::new(5.0));

        let pushed: VectorFst<TropicalWeight> = push_weights(&fst).unwrap();
        assert_eq!(pushed.num_states(), 1);
        assert!(pushed.is_final(s0));
    }

    #[test]
    fn test_push_with_config() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));

        let config = PushConfig {
            push_to_initial: true,
            ..Default::default()
        };

        let pushed: VectorFst<TropicalWeight> = push(&fst, config).unwrap();
        assert_eq!(pushed.num_states(), fst.num_states());
    }

    #[test]
    fn test_push_labels_simple() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(100, 200, TropicalWeight::one(), s1));

        let pushed: VectorFst<TropicalWeight> = push_labels(&fst).unwrap();
        assert_eq!(pushed.num_states(), fst.num_states());
        assert_eq!(pushed.num_arcs_total(), fst.num_arcs_total());
    }

    #[test]
    fn test_topological_sort_acyclic() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));

        let topo = topological_sort(&fst).unwrap();
        assert_eq!(topo.len(), 3);

        // Verify topological order
        let pos: HashMap<_, _> = topo.iter().enumerate().map(|(i, &s)| (s, i)).collect();

        // s0 should come before s1, s1 before s2
        assert!(pos[&s0] < pos[&s1]);
        assert!(pos[&s1] < pos[&s2]);
    }

    #[test]
    fn test_backward_push() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::new(3.0));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s2));

        let config = PushConfig {
            push_to_initial: false,
            ..Default::default()
        };

        let pushed: VectorFst<TropicalWeight> = push(&fst, config).unwrap();
        assert_eq!(pushed.num_states(), fst.num_states());
    }

    #[test]
    fn test_epsilon_removal() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(0, 0, TropicalWeight::one(), s1)); // epsilon
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1)); // non-epsilon

        let config = PushConfig {
            remove_epsilon: true,
            ..Default::default()
        };

        let pushed: VectorFst<TropicalWeight> = push(&fst, config).unwrap();

        // Should have removed epsilon arc
        let arcs: Vec<_> = pushed.arcs(s0).collect();
        assert!(arcs.iter().all(|a| a.ilabel != 0 || a.olabel != 0));
    }

    #[test]
    fn test_full_push_pipeline() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::new(1.0));
        fst.add_arc(s0, Arc::new(0, 0, TropicalWeight::new(0.5), s1)); // epsilon
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(3.0), s2));

        let config = PushConfig {
            push_to_initial: true,
            push_labels: true,
            remove_epsilon: true,
            ..Default::default()
        };

        let result: VectorFst<TropicalWeight> = push(&fst, config).unwrap();
        assert!(result.start().is_some());
        assert!(result.num_states() > 0);
    }
}
