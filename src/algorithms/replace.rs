//! FST replacement algorithm for context-free grammar expansion
//!
//! Implements replacement of non-terminal symbols with finite-state transducers,
//! enabling context-free grammar processing and recursive FST construction.

use crate::arc::Arc;
use crate::fst::{Fst, Label, MutableFst, StateId, VectorFst};
use crate::semiring::Semiring;
use crate::Result;
use std::collections::{HashMap, HashSet, VecDeque};

/// Configuration for replacement algorithm
#[derive(Debug, Clone)]
pub struct ReplaceConfig {
    /// Maximum replacement depth to prevent infinite recursion
    pub max_depth: usize,
    /// Whether to perform left-to-right or right-to-left replacement
    pub left_to_right: bool,
    /// Enable epsilon removal after replacement
    pub remove_epsilon: bool,
    /// Call stack for detecting cycles
    pub enable_cycle_detection: bool,
    /// Return type: epsilon or explicit
    pub return_arc_type: ReturnArcType,
}

impl Default for ReplaceConfig {
    fn default() -> Self {
        Self {
            max_depth: 100,
            left_to_right: true,
            remove_epsilon: false,
            enable_cycle_detection: true,
            return_arc_type: ReturnArcType::Epsilon,
        }
    }
}

/// Type of arc used for returns from non-terminal replacements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnArcType {
    /// Use epsilon arcs for returns
    Epsilon,
    /// Use explicit return symbols
    Explicit,
}

/// Replace FST implementation for context-free grammar expansion
#[derive(Debug, Clone)]
pub struct ReplaceFst<W: Semiring> {
    /// Root non-terminal symbol
    pub root: Label,
    /// Grammar rules mapping non-terminals to replacement FSTs
    pub rules: HashMap<Label, VectorFst<W>>,
    /// Configuration for replacement algorithm
    pub config: ReplaceConfig,
}

impl<W: Semiring> ReplaceFst<W> {
    /// Create a new replace FST with specified root and grammar rules
    pub fn new(root: Label, rules: HashMap<Label, VectorFst<W>>) -> Self {
        Self {
            root,
            rules,
            config: ReplaceConfig::default(),
        }
    }

    /// Create a new replace FST with custom configuration
    pub fn with_config(
        root: Label,
        rules: HashMap<Label, VectorFst<W>>,
        config: ReplaceConfig,
    ) -> Self {
        Self {
            root,
            rules,
            config,
        }
    }

    /// Create a simple replace FST with just a root
    pub fn simple(root: Label) -> Self {
        Self {
            root,
            rules: HashMap::new(),
            config: ReplaceConfig::default(),
        }
    }
}

/// Context for replacement expansion
#[derive(Debug, Clone)]
struct ReplaceContext {
    /// Current call stack for cycle detection
    call_stack: Vec<Label>,
    /// Current expansion depth
    depth: usize,
    /// State mappings for each expansion level
    state_mappings: Vec<HashMap<StateId, StateId>>,
}

impl ReplaceContext {
    fn new() -> Self {
        Self {
            call_stack: Vec::new(),
            depth: 0,
            state_mappings: Vec::new(),
        }
    }

    fn push_call(&mut self, label: Label) -> bool {
        if self.call_stack.contains(&label) {
            false // Cycle detected
        } else {
            self.call_stack.push(label);
            self.depth += 1;
            self.state_mappings.push(HashMap::new());
            true
        }
    }

    fn pop_call(&mut self) {
        if !self.call_stack.is_empty() {
            self.call_stack.pop();
            self.depth -= 1;
            self.state_mappings.pop();
        }
    }
}

/// Replace non-terminal symbols with finite-state transducers
pub fn replace<W, M>(replace_fst: &ReplaceFst<W>) -> Result<M>
where
    W: Semiring + Clone,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();
    let mut context = ReplaceContext::new();

    // Handle empty rules case
    if replace_fst.rules.is_empty() {
        let s0 = result.add_state();
        result.set_start(s0);
        result.set_final(s0, W::one());
        return Ok(result);
    }

    // Start replacement from the root symbol
    if let Some(root_fst) = replace_fst.rules.get(&replace_fst.root) {
        expand_grammar(
            root_fst,
            &mut result,
            &replace_fst.rules,
            &mut context,
            &replace_fst.config,
        )?;
    } else {
        // Create simple FST if no rule for root
        let s0 = result.add_state();
        result.set_start(s0);
        result.set_final(s0, W::one());
    }

    // Post-process the result
    if replace_fst.config.remove_epsilon {
        result = remove_epsilon_transitions(&result)?;
    }

    Ok(result)
}

/// Expand grammar starting from root
fn expand_grammar<W, M>(
    root_fst: &VectorFst<W>,
    result: &mut M,
    rules: &HashMap<Label, VectorFst<W>>,
    context: &mut ReplaceContext,
    config: &ReplaceConfig,
) -> Result<HashMap<StateId, StateId>>
where
    W: Semiring + Clone,
    M: MutableFst<W>,
{
    // Check depth limit
    if context.depth >= config.max_depth {
        return Err(crate::Error::Algorithm(format!(
            "Maximum replacement depth {} exceeded",
            config.max_depth
        )));
    }

    // Create state mapping for this FST
    let mut state_map = HashMap::new();

    // Copy states from root FST
    for state in root_fst.states() {
        let new_state = result.add_state();
        state_map.insert(state, new_state);

        // Copy final weights
        if let Some(weight) = root_fst.final_weight(state) {
            result.set_final(new_state, weight.clone());
        }
    }

    // Set start state if this is the first expansion
    if context.depth == 0 {
        if let Some(start) = root_fst.start() {
            if let Some(&new_start) = state_map.get(&start) {
                result.set_start(new_start);
            }
        }
    }

    // Store current state mapping
    if context.state_mappings.len() <= context.depth {
        context.state_mappings.push(state_map.clone());
    } else {
        context.state_mappings[context.depth] = state_map.clone();
    }

    // Process arcs with replacement
    for state in root_fst.states() {
        let mapped_state = state_map[&state];

        for arc in root_fst.arcs(state) {
            let mapped_nextstate = state_map[&arc.nextstate];

            // Check if this arc represents a non-terminal
            if let Some(replacement_fst) = rules.get(&arc.ilabel) {
                // This is a non-terminal - perform replacement
                expand_non_terminal(
                    replacement_fst,
                    result,
                    rules,
                    mapped_state,
                    mapped_nextstate,
                    &arc,
                    context,
                    config,
                )?;
            } else {
                // Regular terminal arc - copy directly
                result.add_arc(
                    mapped_state,
                    Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), mapped_nextstate),
                );
            }
        }
    }

    Ok(state_map)
}

/// Expand a non-terminal symbol
fn expand_non_terminal<W, M>(
    replacement_fst: &VectorFst<W>,
    result: &mut M,
    rules: &HashMap<Label, VectorFst<W>>,
    from_state: StateId,
    to_state: StateId,
    original_arc: &Arc<W>,
    context: &mut ReplaceContext,
    config: &ReplaceConfig,
) -> Result<()>
where
    W: Semiring + Clone,
    M: MutableFst<W>,
{
    // Check for cycles if enabled
    if config.enable_cycle_detection && !context.push_call(original_arc.ilabel) {
        // Cycle detected - create epsilon transition to avoid infinite recursion
        result.add_arc(
            from_state,
            Arc::new(
                0,
                original_arc.olabel,
                original_arc.weight.clone(),
                to_state,
            ),
        );
        return Ok(());
    }

    // Recursively expand the replacement FST
    let replacement_states = expand_grammar(replacement_fst, result, rules, context, config)?;

    // Connect the replacement to the main graph
    connect_replacement(
        replacement_fst,
        result,
        &replacement_states,
        from_state,
        to_state,
        original_arc,
        config,
    )?;

    // Pop the call stack
    if config.enable_cycle_detection {
        context.pop_call();
    }

    Ok(())
}

/// Connect the replacement FST to the main graph
fn connect_replacement<W, M>(
    replacement_fst: &VectorFst<W>,
    result: &mut M,
    replacement_states: &HashMap<StateId, StateId>,
    from_state: StateId,
    to_state: StateId,
    original_arc: &Arc<W>,
    config: &ReplaceConfig,
) -> Result<()>
where
    W: Semiring + Clone,
    M: MutableFst<W>,
{
    // Find start state of replacement
    if let Some(replacement_start) = replacement_fst.start() {
        if let Some(&mapped_start) = replacement_states.get(&replacement_start) {
            // Create entry transition
            let entry_ilabel = match config.return_arc_type {
                ReturnArcType::Epsilon => 0,
                ReturnArcType::Explicit => original_arc.ilabel,
            };

            result.add_arc(
                from_state,
                Arc::new(
                    entry_ilabel,
                    original_arc.olabel,
                    original_arc.weight.clone(),
                    mapped_start,
                ),
            );
        }
    }

    // Connect final states to continuation
    for replacement_state in replacement_fst.states() {
        if let Some(final_weight) = replacement_fst.final_weight(replacement_state) {
            if let Some(&mapped_state) = replacement_states.get(&replacement_state) {
                // Create exit transition
                let exit_weight = match config.return_arc_type {
                    ReturnArcType::Epsilon => final_weight.clone(),
                    ReturnArcType::Explicit => final_weight.clone(),
                };

                result.add_arc(mapped_state, Arc::new(0, 0, exit_weight, to_state));
            }
        }
    }

    Ok(())
}

/// Remove epsilon transitions from the result FST
fn remove_epsilon_transitions<W, M>(fst: &M) -> Result<M>
where
    W: Semiring + Clone,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();

    // Copy structure without epsilon arcs
    for _ in 0..fst.num_states() {
        result.add_state();
    }

    if let Some(start) = fst.start() {
        result.set_start(start);
    }

    // Process each state
    for state in fst.states() {
        // Copy final weights
        if let Some(weight) = fst.final_weight(state) {
            result.set_final(state, weight.clone());
        }

        // Copy non-epsilon arcs
        for arc in fst.arcs(state) {
            if arc.ilabel != 0 || arc.olabel != 0 {
                result.add_arc(state, arc.clone());
            } else {
                // For epsilon arcs, we need epsilon closure computation
                // For now, we skip them (simplified implementation)
            }
        }
    }

    Ok(result)
}

/// Compute epsilon closure for a set of states
#[allow(dead_code)]
fn compute_epsilon_closure<W, F>(fst: &F, states: &HashSet<StateId>) -> Result<HashMap<StateId, W>>
where
    W: Semiring + Clone,
    F: Fst<W>,
{
    let mut closure = HashMap::new();
    let mut queue = VecDeque::new();

    // Initialize with input states
    for &state in states {
        closure.insert(state, W::one());
        queue.push_back((state, W::one()));
    }

    // Process epsilon transitions
    while let Some((current_state, current_weight)) = queue.pop_front() {
        for arc in fst.arcs(current_state) {
            if arc.ilabel == 0 && arc.olabel == 0 {
                // This is an epsilon arc
                let new_weight = current_weight.times(&arc.weight);

                let should_update = match closure.get(&arc.nextstate) {
                    None => true,
                    Some(existing_weight) => {
                        let combined = existing_weight.plus(&new_weight);
                        if combined != *existing_weight {
                            closure.insert(arc.nextstate, combined);
                            false // Don't add to queue again with old weight
                        } else {
                            false
                        }
                    }
                };

                if should_update {
                    closure.insert(arc.nextstate, new_weight.clone());
                    queue.push_back((arc.nextstate, new_weight));
                }
            }
        }
    }

    Ok(closure)
}

/// Validate grammar for consistency and detect obvious problems
#[allow(dead_code)]
pub fn validate_grammar<W>(rules: &HashMap<Label, VectorFst<W>>) -> Result<()>
where
    W: Semiring,
{
    // Check for empty rules
    if rules.is_empty() {
        return Ok(()); // Empty grammar is valid
    }

    // Check each rule FST for basic validity
    for (label, fst) in rules {
        if fst.num_states() == 0 {
            return Err(crate::Error::Algorithm(format!(
                "Rule for label {label} has no states"
            )));
        }

        if fst.start().is_none() {
            return Err(crate::Error::Algorithm(format!(
                "Rule for label {label} has no start state"
            )));
        }

        // Check that FST is connected
        let mut reachable = HashSet::new();
        if let Some(start) = fst.start() {
            compute_reachable_states(fst, start, &mut reachable);
        }

        if reachable.len() != fst.num_states() {
            return Err(crate::Error::Algorithm(format!(
                "Rule for label {label} has unreachable states"
            )));
        }
    }

    Ok(())
}

/// Compute reachable states from a start state
#[allow(dead_code)]
fn compute_reachable_states<W, F>(fst: &F, start: StateId, reachable: &mut HashSet<StateId>)
where
    W: Semiring,
    F: Fst<W>,
{
    let mut stack = vec![start];

    while let Some(state) = stack.pop() {
        if reachable.insert(state) {
            for arc in fst.arcs(state) {
                stack.push(arc.nextstate);
            }
        }
    }
}

/// Create a replacement FST from a simple string mapping
#[allow(dead_code)]
pub fn from_string_rules<W>(
    root: Label,
    string_rules: HashMap<Label, String>,
) -> Result<ReplaceFst<W>>
where
    W: Semiring + Clone,
{
    let mut rules = HashMap::new();

    for (label, string) in string_rules {
        let fst = create_string_fst(&string)?;
        rules.insert(label, fst);
    }

    Ok(ReplaceFst::new(root, rules))
}

/// Create an FST from a string
#[allow(dead_code)]
fn create_string_fst<W>(string: &str) -> Result<VectorFst<W>>
where
    W: Semiring + Clone,
{
    let mut fst = VectorFst::new();

    if string.is_empty() {
        // Empty string FST
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, W::one());
        return Ok(fst);
    }

    let chars: Vec<char> = string.chars().collect();
    let mut states = Vec::new();

    // Create states
    for _ in 0..=chars.len() {
        states.push(fst.add_state());
    }

    fst.set_start(states[0]);
    fst.set_final(states[chars.len()], W::one());

    // Create arcs for each character
    for (i, &ch) in chars.iter().enumerate() {
        let label = ch as u32;
        fst.add_arc(states[i], Arc::new(label, label, W::one(), states[i + 1]));
    }

    Ok(fst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_replace_fst_simple() {
        let replace_fst = ReplaceFst::<TropicalWeight>::simple(42);
        assert_eq!(replace_fst.root, 42);
        assert!(replace_fst.rules.is_empty());
    }

    #[test]
    fn test_replace_config() {
        let config = ReplaceConfig {
            max_depth: 50,
            left_to_right: false,
            remove_epsilon: true,
            enable_cycle_detection: false,
            return_arc_type: ReturnArcType::Explicit,
        };

        assert_eq!(config.max_depth, 50);
        assert!(!config.left_to_right);
        assert!(config.remove_epsilon);
        assert!(!config.enable_cycle_detection);
        assert_eq!(config.return_arc_type, ReturnArcType::Explicit);
    }

    #[test]
    fn test_replace_basic() {
        let replace_fst = ReplaceFst::<TropicalWeight>::simple(1);
        let result: VectorFst<TropicalWeight> = replace(&replace_fst).unwrap();

        assert!(result.start().is_some());
        assert_eq!(result.num_states(), 1);

        let start_state = result.start().unwrap();
        assert!(result.is_final(start_state));
        assert_eq!(
            result.final_weight(start_state),
            Some(&TropicalWeight::one())
        );
    }

    #[test]
    fn test_replace_with_rules() {
        let mut rules = HashMap::new();

        // Create a simple rule: 1 -> "a"
        let mut rule_fst = VectorFst::<TropicalWeight>::new();
        let s0 = rule_fst.add_state();
        let s1 = rule_fst.add_state();
        rule_fst.set_start(s0);
        rule_fst.set_final(s1, TropicalWeight::one());
        rule_fst.add_arc(
            s0,
            Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s1),
        );

        rules.insert(1, rule_fst);

        let replace_fst = ReplaceFst::new(1, rules);
        let result: VectorFst<TropicalWeight> = replace(&replace_fst).unwrap();

        assert!(result.start().is_some());
        assert!(result.num_states() >= 2);
    }

    #[test]
    fn test_replace_with_config() {
        let mut rules = HashMap::new();

        let mut rule_fst = VectorFst::<TropicalWeight>::new();
        let s0 = rule_fst.add_state();
        let s1 = rule_fst.add_state();
        rule_fst.set_start(s0);
        rule_fst.set_final(s1, TropicalWeight::one());
        rule_fst.add_arc(
            s0,
            Arc::new('b' as u32, 'b' as u32, TropicalWeight::one(), s1),
        );

        rules.insert(2, rule_fst);

        let config = ReplaceConfig {
            max_depth: 10,
            remove_epsilon: true,
            ..Default::default()
        };

        let replace_fst = ReplaceFst::with_config(2, rules, config);
        let result: VectorFst<TropicalWeight> = replace(&replace_fst).unwrap();

        assert!(result.start().is_some());
    }

    #[test]
    fn test_replace_context() {
        let mut context = ReplaceContext::new();

        assert_eq!(context.depth, 0);
        assert!(context.call_stack.is_empty());

        assert!(context.push_call(1));
        assert_eq!(context.depth, 1);
        assert_eq!(context.call_stack.len(), 1);

        // Test cycle detection
        assert!(!context.push_call(1));

        context.pop_call();
        assert_eq!(context.depth, 0);
    }

    #[test]
    fn test_validate_grammar() {
        let mut rules = HashMap::new();

        // Valid rule
        let mut rule_fst = VectorFst::<TropicalWeight>::new();
        let s0 = rule_fst.add_state();
        let s1 = rule_fst.add_state();
        rule_fst.set_start(s0);
        rule_fst.set_final(s1, TropicalWeight::one());
        rule_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));

        rules.insert(1, rule_fst);

        assert!(validate_grammar(&rules).is_ok());

        // Invalid rule - no start state
        let mut invalid_fst = VectorFst::<TropicalWeight>::new();
        invalid_fst.add_state();
        rules.insert(2, invalid_fst);

        assert!(validate_grammar(&rules).is_err());
    }

    #[test]
    fn test_create_string_fst() {
        let fst: VectorFst<TropicalWeight> = create_string_fst("abc").unwrap();

        assert!(fst.start().is_some());
        assert_eq!(fst.num_states(), 4); // 3 chars + final state

        // Test empty string
        let empty_fst: VectorFst<TropicalWeight> = create_string_fst("").unwrap();
        assert_eq!(empty_fst.num_states(), 1);
        assert!(empty_fst.is_final(empty_fst.start().unwrap()));
    }

    #[test]
    fn test_from_string_rules() {
        let mut string_rules = HashMap::new();
        string_rules.insert(1, "hello".to_string());
        string_rules.insert(2, "world".to_string());

        let replace_fst: ReplaceFst<TropicalWeight> = from_string_rules(1, string_rules).unwrap();

        assert_eq!(replace_fst.root, 1);
        assert_eq!(replace_fst.rules.len(), 2);

        let result: VectorFst<TropicalWeight> = replace(&replace_fst).unwrap();
        assert!(result.start().is_some());
    }

    #[test]
    fn test_return_arc_types() {
        assert_eq!(ReturnArcType::Epsilon, ReturnArcType::Epsilon);
        assert_ne!(ReturnArcType::Epsilon, ReturnArcType::Explicit);
    }

    #[test]
    fn test_epsilon_closure() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);

        // Add epsilon transitions
        fst.add_arc(s0, Arc::new(0, 0, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(0, 0, TropicalWeight::new(0.5), s2));

        let mut states = HashSet::new();
        states.insert(s0);

        let closure = compute_epsilon_closure(&fst, &states).unwrap();

        assert!(closure.contains_key(&s0));
        assert!(closure.contains_key(&s1));
        assert!(closure.contains_key(&s2));
    }

    #[test]
    fn test_complex_replacement() {
        let mut rules = HashMap::new();

        // Rule 1: A -> a B
        let mut rule_a = VectorFst::<TropicalWeight>::new();
        let s0 = rule_a.add_state();
        let s1 = rule_a.add_state();
        let s2 = rule_a.add_state();
        rule_a.set_start(s0);
        rule_a.set_final(s2, TropicalWeight::one());
        rule_a.add_arc(
            s0,
            Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s1),
        );
        rule_a.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2)); // B

        // Rule 2: B -> b
        let mut rule_b = VectorFst::<TropicalWeight>::new();
        let s0 = rule_b.add_state();
        let s1 = rule_b.add_state();
        rule_b.set_start(s0);
        rule_b.set_final(s1, TropicalWeight::one());
        rule_b.add_arc(
            s0,
            Arc::new('b' as u32, 'b' as u32, TropicalWeight::one(), s1),
        );

        rules.insert(1, rule_a); // A
        rules.insert(2, rule_b); // B

        let replace_fst = ReplaceFst::new(1, rules);
        let result: VectorFst<TropicalWeight> = replace(&replace_fst).unwrap();

        assert!(result.start().is_some());
        assert!(result.num_states() > 2); // Should expand both A and B
    }

    #[test]
    fn test_cycle_detection() {
        let mut rules = HashMap::new();

        // Create a recursive rule: A -> A a (left recursion)
        let mut rule_a = VectorFst::<TropicalWeight>::new();
        let s0 = rule_a.add_state();
        let s1 = rule_a.add_state();
        let s2 = rule_a.add_state();
        rule_a.set_start(s0);
        rule_a.set_final(s2, TropicalWeight::one());
        rule_a.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1)); // A (recursive)
        rule_a.add_arc(
            s1,
            Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s2),
        );

        rules.insert(1, rule_a);

        let config = ReplaceConfig {
            enable_cycle_detection: true,
            ..Default::default()
        };

        let replace_fst = ReplaceFst::with_config(1, rules, config);
        let result: VectorFst<TropicalWeight> = replace(&replace_fst).unwrap();

        // Should not crash due to infinite recursion
        assert!(result.start().is_some());
    }

    #[test]
    fn test_max_depth_limit() {
        let mut rules = HashMap::new();

        // Create deep nesting: A -> B, B -> C, C -> D, etc.
        for i in 1..=10 {
            let mut rule = VectorFst::<TropicalWeight>::new();
            let s0 = rule.add_state();
            let s1 = rule.add_state();
            rule.set_start(s0);
            rule.set_final(s1, TropicalWeight::one());

            if i < 10 {
                rule.add_arc(s0, Arc::new(i + 1, i + 1, TropicalWeight::one(), s1));
            } else {
                rule.add_arc(
                    s0,
                    Arc::new('x' as u32, 'x' as u32, TropicalWeight::one(), s1),
                );
            }

            rules.insert(i, rule);
        }

        let config = ReplaceConfig {
            max_depth: 5, // Should fail at depth 5
            ..Default::default()
        };

        let replace_fst = ReplaceFst::with_config(1, rules, config);
        let result = replace::<TropicalWeight, VectorFst<TropicalWeight>>(&replace_fst);

        assert!(result.is_err()); // Should exceed max depth
    }
}
