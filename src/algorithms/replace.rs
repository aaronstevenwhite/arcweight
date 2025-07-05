//! FST replacement algorithm for context-free grammar expansion
//!
//! Implements replacement of non-terminal symbols with finite-state transducers,
//! enabling context-free grammar processing and recursive FST construction.

use crate::arc::Arc;
use crate::fst::{Fst, Label, MutableFst, StateId, VectorFst};
use crate::semiring::Semiring;
use crate::Result;
use std::collections::HashMap;

/// Grammar rule mapping non-terminals to VectorFSTs (simplified for now)
pub type _GrammarRules<W> = HashMap<Label, VectorFst<W>>;

/// Configuration for replacement algorithm
#[derive(Debug, Clone)]
pub struct ReplaceConfig {
    /// Maximum replacement depth to prevent infinite recursion
    pub max_depth: usize,
    /// Whether to perform left-to-right or right-to-left replacement
    pub left_to_right: bool,
}

impl Default for ReplaceConfig {
    fn default() -> Self {
        Self {
            max_depth: 100,
            left_to_right: true,
        }
    }
}

/// Replace FST implementation for context-free grammar expansion
#[derive(Debug)]
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

    /// Create a simple replace FST with just a root (for backwards compatibility)
    pub fn simple(root: Label) -> Self {
        Self {
            root,
            rules: HashMap::new(),
            config: ReplaceConfig::default(),
        }
    }
}

/// Replace non-terminal symbols with finite-state transducers
///
/// Implements context-free grammar expansion by replacing non-terminal symbols
/// in FSTs with corresponding sub-FSTs. This enables processing of context-free
/// languages and recursive grammar structures using finite-state technology.
///
/// # Algorithm Details
///
/// - **Non-Terminal Expansion:** Replace special symbols with complete FSTs
/// - **Recursive Replacement:** Support nested and recursive grammar rules
/// - **Time Complexity:** O(expansion_factor × |V| × |E|) depending on grammar
/// - **Space Complexity:** O(expanded_size) for the resulting FST
/// - **Language Relationship:** Implements context-free language recognition
///
/// # Mathematical Foundation
///
/// Replacement implements context-free grammar (CFG) expansion:
/// - **Non-Terminals:** Special symbols representing grammar rules
/// - **Productions:** Mapping from non-terminals to FST fragments
/// - **Expansion:** Recursive substitution of non-terminals with productions
/// - **Termination:** Process terminates when no non-terminals remain
///
/// # Examples
///
/// ## Basic Replacement
///
/// ```rust
/// use arcweight::prelude::*;
/// use arcweight::algorithms::{replace, ReplaceFst};
/// use std::collections::HashMap;
///
/// // Create a simple grammar rule: A -> "hello" (terminal symbol 100)
/// let mut hello_fst = VectorFst::<TropicalWeight>::new();
/// let s0 = hello_fst.add_state();
/// let s1 = hello_fst.add_state();
/// hello_fst.set_start(s0);
/// hello_fst.set_final(s1, TropicalWeight::one());
/// hello_fst.add_arc(s0, Arc::new(100, 100, TropicalWeight::one(), s1)); // terminal "hello"
///
/// // Create grammar rules - use non-terminal 1 that gets replaced by terminal 100
/// let mut rules = HashMap::new();
/// rules.insert(1, hello_fst);
///
/// // Create replace FST for grammar expansion
/// let replace_fst = ReplaceFst::new(1, rules);
///
/// // Perform replacement
/// let expanded: VectorFst<TropicalWeight> = replace(&replace_fst)?;
///
/// // Result contains expanded grammar structure
/// assert!(expanded.start().is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::Algorithm`](crate::Error::Algorithm) if:
/// - The input FST is invalid, corrupted, or malformed
/// - Memory allocation fails during grammar expansion
/// - Non-terminal replacement creates invalid or inconsistent FST structures
/// - Recursive replacement depth exceeds configured limits
/// - Cycle detection fails in recursive grammar expansion
///
/// # See Also
///
/// - [`compose()`](crate::algorithms::compose()) for FST composition in grammar processing
/// - [`union()`](crate::algorithms::union()) for combining grammar alternatives
/// - [`concat()`](crate::algorithms::concat()) for sequencing grammar elements
pub fn replace<W, M>(replace_fst: &ReplaceFst<W>) -> Result<M>
where
    W: Semiring + Clone,
    M: MutableFst<W> + Default,
{
    let mut result = M::default();

    // If no rules are provided, create a simple FST for the root
    if replace_fst.rules.is_empty() {
        let s0 = result.add_state();
        result.set_start(s0);
        result.set_final(s0, W::one());
        return Ok(result);
    }

    // Start replacement from the root symbol
    if let Some(root_fst) = replace_fst.rules.get(&replace_fst.root) {
        // Expand the root FST structure with replacements
        expand_fst_into_result(
            root_fst,
            &mut result,
            &replace_fst.rules,
            0,
            replace_fst.config.max_depth,
        )?;
    } else {
        // No rule for root, create empty FST
        let s0 = result.add_state();
        result.set_start(s0);
        result.set_final(s0, W::one());
    }

    Ok(result)
}

/// Expand an FST into the result, replacing non-terminals recursively
fn expand_fst_into_result<W, M>(
    source_fst: &VectorFst<W>,
    result: &mut M,
    rules: &HashMap<Label, VectorFst<W>>,
    depth: usize,
    max_depth: usize,
) -> Result<HashMap<StateId, StateId>>
where
    W: Semiring + Clone,
    M: MutableFst<W>,
{
    use std::collections::HashMap;

    if depth >= max_depth {
        return Err(crate::Error::Algorithm(
            "Maximum replacement depth exceeded - possible recursive grammar".into(),
        ));
    }

    // Map from source states to result states
    let mut state_map = HashMap::new();

    // Create states in result FST
    for state in source_fst.states() {
        let new_state = result.add_state();
        state_map.insert(state, new_state);
    }

    // Set start state
    if let Some(start) = source_fst.start() {
        if let Some(&mapped_start) = state_map.get(&start) {
            if result.start().is_none() {
                result.set_start(mapped_start);
            }
        }
    }

    // Process arcs and final states
    for state in source_fst.states() {
        let mapped_state = state_map[&state];

        // Set final weight if state is final
        if let Some(weight) = source_fst.final_weight(state) {
            result.set_final(mapped_state, weight.clone());
        }

        // Process outgoing arcs
        for arc in source_fst.arcs(state) {
            let mapped_nextstate = state_map[&arc.nextstate];

            // Check if this arc represents a non-terminal to be replaced
            if let Some(replacement_fst) = rules.get(&arc.ilabel) {
                // This is a non-terminal - replace it with the corresponding FST
                expand_and_connect_replacement(
                    replacement_fst,
                    result,
                    rules,
                    mapped_state,
                    mapped_nextstate,
                    &arc,
                    depth + 1,
                    max_depth,
                )?;
            } else {
                // Regular terminal symbol - copy the arc
                result.add_arc(
                    mapped_state,
                    Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), mapped_nextstate),
                );
            }
        }
    }

    Ok(state_map)
}

/// Expand a replacement FST and connect it between two states
fn expand_and_connect_replacement<W, M>(
    replacement_fst: &VectorFst<W>,
    result: &mut M,
    rules: &HashMap<Label, VectorFst<W>>,
    from_state: StateId,
    to_state: StateId,
    original_arc: &Arc<W>,
    depth: usize,
    max_depth: usize,
) -> Result<()>
where
    W: Semiring + Clone,
    M: MutableFst<W>,
{
    // Create a subgraph for the replacement FST
    let replacement_map = expand_fst_into_result(replacement_fst, result, rules, depth, max_depth)?;

    // Connect the replacement to the main graph
    if let Some(replacement_start) = replacement_fst.start() {
        if let Some(&mapped_start) = replacement_map.get(&replacement_start) {
            // Add epsilon transition from from_state to replacement start
            result.add_arc(
                from_state,
                Arc::new(
                    0, // epsilon input
                    original_arc.olabel,
                    original_arc.weight.clone(),
                    mapped_start,
                ),
            );
        }
    }

    // Connect final states of replacement to to_state
    for state in replacement_fst.states() {
        if let Some(final_weight) = replacement_fst.final_weight(state) {
            if let Some(&mapped_state) = replacement_map.get(&state) {
                // Add epsilon transition from replacement final state to to_state
                result.add_arc(
                    mapped_state,
                    Arc::new(
                        0, // epsilon input
                        0, // epsilon output
                        final_weight.clone(),
                        to_state,
                    ),
                );
            }
        }
    }

    Ok(())
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
    fn test_replace_fst_simple_zero_label() {
        let replace_fst = ReplaceFst::<TropicalWeight>::simple(0);
        assert_eq!(replace_fst.root, 0);
        assert!(replace_fst.rules.is_empty());
    }

    #[test]
    fn test_replace_fst_different_weights() {
        // Test with different semiring types
        let tropical_replace = ReplaceFst::<TropicalWeight>::simple(1);
        assert_eq!(tropical_replace.root, 1);

        let log_replace = ReplaceFst::<LogWeight>::simple(2);
        assert_eq!(log_replace.root, 2);
    }

    #[test]
    fn test_replace_basic() {
        let replace_fst = ReplaceFst::<TropicalWeight>::simple(1);
        let result: VectorFst<TropicalWeight> = replace(&replace_fst).unwrap();

        // Implementation should create single-state FST when no rules
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
    fn test_replace_with_different_root_labels() {
        // Test with different root labels
        let replace_fst1 = ReplaceFst::<TropicalWeight>::simple(100);
        let result1: VectorFst<TropicalWeight> = replace(&replace_fst1).unwrap();

        let replace_fst2 = ReplaceFst::<TropicalWeight>::simple(200);
        let result2: VectorFst<TropicalWeight> = replace(&replace_fst2).unwrap();

        // Both should produce the same basic structure (current implementation)
        assert_eq!(result1.num_states(), result2.num_states());
        assert_eq!(result1.start().is_some(), result2.start().is_some());
    }

    #[test]
    fn test_replace_consistency() {
        // Test that multiple calls with same input produce same result
        let replace_fst = ReplaceFst::<TropicalWeight>::simple(5);

        let result1: VectorFst<TropicalWeight> = replace(&replace_fst).unwrap();
        let result2: VectorFst<TropicalWeight> = replace(&replace_fst).unwrap();

        assert_eq!(result1.num_states(), result2.num_states());
        assert_eq!(result1.start(), result2.start());
        assert_eq!(result1.is_final(0), result2.is_final(0));
    }

    #[test]
    fn test_replace_fst_structure() {
        let replace_fst = ReplaceFst::<TropicalWeight>::simple(42);
        let result: VectorFst<TropicalWeight> = replace(&replace_fst).unwrap();

        // Verify basic structure of current implementation
        assert!(result.start().is_some());
        assert_eq!(result.num_states(), 1);
        assert_eq!(result.num_arcs_total(), 0); // No arcs in single-state FST

        let start_state = result.start().unwrap();
        assert_eq!(start_state, 0); // Should be state 0
        assert!(result.is_final(start_state));
    }

    #[test]
    fn test_replace_fst_weights() {
        // Test with TropicalWeight
        let tropical_replace = ReplaceFst::<TropicalWeight>::simple(1);
        let tropical_result: VectorFst<TropicalWeight> = replace(&tropical_replace).unwrap();

        let start = tropical_result.start().unwrap();
        assert_eq!(
            tropical_result.final_weight(start),
            Some(&TropicalWeight::one())
        );

        // Test with LogWeight
        let log_replace = ReplaceFst::<LogWeight>::simple(1);
        let log_result: VectorFst<LogWeight> = replace(&log_replace).unwrap();

        let start = log_result.start().unwrap();
        assert_eq!(log_result.final_weight(start), Some(&LogWeight::one()));
    }

    #[test]
    fn test_replace_fst_debug() {
        let replace_fst = ReplaceFst::<TropicalWeight>::simple(123);
        let debug_str = format!("{replace_fst:?}");

        // Should contain the root label and type information
        assert!(debug_str.contains("ReplaceFst"));
        assert!(debug_str.contains("root: 123"));
    }

    #[test]
    fn test_replace_fst_large_label() {
        // Test with large label value
        let large_label = u32::MAX;
        let replace_fst = ReplaceFst::<TropicalWeight>::simple(large_label);
        assert_eq!(replace_fst.root, large_label);

        let result: VectorFst<TropicalWeight> = replace(&replace_fst).unwrap();
        assert!(result.start().is_some());
        assert_eq!(result.num_states(), 1);
    }

    #[test]
    fn test_replace_result_properties() {
        let replace_fst = ReplaceFst::<TropicalWeight>::simple(7);
        let result: VectorFst<TropicalWeight> = replace(&replace_fst).unwrap();

        // Test FST properties
        assert!(!result.is_empty()); // Has start state
        assert!(result.start().is_some());

        let start_state = result.start().unwrap();
        assert!(result.is_final(start_state));
        assert_eq!(result.num_arcs(start_state), 0); // No outgoing arcs

        // Test that it accepts empty string
        assert!(result.is_final(start_state));
    }

    #[test]
    fn test_replace_multiple_creations() {
        // Test creating multiple ReplaceFst instances
        let labels = vec![1, 10, 100, 1000];

        for label in labels {
            let replace_fst = ReplaceFst::<TropicalWeight>::simple(label);
            assert_eq!(replace_fst.root, label);

            let result: VectorFst<TropicalWeight> = replace(&replace_fst).unwrap();
            assert!(result.start().is_some());
            assert_eq!(result.num_states(), 1);
        }
    }

    #[test]
    fn test_replace_fst_different_semirings() {
        // Test ReplaceFst works with different semiring types

        // TropicalWeight
        let tropical = ReplaceFst::<TropicalWeight>::simple(1);
        assert_eq!(tropical.root, 1);

        // LogWeight
        let log = ReplaceFst::<LogWeight>::simple(2);
        assert_eq!(log.root, 2);

        // Both should work with replace function
        let tropical_result: VectorFst<TropicalWeight> = replace(&tropical).unwrap();
        let log_result: VectorFst<LogWeight> = replace(&log).unwrap();

        assert!(tropical_result.start().is_some());
        assert!(log_result.start().is_some());
    }

    #[test]
    fn test_replace_result_iteration() {
        let replace_fst = ReplaceFst::<TropicalWeight>::simple(99);
        let result: VectorFst<TropicalWeight> = replace(&replace_fst).unwrap();

        // Test iterating over states
        let states: Vec<_> = result.states().collect();
        assert_eq!(states.len(), 1);
        assert_eq!(states[0], 0);

        // Test iterating over arcs (should be empty)
        let arcs: Vec<_> = result.arcs(states[0]).collect();
        assert_eq!(arcs.len(), 0);
    }

    #[test]
    fn test_replace_edge_cases() {
        // Test with minimum label
        let min_replace = ReplaceFst::<TropicalWeight>::simple(0);
        let min_result: VectorFst<TropicalWeight> = replace(&min_replace).unwrap();
        assert!(min_result.start().is_some());

        // Test with maximum label
        let max_replace = ReplaceFst::<TropicalWeight>::simple(u32::MAX);
        let max_result: VectorFst<TropicalWeight> = replace(&max_replace).unwrap();
        assert!(max_result.start().is_some());
    }

    #[test]
    fn test_replace_with_rules() {
        // Test replacement with actual grammar rules
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

        // Should have created an FST with the rule structure
        assert!(result.start().is_some());
        assert!(result.num_states() >= 2); // At least start and final states
    }
}
