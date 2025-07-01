//! FST property traits and computation

use crate::fst::{Fst, StateId};
use crate::semiring::Semiring;
use bitflags::bitflags;
use std::collections::HashSet;

bitflags! {
    /// FST property flags for optimization and analysis
    ///
    /// These flags track structural and algorithmic properties of FSTs,
    /// enabling algorithm selection and optimization. Properties are computed
    /// once and cached for efficient access.
    ///
    /// # Property Categories
    ///
    /// ## Structural Properties
    /// - **Acceptor/Transducer:** Whether input equals output labels
    /// - **Connectivity:** Reachability and strong connectivity
    /// - **Topology:** Cycles, linearity, and sorting
    ///
    /// ## Epsilon Properties
    /// - **Epsilon transitions:** Presence and location of ε-transitions
    /// - **Determinism:** Input/output determinism properties
    ///
    /// ## Weight Properties
    /// - **Weighted/Unweighted:** Presence of non-trivial weights
    /// - **Functional:** Single output per input property
    ///
    /// # Usage in Algorithms
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// fn optimize_fst<W: DivisibleSemiring + std::hash::Hash + Eq + Ord>(
    ///     fst: &VectorFst<W>
    ///     ) -> Result<VectorFst<W>> {
    ///     let props = compute_properties(fst);
    ///
    ///     // Choose algorithm based on properties
    ///     if props.has_property(PropertyFlags::ACCEPTOR) {
    ///         // Use standard minimization for acceptors
    ///         minimize(fst)
    ///     } else if props.has_property(PropertyFlags::INPUT_DETERMINISTIC) {
    ///         // Skip determinization step
    ///         minimize(fst)
    ///     } else {
    ///         // Full determinize-then-minimize pipeline
    ///         let det_fst: VectorFst<W> = determinize(fst)?;
    ///         minimize(&det_fst)
    ///     }
    /// }
    ///
    /// // Example usage
    /// let mut fst = VectorFst::<TropicalWeight>::new();
    /// let s0 = fst.add_state();
    /// fst.set_start(s0);
    /// fst.set_final(s0, TropicalWeight::one());
    /// let result = optimize_fst(&fst).unwrap();
    /// assert!(result.num_states() > 0);
    /// ```
    ///
    /// # Performance Benefits
    ///
    /// - **Algorithm Selection:** Choose optimal algorithms based on FST structure
    /// - **Early Termination:** Skip unnecessary operations for certain properties
    /// - **Memory Optimization:** Specialized data structures for specific properties
    /// - **Composition Optimization:** Filter selection based on epsilon patterns
    ///
    /// # See Also
    ///
    /// - [`compute_properties`] for computing properties from FST structure
    /// - [`FstProperties`] for the property container type
    /// - [Architecture Guide](../../docs/architecture/README.md) for implementation details
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PropertyFlags: u64 {
        /// FST is an acceptor (input labels = output labels on all arcs)
        ///
        /// An acceptor FST represents a regular language rather than a transduction.
        /// This enables optimizations in minimization and determinization algorithms.
        ///
        /// **Optimization Impact:** Allows use of acceptor-specific algorithms
        const ACCEPTOR = 1 << 0;

        /// FST has no epsilon (ε) transitions
        ///
        /// No arcs have epsilon (label 0) on either input or output.
        /// This property enables more efficient composition and search algorithms.
        ///
        /// **Optimization Impact:** Avoids epsilon removal preprocessing
        const NO_EPSILONS = 1 << 1;

        /// FST has epsilon (ε) transitions
        ///
        /// At least one arc has epsilon (label 0) on input or output.
        /// Epsilon transitions can consume no input while producing output or vice versa.
        const EPSILONS = 1 << 2;

        /// FST has no input epsilon transitions
        ///
        /// No arcs have epsilon (label 0) on the input side.
        /// Input-deterministic FSTs often have this property.
        const NO_INPUT_EPSILONS = 1 << 3;

        /// FST has input epsilon transitions
        ///
        /// At least one arc has epsilon (label 0) on the input side.
        /// These transitions consume no input symbols.
        const INPUT_EPSILONS = 1 << 4;

        /// FST has no output epsilon transitions
        ///
        /// No arcs have epsilon (label 0) on the output side.
        /// This is common in acceptors and some transducers.
        const NO_OUTPUT_EPSILONS = 1 << 5;

        /// FST has output epsilon transitions
        ///
        /// At least one arc has epsilon (label 0) on the output side.
        /// These transitions produce no output symbols.
        const OUTPUT_EPSILONS = 1 << 6;

        /// FST is deterministic on input labels
        ///
        /// From any state, there is at most one outgoing arc for each input label.
        /// This property is crucial for efficient lookup and composition.
        ///
        /// **Optimization Impact:** Enables deterministic algorithms, avoids subset construction
        const INPUT_DETERMINISTIC = 1 << 7;

        /// FST is deterministic on output labels
        ///
        /// From any state, there is at most one outgoing arc for each output label.
        /// Less commonly used than input determinism but useful for some algorithms.
        const OUTPUT_DETERMINISTIC = 1 << 8;

        /// FST is functional (single output sequence per input)
        ///
        /// Each input sequence has at most one corresponding output sequence.
        /// This property is important for unambiguous transductions.
        ///
        /// **Optimization Impact:** Simplifies composition and inversion algorithms
        const FUNCTIONAL = 1 << 9;

        /// FST has no ambiguous input paths
        ///
        /// Each input sequence corresponds to a unique path through the FST.
        /// This is a stronger property than functional.
        const UNAMBIGUOUS = 1 << 10;

        /// All states are accessible from the start state
        ///
        /// Every state can be reached by following some path from the start state.
        /// FSTs without this property have unreachable states that can be removed.
        ///
        /// **Optimization Impact:** Indicates no dead states to clean up
        const ACCESSIBLE = 1 << 11;

        /// All states are coaccessible (can reach some final state)
        ///
        /// Every state has a path to at least one final state.
        /// FSTs without this property have dead-end states.
        ///
        /// **Optimization Impact:** Indicates no dead-end states to remove
        const COACCESSIBLE = 1 << 12;

        /// FST is strongly connected (accessible + coaccessible)
        ///
        /// Equivalent to having both ACCESSIBLE and COACCESSIBLE properties.
        /// This is the most connected form an FST can have.
        const CONNECTED = 1 << 13;

        /// FST contains cycles (paths from states back to themselves)
        ///
        /// The FST has at least one cycle in its state graph.
        /// Cyclic FSTs can accept infinite languages.
        const CYCLIC = 1 << 14;

        /// FST is acyclic (no cycles in state graph)
        ///
        /// The FST has no cycles, making it a directed acyclic graph (DAG).
        /// Acyclic FSTs accept finite languages and enable certain optimizations.
        ///
        /// **Optimization Impact:** Enables topological sorting and dynamic programming algorithms
        const ACYCLIC = 1 << 15;

        /// FST is acyclic when considering only initial state reachability
        ///
        /// There are no cycles reachable from the start state.
        /// This is weaker than full acyclicity.
        const INITIAL_ACYCLIC = 1 << 16;

        /// FST states are topologically sorted
        ///
        /// State IDs are assigned such that arcs only go from lower to higher state IDs.
        /// This property implies acyclicity and enables efficient algorithms.
        ///
        /// **Optimization Impact:** Enables linear-time algorithms on acyclic FSTs
        const TOP_SORTED = 1 << 17;

        /// Arcs from each state are sorted by input label
        ///
        /// For each state, outgoing arcs are ordered by increasing input label.
        /// This enables binary search for arc lookup.
        ///
        /// **Optimization Impact:** O(log k) arc lookup instead of O(k) where k is the number of arcs
        const INPUT_SORTED = 1 << 18;

        /// Arcs from each state are sorted by output label
        ///
        /// For each state, outgoing arcs are ordered by increasing output label.
        /// This is less commonly used than input sorting.
        const OUTPUT_SORTED = 1 << 19;

        /// FST has non-trivial weights (not all weights are semiring one)
        ///
        /// At least one arc or final state has a weight different from the semiring's
        /// multiplicative identity (one).
        const WEIGHTED = 1 << 20;

        /// FST is unweighted (all weights are semiring one)
        ///
        /// All arc weights and final weights equal the semiring's multiplicative identity.
        /// This enables optimizations that ignore weight computation.
        ///
        /// **Optimization Impact:** Simplifies algorithms to focus on structure only
        const UNWEIGHTED = 1 << 21;

        /// FST is string-like (represents a single linear path)
        ///
        /// The FST has no branching - each state has at most one outgoing arc.
        /// String FSTs represent single sequences rather than languages.
        ///
        /// **Optimization Impact:** Enables specialized string algorithms
        const STRING = 1 << 22;
    }
}

/// FST properties
#[derive(Debug, Clone, Copy)]
pub struct FstProperties {
    /// Known properties
    pub known: PropertyFlags,
    /// Property values
    pub properties: PropertyFlags,
}

impl Default for FstProperties {
    fn default() -> Self {
        Self {
            known: PropertyFlags::empty(),
            properties: PropertyFlags::empty(),
        }
    }
}

impl FstProperties {
    // Property constants - re-exported from PropertyFlags for convenience
    /// FST is an acceptor (input = output)
    pub const ACCEPTOR: PropertyFlags = PropertyFlags::ACCEPTOR;
    /// FST has no epsilon transitions
    pub const NO_EPSILONS: PropertyFlags = PropertyFlags::NO_EPSILONS;
    /// FST has epsilon transitions
    pub const EPSILONS: PropertyFlags = PropertyFlags::EPSILONS;
    /// FST has no input epsilons
    pub const NO_INPUT_EPSILONS: PropertyFlags = PropertyFlags::NO_INPUT_EPSILONS;
    /// FST has input epsilons (alias)
    pub const I_EPSILONS: PropertyFlags = PropertyFlags::INPUT_EPSILONS;
    /// FST has no output epsilons
    pub const NO_OUTPUT_EPSILONS: PropertyFlags = PropertyFlags::NO_OUTPUT_EPSILONS;
    /// FST has output epsilons (alias)
    pub const O_EPSILONS: PropertyFlags = PropertyFlags::OUTPUT_EPSILONS;
    /// FST is deterministic on input
    pub const INPUT_DETERMINISTIC: PropertyFlags = PropertyFlags::INPUT_DETERMINISTIC;
    /// FST is deterministic on input (alias)
    pub const I_DETERMINISTIC: PropertyFlags = PropertyFlags::INPUT_DETERMINISTIC;
    /// FST is deterministic on output
    pub const OUTPUT_DETERMINISTIC: PropertyFlags = PropertyFlags::OUTPUT_DETERMINISTIC;
    /// FST has functional property
    pub const FUNCTIONAL: PropertyFlags = PropertyFlags::FUNCTIONAL;
    /// FST has no ambiguous paths
    pub const UNAMBIGUOUS: PropertyFlags = PropertyFlags::UNAMBIGUOUS;
    /// All states are accessible
    pub const ACCESSIBLE: PropertyFlags = PropertyFlags::ACCESSIBLE;
    /// All states are coaccessible
    pub const COACCESSIBLE: PropertyFlags = PropertyFlags::COACCESSIBLE;
    /// FST is strongly connected
    pub const CONNECTED: PropertyFlags = PropertyFlags::CONNECTED;
    /// FST is cyclic
    pub const CYCLIC: PropertyFlags = PropertyFlags::CYCLIC;
    /// FST is acyclic
    pub const ACYCLIC: PropertyFlags = PropertyFlags::ACYCLIC;
    /// FST has initial state
    pub const INITIAL_ACYCLIC: PropertyFlags = PropertyFlags::INITIAL_ACYCLIC;
    /// FST is topologically sorted
    pub const TOP_SORTED: PropertyFlags = PropertyFlags::TOP_SORTED;
    /// Arcs are sorted by input label
    pub const INPUT_SORTED: PropertyFlags = PropertyFlags::INPUT_SORTED;
    /// Arcs are sorted by output label
    pub const OUTPUT_SORTED: PropertyFlags = PropertyFlags::OUTPUT_SORTED;
    /// FST is weighted
    pub const WEIGHTED: PropertyFlags = PropertyFlags::WEIGHTED;
    /// FST is unweighted
    pub const UNWEIGHTED: PropertyFlags = PropertyFlags::UNWEIGHTED;
    /// FST is string-like (linear path)
    pub const STRING: PropertyFlags = PropertyFlags::STRING;

    /// Check if a property is known
    pub fn is_known(&self, flag: PropertyFlags) -> bool {
        self.known.contains(flag)
    }

    /// Check if a property is set
    pub fn has_property(&self, flag: PropertyFlags) -> bool {
        self.is_known(flag) && self.properties.contains(flag)
    }

    /// Check if a property is set (alias for has_property for compatibility)
    pub fn contains(&self, flag: PropertyFlags) -> bool {
        self.has_property(flag)
    }

    /// Set a property
    pub fn set_property(&mut self, flag: PropertyFlags, value: bool) {
        self.known |= flag;
        if value {
            self.properties |= flag;
        } else {
            self.properties &= !flag;
        }
    }

    /// Invalidate all properties
    pub fn invalidate_all(&mut self) {
        self.known = PropertyFlags::empty();
        self.properties = PropertyFlags::empty();
    }
}

/// Compute FST properties through structural analysis
///
/// Analyzes the FST structure to determine which properties hold, including
/// epsilon patterns, determinism, connectivity, and weight characteristics.
/// Properties are computed once and can be cached for efficient access.
///
/// # Algorithm
///
/// - **Time Complexity:** O(|V| + |E|) where V = states, E = arcs
/// - **Space Complexity:** O(|V|) for visited state tracking
/// - **Analysis:** Single pass through all states and arcs
///
/// # Properties Computed
///
/// ## Always Computed
/// - Epsilon transition patterns
/// - Acceptor vs transducer classification  
/// - Weight presence (weighted vs unweighted)
/// - Connectivity (accessible, coaccessible)
/// - Topology (cyclic vs acyclic)
/// - String property (linear vs branching)
///
/// ## Not Currently Computed
/// - Input/output determinism (requires more complex analysis)
/// - Functional property (requires path analysis)
/// - Arc sorting properties (requires per-state sorting checks)
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Analyze a simple acceptor FST
/// let mut fst = VectorFst::<BooleanWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// fst.set_start(s0);
/// fst.set_final(s1, BooleanWeight::one());
/// fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, BooleanWeight::one(), s1));
///
/// let props = compute_properties(&fst);
///
/// // Check computed properties
/// assert!(props.has_property(PropertyFlags::ACCEPTOR));
/// assert!(props.has_property(PropertyFlags::UNWEIGHTED));
/// assert!(props.has_property(PropertyFlags::ACYCLIC));
/// assert!(props.has_property(PropertyFlags::NO_EPSILONS));
/// assert!(props.has_property(PropertyFlags::STRING));
/// ```
///
/// ## Property-Based Optimization
///
/// ```rust
/// use arcweight::prelude::*;
///
/// fn optimize_based_on_properties<W: StarSemiring + std::hash::Hash + Eq + Ord>(
///     fst: &VectorFst<W>
/// ) -> Result<VectorFst<W>> {
///     let props = compute_properties(fst);
///     
///     // Skip operations based on properties
///     let mut result = fst.clone();
///     
///     if !props.has_property(PropertyFlags::ACCESSIBLE) {
///         result = connect(&result)?; // Remove unreachable states
///     }
///     
///     if props.has_property(PropertyFlags::NO_EPSILONS) {
///         // Skip epsilon removal
///         println!("No epsilons detected, skipping removal");
///     } else {
///         result = remove_epsilons(&result)?;
///     }
///     
///     if props.has_property(PropertyFlags::ACYCLIC) {
///         // Use topological sort for acyclic FSTs
///         result = topsort(&result)?;
///     }
///     
///     Ok(result)
/// }
///
/// // Example usage  
/// let mut fst = VectorFst::<BooleanWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
/// fst.set_start(s0);
/// fst.set_final(s1, BooleanWeight::one());
/// fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, BooleanWeight::one(), s1));
///
/// let optimized = optimize_based_on_properties(&fst).unwrap();
/// assert!(optimized.num_states() > 0);
/// ```
///
/// # Performance Considerations
///
/// - **Caching:** Properties should be computed once and cached in the FST
/// - **Incremental:** Modifying FSTs should invalidate affected properties
/// - **Selective:** Only compute expensive properties when needed
/// - **Lazy:** Some properties can be computed on-demand
///
/// # Errors
///
/// This function does not return errors in the current implementation, but future
/// versions may return errors if:
/// - The input FST structure is corrupted or invalid  
/// - Memory allocation fails during property computation
/// - Complex property analysis encounters unsupported FST patterns
///
/// # See Also
///
/// - [`PropertyFlags`] for the complete list of trackable properties
/// - [`FstProperties`] for the property container with known/unknown tracking
/// - [Architecture Guide](../../docs/architecture/README.md) for implementation details
pub fn compute_properties<W: Semiring, F: Fst<W>>(fst: &F) -> FstProperties {
    let mut props = FstProperties::default();

    // check for start state
    if fst.start().is_none() {
        // Empty FST has specific default properties
        props.set_property(PropertyFlags::ACCEPTOR, true);
        props.set_property(PropertyFlags::UNWEIGHTED, true);
        props.set_property(PropertyFlags::WEIGHTED, false);
        props.set_property(PropertyFlags::ACYCLIC, true);
        props.set_property(PropertyFlags::INITIAL_ACYCLIC, true);
        props.set_property(PropertyFlags::TOP_SORTED, true);
        props.set_property(PropertyFlags::ACCESSIBLE, true);
        props.set_property(PropertyFlags::COACCESSIBLE, true);
        props.set_property(PropertyFlags::CONNECTED, true);
        props.set_property(PropertyFlags::INPUT_DETERMINISTIC, true);
        props.set_property(PropertyFlags::OUTPUT_DETERMINISTIC, true);
        props.set_property(PropertyFlags::FUNCTIONAL, true);
        props.set_property(PropertyFlags::NO_EPSILONS, true);
        props.set_property(PropertyFlags::EPSILONS, false);
        props.set_property(PropertyFlags::NO_INPUT_EPSILONS, true);
        props.set_property(PropertyFlags::INPUT_EPSILONS, false);
        props.set_property(PropertyFlags::NO_OUTPUT_EPSILONS, true);
        props.set_property(PropertyFlags::OUTPUT_EPSILONS, false);
        return props;
    }

    // check basic properties
    let mut has_epsilons = false;
    let mut has_input_epsilons = false;
    let mut has_output_epsilons = false;
    let mut is_acceptor = true;
    let mut is_unweighted = true;

    // For cycle detection, track states we've visited
    let mut visited = HashSet::new();
    let mut rec_stack = HashSet::new();

    // DFS to check for cycles
    fn has_cycle<W: Semiring, F: Fst<W>>(
        fst: &F,
        state: StateId,
        visited: &mut HashSet<StateId>,
        rec_stack: &mut HashSet<StateId>,
    ) -> bool {
        visited.insert(state);
        rec_stack.insert(state);

        for arc in fst.arcs(state) {
            if !visited.contains(&arc.nextstate) {
                if has_cycle(fst, arc.nextstate, visited, rec_stack) {
                    return true;
                }
            } else if rec_stack.contains(&arc.nextstate) {
                return true;
            }
        }

        rec_stack.remove(&state);
        false
    }

    // Start DFS from start state if it exists
    let is_acyclic = if let Some(start_state) = fst.start() {
        !has_cycle(fst, start_state, &mut visited, &mut rec_stack)
    } else {
        // No start state means no reachable states, so it's acyclic
        true
    };

    // Check if all states are accessible from start state
    let accessible_states = visited.clone();
    let is_accessible = if fst.start().is_some() {
        accessible_states.len() == fst.num_states()
    } else {
        // No start state means no states are accessible
        fst.num_states() == 0
    };

    // For coaccessible, we need to check if all states can reach a final state
    // using backward DFS from final states
    let is_coaccessible = compute_coaccessible(fst);

    // Check if FST is string-like (linear path with no branching and no cycles)
    let mut is_string = true;
    for state in fst.states() {
        let num_arcs = fst.num_arcs(state);
        if num_arcs > 1 {
            is_string = false;
            break;
        }
    }
    // String FST must also be acyclic
    if !is_acyclic {
        is_string = false;
    }

    // Check determinism properties and other arc-based properties
    let (is_input_deterministic, is_output_deterministic) = compute_determinism(fst);
    let is_functional = compute_functional(fst);
    
    // Check other properties by examining arcs
    for state in fst.states() {
        for arc in fst.arcs(state) {
            if arc.is_epsilon() {
                has_epsilons = true;
            }
            if arc.is_epsilon_input() {
                has_input_epsilons = true;
            }
            if arc.is_epsilon_output() {
                has_output_epsilons = true;
            }
            if arc.ilabel != arc.olabel {
                is_acceptor = false;
            }
            if !<W as num_traits::One>::is_one(&arc.weight) {
                is_unweighted = false;
            }
        }
    }

    // Set all computed properties
    props.set_property(PropertyFlags::NO_EPSILONS, !has_epsilons);
    props.set_property(PropertyFlags::EPSILONS, has_epsilons);
    props.set_property(PropertyFlags::NO_INPUT_EPSILONS, !has_input_epsilons);
    props.set_property(PropertyFlags::INPUT_EPSILONS, has_input_epsilons);
    props.set_property(PropertyFlags::NO_OUTPUT_EPSILONS, !has_output_epsilons);
    props.set_property(PropertyFlags::OUTPUT_EPSILONS, has_output_epsilons);
    props.set_property(PropertyFlags::ACCEPTOR, is_acceptor);
    props.set_property(PropertyFlags::UNWEIGHTED, is_unweighted);
    props.set_property(PropertyFlags::WEIGHTED, !is_unweighted);
    props.set_property(PropertyFlags::ACYCLIC, is_acyclic);
    props.set_property(PropertyFlags::CYCLIC, !is_acyclic);
    props.set_property(PropertyFlags::ACCESSIBLE, is_accessible);
    props.set_property(PropertyFlags::COACCESSIBLE, is_coaccessible);
    props.set_property(PropertyFlags::CONNECTED, is_accessible && is_coaccessible);
    props.set_property(PropertyFlags::INPUT_DETERMINISTIC, is_input_deterministic);
    props.set_property(PropertyFlags::OUTPUT_DETERMINISTIC, is_output_deterministic);
    props.set_property(PropertyFlags::FUNCTIONAL, is_functional);
    props.set_property(PropertyFlags::INITIAL_ACYCLIC, is_acyclic);
    props.set_property(PropertyFlags::TOP_SORTED, is_acyclic);
    props.set_property(PropertyFlags::STRING, is_string);

    props
}

/// Compute coaccessible property using backward reachability analysis
///
/// A state is coaccessible if there exists a path from that state to some final state.
/// This function performs backward reachability analysis starting from all final states.
fn compute_coaccessible<W: Semiring, F: Fst<W>>(fst: &F) -> bool {
    if fst.num_states() == 0 {
        return true; // Empty FST is vacuously coaccessible
    }

    // Collect all final states
    let mut final_states = Vec::new();
    for state in fst.states() {
        if fst.is_final(state) {
            final_states.push(state);
        }
    }

    if final_states.is_empty() {
        return false; // No final states means no state is coaccessible
    }

    // Build reverse graph for backward reachability
    let mut reverse_graph: std::collections::HashMap<StateId, Vec<StateId>> = 
        std::collections::HashMap::new();
    
    for state in fst.states() {
        for arc in fst.arcs(state) {
            reverse_graph.entry(arc.nextstate).or_insert_with(Vec::new).push(state);
        }
    }

    // Perform backward DFS from all final states
    let mut visited = std::collections::HashSet::new();
    let mut stack = final_states.clone();

    while let Some(state) = stack.pop() {
        if visited.insert(state) {
            if let Some(predecessors) = reverse_graph.get(&state) {
                for &pred in predecessors {
                    if !visited.contains(&pred) {
                        stack.push(pred);
                    }
                }
            }
        }
    }

    // All states should be reachable from final states for coaccessible property
    visited.len() == fst.num_states()
}

/// Compute input and output determinism properties
///
/// Returns (is_input_deterministic, is_output_deterministic) where:
/// - Input deterministic: for each state, at most one arc per input label
/// - Output deterministic: for each state, at most one arc per output label
fn compute_determinism<W: Semiring, F: Fst<W>>(fst: &F) -> (bool, bool) {
    let mut is_input_deterministic = true;
    let mut is_output_deterministic = true;

    for state in fst.states() {
        let mut input_labels = std::collections::HashSet::new();
        let mut output_labels = std::collections::HashSet::new();

        for arc in fst.arcs(state) {
            // Check input determinism
            if !input_labels.insert(arc.ilabel) {
                is_input_deterministic = false;
            }

            // Check output determinism
            if !output_labels.insert(arc.olabel) {
                is_output_deterministic = false;
            }

            // Early exit if both are false
            if !is_input_deterministic && !is_output_deterministic {
                return (false, false);
            }
        }
    }

    (is_input_deterministic, is_output_deterministic)
}

/// Compute functional property
///
/// An FST is functional if each input string has at most one corresponding output string.
/// This is a conservative approximation - we check if the FST is both input-deterministic
/// and has no epsilon cycles, which guarantees functionality.
fn compute_functional<W: Semiring, F: Fst<W>>(fst: &F) -> bool {
    // Simple approximation: functional if input deterministic and no epsilon cycles
    let (is_input_deterministic, _) = compute_determinism(fst);
    
    if !is_input_deterministic {
        return false;
    }

    // Check for epsilon cycles - functional FSTs shouldn't have epsilon cycles
    // that could generate multiple outputs for the same input

    fn has_epsilon_cycle<W: Semiring, F: Fst<W>>(
        fst: &F,
        state: StateId,
        visited: &mut std::collections::HashSet<StateId>,
        rec_stack: &mut std::collections::HashSet<StateId>,
    ) -> bool {
        visited.insert(state);
        rec_stack.insert(state);

        for arc in fst.arcs(state) {
            // Only consider epsilon transitions for cycles
            if arc.is_epsilon() {
                if !visited.contains(&arc.nextstate) {
                    if has_epsilon_cycle(fst, arc.nextstate, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(&arc.nextstate) {
                    return true;
                }
            }
        }

        rec_stack.remove(&state);
        false
    }

    // Check for epsilon cycles from any reachable state
    if let Some(start_state) = fst.start() {
        // First, find all reachable states
        let mut reachable_states: std::collections::HashSet<StateId> = std::collections::HashSet::new();
        let mut stack = vec![start_state];
        
        while let Some(state) = stack.pop() {
            if reachable_states.insert(state) {
                for arc in fst.arcs(state) {
                    if !reachable_states.contains(&arc.nextstate) {
                        stack.push(arc.nextstate);
                    }
                }
            }
        }
        
        // Check for epsilon cycles starting from any reachable state
        for &state in &reachable_states {
            let mut local_visited: std::collections::HashSet<StateId> = std::collections::HashSet::new();
            let mut local_rec_stack: std::collections::HashSet<StateId> = std::collections::HashSet::new();
            if has_epsilon_cycle(fst, state, &mut local_visited, &mut local_rec_stack) {
                return false;
            }
        }
        
        true // No epsilon cycles found
    } else {
        true // Empty FST is vacuously functional
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use num_traits::One;

    #[test]
    fn test_empty_fst_properties() {
        let fst = VectorFst::<TropicalWeight>::new();
        let props = compute_properties(&fst);

        // Empty FST should have certain properties
        assert!(props.contains(PropertyFlags::ACCEPTOR));
        assert!(props.contains(PropertyFlags::UNWEIGHTED));
        assert!(props.contains(PropertyFlags::ACYCLIC));
        assert!(props.contains(PropertyFlags::INITIAL_ACYCLIC));
        assert!(props.contains(PropertyFlags::TOP_SORTED));
        assert!(props.contains(PropertyFlags::ACCESSIBLE));
        assert!(props.contains(PropertyFlags::COACCESSIBLE));
    }

    #[test]
    fn test_single_state_fst_properties() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::one());

        let props = compute_properties(&fst);

        assert!(props.contains(PropertyFlags::ACCEPTOR));
        assert!(props.contains(PropertyFlags::UNWEIGHTED));
        assert!(props.contains(PropertyFlags::ACYCLIC));
        assert!(props.contains(PropertyFlags::ACCESSIBLE));
        assert!(props.contains(PropertyFlags::COACCESSIBLE));
    }

    #[test]
    fn test_simple_acceptor_properties() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));

        let props = compute_properties(&fst);

        // Should be an acceptor (input == output labels)
        assert!(props.contains(PropertyFlags::ACCEPTOR));
        assert!(props.contains(PropertyFlags::UNWEIGHTED));
        assert!(props.contains(PropertyFlags::ACYCLIC));
        assert!(props.contains(PropertyFlags::ACCESSIBLE));
        assert!(props.contains(PropertyFlags::COACCESSIBLE));
    }

    #[test]
    fn test_transducer_properties() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::one(), s1)); // Different input/output

        let props = compute_properties(&fst);

        // Should NOT be an acceptor
        assert!(!props.contains(PropertyFlags::ACCEPTOR));
        assert!(props.contains(PropertyFlags::UNWEIGHTED));
    }

    #[test]
    fn test_weighted_fst_properties() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.5), s1)); // Non-unit weight

        let props = compute_properties(&fst);

        // Should be weighted
        assert!(!props.contains(PropertyFlags::UNWEIGHTED));
        assert!(props.contains(PropertyFlags::ACCEPTOR));
    }

    #[test]
    fn test_cyclic_fst_properties() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s0)); // Creates cycle

        let props = compute_properties(&fst);

        // Should be cyclic
        assert!(!props.contains(PropertyFlags::ACYCLIC));
        assert!(props.contains(PropertyFlags::ACCESSIBLE));
        assert!(props.contains(PropertyFlags::COACCESSIBLE));
    }

    #[test]
    fn test_epsilon_fst_properties() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());
        fst.add_arc(s0, Arc::epsilon(TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(1, 1, TropicalWeight::one(), s2));

        let props = compute_properties(&fst);

        // Should have epsilon transitions
        assert!(props.contains(PropertyFlags::EPSILONS));
        assert!(props.contains(PropertyFlags::INPUT_EPSILONS));
        assert!(props.contains(PropertyFlags::OUTPUT_EPSILONS));
    }

    #[test]
    fn test_deterministic_properties() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.set_final(s2, TropicalWeight::one());

        // Add two arcs with same input label (non-deterministic)
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::one(), s2));

        let props = compute_properties(&fst);

        // Should not be input deterministic
        assert!(!props.contains(PropertyFlags::INPUT_DETERMINISTIC));
        // Should not be functional either
        assert!(!props.contains(PropertyFlags::FUNCTIONAL));
    }
    
    #[test]
    fn test_input_deterministic_fst() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Different input labels - should be deterministic
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s2));
        fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::one(), s2));

        let props = compute_properties(&fst);

        // Should be input deterministic
        assert!(props.contains(PropertyFlags::INPUT_DETERMINISTIC));
        // Should be functional
        assert!(props.contains(PropertyFlags::FUNCTIONAL));
    }
    
    #[test]
    fn test_output_deterministic_fst() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Same output labels - should not be output deterministic
        fst.add_arc(s0, Arc::new(1, 10, TropicalWeight::one(), s1));
        fst.add_arc(s0, Arc::new(2, 10, TropicalWeight::one(), s2)); // Same output label

        let props = compute_properties(&fst);

        // Should be input deterministic but not output deterministic
        assert!(props.contains(PropertyFlags::INPUT_DETERMINISTIC));
        assert!(!props.contains(PropertyFlags::OUTPUT_DETERMINISTIC));
    }
    
    #[test]
    fn test_coaccessible_property() {
        // Test FST where not all states can reach final states
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state(); // Dead end state

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));
        fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::one(), s3)); // Dead end

        let props = compute_properties(&fst);

        // Should be accessible but not coaccessible (s3 can't reach final state)
        assert!(props.contains(PropertyFlags::ACCESSIBLE));
        assert!(!props.contains(PropertyFlags::COACCESSIBLE));
        assert!(!props.contains(PropertyFlags::CONNECTED));
    }
    
    #[test]
    fn test_fully_connected_fst() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));

        let props = compute_properties(&fst);

        // Should be both accessible and coaccessible, hence connected
        assert!(props.contains(PropertyFlags::ACCESSIBLE));
        assert!(props.contains(PropertyFlags::COACCESSIBLE));
        assert!(props.contains(PropertyFlags::CONNECTED));
    }
    
    #[test]
    fn test_functional_with_epsilon_cycles() {
        // Create an FST that has epsilon cycles that actually affect functionality
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        let s3 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());
        fst.set_final(s3, TropicalWeight::one());

        // Create a scenario where epsilon cycles can create multiple outputs
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::epsilon(TropicalWeight::one(), s0)); // Epsilon cycle back to start
        fst.add_arc(s1, Arc::new(1, 2, TropicalWeight::one(), s2)); // Different output for same input
        fst.add_arc(s0, Arc::new(1, 3, TropicalWeight::one(), s3)); // Another path for input "1"

        let props = compute_properties(&fst);

        // Should not be functional due to non-determinism (multiple arcs with label 1 from s0)
        assert!(!props.contains(PropertyFlags::FUNCTIONAL));
        assert!(!props.contains(PropertyFlags::INPUT_DETERMINISTIC)); // This should be false now
        assert!(props.contains(PropertyFlags::EPSILONS));
        assert!(props.contains(PropertyFlags::CYCLIC));
    }
    
    #[test]
    fn test_empty_fst_special_cases() {
        let fst = VectorFst::<TropicalWeight>::new();
        let props = compute_properties(&fst);

        // Empty FST should be vacuously functional and deterministic
        assert!(props.contains(PropertyFlags::INPUT_DETERMINISTIC));
        assert!(props.contains(PropertyFlags::OUTPUT_DETERMINISTIC));
        assert!(props.contains(PropertyFlags::FUNCTIONAL));
        assert!(props.contains(PropertyFlags::ACCESSIBLE));
        assert!(props.contains(PropertyFlags::COACCESSIBLE));
        assert!(props.contains(PropertyFlags::CONNECTED));
    }
    
    #[test]
    fn test_no_final_states() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        // No final states set
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));

        let props = compute_properties(&fst);

        // Should not be coaccessible (no final states to reach)
        assert!(!props.contains(PropertyFlags::COACCESSIBLE));
        assert!(!props.contains(PropertyFlags::CONNECTED));
    }

    #[test]
    fn test_string_properties() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Linear chain (string-like structure)
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));

        let props = compute_properties(&fst);

        // Should be string-like
        assert!(props.contains(PropertyFlags::STRING));
        assert!(props.contains(PropertyFlags::ACYCLIC));
        assert!(props.contains(PropertyFlags::TOP_SORTED));
    }

    #[test]
    fn test_properties_bitwise_operations() {
        let props1 = PropertyFlags::ACCEPTOR | PropertyFlags::UNWEIGHTED;
        let props2 = PropertyFlags::ACYCLIC | PropertyFlags::UNWEIGHTED;

        // Test intersection
        let intersection = props1 & props2;
        assert!(intersection.contains(PropertyFlags::UNWEIGHTED));
        assert!(!intersection.contains(PropertyFlags::ACCEPTOR));
        assert!(!intersection.contains(PropertyFlags::ACYCLIC));

        // Test union
        let union = props1 | props2;
        assert!(union.contains(PropertyFlags::ACCEPTOR));
        assert!(union.contains(PropertyFlags::UNWEIGHTED));
        assert!(union.contains(PropertyFlags::ACYCLIC));

        // Test complement
        let complement = !props1;
        assert!(!complement.contains(PropertyFlags::ACCEPTOR));
        assert!(!complement.contains(PropertyFlags::UNWEIGHTED));
    }

    #[test]
    fn test_properties_compatibility() {
        // Test that certain properties are mutually exclusive or inclusive
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::one());

        let props = compute_properties(&fst);

        // Accessible and coaccessible should both be true for connected FSTs
        if props.contains(PropertyFlags::ACCESSIBLE) {
            assert!(props.contains(PropertyFlags::COACCESSIBLE));
        }

        // String implies acyclic
        if props.contains(PropertyFlags::STRING) {
            assert!(props.contains(PropertyFlags::ACYCLIC));
        }

        // Top-sorted implies acyclic
        if props.contains(PropertyFlags::TOP_SORTED) {
            assert!(props.contains(PropertyFlags::ACYCLIC));
        }
    }
}
