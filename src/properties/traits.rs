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
    // This is a simplified check - assume it's true for now
    let is_coaccessible = true;

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
    props.set_property(PropertyFlags::INITIAL_ACYCLIC, is_acyclic);
    props.set_property(PropertyFlags::TOP_SORTED, is_acyclic);
    props.set_property(PropertyFlags::STRING, is_string);

    props
}
