//! FST property traits and computation

use crate::fst::{Fst, StateId};
use crate::semiring::Semiring;
use bitflags::bitflags;
use std::collections::HashSet;

bitflags! {
    /// FST property flags
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PropertyFlags: u64 {
        /// FST is an acceptor (input = output)
        const ACCEPTOR = 1 << 0;
        /// FST has no epsilon transitions
        const NO_EPSILONS = 1 << 1;
        /// FST has epsilon transitions
        const EPSILONS = 1 << 2;
        /// FST has no input epsilons
        const NO_INPUT_EPSILONS = 1 << 3;
        /// FST has input epsilons
        const INPUT_EPSILONS = 1 << 4;
        /// FST has no output epsilons
        const NO_OUTPUT_EPSILONS = 1 << 5;
        /// FST has output epsilons
        const OUTPUT_EPSILONS = 1 << 6;
        /// FST is deterministic on input
        const INPUT_DETERMINISTIC = 1 << 7;
        /// FST is deterministic on output
        const OUTPUT_DETERMINISTIC = 1 << 8;
        /// FST has functional property
        const FUNCTIONAL = 1 << 9;
        /// FST has no ambiguous paths
        const UNAMBIGUOUS = 1 << 10;
        /// All states are accessible
        const ACCESSIBLE = 1 << 11;
        /// All states are coaccessible
        const COACCESSIBLE = 1 << 12;
        /// FST is strongly connected
        const CONNECTED = 1 << 13;
        /// FST is cyclic
        const CYCLIC = 1 << 14;
        /// FST is acyclic
        const ACYCLIC = 1 << 15;
        /// FST has initial state
        const INITIAL_ACYCLIC = 1 << 16;
        /// FST is topologically sorted
        const TOP_SORTED = 1 << 17;
        /// Arcs are sorted by input label
        const INPUT_SORTED = 1 << 18;
        /// Arcs are sorted by output label
        const OUTPUT_SORTED = 1 << 19;
        /// FST is weighted
        const WEIGHTED = 1 << 20;
        /// FST is unweighted
        const UNWEIGHTED = 1 << 21;
        /// FST is string-like (linear path)
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

/// Compute FST properties
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
        rec_stack: &mut HashSet<StateId>
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