//! FST property traits and computation

use crate::fst::Fst;
use crate::semiring::Semiring;
use bitflags::bitflags;

bitflags! {
    /// FST property flags
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PropertyFlags: u64 {
        /// FST is an acceptor (input = output)
        const ACCEPTOR = 1 << 0;
        /// FST has no epsilon transitions
        const NO_EPSILONS = 1 << 1;
        /// FST has no input epsilons
        const NO_INPUT_EPSILONS = 1 << 2;
        /// FST has no output epsilons
        const NO_OUTPUT_EPSILONS = 1 << 3;
        /// FST is deterministic on input
        const INPUT_DETERMINISTIC = 1 << 4;
        /// FST is deterministic on output
        const OUTPUT_DETERMINISTIC = 1 << 5;
        /// FST has functional property
        const FUNCTIONAL = 1 << 6;
        /// FST has no ambiguous paths
        const UNAMBIGUOUS = 1 << 7;
        /// All states are accessible
        const ACCESSIBLE = 1 << 8;
        /// All states are coaccessible
        const COACCESSIBLE = 1 << 9;
        /// FST is strongly connected
        const CONNECTED = 1 << 10;
        /// FST is cyclic
        const CYCLIC = 1 << 11;
        /// FST is acyclic
        const ACYCLIC = 1 << 12;
        /// FST has initial state
        const INITIAL_ACYCLIC = 1 << 13;
        /// FST is topologically sorted
        const TOP_SORTED = 1 << 14;
        /// Arcs are sorted by input label
        const INPUT_SORTED = 1 << 15;
        /// Arcs are sorted by output label
        const OUTPUT_SORTED = 1 << 16;
        /// FST is weighted
        const WEIGHTED = 1 << 17;
        /// FST is unweighted
        const UNWEIGHTED = 1 << 18;
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
    /// Check if a property is known
    pub fn is_known(&self, flag: PropertyFlags) -> bool {
        self.known.contains(flag)
    }
    
    /// Check if a property is set
    pub fn has_property(&self, flag: PropertyFlags) -> bool {
        self.is_known(flag) && self.properties.contains(flag)
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
        return props;
    }
    
    // check basic properties
    let mut has_epsilons = false;
    let mut has_input_epsilons = false;
    let mut has_output_epsilons = false;
    let mut is_acceptor = true;
    let mut is_unweighted = true;
    
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
    
    props.set_property(PropertyFlags::NO_EPSILONS, !has_epsilons);
    props.set_property(PropertyFlags::NO_INPUT_EPSILONS, !has_input_epsilons);
    props.set_property(PropertyFlags::NO_OUTPUT_EPSILONS, !has_output_epsilons);
    props.set_property(PropertyFlags::ACCEPTOR, is_acceptor);
    props.set_property(PropertyFlags::UNWEIGHTED, is_unweighted);
    props.set_property(PropertyFlags::WEIGHTED, !is_unweighted);
    
    props
}