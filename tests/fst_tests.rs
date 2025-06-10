//! Comprehensive tests for FST module

use arcweight::prelude::*;
use arcweight::fst::*;
use num_traits::One;
use proptest::prelude::*;

#[cfg(test)]
mod vector_fst_tests {
    use super::*;

    #[test]
    fn test_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();
        
        assert_eq!(fst.num_states(), 0);
        assert!(fst.is_empty());
        assert_eq!(fst.start(), None);
        assert_eq!(fst.num_arcs_total(), 0);
    }

    #[test]
    fn test_add_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        
        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        assert_eq!(fst.num_states(), 2);
        
        // FST is considered empty until start state is set
        fst.set_start(s0);
        assert!(!fst.is_empty());
    }

    #[test]
    fn test_start_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        
        assert_eq!(fst.start(), None);
        
        fst.set_start(s0);
        assert_eq!(fst.start(), Some(s0));
    }

    #[test]
    fn test_final_states() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        
        assert!(!fst.is_final(s0));
        assert!(!fst.is_final(s1));
        
        fst.set_final(s1, TropicalWeight::new(2.5));
        assert!(!fst.is_final(s0));
        assert!(fst.is_final(s1));
        assert_eq!(*fst.final_weight(s1).unwrap().value(), 2.5);
        
        fst.remove_final(s1);
        assert!(!fst.is_final(s1));
    }

    #[test]
    fn test_add_arcs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        
        assert_eq!(fst.num_arcs(s0), 0);
        assert_eq!(fst.num_arcs(s1), 0);
        
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(1.5), s1));
        fst.add_arc(s0, Arc::new(3, 4, TropicalWeight::new(2.0), s1));
        
        assert_eq!(fst.num_arcs(s0), 2);
        assert_eq!(fst.num_arcs(s1), 0);
        assert_eq!(fst.num_arcs_total(), 2);
    }

    #[test]
    fn test_arc_iteration() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(1.5), s1));
        fst.add_arc(s0, Arc::new(3, 4, TropicalWeight::new(2.0), s1));
        
        let arcs: Vec<_> = fst.arcs(s0).collect();
        assert_eq!(arcs.len(), 2);
        
        assert_eq!(arcs[0].ilabel, 1);
        assert_eq!(arcs[0].olabel, 2);
        assert_eq!(*arcs[0].weight.value(), 1.5);
        assert_eq!(arcs[0].nextstate, s1);
        
        assert_eq!(arcs[1].ilabel, 3);
        assert_eq!(arcs[1].olabel, 4);
        assert_eq!(*arcs[1].weight.value(), 2.0);
        assert_eq!(arcs[1].nextstate, s1);
    }

    #[test]
    fn test_states_iteration() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        
        let states: Vec<_> = fst.states().collect();
        assert_eq!(states, vec![s0, s1, s2]);
    }

    #[test]
    fn test_delete_arcs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s1));
        fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::new(3.0), s1));
        
        assert_eq!(fst.num_arcs(s0), 3);
        
        fst.delete_arc(s0, 1); // delete middle arc
        assert_eq!(fst.num_arcs(s0), 2);
        
        fst.delete_arcs(s0); // delete all arcs
        assert_eq!(fst.num_arcs(s0), 0);
    }

    #[test]
    fn test_clear() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        
        assert_eq!(fst.num_states(), 2);
        assert_eq!(fst.start(), Some(s0));
        assert!(fst.is_final(s1));
        
        fst.clear();
        
        assert_eq!(fst.num_states(), 0);
        assert_eq!(fst.start(), None);
        assert_eq!(fst.num_arcs_total(), 0);
    }

    #[test]
    fn test_reserve() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        
        // These shouldn't panic
        fst.reserve_states(100);
        
        let s0 = fst.add_state();
        fst.reserve_arcs(s0, 50);
    }

    #[test]
    fn test_epsilon_arcs() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        
        let epsilon_arc = Arc::epsilon(TropicalWeight::new(1.0), s1);
        fst.add_arc(s0, epsilon_arc);
        
        let arcs: Vec<_> = fst.arcs(s0).collect();
        assert_eq!(arcs.len(), 1);
        assert!(arcs[0].is_epsilon());
        assert_eq!(arcs[0].ilabel, NO_LABEL);
        assert_eq!(arcs[0].olabel, NO_LABEL);
    }

    #[test]
    fn test_expanded_fst_trait() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s1));
        
        let arcs_slice = fst.arcs_slice(s0);
        assert_eq!(arcs_slice.len(), 2);
        assert_eq!(arcs_slice[0].ilabel, 1);
        assert_eq!(arcs_slice[1].ilabel, 2);
    }
}

#[cfg(test)]
mod const_fst_tests {
    use super::*;

    #[test]
    fn test_const_fst_creation() {
        // Create a vector FST first
        let mut vector_fst = VectorFst::<TropicalWeight>::new();
        let s0 = vector_fst.add_state();
        let s1 = vector_fst.add_state();
        
        vector_fst.set_start(s0);
        vector_fst.set_final(s1, TropicalWeight::one());
        vector_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        
        // Convert to const FST
        let const_fst = ConstFst::from_fst(&vector_fst).unwrap();
        
        assert_eq!(const_fst.num_states(), vector_fst.num_states());
        assert_eq!(const_fst.start(), vector_fst.start());
        assert_eq!(const_fst.num_arcs(s0), vector_fst.num_arcs(s0));
        assert!(const_fst.is_final(s1));
    }

    #[test]
    fn test_const_fst_immutable() {
        let mut vector_fst = VectorFst::<TropicalWeight>::new();
        let s0 = vector_fst.add_state();
        let s1 = vector_fst.add_state();
        
        vector_fst.set_start(s0);
        vector_fst.set_final(s1, TropicalWeight::one());
        vector_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        
        let const_fst = ConstFst::from_fst(&vector_fst).unwrap();
        
        // Verify that const FST provides read-only access
        let arcs: Vec<_> = const_fst.arcs(s0).collect();
        assert_eq!(arcs.len(), 1);
        assert_eq!(arcs[0].ilabel, 1);
    }
}

#[cfg(test)]
mod cache_fst_tests {
    use super::*;

    #[test]
    fn test_cache_fst_creation() {
        let mut base_fst = VectorFst::<TropicalWeight>::new();
        let s0 = base_fst.add_state();
        let s1 = base_fst.add_state();
        
        base_fst.set_start(s0);
        base_fst.set_final(s1, TropicalWeight::one());
        base_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        
        let cache_fst = CacheFst::new(base_fst);
        
        assert_eq!(cache_fst.num_states(), 2);
        assert_eq!(cache_fst.start(), Some(s0));
        assert!(cache_fst.is_final(s1));
    }

    #[test]
    fn test_cache_fst_lazy_computation() {
        let mut base_fst = VectorFst::<TropicalWeight>::new();
        let s0 = base_fst.add_state();
        let s1 = base_fst.add_state();
        
        base_fst.set_start(s0);
        base_fst.set_final(s1, TropicalWeight::one());
        base_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        
        let cache_fst = CacheFst::new(base_fst);
        
        // Access arcs to trigger caching
        let arcs: Vec<_> = cache_fst.arcs(s0).collect();
        assert_eq!(arcs.len(), 1);
        
        // Second access should use cache
        let arcs2: Vec<_> = cache_fst.arcs(s0).collect();
        assert_eq!(arcs2.len(), 1);
        assert_eq!(arcs[0].ilabel, arcs2[0].ilabel);
    }
}

#[cfg(test)]
mod arc_tests {
    use super::*;

    #[test]
    fn test_arc_creation() {
        let arc = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        
        assert_eq!(arc.ilabel, 1);
        assert_eq!(arc.olabel, 2);
        assert_eq!(*arc.weight.value(), 3.0);
        assert_eq!(arc.nextstate, 4);
        assert!(!arc.is_epsilon());
    }

    #[test]
    fn test_epsilon_arc() {
        let arc = Arc::epsilon(TropicalWeight::new(1.0), 5);
        
        assert_eq!(arc.ilabel, NO_LABEL);
        assert_eq!(arc.olabel, NO_LABEL);
        assert_eq!(*arc.weight.value(), 1.0);
        assert_eq!(arc.nextstate, 5);
        assert!(arc.is_epsilon());
    }

    #[test]
    fn test_arc_with_same_labels() {
        let arc = Arc::new(3, 3, TropicalWeight::new(2.0), 6);
        
        assert_eq!(arc.ilabel, 3);
        assert_eq!(arc.olabel, 3);
        assert_eq!(*arc.weight.value(), 2.0);
        assert_eq!(arc.nextstate, 6);
        assert!(!arc.is_epsilon());
    }

    #[test]
    fn test_arc_display() {
        let arc = Arc::new(1, 2, TropicalWeight::new(3.0), 4);
        let display = format!("{}", arc);
        assert!(display.contains("1"));
        assert!(display.contains("2"));
        assert!(display.contains("3"));
        assert!(display.contains("4"));
    }

    #[test]
    fn test_arc_equality() {
        let arc1 = Arc::new(1, 1, TropicalWeight::new(1.0), 1);
        let arc2 = Arc::new(2, 2, TropicalWeight::new(2.0), 2);
        let arc3 = Arc::new(1, 1, TropicalWeight::new(1.0), 1);
        
        assert_ne!(arc1, arc2);
        assert_eq!(arc1, arc3);
    }
}

// Property-based tests
proptest! {
    #[test]
    fn fst_state_consistency(num_states in 1..50usize) {
        let mut fst = VectorFst::<TropicalWeight>::new();
        
        for _ in 0..num_states {
            fst.add_state();
        }
        
        assert_eq!(fst.num_states(), num_states);
        
        // All states should be valid
        for state in fst.states() {
            assert!(state < num_states as StateId);
        }
        
        // Arc counts should be consistent
        let total_arcs = fst.num_arcs_total();
        let sum_arcs: usize = fst.states().map(|s| fst.num_arcs(s)).sum();
        assert_eq!(total_arcs, sum_arcs);
    }
    
    #[test]
    fn arc_creation_consistency(
        ilabel: u32,
        olabel: u32,
        weight: f32,
        nextstate: u32,
    ) {
        let arc = Arc::new(ilabel, olabel, TropicalWeight::new(weight), nextstate);
        
        assert_eq!(arc.ilabel, ilabel);
        assert_eq!(arc.olabel, olabel);
        assert_eq!(*arc.weight.value(), weight);
        assert_eq!(arc.nextstate, nextstate);
        
        // Epsilon check
        let is_epsilon = ilabel == NO_LABEL && olabel == NO_LABEL;
        assert_eq!(arc.is_epsilon(), is_epsilon);
    }
    
    #[test]
    fn fst_final_weight_consistency(weight: f32) {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let state = fst.add_state();
        
        // Initially not final
        assert!(!fst.is_final(state));
        assert!(fst.final_weight(state).is_none());
        
        // Set final weight
        let w = TropicalWeight::new(weight);
        fst.set_final(state, w.clone());
        
        assert!(fst.is_final(state));
        assert_eq!(fst.final_weight(state).unwrap(), &w);
        
        // Remove final weight
        fst.remove_final(state);
        assert!(!fst.is_final(state));
    }
    
    #[test]
    fn fst_arc_operations(num_arcs in 0..20usize) {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        
        // Add arcs
        for i in 0..num_arcs {
            fst.add_arc(s0, Arc::new(i as u32, i as u32, TropicalWeight::new(i as f32), s1));
        }
        
        assert_eq!(fst.num_arcs(s0), num_arcs);
        assert_eq!(fst.num_arcs_total(), num_arcs);
        
        // Check arc iteration
        let arcs: Vec<_> = fst.arcs(s0).collect();
        assert_eq!(arcs.len(), num_arcs);
        
        // Check expanded FST trait
        let arcs_slice = fst.arcs_slice(s0);
        assert_eq!(arcs_slice.len(), num_arcs);
        
        // Delete all arcs
        fst.delete_arcs(s0);
        assert_eq!(fst.num_arcs(s0), 0);
        assert_eq!(fst.num_arcs_total(), 0);
    }
}