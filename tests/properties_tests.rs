//! Comprehensive tests for properties module

use arcweight::prelude::*;
use arcweight::properties::*;
use proptest::prelude::*;

#[cfg(test)]
mod fst_properties_tests {
    use super::*;

    #[test]
    fn test_empty_fst_properties() {
        let fst = VectorFst::<TropicalWeight>::new();
        let props = fst.properties();
        
        // Empty FST should have certain properties
        assert!(props.contains(FstProperties::ACCEPTOR));
        assert!(props.contains(FstProperties::UNWEIGHTED));
        assert!(props.contains(FstProperties::ACYCLIC));
        assert!(props.contains(FstProperties::INITIAL_ACYCLIC));
        assert!(props.contains(FstProperties::TOP_SORTED));
        assert!(props.contains(FstProperties::ACCESSIBLE));
        assert!(props.contains(FstProperties::COACCESSIBLE));
    }

    #[test]
    fn test_single_state_fst_properties() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::one());
        
        let props = fst.properties();
        
        assert!(props.contains(FstProperties::ACCEPTOR));
        assert!(props.contains(FstProperties::UNWEIGHTED));
        assert!(props.contains(FstProperties::ACYCLIC));
        assert!(props.contains(FstProperties::ACCESSIBLE));
        assert!(props.contains(FstProperties::COACCESSIBLE));
    }

    #[test]
    fn test_simple_acceptor_properties() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        
        let props = fst.properties();
        
        // Should be an acceptor (input == output labels)
        assert!(props.contains(FstProperties::ACCEPTOR));
        assert!(props.contains(FstProperties::UNWEIGHTED));
        assert!(props.contains(FstProperties::ACYCLIC));
        assert!(props.contains(FstProperties::ACCESSIBLE));
        assert!(props.contains(FstProperties::COACCESSIBLE));
    }

    #[test]
    fn test_transducer_properties() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::one(), s1)); // Different input/output
        
        let props = fst.properties();
        
        // Should NOT be an acceptor
        assert!(!props.contains(FstProperties::ACCEPTOR));
        assert!(props.contains(FstProperties::UNWEIGHTED));
    }

    #[test]
    fn test_weighted_fst_properties() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.5), s1)); // Non-unit weight
        
        let props = fst.properties();
        
        // Should be weighted
        assert!(!props.contains(FstProperties::UNWEIGHTED));
        assert!(props.contains(FstProperties::ACCEPTOR));
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
        
        let props = fst.properties();
        
        // Should be cyclic
        assert!(!props.contains(FstProperties::ACYCLIC));
        assert!(props.contains(FstProperties::ACCESSIBLE));
        assert!(props.contains(FstProperties::COACCESSIBLE));
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
        
        let props = fst.properties();
        
        // Should have epsilon transitions
        assert!(props.contains(FstProperties::EPSILONS));
        assert!(props.contains(FstProperties::I_EPSILONS));
        assert!(props.contains(FstProperties::O_EPSILONS));
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
        
        let props = fst.properties();
        
        // Should NOT be deterministic
        assert!(!props.contains(FstProperties::I_DETERMINISTIC));
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
        
        let props = fst.properties();
        
        // Should be string-like
        assert!(props.contains(FstProperties::STRING));
        assert!(props.contains(FstProperties::ACYCLIC));
        assert!(props.contains(FstProperties::TOP_SORTED));
    }

    #[test]
    fn test_properties_bitwise_operations() {
        let props1 = FstProperties::ACCEPTOR | FstProperties::UNWEIGHTED;
        let props2 = FstProperties::ACYCLIC | FstProperties::UNWEIGHTED;
        
        // Test intersection
        let intersection = props1 & props2;
        assert!(intersection.contains(FstProperties::UNWEIGHTED));
        assert!(!intersection.contains(FstProperties::ACCEPTOR));
        assert!(!intersection.contains(FstProperties::ACYCLIC));
        
        // Test union
        let union = props1 | props2;
        assert!(union.contains(FstProperties::ACCEPTOR));
        assert!(union.contains(FstProperties::UNWEIGHTED));
        assert!(union.contains(FstProperties::ACYCLIC));
        
        // Test complement
        let complement = !props1;
        assert!(!complement.contains(FstProperties::ACCEPTOR));
        assert!(!complement.contains(FstProperties::UNWEIGHTED));
    }

    #[test]
    fn test_properties_compatibility() {
        // Test that certain properties are mutually exclusive or inclusive
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::one());
        
        let props = fst.properties();
        
        // Accessible and coaccessible should both be true for connected FSTs
        if props.contains(FstProperties::ACCESSIBLE) {
            assert!(props.contains(FstProperties::COACCESSIBLE));
        }
        
        // String implies acyclic
        if props.contains(FstProperties::STRING) {
            assert!(props.contains(FstProperties::ACYCLIC));
        }
        
        // Top-sorted implies acyclic
        if props.contains(FstProperties::TOP_SORTED) {
            assert!(props.contains(FstProperties::ACYCLIC));
        }
    }
}

// Property-based tests
proptest! {
    #[test]
    fn properties_consistency(num_states in 1..10usize, num_arcs in 0..20usize) {
        let mut fst = VectorFst::<TropicalWeight>::new();
        
        for _ in 0..num_states {
            fst.add_state();
        }
        
        if num_states > 0 {
            fst.set_start(0);
            if num_states > 1 {
                fst.set_final((num_states - 1) as u32, TropicalWeight::one());
            }
            
            for i in 0..num_arcs.min(num_states * 3) {
                let from = (i % num_states) as u32;
                let to = ((i + 1) % num_states) as u32;
                let label = (i % 5) as u32 + 1;
                fst.add_arc(from, Arc::new(label, label, TropicalWeight::one(), to));
            }
        }
        
        let props = fst.properties();
        
        // Basic consistency checks
        if props.contains(FstProperties::STRING) {
            prop_assert!(props.contains(FstProperties::ACYCLIC));
        }
        
        if props.contains(FstProperties::TOP_SORTED) {
            prop_assert!(props.contains(FstProperties::ACYCLIC));
        }
        
        // Empty FST should have specific properties
        if fst.num_states() == 0 {
            prop_assert!(props.contains(FstProperties::ACYCLIC));
        }
    }
    
    #[test]
    fn acceptor_property_correctness(
        labels: Vec<u32>,
        different_output: bool,
    ) {
        let mut fst = VectorFst::<TropicalWeight>::new();
        
        if !labels.is_empty() {
            for _i in 0..=labels.len() {
                fst.add_state();
            }
            
            fst.set_start(0);
            fst.set_final(labels.len() as u32, TropicalWeight::one());
            
            for (i, &label) in labels.iter().enumerate() {
                let output_label = if different_output && i == 0 {
                    label + 1000 // Make it different
                } else {
                    label
                };
                
                fst.add_arc(
                    i as u32,
                    Arc::new(label, output_label, TropicalWeight::one(), (i + 1) as u32)
                );
            }
            
            let props = fst.properties();
            
            // If we made any output different, should not be acceptor
            if different_output {
                prop_assert!(!props.contains(FstProperties::ACCEPTOR));
            } else {
                prop_assert!(props.contains(FstProperties::ACCEPTOR));
            }
        }
    }
    
    #[test]
    fn epsilon_property_detection(has_epsilon: bool) {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        
        if has_epsilon {
            fst.add_arc(s0, Arc::epsilon(TropicalWeight::one(), s1));
        } else {
            fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
        }
        
        let props = fst.properties();
        
        if has_epsilon {
            prop_assert!(props.contains(FstProperties::EPSILONS));
            prop_assert!(props.contains(FstProperties::I_EPSILONS));
            prop_assert!(props.contains(FstProperties::O_EPSILONS));
        } else {
            prop_assert!(!props.contains(FstProperties::EPSILONS));
            prop_assert!(!props.contains(FstProperties::I_EPSILONS));
            prop_assert!(!props.contains(FstProperties::O_EPSILONS));
        }
    }
    
    #[test]
    fn weighted_property_detection(use_non_unit_weight: bool) {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        
        let weight = if use_non_unit_weight {
            TropicalWeight::new(2.5)
        } else {
            TropicalWeight::one()
        };
        
        fst.add_arc(s0, Arc::new(1, 1, weight, s1));
        
        let props = fst.properties();
        
        if use_non_unit_weight {
            prop_assert!(!props.contains(FstProperties::UNWEIGHTED));
        } else {
            prop_assert!(props.contains(FstProperties::UNWEIGHTED));
        }
    }
}