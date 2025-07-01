//! Integration tests for FST operations and interactions

use arcweight::prelude::*;
use num_traits::One;

#[test]
fn test_basic_fst_operations() {
    let mut fst = VectorFst::<TropicalWeight>::new();

    let s0 = fst.add_state();
    let s1 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());

    fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(3.0), s1));

    assert_eq!(fst.num_states(), 2);
    assert_eq!(fst.num_arcs(s0), 1);
    assert_eq!(fst.start(), Some(s0));
    assert!(fst.is_final(s1));
}

#[test]
fn test_fst_types_conversion() {
    // Create a VectorFst
    let mut vector_fst = VectorFst::<TropicalWeight>::new();
    let s0 = vector_fst.add_state();
    let s1 = vector_fst.add_state();
    vector_fst.set_start(s0);
    vector_fst.set_final(s1, TropicalWeight::one());
    vector_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));

    // Convert to ConstFst
    let const_fst: ConstFst<TropicalWeight> = ConstFst::from_fst(&vector_fst).unwrap();
    assert_eq!(const_fst.num_states(), vector_fst.num_states());
    assert_eq!(const_fst.num_arcs_total(), vector_fst.num_arcs_total());
    assert_eq!(const_fst.start(), vector_fst.start());
}

#[test]
fn test_fst_iteration() {
    let mut fst = VectorFst::<TropicalWeight>::new();
    
    // Create a more complex FST
    let states: Vec<_> = (0..5).map(|_| fst.add_state()).collect();
    fst.set_start(states[0]);
    fst.set_final(states[4], TropicalWeight::one());

    // Add arcs in a pattern
    for i in 0..4 {
        fst.add_arc(states[i], Arc::new(i as u32 + 1, i as u32 + 1, TropicalWeight::new(i as f32), states[i + 1]));
    }

    // Test state iteration
    let collected_states: Vec<_> = fst.states().collect();
    assert_eq!(collected_states, states);

    // Test arc iteration
    for (i, state) in states[..4].iter().enumerate() {
        let arcs: Vec<_> = fst.arcs(*state).collect();
        assert_eq!(arcs.len(), 1);
        assert_eq!(arcs[0].ilabel, i as u32 + 1);
    }
}

#[test]
fn test_fst_properties_computation() {
    // Test various FST configurations and their properties
    
    // Empty FST
    let empty_fst = VectorFst::<TropicalWeight>::new();
    let empty_props = empty_fst.properties();
    assert!(empty_props.contains(FstProperties::ACYCLIC));

    // Linear FST
    let mut linear_fst = VectorFst::<TropicalWeight>::new();
    let s0 = linear_fst.add_state();
    let s1 = linear_fst.add_state();
    let s2 = linear_fst.add_state();
    linear_fst.set_start(s0);
    linear_fst.set_final(s2, TropicalWeight::one());
    linear_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
    linear_fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));

    let linear_props = linear_fst.properties();
    assert!(linear_props.contains(FstProperties::ACYCLIC));
    assert!(linear_props.contains(FstProperties::ACCEPTOR));
    assert!(linear_props.contains(FstProperties::UNWEIGHTED));

    // Cyclic FST
    let mut cyclic_fst = VectorFst::<TropicalWeight>::new();
    let s0 = cyclic_fst.add_state();
    cyclic_fst.set_start(s0);
    cyclic_fst.set_final(s0, TropicalWeight::one());
    cyclic_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s0));

    let cyclic_props = cyclic_fst.properties();
    assert!(!cyclic_props.contains(FstProperties::ACYCLIC));
}

#[test]
fn test_fst_mutability() {
    let mut fst = VectorFst::<TropicalWeight>::new();
    
    // Add and remove states
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    
    assert_eq!(fst.num_states(), 3);
    
    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::new(5.0));
    
    // Add arcs
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
    fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(2.0), s2));
    
    // Modify final weight
    assert!(fst.is_final(s2));
    assert_eq!(*fst.final_weight(s2).unwrap().value(), 5.0);
    
    fst.set_final(s2, TropicalWeight::new(10.0));
    assert_eq!(*fst.final_weight(s2).unwrap().value(), 10.0);
    
    // Remove final status
    fst.remove_final(s2);
    assert!(!fst.is_final(s2));
}

#[test]
fn test_fst_cloning_and_equality() {
    let mut original = VectorFst::<TropicalWeight>::new();
    let s0 = original.add_state();
    let s1 = original.add_state();
    original.set_start(s0);
    original.set_final(s1, TropicalWeight::one());
    original.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(3.0), s1));

    // Clone the FST
    let cloned = original.clone();
    
    // Verify they have the same structure
    assert_eq!(cloned.num_states(), original.num_states());
    assert_eq!(cloned.num_arcs_total(), original.num_arcs_total());
    assert_eq!(cloned.start(), original.start());
    
    // Verify independence
    original.add_state();
    assert_ne!(cloned.num_states(), original.num_states());
}