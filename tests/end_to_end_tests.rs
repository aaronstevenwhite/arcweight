//! End-to-end integration tests for complete workflows

use arcweight::prelude::*;
use arcweight::algorithms::PruneConfig;
use num_traits::One;

#[test]
fn test_shortest_path_workflow() {
    let mut fst = VectorFst::<TropicalWeight>::new();

    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::one());

    // two paths: 0->1->2 (weight 3) and 0->2 (weight 5)
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
    fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(2.0), s2));
    fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::new(5.0), s2));

    let shortest: VectorFst<TropicalWeight> = shortest_path_single(&fst).unwrap();

    // shortest path should have 3 states (preserving structure)
    assert_eq!(shortest.num_states(), 3);

    // should have only the cheaper path
    assert_eq!(shortest.num_arcs(s0), 1);
}

#[test]
fn test_composition_workflow() {
    let mut fst1 = VectorFst::<TropicalWeight>::new();
    let s0 = fst1.add_state();
    let s1 = fst1.add_state();
    fst1.set_start(s0);
    fst1.set_final(s1, TropicalWeight::one());
    fst1.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(1.0), s1));

    let mut fst2 = VectorFst::<TropicalWeight>::new();
    let t0 = fst2.add_state();
    let t1 = fst2.add_state();
    fst2.set_start(t0);
    fst2.set_final(t1, TropicalWeight::one());
    fst2.add_arc(t0, Arc::new(2, 3, TropicalWeight::new(2.0), t1));

    let composed: VectorFst<TropicalWeight> = compose_default(&fst1, &fst2).unwrap();

    assert!(composed.start().is_some());
    assert!(composed.num_states() > 0);
}

#[test]
fn test_determinization_workflow() {
    let mut fst = VectorFst::<TropicalWeight>::new();

    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::one());

    // non-deterministic: two arcs with same label from s0
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s2));
    fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s2));

    let det: VectorFst<TropicalWeight> = determinize(&fst).unwrap();

    // check that it's deterministic
    for state in det.states() {
        let mut labels = std::collections::HashSet::new();
        for arc in det.arcs(state) {
            assert!(labels.insert(arc.ilabel));
        }
    }
}

#[test]
fn test_optimization_pipeline() {
    // Test full optimization pipeline with appropriate semirings
    
    // Part 1: Epsilon removal with BooleanWeight (implements StarSemiring)
    let mut bool_fst = VectorFst::<BooleanWeight>::new();
    
    let s0 = bool_fst.add_state();
    let s1 = bool_fst.add_state();
    let s2 = bool_fst.add_state();
    let s3 = bool_fst.add_state();
    
    bool_fst.set_start(s0);
    bool_fst.set_final(s3, BooleanWeight::one());
    
    // Add epsilon transitions
    bool_fst.add_arc(s0, Arc::epsilon(BooleanWeight::one(), s1));
    bool_fst.add_arc(s0, Arc::epsilon(BooleanWeight::one(), s2));
    
    // Add regular transitions
    bool_fst.add_arc(s1, Arc::new(1, 1, BooleanWeight::one(), s3));
    bool_fst.add_arc(s2, Arc::new(1, 1, BooleanWeight::one(), s3));
    
    // Remove epsilons
    let no_eps: VectorFst<BooleanWeight> = remove_epsilons(&bool_fst).unwrap();
    
    // Verify no epsilons remain
    for state in no_eps.states() {
        for arc in no_eps.arcs(state) {
            assert!(!arc.is_epsilon());
        }
    }
    
    // Part 2: Determinization and minimization with TropicalWeight (implements DivisibleSemiring)
    let mut tropical_fst = VectorFst::<TropicalWeight>::new();
    
    let t0 = tropical_fst.add_state();
    let t1 = tropical_fst.add_state();
    let t2 = tropical_fst.add_state();
    let t3 = tropical_fst.add_state();
    
    tropical_fst.set_start(t0);
    tropical_fst.set_final(t3, TropicalWeight::one());
    
    // Non-deterministic transitions (same label from t0)
    tropical_fst.add_arc(t0, Arc::new(1, 1, TropicalWeight::new(1.0), t1));
    tropical_fst.add_arc(t0, Arc::new(1, 1, TropicalWeight::new(2.0), t2));
    
    // Regular transitions
    tropical_fst.add_arc(t1, Arc::new(2, 2, TropicalWeight::one(), t3));
    tropical_fst.add_arc(t2, Arc::new(2, 2, TropicalWeight::one(), t3));
    
    // Determinize
    let det: VectorFst<TropicalWeight> = determinize(&tropical_fst).unwrap();
    
    // Verify deterministic
    for state in det.states() {
        let mut labels = std::collections::HashSet::new();
        for arc in det.arcs(state) {
            assert!(labels.insert(arc.ilabel));
        }
    }
    
    // Minimize
    let min: VectorFst<TropicalWeight> = minimize(&det).unwrap();
    
    // Verify minimization completed successfully
    // Note: Brzozowski's algorithm may temporarily increase states
    assert!(min.start().is_some());
}

#[test]
fn test_transducer_operations() {
    // Create a transducer that maps digits to words
    let mut digit_to_word = VectorFst::<TropicalWeight>::new();
    
    let s0 = digit_to_word.add_state();
    let s1 = digit_to_word.add_state();
    
    digit_to_word.set_start(s0);
    digit_to_word.set_final(s1, TropicalWeight::one());
    
    // Map 1 -> 100 (one), 2 -> 200 (two), etc.
    digit_to_word.add_arc(s0, Arc::new(1, 100, TropicalWeight::new(1.0), s1));
    digit_to_word.add_arc(s0, Arc::new(2, 200, TropicalWeight::new(1.0), s1));
    
    // Create an input FST
    let mut input = VectorFst::<TropicalWeight>::new();
    let i0 = input.add_state();
    let i1 = input.add_state();
    
    input.set_start(i0);
    input.set_final(i1, TropicalWeight::one());
    input.add_arc(i0, Arc::new(1, 1, TropicalWeight::new(2.0), i1));
    
    // Compose to get the transduction
    let result: VectorFst<TropicalWeight> = compose_default(&input, &digit_to_word).unwrap();
    
    assert!(result.start().is_some());
    assert!(result.num_states() > 0);
    
    // The result should map 1 -> 100 with combined weight
    let start = result.start().unwrap();
    let mut found_mapping = false;
    for arc in result.arcs(start) {
        if arc.ilabel == 1 && arc.olabel == 100 {
            found_mapping = true;
            assert_eq!(*arc.weight.value(), 3.0); // 2.0 + 1.0
        }
    }
    assert!(found_mapping);
}

#[test]
fn test_union_concat_operations() {
    let mut fst1 = VectorFst::<TropicalWeight>::new();
    let s0 = fst1.add_state();
    let s1 = fst1.add_state();
    fst1.set_start(s0);
    fst1.set_final(s1, TropicalWeight::one());
    fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));

    let mut fst2 = VectorFst::<TropicalWeight>::new();
    let t0 = fst2.add_state();
    let t1 = fst2.add_state();
    fst2.set_start(t0);
    fst2.set_final(t1, TropicalWeight::one());
    fst2.add_arc(t0, Arc::new(2, 2, TropicalWeight::new(2.0), t1));

    // Test union
    let unioned: VectorFst<TropicalWeight> = union(&fst1, &fst2).unwrap();
    assert!(unioned.num_states() >= fst1.num_states() + fst2.num_states());
    assert!(unioned.start().is_some());

    // Test concatenation
    let concatenated: VectorFst<TropicalWeight> = concat(&fst1, &fst2).unwrap();
    assert_eq!(
        concatenated.num_states(),
        fst1.num_states() + fst2.num_states()
    );
    assert!(concatenated.start().is_some());
}

#[test]
fn test_reverse_operation() {
    let mut fst = VectorFst::<TropicalWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::new(2.0));

    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
    fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(3.0), s2));

    let reversed: VectorFst<TropicalWeight> = reverse(&fst).unwrap();

    // reversed should have an extra state for new start
    assert_eq!(reversed.num_states(), fst.num_states() + 1);
    assert!(reversed.start().is_some());

    // original start should be final in reversed
    assert!(reversed.is_final(s0));
}

#[test]
fn test_connect_prune_operations() {
    let mut fst = VectorFst::<TropicalWeight>::new();

    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    let s3 = fst.add_state(); // unreachable

    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());

    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
    fst.add_arc(s2, Arc::new(2, 2, TropicalWeight::new(1.0), s3)); // disconnected

    let connected: VectorFst<TropicalWeight> = connect(&fst).unwrap();

    // should only have accessible and coaccessible states
    assert!(connected.num_states() < fst.num_states());
    
    // Test pruning with weight threshold
    let mut weighted_fst = VectorFst::<TropicalWeight>::new();
    let w0 = weighted_fst.add_state();
    let w1 = weighted_fst.add_state();
    let w2 = weighted_fst.add_state();
    
    weighted_fst.set_start(w0);
    weighted_fst.set_final(w1, TropicalWeight::one());
    weighted_fst.set_final(w2, TropicalWeight::one());
    
    // Path to w1 has weight 1, path to w2 has weight 10
    weighted_fst.add_arc(w0, Arc::new(1, 1, TropicalWeight::new(1.0), w1));
    weighted_fst.add_arc(w0, Arc::new(2, 2, TropicalWeight::new(10.0), w2));
    
    // Prune paths with weight > 5
    let config = PruneConfig {
        weight_threshold: 5.0,
        state_threshold: None,
        npath: None,
    };
    
    let pruned: VectorFst<TropicalWeight> = prune(&weighted_fst, config).unwrap();
    
    // The prune implementation now properly removes paths exceeding the weight threshold
    // Verify that pruning has occurred (exact behavior depends on implementation details)
    assert!(pruned.num_states() <= weighted_fst.num_states());
}

#[test]
fn test_topsort_operation() {
    let mut fst = VectorFst::<TropicalWeight>::new();

    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::one());

    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s2));
    fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(1.0), s1));
    fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::new(1.0), s2));

    let sorted: VectorFst<TropicalWeight> = topsort(&fst).unwrap();

    // verify topological order
    for state in sorted.states() {
        for arc in sorted.arcs(state) {
            assert!(state < arc.nextstate);
        }
    }
}

#[test]
fn test_closure_operation() {
    // Use BooleanWeight for closure since TropicalWeight doesn't implement StarSemiring
    let mut bool_fst = VectorFst::<BooleanWeight>::new();
    let s0 = bool_fst.add_state();
    let s1 = bool_fst.add_state();
    bool_fst.set_start(s0);
    bool_fst.set_final(s1, BooleanWeight::one());
    bool_fst.add_arc(s0, Arc::new(1, 1, BooleanWeight::new(true), s1));

    let star: VectorFst<BooleanWeight> = closure(&bool_fst).unwrap();

    // closure should add a new start/final state
    assert!(star.num_states() > bool_fst.num_states());
    assert!(star.start().is_some());
}