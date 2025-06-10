//! Integration tests

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
fn test_shortest_path() {
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
fn test_composition() {
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
fn test_determinization() {
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
fn test_union() {
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
    
    let unioned: VectorFst<TropicalWeight> = union(&fst1, &fst2).unwrap();
    
    // union should have states from both FSTs plus new start
    assert!(unioned.num_states() >= fst1.num_states() + fst2.num_states());
    assert!(unioned.start().is_some());
}

#[test]
fn test_concat() {
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
    
    let concatenated: VectorFst<TropicalWeight> = concat(&fst1, &fst2).unwrap();
    
    assert_eq!(concatenated.num_states(), fst1.num_states() + fst2.num_states());
    assert!(concatenated.start().is_some());
}

#[test]
fn test_closure() {
    let mut fst = VectorFst::<TropicalWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
    
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

#[test]
fn test_reverse() {
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
fn test_semiring_operations() {
    // tropical
    let t1 = TropicalWeight::new(3.0);
    let t2 = TropicalWeight::new(5.0);
    assert_eq!(t1.plus(&t2), TropicalWeight::new(3.0)); // min
    assert_eq!(t1.times(&t2), TropicalWeight::new(8.0)); // plus
    
    // probability
    let p1 = ProbabilityWeight::new(0.3);
    let p2 = ProbabilityWeight::new(0.5);
    assert_eq!(p1.plus(&p2), ProbabilityWeight::new(0.8));
    assert_eq!(p1.times(&p2), ProbabilityWeight::new(0.15));
    
    // boolean
    let b1 = BooleanWeight::new(true);
    let b2 = BooleanWeight::new(false);
    assert_eq!(b1.plus(&b2), BooleanWeight::new(true)); // or
    assert_eq!(b1.times(&b2), BooleanWeight::new(false)); // and
}

#[test]
fn test_epsilon_removal() {
    let mut fst = VectorFst::<TropicalWeight>::new();
    
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    
    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::one());
    
    // path with epsilon
    fst.add_arc(s0, Arc::epsilon(TropicalWeight::new(1.0), s1));
    fst.add_arc(s1, Arc::new(1, 1, TropicalWeight::new(2.0), s2));
    
    // Use BooleanWeight for epsilon removal since TropicalWeight doesn't implement StarSemiring
    let mut bool_fst = VectorFst::<BooleanWeight>::new();
    let s0 = bool_fst.add_state();
    let s1 = bool_fst.add_state();
    let s2 = bool_fst.add_state();
    
    bool_fst.set_start(s0);
    bool_fst.set_final(s2, BooleanWeight::one());
    
    bool_fst.add_arc(s0, Arc::epsilon(BooleanWeight::new(true), s1));
    bool_fst.add_arc(s1, Arc::new(1, 1, BooleanWeight::new(true), s2));
    
    let no_eps: VectorFst<BooleanWeight> = remove_epsilons(&bool_fst).unwrap();
    
    // check no epsilon transitions remain
    for state in no_eps.states() {
        for arc in no_eps.arcs(state) {
            assert!(!arc.is_epsilon());
        }
    }
}

#[test]
fn test_connect() {
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
}

#[test]
fn test_topsort() {
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