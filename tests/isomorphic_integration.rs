use arcweight::prelude::*;

#[test]
fn test_isomorphic_after_operations() {
    // Create an FST
    let mut fst1 = VectorFst::<TropicalWeight>::new();
    let s0 = fst1.add_state();
    let s1 = fst1.add_state();
    let s2 = fst1.add_state();
    fst1.set_start(s0);
    fst1.set_final(s2, TropicalWeight::new(1.0));
    fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
    fst1.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(0.3), s2));

    // Clone it
    let fst2 = fst1.clone();

    // They should be isomorphic
    assert!(isomorphic(&fst1, &fst2).unwrap());

    // Apply reverse operation to both
    let rev1: VectorFst<TropicalWeight> = reverse(&fst1).unwrap();
    let rev2: VectorFst<TropicalWeight> = reverse(&fst2).unwrap();

    // Reversed versions should also be isomorphic
    assert!(isomorphic(&rev1, &rev2).unwrap());
}

#[test]
fn test_isomorphic_with_different_construction_order() {
    // FST1: add arcs in one order
    let mut fst1 = VectorFst::<TropicalWeight>::new();
    let s0 = fst1.add_state();
    let s1 = fst1.add_state();
    let s2 = fst1.add_state();
    fst1.set_start(s0);
    fst1.set_final(s2, TropicalWeight::one());
    fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
    fst1.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(0.3), s2));
    fst1.add_arc(s1, Arc::new(3, 3, TropicalWeight::new(0.2), s2));

    // FST2: add arcs in different order but same structure
    let mut fst2 = VectorFst::<TropicalWeight>::new();
    let t0 = fst2.add_state();
    let t1 = fst2.add_state();
    let t2 = fst2.add_state();
    fst2.set_start(t0);
    fst2.set_final(t2, TropicalWeight::one());
    // Different arc addition order
    fst2.add_arc(t1, Arc::new(3, 3, TropicalWeight::new(0.2), t2));
    fst2.add_arc(t0, Arc::new(2, 2, TropicalWeight::new(0.3), t2));
    fst2.add_arc(t0, Arc::new(1, 1, TropicalWeight::new(0.5), t1));

    assert!(isomorphic(&fst1, &fst2).unwrap());
}

#[test]
fn test_isomorphic_complex_fst() {
    // Build a more complex FST with multiple paths
    let mut fst1 = VectorFst::<TropicalWeight>::new();
    let states1: Vec<_> = (0..6).map(|_| fst1.add_state()).collect();
    fst1.set_start(states1[0]);
    fst1.set_final(states1[5], TropicalWeight::new(2.0));

    // Create a diamond structure with additional complexity
    fst1.add_arc(
        states1[0],
        Arc::new(1, 1, TropicalWeight::new(0.1), states1[1]),
    );
    fst1.add_arc(
        states1[0],
        Arc::new(2, 2, TropicalWeight::new(0.2), states1[2]),
    );
    fst1.add_arc(
        states1[1],
        Arc::new(3, 3, TropicalWeight::new(0.3), states1[3]),
    );
    fst1.add_arc(
        states1[2],
        Arc::new(4, 4, TropicalWeight::new(0.4), states1[3]),
    );
    fst1.add_arc(
        states1[3],
        Arc::new(5, 5, TropicalWeight::new(0.5), states1[4]),
    );
    fst1.add_arc(
        states1[4],
        Arc::new(6, 6, TropicalWeight::new(0.6), states1[5]),
    );
    // Add some epsilon transitions
    fst1.add_arc(
        states1[1],
        Arc::epsilon(TropicalWeight::new(0.15), states1[4]),
    );

    // Build FST2 with same structure
    let mut fst2 = VectorFst::<TropicalWeight>::new();
    let states2: Vec<_> = (0..6).map(|_| fst2.add_state()).collect();
    fst2.set_start(states2[0]);
    fst2.set_final(states2[5], TropicalWeight::new(2.0));

    fst2.add_arc(
        states2[0],
        Arc::new(1, 1, TropicalWeight::new(0.1), states2[1]),
    );
    fst2.add_arc(
        states2[0],
        Arc::new(2, 2, TropicalWeight::new(0.2), states2[2]),
    );
    fst2.add_arc(
        states2[1],
        Arc::new(3, 3, TropicalWeight::new(0.3), states2[3]),
    );
    fst2.add_arc(
        states2[2],
        Arc::new(4, 4, TropicalWeight::new(0.4), states2[3]),
    );
    fst2.add_arc(
        states2[3],
        Arc::new(5, 5, TropicalWeight::new(0.5), states2[4]),
    );
    fst2.add_arc(
        states2[4],
        Arc::new(6, 6, TropicalWeight::new(0.6), states2[5]),
    );
    fst2.add_arc(
        states2[1],
        Arc::epsilon(TropicalWeight::new(0.15), states2[4]),
    );

    assert!(isomorphic(&fst1, &fst2).unwrap());
}

#[test]
fn test_non_isomorphic_after_modification() {
    // Create two initially isomorphic FSTs
    let mut fst1 = VectorFst::<TropicalWeight>::new();
    let s0 = fst1.add_state();
    let s1 = fst1.add_state();
    fst1.set_start(s0);
    fst1.set_final(s1, TropicalWeight::new(1.0));
    fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));

    let mut fst2 = fst1.clone();

    // Initially isomorphic
    assert!(isomorphic(&fst1, &fst2).unwrap());

    // Modify fst2 by adding an arc
    fst2.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(0.3), s1));

    // No longer isomorphic
    assert!(!isomorphic(&fst1, &fst2).unwrap());
}

#[test]
fn test_isomorphic_with_concat() {
    // Create simple FSTs
    let mut fst_a = VectorFst::<TropicalWeight>::new();
    let a0 = fst_a.add_state();
    let a1 = fst_a.add_state();
    fst_a.set_start(a0);
    fst_a.set_final(a1, TropicalWeight::one());
    fst_a.add_arc(a0, Arc::new(1, 1, TropicalWeight::new(0.5), a1));

    let mut fst_b = VectorFst::<TropicalWeight>::new();
    let b0 = fst_b.add_state();
    let b1 = fst_b.add_state();
    fst_b.set_start(b0);
    fst_b.set_final(b1, TropicalWeight::one());
    fst_b.add_arc(b0, Arc::new(2, 2, TropicalWeight::new(0.3), b1));

    // Concatenate in same order
    let concat1: VectorFst<TropicalWeight> = concat(&fst_a, &fst_b).unwrap();
    let concat2: VectorFst<TropicalWeight> = concat(&fst_a, &fst_b).unwrap();

    // Results should be isomorphic
    assert!(isomorphic(&concat1, &concat2).unwrap());
}

#[test]
fn test_isomorphic_with_union() {
    // Create simple FSTs
    let mut fst_a = VectorFst::<TropicalWeight>::new();
    let a0 = fst_a.add_state();
    let a1 = fst_a.add_state();
    fst_a.set_start(a0);
    fst_a.set_final(a1, TropicalWeight::new(1.0));
    fst_a.add_arc(a0, Arc::new(1, 1, TropicalWeight::new(0.5), a1));

    let mut fst_b = VectorFst::<TropicalWeight>::new();
    let b0 = fst_b.add_state();
    let b1 = fst_b.add_state();
    fst_b.set_start(b0);
    fst_b.set_final(b1, TropicalWeight::new(2.0));
    fst_b.add_arc(b0, Arc::new(2, 2, TropicalWeight::new(0.3), b1));

    // Union in same order
    let union1: VectorFst<TropicalWeight> = union(&fst_a, &fst_b).unwrap();
    let union2: VectorFst<TropicalWeight> = union(&fst_a, &fst_b).unwrap();

    // Results should be isomorphic
    assert!(isomorphic(&union1, &union2).unwrap());
}

#[test]
fn test_isomorphic_self_loop_fst() {
    // FST with self-loop
    let mut fst1 = VectorFst::<TropicalWeight>::new();
    let s0 = fst1.add_state();
    fst1.set_start(s0);
    fst1.set_final(s0, TropicalWeight::one());
    fst1.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s0));

    let mut fst2 = VectorFst::<TropicalWeight>::new();
    let t0 = fst2.add_state();
    fst2.set_start(t0);
    fst2.set_final(t0, TropicalWeight::one());
    fst2.add_arc(t0, Arc::new(1, 1, TropicalWeight::new(0.5), t0));

    assert!(isomorphic(&fst1, &fst2).unwrap());
}
