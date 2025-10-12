//! Integration tests for shortest distance and arc operation algorithms

use arcweight::prelude::*;

#[test]
fn test_shortest_distance_then_arc_sum() {
    // Test composition: compute distances, then sum duplicate arcs
    let mut fst = VectorFst::<TropicalWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::one());

    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
    fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(2.0), s2));

    // Compute distances
    let distances = shortest_distance(&fst).unwrap();
    assert_eq!(distances[s0 as usize], TropicalWeight::one());
    assert_eq!(distances[s1 as usize], TropicalWeight::new(1.0));
    assert_eq!(distances[s2 as usize], TropicalWeight::new(3.0));

    // Add duplicate arc
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));

    // Sum duplicates
    arc_sum(&mut fst).unwrap();
    assert_eq!(fst.num_arcs(s0), 1);

    // Verify distances still correct after arc_sum
    let distances2 = shortest_distance(&fst).unwrap();
    assert_eq!(distances2[s1 as usize], TropicalWeight::new(1.0)); // min(1,2) = 1
}

#[test]
fn test_arc_sum_then_arc_unique() {
    let mut fst = VectorFst::<TropicalWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();

    fst.set_start(s0);

    // Add arcs that will become duplicates after arc_sum
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));

    // Sum them
    arc_sum(&mut fst).unwrap();
    assert_eq!(fst.num_arcs(s0), 1);

    // Add exact duplicate
    let arc = fst.arcs(s0).next().unwrap().clone();
    fst.add_arc(s0, arc);

    // Remove exact duplicate
    arc_unique(&mut fst).unwrap();
    assert_eq!(fst.num_arcs(s0), 1);
}

#[test]
fn test_shortest_distance_after_arc_unique() {
    let mut fst = VectorFst::<TropicalWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());

    // Add exact duplicates
    let arc = Arc::new(1, 1, TropicalWeight::new(1.0), s1);
    fst.add_arc(s0, arc.clone());
    fst.add_arc(s0, arc.clone());
    fst.add_arc(s0, arc);

    arc_unique(&mut fst).unwrap();

    // Should have 1 arc with weight 1.0
    let distances = shortest_distance(&fst).unwrap();
    assert_eq!(distances[s1 as usize], TropicalWeight::new(1.0));
}

#[test]
fn test_all_algorithms_on_complex_fst() {
    let mut fst = VectorFst::<TropicalWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    let s3 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s3, TropicalWeight::one());

    // Build complex FST with duplicates
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1)); // Will be summed
    fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s2));
    fst.add_arc(s2, Arc::new(3, 3, TropicalWeight::new(1.0), s3));

    // Initial distances
    let dist1 = shortest_distance(&fst).unwrap();

    // Apply arc_sum
    arc_sum(&mut fst).unwrap();

    // Distances after arc_sum
    let dist2 = shortest_distance(&fst).unwrap();

    // Should be same or better (tropical min)
    assert!(dist2[s1 as usize] <= dist1[s1 as usize]);

    // Add exact duplicate
    let arc = Arc::new(2, 2, TropicalWeight::new(1.0), s2);
    fst.add_arc(s1, arc);

    // Remove duplicates
    arc_unique(&mut fst).unwrap();

    // Final distances
    let dist3 = shortest_distance(&fst).unwrap();
    assert_eq!(dist3[s0 as usize], TropicalWeight::one());
    assert_eq!(dist3[s1 as usize], TropicalWeight::new(1.0));
    assert_eq!(dist3[s2 as usize], TropicalWeight::new(2.0));
    assert_eq!(dist3[s3 as usize], TropicalWeight::new(3.0));
}

#[test]
fn test_probability_semiring_workflow() {
    let mut fst = VectorFst::<ProbabilityWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s1, ProbabilityWeight::one());

    // Multiple paths that sum in probability semiring
    fst.add_arc(s0, Arc::new(1, 1, ProbabilityWeight::new(0.3), s1));
    fst.add_arc(s0, Arc::new(1, 1, ProbabilityWeight::new(0.4), s1));

    // Before arc_sum: 2 arcs
    assert_eq!(fst.num_arcs(s0), 2);

    // Compute distance (sum of probabilities)
    let dist1 = shortest_distance(&fst).unwrap();

    // Sum arcs
    arc_sum(&mut fst).unwrap();
    assert_eq!(fst.num_arcs(s0), 1);

    // Distance should be same
    let dist2 = shortest_distance(&fst).unwrap();
    assert_eq!(dist1[s1 as usize], dist2[s1 as usize]);
    assert_eq!(dist2[s1 as usize], ProbabilityWeight::new(0.7));
}

#[test]
fn test_cyclic_fst_with_arc_operations() {
    let mut fst = VectorFst::<TropicalWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());

    // Create cycle with high weight (won't improve shortest distance)
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
    fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(10.0), s0));

    // Shortest distance should converge
    let distances = shortest_distance(&fst).unwrap();
    assert_eq!(distances[s0 as usize], TropicalWeight::one());
    assert_eq!(distances[s1 as usize], TropicalWeight::new(1.0));

    // Add duplicate of first arc
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));

    // Remove duplicate
    arc_unique(&mut fst).unwrap();
    assert_eq!(fst.num_arcs(s0), 1);

    // Distances unchanged
    let distances2 = shortest_distance(&fst).unwrap();
    assert_eq!(distances, distances2);
}

#[test]
fn test_empty_and_edge_cases() {
    // Empty FST
    let mut empty = VectorFst::<TropicalWeight>::new();
    arc_sum(&mut empty).unwrap();
    arc_unique(&mut empty).unwrap();

    // Single state
    let mut single = VectorFst::<TropicalWeight>::new();
    let s0 = single.add_state();
    single.set_start(s0);
    single.set_final(s0, TropicalWeight::one());

    let distances = shortest_distance(&single).unwrap();
    assert_eq!(distances[s0 as usize], TropicalWeight::one());

    arc_sum(&mut single).unwrap();
    arc_unique(&mut single).unwrap();

    // State with no arcs
    let mut no_arcs = VectorFst::<TropicalWeight>::new();
    let s0 = no_arcs.add_state();
    let _s1 = no_arcs.add_state();
    no_arcs.set_start(s0);

    arc_sum(&mut no_arcs).unwrap();
    arc_unique(&mut no_arcs).unwrap();
}
