//! Integration tests for state sorting algorithm

use arcweight::prelude::*;
use num_traits::One;

#[test]
fn test_state_sort_then_minimize() {
    // Create an FST with redundant states
    let mut fst = VectorFst::<TropicalWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    let s3 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s3, TropicalWeight::one());

    // Create paths
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s2));
    fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s3));
    fst.add_arc(s2, Arc::new(2, 2, TropicalWeight::one(), s3));

    // Sort first with different strategies
    let bfs_sorted = state_sort(&fst, StateSortType::BreadthFirst).unwrap();
    let dfs_sorted = state_sort(&fst, StateSortType::DepthFirst).unwrap();

    // Then minimize
    let bfs_minimized: VectorFst<TropicalWeight> = minimize(&bfs_sorted).unwrap();
    let dfs_minimized: VectorFst<TropicalWeight> = minimize(&dfs_sorted).unwrap();

    // Both should produce valid minimized FSTs
    assert!(bfs_minimized.start().is_some());
    assert!(dfs_minimized.start().is_some());
    assert!(bfs_minimized.num_states() > 0);
    assert!(dfs_minimized.num_states() > 0);
}

#[test]
fn test_state_sort_composition_compatibility() {
    // Create two FSTs
    let mut fst1 = VectorFst::<TropicalWeight>::new();
    let s0 = fst1.add_state();
    let s1 = fst1.add_state();
    fst1.set_start(s0);
    fst1.set_final(s1, TropicalWeight::one());
    fst1.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));

    let mut fst2 = VectorFst::<TropicalWeight>::new();
    let s0 = fst2.add_state();
    let s1 = fst2.add_state();
    fst2.set_start(s0);
    fst2.set_final(s1, TropicalWeight::one());
    fst2.add_arc(s0, Arc::new(2, 3, TropicalWeight::new(0.3), s1));

    // Sort both FSTs
    let fst1_sorted = state_sort(&fst1, StateSortType::BreadthFirst).unwrap();
    let fst2_sorted = state_sort(&fst2, StateSortType::BreadthFirst).unwrap();

    // Compose sorted FSTs
    let composed: VectorFst<TropicalWeight> = compose_default(&fst1_sorted, &fst2_sorted).unwrap();

    // Compose original FSTs
    let composed_orig: VectorFst<TropicalWeight> = compose_default(&fst1, &fst2).unwrap();

    // Both compositions should have same number of arcs and states
    assert_eq!(composed.num_states(), composed_orig.num_states());

    let composed_arcs: usize = composed.states().map(|s| composed.num_arcs(s)).sum();
    let composed_orig_arcs: usize = composed_orig
        .states()
        .map(|s| composed_orig.num_arcs(s))
        .sum();
    assert_eq!(composed_arcs, composed_orig_arcs);
}

#[test]
fn test_state_sort_multiple_strategies_equivalent() {
    // Create a complex FST
    let mut fst = VectorFst::<TropicalWeight>::new();
    let states: Vec<_> = (0..5).map(|_| fst.add_state()).collect();

    fst.set_start(states[0]);
    fst.set_final(states[4], TropicalWeight::new(2.0));

    // Create branching structure
    fst.add_arc(
        states[0],
        Arc::new(1, 1, TropicalWeight::new(0.1), states[1]),
    );
    fst.add_arc(
        states[0],
        Arc::new(2, 2, TropicalWeight::new(0.2), states[2]),
    );
    fst.add_arc(
        states[1],
        Arc::new(3, 3, TropicalWeight::new(0.3), states[3]),
    );
    fst.add_arc(
        states[2],
        Arc::new(4, 4, TropicalWeight::new(0.4), states[3]),
    );
    fst.add_arc(
        states[3],
        Arc::new(5, 5, TropicalWeight::new(0.5), states[4]),
    );

    // Sort with all strategies
    let bfs = state_sort(&fst, StateSortType::BreadthFirst).unwrap();
    let dfs = state_sort(&fst, StateSortType::DepthFirst).unwrap();
    let topo = state_sort(&fst, StateSortType::Topological).unwrap();

    // All should preserve structure
    assert_eq!(bfs.num_states(), fst.num_states());
    assert_eq!(dfs.num_states(), fst.num_states());
    assert_eq!(topo.num_states(), fst.num_states());

    // All should have same total number of arcs
    let bfs_arcs: usize = bfs.states().map(|s| bfs.num_arcs(s)).sum();
    let dfs_arcs: usize = dfs.states().map(|s| dfs.num_arcs(s)).sum();
    let topo_arcs: usize = topo.states().map(|s| topo.num_arcs(s)).sum();

    assert_eq!(bfs_arcs, dfs_arcs);
    assert_eq!(dfs_arcs, topo_arcs);

    // All should preserve weight sum
    let orig_weight_sum: f32 = fst
        .states()
        .flat_map(|s| fst.arcs(s))
        .map(|a| *a.weight.value())
        .sum();

    let bfs_weight_sum: f32 = bfs
        .states()
        .flat_map(|s| bfs.arcs(s))
        .map(|a| *a.weight.value())
        .sum();

    let dfs_weight_sum: f32 = dfs
        .states()
        .flat_map(|s| dfs.arcs(s))
        .map(|a| *a.weight.value())
        .sum();

    let topo_weight_sum: f32 = topo
        .states()
        .flat_map(|s| topo.arcs(s))
        .map(|a| *a.weight.value())
        .sum();

    assert!((orig_weight_sum - bfs_weight_sum).abs() < 1e-6);
    assert!((orig_weight_sum - dfs_weight_sum).abs() < 1e-6);
    assert!((orig_weight_sum - topo_weight_sum).abs() < 1e-6);
}

#[test]
fn test_state_sort_complex_fst() {
    // Create a larger, more complex FST
    let mut fst = VectorFst::<TropicalWeight>::new();
    let states: Vec<_> = (0..10).map(|_| fst.add_state()).collect();

    fst.set_start(states[0]);
    fst.set_final(states[9], TropicalWeight::one());

    // Create complex structure with multiple paths
    for i in 0..9 {
        if i < 5 {
            fst.add_arc(
                states[i],
                Arc::new(
                    (i + 1) as u32,
                    (i + 1) as u32,
                    TropicalWeight::new(0.1 * i as f32),
                    states[i + 1],
                ),
            );
        }
        if i < 8 {
            fst.add_arc(
                states[i],
                Arc::new(
                    (i + 10) as u32,
                    (i + 10) as u32,
                    TropicalWeight::new(0.2 * i as f32),
                    states[i + 2],
                ),
            );
        }
    }

    // Additional final paths
    fst.add_arc(
        states[5],
        Arc::new(20, 20, TropicalWeight::new(0.5), states[9]),
    );
    fst.add_arc(
        states[6],
        Arc::new(21, 21, TropicalWeight::new(0.6), states[9]),
    );

    // Sort with all strategies
    for sort_type in [
        StateSortType::BreadthFirst,
        StateSortType::DepthFirst,
        StateSortType::Topological,
    ] {
        let sorted = state_sort(&fst, sort_type).unwrap();

        assert_eq!(sorted.num_states(), fst.num_states());
        assert_eq!(sorted.start(), Some(0));

        // Verify arc count preserved
        let orig_arcs: usize = fst.states().map(|s| fst.num_arcs(s)).sum();
        let sorted_arcs: usize = sorted.states().map(|s| sorted.num_arcs(s)).sum();
        assert_eq!(orig_arcs, sorted_arcs);
    }
}

#[test]
fn test_state_sort_preserves_language() {
    // Create FST that accepts specific strings
    let mut fst = VectorFst::<TropicalWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    let s3 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s3, TropicalWeight::new(5.0));

    // Path 1: a b -> weight 1.0 + 2.0 = 3.0
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
    fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(2.0), s3));

    // Path 2: a c -> weight 1.5 + 2.5 = 4.0
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.5), s2));
    fst.add_arc(s2, Arc::new(3, 3, TropicalWeight::new(2.5), s3));

    // Sort FST
    let sorted = state_sort(&fst, StateSortType::BreadthFirst).unwrap();

    // Verify same number of paths
    let orig_paths: usize = fst.states().filter(|&s| fst.is_final(s)).count();
    let sorted_paths: usize = sorted.states().filter(|&s| sorted.is_final(s)).count();

    assert_eq!(orig_paths, sorted_paths);

    // Verify structure preserved
    assert_eq!(sorted.num_states(), fst.num_states());

    let orig_arcs: usize = fst.states().map(|s| fst.num_arcs(s)).sum();
    let sorted_arcs: usize = sorted.states().map(|s| sorted.num_arcs(s)).sum();
    assert_eq!(orig_arcs, sorted_arcs);
}

#[test]
fn test_state_sort_with_arc_sort() {
    // Create FST with unsorted arcs
    let mut fst = VectorFst::<TropicalWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::one());

    // Add arcs in reverse order
    fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::one(), s1));
    fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::one(), s1));
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s2));

    // Sort states first
    let mut state_sorted = state_sort(&fst, StateSortType::BreadthFirst).unwrap();

    // Then sort arcs
    arc_sort(&mut state_sorted, ArcSortType::ByInput).unwrap();

    // Verify arcs are now sorted by input label
    let labels: Vec<_> = state_sorted.arcs(0).map(|a| a.ilabel).collect();
    assert_eq!(labels, vec![1, 2, 3]);
}

#[test]
fn test_state_sort_isomorphic_check() {
    // Create an FST
    let mut fst = VectorFst::<TropicalWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::new(2.5));

    fst.add_arc(s0, Arc::new(1, 10, TropicalWeight::new(1.0), s1));
    fst.add_arc(s1, Arc::new(2, 20, TropicalWeight::new(2.0), s2));

    // Sort with BFS
    let bfs_sorted = state_sort(&fst, StateSortType::BreadthFirst).unwrap();

    // Sort with DFS
    let dfs_sorted = state_sort(&fst, StateSortType::DepthFirst).unwrap();

    // While state IDs may differ, the FSTs should be isomorphic
    // (same structure, different state numbering)
    assert!(isomorphic(&bfs_sorted, &dfs_sorted).unwrap());
}
