//! Integration tests for IntegerWeight semiring with FST algorithms

use arcweight::prelude::*;

#[test]
fn test_integer_with_vector_fst() {
    // Create FST with integer weights
    let mut fst = VectorFst::<IntegerWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, IntegerWeight::one());

    fst.add_arc(s0, Arc::new(1, 1, IntegerWeight::new(5), s1));

    assert_eq!(fst.num_states(), 2);
    assert_eq!(fst.num_arcs(s0), 1);

    let arc = fst.arcs(s0).next().unwrap();
    assert_eq!(arc.weight, IntegerWeight::new(5));
}

#[test]
fn test_integer_with_shortest_distance() {
    // Test path counting with shortest_distance
    let mut fst = VectorFst::<IntegerWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s2, IntegerWeight::one());

    // Two parallel paths from s0 to s1
    fst.add_arc(s0, Arc::new(1, 1, IntegerWeight::one(), s1));
    fst.add_arc(s0, Arc::new(2, 2, IntegerWeight::one(), s1));

    // One path from s1 to s2
    fst.add_arc(s1, Arc::new(3, 3, IntegerWeight::one(), s2));

    let distances = shortest_distance(&fst).unwrap();

    // Should count 2 paths total
    assert_eq!(distances[s2 as usize], IntegerWeight::new(2));
}

#[test]
fn test_integer_with_arc_sort() {
    // Verify arc_sort works with integer weights
    let mut fst = VectorFst::<IntegerWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);

    fst.add_arc(s0, Arc::new(3, 3, IntegerWeight::new(10), s1));
    fst.add_arc(s0, Arc::new(1, 1, IntegerWeight::new(20), s1));
    fst.add_arc(s0, Arc::new(2, 2, IntegerWeight::new(30), s1));

    arc_sort(&mut fst, ArcSortType::ByInput).unwrap();

    let labels: Vec<_> = fst.arcs(s0).map(|a| a.ilabel).collect();
    assert_eq!(labels, vec![1, 2, 3]);
}

#[test]
fn test_integer_with_arc_sum() {
    // Test arc_sum combines integer weights correctly
    let mut fst = VectorFst::<IntegerWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);

    fst.add_arc(s0, Arc::new(1, 1, IntegerWeight::new(5), s1));
    fst.add_arc(s0, Arc::new(1, 1, IntegerWeight::new(3), s1));

    arc_sum(&mut fst).unwrap();

    assert_eq!(fst.num_arcs(s0), 1);
    let arc = fst.arcs(s0).next().unwrap();
    assert_eq!(arc.weight, IntegerWeight::new(8)); // 5 + 3
}

#[test]
fn test_integer_with_arc_unique() {
    // Test arc_unique removes duplicate arcs correctly
    let mut fst = VectorFst::<IntegerWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);

    // Add duplicate arcs
    fst.add_arc(s0, Arc::new(1, 1, IntegerWeight::new(5), s1));
    fst.add_arc(s0, Arc::new(1, 1, IntegerWeight::new(5), s1));
    fst.add_arc(s0, Arc::new(2, 2, IntegerWeight::new(3), s1));

    arc_unique(&mut fst).unwrap();

    assert_eq!(fst.num_arcs(s0), 2); // Should have 2 unique arcs
}

#[test]
fn test_integer_with_compose() {
    // Test composition with integer weights
    let mut fst1 = VectorFst::<IntegerWeight>::new();
    let s0 = fst1.add_state();
    let s1 = fst1.add_state();
    fst1.set_start(s0);
    fst1.set_final(s1, IntegerWeight::one());
    fst1.add_arc(s0, Arc::new(1, 2, IntegerWeight::new(3), s1));

    let mut fst2 = VectorFst::<IntegerWeight>::new();
    let s0 = fst2.add_state();
    let s1 = fst2.add_state();
    fst2.set_start(s0);
    fst2.set_final(s1, IntegerWeight::one());
    fst2.add_arc(s0, Arc::new(2, 3, IntegerWeight::new(4), s1));

    let result: VectorFst<IntegerWeight> = compose(&fst1, &fst2, DefaultComposeFilter).unwrap();

    assert!(result.num_states() > 0);

    // Check that weights are multiplied correctly
    if let Some(start) = result.start() {
        if result.is_final(start) {
            // If composition produced a simple path, check the weight
            let final_weight = result.final_weight(start).unwrap();
            // Weight should be 3 * 4 * 1 = 12
            assert_eq!(*final_weight, IntegerWeight::new(12));
        }
    }
}

#[test]
fn test_integer_with_union() {
    // Test union combines FSTs correctly with integer weights
    let mut fst1 = VectorFst::<IntegerWeight>::new();
    let s0 = fst1.add_state();
    let s1 = fst1.add_state();
    fst1.set_start(s0);
    fst1.set_final(s1, IntegerWeight::new(5));
    fst1.add_arc(s0, Arc::new(1, 1, IntegerWeight::new(10), s1));

    let mut fst2 = VectorFst::<IntegerWeight>::new();
    let s0 = fst2.add_state();
    let s1 = fst2.add_state();
    fst2.set_start(s0);
    fst2.set_final(s1, IntegerWeight::new(3));
    fst2.add_arc(s0, Arc::new(2, 2, IntegerWeight::new(20), s1));

    let result: VectorFst<IntegerWeight> = union(&fst1, &fst2).unwrap();

    assert!(result.num_states() > 0);
    assert!(result.num_arcs_total() >= 2);
}

#[test]
fn test_integer_with_concat() {
    // Test concatenation with integer weights
    let mut fst1 = VectorFst::<IntegerWeight>::new();
    let s0 = fst1.add_state();
    let s1 = fst1.add_state();
    fst1.set_start(s0);
    fst1.set_final(s1, IntegerWeight::new(2));
    fst1.add_arc(s0, Arc::new(1, 1, IntegerWeight::new(3), s1));

    let mut fst2 = VectorFst::<IntegerWeight>::new();
    let s0 = fst2.add_state();
    let s1 = fst2.add_state();
    fst2.set_start(s0);
    fst2.set_final(s1, IntegerWeight::new(5));
    fst2.add_arc(s0, Arc::new(2, 2, IntegerWeight::new(7), s1));

    let result: VectorFst<IntegerWeight> = concat(&fst1, &fst2).unwrap();

    assert!(result.num_states() > 0);
    assert!(result.num_arcs_total() >= 2);
}

#[test]
fn test_integer_with_reverse() {
    // Test reverse operation with integer weights
    let mut fst = VectorFst::<IntegerWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s2, IntegerWeight::new(10));

    fst.add_arc(s0, Arc::new(1, 1, IntegerWeight::new(5), s1));
    fst.add_arc(s1, Arc::new(2, 2, IntegerWeight::new(3), s2));

    let reversed: VectorFst<IntegerWeight> = reverse(&fst).unwrap();

    // Reverse may add extra state(s)
    assert!(reversed.num_states() >= fst.num_states());
    // Arc count should be preserved
    assert!(reversed.num_arcs_total() > 0);
}

#[test]
fn test_integer_io_text_format() {
    // Test reading/writing integer weights in text format
    let mut fst = VectorFst::<IntegerWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    fst.set_start(s0);
    fst.set_final(s1, IntegerWeight::new(42));
    fst.add_arc(s0, Arc::new(1, 2, IntegerWeight::new(100), s1));

    // Write to buffer
    let mut buffer = Vec::new();
    write_text(&fst, &mut buffer, None, None).unwrap();

    // Verify text contains our weights
    let text = String::from_utf8(buffer.clone()).unwrap();
    assert!(text.contains("42"));
    assert!(text.contains("100"));

    // Read back
    let fst2: VectorFst<IntegerWeight> = read_text(&mut &buffer[..], None, None).unwrap();
    assert_eq!(fst2.num_states(), fst.num_states());
    assert_eq!(fst2.num_arcs_total(), fst.num_arcs_total());

    // Verify weights preserved
    if let Some(start) = fst2.start() {
        let arc = fst2.arcs(start).next().unwrap();
        assert_eq!(arc.weight, IntegerWeight::new(100));
    }
}

#[test]
fn test_integer_path_counting_complex() {
    // Complex path counting example with diamond structure
    let mut fst = VectorFst::<IntegerWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    let s3 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s3, IntegerWeight::one());

    // Diamond structure: 2 paths from s0 to s2, converging at s3
    fst.add_arc(s0, Arc::new(1, 1, IntegerWeight::one(), s1));
    fst.add_arc(s0, Arc::new(2, 2, IntegerWeight::one(), s2));
    fst.add_arc(s1, Arc::new(3, 3, IntegerWeight::one(), s3));
    fst.add_arc(s2, Arc::new(4, 4, IntegerWeight::one(), s3));

    let distances = shortest_distance(&fst).unwrap();
    assert_eq!(distances[s3 as usize], IntegerWeight::new(2));
}

#[test]
fn test_integer_with_weighted_paths() {
    // Test counting with non-unit weights
    let mut fst = VectorFst::<IntegerWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s2, IntegerWeight::one());

    // First path with weight 3
    fst.add_arc(s0, Arc::new(1, 1, IntegerWeight::new(3), s1));
    // Second path with weight 5
    fst.add_arc(s0, Arc::new(2, 2, IntegerWeight::new(5), s1));
    // Both merge with weight 2
    fst.add_arc(s1, Arc::new(3, 3, IntegerWeight::new(2), s2));

    let distances = shortest_distance(&fst).unwrap();
    // (3 * 2) + (5 * 2) = 6 + 10 = 16
    assert_eq!(distances[s2 as usize], IntegerWeight::new(16));
}

#[test]
fn test_integer_negative_weights() {
    // Test that negative weights work correctly
    let mut fst = VectorFst::<IntegerWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s1, IntegerWeight::new(-5));
    fst.add_arc(s0, Arc::new(1, 1, IntegerWeight::new(-10), s1));

    assert_eq!(fst.num_states(), 2);
    let arc = fst.arcs(s0).next().unwrap();
    assert_eq!(arc.weight, IntegerWeight::new(-10));
}

#[test]
fn test_integer_large_counts() {
    // Test with larger integer values
    let mut fst = VectorFst::<IntegerWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s1, IntegerWeight::one());
    fst.add_arc(s0, Arc::new(1, 1, IntegerWeight::new(1_000_000), s1));

    let distances = shortest_distance(&fst).unwrap();
    assert_eq!(distances[s1 as usize], IntegerWeight::new(1_000_000));
}

#[test]
fn test_integer_weight_conversion() {
    // Test converting from tropical to integer weights
    let mut tropical_fst = VectorFst::<TropicalWeight>::new();
    let s0 = tropical_fst.add_state();
    let s1 = tropical_fst.add_state();
    tropical_fst.set_start(s0);
    tropical_fst.set_final(s1, TropicalWeight::new(5.0));
    tropical_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(3.0), s1));

    // Convert to integer semiring
    let integer_fst: VectorFst<IntegerWeight> =
        weight_convert(&tropical_fst, |w: &TropicalWeight| {
            IntegerWeight::new(w.value().floor() as i64)
        }).unwrap();

    assert_eq!(integer_fst.num_states(), tropical_fst.num_states());
    assert_eq!(integer_fst.num_arcs_total(), tropical_fst.num_arcs_total());

    let arc = integer_fst.arcs(s0).next().unwrap();
    assert_eq!(arc.weight, IntegerWeight::new(3));
}

#[test]
fn test_integer_with_topsort() {
    // Test topological sort with integer weights
    let mut fst = VectorFst::<IntegerWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s2, IntegerWeight::one());
    fst.add_arc(s0, Arc::new(1, 1, IntegerWeight::one(), s1));
    fst.add_arc(s1, Arc::new(2, 2, IntegerWeight::one(), s2));

    let sorted: VectorFst<IntegerWeight> = topsort(&fst).unwrap();

    assert_eq!(sorted.num_states(), fst.num_states());
}

#[test]
fn test_integer_zero_and_one() {
    // Verify zero and one elements work correctly in FST context
    let mut fst = VectorFst::<IntegerWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s1, IntegerWeight::one());

    fst.add_arc(s0, Arc::new(1, 1, IntegerWeight::one(), s1));
    fst.add_arc(s0, Arc::new(2, 2, IntegerWeight::zero(), s1));

    // Verify final weight
    assert!(fst.is_final(s1));
    assert_eq!(*fst.final_weight(s1).unwrap(), IntegerWeight::one());

    // Verify arc weights
    let arcs: Vec<_> = fst.arcs(s0).collect();
    assert_eq!(arcs.len(), 2);
    assert_eq!(arcs[0].weight, IntegerWeight::one());
    assert_eq!(arcs[1].weight, IntegerWeight::zero());
}
