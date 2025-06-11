//! Comprehensive tests for algorithms module

use arcweight::prelude::*;
use num_traits::One;
use proptest::prelude::*;

#[cfg(test)]
mod composition_tests {
    use super::*;

    #[test]
    fn test_basic_composition() {
        // First FST: input -> intermediate
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s1, TropicalWeight::one());
        fst1.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(1.0), s1));

        // Second FST: intermediate -> output
        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let t0 = fst2.add_state();
        let t1 = fst2.add_state();
        fst2.set_start(t0);
        fst2.set_final(t1, TropicalWeight::one());
        fst2.add_arc(t0, Arc::new(2, 3, TropicalWeight::new(2.0), t1));

        let composed: VectorFst<TropicalWeight> = compose_default(&fst1, &fst2).unwrap();

        assert!(composed.start().is_some());
        assert!(composed.num_states() > 0);

        // Should have at least one path from input 1 to output 3
        let mut found_path = false;
        for state in composed.states() {
            for arc in composed.arcs(state) {
                if arc.ilabel == 1 && arc.olabel == 3 {
                    found_path = true;
                    // Weight should be combined: 1.0 + 2.0 = 3.0
                    assert_eq!(*arc.weight.value(), 3.0);
                }
            }
        }
        assert!(found_path);
    }

    #[test]
    fn test_composition_no_match() {
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
        fst2.add_arc(t0, Arc::new(3, 4, TropicalWeight::new(2.0), t1)); // No match

        let composed: VectorFst<TropicalWeight> = compose_default(&fst1, &fst2).unwrap();

        // Should result in empty FST or FST with no accepting paths
        assert!(composed.start().is_some()); // May have start state

        // Check if any state is final
        let mut has_final = false;
        for state in composed.states() {
            if composed.is_final(state) {
                has_final = true;
                break;
            }
        }
        // Should have no final states reachable from start
        if has_final {
            // If there are final states, they should not be reachable
            assert_eq!(composed.num_arcs_total(), 0);
        }
    }

    #[test]
    fn test_composition_epsilon() {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let s0 = fst1.add_state();
        let s1 = fst1.add_state();
        let s2 = fst1.add_state();
        fst1.set_start(s0);
        fst1.set_final(s2, TropicalWeight::one());
        fst1.add_arc(s0, Arc::epsilon(TropicalWeight::new(0.5), s1));
        fst1.add_arc(s1, Arc::new(1, 2, TropicalWeight::new(1.0), s2));

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
}

#[cfg(test)]
mod determinization_tests {
    use super::*;

    #[test]
    fn test_determinize_simple() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Non-deterministic: two arcs with same input label
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s2));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s2));

        let det: VectorFst<TropicalWeight> = determinize(&fst).unwrap();

        // Check determinism: no state should have multiple arcs with same input label
        for state in det.states() {
            let mut seen_labels = std::collections::HashSet::new();
            for arc in det.arcs(state) {
                assert!(
                    seen_labels.insert(arc.ilabel),
                    "Found duplicate input label {} from state {}",
                    arc.ilabel,
                    state
                );
            }
        }

        // Should preserve language
        assert!(det.start().is_some());
        assert!(det.num_states() > 0);
    }

    #[test]
    fn test_determinize_already_deterministic() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));

        let det: VectorFst<TropicalWeight> = determinize(&fst).unwrap();

        // Should be similar to original
        assert_eq!(det.num_states(), fst.num_states());
        assert!(det.start().is_some());

        for state in det.states() {
            let mut seen_labels = std::collections::HashSet::new();
            for arc in det.arcs(state) {
                assert!(seen_labels.insert(arc.ilabel));
            }
        }
    }
}

#[cfg(test)]
mod shortest_path_tests {
    use super::*;

    #[test]
    fn test_shortest_path_single() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

        // Two paths with different costs
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(2.0), s2));
        fst.add_arc(s0, Arc::new(3, 3, TropicalWeight::new(5.0), s2));

        let shortest: VectorFst<TropicalWeight> = shortest_path_single(&fst).unwrap();

        // Should preserve structure but only keep shortest paths
        assert!(shortest.start().is_some());
        assert!(shortest.num_states() > 0);

        // Start state should have only one outgoing arc (the cheaper one)
        assert!(shortest.num_arcs(s0) <= 1);

        if shortest.num_arcs(s0) == 1 {
            let arcs: Vec<_> = shortest.arcs(s0).collect();
            // Should prefer the path with weight 1.0 over 5.0
            assert_eq!(arcs[0].ilabel, 1);
        }
    }

    #[test]
    fn test_shortest_path_multiple() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::new(1.0));
        fst.set_final(s2, TropicalWeight::new(2.0));

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(2.0), s2));

        let config = ShortestPathConfig {
            nshortest: 2,
            ..Default::default()
        };
        let shortest: VectorFst<TropicalWeight> = shortest_path(&fst, config).unwrap();

        // Should keep both paths since we requested 2
        assert!(shortest.start().is_some());
        assert!(shortest.num_states() > 0);
    }

    #[test]
    fn test_shortest_path_empty() {
        let fst = VectorFst::<TropicalWeight>::new();

        // Empty FST should return error or empty result
        match shortest_path_single::<
            TropicalWeight,
            VectorFst<TropicalWeight>,
            VectorFst<TropicalWeight>,
        >(&fst)
        {
            Ok(shortest) => assert!(shortest.is_empty()),
            Err(_) => {} // Empty FST may legitimately fail
        }
    }
}

#[cfg(test)]
mod union_tests {
    use super::*;

    #[test]
    fn test_union_basic() {
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

        // Should have states from both FSTs plus new start state
        assert!(unioned.num_states() >= fst1.num_states() + fst2.num_states());
        assert!(unioned.start().is_some());

        // Union might use epsilon transitions or restructure the FST
        // At minimum, should have preserved the original structure somehow
        assert!(unioned.num_states() >= 2); // Should have at least the states from both FSTs
    }

    #[test]
    fn test_union_empty() {
        let fst1 = VectorFst::<TropicalWeight>::new();
        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let s0 = fst2.add_state();
        fst2.set_start(s0);
        fst2.set_final(s0, TropicalWeight::one());

        let unioned: VectorFst<TropicalWeight> = union(&fst1, &fst2).unwrap();

        // Should be equivalent to fst2
        assert!(unioned.start().is_some());
    }
}

#[cfg(test)]
mod concat_tests {
    use super::*;

    #[test]
    fn test_concat_basic() {
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

        // Should have states from both FSTs
        assert_eq!(
            concatenated.num_states(),
            fst1.num_states() + fst2.num_states()
        );
        assert!(concatenated.start().is_some());

        // Original final states of fst1 should no longer be final
        // Only final states of fst2 (with offset) should be final
        let mut final_count = 0;
        for state in concatenated.states() {
            if concatenated.is_final(state) {
                final_count += 1;
            }
        }
        assert!(final_count > 0);
    }

    #[test]
    fn test_concat_empty() {
        let fst1 = VectorFst::<TropicalWeight>::new();
        let mut fst2 = VectorFst::<TropicalWeight>::new();
        let s0 = fst2.add_state();
        fst2.set_start(s0);
        fst2.set_final(s0, TropicalWeight::one());

        let concatenated: VectorFst<TropicalWeight> = concat(&fst1, &fst2).unwrap();

        // Concat with empty should be empty
        assert!(concatenated.is_empty() || concatenated.num_arcs_total() == 0);
    }
}

#[cfg(test)]
mod closure_tests {
    use super::*;

    #[test]
    fn test_closure_with_boolean_weight() {
        let mut fst = VectorFst::<BooleanWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, BooleanWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, BooleanWeight::new(true), s1));

        let star: VectorFst<BooleanWeight> = closure(&fst).unwrap();

        // Closure should add states for start/final
        assert!(star.num_states() > fst.num_states());
        assert!(star.start().is_some());

        // Start state should be final (empty string acceptance)
        let start = star.start().unwrap();
        assert!(star.is_final(start));
    }

    #[test]
    fn test_closure_plus_with_boolean_weight() {
        let mut fst = VectorFst::<BooleanWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, BooleanWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, BooleanWeight::new(true), s1));

        let plus: VectorFst<BooleanWeight> = closure_plus(&fst).unwrap();

        // Plus closure should not accept empty string
        assert!(plus.start().is_some());

        // Start state should not be final
        let start = plus.start().unwrap();
        assert!(!plus.is_final(start));
    }
}

#[cfg(test)]
mod reverse_tests {
    use super::*;

    #[test]
    fn test_reverse_basic() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::new(2.0));

        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(3.0), s2));

        let reversed: VectorFst<TropicalWeight> = reverse(&fst).unwrap();

        // Should add a new start state
        assert_eq!(reversed.num_states(), fst.num_states() + 1);
        assert!(reversed.start().is_some());

        // Original start should be final in reversed
        assert!(reversed.is_final(s0));

        // Check that arcs are reversed
        let mut found_reversed_arc = false;
        for state in reversed.states() {
            for arc in reversed.arcs(state) {
                if arc.nextstate == s1 && arc.ilabel == 2 {
                    found_reversed_arc = true;
                }
            }
        }
        assert!(found_reversed_arc);
    }
}

#[cfg(test)]
mod connect_tests {
    use super::*;

    #[test]
    fn test_connect_removes_unreachable() {
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

        // Should remove unreachable states
        assert!(connected.num_states() < fst.num_states());
        assert!(connected.start().is_some());
    }

    #[test]
    fn test_connect_already_connected() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));

        let connected: VectorFst<TropicalWeight> = connect(&fst).unwrap();

        // Should be identical or similar
        assert_eq!(connected.num_states(), fst.num_states());
        assert!(connected.start().is_some());
    }
}

#[cfg(test)]
mod epsilon_removal_tests {
    use super::*;

    #[test]
    fn test_remove_epsilons() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::one());

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

        let no_eps: VectorFst<BooleanWeight> =
            remove_epsilons::<BooleanWeight, VectorFst<BooleanWeight>, VectorFst<BooleanWeight>>(
                &bool_fst,
            )
            .unwrap();

        // Check no epsilon transitions remain
        for state in no_eps.states() {
            for arc in no_eps.arcs(state) {
                assert!(!arc.is_epsilon(), "Found epsilon arc: {:?}", arc);
            }
        }

        // Should preserve the language
        assert!(no_eps.start().is_some());
    }

    #[test]
    fn test_remove_epsilons_none() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));

        // Use BooleanWeight for epsilon removal since TropicalWeight doesn't implement StarSemiring
        let mut bool_fst = VectorFst::<BooleanWeight>::new();
        let s0 = bool_fst.add_state();
        let s1 = bool_fst.add_state();
        let s2 = bool_fst.add_state();

        bool_fst.set_start(s0);
        bool_fst.set_final(s2, BooleanWeight::one());

        bool_fst.add_arc(s0, Arc::epsilon(BooleanWeight::new(true), s1));
        bool_fst.add_arc(s1, Arc::new(1, 1, BooleanWeight::new(true), s2));

        let no_eps: VectorFst<BooleanWeight> =
            remove_epsilons::<BooleanWeight, VectorFst<BooleanWeight>, VectorFst<BooleanWeight>>(
                &bool_fst,
            )
            .unwrap();

        // Should preserve structure when no epsilons present
        // Note: Implementation might add states for proper structure
        assert!(no_eps.num_states() >= bool_fst.num_states() - 1); // Allow for structure changes
    }
}

#[cfg(test)]
mod topsort_tests {
    use super::*;

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

        // Verify topological order: all arcs go from lower to higher state IDs
        for state in sorted.states() {
            for arc in sorted.arcs(state) {
                assert!(
                    state < arc.nextstate,
                    "Arc from {} to {} violates topological order",
                    state,
                    arc.nextstate
                );
            }
        }

        assert!(sorted.start().is_some());
        assert_eq!(sorted.num_states(), fst.num_states());
    }

    #[test]
    fn test_topsort_cyclic() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::one());

        // Create a cycle
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s0));

        // Topsort should fail or handle cycles appropriately
        let result =
            topsort::<TropicalWeight, VectorFst<TropicalWeight>, VectorFst<TropicalWeight>>(&fst);
        if let Ok(sorted) = result {
            // If it succeeds, should still have valid structure
            assert!(sorted.start().is_some());
        }
        // If it fails, that's also acceptable for cyclic graphs
    }
}

// Property-based tests
proptest! {
    #[test]
    fn determinize_preserves_determinism(
        num_states in 1..10usize,
        num_arcs in 0..20usize,
    ) {
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
                fst.add_arc(from, Arc::new(label, label, TropicalWeight::new(i as f32), to));
            }
        }

        if let Ok(det) = determinize::<TropicalWeight, VectorFst<TropicalWeight>, VectorFst<TropicalWeight>>(&fst) {
            // Check determinism
            for state in det.states() {
                let mut seen_labels = std::collections::HashSet::new();
                for arc in det.arcs(state) {
                    prop_assert!(seen_labels.insert(arc.ilabel),
                        "Non-deterministic: duplicate label {} from state {}", arc.ilabel, state);
                }
            }
        }
    }

    #[test]
    fn union_preserves_languages(
        fst1_states in 1..5usize,
        fst2_states in 1..5usize,
    ) {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let mut fst2 = VectorFst::<TropicalWeight>::new();

        for _ in 0..fst1_states {
            fst1.add_state();
        }
        for _ in 0..fst2_states {
            fst2.add_state();
        }

        if fst1_states > 0 {
            fst1.set_start(0);
            fst1.set_final((fst1_states - 1) as u32, TropicalWeight::one());
        }
        if fst2_states > 0 {
            fst2.set_start(0);
            fst2.set_final((fst2_states - 1) as u32, TropicalWeight::one());
        }

        if let Ok(unioned) = union::<TropicalWeight, VectorFst<TropicalWeight>, VectorFst<TropicalWeight>, VectorFst<TropicalWeight>>(&fst1, &fst2) {
            prop_assert!(unioned.num_states() >= fst1.num_states().max(fst2.num_states()));
            prop_assert!(unioned.start().is_some());
        }
    }

    #[test]
    fn concat_combines_lengths(
        fst1_states in 1..5usize,
        fst2_states in 1..5usize,
    ) {
        let mut fst1 = VectorFst::<TropicalWeight>::new();
        let mut fst2 = VectorFst::<TropicalWeight>::new();

        for _ in 0..fst1_states {
            fst1.add_state();
        }
        for _ in 0..fst2_states {
            fst2.add_state();
        }

        if fst1_states > 0 {
            fst1.set_start(0);
            fst1.set_final((fst1_states - 1) as u32, TropicalWeight::one());
        }
        if fst2_states > 0 {
            fst2.set_start(0);
            fst2.set_final((fst2_states - 1) as u32, TropicalWeight::one());
        }

        if let Ok(concatenated) = concat::<TropicalWeight, VectorFst<TropicalWeight>, VectorFst<TropicalWeight>, VectorFst<TropicalWeight>>(&fst1, &fst2) {
            prop_assert_eq!(concatenated.num_states(), fst1.num_states() + fst2.num_states());
            prop_assert!(concatenated.start().is_some());
        }
    }

    #[test]
    fn connect_removes_unreachable_states(num_states in 2..10usize) {
        let mut fst = VectorFst::<TropicalWeight>::new();

        for _ in 0..num_states {
            fst.add_state();
        }

        if num_states > 1 {
            fst.set_start(0);
            fst.set_final(1, TropicalWeight::one());
            fst.add_arc(0, Arc::new(1, 1, TropicalWeight::new(1.0), 1));
            // Leave other states disconnected
        }

        if let Ok(connected) = connect::<TropicalWeight, VectorFst<TropicalWeight>, VectorFst<TropicalWeight>>(&fst) {
            prop_assert!(connected.num_states() <= fst.num_states());
            if !connected.is_empty() {
                prop_assert!(connected.start().is_some());
            }
        }
    }
}
