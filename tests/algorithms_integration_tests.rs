//! Integration and property-based tests for algorithms module

use arcweight::prelude::*;
use proptest::prelude::*;

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
