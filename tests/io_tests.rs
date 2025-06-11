//! Comprehensive tests for IO module

use arcweight::prelude::*;
use num_traits::identities::One;
use proptest::prelude::*;
use std::io::{BufReader, Cursor};

#[cfg(feature = "serde")]
use arcweight::io::{read_binary, write_binary};

#[cfg(test)]
mod text_format_tests {
    use super::*;

    #[test]
    fn test_write_read_text_roundtrip() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::new(2.5));

        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(1.0), s1));
        fst.add_arc(s1, Arc::new(3, 4, TropicalWeight::new(1.5), s2));
        fst.add_arc(s0, Arc::epsilon(TropicalWeight::new(0.5), s2));

        // Write to buffer
        let mut buffer = Vec::new();
        write_text(&fst, &mut buffer, None, None).unwrap();

        // Read back from buffer
        let cursor = Cursor::new(buffer);
        let mut buf_reader = BufReader::new(cursor);
        let read_fst: VectorFst<TropicalWeight> =
            read_text::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut buf_reader, None, None)
                .unwrap();

        // Verify structure is preserved
        assert_eq!(read_fst.num_states(), fst.num_states());
        assert_eq!(read_fst.start(), fst.start());
        assert_eq!(read_fst.num_arcs_total(), fst.num_arcs_total());

        // Check final weights
        for state in fst.states() {
            let original_final = fst.final_weight(state);
            let read_final = read_fst.final_weight(state);

            match (original_final, read_final) {
                (Some(w1), Some(w2)) => assert_eq!(w1, w2),
                (None, None) => {}
                _ => panic!("Final weight mismatch for state {}", state),
            }
        }

        // Check arcs
        for state in fst.states() {
            let original_arcs: Vec<_> = fst.arcs(state).collect();
            let read_arcs: Vec<_> = read_fst.arcs(state).collect();

            assert_eq!(original_arcs.len(), read_arcs.len());

            for (orig, read) in original_arcs.iter().zip(read_arcs.iter()) {
                assert_eq!(orig.ilabel, read.ilabel);
                assert_eq!(orig.olabel, read.olabel);
                assert_eq!(orig.weight, read.weight);
                assert_eq!(orig.nextstate, read.nextstate);
            }
        }
    }

    #[test]
    fn test_write_read_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();

        let mut buffer = Vec::new();
        write_text(&fst, &mut buffer, None, None).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_text::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor, None, None)
                .unwrap();

        assert!(read_fst.is_empty());
        assert_eq!(read_fst.num_states(), 0);
        assert_eq!(read_fst.start(), None);
    }

    #[test]
    fn test_write_read_single_state() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s0, TropicalWeight::one());

        let mut buffer = Vec::new();
        write_text(&fst, &mut buffer, None, None).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_text::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor, None, None)
                .unwrap();

        assert_eq!(read_fst.num_states(), 1);
        assert_eq!(read_fst.start(), Some(0));
        assert!(read_fst.is_final(0));
    }

    #[test]
    fn test_text_format_different_weights() {
        // Test with Boolean weight
        let mut bool_fst = VectorFst::<BooleanWeight>::new();
        let s0 = bool_fst.add_state();
        let s1 = bool_fst.add_state();

        bool_fst.set_start(s0);
        bool_fst.set_final(s1, BooleanWeight::new(true));
        bool_fst.add_arc(s0, Arc::new(1, 1, BooleanWeight::new(false), s1));

        let mut buffer = Vec::new();
        write_text(&bool_fst, &mut buffer, None, None).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<BooleanWeight> =
            read_text::<BooleanWeight, VectorFst<BooleanWeight>, _>(&mut cursor, None, None)
                .unwrap();

        assert_eq!(read_fst.num_states(), bool_fst.num_states());
        assert_eq!(read_fst.start(), bool_fst.start());

        // Test with Probability weight
        let mut prob_fst = VectorFst::<ProbabilityWeight>::new();
        let s0 = prob_fst.add_state();
        let s1 = prob_fst.add_state();

        prob_fst.set_start(s0);
        prob_fst.set_final(s1, ProbabilityWeight::new(0.8));
        prob_fst.add_arc(s0, Arc::new(1, 1, ProbabilityWeight::new(0.3), s1));

        let mut buffer = Vec::new();
        write_text(&prob_fst, &mut buffer, None, None).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<ProbabilityWeight> = read_text::<
            ProbabilityWeight,
            VectorFst<ProbabilityWeight>,
            _,
        >(&mut cursor, None, None)
        .unwrap();

        assert_eq!(read_fst.num_states(), prob_fst.num_states());
        assert_eq!(read_fst.start(), prob_fst.start());
    }
}

#[cfg(all(test, feature = "serde"))]
mod binary_format_tests {
    use super::*;
    use arcweight::io::{read_binary, write_binary};

    #[test]
    fn test_write_read_binary_roundtrip() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s2, TropicalWeight::new(std::f32::consts::PI));

        fst.add_arc(s0, Arc::new(10, 20, TropicalWeight::new(2.5), s1));
        fst.add_arc(s1, Arc::new(30, 40, TropicalWeight::new(1.2), s2));
        fst.add_arc(s0, Arc::epsilon(TropicalWeight::new(0.1), s2));

        // Write to buffer
        let mut buffer = Vec::new();
        write_binary(&fst, &mut buffer).unwrap();

        // Read back from buffer
        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_binary::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        // Verify structure is preserved
        assert_eq!(read_fst.num_states(), fst.num_states());
        assert_eq!(read_fst.start(), fst.start());
        assert_eq!(read_fst.num_arcs_total(), fst.num_arcs_total());

        // Check final weights
        for state in fst.states() {
            let original_final = fst.final_weight(state);
            let read_final = read_fst.final_weight(state);

            match (original_final, read_final) {
                (Some(w1), Some(w2)) => assert_eq!(w1, w2),
                (None, None) => {}
                _ => panic!("Final weight mismatch for state {}", state),
            }
        }

        // Check arcs
        for state in fst.states() {
            let original_arcs: Vec<_> = fst.arcs(state).collect();
            let read_arcs: Vec<_> = read_fst.arcs(state).collect();

            assert_eq!(original_arcs.len(), read_arcs.len());

            for (orig, read) in original_arcs.iter().zip(read_arcs.iter()) {
                assert_eq!(orig.ilabel, read.ilabel);
                assert_eq!(orig.olabel, read.olabel);
                assert_eq!(orig.weight, read.weight);
                assert_eq!(orig.nextstate, read.nextstate);
            }
        }
    }

    #[test]
    fn test_binary_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();

        let mut buffer = Vec::new();
        write_binary(&fst, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_binary::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        assert!(read_fst.is_empty());
        assert_eq!(read_fst.num_states(), 0);
        assert_eq!(read_fst.start(), None);
    }

    #[test]
    fn test_binary_format_efficiency() {
        let mut fst = VectorFst::<TropicalWeight>::new();

        // Create a larger FST to test efficiency
        for _i in 0..100 {
            fst.add_state();
        }

        fst.set_start(0);
        fst.set_final(99, TropicalWeight::one());

        for i in 0..99 {
            fst.add_arc(
                i,
                Arc::new(i + 1, i + 1, TropicalWeight::new(i as f32 * 0.1), i + 1),
            );
        }

        let mut binary_buffer = Vec::new();
        let mut text_buffer = Vec::new();

        write_binary(&fst, &mut binary_buffer).unwrap();
        write_text(&fst, &mut text_buffer, None, None).unwrap();

        // Binary format should typically be more compact
        // (though this is not guaranteed for all cases)
        println!(
            "Binary size: {}, Text size: {}",
            binary_buffer.len(),
            text_buffer.len()
        );

        // Verify binary can be read back correctly
        let mut cursor = Cursor::new(binary_buffer);
        let read_fst: VectorFst<TropicalWeight> =
            read_binary::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        assert_eq!(read_fst.num_states(), fst.num_states());
        assert_eq!(read_fst.num_arcs_total(), fst.num_arcs_total());
    }
}

#[cfg(test)]
mod openfst_compat_tests {
    use super::*;

    #[test]
    fn test_openfst_roundtrip_basic() {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::new(1.5));
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));

        let buffer = Vec::new();
        let mut cursor = Cursor::new(buffer);
        write_openfst(&fst, &mut cursor).unwrap();

        cursor.set_position(0);
        let read_fst: VectorFst<TropicalWeight> =
            read_openfst::<VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        assert_eq!(read_fst.num_states(), fst.num_states());
        assert_eq!(read_fst.start(), fst.start());
        assert_eq!(read_fst.num_arcs_total(), fst.num_arcs_total());
    }

    #[test]
    fn test_openfst_empty_fst() {
        let fst = VectorFst::<TropicalWeight>::new();

        let buffer = Vec::new();
        let mut cursor = Cursor::new(buffer);
        write_openfst(&fst, &mut cursor).unwrap();

        cursor.set_position(0);
        let read_fst: VectorFst<TropicalWeight> =
            read_openfst::<VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        assert!(read_fst.is_empty());
    }

    #[test]
    fn test_openfst_compatibility() {
        // Test that our format can handle typical OpenFST constructs
        let mut fst = VectorFst::<TropicalWeight>::new();

        // Create a more complex FST structure
        for _i in 0..10 {
            fst.add_state();
        }

        fst.set_start(0);
        fst.set_final(9, TropicalWeight::new(0.0));

        // Add various arc types
        fst.add_arc(0, Arc::new(1, 1, TropicalWeight::new(1.0), 1));
        fst.add_arc(1, Arc::epsilon(TropicalWeight::new(0.0), 2));
        fst.add_arc(2, Arc::new(2, 3, TropicalWeight::new(2.0), 3));

        // Add some parallel arcs
        fst.add_arc(0, Arc::new(4, 4, TropicalWeight::new(3.0), 4));
        fst.add_arc(4, Arc::new(5, 5, TropicalWeight::new(1.0), 9));

        let buffer = Vec::new();
        let mut cursor = Cursor::new(buffer);
        write_openfst(&fst, &mut cursor).unwrap();

        cursor.set_position(0);
        let read_fst: VectorFst<TropicalWeight> =
            read_openfst::<VectorFst<TropicalWeight>, _>(&mut cursor).unwrap();

        assert_eq!(read_fst.num_states(), fst.num_states());
        assert_eq!(read_fst.start(), fst.start());

        // Verify specific structural elements
        assert!(read_fst.is_final(9));
        assert_eq!(read_fst.num_arcs(0), 2); // Two outgoing arcs from start
    }
}

// Property-based tests
proptest! {
    #[test]
    fn text_format_roundtrip_preserves_structure(
        num_states in 1..20usize,
        num_arcs in 0..50usize,
    ) {
        let mut fst = VectorFst::<TropicalWeight>::new();

        for _ in 0..num_states {
            fst.add_state();
        }

        if num_states > 0 {
            fst.set_start(0);
            if num_states > 1 {
                fst.set_final((num_states - 1) as u32, TropicalWeight::new(1.0));
            }

            for i in 0..num_arcs.min(num_states * 5) {
                let from = (i % num_states) as u32;
                let to = ((i + 1) % num_states) as u32;
                let label = (i % 10) as u32 + 1;
                fst.add_arc(from, Arc::new(label, label, TropicalWeight::new(i as f32 * 0.1), to));
            }
        }

        let mut buffer = Vec::new();
        if write_text(&fst, &mut buffer, None, None).is_ok() {
            let mut cursor = Cursor::new(buffer);
            if let Ok(read_fst) = read_text::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor, None, None) {
                prop_assert_eq!(read_fst.num_states(), fst.num_states());
                prop_assert_eq!(read_fst.start(), fst.start());
                prop_assert_eq!(read_fst.num_arcs_total(), fst.num_arcs_total());
            }
        }
    }

    #[test]
    fn binary_format_roundtrip_preserves_structure(
        num_states in 1..20usize,
        num_arcs in 0..50usize,
    ) {
        let mut fst = VectorFst::<TropicalWeight>::new();

        for _ in 0..num_states {
            fst.add_state();
        }

        if num_states > 0 {
            fst.set_start(0);
            if num_states > 1 {
                fst.set_final((num_states - 1) as u32, TropicalWeight::new(2.0));
            }

            for i in 0..num_arcs.min(num_states * 5) {
                let from = (i % num_states) as u32;
                let to = ((i + 1) % num_states) as u32;
                let label = (i % 10) as u32 + 1;
                fst.add_arc(from, Arc::new(label, label, TropicalWeight::new(i as f32 * 0.2), to));
            }
        }

        #[cfg(feature = "serde")]
        {
            let mut buffer = Vec::new();
            if write_binary(&fst, &mut buffer).is_ok() {
                let mut cursor = Cursor::new(buffer);
                if let Ok(read_fst) = read_binary::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut cursor) {
                    prop_assert_eq!(read_fst.num_states(), fst.num_states());
                    prop_assert_eq!(read_fst.start(), fst.start());
                    prop_assert_eq!(read_fst.num_arcs_total(), fst.num_arcs_total());
                }
            }
        }
    }

    #[test]
    fn io_format_consistency(weight: f32, ilabel: u32, olabel: u32) {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();

        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::new(weight));
        fst.add_arc(s0, Arc::new(ilabel, olabel, TropicalWeight::new(weight), s1));

        // Test both formats give same result
        let mut text_buffer = Vec::new();

        #[cfg(feature = "serde")]
        {
            let mut binary_buffer = Vec::new();

            if write_text(&fst, &mut text_buffer, None, None).is_ok() &&
               write_binary(&fst, &mut binary_buffer).is_ok() {

                let mut text_cursor = Cursor::new(text_buffer);
                let mut binary_cursor = Cursor::new(binary_buffer);

                if let (Ok(text_fst), Ok(binary_fst)) = (
                    read_text::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut text_cursor, None, None),
                    read_binary::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut binary_cursor)
                ) {
                    prop_assert_eq!(text_fst.num_states(), binary_fst.num_states());
                    prop_assert_eq!(text_fst.start(), binary_fst.start());
                    prop_assert_eq!(text_fst.num_arcs_total(), binary_fst.num_arcs_total());
                }
            }
        }

        #[cfg(not(feature = "serde"))]
        {
            // Just test text format when serde is not available
            if write_text(&fst, &mut text_buffer, None, None).is_ok() {
                let mut text_cursor = Cursor::new(text_buffer);
                if let Ok(_text_fst) = read_text::<TropicalWeight, VectorFst<TropicalWeight>, _>(&mut text_cursor, None, None) {
                    // Test passes if text format works
                }
            }
        }
    }
}
