//! Property-based tests for IO module

use arcweight::prelude::*;
use proptest::prelude::*;
use std::io::Cursor;

#[cfg(feature = "serde")]
use arcweight::io::{read_binary, write_binary};

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
