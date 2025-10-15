use arcweight::prelude::*;

#[test]
fn test_encode_decode_preserves_structure() {
    // Create an FST with complex structure
    let mut fst = VectorFst::<TropicalWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    let s3 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s3, TropicalWeight::new(1.0));

    fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));
    fst.add_arc(s0, Arc::new(3, 4, TropicalWeight::new(0.3), s2));
    fst.add_arc(s1, Arc::new(5, 6, TropicalWeight::new(0.2), s3));
    fst.add_arc(s2, Arc::new(7, 8, TropicalWeight::new(0.4), s3));

    let (encoded, table) = encode(&fst).unwrap();
    let decoded = decode(&encoded, &table).unwrap();

    // Structure should be preserved
    assert_eq!(decoded.num_states(), fst.num_states());
    assert_eq!(decoded.num_arcs_total(), fst.num_arcs_total());
    assert_eq!(decoded.start(), fst.start());

    // All arcs should match
    for state in fst.states() {
        let orig_arcs: Vec<_> = fst.arcs(state).collect();
        let dec_arcs: Vec<_> = decoded.arcs(state).collect();

        assert_eq!(orig_arcs.len(), dec_arcs.len());

        for (orig, dec) in orig_arcs.iter().zip(dec_arcs.iter()) {
            assert_eq!(orig.ilabel, dec.ilabel);
            assert_eq!(orig.olabel, dec.olabel);
            assert_eq!(orig.weight, dec.weight);
            assert_eq!(orig.nextstate, dec.nextstate);
        }
    }
}

#[test]
fn test_encode_decode_with_determinization() {
    // Create a simple non-deterministic FST
    let mut fst = VectorFst::<TropicalWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());
    fst.set_final(s2, TropicalWeight::one());

    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.3), s2));

    // Encode/decode should preserve the structure
    let (encoded, table) = encode(&fst).unwrap();

    // Encoded FST should have same structure
    assert_eq!(encoded.num_states(), fst.num_states());
    assert_eq!(encoded.num_arcs_total(), fst.num_arcs_total());

    let decoded = decode(&encoded, &table).unwrap();
    assert_eq!(decoded.num_states(), fst.num_states());
    assert_eq!(decoded.num_arcs_total(), fst.num_arcs_total());
}

#[test]
fn test_encode_decode_with_minimization() {
    // Create FST that could be minimized
    let mut fst = VectorFst::<TropicalWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::new(1.0));
    fst.set_final(s2, TropicalWeight::new(1.0));

    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
    fst.add_arc(s0, Arc::new(2, 2, TropicalWeight::new(0.5), s2));

    let (encoded, table) = encode(&fst).unwrap();
    let decoded = decode(&encoded, &table).unwrap();

    // Should preserve structure
    assert_eq!(decoded.num_states(), fst.num_states());
}

#[test]
fn test_encode_decode_complex_transducer() {
    // Create a complex transducer
    let mut fst = VectorFst::<TropicalWeight>::new();
    let states: Vec<_> = (0..10).map(|_| fst.add_state()).collect();

    fst.set_start(states[0]);
    fst.set_final(states[9], TropicalWeight::new(2.0));

    // Add various arcs with different labels and weights
    for i in 0..9 {
        fst.add_arc(
            states[i],
            Arc::new(
                (i + 1) as u32,
                (i + 2) as u32,
                TropicalWeight::new((i as f32 + 1.0) * 0.1),
                states[i + 1],
            ),
        );
    }

    // Add some epsilon transitions
    fst.add_arc(
        states[2],
        Arc::epsilon(TropicalWeight::new(0.05), states[5]),
    );
    fst.add_arc(
        states[4],
        Arc::epsilon(TropicalWeight::new(0.15), states[7]),
    );

    let (encoded, table) = encode(&fst).unwrap();
    let decoded = decode(&encoded, &table).unwrap();

    assert_eq!(decoded.num_states(), fst.num_states());
    assert_eq!(decoded.num_arcs_total(), fst.num_arcs_total());

    // Verify all arcs match
    for state in fst.states() {
        let orig_arcs: Vec<_> = fst.arcs(state).collect();
        let dec_arcs: Vec<_> = decoded.arcs(state).collect();

        assert_eq!(orig_arcs.len(), dec_arcs.len());

        for (orig, dec) in orig_arcs.iter().zip(dec_arcs.iter()) {
            assert_eq!(orig.ilabel, dec.ilabel);
            assert_eq!(orig.olabel, dec.olabel);
            assert_eq!(orig.weight, dec.weight);
            assert_eq!(orig.nextstate, dec.nextstate);
        }
    }
}

#[test]
fn test_encode_with_multiple_identical_arcs() {
    // Test encoding efficiency with duplicate arcs
    let mut fst = VectorFst::<TropicalWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());

    // Add many identical arcs
    for _ in 0..10 {
        fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));
    }

    let (encoded, table) = encode(&fst).unwrap();

    // Should only have one unique encoding
    assert_eq!(table.size(), 1);

    // All encoded arcs should have same label
    let arcs: Vec<_> = encoded.arcs(s0).collect();
    assert_eq!(arcs.len(), 10);

    let first_label = arcs[0].ilabel;
    for arc in &arcs {
        assert_eq!(arc.ilabel, first_label);
    }
}

#[test]
fn test_encode_decode_with_cycles() {
    // Create FST with cycles
    let mut fst = VectorFst::<TropicalWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::one());

    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
    fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(0.3), s2));
    fst.add_arc(s1, Arc::new(3, 3, TropicalWeight::new(0.2), s0)); // Cycle back
    fst.add_arc(s2, Arc::new(4, 4, TropicalWeight::new(0.1), s0)); // Another cycle

    let (encoded, table) = encode(&fst).unwrap();
    let decoded = decode(&encoded, &table).unwrap();

    // Verify structure preserved including cycles
    assert_eq!(decoded.num_states(), fst.num_states());
    assert_eq!(decoded.num_arcs_total(), fst.num_arcs_total());

    for state in fst.states() {
        let orig_arcs: Vec<_> = fst.arcs(state).collect();
        let dec_arcs: Vec<_> = decoded.arcs(state).collect();

        assert_eq!(orig_arcs.len(), dec_arcs.len());
    }
}

#[test]
fn test_encode_decode_with_boolean_weight() {
    // Test with a different semiring
    let mut fst = VectorFst::<BooleanWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s1, BooleanWeight::one());

    fst.add_arc(s0, Arc::new(1, 2, BooleanWeight::one(), s1));
    fst.add_arc(s0, Arc::new(3, 4, BooleanWeight::one(), s1));

    let (encoded, table) = encode(&fst).unwrap();
    let decoded = decode(&encoded, &table).unwrap();

    assert_eq!(decoded.num_states(), fst.num_states());
    assert_eq!(decoded.num_arcs_total(), fst.num_arcs_total());

    // Verify arcs match
    for state in fst.states() {
        let orig_arcs: Vec<_> = fst.arcs(state).collect();
        let dec_arcs: Vec<_> = decoded.arcs(state).collect();

        assert_eq!(orig_arcs.len(), dec_arcs.len());

        for (orig, dec) in orig_arcs.iter().zip(dec_arcs.iter()) {
            assert_eq!(orig.ilabel, dec.ilabel);
            assert_eq!(orig.olabel, dec.olabel);
            assert_eq!(orig.weight, dec.weight);
        }
    }
}

#[test]
fn test_encode_table_reuse() {
    // Test that encoding table can be reused
    let mut fst1 = VectorFst::<TropicalWeight>::new();
    let s0 = fst1.add_state();
    let s1 = fst1.add_state();
    fst1.set_start(s0);
    fst1.set_final(s1, TropicalWeight::one());
    fst1.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));

    let mut fst2 = VectorFst::<TropicalWeight>::new();
    let t0 = fst2.add_state();
    let t1 = fst2.add_state();
    fst2.set_start(t0);
    fst2.set_final(t1, TropicalWeight::one());
    fst2.add_arc(t0, Arc::new(1, 2, TropicalWeight::new(0.5), t1));

    let (encoded1, table1) = encode(&fst1).unwrap();
    let (encoded2, table2) = encode(&fst2).unwrap();

    // Same arc tuple should get same encoding (in their respective tables)
    assert_eq!(table1.size(), table2.size());

    // Both should decode correctly
    let decoded1 = decode(&encoded1, &table1).unwrap();
    let decoded2 = decode(&encoded2, &table2).unwrap();

    assert_eq!(decoded1.num_arcs_total(), fst1.num_arcs_total());
    assert_eq!(decoded2.num_arcs_total(), fst2.num_arcs_total());
}
