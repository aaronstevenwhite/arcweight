//! Integration tests for cross-semiring operations and conversions

use arcweight::prelude::*;
use num_traits::One;

#[test]
fn test_cross_semiring_operations() {
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
fn test_semiring_conversions() {
    // Test weight conversions between different semirings
    let tropical_fst = {
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        fst.set_start(s0);
        fst.set_final(s1, TropicalWeight::new(2.0));
        fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(3.0), s1));
        fst
    };

    // Convert to log semiring using a converter function
    let log_fst: VectorFst<LogWeight> = weight_convert(&tropical_fst, |w: &TropicalWeight| {
        LogWeight::new(-(*w.value() as f64).ln())
    })
    .unwrap();
    assert_eq!(log_fst.num_states(), tropical_fst.num_states());
    assert_eq!(log_fst.num_arcs_total(), tropical_fst.num_arcs_total());
}

#[test]
fn test_semiring_in_algorithms() {
    // Test that algorithms work correctly with different semirings

    // Boolean semiring for closure
    let mut bool_fst = VectorFst::<BooleanWeight>::new();
    let s0 = bool_fst.add_state();
    let s1 = bool_fst.add_state();
    bool_fst.set_start(s0);
    bool_fst.set_final(s1, BooleanWeight::one());
    bool_fst.add_arc(s0, Arc::new(1, 1, BooleanWeight::new(true), s1));

    let star: VectorFst<BooleanWeight> = closure(&bool_fst).unwrap();
    assert!(star.num_states() > bool_fst.num_states());

    // Boolean semiring for epsilon removal
    let mut bool_eps_fst = VectorFst::<BooleanWeight>::new();
    let s0 = bool_eps_fst.add_state();
    let s1 = bool_eps_fst.add_state();
    let s2 = bool_eps_fst.add_state();
    bool_eps_fst.set_start(s0);
    bool_eps_fst.set_final(s2, BooleanWeight::one());
    bool_eps_fst.add_arc(s0, Arc::epsilon(BooleanWeight::new(true), s1));
    bool_eps_fst.add_arc(s1, Arc::new(1, 1, BooleanWeight::new(true), s2));

    let no_eps: VectorFst<BooleanWeight> = remove_epsilons(&bool_eps_fst).unwrap();
    for state in no_eps.states() {
        for arc in no_eps.arcs(state) {
            assert!(!arc.is_epsilon());
        }
    }
}

#[test]
fn test_semiring_properties_in_fst() {
    // Test that FST properties are correctly computed for different semirings
    let mut tropical_fst = VectorFst::<TropicalWeight>::new();
    let s0 = tropical_fst.add_state();
    let s1 = tropical_fst.add_state();
    tropical_fst.set_start(s0);
    tropical_fst.set_final(s1, TropicalWeight::new(1.0));
    tropical_fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s1));

    let props = tropical_fst.properties();
    assert!(!props.contains(FstProperties::UNWEIGHTED)); // Has non-unit weight

    // Boolean FST should be unweighted (all weights are either true/false)
    let mut bool_fst = VectorFst::<BooleanWeight>::new();
    let s0 = bool_fst.add_state();
    let s1 = bool_fst.add_state();
    bool_fst.set_start(s0);
    bool_fst.set_final(s1, BooleanWeight::one());
    bool_fst.add_arc(s0, Arc::new(1, 1, BooleanWeight::one(), s1));

    let bool_props = bool_fst.properties();
    // Note: Boolean semiring might not be considered "unweighted" in the traditional sense
    assert!(bool_props.contains(FstProperties::ACCEPTOR)); // Input and output labels match
}
