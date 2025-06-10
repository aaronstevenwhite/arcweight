//! Property-based tests using proptest

use proptest::prelude::*;
use arcweight::prelude::*;
use num_traits::{Zero, One};

proptest! {
    #[test]
    fn tropical_semiring_associativity(a in -100.0..100.0f32, b in -100.0..100.0f32, c in -100.0..100.0f32) {
        let w1 = TropicalWeight::new(a);
        let w2 = TropicalWeight::new(b);
        let w3 = TropicalWeight::new(c);
        
        // (a + b) + c = a + (b + c)
        let left = w1.plus(&w2).plus(&w3);
        let right = w1.plus(&w2.plus(&w3));
        prop_assert!(left.approx_eq(&right, 1e-4));
        
        // (a * b) * c = a * (b * c)
        let left = w1.times(&w2).times(&w3);
        let right = w1.times(&w2.times(&w3));
        prop_assert!(left.approx_eq(&right, 1e-4));
    }
    
    #[test]
    fn tropical_identity_elements(a in -100.0..100.0f32) {
        let w = TropicalWeight::new(a);
        
        // w + zero = w
        prop_assert!(w.plus(&TropicalWeight::zero()).approx_eq(&w, 1e-4));
        
        // w * one = w
        prop_assert!(w.times(&TropicalWeight::one()).approx_eq(&w, 1e-4));
    }
    
    #[test]
    fn probability_semiring_properties(a in 0.0..=1.0, b in 0.0..=1.0) {
        let w1 = ProbabilityWeight::new(a);
        let w2 = ProbabilityWeight::new(b);
        
        // sum should be >= max(a, b)
        let sum = w1.plus(&w2);
        assert!(*sum.value() >= a.max(b));
        
        // product should be <= min(a, b)
        let prod = w1.times(&w2);
        assert!(*prod.value() <= a.min(b));
    }
    
    #[test]
    fn boolean_idempotence(a: bool) {
        let w = BooleanWeight::new(a);
        
        // w + w = w (idempotent)
        assert_eq!(w.plus(&w), w);
    }
    
    #[test]
    fn fst_state_consistency(num_states in 1..100usize) {
        let mut fst = VectorFst::<TropicalWeight>::new();
        
        for _ in 0..num_states {
            fst.add_state();
        }
        
        assert_eq!(fst.num_states(), num_states);
        
        // all states should be valid
        for state in fst.states() {
            assert!(state < num_states as StateId);
        }
    }
    
    #[test]
    fn arc_consistency(
        ilabel: u32,
        olabel: u32,
        weight: f32,
        nextstate in 0..100u32,
    ) {
        let arc = Arc::new(
            ilabel,
            olabel,
            TropicalWeight::new(weight),
            nextstate,
        );
        
        assert_eq!(arc.ilabel, ilabel);
        assert_eq!(arc.olabel, olabel);
        assert_eq!(*arc.weight.value(), weight);
        assert_eq!(arc.nextstate, nextstate);
    }
}