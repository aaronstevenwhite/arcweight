//! Comprehensive tests for semiring module

use arcweight::prelude::*;
use arcweight::semiring::*;
use num_traits::{Zero, One};
use proptest::prelude::*;

#[cfg(test)]
mod tropical_weight_tests {
    use super::*;

    #[test]
    fn test_tropical_weight_creation() {
        let w = TropicalWeight::new(5.0);
        assert_eq!(*w.value(), 5.0);
    }

    #[test]
    fn test_tropical_zero_one() {
        let zero = TropicalWeight::zero();
        let one = TropicalWeight::one();
        
        assert!(Semiring::is_zero(&zero));
        assert!(Semiring::is_one(&one));
        assert_eq!(*one.value(), 0.0);
        assert!(zero.value().is_infinite());
    }

    #[test]
    fn test_tropical_addition() {
        let w1 = TropicalWeight::new(3.0);
        let w2 = TropicalWeight::new(5.0);
        let result = w1.plus(&w2);
        
        assert_eq!(*result.value(), 3.0); // min operation
    }

    #[test]
    fn test_tropical_multiplication() {
        let w1 = TropicalWeight::new(3.0);
        let w2 = TropicalWeight::new(5.0);
        let result = w1.times(&w2);
        
        assert_eq!(*result.value(), 8.0); // addition operation
    }

    #[test]
    fn test_tropical_zero_multiplication() {
        let w = TropicalWeight::new(5.0);
        let zero = TropicalWeight::zero();
        let result = w.times(&zero);
        
        assert!(Semiring::is_zero(&result));
    }

    #[test]
    fn test_tropical_one_multiplication() {
        let w = TropicalWeight::new(5.0);
        let one = TropicalWeight::one();
        let result = w.times(&one);
        
        assert_eq!(result, w);
    }

    #[test]
    fn test_tropical_display() {
        let w = TropicalWeight::new(5.0);
        let zero = TropicalWeight::zero();
        
        assert_eq!(format!("{}", w), "5");
        assert_eq!(format!("{}", zero), "∞");
    }

    #[test]
    fn test_tropical_division() {
        let w1 = TropicalWeight::new(8.0);
        let w2 = TropicalWeight::new(3.0);
        
        let result = w1.divide(&w2).unwrap();
        assert_eq!(*result.value(), 5.0);
        
        // Division by zero should return None
        let zero = TropicalWeight::zero();
        assert!(w1.divide(&zero).is_none());
    }

    #[test]
    fn test_tropical_properties() {
        let props = TropicalWeight::properties();
        assert!(props.left_semiring);
        assert!(props.right_semiring);
        assert!(props.commutative);
        assert!(props.idempotent);
        assert!(props.path);
    }

    #[test]
    fn test_tropical_approx_eq() {
        let w1 = TropicalWeight::new(5.000001);
        let w2 = TropicalWeight::new(5.0);
        
        assert!(w1.approx_eq(&w2, 0.001));
        assert!(!w1.approx_eq(&w2, 0.0000001));
    }
}

#[cfg(test)]
mod probability_weight_tests {
    use super::*;

    #[test]
    fn test_probability_weight_creation() {
        let w = ProbabilityWeight::new(0.5);
        assert_eq!(*w.value(), 0.5);
    }

    #[test]
    fn test_probability_zero_one() {
        let zero = ProbabilityWeight::zero();
        let one = ProbabilityWeight::one();
        
        assert!(Semiring::is_zero(&zero));
        assert!(Semiring::is_one(&one));
        assert_eq!(*zero.value(), 0.0);
        assert_eq!(*one.value(), 1.0);
    }

    #[test]
    fn test_probability_addition() {
        let w1 = ProbabilityWeight::new(0.3);
        let w2 = ProbabilityWeight::new(0.5);
        let result = w1.plus(&w2);
        
        assert_eq!(*result.value(), 0.8); // regular addition
    }

    #[test]
    fn test_probability_multiplication() {
        let w1 = ProbabilityWeight::new(0.3);
        let w2 = ProbabilityWeight::new(0.5);
        let result = w1.times(&w2);
        
        assert_eq!(*result.value(), 0.15); // regular multiplication
    }

    #[test]
    fn test_probability_zero_operations() {
        let w = ProbabilityWeight::new(0.5);
        let zero = ProbabilityWeight::zero();
        
        let add_result = w.plus(&zero);
        let mul_result = w.times(&zero);
        
        assert_eq!(add_result, w);
        assert!(Semiring::is_zero(&mul_result));
    }

    #[test]
    fn test_probability_one_operations() {
        let w = ProbabilityWeight::new(0.5);
        let one = ProbabilityWeight::one();
        
        let mul_result = w.times(&one);
        assert_eq!(mul_result, w);
    }
}

#[cfg(test)]
mod boolean_weight_tests {
    use super::*;

    #[test]
    fn test_boolean_weight_creation() {
        let w_true = BooleanWeight::new(true);
        let w_false = BooleanWeight::new(false);
        
        assert_eq!(*w_true.value(), true);
        assert_eq!(*w_false.value(), false);
    }

    #[test]
    fn test_boolean_zero_one() {
        let zero = BooleanWeight::zero();
        let one = BooleanWeight::one();
        
        assert!(Semiring::is_zero(&zero));
        assert!(Semiring::is_one(&one));
        assert_eq!(*zero.value(), false);
        assert_eq!(*one.value(), true);
    }

    #[test]
    fn test_boolean_addition() {
        let w_true = BooleanWeight::new(true);
        let w_false = BooleanWeight::new(false);
        
        // OR operation
        assert_eq!(*w_true.plus(&w_false).value(), true);
        assert_eq!(*w_false.plus(&w_true).value(), true);
        assert_eq!(*w_false.plus(&w_false).value(), false);
        assert_eq!(*w_true.plus(&w_true).value(), true);
    }

    #[test]
    fn test_boolean_multiplication() {
        let w_true = BooleanWeight::new(true);
        let w_false = BooleanWeight::new(false);
        
        // AND operation
        assert_eq!(*w_true.times(&w_false).value(), false);
        assert_eq!(*w_false.times(&w_true).value(), false);
        assert_eq!(*w_false.times(&w_false).value(), false);
        assert_eq!(*w_true.times(&w_true).value(), true);
    }

    #[test]
    fn test_boolean_idempotence() {
        let w_true = BooleanWeight::new(true);
        let w_false = BooleanWeight::new(false);
        
        // w + w = w (idempotent)
        assert_eq!(w_true.plus(&w_true), w_true);
        assert_eq!(w_false.plus(&w_false), w_false);
    }
}

#[cfg(test)]
mod log_weight_tests {
    use super::*;

    #[test]
    fn test_log_weight_creation() {
        let w = LogWeight::new(2.0);
        assert_eq!(*w.value(), 2.0);
    }

    #[test]
    fn test_log_zero_one() {
        let zero = LogWeight::zero();
        let one = LogWeight::one();
        
        assert!(Semiring::is_zero(&zero));
        assert!(Semiring::is_one(&one));
        assert!(zero.value().is_infinite());
        assert_eq!(*one.value(), 0.0);
    }

    #[test]
    fn test_log_addition() {
        let w1 = LogWeight::new(1.0);
        let w2 = LogWeight::new(2.0);
        let result = w1.plus(&w2);
        
        // -log(exp(-1) + exp(-2)) ≈ 0.687
        assert!(result.approx_eq(&LogWeight::new(0.6867), 0.001));
    }

    #[test]
    fn test_log_multiplication() {
        let w1 = LogWeight::new(1.0);
        let w2 = LogWeight::new(2.0);
        let result = w1.times(&w2);
        
        assert_eq!(*result.value(), 3.0); // addition in log space
    }
}

#[cfg(test)]
mod minmax_weight_tests {
    use super::*;

    #[test]
    fn test_min_weight() {
        let w1 = MinWeight::new(5.0);
        let w2 = MinWeight::new(3.0);
        
        let result = w1.plus(&w2);
        assert_eq!(*result.value(), 3.0); // min operation
        
        let result = w1.times(&w2);
        assert_eq!(*result.value(), 5.0); // max operation
    }

    #[test]
    fn test_max_weight() {
        let w1 = MaxWeight::new(5.0);
        let w2 = MaxWeight::new(3.0);
        
        let result = w1.plus(&w2);
        assert_eq!(*result.value(), 5.0); // max operation
        
        let result = w1.times(&w2);
        assert_eq!(*result.value(), 3.0); // min operation
    }
}

#[cfg(test)]
mod string_weight_tests {
    use super::*;

    #[test]
    fn test_string_weight_creation() {
        let w = StringWeight::from_string("hello");
        assert_eq!(w.to_string().unwrap(), "hello");
    }

    #[test]
    fn test_string_zero_one() {
        let zero = StringWeight::zero();
        let one = StringWeight::one();
        
        assert!(Semiring::is_zero(&zero));
        assert!(Semiring::is_one(&one));
        assert_eq!(one.as_bytes(), &[]);
    }

    #[test]
    fn test_string_concatenation() {
        let w1 = StringWeight::from_string("hello");
        let w2 = StringWeight::from_string("world");
        
        let result = w1.times(&w2);
        assert_eq!(result.to_string().unwrap(), "helloworld");
    }

    #[test]
    fn test_string_lcp() {
        let w1 = StringWeight::from_string("hello");
        let w2 = StringWeight::from_string("help");
        
        let result = w1.plus(&w2);
        assert_eq!(result.to_string().unwrap(), "hel");
    }
}

#[cfg(test)]
mod product_weight_tests {
    use super::*;

    #[test]
    fn test_product_weight_creation() {
        let w = ProductWeight::new(TropicalWeight::new(2.0), ProbabilityWeight::new(0.5));
        assert_eq!(*w.w1.value(), 2.0);
        assert_eq!(*w.w2.value(), 0.5);
    }

    #[test]
    fn test_product_zero_one() {
        let zero = ProductWeight::<TropicalWeight, ProbabilityWeight>::zero();
        let one = ProductWeight::<TropicalWeight, ProbabilityWeight>::one();
        
        assert!(Semiring::is_zero(&zero));
        assert!(Semiring::is_one(&one));
        assert!(Semiring::is_zero(&zero.w1));
        assert!(Semiring::is_zero(&zero.w2));
        assert!(Semiring::is_one(&one.w1));
        assert!(Semiring::is_one(&one.w2));
    }

    #[test]
    fn test_product_operations() {
        let w1 = ProductWeight::new(TropicalWeight::new(2.0), ProbabilityWeight::new(0.3));
        let w2 = ProductWeight::new(TropicalWeight::new(3.0), ProbabilityWeight::new(0.5));
        
        let add_result = w1.plus(&w2);
        let mul_result = w1.times(&w2);
        
        // Should apply operations component-wise
        assert_eq!(*add_result.w1.value(), 2.0); // min(2, 3)
        assert_eq!(*add_result.w2.value(), 0.8); // 0.3 + 0.5
        
        assert_eq!(*mul_result.w1.value(), 5.0); // 2 + 3
        assert_eq!(*mul_result.w2.value(), 0.15); // 0.3 * 0.5
    }
}

// Property-based tests
proptest! {
    #[test]
    fn tropical_semiring_axioms(a: f32, b: f32, c: f32) {
        let wa = TropicalWeight::new(a);
        let wb = TropicalWeight::new(b);
        let wc = TropicalWeight::new(c);
        
        // Associativity of addition
        let left = wa.plus(&wb).plus(&wc);
        let right = wa.plus(&wb.plus(&wc));
        assert!(left.approx_eq(&right, 1e-6));
        
        // Associativity of multiplication
        let left = wa.times(&wb).times(&wc);
        let right = wa.times(&wb.times(&wc));
        assert!(left.approx_eq(&right, 1e-6));
        
        // Commutativity of addition
        assert!(wa.plus(&wb).approx_eq(&wb.plus(&wa), 1e-6));
        
        // Commutativity of multiplication
        assert!(wa.times(&wb).approx_eq(&wb.times(&wa), 1e-6));
        
        // Distributivity
        let left = wa.plus(&wb).times(&wc);
        let right = wa.times(&wc).plus(&wb.times(&wc));
        assert!(left.approx_eq(&right, 1e-6));
    }
    
    #[test]
    fn tropical_identity_laws(a: f32) {
        let w = TropicalWeight::new(a);
        let zero = TropicalWeight::zero();
        let one = TropicalWeight::one();
        
        // Additive identity
        assert!(w.plus(&zero).approx_eq(&w, 1e-6));
        assert!(zero.plus(&w).approx_eq(&w, 1e-6));
        
        // Multiplicative identity
        assert!(w.times(&one).approx_eq(&w, 1e-6));
        assert!(one.times(&w).approx_eq(&w, 1e-6));
        
        // Annihilation by zero
        assert!(Semiring::is_zero(&w.times(&zero)));
        assert!(Semiring::is_zero(&zero.times(&w)));
    }
    
    #[test]
    fn probability_weight_bounds(a in 0.0..=1.0, b in 0.0..=1.0) {
        let wa = ProbabilityWeight::new(a);
        let wb = ProbabilityWeight::new(b);
        
        // Addition should be bounded (probability semiring might be defined differently)
        let sum = wa.plus(&wb);
        // Note: In standard probability semiring, addition is regular addition (not bounded by 1.0)
        assert!(*sum.value() >= a.max(b));
        
        // Multiplication should be bounded
        let prod = wa.times(&wb);
        assert!(*prod.value() <= a.min(b));
        assert!(*prod.value() >= 0.0);
    }
    
    #[test]
    fn boolean_weight_correctness(a: bool, b: bool) {
        let wa = BooleanWeight::new(a);
        let wb = BooleanWeight::new(b);
        
        // OR operation
        assert_eq!(*wa.plus(&wb).value(), a || b);
        
        // AND operation
        assert_eq!(*wa.times(&wb).value(), a && b);
        
        // Idempotence
        assert_eq!(wa.plus(&wa), wa);
    }
}