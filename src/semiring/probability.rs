//! Probability semiring implementation

use super::traits::*;
use core::fmt;
use core::ops::{Add, Mul};
use core::str::FromStr;
use num_traits::{One, Zero};
use ordered_float::OrderedFloat;

/// Probability weight for stochastic modeling and Bayesian inference
///
/// The probability semiring provides the mathematical foundation for probabilistic
/// modeling in applications where events are independent and path probabilities
/// should be combined through standard probability theory. This semiring works
/// directly with probability values in the range [0, 1].
///
/// # Mathematical Semantics
///
/// - **Value Range:** Real numbers [0, 1] ⊂ ℝ (probabilities between 0 and 1)
/// - **Addition (⊕):** `a + b` - combines mutually exclusive events
/// - **Multiplication (⊗):** `a × b` - combines independent events
/// - **Zero (0̄):** `0.0` - represents impossible event
/// - **One (1̄):** `1.0` - represents certain event
///
/// # Important Constraints
///
/// **Probability Validity:** The probability semiring requires that the sum of
/// probabilities along alternative paths from any state must not exceed 1.0.
/// This constraint ensures valid probabilistic semantics but must be enforced
/// by the application.
///
/// **Numerical Limitations:** For very small probabilities (< 1e-15), consider
/// using [`LogWeight`](crate::semiring::LogWeight) to avoid underflow issues.
///
/// # Use Cases
///
/// ## Probabilistic Parsing and Grammar Models
/// ```rust
/// use arcweight::prelude::*;
///
/// // Grammar rule probabilities
/// let noun_phrase_prob = ProbabilityWeight::new(0.6);    // 60% chance
/// let verb_phrase_prob = ProbabilityWeight::new(0.8);    // 80% chance
/// let sentence_prob = ProbabilityWeight::new(0.9);       // 90% chance
///
/// // Probability of complete parse (independent events)
/// let parse_prob = noun_phrase_prob
///     .times(&verb_phrase_prob)
///     .times(&sentence_prob);  // 0.6 × 0.8 × 0.9 = 0.432
///
/// println!("Parse probability: {:.3}", parse_prob.value());  // 0.432
/// ```
///
/// ## Speech Recognition Confidence Scoring
/// ```rust
/// use arcweight::prelude::*;
///
/// // Acoustic model confidence scores
/// let acoustic_conf = ProbabilityWeight::new(0.85);
/// let language_conf = ProbabilityWeight::new(0.92);
/// let pronunciation_conf = ProbabilityWeight::new(0.78);
///
/// // Combined confidence (assuming independence)
/// let overall_conf = acoustic_conf
///     .times(&language_conf)
///     .times(&pronunciation_conf);  // 0.85 × 0.92 × 0.78 ≈ 0.610
///
/// // Alternative hypotheses (mutually exclusive)
/// let hypothesis1 = ProbabilityWeight::new(0.6);
/// let hypothesis2 = ProbabilityWeight::new(0.3);
/// let hypothesis3 = ProbabilityWeight::new(0.1);
///
/// let total_prob = hypothesis1
///     .plus(&hypothesis2)
///     .plus(&hypothesis3);  // 0.6 + 0.3 + 0.1 = 1.0
/// ```
///
/// ## Machine Translation Quality Assessment
/// ```rust
/// use arcweight::prelude::*;
///
/// // Translation model components
/// let fluency_score = ProbabilityWeight::new(0.88);      // Translation fluency
/// let adequacy_score = ProbabilityWeight::new(0.75);     // Semantic adequacy
/// let alignment_score = ProbabilityWeight::new(0.92);    // Word alignment quality
///
/// // Overall translation quality (composite score)
/// let translation_quality = fluency_score
///     .times(&adequacy_score)
///     .times(&alignment_score);  // 0.88 × 0.75 × 0.92 ≈ 0.607
///
/// // Multiple translation candidates
/// let candidate1 = ProbabilityWeight::new(0.4);  // Best translation
/// let candidate2 = ProbabilityWeight::new(0.35); // Second best
/// let candidate3 = ProbabilityWeight::new(0.25); // Third option
///
/// let total_mass = candidate1.plus(&candidate2).plus(&candidate3);  // 1.0
/// ```
///
/// ## Bayesian Network Inference
/// ```rust
/// use arcweight::prelude::*;
///
/// // Prior probabilities
/// let prior_disease = ProbabilityWeight::new(0.001);     // 0.1% disease prevalence
/// let prior_healthy = ProbabilityWeight::new(0.999);     // 99.9% healthy
///
/// // Likelihood: P(test_positive | condition)
/// let test_given_disease = ProbabilityWeight::new(0.95); // 95% sensitivity
/// let test_given_healthy = ProbabilityWeight::new(0.02); // 2% false positive
///
/// // Joint probabilities: P(condition, test_result)
/// let disease_and_positive = prior_disease.times(&test_given_disease);
/// let healthy_and_positive = prior_healthy.times(&test_given_healthy);
///
/// // Marginal: P(test_positive)
/// let marginal_positive = disease_and_positive.plus(&healthy_and_positive);
///
/// // Posterior: P(disease | test_positive) = P(disease, positive) / P(positive)
/// let posterior_disease = disease_and_positive.divide(&marginal_positive).unwrap();
/// println!("Posterior probability of disease: {:.3}", posterior_disease.value());
/// ```
///
/// # Working with FSTs
///
/// ```rust
/// use arcweight::prelude::*;
///
/// let prob1 = ProbabilityWeight::new(0.7);  // 70% probability
/// let prob2 = ProbabilityWeight::new(0.3);  // 30% probability
///
/// // Addition combines mutually exclusive events
/// let combined = prob1 + prob2;  // 0.7 + 0.3 = 1.0
/// assert!((combined.value() - 1.0).abs() < 1e-15);
///
/// // Multiplication combines independent events
/// let joint = prob1 * prob2;  // 0.7 × 0.3 = 0.21
/// assert!((joint.value() - 0.21).abs() < 1e-15);
///
/// // Identity elements
/// assert_eq!(ProbabilityWeight::zero(), ProbabilityWeight::new(0.0));
/// assert_eq!(ProbabilityWeight::one(), ProbabilityWeight::new(1.0));
/// ```
///
/// # Probabilistic Properties
///
/// ## Conditional Probability and Division
/// ```rust
/// use arcweight::prelude::*;
///
/// // P(A and B) = 0.3, P(B) = 0.6
/// let joint_prob = ProbabilityWeight::new(0.3);
/// let marginal_prob = ProbabilityWeight::new(0.6);
///
/// // Conditional probability: P(A | B) = P(A and B) / P(B)
/// let conditional = joint_prob.divide(&marginal_prob).unwrap();
/// assert!((conditional.value() - 0.5).abs() < 1e-15);  // P(A | B) = 0.5
/// ```
///
/// ## Kleene Star for Geometric Distribution
/// ```rust
/// use arcweight::prelude::*;
///
/// // Probability of continuing (< 1.0 for convergence)
/// let continue_prob = ProbabilityWeight::new(0.8);
///
/// // Expected number of iterations: 1 / (1 - p)
/// let expected_iterations = continue_prob.star();
/// assert!((expected_iterations.value() - 5.0).abs() < 1e-10);  // 1/(1-0.8) = 5
/// ```
///
/// # Advanced Probabilistic Modeling
///
/// ## Normalized Probability Distributions
/// ```rust
/// use arcweight::prelude::*;
///
/// // Unnormalized probabilities
/// let weights = vec![
///     ProbabilityWeight::new(2.0),  // Unnormalized
///     ProbabilityWeight::new(3.0),
///     ProbabilityWeight::new(1.0),
/// ];
///
/// // Compute normalization constant
/// let total = weights.iter()
///     .fold(ProbabilityWeight::zero(), |acc, &w| acc.plus(&w));
///
/// // Normalize to valid probability distribution
/// let normalized: Vec<_> = weights.iter()
///     .map(|&w| w.divide(&total).unwrap())
///     .collect();
///
/// // Verify normalization
/// let sum = normalized.iter()
///     .fold(ProbabilityWeight::zero(), |acc, &w| acc.plus(&w));
/// assert!((sum.value() - 1.0).abs() < 1e-10);
/// ```
///
/// ## Chain Rule Implementation
/// ```rust
/// use arcweight::prelude::*;
///
/// // Sequence probability using chain rule: P(w1, w2, w3) = P(w1) × P(w2|w1) × P(w3|w1,w2)
/// let p_w1 = ProbabilityWeight::new(0.3);        // P(w1)
/// let p_w2_given_w1 = ProbabilityWeight::new(0.5); // P(w2 | w1)
/// let p_w3_given_w1w2 = ProbabilityWeight::new(0.8); // P(w3 | w1, w2)
///
/// let sequence_prob = p_w1
///     .times(&p_w2_given_w1)
///     .times(&p_w3_given_w1w2);  // 0.3 × 0.5 × 0.8 = 0.12
/// ```
///
/// # Performance Characteristics
///
/// - **Arithmetic:** Both addition and multiplication are O(1) floating-point operations
/// - **Memory:** 8 bytes per weight (single f64)
/// - **Precision:** Double precision for accurate probability computation
/// - **Range:** Supports probabilities from ~1e-308 to 1.0
/// - **Underflow Risk:** Very small probabilities may underflow (use LogWeight instead)
///
/// # Numerical Considerations
///
/// - **Valid Range:** Values must be in [0, 1] for valid probability semantics
/// - **Underflow:** Products of many small probabilities may underflow to 0
/// - **Overflow:** Sums can exceed 1.0, violating probability constraints
/// - **Precision:** IEEE 754 double precision provides ~15-16 decimal digits
///
/// # Integration with FST Algorithms
///
/// Probability weights integrate naturally with FST algorithms:
/// - **Composition:** Combines probabilistic models
/// - **Shortest Path:** Finds maximum probability paths (when using appropriate metrics)
/// - **Forward-Backward:** Computes path probabilities
/// - **Determinization:** Maintains probability distributions
///
/// # Migration to Log Semiring
///
/// For numerically challenging applications:
/// ```rust
/// use arcweight::prelude::*;
///
/// // Convert probability to log weight for stability
/// let prob = ProbabilityWeight::new(1e-20);
/// let log_weight = LogWeight::from_probability(*prob.value());
///
/// // Perform computation in log space
/// let result_log = log_weight.times(&log_weight);
///
/// // Convert back if needed
/// let result_prob = ProbabilityWeight::new(result_log.to_probability());
/// ```
///
/// # See Also
///
/// - [Core Concepts - Probability Semiring](../../docs/core-concepts/semirings.md#probability-semiring) for mathematical background
/// - [`LogWeight`](crate::semiring::LogWeight) for numerically stable probability computation
/// - [`TropicalWeight`](crate::semiring::TropicalWeight) for optimization problems
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProbabilityWeight(OrderedFloat<f64>);

impl ProbabilityWeight {
    /// Create a new probability weight
    pub fn new(value: f64) -> Self {
        debug_assert!(value >= 0.0, "Probability must be non-negative");
        Self(OrderedFloat(value))
    }
}

impl fmt::Display for ProbabilityWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Zero for ProbabilityWeight {
    fn zero() -> Self {
        Self::new(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl One for ProbabilityWeight {
    fn one() -> Self {
        Self::new(1.0)
    }
}

impl Add for ProbabilityWeight {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Mul for ProbabilityWeight {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl Semiring for ProbabilityWeight {
    type Value = f64;

    fn new(value: Self::Value) -> Self {
        Self::new(value)
    }

    fn value(&self) -> &Self::Value {
        &self.0
    }

    fn properties() -> SemiringProperties {
        SemiringProperties {
            left_semiring: true,
            right_semiring: true,
            commutative: true,
            idempotent: false,
            path: false,
        }
    }

    fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
        (self.0 - other.0).abs() < epsilon
    }
}

impl DivisibleSemiring for ProbabilityWeight {
    fn divide(&self, other: &Self) -> Option<Self> {
        if <Self as num_traits::Zero>::is_zero(other) {
            None
        } else {
            Some(Self(self.0 / other.0))
        }
    }
}

impl StarSemiring for ProbabilityWeight {
    fn star(&self) -> Self {
        if *self.0 >= 1.0 {
            Self(OrderedFloat(f64::INFINITY))
        } else {
            Self(OrderedFloat(1.0 / (1.0 - *self.0)))
        }
    }
}

impl FromStr for ProbabilityWeight {
    type Err = std::num::ParseFloatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<f64>().map(Self::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{One, Zero};

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

    #[test]
    fn test_probability_display() {
        let w = ProbabilityWeight::new(0.5);
        assert_eq!(format!("{}", w), "0.5");
    }

    #[test]
    fn test_probability_division() {
        let w1 = ProbabilityWeight::new(0.6);
        let w2 = ProbabilityWeight::new(0.3);

        let result = w1.divide(&w2).unwrap();
        assert_eq!(*result.value(), 2.0);

        // Division by zero should return None
        let zero = ProbabilityWeight::zero();
        assert!(w1.divide(&zero).is_none());
    }

    #[test]
    fn test_probability_star() {
        // For p < 1, star should be 1/(1-p)
        let w = ProbabilityWeight::new(0.5);
        let star_result = w.star();
        assert_eq!(*star_result.value(), 2.0);

        // For p >= 1, star should be infinity
        let w_one = ProbabilityWeight::new(1.0);
        assert!(w_one.star().value().is_infinite());

        let w_greater = ProbabilityWeight::new(1.5);
        assert!(w_greater.star().value().is_infinite());
    }

    #[test]
    fn test_probability_properties() {
        let props = ProbabilityWeight::properties();
        assert!(props.left_semiring);
        assert!(props.right_semiring);
        assert!(props.commutative);
        assert!(!props.idempotent);
        assert!(!props.path);
    }

    #[test]
    fn test_probability_approx_eq() {
        let w1 = ProbabilityWeight::new(0.5000001);
        let w2 = ProbabilityWeight::new(0.5);

        assert!(w1.approx_eq(&w2, 0.001));
        assert!(!w1.approx_eq(&w2, 0.00000001));
    }

    #[test]
    fn test_probability_from_str() {
        assert_eq!(
            ProbabilityWeight::from_str("0.5").unwrap(),
            ProbabilityWeight::new(0.5)
        );
    }

    #[test]
    fn test_probability_operator_overloads() {
        let w1 = ProbabilityWeight::new(0.3);
        let w2 = ProbabilityWeight::new(0.5);

        // Test + operator (addition)
        assert_eq!(w1 + w2, ProbabilityWeight::new(0.8));

        // Test * operator (multiplication)
        assert_eq!(w1 * w2, ProbabilityWeight::new(0.15));
    }

    #[test]
    fn test_probability_identity_laws() {
        let w = ProbabilityWeight::new(0.5);
        let zero = ProbabilityWeight::zero();
        let one = ProbabilityWeight::one();

        // Additive identity
        assert_eq!(w + zero, w);
        assert_eq!(zero + w, w);

        // Multiplicative identity
        assert_eq!(w * one, w);
        assert_eq!(one * w, w);

        // Annihilation by zero
        assert!(Semiring::is_zero(&(w * zero)));
        assert!(Semiring::is_zero(&(zero * w)));
    }

    #[test]
    fn test_probability_semiring_axioms() {
        let a = ProbabilityWeight::new(0.2);
        let b = ProbabilityWeight::new(0.3);
        let c = ProbabilityWeight::new(0.4);
        let tolerance = 1e-10;

        // Associativity of addition
        assert!(((a + b) + c).approx_eq(&(a + (b + c)), tolerance));

        // Associativity of multiplication
        assert!(((a * b) * c).approx_eq(&(a * (b * c)), tolerance));

        // Commutativity of addition
        assert_eq!(a + b, b + a);

        // Commutativity of multiplication
        assert_eq!(a * b, b * a);

        // Distributivity
        assert!(((a + b) * c).approx_eq(&((a * c) + (b * c)), tolerance));
    }

    // Property-based tests
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn test_probability_bounds_property(a in 0.0..=1.0, b in 0.0..=1.0) {
                let w1 = ProbabilityWeight::new(a);
                let w2 = ProbabilityWeight::new(b);

                // sum should be >= max(a, b)
                let sum = w1.plus(&w2);
                assert!(*sum.value() >= a.max(b));

                // product should be <= min(a, b)
                let prod = w1.times(&w2);
                assert!(*prod.value() <= a.min(b));
                assert!(*prod.value() >= 0.0);
            }
        }
    }
}
