//! Real number semiring implementation with multiplicative inverses

use super::traits::*;
use core::fmt;
use core::ops::{Add, Mul};
use num_traits::{One, Zero};
use ordered_float::OrderedFloat;
use std::str::FromStr;

/// Real number weight for algebraic computations requiring multiplicative inverses
///
/// The Real semiring provides a mathematical framework for computations over real numbers
/// where multiplicative inverses are required. Unlike tropical or log semirings that use
/// specialized operations, this semiring uses standard arithmetic with addition for ⊕
/// and multiplication for ⊗, making it suitable for linear algebraic computations,
/// normalization procedures, and algorithms requiring division operations.
///
/// # Mathematical Semantics
///
/// - **Value Range:** Real numbers ℝ
/// - **Addition (⊕):** `a + b` - standard real number addition
/// - **Multiplication (⊗):** `a × b` - standard real number multiplication
/// - **Zero (0̄):** `0.0` - additive identity
/// - **One (1̄):** `1.0` - multiplicative identity
/// - **Inverse:** `1/a` for `a ≠ 0` - multiplicative inverse
///
/// # Key Properties
///
/// - **Commutative:** Both addition and multiplication are commutative
/// - **Associative:** Operations follow standard arithmetic associativity
/// - **Distributive:** Multiplication distributes over addition
/// - **Invertible:** Non-zero elements have multiplicative inverses
/// - **Field-like:** Approaches field structure (missing additive inverses)
///
/// # Use Cases
///
/// ## Linear Algebra and Matrix Operations
/// ```rust
/// use arcweight::prelude::*;
///
/// // Matrix element operations requiring division
/// let det_a = RealWeight::new(4.0);  // Matrix determinant component
/// let det_b = RealWeight::new(2.0);  // Another component
///
/// // Compute matrix element: det_a / det_b
/// if let Some(inv_b) = det_b.inverse() {
///     let matrix_element = det_a.times(&inv_b);  // 4.0 / 2.0 = 2.0
///     assert_eq!(matrix_element, RealWeight::new(2.0));
/// }
/// ```
///
/// ## Statistical Normalization
/// ```rust
/// use arcweight::prelude::*;
///
/// // Normalize a set of weights to sum to 1.0
/// let weights = vec![
///     RealWeight::new(10.0),
///     RealWeight::new(20.0),
///     RealWeight::new(30.0),
/// ];
///
/// // Calculate total
/// let total = weights.iter()
///     .fold(RealWeight::zero(), |acc, w| acc.plus(w));  // 60.0
///
/// // Normalize using multiplicative inverse
/// if let Some(inv_total) = total.inverse() {
///     let normalized: Vec<_> = weights.iter()
///         .map(|w| w.times(&inv_total))
///         .collect();
///     
///     // Verify normalization: should sum to 1.0
///     let sum = normalized.iter()
///         .fold(RealWeight::zero(), |acc, w| acc.plus(w));
///     assert!((sum.value() - 1.0).abs() < 1e-10);
/// }
/// ```
///
/// ## Signal Processing and Filtering
/// ```rust
/// use arcweight::prelude::*;
///
/// // Digital filter coefficient computation
/// let signal_power = RealWeight::new(100.0);   // Signal power
/// let noise_power = RealWeight::new(25.0);     // Noise power
///
/// // Signal-to-noise ratio computation
/// if let Some(invₙoise) = noise_power.inverse() {
///     let snr = signal_power.times(&invₙoise);  // 100 / 25 = 4.0
///     println!("SNR: {}", snr);  // 4.0
/// }
/// ```
///
/// ## Economic and Financial Modeling
/// ```rust
/// use arcweight::prelude::*;
///
/// // Price elasticity calculation
/// let price_change = RealWeight::new(0.10);     // 10% price increase
/// let demand_change = RealWeight::new(-0.05);   // 5% demand decrease
///
/// // Note: This demonstrates the mathematical operations;
/// // real elasticity would use absolute values and proper signs
/// let elasticity_magnitude = demand_change.value().abs();
/// let price_magnitude = price_change.value();
///
/// let elasticityₙum = RealWeight::new(elasticity_magnitude);
/// if let Some(inv_price) = RealWeight::new(*price_magnitude).inverse() {
///     let elasticity = elasticityₙum.times(&inv_price);
///     // Elasticity coefficient: 0.05 / 0.10 = 0.5
/// }
/// ```
///
/// ## Weight Normalization in FSTs
/// ```rust
/// use arcweight::prelude::*;
///
/// // Normalize FST arc weights for probabilistic interpretation
/// let arc_weights = vec![
///     RealWeight::new(3.0),
///     RealWeight::new(6.0),
///     RealWeight::new(9.0),
/// ];
///
/// let total_weight = arc_weights.iter()
///     .fold(RealWeight::zero(), |acc, w| acc.plus(w));
///
/// if let Some(normalizer) = total_weight.inverse() {
///     let normalized_weights: Vec<_> = arc_weights.iter()
///         .map(|w| w.times(&normalizer))
///         .collect();
///     
///     // Weights now sum to 1.0: [1/6, 2/6, 3/6] = [0.167, 0.333, 0.5]
/// }
/// ```
///
/// # Working with FSTs
///
/// ```rust
/// use arcweight::prelude::*;
///
/// let a = RealWeight::new(6.0);
/// let b = RealWeight::new(2.0);
///
/// // Standard arithmetic operations
/// let sum = a.plus(&b);        // 6.0 + 2.0 = 8.0
/// let product = a.times(&b);   // 6.0 × 2.0 = 12.0
///
/// // Division via multiplicative inverse
/// if let Some(inv_b) = b.inverse() {
///     let quotient = a.times(&inv_b);  // 6.0 ÷ 2.0 = 3.0
///     assert_eq!(quotient, RealWeight::new(3.0));
/// }
///
/// // Identity elements
/// assert_eq!(RealWeight::zero(), RealWeight::new(0.0));  // Additive identity
/// assert_eq!(RealWeight::one(), RealWeight::new(1.0));   // Multiplicative identity
/// ```
///
/// # Advanced Applications
///
/// ## Solving Linear Systems
/// ```rust
/// use arcweight::prelude::*;
///
/// // Solve simple equation: ax = b for x
/// let a = RealWeight::new(5.0);
/// let b = RealWeight::new(15.0);
///
/// if let Some(inv_a) = a.inverse() {
///     let x = b.times(&inv_a);  // x = b/a = 15/5 = 3
///     
///     // Verify: a × x = b
///     let verification = a.times(&x);
///     assert_eq!(verification, b);
/// }
/// ```
///
/// ## Geometric Mean Computation
/// ```rust
/// use arcweight::prelude::*;
///
/// // Compute geometric mean: (a × b × c)^(1/3)
/// let values = vec![
///     RealWeight::new(8.0),
///     RealWeight::new(27.0),
///     RealWeight::new(64.0),
/// ];
///
/// // Product of all values
/// let product = values.iter()
///     .fold(RealWeight::one(), |acc, v| acc.times(v));  // 8 × 27 × 64
///
/// // For geometric mean, we'd need nth roots, but this shows the principle
/// // In practice: geometric_mean = product^(1/n)
/// ```
///
/// ## Bayesian Weight Updates
/// ```rust
/// use arcweight::prelude::*;
///
/// // Bayesian posterior computation with normalization
/// let prior = RealWeight::new(0.3);
/// let likelihood = RealWeight::new(0.8);
/// let evidence = RealWeight::new(0.6);
///
/// // Posterior = (prior × likelihood) / evidence
/// let numerator = prior.times(&likelihood);
/// if let Some(inv_evidence) = evidence.inverse() {
///     let posterior = numerator.times(&inv_evidence);
///     // Posterior = (0.3 × 0.8) / 0.6 = 0.24 / 0.6 = 0.4
/// }
/// ```
///
/// # Performance Characteristics
///
/// - **Arithmetic:** O(1) operations using hardware floating-point units
/// - **Memory:** 8 bytes per weight (f64 precision)
/// - **Precision:** IEEE 754 double precision (~15-17 significant digits)
/// - **Range:** Approximately ±1.8 × 10³08
/// - **Inverse Computation:** O(1) hardware division operation
/// - **Numerical Stability:** Handles normal range well, guards against division by zero
///
/// # Mathematical Properties
///
/// The Real semiring satisfies all semiring axioms plus additional structure:
/// - **Semiring Axioms:** ✓ All standard semiring properties hold
/// - **Commutativity:** ✓ Both operations are commutative
/// - **Invertibility:** ✓ All non-zero elements have multiplicative inverses
/// - **Distributivity:** ✓ Standard arithmetic distributivity
/// - **Field-like Structure:** Nearly a field (missing only additive inverses)
///
/// # Numerical Considerations
///
/// ## Precision and Stability
/// - Uses `f64` for high precision arithmetic
/// - Guards against division by zero and near-zero values
/// - Provides configurable epsilon for floating-point comparisons
/// - Handles infinity and NaN according to IEEE 754 standards
///
/// ## Edge Cases
/// - **Zero Division:** `0.inverse()` returns `None`
/// - **Infinity:** Operations with infinity follow IEEE 754 rules
/// - **Very Small Numbers:** May underflow to zero; inverse may overflow
/// - **Very Large Numbers:** May overflow to infinity
///
/// # Integration with FST Algorithms
///
/// Real weights work with all FST algorithms:
/// - **Shortest Path:** Standard algorithms apply (though paths are additive, not minimal)
/// - **Composition:** Supports full composition with weight combination
/// - **Determinization:** Maintains arithmetic structure through determinization
/// - **Minimization:** Preserves mathematical relationships during minimization
/// - **Weight Pushing:** Enables sophisticated normalization and optimization
///
/// # See Also
///
/// - [`DivisibleSemiring`] for division without full invertibility
/// - [`crate::semiring::TropicalWeight`] for shortest-path optimization
/// - [`crate::semiring::LogWeight`] for probabilistic computations with better numerical stability
/// - [Core Concepts - Real Semiring](../../docs/core-concepts/semirings.md#real-semiring) for mathematical background
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RealWeight(OrderedFloat<f64>);

impl RealWeight {
    /// Create a new real weight
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// let weight = RealWeight::new(3.14159);
    /// assert_eq!(weight.value(), &3.14159);
    /// ```
    pub fn new(value: f64) -> Self {
        Self(OrderedFloat(value))
    }

    /// Create weight from integer
    ///
    /// Convenience method for creating weights from integer values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// let weight = RealWeight::from_int(42);
    /// assert_eq!(weight, RealWeight::new(42.0));
    /// ```
    pub fn from_int(value: i32) -> Self {
        Self::new(value as f64)
    }

    /// Create weight from rational numbers
    ///
    /// Creates a weight representing the fraction numerator/denominator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// let weight = RealWeight::from_fraction(3, 4);  // 3/4 = 0.75
    /// assert_eq!(weight, RealWeight::new(0.75));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if denominator is zero.
    pub fn from_fraction(numerator: i32, denominator: i32) -> Self {
        assert_ne!(denominator, 0, "Denominator cannot be zero");
        Self::new(numerator as f64 / denominator as f64)
    }

    /// Get the raw floating-point value
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// let weight = RealWeight::new(2.718);
    /// assert_eq!(weight.as_f64(), 2.718);
    /// ```
    pub fn as_f64(&self) -> f64 {
        *self.0
    }

    /// Check if the weight represents positive infinity
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// let inf_weight = RealWeight::new(f64::INFINITY);
    /// assert!(inf_weight.is_infinite());
    ///
    /// let normal_weight = RealWeight::new(42.0);
    /// assert!(!normal_weight.is_infinite());
    /// ```
    pub fn is_infinite(&self) -> bool {
        self.0.is_infinite()
    }

    /// Check if the weight is NaN (Not a Number)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// let nan_weight = RealWeight::new(f64::NAN);
    /// assert!(nan_weight.is_nan());
    ///
    /// let normal_weight = RealWeight::new(42.0);
    /// assert!(!normal_weight.is_nan());
    /// ```
    pub fn is_nan(&self) -> bool {
        self.0.is_nan()
    }

    /// Compute absolute value
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// let negative = RealWeight::new(-5.0);
    /// let positive = negative.abs();
    /// assert_eq!(positive, RealWeight::new(5.0));
    /// ```
    pub fn abs(&self) -> Self {
        Self::new(self.0.abs())
    }

    /// Compute square root
    ///
    /// Returns `None` for negative values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// let weight = RealWeight::new(16.0);
    /// if let Some(sqrt_weight) = weight.sqrt() {
    ///     assert_eq!(sqrt_weight, RealWeight::new(4.0));
    /// }
    ///
    /// let negative = RealWeight::new(-4.0);
    /// assert_eq!(negative.sqrt(), None);
    /// ```
    pub fn sqrt(&self) -> Option<Self> {
        if *self.0 < 0.0 {
            None
        } else {
            Some(Self::new(self.0.sqrt()))
        }
    }

    /// Raise to a power
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// let base = RealWeight::new(2.0);
    /// let result = base.pow(3.0);  // 2³ = 8
    /// assert_eq!(result, RealWeight::new(8.0));
    /// ```
    pub fn pow(&self, exponent: f64) -> Self {
        Self::new(self.0.powf(exponent))
    }
}

impl fmt::Display for RealWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_infinite() {
            if self.0.is_sign_positive() {
                write!(f, "∞")
            } else {
                write!(f, "-∞")
            }
        } else if self.0.is_nan() {
            write!(f, "NaN")
        } else {
            write!(f, "{}", self.0)
        }
    }
}

impl Zero for RealWeight {
    fn zero() -> Self {
        Self::new(0.0)
    }

    fn is_zero(&self) -> bool {
        *self.0 == 0.0
    }
}

impl One for RealWeight {
    fn one() -> Self {
        Self::new(1.0)
    }
}

impl Add for RealWeight {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(*self.0 + *rhs.0)
    }
}

impl Mul for RealWeight {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(*self.0 * *rhs.0)
    }
}

impl Semiring for RealWeight {
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
            idempotent: false, // a + a ≠ a for most real numbers
            path: false,       // a + b ∉ {a, b} in general
        }
    }

    fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
        (*self.0 - *other.0).abs() < epsilon
    }
}

impl DivisibleSemiring for RealWeight {
    fn divide(&self, other: &Self) -> Option<Self> {
        if Zero::is_zero(other) {
            None
        } else {
            Some(Self::new(*self.0 / *other.0))
        }
    }
}

impl InvertibleSemiring for RealWeight {
    fn inverse(&self) -> Option<Self> {
        if Zero::is_zero(self) {
            None
        } else {
            Some(Self::new(1.0 / *self.0))
        }
    }
}

impl NaturallyOrderedSemiring for RealWeight {}

impl FromStr for RealWeight {
    type Err = std::num::ParseFloatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<f64>().map(Self::new)
    }
}

impl Default for RealWeight {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_construction() {
        let w1 = RealWeight::new(3.5);
        assert_eq!(w1.as_f64(), 3.5);
        assert_eq!(*w1.value(), 3.5);

        let w2 = RealWeight::from_int(42);
        assert_eq!(w2, RealWeight::new(42.0));

        let w3 = RealWeight::from_fraction(3, 4);
        assert_eq!(w3, RealWeight::new(0.75));
    }

    #[test]
    #[should_panic(expected = "Denominator cannot be zero")]
    fn test_fraction_zero_denominator() {
        RealWeight::from_fraction(1, 0);
    }

    #[test]
    fn test_semiring_identities() {
        let zero = RealWeight::zero();
        let one = RealWeight::one();

        assert_eq!(zero, RealWeight::new(0.0));
        assert_eq!(one, RealWeight::new(1.0));
        assert!(Zero::is_zero(&zero));
        assert!(!Zero::is_zero(&one));
    }

    #[test]
    fn test_arithmetic_operations() {
        let a = RealWeight::new(6.0);
        let b = RealWeight::new(4.0);

        // Addition (plus operation)
        let sum = a.plus(&b);
        assert_eq!(sum, RealWeight::new(10.0));

        // Multiplication (times operation)
        let product = a.times(&b);
        assert_eq!(product, RealWeight::new(24.0));

        // Operator overloads
        let sum2 = a + b;
        let product2 = a * b;
        assert_eq!(sum, sum2);
        assert_eq!(product, product2);
    }

    #[test]
    fn test_semiring_properties() {
        let a = RealWeight::new(2.0);
        let b = RealWeight::new(3.0);
        let c = RealWeight::new(5.0);
        let zero = RealWeight::zero();
        let one = RealWeight::one();

        // Associativity of addition
        let left_assoc = (a.plus(&b)).plus(&c);
        let right_assoc = a.plus(&(b.plus(&c)));
        assert!(left_assoc.approx_eq(&right_assoc, 1e-10));

        // Commutativity of addition
        let ab = a.plus(&b);
        let ba = b.plus(&a);
        assert_eq!(ab, ba);

        // Additive identity
        let a_plus_zero = a.plus(&zero);
        assert_eq!(a_plus_zero, a);

        // Associativity of multiplication
        let left_mult_assoc = (a.times(&b)).times(&c);
        let right_mult_assoc = a.times(&(b.times(&c)));
        assert!(left_mult_assoc.approx_eq(&right_mult_assoc, 1e-10));

        // Commutativity of multiplication
        let ab_mult = a.times(&b);
        let ba_mult = b.times(&a);
        assert_eq!(ab_mult, ba_mult);

        // Multiplicative identity
        let a_times_one = a.times(&one);
        assert_eq!(a_times_one, a);

        // Zero is annihilator
        let a_times_zero = a.times(&zero);
        assert_eq!(a_times_zero, zero);

        // Distributivity
        let left_dist = a.times(&(b.plus(&c)));
        let right_dist = (a.times(&b)).plus(&(a.times(&c)));
        assert!(left_dist.approx_eq(&right_dist, 1e-10));
    }

    #[test]
    fn test_properties_structure() {
        let props = RealWeight::properties();
        assert!(props.left_semiring);
        assert!(props.right_semiring);
        assert!(props.commutative);
        assert!(!props.idempotent); // Real addition is not idempotent
        assert!(!props.path); // Real addition doesn't have path property
    }

    #[test]
    fn test_divisible_semiring() {
        let a = RealWeight::new(15.0);
        let b = RealWeight::new(3.0);
        let zero = RealWeight::zero();

        // Normal division
        let quotient = a.divide(&b).unwrap();
        assert_eq!(quotient, RealWeight::new(5.0));

        // Verify division identity: a = (a/b) * b
        let verification = quotient.times(&b);
        assert!(verification.approx_eq(&a, 1e-10));

        // Division by zero
        assert_eq!(a.divide(&zero), None);
    }

    #[test]
    fn test_invertible_semiring() {
        let a = RealWeight::new(4.0);
        let zero = RealWeight::zero();
        let one = RealWeight::one();

        // Normal inverse
        let inv_a = a.inverse().unwrap();
        assert_eq!(inv_a, RealWeight::new(0.25));

        // Inverse identity: a * a^(-1) = 1
        let identity_check = a.times(&inv_a);
        assert!(identity_check.approx_eq(&one, 1e-10));

        // Zero has no inverse
        assert_eq!(zero.inverse(), None);

        // One is its own inverse
        let inv_one = one.inverse().unwrap();
        assert_eq!(inv_one, one);
    }

    #[test]
    fn test_inverse_properties() {
        let values = [0.1, 0.5, 1.0, 2.0, 10.0, 100.0];
        let one = RealWeight::one();

        for &val in &values {
            let w = RealWeight::new(val);
            if let Some(inv_w) = w.inverse() {
                // w * w^(-1) = 1
                let product = w.times(&inv_w);
                assert!(
                    product.approx_eq(&one, 1e-10),
                    "Failed for value {}: {} * {} = {}",
                    val,
                    w,
                    inv_w,
                    product
                );

                // (w^(-1))^(-1) = w
                if let Some(inv_inv_w) = inv_w.inverse() {
                    assert!(
                        inv_inv_w.approx_eq(&w, 1e-10),
                        "Double inverse failed for value {}",
                        val
                    );
                }
            }
        }
    }

    #[test]
    fn test_mathematical_operations() {
        let a = RealWeight::new(-5.0);
        assert_eq!(a.abs(), RealWeight::new(5.0));

        let b = RealWeight::new(16.0);
        assert_eq!(b.sqrt().unwrap(), RealWeight::new(4.0));

        let negative = RealWeight::new(-4.0);
        assert_eq!(negative.sqrt(), None);

        let base = RealWeight::new(2.0);
        assert_eq!(base.pow(3.0), RealWeight::new(8.0));
    }

    #[test]
    fn test_special_values() {
        let inf = RealWeight::new(f64::INFINITY);
        let neg_inf = RealWeight::new(f64::NEG_INFINITY);
        let nan = RealWeight::new(f64::NAN);

        assert!(inf.is_infinite());
        assert!(neg_inf.is_infinite());
        assert!(nan.is_nan());

        // Test display formatting
        assert_eq!(format!("{}", inf), "∞");
        assert_eq!(format!("{}", neg_inf), "-∞");
        assert_eq!(format!("{}", nan), "NaN");

        // Infinity operations
        assert_eq!(inf.inverse(), Some(RealWeight::new(0.0)));
        assert_eq!(RealWeight::new(0.0).inverse(), None);
    }

    #[test]
    fn test_normalization_example() {
        let weights = [
            RealWeight::new(10.0),
            RealWeight::new(20.0),
            RealWeight::new(30.0),
        ];

        let total = weights
            .iter()
            .fold(RealWeight::zero(), |acc, w| acc.plus(w));
        assert_eq!(total, RealWeight::new(60.0));

        if let Some(inv_total) = total.inverse() {
            let normalized: Vec<_> = weights.iter().map(|w| w.times(&inv_total)).collect();

            // Check individual normalized values
            assert!(normalized[0].approx_eq(&RealWeight::new(1.0 / 6.0), 1e-10));
            assert!(normalized[1].approx_eq(&RealWeight::new(2.0 / 6.0), 1e-10));
            assert!(normalized[2].approx_eq(&RealWeight::new(3.0 / 6.0), 1e-10));

            // Verify they sum to 1.0
            let sum = normalized
                .iter()
                .fold(RealWeight::zero(), |acc, w| acc.plus(w));
            assert!(sum.approx_eq(&RealWeight::one(), 1e-10));
        }
    }

    #[test]
    fn test_linear_system_solving() {
        // Solve ax = b for x
        let a = RealWeight::new(5.0);
        let b = RealWeight::new(15.0);

        if let Some(inv_a) = a.inverse() {
            let x = b.times(&inv_a);
            assert_eq!(x, RealWeight::new(3.0));

            // Verify: a * x = b
            let verification = a.times(&x);
            assert!(verification.approx_eq(&b, 1e-10));
        }
    }

    #[test]
    fn test_approximate_equality() {
        let a = RealWeight::new(1.0);
        let b = RealWeight::new(1.0000001);
        let c = RealWeight::new(1.1);

        assert!(a.approx_eq(&b, 1e-6));
        assert!(!a.approx_eq(&b, 1e-8));
        assert!(!a.approx_eq(&c, 1e-6));
    }

    #[test]
    fn test_ordering() {
        let a = RealWeight::new(1.0);
        let b = RealWeight::new(2.0);
        let c = RealWeight::new(1.0);

        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, c);
        assert!(a <= c);
        assert!(a >= c);
    }

    #[test]
    fn test_string_parsing() {
        assert_eq!("3.5".parse::<RealWeight>().unwrap(), RealWeight::new(3.5));
        assert_eq!("-2.5".parse::<RealWeight>().unwrap(), RealWeight::new(-2.5));
        assert!("not_a_number".parse::<RealWeight>().is_err());
    }

    #[test]
    fn test_default() {
        let default_weight = RealWeight::default();
        assert_eq!(default_weight, RealWeight::zero());
    }

    #[test]
    fn test_in_place_operations() {
        let mut a = RealWeight::new(5.0);
        let b = RealWeight::new(3.0);

        a.plus_assign(&b);
        assert_eq!(a, RealWeight::new(8.0));

        a.times_assign(&b);
        assert_eq!(a, RealWeight::new(24.0));
    }

    #[test]
    fn test_numerical_stability() {
        // Test with very small numbers
        let small = RealWeight::new(1e-100);
        let large = RealWeight::new(1e100);

        if let Some(inv_small) = small.inverse() {
            assert!(inv_small.is_infinite() || inv_small.as_f64() > 1e90);
        }

        if let Some(inv_large) = large.inverse() {
            assert!(inv_large.as_f64() < 1e-90);
        }

        // Test overflow/underflow behavior
        let very_large = RealWeight::new(f64::MAX);
        let very_small = RealWeight::new(f64::MIN_POSITIVE);

        assert!(!very_large.is_infinite());
        assert!(!Zero::is_zero(&very_small));
    }

    #[test]
    fn test_bayesian_example() {
        let prior = RealWeight::new(0.3);
        let likelihood = RealWeight::new(0.8);
        let evidence = RealWeight::new(0.6);

        let numerator = prior.times(&likelihood);
        assert_eq!(numerator, RealWeight::new(0.24));

        if let Some(inv_evidence) = evidence.inverse() {
            let posterior = numerator.times(&inv_evidence);
            assert!(posterior.approx_eq(&RealWeight::new(0.4), 1e-10));
        }
    }

    #[test]
    fn test_edge_cases() {
        let zero = RealWeight::zero();
        let _one = RealWeight::one();
        let inf = RealWeight::new(f64::INFINITY);

        // Edge case: inverse of very small number
        let tiny = RealWeight::new(f64::MIN_POSITIVE);
        if let Some(inv_tiny) = tiny.inverse() {
            assert!(inv_tiny.as_f64() > 1e100);
        }

        // Edge case: operations with infinity
        let finite = RealWeight::new(42.0);
        let inf_plus_finite = inf.plus(&finite);
        assert!(inf_plus_finite.is_infinite());

        let inf_times_finite = inf.times(&finite);
        assert!(inf_times_finite.is_infinite());

        // Edge case: division involving infinity
        if let Some(divided) = inf.divide(&finite) {
            assert!(divided.is_infinite());
        }

        if let Some(inv_inf) = inf.inverse() {
            assert_eq!(inv_inf, zero);
        }
    }
}
