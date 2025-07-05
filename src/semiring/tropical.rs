//! Tropical semiring implementation

use super::traits::*;
use core::fmt;
use core::ops::{Add, Mul};
use core::str::FromStr;
use num_traits::{One, Zero};
use ordered_float::OrderedFloat;

/// Tropical weight for shortest-path computation and optimization
///
/// The tropical semiring provides the mathematical foundation for shortest-path
/// algorithms and min-cost optimization problems. In this semiring, "addition"
/// selects the minimum cost path, while "multiplication" accumulates costs.
///
/// # Mathematical Semantics
///
/// - **Value Range:** Real numbers ℝ plus positive infinity (+∞)
/// - **Addition (⊕):** `min(a, b)` - selects the better of two alternatives
/// - **Multiplication (⊗):** `a + b` - accumulates cost along a path
/// - **Zero (0̄):** `+∞` - represents impossible/blocked paths
/// - **One (1̄):** `0.0` - represents free/zero-cost transitions
///
/// # Use Cases
///
/// ## Shortest Path Problems
/// ```rust
/// use arcweight::prelude::*;
///
/// // Represent path costs in a graph
/// let path1_cost = TropicalWeight::new(3.5);  // Cost via route 1
/// let path2_cost = TropicalWeight::new(2.1);  // Cost via route 2
///
/// // Select minimum cost path
/// let best_path = path1_cost.plus(&path2_cost);  // min(3.5, 2.1) = 2.1
/// println!("Best path cost: {}", best_path);  // 2.1
///
/// // Accumulate cost along chosen path
/// let segment1 = TropicalWeight::new(1.0);
/// let segment2 = TropicalWeight::new(1.1);
/// let total_cost = segment1.times(&segment2);  // 1.0 + 1.1 = 2.1
/// assert_eq!(total_cost, best_path);
/// ```
///
/// ## Edit Distance Computation
/// ```rust
/// use arcweight::prelude::*;
///
/// // Edit operations have costs
/// let insertion_cost = TropicalWeight::new(1.0);
/// let deletion_cost = TropicalWeight::new(1.0);
/// let substitution_cost = TropicalWeight::new(1.5);
///
/// // Choose minimum cost operation
/// let min_edit = insertion_cost
///     .plus(&deletion_cost)
///     .plus(&substitution_cost);  // min(1.0, 1.0, 1.5) = 1.0
///
/// // Build total edit distance by accumulating operations
/// let edit_sequence = min_edit.times(&insertion_cost);  // 1.0 + 1.0 = 2.0
/// ```
///
/// ## FST Weights in Practice
/// ```rust
/// use arcweight::prelude::*;
///
/// // Build weighted FST for spell correction
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s1, TropicalWeight::one());  // Accept with zero cost
///
/// // Add edit operation: substitute 'a' -> 'e' with cost 0.5
/// fst.add_arc(s0, Arc::new(
///     'a' as u32,
///     'e' as u32,
///     TropicalWeight::new(0.5),  // Substitution cost
///     s1
/// ));
/// ```
///
/// # Working with FSTs
///
/// ```rust
/// use arcweight::prelude::*;
///
/// let w1 = TropicalWeight::new(0.5);
/// let w2 = TropicalWeight::new(0.3);
///
/// // Addition is minimum (path selection)
/// let sum = w1 + w2;
/// assert_eq!(sum, TropicalWeight::new(0.3));
///
/// // Multiplication is addition (cost accumulation)
/// let product = w1 * w2;
/// assert_eq!(product, TropicalWeight::new(0.8));
///
/// // Zero is infinity (impossible path)
/// assert!(<TropicalWeight as num_traits::Zero>::is_zero(&TropicalWeight::zero()));
/// assert_eq!(TropicalWeight::zero(), TropicalWeight::INFINITY);
///
/// // One is 0.0 (free transition)
/// assert_eq!(TropicalWeight::one(), TropicalWeight::new(0.0));
/// ```
///
/// # Numerical Considerations
///
/// - **Precision:** Uses `f32` for memory efficiency in large FSTs
/// - **Infinity:** Represents truly unreachable states (use `TropicalWeight::zero()`)
/// - **Overflow:** Addition is overflow-safe (min operation), multiplication can overflow
/// - **Comparison:** Implements total ordering for shortest-path algorithms
///
/// # Performance Characteristics
///
/// - **Arithmetic:** Both addition (min) and multiplication (+) are O(1)
/// - **Memory:** 4 bytes per weight (single f32)
/// - **Comparison:** Fast floating-point comparison for priority queues
/// - **Hash/Eq:** Supports use in hash maps for efficient algorithm implementation
///
/// # See Also
///
/// - [Core Concepts - Tropical Semiring](../../docs/core-concepts/semirings.md#tropical-semiring) for mathematical background
/// - [`LogWeight`](crate::semiring::LogWeight) for numerically stable probability computation
/// - [`shortest_path()`](crate::algorithms::shortest_path) for algorithms using this semiring
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TropicalWeight(OrderedFloat<f32>);

impl TropicalWeight {
    /// Positive infinity (zero element)
    pub const INFINITY: Self = Self(OrderedFloat(f32::INFINITY));

    /// Create a new tropical weight
    ///
    /// # Examples
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let weight = TropicalWeight::new(0.5);
    /// assert_eq!(weight.value(), &0.5);
    ///
    /// let zero_weight = TropicalWeight::new(f32::INFINITY);
    /// assert!(<TropicalWeight as num_traits::Zero>::is_zero(&zero_weight));
    /// ```
    pub fn new(value: f32) -> Self {
        Self(OrderedFloat(value))
    }
}

impl fmt::Display for TropicalWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_infinite() {
            write!(f, "∞")
        } else {
            write!(f, "{}", self.0)
        }
    }
}

impl Zero for TropicalWeight {
    fn zero() -> Self {
        Self::INFINITY
    }

    fn is_zero(&self) -> bool {
        self.0.is_infinite()
    }
}

impl One for TropicalWeight {
    fn one() -> Self {
        Self::new(0.0)
    }
}

impl Add for TropicalWeight {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.min(rhs.0))
    }
}

impl Mul for TropicalWeight {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if <Self as num_traits::Zero>::is_zero(&self) || <Self as num_traits::Zero>::is_zero(&rhs) {
            Self::zero()
        } else {
            Self(self.0 + rhs.0)
        }
    }
}

impl Semiring for TropicalWeight {
    type Value = f32;

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
            idempotent: true,
            path: true,
        }
    }

    fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
        if <Self as num_traits::Zero>::is_zero(self) && <Self as num_traits::Zero>::is_zero(other) {
            true
        } else {
            (self.0 - other.0).abs() < epsilon as f32
        }
    }
}

impl NaturallyOrderedSemiring for TropicalWeight {}

impl DivisibleSemiring for TropicalWeight {
    fn divide(&self, other: &Self) -> Option<Self> {
        if <Self as num_traits::Zero>::is_zero(other) {
            None
        } else if <Self as num_traits::Zero>::is_zero(self) {
            Some(Self::zero())
        } else {
            Some(Self(self.0 - other.0))
        }
    }
}

impl FromStr for TropicalWeight {
    type Err = std::num::ParseFloatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "∞" || s == "inf" || s == "infinity" {
            Ok(Self::INFINITY)
        } else {
            s.parse::<f32>().map(Self::new)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{One, Zero};

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

        assert_eq!(format!("{w}"), "5");
        assert_eq!(format!("{zero}"), "∞");
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
        let w1 = TropicalWeight::new(5.000_001);
        let w2 = TropicalWeight::new(5.0);

        assert!(w1.approx_eq(&w2, 0.001));
        assert!(!w1.approx_eq(&w2, 0.000_000_1));
    }

    #[test]
    fn test_tropical_from_str() {
        assert_eq!(
            TropicalWeight::from_str("5.0").unwrap(),
            TropicalWeight::new(5.0)
        );
        assert_eq!(
            TropicalWeight::from_str("∞").unwrap(),
            TropicalWeight::INFINITY
        );
        assert_eq!(
            TropicalWeight::from_str("inf").unwrap(),
            TropicalWeight::INFINITY
        );
        assert_eq!(
            TropicalWeight::from_str("infinity").unwrap(),
            TropicalWeight::INFINITY
        );
    }

    #[test]
    fn test_tropical_operator_overloads() {
        let w1 = TropicalWeight::new(3.0);
        let w2 = TropicalWeight::new(5.0);

        // Test + operator (min)
        assert_eq!(w1 + w2, TropicalWeight::new(3.0));

        // Test * operator (addition)
        assert_eq!(w1 * w2, TropicalWeight::new(8.0));
    }

    #[test]
    fn test_tropical_identity_laws() {
        let w = TropicalWeight::new(5.0);
        let zero = TropicalWeight::zero();
        let one = TropicalWeight::one();

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
    fn test_tropical_semiring_axioms() {
        let a = TropicalWeight::new(2.0);
        let b = TropicalWeight::new(3.0);
        let c = TropicalWeight::new(4.0);

        // Associativity of addition
        assert_eq!((a + b) + c, a + (b + c));

        // Associativity of multiplication
        assert_eq!((a * b) * c, a * (b * c));

        // Commutativity of addition
        assert_eq!(a + b, b + a);

        // Commutativity of multiplication
        assert_eq!(a * b, b * a);

        // Distributivity
        assert_eq!((a + b) * c, (a * c) + (b * c));
    }

    // Property-based tests
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn test_tropical_associativity_property(a in -100.0..100.0f32, b in -100.0..100.0f32, c in -100.0..100.0f32) {
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
            fn test_tropical_identity_property(a in -100.0..100.0f32) {
                let w = TropicalWeight::new(a);

                // w + zero = w
                prop_assert!(w.plus(&TropicalWeight::zero()).approx_eq(&w, 1e-4));

                // w * one = w
                prop_assert!(w.times(&TropicalWeight::one()).approx_eq(&w, 1e-4));
            }
        }
    }
}
