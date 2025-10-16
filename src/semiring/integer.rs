//! Integer semiring for exact arithmetic
//!
//! The integer semiring provides exact integer arithmetic without the precision
//! issues associated with floating-point operations. This makes it particularly
//! useful for counting, combinatorial, and exact arithmetic applications.
//!
//! # Mathematical Semantics
//!
//! - **Value Range:** Signed 64-bit integers (-2^63 to 2^63-1)
//! - **Addition (⊕):** Standard addition `a + b`
//! - **Multiplication (⊗):** Standard multiplication `a × b`
//! - **Zero (0̄):** 0 (additive identity)
//! - **One (1̄):** 1 (multiplicative identity)
//!
//! # Use Cases
//!
//! ## Path Counting
//! ```rust
//! use arcweight::prelude::*;
//!
//! // Count number of paths through an FST
//! let mut fst = VectorFst::<IntegerWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! let s2 = fst.add_state();
//!
//! fst.set_start(s0);
//! fst.set_final(s2, IntegerWeight::one());
//!
//! // Two parallel paths from s0 to s1
//! fst.add_arc(s0, Arc::new(1, 1, IntegerWeight::one(), s1));
//! fst.add_arc(s0, Arc::new(2, 2, IntegerWeight::one(), s1));
//!
//! // One path from s1 to s2
//! fst.add_arc(s1, Arc::new(3, 3, IntegerWeight::one(), s2));
//!
//! // shortest_distance counts total paths (2 paths total)
//! # #[cfg(feature = "algorithms")]
//! # {
//! let distances = shortest_distance(&fst).unwrap();
//! assert_eq!(distances[s2 as usize], IntegerWeight::new(2));
//! # }
//! ```
//!
//! ## Exact Arithmetic
//! ```rust
//! use arcweight::prelude::*;
//!
//! let w1 = IntegerWeight::new(100);
//! let w2 = IntegerWeight::new(50);
//!
//! // No floating-point precision loss
//! assert_eq!(w1.plus(&w2), IntegerWeight::new(150));
//! assert_eq!(w1.times(&w2), IntegerWeight::new(5000));
//! ```
//!
//! ## Combinatorial Applications
//! ```rust
//! use arcweight::prelude::*;
//!
//! // Count combinations in an FST
//! let combinations = IntegerWeight::new(6);
//! let choices = IntegerWeight::new(4);
//!
//! // Total possibilities
//! let total = combinations.times(&choices);
//! assert_eq!(total, IntegerWeight::new(24));
//! ```
//!
//! # Algebraic Properties
//!
//! The integer semiring forms a commutative semiring under standard arithmetic:
//! - **Commutative:** Both addition and multiplication are commutative
//! - **Associative:** Both operations are associative
//! - **Distributive:** Multiplication distributes over addition
//! - **Not Idempotent:** `a ⊕ a ≠ a` (except for 0)
//! - **Path Property:** Does NOT have path property (combines values)
//!
//! # Overflow Behavior
//!
//! Operations use wrapping semantics to avoid panics on overflow:
//! - Addition uses `wrapping_add`
//! - Multiplication uses `wrapping_mul`
//!
//! For applications requiring overflow detection, consider using external
//! validation or a custom semiring with checked arithmetic.
//!
//! ```rust
//! use arcweight::prelude::*;
//!
//! // Overflow wraps around (does not panic)
//! let large = IntegerWeight::new(i64::MAX);
//! let result = large.plus(&IntegerWeight::one());
//! assert_eq!(result, IntegerWeight::new(i64::MIN)); // Wraps to MIN
//! ```
//!
//! # Performance Characteristics
//!
//! - **Arithmetic:** O(1) integer operations
//! - **Memory:** 8 bytes per weight (i64)
//! - **Comparison:** Fast integer comparison
//! - **Copy:** Cheap to copy (implements Copy trait)
//!
//! # Integration with FST Algorithms
//!
//! IntegerWeight works with all FST algorithms:
//! - **Composition:** Multiplies weights along paths
//! - **Union:** Adds weights from parallel paths
//! - **Shortest distance:** Computes total path weights
//! - **Determinization:** Combines weights from equivalent states
//!
//! # See Also
//!
//! - [`TropicalWeight`](crate::semiring::TropicalWeight) for optimization problems
//! - [`BooleanWeight`](crate::semiring::BooleanWeight) for recognition
//! - [`ProbabilityWeight`](crate::semiring::ProbabilityWeight) for probabilistic FSTs

use super::traits::*;
use core::fmt;
use core::ops::{Add, Mul};
use core::str::FromStr;
use num_traits::{One, Zero};

/// Integer weight for exact arithmetic operations
///
/// A semiring weight based on standard integer arithmetic, useful for
/// counting, combinatorial applications, and situations requiring exact
/// arithmetic without floating-point precision issues.
///
/// # Examples
///
/// ```
/// use arcweight::prelude::*;
///
/// let w1 = IntegerWeight::new(5);
/// let w2 = IntegerWeight::new(3);
///
/// // Addition
/// assert_eq!(w1.plus(&w2), IntegerWeight::new(8));
///
/// // Multiplication
/// assert_eq!(w1.times(&w2), IntegerWeight::new(15));
///
/// // Operator overloads
/// assert_eq!(w1 + w2, IntegerWeight::new(8));
/// assert_eq!(w1 * w2, IntegerWeight::new(15));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IntegerWeight(i64);

impl IntegerWeight {
    /// Creates a new integer weight with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let weight = IntegerWeight::new(42);
    /// assert_eq!(*weight.value(), 42);
    /// ```
    pub const fn new(value: i64) -> Self {
        Self(value)
    }

    /// Returns the underlying integer value.
    ///
    /// # Examples
    ///
    /// ```
    /// use arcweight::prelude::*;
    ///
    /// let weight = IntegerWeight::new(100);
    /// assert_eq!(*weight.value(), 100);
    /// ```
    pub const fn value(&self) -> &i64 {
        &self.0
    }
}

impl fmt::Display for IntegerWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Zero for IntegerWeight {
    fn zero() -> Self {
        Self::new(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for IntegerWeight {
    fn one() -> Self {
        Self::new(1)
    }
}

impl Add for IntegerWeight {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // Use wrapping_add to avoid overflow panics
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl Mul for IntegerWeight {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // Use wrapping_mul to avoid overflow panics
        Self(self.0.wrapping_mul(rhs.0))
    }
}

impl Semiring for IntegerWeight {
    type Value = i64;

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
            path: true,
        }
    }
}

impl FromStr for IntegerWeight {
    type Err = std::num::ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<i64>().map(Self::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_weight_creation() {
        let w = IntegerWeight::new(42);
        assert_eq!(*w.value(), 42);
    }

    #[test]
    fn test_integer_weight_value() {
        let w = IntegerWeight::new(100);
        assert_eq!(*w.value(), 100);

        let w_negative = IntegerWeight::new(-50);
        assert_eq!(*w_negative.value(), -50);
    }

    #[test]
    fn test_integer_weight_zero() {
        let zero = IntegerWeight::zero();
        assert_eq!(*zero.value(), 0);
        assert!(Zero::is_zero(&zero));
    }

    #[test]
    fn test_integer_weight_one() {
        let one = IntegerWeight::one();
        assert_eq!(*one.value(), 1);
        assert!(!Zero::is_zero(&one));
    }

    #[test]
    fn test_integer_addition() {
        let w1 = IntegerWeight::new(5);
        let w2 = IntegerWeight::new(3);
        let result = w1.plus(&w2);
        assert_eq!(*result.value(), 8);
    }

    #[test]
    fn test_integer_multiplication() {
        let w1 = IntegerWeight::new(5);
        let w2 = IntegerWeight::new(3);
        let result = w1.times(&w2);
        assert_eq!(*result.value(), 15);
    }

    #[test]
    fn test_additive_identity() {
        let w = IntegerWeight::new(42);
        let zero = IntegerWeight::zero();

        assert_eq!(w.plus(&zero), w);
        assert_eq!(zero.plus(&w), w);
    }

    #[test]
    fn test_multiplicative_identity() {
        let w = IntegerWeight::new(42);
        let one = IntegerWeight::one();

        assert_eq!(w.times(&one), w);
        assert_eq!(one.times(&w), w);
    }

    #[test]
    fn test_annihilation() {
        let w = IntegerWeight::new(42);
        let zero = IntegerWeight::zero();

        let result1 = w.times(&zero);
        let result2 = zero.times(&w);

        assert!(Zero::is_zero(&result1));
        assert!(Zero::is_zero(&result2));
    }

    #[test]
    fn test_additive_associativity() {
        let a = IntegerWeight::new(2);
        let b = IntegerWeight::new(3);
        let c = IntegerWeight::new(5);

        assert_eq!((a + b) + c, a + (b + c));
    }

    #[test]
    fn test_additive_commutativity() {
        let a = IntegerWeight::new(7);
        let b = IntegerWeight::new(11);

        assert_eq!(a + b, b + a);
    }

    #[test]
    fn test_multiplicative_associativity() {
        let a = IntegerWeight::new(2);
        let b = IntegerWeight::new(3);
        let c = IntegerWeight::new(5);

        assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn test_multiplicative_commutativity() {
        let a = IntegerWeight::new(7);
        let b = IntegerWeight::new(11);

        assert_eq!(a * b, b * a);
    }

    #[test]
    fn test_left_distributivity() {
        let a = IntegerWeight::new(2);
        let b = IntegerWeight::new(3);
        let c = IntegerWeight::new(5);

        // a * (b + c) = (a * b) + (a * c)
        assert_eq!(a * (b + c), (a * b) + (a * c));
    }

    #[test]
    fn test_right_distributivity() {
        let a = IntegerWeight::new(2);
        let b = IntegerWeight::new(3);
        let c = IntegerWeight::new(5);

        // (a + b) * c = (a * c) + (b * c)
        assert_eq!((a + b) * c, (a * c) + (b * c));
    }

    #[test]
    fn test_is_zero() {
        let zero = IntegerWeight::new(0);
        let nonzero = IntegerWeight::new(5);

        assert!(Semiring::is_zero(&zero));
        assert!(!Semiring::is_zero(&nonzero));
    }

    #[test]
    fn test_is_one() {
        let one = IntegerWeight::new(1);
        let not_one = IntegerWeight::new(5);

        assert!(Semiring::is_one(&one));
        assert!(!Semiring::is_one(&not_one));
    }

    #[test]
    fn test_approx_eq() {
        let w1 = IntegerWeight::new(42);
        let w2 = IntegerWeight::new(42);
        let w3 = IntegerWeight::new(43);

        assert!(w1.approx_eq(&w2, 0.0));
        assert!(!w1.approx_eq(&w3, 0.0));
    }

    #[test]
    fn test_semiring_properties() {
        let props = IntegerWeight::properties();
        assert!(props.left_semiring);
        assert!(props.right_semiring);
        assert!(props.commutative);
        assert!(!props.idempotent);
        assert!(props.path);
    }

    #[test]
    fn test_add_operator() {
        let w1 = IntegerWeight::new(10);
        let w2 = IntegerWeight::new(20);

        assert_eq!(w1 + w2, IntegerWeight::new(30));
    }

    #[test]
    fn test_mul_operator() {
        let w1 = IntegerWeight::new(10);
        let w2 = IntegerWeight::new(20);

        assert_eq!(w1 * w2, IntegerWeight::new(200));
    }

    #[test]
    fn test_display() {
        let w = IntegerWeight::new(42);
        assert_eq!(format!("{}", w), "42");

        let w_negative = IntegerWeight::new(-100);
        assert_eq!(format!("{}", w_negative), "-100");
    }

    #[test]
    fn test_from_str() {
        assert_eq!(
            IntegerWeight::from_str("42").unwrap(),
            IntegerWeight::new(42)
        );
        assert_eq!(
            IntegerWeight::from_str("-100").unwrap(),
            IntegerWeight::new(-100)
        );
        assert_eq!(IntegerWeight::from_str("0").unwrap(), IntegerWeight::zero());
    }

    #[test]
    fn test_from_str_invalid() {
        assert!(IntegerWeight::from_str("not_a_number").is_err());
        assert!(IntegerWeight::from_str("12.34").is_err());
        assert!(IntegerWeight::from_str("").is_err());
    }

    #[test]
    fn test_negative_values() {
        let w1 = IntegerWeight::new(-5);
        let w2 = IntegerWeight::new(3);

        assert_eq!(w1 + w2, IntegerWeight::new(-2));
        assert_eq!(w1 * w2, IntegerWeight::new(-15));
    }

    #[test]
    fn test_negative_multiplication() {
        let w1 = IntegerWeight::new(-5);
        let w2 = IntegerWeight::new(-3);

        assert_eq!(w1 * w2, IntegerWeight::new(15));
    }

    #[test]
    fn test_zero_operations() {
        let zero = IntegerWeight::zero();
        let w = IntegerWeight::new(42);

        // Zero addition
        assert_eq!(zero + w, w);
        assert_eq!(w + zero, w);

        // Zero multiplication
        assert_eq!(zero * w, zero);
        assert_eq!(w * zero, zero);
    }

    #[test]
    fn test_overflow_addition() {
        let max = IntegerWeight::new(i64::MAX);
        let one = IntegerWeight::new(1);

        // Should wrap to MIN without panicking
        let result = max + one;
        assert_eq!(result, IntegerWeight::new(i64::MIN));
    }

    #[test]
    fn test_overflow_multiplication() {
        let large = IntegerWeight::new(i64::MAX);
        let two = IntegerWeight::new(2);

        // Should wrap without panicking
        let result = large * two;
        // Verify it doesn't panic (exact value depends on wrapping behavior)
        assert!(result.value() != large.value());
    }

    #[test]
    fn test_large_values() {
        let w1 = IntegerWeight::new(1_000_000_000);
        let w2 = IntegerWeight::new(2_000_000_000);

        assert_eq!(w1 + w2, IntegerWeight::new(3_000_000_000));
    }

    #[test]
    fn test_clone() {
        let w1 = IntegerWeight::new(42);
        #[allow(clippy::clone_on_copy)]
        let w2 = w1.clone();

        assert_eq!(w1, w2);
        assert_eq!(*w1.value(), *w2.value());
    }

    #[test]
    fn test_copy() {
        let w1 = IntegerWeight::new(42);
        let w2 = w1; // Copy, not move

        assert_eq!(w1, w2);
        assert_eq!(*w1.value(), 42); // w1 still valid
    }

    #[test]
    fn test_eq() {
        let w1 = IntegerWeight::new(42);
        let w2 = IntegerWeight::new(42);
        let w3 = IntegerWeight::new(43);

        assert_eq!(w1, w2);
        assert_ne!(w1, w3);
    }

    #[test]
    fn test_ord() {
        let w1 = IntegerWeight::new(10);
        let w2 = IntegerWeight::new(20);
        let w3 = IntegerWeight::new(10);

        assert!(w1 < w2);
        assert!(w2 > w1);
        assert!(w1 <= w3);
        assert!(w1 >= w3);
    }

    #[test]
    fn test_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(IntegerWeight::new(1));
        set.insert(IntegerWeight::new(2));
        set.insert(IntegerWeight::new(1)); // Duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&IntegerWeight::new(1)));
        assert!(set.contains(&IntegerWeight::new(2)));
    }

    #[test]
    fn test_default() {
        let w = IntegerWeight::default();
        assert_eq!(w, IntegerWeight::new(0));
        assert!(Zero::is_zero(&w));
    }

    #[test]
    fn test_not_idempotent() {
        let w = IntegerWeight::new(5);
        let result = w + w;

        // Integer addition is not idempotent (except for 0)
        assert_eq!(result, IntegerWeight::new(10));
        assert_ne!(result, w);
    }

    #[test]
    fn test_multiple_operations() {
        let w1 = IntegerWeight::new(2);
        let w2 = IntegerWeight::new(3);
        let w3 = IntegerWeight::new(4);

        // (2 + 3) * 4 = 20
        let result = (w1 + w2) * w3;
        assert_eq!(result, IntegerWeight::new(20));

        // 2 * (3 + 4) = 14
        let result2 = w1 * (w2 + w3);
        assert_eq!(result2, IntegerWeight::new(14));
    }

    #[test]
    fn test_plus_assign() {
        let mut w = IntegerWeight::new(10);
        w.plus_assign(&IntegerWeight::new(5));
        assert_eq!(w, IntegerWeight::new(15));
    }

    #[test]
    fn test_times_assign() {
        let mut w = IntegerWeight::new(10);
        w.times_assign(&IntegerWeight::new(5));
        assert_eq!(w, IntegerWeight::new(50));
    }
}
