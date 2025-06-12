//! Tropical semiring implementation

use super::traits::*;
use core::fmt;
use core::ops::{Add, Mul};
use core::str::FromStr;
use num_traits::{One, Zero};
use ordered_float::OrderedFloat;

/// Tropical weight (min, +) semiring
/// 
/// The tropical semiring uses minimum for addition and regular addition for multiplication.
/// The zero element is positive infinity and the one element is 0.0.
/// 
/// # Examples
/// 
/// ```
/// use arcweight::prelude::*;
/// 
/// let w1 = TropicalWeight::new(0.5);
/// let w2 = TropicalWeight::new(0.3);
/// 
/// // Addition is minimum
/// let sum = w1 + w2;
/// assert_eq!(sum, TropicalWeight::new(0.3));
/// 
/// // Multiplication is addition
/// let product = w1 * w2;
/// assert_eq!(product, TropicalWeight::new(0.8));
/// 
/// // Zero is infinity
/// assert!(<TropicalWeight as num_traits::Zero>::is_zero(&TropicalWeight::zero()));
/// assert_eq!(TropicalWeight::zero(), TropicalWeight::INFINITY);
/// 
/// // One is 0.0
/// assert_eq!(TropicalWeight::one(), TropicalWeight::new(0.0));
/// ```
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
