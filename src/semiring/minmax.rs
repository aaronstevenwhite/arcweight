//! Min/Max semiring implementations

use super::traits::*;
use ordered_float::OrderedFloat;
use core::fmt;
use core::ops::{Add, Mul};
use num_traits::{One, Zero};

/// Min weight (min, max) semiring
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MinWeight(OrderedFloat<f32>);

impl MinWeight {
    /// Positive infinity (zero element)
    pub const INFINITY: Self = Self(OrderedFloat(f32::INFINITY));
    
    /// Negative infinity (one element)
    pub const NEG_INFINITY: Self = Self(OrderedFloat(f32::NEG_INFINITY));
    
    /// Create a new min weight
    pub fn new(value: f32) -> Self {
        Self(OrderedFloat(value))
    }
}

impl fmt::Display for MinWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_infinite() {
            if self.0.is_sign_positive() {
                write!(f, "∞")
            } else {
                write!(f, "-∞")
            }
        } else {
            write!(f, "{}", self.0)
        }
    }
}

impl Zero for MinWeight {
    fn zero() -> Self {
        Self::INFINITY
    }
    
    fn is_zero(&self) -> bool {
        self.0.is_infinite() && self.0.is_sign_positive()
    }
}

impl One for MinWeight {
    fn one() -> Self {
        Self::NEG_INFINITY
    }
}

impl Add for MinWeight {
    type Output = Self;
    
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.min(rhs.0))
    }
}

impl Mul for MinWeight {
    type Output = Self;
    
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.max(rhs.0))
    }
}

impl Semiring for MinWeight {
    type Value = f32;
    
    fn new(value: Self::Value) -> Self {
        Self::new(value)
    }
    
    fn value(&self) -> &Self::Value {
        &self.0
    }
    
    fn properties() -> SemiringProperties {
        SemiringProperties {
            left_semiring: false,
            right_semiring: false,
            commutative: true,
            idempotent: true,
            path: false,
        }
    }
}

impl NaturallyOrderedSemiring for MinWeight {}

/// Max weight (max, min) semiring
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MaxWeight(OrderedFloat<f32>);

impl MaxWeight {
    /// Negative infinity (zero element)
    pub const NEG_INFINITY: Self = Self(OrderedFloat(f32::NEG_INFINITY));
    
    /// Positive infinity (one element)
    pub const INFINITY: Self = Self(OrderedFloat(f32::INFINITY));
    
    /// Create a new max weight
    pub fn new(value: f32) -> Self {
        Self(OrderedFloat(value))
    }
}

impl fmt::Display for MaxWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_infinite() {
            if self.0.is_sign_positive() {
                write!(f, "∞")
            } else {
                write!(f, "-∞")
            }
        } else {
            write!(f, "{}", self.0)
        }
    }
}

impl Zero for MaxWeight {
    fn zero() -> Self {
        Self::NEG_INFINITY
    }
    
    fn is_zero(&self) -> bool {
        self.0.is_infinite() && self.0.is_sign_negative()
    }
}

impl One for MaxWeight {
    fn one() -> Self {
        Self::INFINITY
    }
}

impl Add for MaxWeight {
    type Output = Self;
    
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.max(rhs.0))
    }
}

impl Mul for MaxWeight {
    type Output = Self;
    
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.min(rhs.0))
    }
}

impl Semiring for MaxWeight {
    type Value = f32;
    
    fn new(value: Self::Value) -> Self {
        Self::new(value)
    }
    
    fn value(&self) -> &Self::Value {
        &self.0
    }
    
    fn properties() -> SemiringProperties {
        SemiringProperties {
            left_semiring: false,
            right_semiring: false,
            commutative: true,
            idempotent: true,
            path: false,
        }
    }
}

impl NaturallyOrderedSemiring for MaxWeight {}