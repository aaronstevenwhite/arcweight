//! Probability semiring implementation

use super::traits::*;
use ordered_float::OrderedFloat;
use core::fmt;
use core::ops::{Add, Mul};
use num_traits::{One, Zero};

/// Probability weight (+, ×) semiring
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
        if <Self as num_traits::Zero>::is_zero(&other) {
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