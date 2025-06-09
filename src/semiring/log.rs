//! Log semiring implementation

use super::traits::*;
use ordered_float::OrderedFloat;
use core::fmt;
use core::ops::{Add, Mul};
use num_traits::{One, Zero};

/// Log weight (-log(p), +) semiring
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LogWeight(OrderedFloat<f64>);

impl LogWeight {
    /// Positive infinity (zero element)
    pub const INFINITY: Self = Self(OrderedFloat(f64::INFINITY));
    
    /// Create a new log weight
    pub fn new(value: f64) -> Self {
        Self(OrderedFloat(value))
    }
    
    /// Convert from probability
    pub fn from_probability(p: f64) -> Self {
        if p == 0.0 {
            Self::INFINITY
        } else {
            Self::new(-p.ln())
        }
    }
    
    /// Convert to probability
    pub fn to_probability(&self) -> f64 {
        if self.0.is_infinite() {
            0.0
        } else {
            (-*self.0).exp()
        }
    }
}

impl fmt::Display for LogWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_infinite() {
            write!(f, "∞")
        } else {
            write!(f, "{}", self.0)
        }
    }
}

impl Zero for LogWeight {
    fn zero() -> Self {
        Self::INFINITY
    }
    
    fn is_zero(&self) -> bool {
        self.0.is_infinite()
    }
}

impl One for LogWeight {
    fn one() -> Self {
        Self::new(0.0)
    }
}

impl Add for LogWeight {
    type Output = Self;
    
    fn add(self, rhs: Self) -> Self::Output {
        if <Self as num_traits::Zero>::is_zero(&self) {
            rhs
        } else if <Self as num_traits::Zero>::is_zero(&rhs) {
            self
        } else {
            let a = -*self.0;
            let b = -*rhs.0;
            Self::new(-(a.max(b) + (1.0 + (-(a - b).abs()).exp()).ln()))
        }
    }
}

impl Mul for LogWeight {
    type Output = Self;
    
    fn mul(self, rhs: Self) -> Self::Output {
        if <Self as num_traits::Zero>::is_zero(&self) || <Self as num_traits::Zero>::is_zero(&rhs) {
            Self::zero()
        } else {
            Self(self.0 + rhs.0)
        }
    }
}

impl Semiring for LogWeight {
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
        if <Self as num_traits::Zero>::is_zero(&self) && <Self as num_traits::Zero>::is_zero(&other) {
            true
        } else {
            (self.0 - other.0).abs() < epsilon
        }
    }
}

impl DivisibleSemiring for LogWeight {
    fn divide(&self, other: &Self) -> Option<Self> {
        if <Self as num_traits::Zero>::is_zero(&other) {
            None
        } else if <Self as num_traits::Zero>::is_zero(&self) {
            Some(Self::zero())
        } else {
            Some(Self(self.0 - other.0))
        }
    }
}