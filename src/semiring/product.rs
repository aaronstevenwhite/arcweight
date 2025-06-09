//! Product semiring implementation

use super::traits::*;
use core::fmt;
use core::ops::{Add, Mul};
use num_traits::{One, Zero};

/// Product of two semirings
#[derive(Clone, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProductWeight<W1: Semiring, W2: Semiring> {
    /// First component
    pub w1: W1,
    /// Second component
    pub w2: W2,
}

impl<W1: Semiring, W2: Semiring> ProductWeight<W1, W2> {
    /// Create a new product weight
    pub fn new(w1: W1, w2: W2) -> Self {
        Self { w1, w2 }
    }
}

impl<W1: Semiring, W2: Semiring> fmt::Display for ProductWeight<W1, W2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.w1, self.w2)
    }
}

impl<W1: Semiring, W2: Semiring> Zero for ProductWeight<W1, W2> {
    fn zero() -> Self {
        Self::new(W1::zero(), W2::zero())
    }
    
    fn is_zero(&self) -> bool {
        Semiring::is_zero(&self.w1) && Semiring::is_zero(&self.w2)
    }
}

impl<W1: Semiring, W2: Semiring> One for ProductWeight<W1, W2> {
    fn one() -> Self {
        Self::new(W1::one(), W2::one())
    }
}

impl<W1: Semiring, W2: Semiring> Add for ProductWeight<W1, W2> {
    type Output = Self;
    
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.w1 + rhs.w1,
            self.w2 + rhs.w2,
        )
    }
}

impl<W1: Semiring, W2: Semiring> Mul for ProductWeight<W1, W2> {
    type Output = Self;
    
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.w1 * rhs.w1,
            self.w2 * rhs.w2,
        )
    }
}

impl<W1: Semiring, W2: Semiring> Semiring for ProductWeight<W1, W2> {
    type Value = (W1::Value, W2::Value);
    
    fn new(value: Self::Value) -> Self {
        Self::new(W1::new(value.0), W2::new(value.1))
    }
    
    fn value(&self) -> &Self::Value {
        // this requires unsafe or a different approach
        unimplemented!("ProductWeight::value requires redesign")
    }
    
    fn properties() -> SemiringProperties {
        let p1 = W1::properties();
        let p2 = W2::properties();
        
        SemiringProperties {
            left_semiring: p1.left_semiring && p2.left_semiring,
            right_semiring: p1.right_semiring && p2.right_semiring,
            commutative: p1.commutative && p2.commutative,
            idempotent: p1.idempotent && p2.idempotent,
            path: p1.path && p2.path,
        }
    }
    
    fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
        self.w1.approx_eq(&other.w1, epsilon) && 
        self.w2.approx_eq(&other.w2, epsilon)
    }
}

impl<W1, W2> DivisibleSemiring for ProductWeight<W1, W2>
where
    W1: DivisibleSemiring,
    W2: DivisibleSemiring,
{
    fn divide(&self, other: &Self) -> Option<Self> {
        match (self.w1.divide(&other.w1), self.w2.divide(&other.w2)) {
            (Some(w1), Some(w2)) => Some(Self::new(w1, w2)),
            _ => None,
        }
    }
}

impl<W1, W2> StarSemiring for ProductWeight<W1, W2>
where
    W1: StarSemiring,
    W2: StarSemiring,
{
    fn star(&self) -> Self {
        Self::new(self.w1.star(), self.w2.star())
    }
}