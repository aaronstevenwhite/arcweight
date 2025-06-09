//! Core semiring traits for weight types

use core::fmt::{Debug, Display};
use core::ops::{Add, Mul};
use num_traits::{One, Zero};

/// Core semiring trait defining algebraic operations for weights
pub trait Semiring: 
    Clone 
    + Debug 
    + Display 
    + PartialEq 
    + PartialOrd
    + Add<Output = Self>
    + Mul<Output = Self>
    + Zero
    + One
    + Send
    + Sync
    + 'static
{
    /// Type of the underlying value
    type Value: Clone + Debug + PartialEq + PartialOrd;
    
    /// Create a new weight from a value
    fn new(value: Self::Value) -> Self;
    
    /// Get the underlying value
    fn value(&self) -> &Self::Value;
    
    /// Semiring addition (⊕)
    fn plus(&self, other: &Self) -> Self {
        self.clone() + other.clone()
    }
    
    /// Semiring multiplication (⊗)
    fn times(&self, other: &Self) -> Self {
        self.clone() * other.clone()
    }
    
    /// In-place semiring addition
    fn plus_assign(&mut self, other: &Self) {
        *self = self.plus(other);
    }
    
    /// In-place semiring multiplication  
    fn times_assign(&mut self, other: &Self) {
        *self = self.times(other);
    }
    
    /// Check if weight is zero (additive identity)
    fn is_zero(&self) -> bool {
        self == &Self::zero()
    }
    
    /// Check if weight is one (multiplicative identity)
    fn is_one(&self) -> bool {
        self == &Self::one()
    }
    
    /// Semiring properties
    fn properties() -> SemiringProperties {
        SemiringProperties::default()
    }
    
    /// Approximate equality for floating-point weights
    fn approx_eq(&self, other: &Self, _epsilon: f64) -> bool {
        self == other
    }
}

/// Properties of a semiring
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemiringProperties {
    /// Left semiring (a ⊕ b) ⊗ c = (a ⊗ c) ⊕ (b ⊗ c)
    pub left_semiring: bool,
    /// Right semiring c ⊗ (a ⊕ b) = (c ⊗ a) ⊕ (c ⊗ b)
    pub right_semiring: bool,
    /// Commutative (a ⊗ b = b ⊗ a)
    pub commutative: bool,
    /// Idempotent (a ⊕ a = a)
    pub idempotent: bool,
    /// Path property
    pub path: bool,
}

impl Default for SemiringProperties {
    fn default() -> Self {
        Self {
            left_semiring: true,
            right_semiring: true,
            commutative: false,
            idempotent: false,
            path: false,
        }
    }
}

/// Trait for weights that can be divided
pub trait DivisibleSemiring: Semiring {
    /// Division operation
    fn divide(&self, other: &Self) -> Option<Self>;
}

/// Trait for weights that support star operation (Kleene closure)
pub trait StarSemiring: Semiring {
    /// Star operation: w* = 1 ⊕ w ⊕ w² ⊕ ...
    fn star(&self) -> Self;
}

/// Trait for weights with natural ordering
pub trait NaturallyOrderedSemiring: Semiring + Ord {}

/// Trait for weights that can be inverted
pub trait InvertibleSemiring: Semiring {
    /// Multiplicative inverse
    fn inverse(&self) -> Option<Self>;
}