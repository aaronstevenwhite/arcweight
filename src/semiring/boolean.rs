//! Boolean semiring implementation

use super::traits::*;
use core::fmt;
use core::ops::{Add, Mul};
use core::str::FromStr;
use num_traits::{One, Zero};

/// Boolean weight (∨, ∧) semiring
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BooleanWeight(bool);

impl BooleanWeight {
    /// Create a new boolean weight
    pub const fn new(value: bool) -> Self {
        Self(value)
    }
}

impl fmt::Display for BooleanWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Zero for BooleanWeight {
    fn zero() -> Self {
        Self::new(false)
    }

    fn is_zero(&self) -> bool {
        !self.0
    }
}

impl One for BooleanWeight {
    fn one() -> Self {
        Self::new(true)
    }
}

impl Add for BooleanWeight {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 || rhs.0)
    }
}

impl Mul for BooleanWeight {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 && rhs.0)
    }
}

impl Semiring for BooleanWeight {
    type Value = bool;

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
}

impl NaturallyOrderedSemiring for BooleanWeight {}

impl StarSemiring for BooleanWeight {
    fn star(&self) -> Self {
        Self::one()
    }
}

impl FromStr for BooleanWeight {
    type Err = std::str::ParseBoolError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<bool>().map(Self::new)
    }
}
