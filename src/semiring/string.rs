//! String semiring implementation

use super::traits::*;
use core::fmt;
use core::ops::{Add, Mul};
use num_traits::{One, Zero};

/// String weight (longest common prefix, concatenation) semiring
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StringWeight(Vec<u8>);

impl StringWeight {
    /// Empty string (one element)
    pub const EMPTY: Self = Self(Vec::new());

    /// Create from bytes
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// Create from string slice
    pub fn from_string(s: &str) -> Self {
        Self(s.as_bytes().to_vec())
    }

    /// Convert to string
    /// 
    /// # Errors
    /// 
    /// Returns an error if:
    /// - The internal byte sequence is not valid UTF-8
    pub fn to_string(&self) -> Result<String, core::str::Utf8Error> {
        core::str::from_utf8(&self.0).map(|s| s.to_string())
    }

    /// Get bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Longest common prefix
    fn lcp(&self, other: &Self) -> Self {
        let len = self.0.len().min(other.0.len());
        let mut i = 0;
        while i < len && self.0[i] == other.0[i] {
            i += 1;
        }
        Self(self.0[..i].to_vec())
    }
}

impl fmt::Display for StringWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_string() {
            Ok(s) => write!(f, "\"{}\"", s),
            Err(_) => write!(f, "{:?}", self.0),
        }
    }
}

impl Zero for StringWeight {
    fn zero() -> Self {
        // special marker for zero (infinity)
        Self(vec![0xFF])
    }

    fn is_zero(&self) -> bool {
        self.0 == vec![0xFF]
    }
}

impl One for StringWeight {
    fn one() -> Self {
        Self::EMPTY
    }
}

impl Add for StringWeight {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if <Self as num_traits::Zero>::is_zero(&self) {
            rhs
        } else if <Self as num_traits::Zero>::is_zero(&rhs) {
            self
        } else {
            self.lcp(&rhs)
        }
    }
}

impl Mul for StringWeight {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if <Self as num_traits::Zero>::is_zero(&self) || <Self as num_traits::Zero>::is_zero(&rhs) {
            Self::zero()
        } else {
            let mut result = self.0;
            result.extend_from_slice(&rhs.0);
            Self(result)
        }
    }
}

impl Semiring for StringWeight {
    type Value = Vec<u8>;

    fn new(value: Self::Value) -> Self {
        Self(value)
    }

    fn value(&self) -> &Self::Value {
        &self.0
    }

    fn properties() -> SemiringProperties {
        SemiringProperties {
            left_semiring: true,
            right_semiring: true,
            commutative: false,
            idempotent: true,
            path: false,
        }
    }
}
