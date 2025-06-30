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
