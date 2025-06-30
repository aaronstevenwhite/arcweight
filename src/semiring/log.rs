//! Log semiring implementation

use super::traits::*;
use core::fmt;
use core::ops::{Add, Mul};
use num_traits::{One, Zero};
use ordered_float::OrderedFloat;

/// Log weight for numerically stable probability computation
///
/// The log semiring addresses numerical stability issues inherent in probability
/// computation by working in the logarithmic domain. This enables handling of very
/// small probabilities (e.g., 10^-100) that would underflow in linear probability
/// space, while maintaining mathematically equivalent probabilistic semantics.
///
/// # Mathematical Semantics
///
/// - **Value Range:** Real numbers ℝ plus positive infinity (+∞)
/// - **Addition (⊕):** `-log(e^(-a) + e^(-b))` (log-sum-exp) - combines probabilities
/// - **Multiplication (⊗):** `a + b` (addition in log space) - multiplies probabilities
/// - **Zero (0̄):** `+∞` - represents impossible event (probability 0)
/// - **One (1̄):** `0.0` - represents certain event (probability 1)
///
/// # Relationship to Probability Semiring
///
/// The log semiring is related to the probability semiring through logarithmic transformation:
/// - If `p, q` are probabilities, then `-log p ⊕ -log q = -log(p + q)`
/// - If `p, q` are probabilities, then `-log p ⊗ -log q = -log(p × q)`
///
/// This preserves probabilistic semantics while providing numerical stability.
///
/// # Use Cases
///
/// ## Large-Vocabulary Speech Recognition
/// ```rust
/// use arcweight::prelude::*;
///
/// // Acoustic model probabilities (very small values)
/// let acoustic_prob = LogWeight::from_probability(1e-50);  // Converts to log space
/// let language_prob = LogWeight::from_probability(0.001);
///
/// // Combine probabilities safely
/// let combined = acoustic_prob.times(&language_prob);  // Multiplication in log space
///
/// // Convert back to probability if needed
/// let final_prob = combined.to_probability();
/// println!("Final probability: {:.2e}", final_prob);  // ~1e-53
/// ```
///
/// ## Machine Translation with Large Models
/// ```rust
/// use arcweight::prelude::*;
///
/// // Translation model scores (often very small probabilities)
/// let phrase_prob = LogWeight::from_probability(1e-20);
/// let alignment_prob = LogWeight::from_probability(1e-15);
/// let reordering_prob = LogWeight::from_probability(0.1);
///
/// // Combine all model scores
/// let translation_score = phrase_prob
///     .times(&alignment_prob)
///     .times(&reordering_prob);
///
/// // Alternative translations (add probabilities)
/// let alternative1 = LogWeight::from_probability(1e-35);
/// let alternative2 = LogWeight::from_probability(2e-35);
/// let combined_alternatives = alternative1.plus(&alternative2);  // LogSumExp
/// ```
///
/// ## Neural Language Model Integration
/// ```rust
/// use arcweight::prelude::*;
///
/// // Softmax probabilities from neural networks
/// let word_probs = vec![
///     LogWeight::from_probability(0.4),   // Most likely word
///     LogWeight::from_probability(0.3),   // Second choice
///     LogWeight::from_probability(0.2),   // Third choice
///     LogWeight::from_probability(0.1),   // Least likely
/// ];
///
/// // Compute probability of any of these words
/// let any_word_prob = word_probs.into_iter()
///     .fold(LogWeight::zero(), |acc, prob| acc.plus(&prob));
///
/// // Should be close to 1.0 (sum of probabilities)
/// assert!((any_word_prob.to_probability() - 1.0).abs() < 1e-10);
/// ```
///
/// ## Sequence Analysis in Bioinformatics
/// ```rust
/// use arcweight::prelude::*;
///
/// // DNA sequence alignment with very long sequences
/// let base_prob = LogWeight::from_probability(0.25);  // Each base equally likely
/// let sequence_length = 1000;
///
/// // Probability of specific sequence (would underflow in linear space)
/// let mut sequence_prob = LogWeight::one();
/// for _ in 0..sequence_length {
///     sequence_prob = sequence_prob.times(&base_prob);
/// }
///
/// // Convert to scientific notation for display
/// println!("Sequence probability: {:.2e}", sequence_prob.to_probability());
/// ```
///
/// # Working with FSTs
///
/// ```rust
/// use arcweight::prelude::*;
///
/// let log_p1 = LogWeight::from_probability(0.5);   // Convert from probability
/// let log_p2 = LogWeight::from_probability(0.25);  // Convert from probability
///
/// // Addition performs log-sum-exp (combines probabilities)
/// let sum = log_p1 + log_p2;  // -log(0.5 + 0.25) = -log(0.75)
/// assert!((sum.to_probability() - 0.75).abs() < 1e-6);
///
/// // Multiplication is addition in log space (multiplies probabilities)
/// let product = log_p1 * log_p2;  // -log(0.5 × 0.25) = -log(0.125)
/// assert!((product.to_probability() - 0.125).abs() < 1e-10);
///
/// // Identity elements
/// assert_eq!(LogWeight::zero(), LogWeight::INFINITY);  // Impossible event
/// assert_eq!(LogWeight::one(), LogWeight::new(0.0));   // Certain event
/// ```
///
/// # Numerical Stability Implementation
///
/// The log semiring implements numerically stable log-sum-exp operation:
/// ```text
/// -log(e^(-a) + e^(-b)) = -max(a,b) - log(1 + e^(-|a-b|))
/// ```
///
/// This formulation prevents overflow/underflow by:
/// - Working with the larger magnitude value first
/// - Computing the difference in a stable manner
/// - Using the identity: `log(1 + x) ≈ x` for small `x`
///
/// # Performance Characteristics
///
/// - **Arithmetic:** Addition is expensive (log-sum-exp), multiplication is O(1)
/// - **Memory:** 8 bytes per weight (single f64)
/// - **Precision:** Double precision for high-accuracy probability computation
/// - **Conversion:** Efficient probability ↔ log conversions available
/// - **Range:** Handles probabilities from ~10^-308 to 1.0
///
/// # Conversion Utilities
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Convert from probability to log weight
/// let prob = 0.001;
/// let log_weight = LogWeight::from_probability(prob);
/// assert_eq!(log_weight.value(), &(-prob.ln()));
///
/// // Convert back to probability
/// let recovered_prob = log_weight.to_probability();
/// assert!((recovered_prob - prob).abs() < 1e-15);
///
/// // Handle edge cases
/// let zero_prob = LogWeight::from_probability(0.0);
/// assert!(<LogWeight as num_traits::Zero>::is_zero(&zero_prob));
/// assert_eq!(zero_prob.to_probability(), 0.0);
/// ```
///
/// # Advanced Usage
///
/// ## Normalization in Log Space
/// ```rust
/// use arcweight::prelude::*;
///
/// // Normalize a probability distribution in log space
/// let log_probs = vec![
///     LogWeight::new(1.0),  // Unnormalized log probabilities
///     LogWeight::new(2.0),
///     LogWeight::new(0.5),
/// ];
///
/// // Compute log partition function (log of sum of probabilities)
/// let log_z = log_probs.iter()
///     .fold(LogWeight::zero(), |acc, &p| acc.plus(&p));
///
/// // Normalize each probability
/// let normalized: Vec<_> = log_probs.iter()
///     .map(|&p| p.divide(&log_z).unwrap())
///     .collect();
///
/// // Verify normalization (sum should be 1.0)
/// let sum = normalized.iter()
///     .fold(LogWeight::zero(), |acc, &p| acc.plus(&p));
/// assert!((sum.to_probability() - 1.0).abs() < 1e-10);
/// ```
///
/// # Integration with FST Algorithms
///
/// Log weights work seamlessly with all FST algorithms while providing numerical stability:
/// - **Shortest Path:** Finds maximum probability paths
/// - **Forward-Backward:** Stable computation of path probabilities
/// - **Composition:** Combines probabilistic models
/// - **Determinization:** Maintains probability distributions
///
/// # See Also
///
/// - [Core Concepts - Log Semiring](../../docs/core-concepts/semirings.md#log-semiring) for mathematical background
/// - [`ProbabilityWeight`](crate::semiring::ProbabilityWeight) for simple probability computation
/// - [`TropicalWeight`](crate::semiring::TropicalWeight) for optimization problems
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
        if <Self as num_traits::Zero>::is_zero(self) && <Self as num_traits::Zero>::is_zero(other) {
            true
        } else {
            (self.0 - other.0).abs() < epsilon
        }
    }
}

impl DivisibleSemiring for LogWeight {
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
