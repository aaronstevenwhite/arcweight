//! Product semiring implementation

use super::traits::*;
use core::fmt;
use core::ops::{Add, Mul};
use num_traits::{One, Zero};
use std::sync::OnceLock;

/// Product semiring combining two independent semiring computations
///
/// The Product semiring enables simultaneous computation over two independent
/// semiring structures, allowing algorithms to track multiple types of weights
/// or costs simultaneously. This is essential for multi-objective optimization,
/// parallel model combination, and composite metric tracking in FST operations.
///
/// # Mathematical Semantics
///
/// Given two semirings (S₁, ⊕₁, ⊗₁, 0̄₁, 1̄₁) and (S₂, ⊕₂, ⊗₂, 0̄₂, 1̄₂):
///
/// - **Value Range:** Cartesian product S₁ × S₂
/// - **Addition (⊕):** `(a₁, a₂) ⊕ (b₁, b₂) = (a₁ ⊕₁ b₁, a₂ ⊕₂ b₂)`
/// - **Multiplication (⊗):** `(a₁, a₂) ⊗ (b₁, b₂) = (a₁ ⊗₁ b₁, a₂ ⊗₂ b₂)`
/// - **Zero (0̄):** `(0̄₁, 0̄₂)` - impossible in both semirings
/// - **One (1̄):** `(1̄₁, 1̄₂)` - identity in both semirings
///
/// # Use Cases
///
/// ## Multi-Objective Optimization
/// ```rust
/// use arcweight::prelude::*;
///
/// // Optimize both cost and time simultaneously
/// type CostTimeWeight = ProductWeight<TropicalWeight, TropicalWeight>;
///
/// let route1 = CostTimeWeight::new(
///     TropicalWeight::new(100.0),  // $100 cost
///     TropicalWeight::new(30.0)    // 30 minute travel time
/// );
///
/// let route2 = CostTimeWeight::new(
///     TropicalWeight::new(80.0),   // $80 cost
///     TropicalWeight::new(45.0)    // 45 minute travel time
/// );
///
/// // Find Pareto optimal routes by component-wise comparison
/// let combined = route1.plus(&route2);  // (min(100,80), min(30,45)) = (80, 30)
/// // Note: This selects best in each dimension independently
/// ```
///
/// ## Cost-Probability Tracking
/// ```rust
/// use arcweight::prelude::*;
///
/// // Track both monetary cost and success probability
/// type CostProbWeight = ProductWeight<TropicalWeight, ProbabilityWeight>;
///
/// let option1 = CostProbWeight::new(
///     TropicalWeight::new(50.0),         // $50 cost
///     ProbabilityWeight::new(0.9)        // 90% success rate
/// );
///
/// let option2 = CostProbWeight::new(
///     TropicalWeight::new(30.0),         // $30 cost  
///     ProbabilityWeight::new(0.7)        // 70% success rate
/// );
///
/// // Sequential operations accumulate both cost and probability
/// let total = option1.times(&option2);  // ($80 total, 63% combined success)
/// ```
///
/// ## Quality-Performance Metrics
/// ```rust
/// use arcweight::prelude::*;
///
/// // Simultaneously track quality score and processing time
/// type QualityTimeWeight = ProductWeight<MaxWeight, TropicalWeight>;
///
/// let processor1 = QualityTimeWeight::new(
///     MaxWeight::new(0.95),        // 95% quality score
///     TropicalWeight::new(10.0)    // 10ms processing time
/// );
///
/// let processor2 = QualityTimeWeight::new(
///     MaxWeight::new(0.88),        // 88% quality score
///     TropicalWeight::new(5.0)     // 5ms processing time
/// );
///
/// // Pipeline combines quality constraints and accumulates time
/// let pipeline = processor1.times(&processor2);
/// // Quality: min(0.95, 0.88) = 0.88, Time: 10 + 5 = 15ms
/// ```
///
/// ## Speech Recognition Scoring
/// ```rust
/// use arcweight::prelude::*;
///
/// // Track acoustic and language model scores separately
/// type AcousticLanguageWeight = ProductWeight<LogWeight, LogWeight>;
///
/// let word1_scores = AcousticLanguageWeight::new(
///     LogWeight::from_probability(0.8),   // Acoustic model confidence
///     LogWeight::from_probability(0.6)    // Language model probability
/// );
///
/// let word2_scores = AcousticLanguageWeight::new(
///     LogWeight::from_probability(0.7),   // Acoustic model confidence
///     LogWeight::from_probability(0.9)    // Language model probability
/// );
///
/// // Sequence combines both model types
/// let sequence_score = word1_scores.times(&word2_scores);
/// // Acoustic: 0.8 × 0.7 = 0.56, Language: 0.6 × 0.9 = 0.54
/// ```
///
/// ## Resource Allocation with Constraints
/// ```rust
/// use arcweight::prelude::*;
///
/// // Track memory usage and computation time
/// type MemoryTimeWeight = ProductWeight<TropicalWeight, TropicalWeight>;
///
/// let algorithm1 = MemoryTimeWeight::new(
///     TropicalWeight::new(512.0),    // 512MB memory
///     TropicalWeight::new(100.0)     // 100ms compute time
/// );
///
/// let algorithm2 = MemoryTimeWeight::new(
///     TropicalWeight::new(256.0),    // 256MB memory
///     TropicalWeight::new(200.0)     // 200ms compute time
/// );
///
/// // Alternative algorithms - choose best in each dimension
/// let best_resources = algorithm1.plus(&algorithm2);  // (256MB, 100ms)
/// ```
///
/// # Working with FSTs
///
/// ```rust
/// use arcweight::prelude::*;
///
/// type DualTropical = ProductWeight<TropicalWeight, TropicalWeight>;
///
/// let weight1 = DualTropical::new(
///     TropicalWeight::new(2.0),
///     TropicalWeight::new(3.0)
/// );
///
/// let weight2 = DualTropical::new(
///     TropicalWeight::new(1.5),
///     TropicalWeight::new(4.0)
/// );
///
/// // Addition operates component-wise
/// let sum = weight1.clone() + weight2.clone();  // (min(2.0, 1.5), min(3.0, 4.0)) = (1.5, 3.0)
///
/// // Multiplication operates component-wise  
/// let product = weight1 * weight2;  // (2.0 + 1.5, 3.0 + 4.0) = (3.5, 7.0)
///
/// // Identity elements
/// let zero = DualTropical::zero();  // (∞, ∞)
/// let one = DualTropical::one();    // (0, 0)
/// ```
///
/// # Advanced Applications
///
/// ## FST Composition with Multiple Weights
/// ```rust
/// use arcweight::prelude::*;
///
/// // Compose FSTs tracking both edit distance and confidence
/// type EditConfidenceWeight = ProductWeight<TropicalWeight, ProbabilityWeight>;
///
/// // Build weighted FST for spell correction
/// let mut fst = VectorFst::<EditConfidenceWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s1, EditConfidenceWeight::new(
///     TropicalWeight::one(),         // No additional edit cost
///     ProbabilityWeight::one()       // Full confidence in acceptance
/// ));
///
/// // Add edit operation: substitute 'a' -> 'e'
/// fst.add_arc(s0, Arc::new(
///     'a' as u32,
///     'e' as u32,
///     EditConfidenceWeight::new(
///         TropicalWeight::new(1.0),      // Edit distance of 1
///         ProbabilityWeight::new(0.8)    // 80% confidence in correction
///     ),
///     s1
/// ));
/// ```
///
/// ## Multi-Model Integration
/// ```rust
/// use arcweight::prelude::*;
///
/// // Combine neural and statistical models
/// type NeuralStatWeight = ProductWeight<LogWeight, TropicalWeight>;
///
/// let neural_score = LogWeight::from_probability(0.85);     // Neural model confidence
/// let statistical_cost = TropicalWeight::new(2.3);         // Statistical model cost
///
/// let combined_weight = NeuralStatWeight::new(neural_score, statistical_cost);
///
/// // Can decompose later for model-specific analysis
/// let neural_component = &combined_weight.w1;
/// let statistical_component = &combined_weight.w2;
/// ```
///
/// ## Conditional Properties and Division
/// ```rust
/// use arcweight::prelude::*;
///
/// // Both component semirings must support division
/// type DivisibleProduct = ProductWeight<LogWeight, ProbabilityWeight>;
///
/// let joint = DivisibleProduct::new(
///     LogWeight::from_probability(0.3),    // P(A ∩ B)
///     ProbabilityWeight::new(0.6)          // P(B)
/// );
///
/// let marginal = DivisibleProduct::new(
///     LogWeight::from_probability(0.6),    // P(B)
///     ProbabilityWeight::new(0.6)          // P(B)
/// );
///
/// // Component-wise division for conditional probabilities
/// if let Some(conditional) = joint.divide(&marginal) {
///     // conditional.w1 ≈ P(A|B), conditional.w2 ≈ 1.0
/// }
/// ```
///
/// # Performance Characteristics
///
/// - **Arithmetic:** O(1) for each component, total O(k) for k-dimensional products
/// - **Memory:** Sum of component memory requirements plus efficient tuple caching
/// - **Comparison:** Component-wise comparison, may be expensive for complex weights
/// - **Properties:** Inherits intersection of component semiring properties
/// - **Scalability:** Suitable for small numbers of dimensions (typically 2-4)
/// - **Value Access:** Thread-safe lazy computation with optimized caching
///
/// # Mathematical Properties
///
/// The Product semiring preserves key algebraic properties:
/// - **Semiring Axioms:** If both components are semirings, product is a semiring
/// - **Commutativity:** Preserved if both components are commutative
/// - **Idempotency:** Preserved if both components are idempotent
/// - **Path Property:** Preserved if both components have path property
/// - **Divisibility:** Available if both components support division
///
/// # Design Considerations
///
/// ## Type Safety
/// The Product semiring enforces type safety at compile time:
/// ```rust
/// use arcweight::prelude::*;
///
/// // Type system prevents mixing incompatible product types
/// type CostTime = ProductWeight<TropicalWeight, TropicalWeight>;
/// type QualityProb = ProductWeight<MaxWeight, ProbabilityWeight>;
///
/// // These types are incompatible - compilation error if mixed
/// ```
///
/// ## Performance vs. Flexibility Trade-offs
/// - **Benefits:** Type-safe multi-dimensional optimization, compositional design
/// - **Costs:** Memory overhead, computational complexity increases with dimensions
/// - **Alternatives:** For high-dimensional problems, consider specialized data structures
///
/// # Integration with FST Algorithms
///
/// Product weights work with all FST algorithms that support their component semirings:
/// - **Shortest Path:** Finds optimal paths in all dimensions simultaneously
/// - **Composition:** Combines multi-weighted transducers
/// - **Determinization:** Maintains multi-dimensional weight structure
/// - **Minimization:** Preserves component-wise optimization properties
///
/// # Implementation Details
///
/// This implementation uses efficient thread-safe lazy evaluation:
/// - Components are stored directly for fast access
/// - Tuple values are computed on-demand with thread-safe caching
/// - Memory overhead is minimized while maintaining full trait compliance
/// - All semiring operations preserve mathematical correctness
/// - Send + Sync requirements are satisfied for multi-threading support
///
/// # See Also
///
/// - [Core Concepts - Product Semiring](../../docs/core-concepts/semirings.md#product-semiring) for mathematical background
/// - [`TropicalWeight`](crate::semiring::TropicalWeight) and [`LogWeight`](crate::semiring::LogWeight) for common component semirings
/// - [`compose()`](crate::algorithms::compose) for multi-weighted FST composition
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProductWeight<W1: Semiring, W2: Semiring> {
    /// First component weight
    pub w1: W1,
    /// Second component weight
    pub w2: W2,
    /// Thread-safe cached tuple value for efficient `value()` method implementation
    /// Uses OnceLock for thread-safe lazy initialization without RefCell/Mutex overhead
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_value: OnceLock<(W1::Value, W2::Value)>,
}

impl<W1: Semiring, W2: Semiring> ProductWeight<W1, W2> {
    /// Create a new product weight from two component weights
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// let weight = ProductWeight::new(
    ///     TropicalWeight::new(5.0),
    ///     BooleanWeight::one()
    /// );
    /// ```
    pub fn new(w1: W1, w2: W2) -> Self {
        Self {
            w1,
            w2,
            cached_value: OnceLock::new(),
        }
    }

    /// Create a new product weight from component values
    ///
    /// This is a convenience method that constructs the component weights
    /// from their underlying values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// let weight = ProductWeight::<TropicalWeight, BooleanWeight>::from_values(5.0, true);
    /// assert_eq!(weight.w1, TropicalWeight::new(5.0));
    /// assert_eq!(weight.w2, BooleanWeight::new(true));
    /// ```
    pub fn from_values(v1: W1::Value, v2: W2::Value) -> Self {
        Self::new(W1::new(v1), W2::new(v2))
    }

    /// Get references to both component weights as a tuple
    ///
    /// This provides efficient access to both components without cloning.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// let weight = ProductWeight::new(
    ///     TropicalWeight::new(5.0),
    ///     BooleanWeight::one()
    /// );
    /// let (w1_ref, w2_ref) = weight.components();
    /// ```
    pub fn components(&self) -> (&W1, &W2) {
        (&self.w1, &self.w2)
    }

    /// Decompose the product weight into its components
    ///
    /// This consumes the product weight and returns the two component weights.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// let weight = ProductWeight::new(
    ///     TropicalWeight::new(5.0),
    ///     BooleanWeight::one()
    /// );
    /// let (w1, w2) = weight.into_components();
    /// ```
    pub fn into_components(self) -> (W1, W2) {
        (self.w1, self.w2)
    }

    /// Apply a function to the first component while preserving the second
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// let weight = ProductWeight::new(
    ///     TropicalWeight::new(5.0),
    ///     BooleanWeight::one()
    /// );
    /// let doubled = weight.map_first(|w| w.times(&TropicalWeight::new(2.0)));
    /// ```
    pub fn map_first<F>(self, f: F) -> Self
    where
        F: FnOnce(W1) -> W1,
    {
        Self::new(f(self.w1), self.w2)
    }

    /// Apply a function to the second component while preserving the first
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// let weight = ProductWeight::new(
    ///     TropicalWeight::new(5.0),
    ///     BooleanWeight::one()
    /// );
    /// let negated = weight.map_second(|w| w.plus(&BooleanWeight::zero()));
    /// ```
    pub fn map_second<F>(self, f: F) -> Self
    where
        F: FnOnce(W2) -> W2,
    {
        Self::new(self.w1, f(self.w2))
    }

    /// Apply functions to both components simultaneously
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// let weight = ProductWeight::new(
    ///     TropicalWeight::new(5.0),
    ///     BooleanWeight::one()
    /// );
    /// let modified = weight.map_both(
    ///     |w1| w1.times(&TropicalWeight::new(2.0)),
    ///     |w2| w2.plus(&BooleanWeight::zero())
    /// );
    /// ```
    pub fn map_both<F1, F2>(self, f1: F1, f2: F2) -> Self
    where
        F1: FnOnce(W1) -> W1,
        F2: FnOnce(W2) -> W2,
    {
        Self::new(f1(self.w1), f2(self.w2))
    }

    /// Compute tuple value with thread-safe lazy initialization
    ///
    /// This method uses OnceLock for efficient, thread-safe lazy evaluation.
    /// The tuple is computed only when first needed and cached permanently.
    fn get_or_init_value(&self) -> &(W1::Value, W2::Value)
    where
        W1::Value: Clone,
        W2::Value: Clone,
    {
        self.cached_value
            .get_or_init(|| (self.w1.value().clone(), self.w2.value().clone()))
    }
}

impl<W1: Semiring, W2: Semiring> fmt::Display for ProductWeight<W1, W2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.w1, self.w2)
    }
}

// Implement PartialEq by comparing components directly (more efficient than value tuples)
impl<W1: Semiring, W2: Semiring> PartialEq for ProductWeight<W1, W2> {
    fn eq(&self, other: &Self) -> bool {
        self.w1 == other.w1 && self.w2 == other.w2
    }
}

impl<W1: Semiring, W2: Semiring> Eq for ProductWeight<W1, W2> {}

// Implement PartialOrd by comparing components lexicographically
impl<W1: Semiring, W2: Semiring> PartialOrd for ProductWeight<W1, W2> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.w1.partial_cmp(&other.w1) {
            Some(std::cmp::Ordering::Equal) => self.w2.partial_cmp(&other.w2),
            other => other,
        }
    }
}

// Implement Ord when both components are Ord
impl<W1, W2> Ord for ProductWeight<W1, W2>
where
    W1: Semiring + Ord,
    W2: Semiring + Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.w1.cmp(&other.w1) {
            std::cmp::Ordering::Equal => self.w2.cmp(&other.w2),
            other => other,
        }
    }
}

// Implement Hash when both components are Hash
impl<W1, W2> std::hash::Hash for ProductWeight<W1, W2>
where
    W1: Semiring + std::hash::Hash,
    W2: Semiring + std::hash::Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.w1.hash(state);
        self.w2.hash(state);
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
        Self::new(self.w1 + rhs.w1, self.w2 + rhs.w2)
    }
}

impl<W1: Semiring, W2: Semiring> Mul for ProductWeight<W1, W2> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.w1 * rhs.w1, self.w2 * rhs.w2)
    }
}

impl<W1, W2> Semiring for ProductWeight<W1, W2>
where
    W1: Semiring,
    W2: Semiring,
    W1::Value: Clone + Send + Sync,
    W2::Value: Clone + Send + Sync,
{
    type Value = (W1::Value, W2::Value);

    fn new(value: Self::Value) -> Self {
        Self::new(W1::new(value.0), W2::new(value.1))
    }

    fn value(&self) -> &Self::Value {
        // Thread-safe lazy evaluation using OnceLock
        // This is both efficient and mathematically sound
        self.get_or_init_value()
    }

    fn plus(&self, other: &Self) -> Self {
        Self::new(self.w1.plus(&other.w1), self.w2.plus(&other.w2))
    }

    fn times(&self, other: &Self) -> Self {
        Self::new(self.w1.times(&other.w1), self.w2.times(&other.w2))
    }

    fn plus_assign(&mut self, other: &Self) {
        self.w1.plus_assign(&other.w1);
        self.w2.plus_assign(&other.w2);
        // Note: OnceLock cannot be cleared, but since we're mutating the components,
        // we create a new instance with fresh cache. This maintains correctness.
        *self = Self::new(self.w1.clone(), self.w2.clone());
    }

    fn times_assign(&mut self, other: &Self) {
        self.w1.times_assign(&other.w1);
        self.w2.times_assign(&other.w2);
        // Note: OnceLock cannot be cleared, but since we're mutating the components,
        // we create a new instance with fresh cache. This maintains correctness.
        *self = Self::new(self.w1.clone(), self.w2.clone());
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
        self.w1.approx_eq(&other.w1, epsilon) && self.w2.approx_eq(&other.w2, epsilon)
    }
}

impl<W1, W2> DivisibleSemiring for ProductWeight<W1, W2>
where
    W1: DivisibleSemiring,
    W2: DivisibleSemiring,
    W1::Value: Clone + Send + Sync,
    W2::Value: Clone + Send + Sync,
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
    W1::Value: Clone + Send + Sync,
    W2::Value: Clone + Send + Sync,
{
    fn star(&self) -> Self {
        Self::new(self.w1.star(), self.w2.star())
    }
}

impl<W1, W2> NaturallyOrderedSemiring for ProductWeight<W1, W2>
where
    W1: NaturallyOrderedSemiring,
    W2: NaturallyOrderedSemiring,
    W1::Value: Clone + Send + Sync,
    W2::Value: Clone + Send + Sync,
{
}

// Default implementation for convenient construction
impl<W1: Semiring + Default, W2: Semiring + Default> Default for ProductWeight<W1, W2> {
    fn default() -> Self {
        Self::new(W1::default(), W2::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semiring::{ProbabilityWeight, TropicalWeight};
    use num_traits::{One, Zero};

    #[test]
    fn test_product_weight_creation() {
        let w = ProductWeight::new(TropicalWeight::new(2.0), ProbabilityWeight::new(0.5));
        assert_eq!(*w.w1.value(), 2.0);
        assert_eq!(*w.w2.value(), 0.5);
    }

    #[test]
    fn test_product_zero_one() {
        let zero = ProductWeight::<TropicalWeight, ProbabilityWeight>::zero();
        let one = ProductWeight::<TropicalWeight, ProbabilityWeight>::one();

        assert!(Semiring::is_zero(&zero));
        assert!(Semiring::is_one(&one));
        assert!(Semiring::is_zero(&zero.w1));
        assert!(Semiring::is_zero(&zero.w2));
        assert!(Semiring::is_one(&one.w1));
        assert!(Semiring::is_one(&one.w2));
    }

    #[test]
    fn test_product_operations() {
        let w1 = ProductWeight::new(TropicalWeight::new(2.0), ProbabilityWeight::new(0.3));
        let w2 = ProductWeight::new(TropicalWeight::new(3.0), ProbabilityWeight::new(0.5));

        let add_result = w1.plus(&w2);
        let mul_result = w1.times(&w2);

        // Should apply operations component-wise
        assert_eq!(*add_result.w1.value(), 2.0); // min(2, 3)
        assert_eq!(*add_result.w2.value(), 0.8); // 0.3 + 0.5

        assert_eq!(*mul_result.w1.value(), 5.0); // 2 + 3
        assert_eq!(*mul_result.w2.value(), 0.15); // 0.3 * 0.5
    }

    #[test]
    fn test_product_display() {
        let w = ProductWeight::new(TropicalWeight::new(2.0), ProbabilityWeight::new(0.5));
        assert_eq!(format!("{w}"), "(2, 0.5)");

        let zero = ProductWeight::<TropicalWeight, ProbabilityWeight>::zero();
        assert_eq!(format!("{zero}"), "(∞, 0)");
    }

    #[test]
    fn test_product_division() {
        let w1 = ProductWeight::new(TropicalWeight::new(5.0), ProbabilityWeight::new(0.6));
        let w2 = ProductWeight::new(TropicalWeight::new(2.0), ProbabilityWeight::new(0.3));

        let result = w1.divide(&w2).unwrap();
        assert_eq!(*result.w1.value(), 3.0); // 5 - 2
        assert_eq!(*result.w2.value(), 2.0); // 0.6 / 0.3

        // Division by zero in either component should return None
        let zero = ProductWeight::<TropicalWeight, ProbabilityWeight>::zero();
        assert!(w1.divide(&zero).is_none());
    }

    #[test]
    fn test_product_properties() {
        let props = ProductWeight::<TropicalWeight, ProbabilityWeight>::properties();

        // Properties are intersection of component properties
        assert!(props.left_semiring);
        assert!(props.right_semiring);
        assert!(props.commutative); // Both tropical and probability are commutative
        assert!(!props.idempotent); // Probability is not idempotent
        assert!(!props.path); // Probability is not path
    }

    #[test]
    fn test_product_approx_eq() {
        let w1 = ProductWeight::new(TropicalWeight::new(2.0001), ProbabilityWeight::new(0.5001));
        let w2 = ProductWeight::new(TropicalWeight::new(2.0), ProbabilityWeight::new(0.5));

        assert!(w1.approx_eq(&w2, 0.001));
        assert!(!w1.approx_eq(&w2, 0.00001));
    }

    #[test]
    fn test_product_operator_overloads() {
        let w1 = ProductWeight::new(TropicalWeight::new(2.0), ProbabilityWeight::new(0.3));
        let w2 = ProductWeight::new(TropicalWeight::new(3.0), ProbabilityWeight::new(0.5));

        // Test + operator
        let sum = w1 + w2;
        assert_eq!(*sum.w1.value(), 2.0); // min(2, 3)
        assert_eq!(*sum.w2.value(), 0.8); // 0.3 + 0.5

        // Test * operator
        let w1 = ProductWeight::new(TropicalWeight::new(2.0), ProbabilityWeight::new(0.3));
        let w2 = ProductWeight::new(TropicalWeight::new(3.0), ProbabilityWeight::new(0.5));
        let product = w1 * w2;
        assert_eq!(*product.w1.value(), 5.0); // 2 + 3
        assert_eq!(*product.w2.value(), 0.15); // 0.3 * 0.5
    }

    #[test]
    fn test_product_identity_laws() {
        let w = ProductWeight::new(TropicalWeight::new(2.0), ProbabilityWeight::new(0.5));
        let zero = ProductWeight::<TropicalWeight, ProbabilityWeight>::zero();
        let one = ProductWeight::<TropicalWeight, ProbabilityWeight>::one();

        // Additive identity
        assert_eq!(w.clone() + zero.clone(), w);
        assert_eq!(zero.clone() + w.clone(), w);

        // Multiplicative identity
        assert_eq!(w.clone() * one.clone(), w);
        assert_eq!(one.clone() * w.clone(), w);

        // Annihilation by zero
        assert!(Semiring::is_zero(&(w.clone() * zero.clone())));
        assert!(Semiring::is_zero(&(zero * w)));
    }

    #[test]
    fn test_product_semiring_axioms() {
        let a = ProductWeight::new(TropicalWeight::new(1.0), ProbabilityWeight::new(0.2));
        let b = ProductWeight::new(TropicalWeight::new(2.0), ProbabilityWeight::new(0.3));
        let c = ProductWeight::new(TropicalWeight::new(3.0), ProbabilityWeight::new(0.4));
        let tolerance = 1e-10;

        // Associativity of addition
        assert!(((a.clone() + b.clone()) + c.clone())
            .approx_eq(&(a.clone() + (b.clone() + c.clone())), tolerance));

        // Associativity of multiplication
        assert!(((a.clone() * b.clone()) * c.clone())
            .approx_eq(&(a.clone() * (b.clone() * c.clone())), tolerance));

        // Commutativity of addition
        assert_eq!(a.clone() + b.clone(), b.clone() + a.clone());

        // Commutativity of multiplication
        assert_eq!(a.clone() * b.clone(), b.clone() * a.clone());

        // Distributivity
        assert!(((a.clone() + b.clone()) * c.clone()).approx_eq(
            &((a.clone() * c.clone()) + (b.clone() * c.clone())),
            tolerance
        ));
    }
}
