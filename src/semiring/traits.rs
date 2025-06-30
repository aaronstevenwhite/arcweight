//! Core semiring traits for weight types

use core::fmt::{Debug, Display};
use core::ops::{Add, Mul};
use num_traits::{One, Zero};

/// Core semiring trait defining algebraic operations for weights
///
/// A semiring is an algebraic structure (S, ⊕, ⊗, 0̄, 1̄) consisting of:
/// - A set S of elements (weights)
/// - Binary operation ⊕ (addition, for combining alternative paths)
/// - Binary operation ⊗ (multiplication, for path concatenation)  
/// - Additive identity 0̄ (represents "no path" or impossible transition)
/// - Multiplicative identity 1̄ (represents "free" or zero-cost transition)
///
/// # Semiring Axioms
///
/// 1. **(S, ⊕, 0̄) is a commutative monoid**
///    - Associativity: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
///    - Commutativity: a ⊕ b = b ⊕ a
///    - Identity: a ⊕ 0̄ = 0̄ ⊕ a = a
///
/// 2. **(S, ⊗, 1̄) is a monoid**
///    - Associativity: (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)
///    - Identity: a ⊗ 1̄ = 1̄ ⊗ a = a
///
/// 3. **⊗ distributes over ⊕**
///    - Left: a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)
///    - Right: (a ⊕ b) ⊗ c = (a ⊗ c) ⊕ (b ⊗ c)
///
/// 4. **0̄ is an annihilator**
///    - a ⊗ 0̄ = 0̄ ⊗ a = 0̄
///
/// # Implementation Guidelines
///
/// When implementing a custom semiring:
/// 1. Ensure all semiring axioms are satisfied
/// 2. Use efficient implementations for `plus()` and `times()`
/// 3. Override `approx_eq()` for floating-point weights
/// 4. Set appropriate `SemiringProperties` for optimizations
/// 5. Consider numerical stability for probabilistic semirings
///
/// # Examples
///
/// ## Using Existing Semirings
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Tropical semiring for shortest path
/// let w1 = TropicalWeight::new(2.5);
/// let w2 = TropicalWeight::new(1.8);
/// let sum = w1.plus(&w2);      // min(2.5, 1.8) = 1.8
/// let product = w1.times(&w2); // 2.5 + 1.8 = 4.3
///
/// // Boolean semiring for recognition
/// let b1 = BooleanWeight::one();  // true
/// let b2 = BooleanWeight::zero(); // false
/// let or_result = b1.plus(&b2);   // true OR false = true
/// let and_result = b1.times(&b2); // true AND false = false
/// ```
///
/// ## Custom Semiring Implementation
///
/// ```rust
/// use arcweight::prelude::*;
/// use num_traits::{Zero, One};
/// use std::ops::{Add, Mul};
/// use std::fmt;
///
/// // Lexicographic semiring (primary_cost, secondary_count)
/// #[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
/// pub struct LexWeight {
///     cost: u32,  // Using u32 for total ordering
///     count: u32,
/// }
///
/// impl fmt::Display for LexWeight {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         write!(f, "({}, {})", self.cost, self.count)
///     }
/// }
///
/// impl Add for LexWeight {
///     type Output = Self;
///     fn add(self, other: Self) -> Self {
///         self.plus(&other)
///     }
/// }
///
/// impl Mul for LexWeight {
///     type Output = Self;
///     fn mul(self, other: Self) -> Self {
///         self.times(&other)
///     }
/// }
///
/// impl Zero for LexWeight {
///     fn zero() -> Self {
///         Self { cost: u32::MAX, count: 0 }  // Infinite cost for zero
///     }
///     
///     fn is_zero(&self) -> bool {
///         self.cost == u32::MAX
///     }
/// }
///
/// impl One for LexWeight {
///     fn one() -> Self {
///         Self { cost: 0, count: 0 }  // Zero cost for identity
///     }
/// }
///
/// impl Semiring for LexWeight {
///     type Value = (u32, u32);
///     
///     fn new(value: Self::Value) -> Self {
///         Self { cost: value.0, count: value.1 }
///     }
///     
///     fn value(&self) -> &Self::Value {
///         // In practice this would store a tuple field
///         unimplemented!("Would need tuple storage for this")
///     }
///     
///     fn plus(&self, other: &Self) -> Self {
///         // Lexicographic minimum
///         if self.cost < other.cost ||
///            (self.cost == other.cost && self.count <= other.count) {
///             self.clone()
///         } else {
///             other.clone()
///         }
///     }
///     
///     fn times(&self, other: &Self) -> Self {
///         Self {
///             cost: self.cost + other.cost,
///             count: self.count + other.count,
///         }
///     }
///     
///     fn properties() -> SemiringProperties {
///         SemiringProperties {
///             left_semiring: true,
///             right_semiring: true,
///             commutative: true,
///             idempotent: false,
///             path: false,
///         }
///     }
///     
///     fn approx_eq(&self, other: &Self, _epsilon: f64) -> bool {
///         self == other
///     }
/// }
///
/// // Example usage
/// let w1 = LexWeight { cost: 5, count: 2 };
/// let w2 = LexWeight { cost: 3, count: 1 };
/// let sum = w1.plus(&w2);  // (3, 1) - lexicographically smaller
/// let product = w1.times(&w2);  // (8, 3) - costs and counts add
/// assert_eq!(sum, w2);
/// assert_eq!(product.cost, 8);
/// ```
///
/// # Common Semirings
///
/// - **[`TropicalWeight`]:** Min-plus algebra for shortest paths
/// - **[`LogWeight`]:** Log semiring for probabilistic computations  
/// - **[`BooleanWeight`]:** Boolean algebra for recognition
/// - **[`ProbabilityWeight`]:** Standard probability semiring
///
/// # Performance Considerations
///
/// - **In-place operations:** Use `plus_assign()` and `times_assign()` when possible
/// - **Numerical stability:** Use [`LogWeight`] instead of [`ProbabilityWeight`] for small probabilities
/// - **Custom semirings:** Profile addition and multiplication operations as they're called frequently
///
/// # See Also
///
/// - [Core Concepts - Semiring Theory](../../docs/core-concepts/semirings.md) for mathematical background
/// - [API Reference - Semirings](../../docs/api-reference.md#semirings) for complete semiring catalog
/// - [`DivisibleSemiring`] for semirings supporting division
/// - [`StarSemiring`] for semirings supporting Kleene closure
///
/// [`TropicalWeight`]: crate::semiring::TropicalWeight
/// [`LogWeight`]: crate::semiring::LogWeight
/// [`BooleanWeight`]: crate::semiring::BooleanWeight
/// [`ProbabilityWeight`]: crate::semiring::ProbabilityWeight
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

/// Properties of a semiring that determine algorithmic optimizations and guarantees
///
/// These properties characterize the mathematical structure of a semiring and enable
/// algorithm implementers to choose appropriate optimizations, select efficient data
/// structures, and guarantee correctness of iterative procedures.
///
/// # Mathematical Properties
///
/// ## Distributivity Properties
/// - **Left semiring:** Multiplication distributes over addition from the left
///   `(a ⊕ b) ⊗ c = (a ⊗ c) ⊕ (b ⊗ c)`
/// - **Right semiring:** Multiplication distributes over addition from the right  
///   `c ⊗ (a ⊕ b) = (c ⊗ a) ⊕ (c ⊗ b)`
///
/// Most practical semirings are both left and right semirings (true semirings).
///
/// ## Optimization Properties
/// - **Commutative:** Multiplication order doesn't matter `a ⊗ b = b ⊗ a`
/// - **Idempotent:** Addition is idempotent `a ⊕ a = a`
/// - **Path property:** Addition selects one operand `a ⊕ b ∈ {a, b}`
///
/// # Algorithmic Implications
///
/// ## Idempotent Semirings
/// Enable optimization in shortest-path algorithms:
/// - **Convergence:** Guarantee that iterative algorithms terminate
/// - **Memoization:** Safe to cache intermediate results
/// - **Pruning:** Can eliminate dominated alternatives early
///
/// ## Path Property
/// Ensures that algorithms find actual optimal solutions:
/// - **Single path:** Addition selects rather than combines
/// - **Optimization:** Enables shortest-path semantics
/// - **Determinism:** Reproducible results with consistent tie-breaking
///
/// ## Commutativity
/// Allows reordering optimizations:
/// - **Parallel computation:** Operations can be reordered safely
/// - **Memory layout:** Can optimize for cache locality
/// - **Batch processing:** Can group operations efficiently
///
/// # Examples by Semiring Type
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Tropical: idempotent, path property, commutative
/// let props = TropicalWeight::properties();
/// assert!(props.idempotent);  // min(a, a) = a
/// assert!(props.path);        // min(a, b) ∈ {a, b}
/// assert!(props.commutative); // a + b = b + a
///
/// // Boolean: idempotent, path property, commutative  
/// let props = BooleanWeight::properties();
/// assert!(props.idempotent);  // a ∨ a = a
/// assert!(props.path);        // a ∨ b ∈ {a, b}
/// assert!(props.commutative); // a ∧ b = b ∧ a
///
/// // Probability: NOT idempotent, NOT path property, commutative
/// let props = ProbabilityWeight::properties();
/// assert!(!props.idempotent); // p + p ≠ p (usually)
/// assert!(!props.path);       // p + q ∉ {p, q} (usually)
/// assert!(props.commutative); // p × q = q × p
/// ```
///
/// # Performance Optimization Guidelines
///
/// ## For Idempotent Semirings
/// - Use hash-based duplicate elimination
/// - Implement early termination in iterative algorithms
/// - Cache results of expensive computations
///
/// ## For Path Property Semirings  
/// - Implement efficient priority queues for shortest-path algorithms
/// - Use deterministic tie-breaking for reproducible results
/// - Optimize for single-path extraction rather than path enumeration
///
/// ## For Commutative Semirings
/// - Reorder operations for numerical stability
/// - Use SIMD instructions for batch operations
/// - Optimize memory access patterns
///
/// # See Also
///
/// - [Core Concepts - Semiring Theory](../../docs/core-concepts/semirings.md#specialized-properties-for-optimization) for mathematical foundations
/// - [`Semiring::properties`] method for querying these properties
/// - FST algorithms that leverage these properties for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemiringProperties {
    /// Left semiring: (a ⊕ b) ⊗ c = (a ⊗ c) ⊕ (b ⊗ c)
    pub left_semiring: bool,
    /// Right semiring: c ⊗ (a ⊕ b) = (c ⊗ a) ⊕ (c ⊗ b)
    pub right_semiring: bool,
    /// Commutative: a ⊗ b = b ⊗ a
    pub commutative: bool,
    /// Idempotent: a ⊕ a = a
    pub idempotent: bool,
    /// Path property: a ⊕ b ∈ {a, b}
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
///
/// A divisible semiring supports division operation, enabling more sophisticated
/// algorithms like weight pushing and composition optimization. The division
/// operation is the inverse of multiplication when it exists.
///
/// # Mathematical Definition
///
/// For a divisible semiring, if `a ⊗ b = c`, then `c.divide(&b) = Some(a)`
/// (when the division is defined).
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Tropical semiring supports division
/// let a = TropicalWeight::new(5.0);
/// let b = TropicalWeight::new(2.0);
/// let c = a.times(&b); // 5.0 + 2.0 = 7.0
///
/// // Division: c / b = a
/// if let Some(result) = c.divide(&b) {
///     assert_eq!(result, a); // 7.0 - 2.0 = 5.0
/// }
/// ```
///
/// # See Also
///
/// - [`TropicalWeight`] implements this trait
/// - [`LogWeight`] implements this trait
/// - [`BooleanWeight`] does NOT implement this (division not meaningful)
///
/// [`TropicalWeight`]: crate::semiring::TropicalWeight
/// [`LogWeight`]: crate::semiring::LogWeight
/// [`BooleanWeight`]: crate::semiring::BooleanWeight
pub trait DivisibleSemiring: Semiring {
    /// Division operation
    fn divide(&self, other: &Self) -> Option<Self>;
}

/// Trait for weights that support star operation (Kleene closure)
///
/// The star semiring enables Kleene closure operations, which compute the infinite
/// sum w* = 1̄ ⊕ w ⊕ w² ⊕ w³ ⊕ ... This is essential for closure algorithms and
/// cyclic path analysis in FSTs.
///
/// # Mathematical Definition
///
/// For a k-closed semiring, the star operation converges:
/// w* = ⊕ᵢ₌₀^∞ wⁱ
///
/// This requires that the infinite sum has a well-defined limit.
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Boolean semiring: w* = true for any w ≠ false
/// let w = BooleanWeight::one();
/// let star = w.star(); // Always true (can reach via 0 or more steps)
///
/// // Note: TropicalWeight does not implement StarSemiring in this implementation
/// // Tropical star operation requires additional mathematical considerations
/// ```
///
/// # Applications
///
/// - **Closure algorithms:** Computing transitive closure of FSTs
/// - **Cycle analysis:** Finding optimal paths through cycles
/// - **Regular expression:** Implementing Kleene star operator
///
/// # See Also
///
/// - [`BooleanWeight`] implements this trait
/// - Note: [`TropicalWeight`] does not implement this trait in the current implementation
/// - [Closure algorithms](crate::algorithms::closure) for usage
///
/// [`BooleanWeight`]: crate::semiring::BooleanWeight
/// [`TropicalWeight`]: crate::semiring::TropicalWeight
pub trait StarSemiring: Semiring {
    /// Star operation: w* = 1 ⊕ w ⊕ w² ⊕ ...
    fn star(&self) -> Self;
}

/// Trait for semirings with natural ordering compatible with semiring operations
///
/// A naturally ordered semiring has a partial order ≤ that is compatible with
/// both addition and multiplication operations. This ordering enables efficient
/// shortest-path algorithms and provides convergence guarantees for iterative
/// procedures.
///
/// # Mathematical Definition
///
/// A semiring (S, ⊕, ⊗, 0̄, 1̄) is naturally ordered if there exists a partial order ≤ such that:
/// 1. For all a, b ∈ S: a ≤ a ⊕ b and b ≤ a ⊕ b
/// 2. If a ≤ b, then a ⊕ c ≤ b ⊕ c and c ⊕ a ≤ c ⊕ b  
/// 3. If a ≤ b, then a ⊗ c ≤ b ⊗ c and c ⊗ a ≤ c ⊗ b
///
/// # Algorithmic Benefits
///
/// Natural ordering enables several important algorithmic optimizations:
/// - **Monotonic algorithms:** Shortest-path algorithms maintain ordering invariants
/// - **Early termination:** Can stop when optimal solution is proven
/// - **Priority queues:** Efficient implementation of Dijkstra's algorithm variants
/// - **Convergence proofs:** Mathematical guarantees for iterative procedures
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // Tropical semiring is naturally ordered by ≤ on real numbers
/// let w1 = TropicalWeight::new(2.0);
/// let w2 = TropicalWeight::new(3.0);
/// assert!(w1 <= w2);  // 2.0 ≤ 3.0
///
/// // Addition preserves ordering: min(2,3) = 2 ≤ max(2,3) = 3
/// let sum = w1.plus(&w2);
/// assert!(sum <= w1 && sum <= w2);  // min(a,b) ≤ a,b
///
/// // Boolean semiring is naturally ordered by false ≤ true  
/// let false_w = BooleanWeight::zero();
/// let true_w = BooleanWeight::one();
/// assert!(false_w <= true_w);
/// ```
///
/// # Implementation Requirements
///
/// Types implementing this trait must:
/// 1. Implement [`Ord`] with total ordering compatible with semiring structure
/// 2. Ensure ordering is preserved by semiring operations
/// 3. Provide efficient comparison operations for algorithm performance
///
/// # See Also
///
/// - [`TropicalWeight`] implements this trait
/// - [`BooleanWeight`] implements this trait  
/// - [`MinWeight`] and [`MaxWeight`] implement this trait
/// - Shortest-path algorithms that require natural ordering
///
/// [`TropicalWeight`]: crate::semiring::TropicalWeight
/// [`BooleanWeight`]: crate::semiring::BooleanWeight
/// [`MinWeight`]: crate::semiring::MinWeight
/// [`MaxWeight`]: crate::semiring::MaxWeight
pub trait NaturallyOrderedSemiring: Semiring + Ord {}

/// Trait for semirings that support multiplicative inverse operations
///
/// An invertible semiring provides multiplicative inverses for non-zero elements,
/// enabling division-like operations and supporting algorithms that require
/// "undoing" multiplication operations. This is particularly useful for weight
/// pushing, normalization, and certain optimization procedures.
///
/// # Mathematical Definition
///
/// For an invertible semiring, each non-zero element a has a multiplicative
/// inverse a⁻¹ such that:
/// - a ⊗ a⁻¹ = a⁻¹ ⊗ a = 1̄
///
/// The zero element 0̄ typically has no inverse (division by zero is undefined).
///
/// # Relationship to Divisible Semirings
///
/// Invertible semirings are stronger than divisible semirings:
/// - **Divisible:** Can divide when the result exists  
/// - **Invertible:** Every element has an inverse (multiplicative group structure)
///
/// Many practical semirings are divisible but not invertible (e.g., tropical semiring).
///
/// # Applications
///
/// Invertible semirings enable advanced algorithms:
/// - **Weight pushing:** Redistributing weights for normalization
/// - **Matrix operations:** Implementing linear algebra over semirings
/// - **Formal series:** Manipulating power series with semiring coefficients
/// - **Constraint solving:** Solving systems of semiring equations
///
/// # Examples
///
/// ```rust
/// use arcweight::prelude::*;
///
/// // RealWeight implements InvertibleSemiring for real number arithmetic
/// let a = RealWeight::new(2.0);
/// let b = RealWeight::new(0.5);
///
/// // Multiplicative inverses exist for non-zero elements
/// if let Some(inv_a) = a.inverse() {
///     // inv_a is 0.5, since 2.0 * 0.5 = 1.0
///     let identity = a.times(&inv_a);
///     assert_eq!(identity, RealWeight::one());
/// }
///
/// if let Some(inv_b) = b.inverse() {
///     // inv_b is 2.0, since 0.5 * 2.0 = 1.0
///     let identity = b.times(&inv_b);
///     assert_eq!(identity, RealWeight::one());
/// }
///
/// // Zero element has no inverse
/// let zero = RealWeight::zero();
/// assert_eq!(zero.inverse(), None);
/// ```
///
/// # Implementation Considerations
///
/// When implementing this trait:
/// 1. **Numerical stability:** Handle floating-point precision issues
/// 2. **Zero handling:** Return `None` for zero elements  
/// 3. **Overflow protection:** Guard against division by very small numbers
/// 4. **Performance:** Optimize for common cases (powers of 2, etc.)
///
/// # Mathematical Properties
///
/// Invertible semirings often have additional structure:
/// - **Group property:** Non-zero elements form a multiplicative group
/// - **Field-like:** May approach field semantics with additive inverses
/// - **Unique factorization:** Enable canonical decompositions
///
/// # See Also
///
/// - [`DivisibleSemiring`] for weaker division operations
/// - Field semirings for full ring/field structure
/// - Advanced algorithms requiring inverse operations
pub trait InvertibleSemiring: Semiring {
    /// Compute multiplicative inverse
    ///
    /// Returns `Some(inverse)` if the inverse exists, `None` for zero element
    /// or if the inverse cannot be computed (e.g., numerical instability).
    ///
    /// # Mathematical Property
    ///
    /// If `Some(inv) = a.inverse()`, then `a.times(&inv)` should equal `Self::one()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use arcweight::prelude::*;
    ///
    /// // RealWeight provides full InvertibleSemiring functionality
    /// let a = RealWeight::new(2.0);
    /// let b = RealWeight::new(0.5);
    ///
    /// // Multiplicative inverses exist for non-zero elements
    /// if let Some(inv_a) = a.inverse() {
    ///     // inv_a is 0.5, since 2.0 * 0.5 = 1.0
    ///     let identity = a.times(&inv_a);
    ///     assert_eq!(identity, RealWeight::one());
    /// }
    ///
    /// if let Some(inv_b) = b.inverse() {
    ///     // inv_b is 2.0, since 0.5 * 2.0 = 1.0
    ///     let identity = b.times(&inv_b);
    ///     assert_eq!(identity, RealWeight::one());
    /// }
    ///
    /// // Zero element has no inverse
    /// let zero = RealWeight::zero();
    /// assert_eq!(zero.inverse(), None);
    ///
    /// // Applications: weight normalization
    /// let weights = vec![
    ///     RealWeight::new(4.0),
    ///     RealWeight::new(2.0),
    ///     RealWeight::new(8.0),
    /// ];
    ///
    /// // Normalize by dividing by total (using inverses)
    /// let total = weights.iter().fold(RealWeight::zero(), |acc, w| acc.plus(w));
    /// if let Some(inv_total) = total.inverse() {
    ///     let normalized: Vec<_> = weights.iter()
    ///         .map(|w| w.times(&inv_total))
    ///         .collect();
    ///     
    ///     // Verify normalization: sum should equal 1.0
    ///     let sum = normalized.iter().fold(RealWeight::zero(), |acc, w| acc.plus(w));
    ///     assert!((sum.as_f64() - 1.0).abs() < 1e-10);
    /// }
    /// ```
    fn inverse(&self) -> Option<Self>;
}
