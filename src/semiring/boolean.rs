//! Boolean semiring implementation

use super::traits::*;
use core::fmt;
use core::ops::{Add, Mul};
use core::str::FromStr;
use num_traits::{One, Zero};

/// Boolean weight for recognition and reachability analysis
///
/// The Boolean semiring provides the simplest non-trivial semiring structure,
/// corresponding to unweighted finite state automata and basic recognition problems.
/// In this semiring, "addition" implements logical OR (path existence), while
/// "multiplication" implements logical AND (path conjunction).
///
/// # Mathematical Semantics
///
/// - **Value Range:** Boolean values {false, true} ≅ {⊥, ⊤}
/// - **Addition (⊕):** `a ∨ b` (logical OR) - combines alternative paths
/// - **Multiplication (⊗):** `a ∧ b` (logical AND) - requires all conditions
/// - **Zero (0̄):** `false` - represents rejection/failure/no path
/// - **One (1̄):** `true` - represents acceptance/success/valid path
///
/// # Use Cases
///
/// ## Regular Expression Matching
/// ```rust
/// use arcweight::prelude::*;
///
/// // Build pattern matcher FST
/// let mut fst = VectorFst::<BooleanWeight>::new();
/// let s0 = fst.add_state();
/// let s1 = fst.add_state();
///
/// fst.set_start(s0);
/// fst.set_final(s1, BooleanWeight::one());  // Accept state
///
/// // Add pattern transition: match 'a'
/// fst.add_arc(s0, Arc::new(
///     'a' as u32,
///     'a' as u32,
///     BooleanWeight::one(),  // Valid transition
///     s1
/// ));
///
/// // The FST now accepts strings containing 'a'
/// ```
///
/// ## Graph Reachability
/// ```rust
/// use arcweight::prelude::*;
///
/// // Check if path exists between nodes
/// let path_exists = BooleanWeight::new(true);
/// let no_path = BooleanWeight::new(false);
///
/// // Combine multiple path possibilities
/// let reachable = path_exists
///     .plus(&no_path)
///     .plus(&path_exists);  // true ∨ false ∨ true = true
///
/// // Check if all intermediate nodes are valid
/// let validᵢntermediate = BooleanWeight::new(true);
/// let path_validity = reachable.times(&validᵢntermediate);  // true ∧ true = true
/// ```
///
/// ## Set Membership Testing
/// ```rust
/// use arcweight::prelude::*;
///
/// // Dictionary lookup FST
/// let word_exists = BooleanWeight::new(true);   // Word found in dictionary
/// let word_missing = BooleanWeight::new(false); // Word not found
///
/// // Multiple dictionary checks
/// let found_anywhere = word_missing
///     .plus(&word_exists);  // false ∨ true = true (found in at least one)
///
/// // Require presence in multiple dictionaries
/// let found_everywhere = word_exists
///     .times(&word_exists); // true ∧ true = true (found in all)
/// ```
///
/// ## Constraint Satisfaction
/// ```rust
/// use arcweight::prelude::*;
///
/// // Grammar rule satisfaction
/// let rule1_satisfied = BooleanWeight::new(true);
/// let rule2_satisfied = BooleanWeight::new(false);
/// let rule3_satisfied = BooleanWeight::new(true);
///
/// // All rules must be satisfied (conjunction)
/// let all_rules = rule1_satisfied
///     .times(&rule2_satisfied)
///     .times(&rule3_satisfied);  // true ∧ false ∧ true = false
///
/// // At least one rule satisfied (disjunction)
/// let any_rule = rule1_satisfied
///     .plus(&rule2_satisfied)
///     .plus(&rule3_satisfied);   // true ∨ false ∨ true = true
/// ```
///
/// # Working with FSTs
///
/// ```rust
/// use arcweight::prelude::*;
///
/// let true_val = BooleanWeight::new(true);
/// let false_val = BooleanWeight::new(false);
///
/// // Addition is logical OR (alternative paths)
/// let or_result = true_val + false_val;
/// assert_eq!(or_result, BooleanWeight::new(true));
///
/// // Multiplication is logical AND (path conjunction)
/// let and_result = true_val * false_val;
/// assert_eq!(and_result, BooleanWeight::new(false));
///
/// // Idempotency properties
/// assert_eq!(true_val + true_val, true_val);   // true ∨ true = true
/// assert_eq!(false_val * false_val, false_val); // false ∧ false = false
///
/// // Identity elements
/// assert_eq!(BooleanWeight::zero(), BooleanWeight::new(false));  // Additive identity
/// assert_eq!(BooleanWeight::one(), BooleanWeight::new(true));    // Multiplicative identity
/// ```
///
/// # Algebraic Properties
///
/// The Boolean semiring forms a complete Boolean algebra with important properties:
///
/// - **Idempotent:** `a ⊕ a = a` and `a ⊗ a = a`
/// - **Commutative:** `a ⊕ b = b ⊕ a` and `a ⊗ b = b ⊗ a`
/// - **Associative:** `(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)`
/// - **Distributive:** `a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)`
/// - **Absorptive:** `a ⊕ (a ⊗ b) = a` and `a ⊗ (a ⊕ b) = a`
/// - **Complemented:** Each element has a complement (negation)
///
/// # Performance Characteristics
///
/// - **Arithmetic:** Both OR and AND operations are O(1) bit operations
/// - **Memory:** 1 byte per weight (single bool with padding)
/// - **Comparison:** Extremely fast integer comparison
/// - **Hash/Eq:** Optimal for hash maps and sets
/// - **Specialized:** Can be bit-packed for massive state spaces
///
/// # Advanced Usage
///
/// ## Kleene Star
/// ```rust
/// use arcweight::prelude::*;
///
/// let weight = BooleanWeight::new(true);
/// let star_result = weight.star();  // Always true for non-zero elements
/// assert_eq!(star_result, BooleanWeight::one());
/// ```
///
/// ## Logical Operations
/// ```rust
/// use arcweight::prelude::*;
///
/// // De Morgan's laws can be implemented using semiring operations
/// let a = BooleanWeight::new(true);
/// let b = BooleanWeight::new(false);
///
/// // ¬(a ∧ b) ≡ ¬a ∨ ¬b (requires external negation function)
/// // Boolean semiring provides the base operations for logical reasoning
/// ```
///
/// # Integration with FST Algorithms
///
/// Boolean weights work optimally with all FST algorithms:
/// - **Composition:** Combines recognizers and transducers
/// - **Union:** Creates alternatives in recognition
/// - **Determinization:** Produces minimal recognizers
/// - **Shortest Path:** Finds any accepting path (since all have same weight)
///
/// # See Also
///
/// - [Core Concepts - Boolean Semiring](../../docs/core-concepts/semirings.md#boolean-semiring) for mathematical background
/// - [`TropicalWeight`](crate::semiring::TropicalWeight) for optimization problems
/// - [`compose()`](crate::algorithms::compose) for building complex recognizers from Boolean FSTs
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
