//! Semiring implementations for weighted FSTs
//!
//! This module provides various semiring implementations that define the algebraic
//! structure for FST weights. Each semiring defines two operations (⊕ and ⊗) with
//! specific properties needed for FST algorithms.
//!
//! ## Available Semirings
//!
//! ### [`TropicalWeight`] - Tropical Semiring (min, +)
//! - **Addition:** minimum operation (a ⊕ b = min(a, b))
//! - **Multiplication:** regular addition (a ⊗ b = a + b)
//! - **Zero:** +∞, **One:** 0.0
//! - **Use case:** Shortest path problems, Viterbi decoding
//! - **Example:** Cost minimization, edit distance
//!
//! ### [`LogWeight`] - Log Semiring (log-add, +)
//! - **Addition:** log(exp(a) + exp(b))
//! - **Multiplication:** regular addition (a ⊗ b = a + b)
//! - **Zero:** +∞, **One:** 0.0
//! - **Use case:** Probabilistic models in log domain
//! - **Example:** Speech recognition, machine translation
//!
//! ### [`ProbabilityWeight`] - Probability Semiring (+, ×)
//! - **Addition:** regular addition (a ⊕ b = a + b)
//! - **Multiplication:** regular multiplication (a ⊗ b = a × b)
//! - **Zero:** 0.0, **One:** 1.0
//! - **Use case:** Direct probability computations
//! - **Example:** Language models, probability estimation
//!
//! ### [`RealWeight`] - Real Number Semiring (+, ×)
//! - **Addition:** regular addition (a ⊕ b = a + b)
//! - **Multiplication:** regular multiplication (a ⊗ b = a × b)
//! - **Zero:** 0.0, **One:** 1.0
//! - **Use case:** Linear algebra, normalization, matrix operations
//! - **Example:** Bayesian inference, signal processing, financial modeling
//!
//! ### [`BooleanWeight`] - Boolean Semiring (∨, ∧)
//! - **Addition:** logical OR (a ⊕ b = a ∨ b)
//! - **Multiplication:** logical AND (a ⊗ b = a ∧ b)
//! - **Zero:** false, **One:** true
//! - **Use case:** Unweighted automata, membership testing
//! - **Example:** Regular expressions, pattern matching
//!
//! ### [`StringWeight`] - String Semiring (LCP, concat)
//! - **Addition:** longest common prefix
//! - **Multiplication:** string concatenation
//! - **Zero:** special marker, **One:** empty string
//! - **Use case:** String operations, edit sequences
//! - **Example:** Computing edit scripts
//!
//! ### [`MinWeight`] / [`MaxWeight`] - MinMax Semirings
//! - **Addition:** min/max operation
//! - **Multiplication:** min/max operation
//! - **Use case:** Optimization problems
//!
//! ### [`ProductWeight`] - Product of Semirings
//! - Combines two semirings into a Cartesian product
//! - **Use case:** Multi-objective optimization
//!
//! ### [`GallicWeight`] - Gallic Semiring
//! - Combines label sequences with weights
//! - **Multiple variants:** LeftGallic, RightGallic, MinGallic, RestrictGallic, UnionGallic
//! - **Use case:** FST composition, output label tracking
//! - **Example:** Advanced transducer algorithms, label-weighted shortest paths
//!
//! ## Choosing a Semiring
//!
//! ```
//! use arcweight::prelude::*;
//!
//! // For shortest path / minimum cost
//! let tropical = TropicalWeight::new(0.5);
//!
//! // For probabilistic models (log domain)
//! let log_prob = LogWeight::new(-2.3); // log probability
//!
//! // For direct probabilities
//! let prob = ProbabilityWeight::new(0.8);
//!
//! // For unweighted automata
//! let boolean = BooleanWeight::new(true);
//!
//! // Arithmetic follows semiring properties
//! let sum = tropical + TropicalWeight::new(0.3); // min(0.5, 0.3) = 0.3
//! let product = tropical * TropicalWeight::new(0.2); // 0.5 + 0.2 = 0.7
//! ```
//!
//! ## Semiring Properties
//!
//! All semirings implement the [`Semiring`] trait and satisfy:
//! - Associativity: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
//! - Commutativity: a ⊕ b = b ⊕ a (exception: StringWeight multiplication is non-commutative)
//! - Distributivity: a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)
//! - Identity elements: 0̄ (additive), 1̄ (multiplicative)
//!
//! Some semirings provide additional properties like [`DivisibleSemiring`]
//! for algorithms that require division operations.

mod boolean;
pub mod gallic;
mod integer;
mod log;
mod minmax;
mod probability;
mod product;
mod real;
mod string;
mod traits;
mod tropical;

pub use boolean::BooleanWeight;
pub use integer::IntegerWeight;
pub use log::LogWeight;
pub use minmax::{MaxWeight, MinWeight};
pub use probability::ProbabilityWeight;
pub use product::ProductWeight;
pub use real::RealWeight;
pub use string::StringWeight;
pub use traits::*;
pub use tropical::TropicalWeight;

// Re-export gallic types for convenience
pub use gallic::{
    GallicWeight, LeftGallic, LeftGallicWeight, MinGallic, MinGallicWeight, RestrictGallic,
    RestrictGallicWeight, RightGallic, RightGallicWeight, StandardGallicWeight, UnionGallic,
};
