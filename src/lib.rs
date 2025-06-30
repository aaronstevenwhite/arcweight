//! # ArcWeight - Weighted Finite State Transducers for Rust
//!
//! A high-performance, type-safe library for constructing, combining, optimizing,
//! and searching weighted finite state transducers (WFSTs) and automata.
//!
//! ## Overview
//!
//! ArcWeight provides a comprehensive toolkit for working with finite state machines,
//! from simple acceptors to complex weighted transducers. Built with performance and
//! correctness in mind, it leverages Rust's type system to ensure compile-time safety
//! while maintaining the flexibility needed for advanced FST operations.
//!
//! ## Key Features
//!
//! ### 🎯 Comprehensive FST Support
//! - **Multiple implementations:** [`fst::VectorFst`] for construction, [`fst::ConstFst`] for deployment
//! - **Lazy evaluation:** On-demand computation with [`fst::LazyFstImpl`]
//! - **Memory efficiency:** Compact representations and caching strategies
//!
//! ### ⚖️ Extensible Semiring Library
//! - **Wide variety of semirings**, including:
//!   - [`semiring::TropicalWeight`]
//!   - [`semiring::ProbabilityWeight`]
//!   - [`semiring::BooleanWeight`]
//!   - [`semiring::LogWeight`]
//!   - [`semiring::ProductWeight`]
//!   - [`semiring::StringWeight`]
//! - **Extensible framework:** Implement custom semirings via traits
//!
//! ### 🚀 Extensive Algorithm Library
//! - **Core operations:** [`algorithms::compose`], [`algorithms::concat`], [`algorithms::union`], [`algorithms::closure`]
//! - **Optimizations:** [`algorithms::minimize`], [`algorithms::determinize`], [`algorithms::remove_epsilons`]
//! - **Path algorithms:** [`algorithms::shortest_path`], [`algorithms::randgen`]
//! - **Advanced transforms:** [`algorithms::synchronize`], [`algorithms::push_weights`]
//!
//! ### 📊 Property Analysis
//! - **Automatic property tracking:** Detect determinism, cyclicity, connectivity
//! - **Optimization guidance:** Algorithm selection based on FST properties
//! - **Efficient computation:** $O(|V| + |E|)$ property analysis
//!
//! ## Quick Start
//!
//! ```rust
//! use arcweight::prelude::*;
//!
//! // Build a simple acceptor for the pattern "ab+"
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! let s2 = fst.add_state();
//!
//! fst.set_start(s0);
//! fst.set_final(s2, TropicalWeight::one());
//!
//! // Add transitions: 'a' -> 'b' -> 'b'*
//! fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::new(1.0), s1));
//! fst.add_arc(s1, Arc::new('b' as u32, 'b' as u32, TropicalWeight::new(0.5), s2));
//! fst.add_arc(s2, Arc::new('b' as u32, 'b' as u32, TropicalWeight::new(0.5), s2));
//!
//! // Find shortest accepting path
//! let shortest: VectorFst<TropicalWeight> = shortest_path(&fst, ShortestPathConfig::default())?;
//!
//! // Minimize the FST
//! let minimal: VectorFst<TropicalWeight> = minimize(&fst)?;
//! # Ok::<(), arcweight::Error>(())
//! ```
//!
//! ## Common Use Cases
//!
//! ### Building a Transducer
//!
//! ```
//! use arcweight::prelude::*;
//!
//! // Create a transducer that uppercases ASCII letters
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let state = fst.add_state();
//! fst.set_start(state);
//! fst.set_final(state, TropicalWeight::one());
//!
//! // Add lowercase->uppercase mappings
//! for c in b'a'..=b'z' {
//!     let lower = c as u32;
//!     let upper = (c - 32) as u32;  // ASCII uppercase
//!     fst.add_arc(state, Arc::new(lower, upper, TropicalWeight::one(), state));
//! }
//! ```
//!
//! ### Composing FSTs
//!
//! ```
//! use arcweight::prelude::*;
//!
//! // Compose two transducers to create a pipeline
//! fn create_pipeline() -> Result<VectorFst<TropicalWeight>> {
//!     let fst1 = VectorFst::<TropicalWeight>::new();  // First transducer
//!     let fst2 = VectorFst::<TropicalWeight>::new();  // Second transducer
//!     
//!     // Compose: output of fst1 feeds into input of fst2
//!     compose_default(&fst1, &fst2)
//! }
//! ```
//!
//! ## Module Organization
//!
//! The library is organized into focused modules, each handling a specific aspect
//! of FST functionality:
//!
//! ### Core Modules
//!
//! - [`arc`] - Arc types representing FST transitions
//! - [`fst`] - FST implementations and traits
//! - [`semiring`] - Weight types and algebraic operations
//!
//! ### Algorithm Modules
//!
//! - [`algorithms`] - FST algorithms and transformations
//! - [`properties`] - Property computation and analysis
//!
//! ### Support Modules
//!
//! - [`io`] - Serialization and file format support
//! - [`utils`] - Symbol tables and utility types
//! - [`prelude`] - Convenient re-exports of common types
//!
//! ## Performance Guidelines
//!
//! 1. **Choose the right FST type:** Use [`fst::VectorFst`] for construction, [`fst::ConstFst`] for deployment
//! 2. **Leverage properties:** Let the library detect and optimize based on FST properties
//! 3. **Batch operations:** Combine multiple operations when possible
//! 4. **Use lazy evaluation:** For large FSTs, consider lazy implementations
//!
//! ## Advanced Topics
//!
//! - **Custom Semirings:** Implement the [`Semiring`] trait for domain-specific weights
//! - **Lazy Computation:** Use [`fst::LazyFstImpl`] for on-demand arc generation
//! - **Memory Optimization:** Employ [`fst::CompactFst`] for memory-constrained environments
//! - **Parallel Algorithms:** Some algorithms support parallel execution (feature-gated)

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

pub mod algorithms;
pub mod arc;
pub mod fst;
pub mod io;
pub mod properties;
pub mod semiring;
pub mod utils;

pub mod prelude;

// Re-export key items at crate root
pub use arc::{Arc, ArcIterator};
pub use fst::{ExpandedFst, Fst, MutableFst, VectorFst};
pub use semiring::{BooleanWeight, ProbabilityWeight, Semiring, TropicalWeight};

/// Library-wide error type
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Invalid FST operation
    #[error("Invalid FST operation: {0}")]
    InvalidOperation(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Algorithm-specific error
    #[error("Algorithm error: {0}")]
    Algorithm(String),
}

/// Library-wide result type
pub type Result<T> = std::result::Result<T, Error>;
