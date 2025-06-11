//! # ArcWeight
//!
//! A high-performance, modular library for weighted finite state transducers (WFSTs).
//!
//! This library provides a comprehensive toolkit for constructing, combining, optimizing,
//! and searching weighted finite-state transducers. It supports all major semiring types
//! and provides a trait-based architecture for maximum extensibility.
//!
//! ## Features
//!
//! - **Comprehensive semiring support**: Tropical, probability, boolean, log, and more
//! - **Multiple FST implementations**: Vector, constant, compact, lazy, and cached
//! - **Full algorithm suite**: Composition, determinization, minimization, shortest path, etc.
//! - **OpenFST compatibility**: Read and write OpenFST format files
//! - **High performance**: Optimized implementations with optional parallelization
//! - **Type-safe**: Leverages Rust's type system for correctness
//!
//! ## Quick Start
//!
//! ```rust
//! use arcweight::prelude::*;
//!
//! // create a simple acceptor
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! let s2 = fst.add_state();
//!
//! fst.set_start(s0);
//! fst.set_final(s2, TropicalWeight::one());
//!
//! // add arcs
//! fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
//! fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(0.3), s2));
//!
//! // find shortest path
//! let config = ShortestPathConfig { nshortest: 1, ..Default::default() };
//! let shortest: VectorFst<TropicalWeight> = shortest_path(&fst, config).unwrap();
//! ```

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod algorithms;
pub mod arc;
pub mod fst;
pub mod io;
pub mod properties;
pub mod semiring;
pub mod utils;

pub mod prelude;

// re-export key items at crate root
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
