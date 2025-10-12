//! Convenient re-exports of commonly used types and functions
//!
//! The prelude module provides a curated collection of the most frequently used
//! items from ArcWeight, allowing you to get started quickly with a single import:
//!
//! ```
//! use arcweight::prelude::*;
//! ```
//!
//! ## What's Included
//!
//! ### Core Types
//!
//! - **FST Types:** [`VectorFst`], [`ConstFst`], [`CompactFst`]
//! - **Arc Type:** [`Arc`] for representing transitions
//! - **Traits:** [`Fst`], [`MutableFst`], [`ExpandedFst`]
//! - **Identifiers:** [`StateId`], [`Label`], [`NO_STATE_ID`], [`NO_LABEL`]
//!
//! ### Semirings
//!
//! All major semiring types and traits:
//! - **Common weights:** [`TropicalWeight`], [`ProbabilityWeight`], [`BooleanWeight`]
//! - **Additional weights:** [`LogWeight`], [`RealWeight`], [`ProductWeight`], [`StringWeight`]
//! - **Semiring traits:** [`Semiring`], [`StarSemiring`], [`DivisibleSemiring`]
//!
//! ### Algorithms
//!
//! Essential FST algorithms:
//! - **Core operations:** [`compose()`], [`concat()`], [`union()`], [`closure()`]
//! - **Optimizations:** [`minimize()`], [`determinize()`], [`remove_epsilons()`]
//! - **Path algorithms:** [`shortest_path()`], [`shortest_path_single()`]
//! - **Transformations:** [`reverse()`], [`project_input()`], [`project_output()`]
//!
//! ### I/O Operations
//!
//! File format support:
//! - **Text format:** [`read_text()`], [`write_text()`]
//! - **OpenFST format:** [`read_openfst()`], [`write_openfst()`]
//! - **Binary format:** [`read_binary()`], [`write_binary()`] (with serde feature)
//!
//! ### Properties and Utilities
//!
//! - **Property analysis:** [`compute_properties()`], [`FstProperties`], [`PropertyFlags`]
//! - **Symbol tables:** [`SymbolTable`] for label-to-string mappings
//! - **Error handling:** [`Error`], [`Result`] types
//! - **Numeric traits:** [`Zero`], [`One`] from num-traits
//!
//! ## Usage Examples
//!
//! ### Basic FST Construction
//!
//! ```
//! use arcweight::prelude::*;
//!
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//!
//! fst.set_start(s0);
//! fst.set_final(s1, TropicalWeight::one());
//! fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));
//! ```
//!
//! ### Algorithm Chaining
//!
//! ```
//! use arcweight::prelude::*;
//!
//! fn process_fst(fst: &VectorFst<TropicalWeight>) -> Result<VectorFst<TropicalWeight>> {
//!     // Chain multiple operations
//!     let deterministic: VectorFst<TropicalWeight> = determinize(fst)?;
//!     let minimal: VectorFst<TropicalWeight> = minimize(&deterministic)?;
//!     let reversed: VectorFst<TropicalWeight> = reverse(&minimal)?;
//!     Ok(reversed)
//! }
//! ```
//!
//! ### Working with Different Semirings
//!
//! ```
//! use arcweight::prelude::*;
//!
//! // Boolean semiring for unweighted FSTs
//! let bool_fst = VectorFst::<BooleanWeight>::new();
//!
//! // Probability semiring for probabilistic models
//! let prob_fst = VectorFst::<ProbabilityWeight>::new();
//!
//! // Log semiring for numerical stability
//! let log_fst = VectorFst::<LogWeight>::new();
//! ```
//!
//! ## Design Philosophy
//!
//! The prelude is designed to include items that are:
//! 1. **Frequently used:** Core types needed in most FST programs
//! 2. **Unambiguous:** No naming conflicts with common Rust items
//! 3. **Essential:** Fundamental to working with the library
//!
//! More specialized items remain in their respective modules to avoid
//! namespace pollution while keeping the prelude focused and ergonomic.

pub use num_traits::{One, Zero};

pub use crate::{
    // algorithms
    algorithms::{
        arc_sum, arc_unique, closure, closure_plus, compose, compose_default, concat, connect,
        determinize, minimize, project_input, project_output, prune, remove_epsilons, reverse,
        shortest_distance, shortest_path, shortest_path_single, topsort, union, weight_convert,
        ComposeFilter, DefaultComposeFilter, ShortestPathConfig,
    },

    // core types
    arc::{Arc, ArcIterator},
    // fst implementations
    fst::{CompactFst, ConstFst, VectorFst},

    fst::{ExpandedFst, Fst, Label, MutableFst, StateId, NO_LABEL, NO_STATE_ID},

    // i/o
    io::{read_openfst, read_text, write_openfst, write_text},

    // optimization
    optimization::{
        optimize_for_performance, prefetch_cache_line, AccessPattern, ArcPool, CacheMetadata,
        OptimizationRecommendation, OptimizedFst, SimdOps,
    },

    // properties
    properties::{compute_properties, FstProperties, PropertyFlags},

    // semirings
    semiring::{
        BooleanWeight, DivisibleSemiring, InvertibleSemiring, LogWeight, MaxWeight, MinWeight,
        NaturallyOrderedSemiring, ProbabilityWeight, ProductWeight, RealWeight, Semiring,
        SemiringProperties, StarSemiring, StringWeight, TropicalWeight,
    },

    // utilities
    utils::SymbolTable,

    // error handling
    Error,
    Result,
};

#[cfg(feature = "serde")]
pub use crate::io::{read_binary, write_binary};
