//! FST algorithms module
//!
//! This module provides a comprehensive collection of algorithms for manipulating
//! and analyzing weighted finite state transducers (FSTs). The algorithms are
//! organized into several categories:
//!
//! ## Core Operations
//!
//! - [`compose()`] - Composition of two FSTs
//! - [`concat()`] - Concatenation of FSTs
//! - [`union()`] - Union of multiple FSTs
//! - [`closure()`], [`closure_plus()`] - Kleene star and plus operations
//! - [`intersect()`] - Intersection of acceptor languages
//! - [`difference()`] - Difference of acceptor languages (A₁ - A₂)
//!
//! ## Optimization Algorithms
//!
//! - [`minimize()`] - State minimization using Brzozowski's algorithm
//! - [`determinize()`] - Convert non-deterministic FST to deterministic
//! - [`remove_epsilons()`] - Remove epsilon transitions
//! - [`connect()`] - Remove non-accessible/coaccessible states
//!
//! ## Path Algorithms
//!
//! - [`shortest_path()`] - Find shortest paths through FST
//! - [`randgen()`] - Generate random paths from FST
//!
//! ## Transformation Algorithms
//!
//! - [`reverse()`] - Reverse the FST direction
//! - [`project_input()`], [`project_output()`] - Project to input/output labels
//! - [`synchronize()`] - Synchronize transducer labels
//! - [`push_weights()`], [`push_labels()`] - Push weights toward initial/final states
//!
//! ## Utility Algorithms
//!
//! - [`prune()`] - Prune arcs/states based on weight thresholds
//! - [`replace()`] - Replace symbols with sub-FSTs
//! - [`topsort()`] - Topological sort of states
//! - [`weight_convert()`] - Convert between semiring types
//!
//! ## Usage Examples
//!
//! Most algorithms follow the pattern of taking input FST(s) and configuration,
//! returning a new FST with the operation applied:
//!
//! ```
//! use arcweight::prelude::*;
//!
//! // Create a simple FST
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! fst.set_start(s0);
//! fst.set_final(s1, TropicalWeight::one());
//!
//! // Apply algorithms
//! let minimized: VectorFst<TropicalWeight> = minimize(&fst)?;
//! let shortest: VectorFst<TropicalWeight> = shortest_path(&fst, ShortestPathConfig::default())?;
//! let reversed: VectorFst<TropicalWeight> = reverse(&fst)?;
//! # Ok::<(), arcweight::Error>(())
//! ```

mod closure;
mod compose;
mod concat;
mod connect;
mod determinize;
mod difference;
mod intersect;
mod minimize;
mod project;
mod prune;
mod push;
mod randgen;
mod replace;
mod reverse;
mod rmepsilon;
mod shortest_path;
mod synchronize;
mod topsort;
mod union;
mod weight_convert;

pub use closure::{closure, closure_plus};
pub use compose::{compose, compose_default, ComposeFilter, DefaultComposeFilter};
pub use concat::concat;
pub use connect::connect;
pub use determinize::determinize;
pub use difference::difference;
pub use intersect::intersect;
pub use minimize::minimize;
pub use project::{project_input, project_output};
pub use prune::{prune, PruneConfig};
pub use push::{push_labels, push_weights};
pub use randgen::{randgen, RandGenConfig};
pub use replace::{replace, ReplaceFst};
pub use reverse::reverse;
pub use rmepsilon::remove_epsilons;
pub use shortest_path::{shortest_path, shortest_path_single, ShortestPathConfig};
pub use synchronize::synchronize;
pub use topsort::topsort;
pub use union::union;
pub use weight_convert::weight_convert;
