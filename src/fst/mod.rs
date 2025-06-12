//! FST implementations
//! 
//! This module provides different FST (Finite State Transducer) implementations,
//! each optimized for specific use cases and performance characteristics.
//! 
//! ## FST Types
//! 
//! ### [`VectorFst`] - Mutable Vector FST
//! - **Use case**: General-purpose, dynamic FST construction
//! - **Performance**: Fast random access, good for small to medium FSTs
//! - **Memory**: Moderate memory usage, grows dynamically
//! - **Best for**: Building FSTs incrementally, frequent modifications
//! 
//! ### [`ConstFst`] - Immutable Const FST  
//! - **Use case**: Read-only FSTs with optimized storage
//! - **Performance**: Very fast traversal, excellent cache locality
//! - **Memory**: Minimal memory footprint, contiguous storage
//! - **Best for**: Large FSTs that don't change, production use
//! 
//! ### [`CompactFst`] - Memory-Efficient FST
//! - **Use case**: Large FSTs with limited memory
//! - **Performance**: Slower access due to compression
//! - **Memory**: Highly compressed representation
//! - **Best for**: Very large FSTs where memory is constrained
//! 
//! ### [`CacheFst`] - Caching Wrapper
//! - **Use case**: Expensive computations with repeated access
//! - **Performance**: Fast for repeated queries, slower for first access
//! - **Memory**: Additional cache overhead
//! - **Best for**: Wrapping lazy or remote FSTs
//! 
//! ### [`LazyFstImpl`] - On-Demand Computation
//! - **Use case**: FSTs computed dynamically
//! - **Performance**: Varies by computation complexity
//! - **Memory**: Only stores computed portions
//! - **Best for**: Large search spaces, composition chains
//! 
//! ## Choosing an FST Type
//! 
//! ```
//! use arcweight::prelude::*;
//! use arcweight::fst::CacheFst;
//! 
//! // For building and modifying FSTs
//! let mut mutable_fst = VectorFst::<TropicalWeight>::new();
//! mutable_fst.add_state();
//! 
//! // For read-only, high-performance access
//! let const_fst = ConstFst::from_fst(&mutable_fst)?;
//! 
//! // For memory-constrained environments  
//! // let compact_fst = CompactFst::new(); // requires complex type parameters
//! 
//! // For caching expensive operations
//! let cached_fst = CacheFst::new(const_fst);
//! # Ok::<(), arcweight::Error>(())
//! ```
//! 
//! ## Core Traits
//! 
//! All FST types implement the core [`Fst`] trait for read operations,
//! while mutable types also implement [`MutableFst`] for modifications.
//! Some types implement [`ExpandedFst`] for direct slice access to arcs.

mod cache_fst;
mod compact_fst;
mod const_fst;
mod lazy_fst;
mod traits;
mod vector_fst;

pub use cache_fst::CacheFst;
pub use compact_fst::CompactFst;
pub use const_fst::ConstFst;
pub use lazy_fst::LazyFstImpl;
pub use traits::*;
pub use vector_fst::VectorFst;
