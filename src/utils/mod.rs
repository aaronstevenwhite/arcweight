//! Utility modules
//!
//! This module provides supporting utilities for FST operations,
//! including symbol tables, queues, and encoding tools.
//!
//! ## Components
//!
//! ### [`SymbolTable`] - Symbol Management
//! - Maps between symbols (strings) and numeric labels
//! - Bidirectional lookup for efficient FST storage
//! - Used in text I/O formats and human-readable representations
//!
//! ### Queue Types - Algorithm Support
//! - [`FifoQueue`] - First-in-first-out queue for breadth-first algorithms
//! - [`LifoQueue`] - Last-in-first-out (stack) for depth-first algorithms  
//! - [`StateQueue`] - Priority queue for state-based algorithms
//! - [`TopOrderQueue`] - Topologically ordered queue for optimization
//!
//! ### [`EncodeMapper`] - Arc Encoding
//! - Encodes FST arcs for space-efficient storage
//! - Supports different encoding strategies
//! - Used in compact representations and serialization
//!
//! ## Usage Examples
//!
//! ```
//! use arcweight::prelude::*;
//!
//! // Symbol table for text processing
//! let mut symbols = SymbolTable::new();
//! let hello_id = symbols.add_symbol("hello");
//! let world_id = symbols.add_symbol("world");
//!
//! // Use in FST construction
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! fst.set_start(s0);
//! fst.set_final(s1, TropicalWeight::one());
//! fst.add_arc(s0, Arc::new(hello_id, world_id, TropicalWeight::one(), s1));
//!
//! // Queue for algorithms
//! use arcweight::utils::Queue;
//! let mut queue = arcweight::utils::FifoQueue::new();
//! queue.enqueue(s0);
//! ```

mod encode;
mod queue;
mod symbol_table;

pub use encode::{EncodeMapper, EncodeType};
pub use queue::{FifoQueue, LifoQueue, Queue, StateQueue, TopOrderQueue};
pub use symbol_table::SymbolTable;
