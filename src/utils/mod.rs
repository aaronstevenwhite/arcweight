//! Utility types and data structures for FST operations
//!
//! This module provides essential supporting utilities that are used throughout
//! the ArcWeight library. These utilities handle symbol management, state exploration
//! queues, and arc encoding for efficient FST manipulation.
//!
//! ## Core Components
//!
//! ### [`SymbolTable`] - Symbol-to-Label Mapping
//!
//! Manages bidirectional mapping between human-readable symbols and numeric labels:
//! - **Purpose:** Convert between strings and FST label integers
//! - **Features:** O(1) lookups, automatic ID assignment, symbol iteration
//! - **Use cases:** Text I/O, debugging, linguistic applications
//!
//! ```
//! use arcweight::utils::SymbolTable;
//!
//! let mut symbols = SymbolTable::new();
//!
//! // Add symbols and get their IDs
//! let cat_id = symbols.add_symbol("cat");
//! let dog_id = symbols.add_symbol("dog");
//!
//! // Bidirectional lookup
//! assert_eq!(symbols.find_id("cat"), Some(cat_id));
//! assert_eq!(symbols.find(cat_id), Some("cat"));
//!
//! // Special symbols
//! assert_eq!(symbols.find(0), Some("<eps>"));  // Epsilon
//! ```
//!
//! ### Queue Types - Algorithm State Exploration
//!
//! Different queue implementations for various FST traversal strategies:
//!
//! #### [`FifoQueue`] - Breadth-First Search
//! - **Order:** First-In-First-Out
//! - **Properties:** Explores states level by level
//! - **Use cases:** Shortest path, reachability analysis
//! - **Algorithms:** Dijkstra's algorithm, BFS-based operations
//!
//! #### [`LifoQueue`] - Depth-First Search  
//! - **Order:** Last-In-First-Out (stack)
//! - **Properties:** Explores deeply before backtracking
//! - **Use cases:** Cycle detection, topological sort
//! - **Algorithms:** DFS-based operations, path enumeration
//!
//! #### [`StateQueue`] - Priority Queue
//! - **Order:** Priority-based (customizable)
//! - **Properties:** Processes states by importance/cost
//! - **Use cases:** A* search, best-first algorithms
//! - **Algorithms:** Weighted shortest path, pruning
//!
//! #### [`TopOrderQueue`] - Topological Order
//! - **Order:** Respects state dependencies
//! - **Properties:** Processes states in topological order
//! - **Use cases:** Acyclic FST optimization
//! - **Algorithms:** Dynamic programming on DAGs
//!
//! ### [`EncodeMapper`] - Arc Compression
//!
//! Provides arc encoding for memory-efficient FST storage:
//! - **Purpose:** Reduce memory footprint of FST arcs
//! - **Strategies:** Label encoding, weight quantization
//! - **Use cases:** Large FSTs, mobile deployment
//!
//! ## Usage Examples
//!
//! ### Building FSTs with Symbol Tables
//!
//! ```
//! use arcweight::prelude::*;
//! use arcweight::utils::SymbolTable;
//!
//! // Create a word-to-phoneme transducer
//! let mut words = SymbolTable::new();
//! let mut phones = SymbolTable::new();
//!
//! // Add vocabulary
//! let hello_id = words.add_symbol("hello");
//! let world_id = words.add_symbol("world");
//! let h_id = phones.add_symbol("h");
//! let eh_id = phones.add_symbol("eh");
//! let l_id = phones.add_symbol("l");
//! let ow_id = phones.add_symbol("ow");
//!
//! // Build FST with meaningful labels
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! fst.set_start(s0);
//! fst.add_arc(s0, Arc::new(hello_id, h_id, TropicalWeight::one(), s1));
//! ```
//!
//! ### Queue Selection for Algorithms
//!
//! ```
//! use arcweight::prelude::*;
//! use arcweight::utils::{Queue, FifoQueue, LifoQueue};
//!
//! fn explore_fst<Q: Queue>(fst: &impl Fst<TropicalWeight>, mut queue: Q) {
//!     if let Some(start) = fst.start() {
//!         queue.enqueue(start);
//!         
//!         while let Some(state) = queue.dequeue() {
//!             // Process state
//!             for arc in fst.arcs(state) {
//!                 queue.enqueue(arc.nextstate);
//!             }
//!         }
//!     }
//! }
//!
//! let fst = VectorFst::<TropicalWeight>::new();
//!
//! // Breadth-first exploration
//! explore_fst(&fst, FifoQueue::new());
//!
//! // Depth-first exploration  
//! explore_fst(&fst, LifoQueue::new());
//! ```
//!
//! ### Advanced Symbol Table Usage
//!
//! ```
//! use arcweight::utils::SymbolTable;
//! use std::fs::File;
//! use std::io::{BufRead, BufReader};
//!
//! // Load symbols from file
//! fn load_symbols(filename: &str) -> std::io::Result<SymbolTable> {
//!     let mut symbols = SymbolTable::new();
//!     let file = File::open(filename)?;
//!     let reader = BufReader::new(file);
//!     
//!     for line in reader.lines() {
//!         let symbol = line?;
//!         symbols.add_symbol(&symbol);
//!     }
//!     
//!     Ok(symbols)
//! }
//! ```
//!
//! ## Performance Considerations
//!
//! ### Symbol Tables
//! - **Memory:** O(n) where n is the number of symbols
//! - **Lookup:** O(1) average case for both directions
//! - **Insertion:** O(1) amortized
//!
//! ### Queues
//! - **FifoQueue:** O(1) enqueue/dequeue
//! - **LifoQueue:** O(1) push/pop
//! - **StateQueue:** O(log n) operations
//! - **TopOrderQueue:** O(1) with preprocessing
//!
//! ## Design Patterns
//!
//! ### Generic Queue Interface
//!
//! All queue types implement the [`Queue`] trait, allowing generic algorithms:
//!
//! ```
//! use arcweight::utils::Queue;
//!
//! fn generic_search<Q: Queue>(queue: &mut Q, start: u32) {
//!     queue.enqueue(start);
//!     while let Some(state) = queue.dequeue() {
//!         // Process state...
//!     }
//! }
//! ```
//!
//! ### Symbol Table Best Practices
//!
//! 1. **Reserve special symbols:** ID 0 for epsilon, low IDs for special tokens
//! 2. **Consistent tables:** Use same symbol table for related FSTs
//! 3. **Persistence:** Save/load symbol tables with FSTs for reproducibility

mod encode;
mod queue;
mod symbol_table;

pub use encode::{EncodeMapper, EncodeType};
pub use queue::{FifoQueue, LifoQueue, Queue, StateQueue, TopOrderQueue};
pub use symbol_table::SymbolTable;
