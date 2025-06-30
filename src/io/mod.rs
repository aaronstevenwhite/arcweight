//! FST serialization and file format support
//!
//! This module provides comprehensive I/O capabilities for reading and writing FSTs
//! in various formats. It supports interoperability with other FST libraries and
//! provides both human-readable and efficient binary formats.
//!
//! ## Supported Formats
//!
//! ### Text Format - Human-Readable Representation
//!
//! - **Functions:** [`read_text()`], [`write_text()`]
//! - **Features:** Easy to debug, version control friendly, editable
//! - **Use cases:** Small FSTs, debugging, manual FST creation
//! - **Format:**
//!   ```text
//!   0 1 a b 0.5
//!   1 2 c d 0.3
//!   2 0.0
//!   ```
//!
//! ### OpenFST Binary Format - Industry Standard
//!
//! - **Functions:** [`read_openfst()`], [`write_openfst()`]
//! - **Features:** Compatible with OpenFST library, efficient storage
//! - **Use cases:** Interoperability with OpenFST tools, production systems
//! - **Supports:** All standard FST types and semirings
//!
//! ### Binary Format - Native Serialization
//!
//! - **Functions:** [`read_binary()`], [`write_binary()`] (requires `serde` feature)
//! - **Features:** Fast serialization, preserves all Rust-specific properties
//! - **Use cases:** Caching, distributed systems, persistence
//! - **Advantages:** Type-safe, versioned, compressed options
//!
//! ## Usage Examples
//!
//! ### Reading and Writing Text Format
//!
//! ```no_run
//! use arcweight::prelude::*;
//! use std::fs::File;
//! use std::io::{BufReader, BufWriter};
//!
//! // Write FST to text file
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! let s0 = fst.add_state();
//! let s1 = fst.add_state();
//! fst.set_start(s0);
//! fst.set_final(s1, TropicalWeight::new(0.0));
//! fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(0.5), s1));
//!
//! let file = File::create("fst.txt").unwrap();
//! let mut writer = BufWriter::new(file);
//! write_text(&fst, &mut writer, None, None).unwrap();
//!
//! // Read FST from text file
//! let file = File::open("fst.txt").unwrap();
//! let mut reader = BufReader::new(file);
//! let loaded_fst: VectorFst<TropicalWeight> = read_text(&mut reader, None, None).unwrap();
//! ```
//!
//! ### OpenFST Compatibility
//!
//! ```no_run
//! use arcweight::prelude::*;
//! use std::fs::File;
//!
//! // Read FST created by OpenFST tools
//! let mut file = File::open("model.fst").unwrap();
//! let fst: VectorFst<TropicalWeight> = read_openfst(&mut file).unwrap();
//!
//! // Process with ArcWeight algorithms
//! let minimized: VectorFst<TropicalWeight> = minimize(&fst).unwrap();
//!
//! // Write back in OpenFST format
//! let mut out_file = File::create("minimized.fst").unwrap();
//! write_openfst(&minimized, &mut out_file).unwrap();
//! ```
//!
//! ### Binary Serialization (with serde feature)
//!
//! ```
//! # #[cfg(feature = "serde")]
//! # {
//! use arcweight::prelude::*;
//! use arcweight::io::{write_binary, read_binary};
//! use std::io::Cursor;
//!
//! # fn example() -> Result<()> {
//! // Serialize FST to in-memory buffer
//! let fst = VectorFst::<LogWeight>::new();
//! let mut buffer = Vec::new();
//! write_binary(&fst, &mut buffer)?;
//!
//! // Deserialize from buffer
//! let mut cursor = Cursor::new(buffer);
//! let loaded: VectorFst<LogWeight> = read_binary(&mut cursor)?;
//! # Ok(())
//! # }
//! # }
//! ```
//!
//! ## Format Selection Guide
//!
//! | Format | Speed | Size | Compatibility | Human-Readable | Use Case |
//! |--------|-------|------|---------------|----------------|----------|
//! | Text | Slow | Large | Universal | Yes | Debugging, small FSTs |
//! | OpenFST | Fast | Medium | OpenFST tools | No | Production, interop |
//! | Binary | Fastest | Small | ArcWeight only | No | Caching, storage |
//!
//! ## Advanced Features
//!
//! ### Symbol Table Integration
//!
//! ```no_run
//! use arcweight::prelude::*;
//! use arcweight::utils::SymbolTable;
//!
//! // Create FST with symbol table
//! let mut symbols = SymbolTable::new();
//! let hello = symbols.add_symbol("hello");
//! let world = symbols.add_symbol("world");
//!
//! let mut fst = VectorFst::<TropicalWeight>::new();
//! // ... build FST using symbol IDs ...
//!
//! // Symbol tables can be saved/loaded separately
//! // This allows human-readable arc labels in text format
//! ```
//!
//! ### Streaming Large FSTs
//!
//! For very large FSTs, consider:
//! - Using memory-mapped files with OpenFST format
//! - Streaming processing with lazy FST implementations
//! - Compressed binary formats with the serde feature
//!
//! ## Error Handling
//!
//! All I/O operations return `Result<T, arcweight::Error>` for proper error handling:
//! - File format errors
//! - Incompatible semiring types
//! - Corrupted data
//! - I/O failures

mod binary_format;
mod openfst_compat;
mod text_format;

#[cfg(feature = "serde")]
pub use binary_format::{read_binary, write_binary};
pub use openfst_compat::{read_openfst, write_openfst};
pub use text_format::{read_text, write_text};
