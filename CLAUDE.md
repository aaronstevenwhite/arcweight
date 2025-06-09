# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Building and Testing
- `cargo build` - Build the library
- `cargo test` - Run all tests
- `cargo test --integration` - Run integration tests only
- `cargo test property_tests` - Run property-based tests
- `cargo bench` - Run performance benchmarks
- `cargo clippy` - Run linter
- `cargo fmt` - Format code

### Development
- `cargo run --example basic_usage` - Run basic usage example
- `cargo run --example composition` - Run composition example
- `cargo run --example shortest_path` - Run shortest path example
- `cargo run --example speech_recognition` - Run speech recognition example

## Architecture Overview

ArcWeight is a weighted finite state transducer (WFST) library with a trait-based architecture:

### Core Design Patterns
- **Trait-based FST types**: All FST implementations conform to `Fst`, `MutableFst`, and `ExpandedFst` traits
- **Generic semiring support**: Algorithms work with any type implementing `Semiring` trait
- **Zero-copy iteration**: Arc iteration avoids allocations in hot paths
- **Optional parallelization**: Rayon-based parallel algorithms enabled via `parallel` feature

### Key Module Structure
- `fst/` - FST trait definitions and implementations (VectorFst, ConstFst, CompactFst, LazyFst, CacheFst)
- `semiring/` - Weight types (TropicalWeight, ProbabilityWeight, BooleanWeight, LogWeight, etc.)
- `algorithms/` - Core FST operations (compose, determinize, minimize, shortest_path, etc.)
- `arc/` - Arc representation and iteration
- `io/` - File I/O including OpenFST compatibility
- `utils/` - Supporting utilities like SymbolTable

### Important Implementation Details
- StateId and Label are type aliases for u32
- NO_STATE_ID and NO_LABEL constants represent invalid/epsilon values
- All algorithms return `Result<T>` with library-specific `Error` type
- The prelude module provides convenient imports for common usage

### Testing Strategy
- Unit tests alongside each module
- Integration tests in `tests/` directory  
- Property-based tests using proptest for algorithm correctness
- Benchmarks in `benches/` directory for performance validation