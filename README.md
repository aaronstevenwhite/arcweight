# ArcWeight

[![Crates.io](https://img.shields.io/crates/v/arcweight.svg)](https://crates.io/crates/arcweight)
[![Documentation](https://docs.rs/arcweight/badge.svg)](https://docs.rs/arcweight)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Build Status](https://github.com/yourusername/arcweight/workflows/CI/badge.svg)](https://github.com/yourusername/arcweight/actions)

A high-performance, modular Rust library for weighted finite state transducers (WFSTs).

ArcWeight provides a comprehensive toolkit for constructing, combining, optimizing, and searching weighted finite-state transducers. It offers functionality comparable to OpenFST with a modern, type-safe Rust API.

## Features

- 🚀 **High Performance**: Optimized implementations with parallel algorithms support
- 🔧 **Modular Design**: Trait-based architecture for maximum extensibility
- 📊 **Rich Semiring Support**: 
  - Tropical, Probability, Boolean, Log semirings
  - String, MinMax, and Product semirings
  - Custom semiring implementations
- 🗄️ **Multiple FST Types**: Vector, constant, compact, lazy, and cached implementations
- 🔄 **Comprehensive Algorithm Suite**:
  - Core operations: composition, determinization, minimization
  - Advanced operations: push, prune, project, replace
  - Path operations: shortest path, random path generation
  - Graph operations: connect, synchronize, topsort
- 📁 **OpenFST Compatible**: Read and write OpenFST format files
- 🦀 **Pure Rust**: No C++ dependencies, fully memory safe
- 🧪 **Extensive Benchmarks**: Performance testing for core operations, memory usage, and parallel algorithms

## Quick Start

Add ArcWeight to your `Cargo.toml`:

```toml
[dependencies]
arcweight = "0.1"
```

Basic example:

```rust
use arcweight::prelude::*;

// Create a simple acceptor
let mut fst = VectorFst::<TropicalWeight>::new();
let s0 = fst.add_state();
let s1 = fst.add_state();
let s2 = fst.add_state();

fst.set_start(s0);
fst.set_final(s2, TropicalWeight::one());

// Add weighted transitions
fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(0.5), s1));
fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(0.3), s2));

// Find the shortest path
let shortest = shortest_path(&fst)?;
```

## Documentation

- [User Guide](docs/guide.md) - Comprehensive guide to using ArcWeight
- [API Reference](https://docs.rs/arcweight) - Detailed API documentation
- [Examples](examples/) - Complete, runnable examples

## Examples

Check out the examples/ directory for more complex usage:

- `basic_usage.rs` - Getting started with FSTs
- `composition.rs` - Composing transducers
- `shortest_path.rs` - Finding optimal paths
- `speech_recognition.rs` - Real-world speech processing example

## Performance

ArcWeight is designed for high performance:

- Zero-copy arc iteration
- Cache-friendly data structures
- Parallel algorithms via Rayon
- Minimal allocations in hot paths
- Comprehensive benchmarking suite

Run benchmarks with:
```bash
cargo bench
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

## Contributing

Contributions are welcome! Please read our Contributing Guide for details.

## Acknowledgments

This library is inspired by the excellent OpenFST library by Cyril Allauzen, Michael Riley, Johan Schalkwyk, Wojciech Skut, and Mehryar Mohri.