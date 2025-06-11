# ArcWeight

[![Crates.io](https://img.shields.io/crates/v/arcweight.svg)](https://crates.io/crates/arcweight)
[![Documentation](https://docs.rs/arcweight/badge.svg)](https://docs.rs/arcweight)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Build Status](https://github.com/aaronstevenwhite/arcweight/workflows/CI/badge.svg)](https://github.com/aaronstevenwhite/arcweight/actions)

A high-performance, modular Rust library for weighted finite state transducers (WFSTs).

ArcWeight provides a comprehensive toolkit for constructing, combining, optimizing, and searching weighted finite-state transducers. It offers functionality comparable to [OpenFst](https://www.openfst.org/) with a modern, type-safe Rust API.

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
- 🧪 **Comprehensive Benchmarking**: Performance testing across all major operations:
  - Core algorithms (composition, determinization, minimization)
  - Memory usage and storage efficiency
  - I/O operations and serialization
  - Parallel algorithm performance
  - Optimization algorithms (epsilon removal, weight pushing)

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
- [Examples](docs/examples/) - Complete, runnable examples with detailed documentation

## Examples

The `examples/` directory contains comprehensive, real-world applications demonstrating ArcWeight's capabilities:

### Text Processing & NLP
- **`edit_distance.rs`** - Compute edit distance with configurable weights and operations
  ```bash
  cargo run --example edit_distance
  ```

- **`word_correction.rs`** - Dictionary-based spell checking and word correction with fuzzy matching
  ```bash
  cargo run --example word_correction
  ```

### Speech & Phonetics
- **`pronunciation_lexicon.rs`** - Pronunciation dictionary with grapheme-to-phoneme (G2P) rules
  ```bash
  cargo run --example pronunciation_lexicon
  ```

- **`phonological_rules.rs`** - Phonological rule systems following Kaplan & Kay's framework
  ```bash
  cargo run --example phonological_rules
  ```

### Morphology & Linguistics
- **`morphological_analyzer.rs`** - Finite state morphology following Karttunen's two-level framework
  ```bash
  cargo run --example morphological_analyzer
  ```

- **`transliteration.rs`** - Cross-script transliteration (Cyrillic, Arabic, Greek to Latin)
  ```bash
  cargo run --example transliteration
  ```

### Text Normalization
- **`number_date_normalizer.rs`** - Text normalization for numbers, dates, and measurements
  ```bash
  cargo run --example number_date_normalizer
  ```

Each example includes detailed documentation in the [Examples Guide](docs/examples/) with theoretical background and implementation details.

## Performance & Benchmarking

ArcWeight is designed for high performance with extensive benchmarking:

- **Zero-copy arc iteration** - Minimal allocations in hot paths
- **Cache-friendly data structures** - Optimized memory layout
- **Parallel algorithms** - Optional Rayon-based parallelization
- **Comprehensive benchmarking suite** covering:
  - Core operations (composition, determinization, minimization)
  - Memory usage and storage efficiency
  - I/O and serialization performance
  - Parallel algorithm scaling
  - Optimization algorithms (epsilon removal, weight pushing)

Run benchmarks with:
```bash
cargo bench
```

The `benches/` directory contains organized benchmark suites:
- `core/` - Basic FST operations and algorithms
- `memory/` - Memory usage and storage benchmarks
- `io/` - Serialization and deserialization performance
- `optimization/` - Advanced optimization algorithms
- `parallel/` - Parallel processing benchmarks

## License

Licensed under [Apache 2.0](LICENSE).

## Development

### Code Quality

ArcWeight maintains high code quality standards with comprehensive tooling:

- **Formatting**: `rustfmt.toml` - Consistent code formatting (100 char width, Unix line endings)
- **Linting**: `clippy.toml` - Strict linting rules optimized for FST library patterns
- **Testing**: Extensive test suite with unit, integration, and property-based tests

### Development Commands

```bash
# Build and test
cargo build
cargo test
cargo test --integration

# Code quality
cargo fmt        # Format code
cargo clippy     # Run linter

# Performance
cargo bench      # Run benchmarks

# Examples
cargo run --example edit_distance
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](docs/contributing.md) for details.