# ArcWeight

[![Crates.io](https://img.shields.io/crates/v/arcweight.svg)](https://crates.io/crates/arcweight)
[![Documentation](https://docs.rs/arcweight/badge.svg)](https://docs.rs/arcweight)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Build Status](https://github.com/aaronstevenwhite/arcweight/workflows/CI/badge.svg)](https://github.com/aaronstevenwhite/arcweight/actions)

**A high-performance, modular Rust library for weighted finite state transducers (WFSTs).**

ArcWeight provides a comprehensive toolkit for constructing, combining, optimizing, and searching weighted finite-state transducers. It offers functionality comparable to [OpenFst](https://www.openfst.org/) with a modern, type-safe Rust API, making it ideal for natural language processing and computational linguistics applications.

## What Are Finite State Transducers?

Finite State Transducers (FSTs) are computational models that define rational relations between strings, enabling the transformation of input sequences into output sequences with associated weights ([Mohri 1997](https://aclanthology.org/J97-2003/)). FSTs have become fundamental tools in computational linguistics and natural language processing for:

- Computing the distance between strings
- Spell checking and correcting text
- Normalizing dates, times, etc.
- Analyzing and generating words

**Why choose ArcWeight over alternatives?**
- 🦀 **Pure Rust**: Memory safety, modern tooling, no C++ dependencies
- ⚡ **Performance**: Optimized algorithms implementing state-of-the-art techniques ([Mohri 2002](https://doi.org/10.1016/S0304-3975(99)00014-6))
- 🔧 **Flexibility**: Extensible design with custom semirings and FST types
- 📚 **Complete**: Comprehensive algorithm suite based on established theoretical foundations

## Comparison with rustfst

| Feature | ArcWeight | rustfst |
|---------|-----------|---------|
| **Primary Focus** | General weighted transducers for NLP | Full OpenFST reimplementation |
| **Use Case** | NLP, speech, linguistics applications | Research, OpenFST migration |
| **Architecture** | Clean, modular trait system | Complex, feature-rich |
| **API Complexity** | Accessible, well-documented | Industrial-strength, academic |
| **FST Types** | Vector, Const, Compact, Lazy, Cache | VectorFst, ConstFst |
| **Semirings** | 8+ types with extensibility | 10+ types |
| **Algorithms** | 20+ core operations | 40+ operations with variants |
| **Composition** | Single-file (~500 lines) | Multi-file (33 files) |
| **Documentation** | Modern, example-rich | Academic, reference-heavy |
| **Code Style** | Rust idioms, type safety | OpenFST compatibility focus |
| **Dependencies** | Minimal | Moderate |
| **Learning Curve** | Gentle | Steep |

### When to Choose Each Library

**Choose ArcWeight when:**
- Building NLP applications (spell checking, morphology, phonology)
- Need clean, maintainable FST code with good documentation
- Want modern Rust idioms and type safety
- Require multiple FST implementations (vector, const, lazy, cached)
- Value simplicity and accessibility over exhaustive features
- Prefer streamlined implementations of core algorithms

**Choose rustfst when:**
- Migrating from OpenFST C++ codebase
- Need maximum feature parity with OpenFST
- Require specialized algorithms (lookahead composition, advanced matchers)
- Working in research setting with OpenFST familiarity
- Need Python bindings
- Require every algorithmic variant from OpenFST

### Architectural Philosophy

**ArcWeight** prioritizes clarity and maintainability:
- Trait-based design with clear separation of concerns
- Single-file implementations for core algorithms
- Comprehensive inline documentation with theoretical background
- Modern Rust patterns (Result types, iterators, zero-cost abstractions)
- Focus on essential functionality with clean abstractions

**rustfst** prioritizes feature completeness:
- Maximum OpenFST compatibility
- Extensive algorithm variants and optimizations
- Complex type hierarchies for specialized use cases
- Academic rigor with research paper references
- Comprehensive coverage of OpenFST functionality

## Features

- 🚀 **High Performance**: Optimized implementations with parallel algorithms support
- 🔧 **Modular Design**: Trait-based architecture for maximum extensibility
- 📊 **Rich Semiring Support**: 
  - Tropical, Probability, Boolean, Log, Real semirings
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

### Installation

Add ArcWeight to your `Cargo.toml`:

```toml
[dependencies]
arcweight = "0.1"
```

**System Requirements:**
- Rust 1.85.0 or later
- 64-bit architecture (x86_64 or aarch64)

### Your First FST

Here's a simple FST that recognizes the word "hello":

```rust
use arcweight::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple FST
    let mut fst = VectorFst::<TropicalWeight>::new();
    
    // Add states
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    let s3 = fst.add_state();
    let s4 = fst.add_state();
    let s5 = fst.add_state();
    
    // Set start and final states
    fst.set_start(s0);
    fst.set_final(s5, TropicalWeight::one());
    
    // Add arcs for "hello"
    fst.add_arc(s0, Arc::new('h' as u32, 'h' as u32, TropicalWeight::one(), s1));
    fst.add_arc(s1, Arc::new('e' as u32, 'e' as u32, TropicalWeight::one(), s2));
    fst.add_arc(s2, Arc::new('l' as u32, 'l' as u32, TropicalWeight::one(), s3));
    fst.add_arc(s3, Arc::new('l' as u32, 'l' as u32, TropicalWeight::one(), s4));
    fst.add_arc(s4, Arc::new('o' as u32, 'o' as u32, TropicalWeight::one(), s5));
    
    // Use the FST - check if "hello" is accepted
    let input = vec!['h' as u32, 'e' as u32, 'l' as u32, 'l' as u32, 'o' as u32];
    let accepted = fst.get_properties().contains(FstProperties::ACCEPTOR);
    
    println!("Created FST with {} states", fst.num_states());
    println!("FST is an acceptor: {}", accepted);
    
    Ok(())
}
```

**Want to dive deeper?**
- **[🧠 Core Concepts](docs/core-concepts/README.md):** Understand FSTs and semirings
- **[💡 Examples](examples/):** Theoretical and applied examples

## Documentation

### 📚 Learning Resources

- **[Core Concepts](docs/core-concepts/README.md):** Mathematical foundations

### 🔗 Reference Documentation

- **[API Reference](https://docs.rs/arcweight):** Complete API documentation with examples
- **[Bibliography](docs/bibliography.bib):** Foundational papers and references

## Examples & Applications

### 🚀 Try It Now

Run these examples to see ArcWeight in action:

```bash
# Spell checking and correction
cargo run --example spell_checking

# Edit distance computation
cargo run --example edit_distance

# Morphological analysis
cargo run --example morphological_analyzer
```

### 📝 Real-World Examples

Run these examples to see ArcWeight in action:

```bash
# Text processing examples
cargo run --example edit_distance
cargo run --example spell_checking

# Speech and phonetics
cargo run --example pronunciation_lexicon
cargo run --example phonological_rules

# Morphology and linguistics
cargo run --example morphological_analyzer
cargo run --example transliteration

# Text normalization
cargo run --example number_date_normalizer
```

All examples include comprehensive inline documentation with theoretical background and implementation details.

## Performance & Benchmarking

ArcWeight is engineered for high performance with extensive optimization and benchmarking:

### ⚡ Performance Features

- **🔥 Zero-copy arc iteration** - Minimal allocations in hot paths
- **🏗️ Cache-friendly data structures** - Optimized memory layout for modern CPUs
- **⚙️ Parallel algorithms** - Optional Rayon-based parallelization for large FSTs
- **🎯 Algorithm selection** - Automatic optimization based on FST properties

### 📊 Benchmarking Suite

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark categories
cargo bench core      # Core FST operations
cargo bench memory    # Memory usage and storage
cargo bench parallel  # Parallel algorithm scaling
```

**Benchmark Categories:**

| Category | Focus | Key Metrics |
|----------|-------|-------------|
| **Core** | Basic operations (compose, determinize, minimize) | Throughput, latency |
| **Memory** | Storage efficiency and memory usage | Bytes per state/arc |
| **I/O** | Serialization and file operations | Read/write speed |
| **Optimization** | Advanced algorithms (push, prune, connect) | Optimization ratio |
| **Parallel** | Multi-threaded performance | Scaling factor |

**Results:** Run `cargo bench` to see performance characteristics on your system

## Community & Support

### 💬 Getting Help

- **📖 [API Documentation](https://docs.rs/arcweight)** - Complete reference
- **🐛 [Issues](https://github.com/aaronstevenwhite/arcweight/issues)** - Bug reports and feature requests
- **💡 [Discussions](https://github.com/aaronstevenwhite/arcweight/discussions)** - Community support and ideas

### 🤝 Contributing

We welcome contributions! Whether you're:

- 🐛 **Reporting bugs** - Help us improve stability
- 💡 **Suggesting features** - Share your ideas for new functionality  
- 📝 **Improving docs** - Make ArcWeight more accessible
- ⚡ **Optimizing performance** - Help us go faster
- 🧪 **Adding tests** - Increase code coverage and reliability

### 🛠️ Development

**Quick Setup:**
```bash
# Clone and build
git clone https://github.com/aaronstevenwhite/arcweight.git
cd arcweight
cargo build

# Run tests
cargo test
cargo test --integration

# Code quality
cargo fmt && cargo clippy

# Run examples
cargo run --example edit_distance

# Build documentation book (requires mdbook and mdbook-katex)
cargo install mdbook mdbook-katex
mdbook build
```

**Code Quality Standards:**
- ✅ Comprehensive test coverage (unit, integration, property-based)
- 🎨 Consistent formatting (`rustfmt.toml`)
- 🔍 Strict linting (`clippy.toml`)
- 📚 Documented public APIs
- ⚡ Performance regression testing

## License

Licensed under [Apache 2.0](LICENSE) - see the license file for details.