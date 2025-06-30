# ArcWeight

[![Crates.io](https://img.shields.io/crates/v/arcweight.svg)](https://crates.io/crates/arcweight)
[![Documentation](https://docs.rs/arcweight/badge.svg)](https://docs.rs/arcweight)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Build Status](https://github.com/aaronstevenwhite/arcweight/workflows/CI/badge.svg)](https://github.com/aaronstevenwhite/arcweight/actions)

**A high-performance, modular Rust library for weighted finite state transducers (WFSTs).**

ArcWeight provides a comprehensive toolkit for constructing, combining, optimizing, and searching weighted finite-state transducers. It offers functionality comparable to [OpenFst](https://www.openfst.org/) with a modern, type-safe Rust API, making it ideal for natural language processing, speech recognition, computational linguistics, and machine learning applications.

## What Are Finite State Transducers?

Finite State Transducers (FSTs) are powerful mathematical models that transform input sequences into output sequences with associated costs or probabilities. Think of them as smart pattern matchers that can:

- **Spell check and correct text** (input: "teh" $\to$ output: "the")
- **Translate between languages** (input: "hello" $\to$ output: "hola") 
- **Convert speech to text** (input: audio features $\to$ output: words)
- **Normalize text** (input: "2nd" $\to$ output: "second")
- **Parse and generate morphology** (input: "running" $\to$ output: "run+ing")

**Why choose ArcWeight over alternatives?**
- 🦀 **Pure Rust**: Memory safety, modern tooling, no C++ dependencies
- ⚡ **Performance**: Optimized algorithms with optional parallelization
- 🔧 **Flexibility**: Extensible design with custom semirings and FST types
- 📚 **Complete**: Comprehensive algorithm suite with detailed documentation

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

### Installation

Add ArcWeight to your `Cargo.toml`:

```toml
[dependencies]
arcweight = "0.1"
```

**System Requirements:**
- Rust 1.75.0 or later
- 64-bit architecture (x86_64 or aarch64)

### Your First FST

Here's a simple spell checker that suggests "hello" for misspelled words:

```rust
use arcweight::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple dictionary FST
    let mut dictionary = VectorFst::<TropicalWeight>::new();
    
    // Build FST that accepts "hello" with cost 0
    let s0 = dictionary.add_state();  // Start
    let s1 = dictionary.add_state();  // After 'h'
    let s2 = dictionary.add_state();  // After 'e'
    let s3 = dictionary.add_state();  // After 'l'
    let s4 = dictionary.add_state();  // After 'l'
    let s5 = dictionary.add_state();  // Final state
    
    dictionary.set_start(s0);
    dictionary.set_final(s5, TropicalWeight::one());
    
    // Add arcs for "hello"
    dictionary.add_arc(s0, Arc::new('h' as u32, 'h' as u32, TropicalWeight::one(), s1));
    dictionary.add_arc(s1, Arc::new('e' as u32, 'e' as u32, TropicalWeight::one(), s2));
    dictionary.add_arc(s2, Arc::new('l' as u32, 'l' as u32, TropicalWeight::one(), s3));
    dictionary.add_arc(s3, Arc::new('l' as u32, 'l' as u32, TropicalWeight::one(), s4));
    dictionary.add_arc(s4, Arc::new('o' as u32, 'o' as u32, TropicalWeight::one(), s5));
    
    // Create edit distance FST (allows 1 character error)
    let edit_distance = build_edit_distance_fst(1)?;
    
    // Compose for spell checking: input_word -> edit_operations -> dictionary_word
    let spell_checker = compose(&edit_distance, &dictionary)?;
    
    println!("Spell checker created with {} states", spell_checker.num_states());
    Ok(())
}
```

**Want to dive deeper?**
- **[🧠 Core Concepts](docs/core-concepts.md)** - Understand FSTs and semirings
- **[💡 Examples](examples/)** - Real-world applications in the examples directory

## Documentation

### 📚 Learning Resources

| Resource | Description | Best For |
|----------|-------------|----------|
| **[Core Concepts](docs/core-concepts/README.md)** | FSTs, semirings, and mathematical foundations | Understanding theory |

### 🔗 Reference Documentation

- **[API Reference](https://docs.rs/arcweight)** - Complete API documentation with examples

## Examples & Applications

### 🚀 Try It Now

Run these examples to see ArcWeight in action:

```bash
# Spell checking and correction
cargo run --example word_correction

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
cargo run --example word_correction
cargo run --example fst_composition

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

**Get Started:**
1. Check out **[good first issues](https://github.com/aaronstevenwhite/arcweight/labels/good%20first%20issue)**
2. Join the discussion in **[GitHub Discussions](https://github.com/aaronstevenwhite/arcweight/discussions)**

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

---

**Made with ❤️ by the ArcWeight community**

*ArcWeight is designed for researchers, engineers, and developers working with finite state methods in natural language processing, speech recognition, and computational linguistics.*