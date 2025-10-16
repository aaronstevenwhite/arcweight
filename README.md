# ArcWeight

[![Crates.io](https://img.shields.io/crates/v/arcweight.svg)](https://crates.io/crates/arcweight)
[![Documentation](https://docs.rs/arcweight/badge.svg)](https://docs.rs/arcweight)
[![Build Status](https://github.com/aaronstevenwhite/arcweight/workflows/CI/badge.svg)](https://github.com/aaronstevenwhite/arcweight/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

A high-performance Rust library for weighted finite-state transducers with comprehensive semiring support.

ArcWeight provides efficient algorithms for constructing, combining, and optimizing weighted finite-state transducers (WFSTs), making it suitable for natural language processing, speech recognition, and computational linguistics applications.

## Features

- **Core FST Operations**: Composition, determinization, minimization, closure, union, concatenation
- **Advanced Algorithms**: Shortest path, weight pushing, epsilon removal, pruning, synchronization
- **Rich Semiring Support**: Tropical, log, probability, boolean, integer, product, and Gallic weights
- **Multiple FST Implementations**: Vector-based, constant, compact, lazy evaluation, and cached
- **Type-Safe Design**: Zero-cost abstractions with trait-based polymorphism
- **OpenFST Compatible**: Read and write OpenFST format files
- **Pure Rust**: Memory-safe implementation with no C++ dependencies
- **Parallel Processing**: Optional Rayon-based parallelization for large FSTs

## Quick Start

Add ArcWeight to your `Cargo.toml`:

```toml
[dependencies]
arcweight = "0.1"
```

### Basic Example

```rust
use arcweight::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple FST
    let mut fst = VectorFst::<TropicalWeight>::new();

    // Add states
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();

    // Set start and final states
    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::one());

    // Add arcs
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
    fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s2));

    // Perform operations
    let minimized = minimize(&fst)?;

    println!("Original states: {}", fst.num_states());
    println!("Minimized states: {}", minimized.num_states());

    Ok(())
}
```

## Examples

ArcWeight includes comprehensive examples demonstrating real-world applications:

```bash
# String edit distance
cargo run --example edit_distance

# Spell checking and correction
cargo run --example spell_checking

# Morphological analysis
cargo run --example morphological_analyzer

# Phonological rules
cargo run --example phonological_rules

# Text normalization
cargo run --example number_date_normalizer
```

See the [`examples/`](examples/) directory for complete implementations with detailed explanations.

## Documentation

- [API Documentation](https://docs.rs/arcweight) - Complete API reference with examples
- [Examples](examples/) - Real-world applications and usage patterns

## Minimum Supported Rust Version (MSRV)

ArcWeight requires Rust 1.85.0 or later.

The MSRV is explicitly tested in CI and will only be increased in minor version updates. When the MSRV is increased, the previous two stable releases will still be supported for six months.

## Performance

ArcWeight is designed for high performance:

- Zero-copy arc iteration minimizes allocations
- Cache-friendly data structures optimize memory access
- Optional parallel algorithms leverage multi-core processors
- Automatic algorithm selection based on FST properties

Run benchmarks on your system:

```bash
cargo bench
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Quick checklist:
- Follow existing code style (run `cargo fmt`)
- Add tests for new functionality (run `cargo test`)
- Update documentation for public APIs (run `cargo doc`)
- Ensure all CI checks pass (run `cargo clippy`)

## Getting Help

- [Documentation](https://docs.rs/arcweight) - API reference and guides
- [Issues](https://github.com/aaronstevenwhite/arcweight/issues) - Bug reports and feature requests
- [Discussions](https://github.com/aaronstevenwhite/arcweight/discussions) - Questions and community support

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Citation

If you use ArcWeight in your research, please cite:

```bibtex
@software{arcweight,
  author = {White, Aaron Steven},
  title = {ArcWeight: A Rust Library for Weighted Finite-State Transducers},
  url = {https://github.com/aaronstevenwhite/arcweight},
  year = {2025}
}
```

## References

ArcWeight implements algorithms based on:

- Mehryar Mohri. 1997. [Finite-State Transducers in Language and Speech Processing](https://aclanthology.org/J97-2003/). *Computational Linguistics* 23(2):269-311.
- Mehryar Mohri. 2002. [Semiring Frameworks and Algorithms for Shortest-Distance Problems](https://doi.org/10.1016/S0304-3975(99)00014-6). *Journal of Automata, Languages and Combinatorics* 7(3):321-350.
- Mehryar Mohri. 2009. [Weighted Automata Algorithms](https://doi.org/10.1007/978-3-642-01492-5_6). In *Handbook of Weighted Automata*, pages 213-254. Springer.
