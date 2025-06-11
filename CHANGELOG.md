# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-06-11

### Added

#### Core Features
- **FST Implementations**
  - `VectorFst` - Mutable FST with dynamic storage
  - `ConstFst` - Immutable FST optimized for static data
  - `CompactFst` - Memory-efficient FST representation
  - `LazyFst` - On-demand computation FST
  - `CacheFst` - Caching layer for expensive computations

- **Semiring Support**
  - `TropicalWeight` - Standard tropical semiring (min, +)
  - `LogWeight` - Log semiring for probabilistic computations
  - `ProbabilityWeight` - Direct probability representation
  - `BooleanWeight` - Boolean semiring
  - `StringWeight` - String concatenation and prefix operations
  - `MinMaxWeight` - Min/max operations
  - `ProductWeight` - Cartesian product of semirings

- **Core Algorithms**
  - Composition - FST composition with multiple filters
  - Determinization - Convert to deterministic FST
  - Minimization - Minimize FST states
  - Epsilon removal - Remove epsilon transitions
  - Weight pushing - Push weights toward initial/final states
  - Shortest path - Find shortest paths through FST
  - Closure - Kleene star and plus operations
  - Union - Union of multiple FSTs
  - Concatenation - Sequential combination
  - Reverse - Reverse FST direction
  - Synchronization - Synchronize transducers
  - Connection - Remove non-accessible states
  - Top sort - Topological ordering

- **I/O Formats**
  - OpenFST binary format compatibility
  - Text format for human-readable FSTs
  - Binary serialization with serde (feature-gated)

- **Properties System**
  - Comprehensive FST property detection
  - Property preservation tracking through operations
  - Optimization hints based on properties

#### Examples
- Edit distance computation with configurable costs
- Spell checking and word correction
- Pronunciation lexicon with G2P rules
- Morphological analysis (two-level morphology)
- Number and date normalization
- Cross-script transliteration
- Phonological rule application

#### Development Tools
- Comprehensive benchmark suite
- Property-based testing with proptest
- Strict clippy configuration
- MSRV enforcement (1.75.0)
- Cross-platform CI/CD

### Infrastructure
- Apache 2.0 license
- Comprehensive documentation
- GitHub Actions CI for Linux, macOS, and Windows
- Feature flags for optional functionality:
  - `parallel` - Rayon-based parallel algorithms
  - `serde` - Serialization support

### Performance
- Zero-copy arc iteration
- Cache-friendly data structures
- Optional parallelization for large FSTs
- Memory-efficient FST representations

### Compatibility
- Rust 1.75.0 or later (MSRV)
- Cross-platform support (Linux, macOS, Windows)
- OpenFST file format compatibility
- No unsafe code in public APIs

### Documentation
- Complete API documentation
- User guide and examples
- Architecture documentation
- Contributing guidelines
- Development guide

### Testing
- Unit tests for all components
- Integration tests for workflows
- Property-based tests for correctness
- Benchmarks for performance tracking
- CI testing on multiple platforms and Rust versions

### Known Limitations
- Initial release - API may evolve
- Some advanced OpenFST features not yet implemented
- Limited to 32-bit state and label IDs

[0.1.0]: https://github.com/aaronstevenwhite/arcweight/releases/tag/v0.1.0