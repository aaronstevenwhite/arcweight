# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-16

Initial release of ArcWeight, a high-performance Rust library for weighted finite-state transducers.

### Added

#### Core FST Types

- `VectorFst` - Mutable FST with dynamic vector-based storage
- `ConstFst` - Immutable FST optimized for read-only operations
- `CompactFst` - Memory-efficient compressed FST representation
- `LazyFst` - On-demand computation wrapper for deferred evaluation
- `CacheFst` - Caching layer for expensive FST operations

#### Semiring Implementations

- `TropicalWeight` - Tropical semiring (min, +) for shortest path problems
- `LogWeight` - Log semiring for probabilistic computations
- `ProbabilityWeight` - Direct probability representation with standard operations
- `BooleanWeight` - Boolean semiring for unweighted automata
- `IntegerWeight` - Integer semiring for counting problems
- `RealWeight` - Real-valued semiring for general numeric computations
- `StringWeight` - String concatenation semiring for label sequences
- `MinMaxWeight` - Min/max semiring for optimization problems
- `ProductWeight<W1, W2>` - Cartesian product of two semirings
- `GallicWeight<W, V>` - Label-weighted semiring with five variants:
  - `LeftGallic` - Left-canonical composition
  - `RightGallic` - Right-canonical composition
  - `MinGallic` - Minimum-weight path selection
  - `RestrictGallic` - Restricted functional composition
  - `UnionGallic` - General union composition

#### Core Algorithms

**Rational Operations**
- `compose()` - Composition of two FSTs with configurable filters
- `concat()` - Sequential concatenation of FSTs
- `union()` - Union of multiple FSTs
- `closure()`, `closure_plus()` - Kleene star and plus closure operations
- `reverse()` - Reverse FST direction (swap initial/final states)

**Optimization Algorithms**
- `determinize()` - Convert non-deterministic FST to deterministic form
- `minimize()` - State minimization using Brzozowski's algorithm
- `remove_epsilons()` - Epsilon transition removal
- `connect()` - Remove non-accessible and non-coaccessible states
- `prune()` - Prune paths above weight threshold

**Path Algorithms**
- `shortest_path()` - Extract k-shortest paths
- `shortest_distance()` - Compute shortest distances from start state
- `randgen()` - Random path generation with configurable sampling

**Transformation Algorithms**
- `project_input()`, `project_output()` - Project to input/output labels
- `encode()`, `decode()` - Encode labels/weights for optimization
- `push_weights()` - Push weights toward initial or final states
- `push_labels()` - Push labels through epsilon transitions
- `reweight()` - Reweight FST using potential function
- `synchronize()` - Synchronize transducer for simultaneous label output
- `weight_convert()` - Convert between compatible semiring types

**Graph Algorithms**
- `state_sort()` - Sort states by BFS, DFS, or topological order
- `topsort()` - Topological sorting of acyclic FSTs
- `condense()` - Condense strongly connected components using Tarjan's algorithm
- `partition()` - Partition states into bisimulation equivalence classes

**Utility Algorithms**
- `arc_sort()` - Sort arcs by input or output label
- `arc_sum()` - Sum arc weights between duplicate transitions
- `arc_unique()` - Remove duplicate arcs
- `isomorphic()` - Test structural isomorphism between FSTs
- `intersect()` - Intersection for acceptor FSTs
- `difference()` - Set difference for acceptor FSTs
- `replace()` - Replace labels with sub-FSTs

#### I/O and Serialization

- OpenFST binary format reading and writing
- Text format for human-readable FSTs
- Serde-based serialization (feature-gated with `serde`)
- Binary format with bincode (feature-gated with `serde`)

#### Properties System

- Comprehensive FST property detection (acceptor, deterministic, acyclic, etc.)
- Automatic property computation and caching
- Property preservation tracking through operations
- Properties used for algorithm selection and optimization

#### Examples

- `edit_distance` - Levenshtein distance with configurable operation costs
- `spell_checking` - Spell checker with correction suggestions
- `morphological_analyzer` - Two-level morphology for word analysis
- `phonological_rules` - Phonological rule application
- `pronunciation_lexicon` - Grapheme-to-phoneme conversion
- `number_date_normalizer` - Text normalization for numbers and dates
- `transliteration` - Cross-script transliteration
- `string_alignment` - Sequence alignment with FSTs

#### Features

- `parallel` (default) - Rayon-based parallel processing for large FSTs
- `serde` (default) - Serialization support via serde and bincode

#### Performance

- Zero-copy arc iteration for minimal allocations
- Cache-friendly data structures optimized for modern CPUs
- Optional parallelization for compute-intensive algorithms
- Multiple FST representations for different use cases
- Automatic algorithm selection based on FST properties

#### Development

- Comprehensive benchmark suite covering all major operations
- Property-based testing with proptest
- Strict clippy configuration for code quality
- Minimum Supported Rust Version (MSRV): 1.85.0
- Cross-platform CI testing (Linux, macOS, Windows)
- Multi-version Rust testing (stable, beta, nightly)

#### Documentation

- Complete API documentation with rustdoc
- Complexity analysis for all algorithms
- Mathematical foundations and algorithm descriptions
- Runnable examples in all public API documentation
- Contributing guidelines and development workflow
- Citation format for academic use

### Notes

- State and label IDs are 32-bit unsigned integers
- Pure Rust implementation with no unsafe code in public APIs
- OpenFST file format compatibility for easy migration
- Apache 2.0 license

[0.1.0]: https://github.com/aaronstevenwhite/arcweight/releases/tag/v0.1.0
