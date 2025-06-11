# ArcWeight Architecture

## System Overview

ArcWeight is a high-performance Rust library for weighted finite state transducers (WFSTs) designed with a trait-based, modular architecture. The library follows Rust conventions and provides type-safe abstractions for FST construction, manipulation, and analysis.

## Core Design Philosophy

### Trait-Based Architecture
- **Separation of Concerns**: Core traits (`Fst`, `MutableFst`, `ExpandedFst`, `LazyFst`) define clean interfaces
- **Generic Programming**: All algorithms work with any type implementing the semiring trait
- **Zero-Cost Abstractions**: Trait implementations compile to efficient machine code
- **Extensibility**: Easy to add custom FST types and semirings

### Type Safety
- **StateId** and **Label** are type aliases for `u32` with special constants (`NO_STATE_ID`, `NO_LABEL`)
- **Semiring** trait ensures mathematical correctness of weight operations
- **Result** types enforce proper error handling throughout the API

## Module Structure

### FST Module (`src/fst/`)

**Core Traits** (`traits.rs`):
- `Fst<W>`: Read-only FST operations for any semiring `W`
- `MutableFst<W>`: FST modification operations
- `ExpandedFst<W>`: FSTs with all states in memory
- `LazyFst<W>`: On-demand FST computation

**Implementations**:
- `VectorFst`: Mutable FST with vector-based storage (most common)
- `ConstFst`: Immutable FST optimized for space and access speed
- `CompactFst`: Space-efficient FST with compressed arc storage
- `LazyFstImpl`: Wrapper for lazy computation of FST operations
- `CacheFst`: Caching layer for expensive FST operations

### Semiring Module (`src/semiring/`)

**Core Traits** (`traits.rs`):
- `Semiring`: Basic semiring operations (plus, times, zero, one)
- `DivisibleSemiring`: Semirings supporting division
- `StarSemiring`: Semirings with Kleene star operation
- `NaturallyOrderedSemiring`: Semirings with natural ordering

**Weight Types**:
- `TropicalWeight`: Min-plus semiring (shortest path, Viterbi)
- `ProbabilityWeight`: Real probabilities [0,1] 
- `BooleanWeight`: Boolean semiring for unweighted FSTs
- `LogWeight`: Log semiring for numerical stability
- `MinWeight`/`MaxWeight`: Min/max semirings
- `StringWeight`: String concatenation semiring
- `ProductWeight`: Cartesian product of two semirings

### Algorithms Module (`src/algorithms/`)

**Core Operations**:
- `compose`: FST composition with configurable filters
- `determinize`: Subset construction determinization
- `minimize`: State minimization using partition refinement
- `shortest_path`: Single-source shortest paths with configurable options

**Construction Operations**:
- `concat`: FST concatenation
- `union`: FST union (alternation)
- `closure`/`closure_plus`: Kleene star and plus operations
- `reverse`: FST reversal

**Transformation Operations**:
- `connect`: Remove non-coaccessible states
- `project_input`/`project_output`: Project to input/output
- `remove_epsilons`: Epsilon removal
- `topsort`: Topological sort of states

**Optimization Operations**:
- `push_weights`/`push_labels`: Weight/label pushing
- `prune`: Remove low-probability paths
- `weight_convert`: Convert between semiring types

**Advanced Operations**:
- `intersect`/`difference`: Set operations on FSTs
- `replace`: FST replacement with context
- `synchronize`: Synchronization for composition
- `randgen`: Random path generation

### Arc Module (`src/arc/`)

**Arc Representation**:
- Input label, output label, weight, next state
- Efficient iteration patterns for hot paths
- Memory-optimized storage

### Properties Module (`src/properties/`)

**FST Properties**:
- Structural properties (acyclic, deterministic, etc.)
- Property inheritance and composition rules
- Optimization decisions based on properties

### I/O Module (`src/io/`)

**Format Support**:
- Binary format for efficient serialization
- Text format for human readability
- OpenFST compatibility for interoperability

### Utils Module (`src/utils/`)

**Supporting Utilities**:
- `SymbolTable`: Bidirectional symbol mapping
- Priority queues for algorithms
- Encoding utilities

## Key Implementation Patterns

### Memory Management
- Owned data in `VectorFst` for maximum flexibility
- Reference counting for shared immutable data
- Zero-copy arc iteration where possible
- Optional compression for large FSTs

### Error Handling
- `Result<T>` return types throughout
- Comprehensive error types with context
- Fail-fast validation where appropriate

### Performance Optimizations
- Generic algorithms compile to specialized code
- Cache-friendly data layouts
- SIMD-friendly operations where applicable
- Optional parallel processing via feature flags

### Parallelization Strategy
- Rayon-based data parallelism for large FSTs
- Algorithm-specific parallel implementations
- Thread-safe immutable FST types
- Configurable parallelism thresholds

## Extension Points

### Custom Semirings
Implement the `Semiring` trait for domain-specific weights:
```rust
impl Semiring for CustomWeight {
    fn plus(&self, other: &Self) -> Self { /* ... */ }
    fn times(&self, other: &Self) -> Self { /* ... */ }
    fn zero() -> Self { /* ... */ }
    fn one() -> Self { /* ... */ }
}
```

### Custom FST Types
Implement core traits for specialized storage:
```rust
impl<W: Semiring> Fst<W> for CustomFst<W> {
    // Required methods
}
```

### Custom Algorithms
Use existing building blocks for specialized operations:
```rust
pub fn custom_algorithm<F, W>(fst: &F) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: Semiring,
{
    // Implementation
}
```

## Integration Patterns

### Library Integration
- `prelude` module for convenient imports
- Feature flags for optional dependencies
- Stable public API with semantic versioning

### Application Patterns
- Builder patterns for complex FST construction
- Iterator patterns for efficient traversal
- Functional composition of operations

## Performance Characteristics

### Complexity Guarantees
- Most algorithms have well-defined time/space complexity
- Worst-case bounds documented for critical operations
- Performance regression testing via benchmarks

### Optimization Strategies
- Algorithm selection based on FST properties
- Lazy evaluation for expensive operations
- Caching for repeated computations
- Memory pool allocation for hot paths

## Future Architecture Considerations

### Planned Enhancements
- GPU acceleration for parallel algorithms
- Streaming FST operations for large data
- Incremental algorithm updates
- Advanced compression techniques

### Research Directions
- Machine learning integration
- Approximate algorithms for performance
- Domain-specific optimizations
- Advanced parallel algorithms 