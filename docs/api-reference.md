# API Reference

This section provides complete API documentation for ArcWeight. The API is organized into several key modules that handle different aspects of weighted finite state transducers.

## Online Documentation

The complete, up-to-date API documentation is available at:
- **[docs.rs/arcweight](https://docs.rs/arcweight)** - Latest stable release documentation
- **[Local Documentation](#local-documentation)** - Generate locally for development builds

## Getting Started

### Prelude
- **[`prelude`](https://docs.rs/arcweight/latest/arcweight/prelude/)** - Convenient re-exports of commonly used types and functions

Most users should start by importing the prelude:
```rust
use arcweight::prelude::*;
```

This provides access to all commonly used types including `VectorFst`, `TropicalWeight`, `Arc`, and core algorithms.

## Core Modules

### Finite State Transducers
- **[`fst`](https://docs.rs/arcweight/latest/arcweight/fst/)** - Core FST data structures and operations
  - **[`VectorFst`](https://docs.rs/arcweight/latest/arcweight/fst/struct.VectorFst.html)** - Mutable FST implementation
  - **[`ConstFst`](https://docs.rs/arcweight/latest/arcweight/fst/struct.ConstFst.html)** - Immutable optimized FST
  - **[`CompactFst`](https://docs.rs/arcweight/latest/arcweight/fst/struct.CompactFst.html)** - Compressed FST representation
  - **[Traits](https://docs.rs/arcweight/latest/arcweight/fst/traits/)** - FST trait definitions

### Semirings
- **[`semiring`](https://docs.rs/arcweight/latest/arcweight/semiring/)** - Abstract semiring traits and implementations
  - **[`TropicalWeight`](https://docs.rs/arcweight/latest/arcweight/semiring/struct.TropicalWeight.html)** - Tropical semiring (min-plus algebra)
  - **[`LogWeight`](https://docs.rs/arcweight/latest/arcweight/semiring/struct.LogWeight.html)** - Log semiring for probability computations
  - **[`BooleanWeight`](https://docs.rs/arcweight/latest/arcweight/semiring/struct.BooleanWeight.html)** - Boolean semiring for unweighted automata
  - **[`ProbabilityWeight`](https://docs.rs/arcweight/latest/arcweight/semiring/struct.ProbabilityWeight.html)** - Probability semiring
  - **[Traits](https://docs.rs/arcweight/latest/arcweight/semiring/traits/)** - Semiring trait definitions

### Algorithms
- **[`algorithms`](https://docs.rs/arcweight/latest/arcweight/algorithms/)** - Core FST algorithms
  - **[`shortest_path`](https://docs.rs/arcweight/latest/arcweight/algorithms/fn.shortest_path.html)** - Shortest path algorithms
  - **[`minimize`](https://docs.rs/arcweight/latest/arcweight/algorithms/fn.minimize.html)** - FST minimization
  - **[`compose`](https://docs.rs/arcweight/latest/arcweight/algorithms/fn.compose.html)** - FST composition
  - **[`union`](https://docs.rs/arcweight/latest/arcweight/algorithms/fn.union.html)** - FST union
  - **[`determinize`](https://docs.rs/arcweight/latest/arcweight/algorithms/fn.determinize.html)** - Determinization
  - **[`reverse`](https://docs.rs/arcweight/latest/arcweight/algorithms/fn.reverse.html)** - FST reversal

### Core Data Structures
- **[`arc`](https://docs.rs/arcweight/latest/arcweight/arc/)** - Arc (transition) types and utilities
  - **[`Arc`](https://docs.rs/arcweight/latest/arcweight/arc/struct.Arc.html)** - FST transition/arc
  - **[`StateId`](https://docs.rs/arcweight/latest/arcweight/arc/type.StateId.html)** - State identifier type
  - **[`Label`](https://docs.rs/arcweight/latest/arcweight/arc/type.Label.html)** - Input/output label type

### Utilities
- **[`utils`](https://docs.rs/arcweight/latest/arcweight/utils/)** - Utility functions and data structures
  - **[`SymbolTable`](https://docs.rs/arcweight/latest/arcweight/utils/struct.SymbolTable.html)** - Symbol table management
- **[`io`](https://docs.rs/arcweight/latest/arcweight/io/)** - Input/output operations for FST serialization
  - **[`read_text`](https://docs.rs/arcweight/latest/arcweight/io/fn.read_text.html)** - Read text format FSTs
  - **[`write_text`](https://docs.rs/arcweight/latest/arcweight/io/fn.write_text.html)** - Write text format FSTs
  - **[`read_openfst`](https://docs.rs/arcweight/latest/arcweight/io/fn.read_openfst.html)** - OpenFST compatibility
- **[`properties`](https://docs.rs/arcweight/latest/arcweight/properties/)** - FST property computation and analysis

## Quick Reference

### Creating an FST
```rust
use arcweight::prelude::*;

let mut fst = VectorFst::<TropicalWeight>::new();
let s0 = fst.add_state();
let s1 = fst.add_state();
fst.set_start(s0);
fst.set_final(s1, TropicalWeight::one());
fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s1));
```

### Loading from File
```rust
use arcweight::prelude::*;

// Text format
let fst: VectorFst<TropicalWeight> = read_text(&mut file, None, None)?;

// OpenFST format
let fst: VectorFst<TropicalWeight> = read_openfst(&mut file)?;
```

### Common Operations
```rust
use arcweight::prelude::*;

// Composition
let result = compose_default(&fst1, &fst2)?;

// Union
let result = union(&fst1, &fst2)?;

// Shortest path
let result = shortest_path_single(&fst)?;

// Minimization
let result = minimize(&fst)?;
```

## Local Documentation

To generate and view the API documentation locally:

```bash
# Generate documentation
cargo doc --open

# Generate with private items (for development)
cargo doc --document-private-items --open
```

## Integration with Code

Many functions in this library are documented with examples that you can run directly. Look for code blocks marked with `rust` in the online documentation - these are executable examples that demonstrate real usage patterns.

## See Also

- [Core Concepts](core-concepts/) - Mathematical background
- [Working with FSTs](working-with-fsts/) - Common usage patterns
- [Examples](examples/) - Practical examples and tutorials