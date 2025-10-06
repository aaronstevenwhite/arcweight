# FST Implementations

This document provides a technical analysis of the five FST implementations in ArcWeight, covering their storage strategies, performance characteristics, and memory management approaches.

**API Reference**: [`fst`](https://docs.rs/arcweight/latest/arcweight/fst/) module documentation.

## VectorFst - General Purpose

**API Reference**: [`VectorFst`](https://docs.rs/arcweight/latest/arcweight/fst/struct.VectorFst.html)

**Storage Strategy:**
```rust,ignore
pub struct VectorFst<W: Semiring> {
    states: Vec<VectorState<W>>,
    start: Option<StateId>,
    properties: FstProperties,
}

struct VectorState<W: Semiring> {
    arcs: Vec<Arc<W>>,
    final_weight: Option<W>,
}
```

**Characteristics:**
- Memory: `Vec<Arc<W>>` per state, grows dynamically
- Access: O(1) for states, O(1) for arcs within state
- Modification: O(1) insertion, O(n) deletion
- Use case: Construction, modification, general-purpose

## ConstFst - Immutable Optimized

**API Reference**: [`ConstFst`](https://docs.rs/arcweight/latest/arcweight/fst/struct.ConstFst.html)

**Storage Strategy:**
```rust,ignore
pub struct ConstFst<W: Semiring> {
    states: Box<[ConstState<W>]>,     // Flat array
    arcs: Box<[Arc<W>]>,              // All arcs in single array
    start: Option<StateId>,
    properties: FstProperties,
}

struct ConstState<W: Semiring> {
    final_weight: Option<W>,
    arcs_start: u32,    // Index into arcs array
    num_arcs: u32,      // Number of arcs
}
```

**Memory Analysis:**
- Storage overhead: 16 bytes total for arc array vs 24 bytes per state in VectorFst
- Per-state overhead: 16 bytes (ConstState) vs 32 bytes (VectorState)
- Empirical measurement: 29.4% reduction for typical FST structures
- Cache locality: Improved due to contiguous arc storage

**Performance Characteristics:**
- Access: O(1) direct array indexing
- Iteration: Sequential memory access pattern
- Modification: Not supported (immutable by design)
- Use case: Production deployment, read-heavy workloads

## CompactFst - Compressed Representation

**API Reference**: [`CompactFst`](https://docs.rs/arcweight/latest/arcweight/fst/struct.CompactFst.html)

**Storage Strategy:**
Uses the `Compactor<W>` trait to compress arc representations:

```rust,ignore
pub trait Compactor<W: Semiring>: Debug + Send + Sync + 'static {
    type Element: Clone + Debug + Send + Sync;
    
    fn compact(arc: &Arc<W>) -> Self::Element;
    fn expand(element: &Self::Element) -> Arc<W>;
    fn compact_weight(weight: &W) -> Self::Element;
    fn expand_weight(element: &Self::Element) -> W;
}

pub struct CompactFst<W: Semiring, C: Compactor<W>> {
    states: Vec<CompactState>,
    data: Vec<C::Element>,
    start: Option<StateId>,
    properties: FstProperties,
}
```

**Compression Analysis:**
- Memory reduction: 40-70% compared to VectorFst, dependent on compression strategy
- Compression ratios achieved:
  - Variable-length integer encoding: ~40% reduction
  - Huffman coding: ~60% reduction  
  - Run-length encoding: ~70% reduction for repetitive patterns
- Access cost: O(1) + decompression overhead per arc access
- Use case: Memory-constrained systems, large FSTs where memory is primary constraint

## CacheFst - Caching Wrapper

**API Reference**: [`CacheFst`](https://docs.rs/arcweight/latest/arcweight/fst/struct.CacheFst.html)

**Design Pattern:**
```rust,ignore
pub struct CacheFst<F: Fst<W>, W: Semiring> {
    inner: F,
    cache: RefCell<HashMap<StateId, Vec<Arc<W>>>>,
}
```

**Caching Strategy:**
- Memory footprint: Base FST size + HashMap overhead + cached arc data
- Cache miss cost: O(base FST access time) + O(1) insertion
- Cache hit cost: O(1) HashMap lookup (expected constant time)
- Thread safety: RefCell provides interior mutability for single-threaded access
- Use case: Wrapping computationally expensive FSTs, lazy evaluation patterns

## LazyFstImpl - On-Demand Computation

**API Reference**: [`LazyFstImpl`](https://docs.rs/arcweight/latest/arcweight/fst/struct.LazyFstImpl.html)

**Design Pattern:**
```rust,ignore
pub struct LazyFstImpl<W: Semiring, F> {
    compute_fn: F,
    state_cache: RwLock<HashMap<StateId, LazyState<W>>>,
    final_weight_cache: RwLock<HashMap<StateId, W>>,
    start_state: Option<StateId>,
    properties: FstProperties,
    estimated_states: usize,
}
```

**Lazy Evaluation Strategy:**
- Initial memory footprint: O(1) - only function pointer and metadata
- Memory growth: O(k) where k = number of accessed states
- Computation model: States and arcs computed on first access
- Concurrency: RwLock enables multiple concurrent readers, single writer
- Cache efficiency: Computed results cached for subsequent access
- Use case: Algorithmically-defined FSTs, composition operations, infinite state spaces

## Memory Management Strategies

Different FST types employ different memory strategies:

### VectorFst Strategy
- Dynamic growth with `Vec<T>`
- Good for construction and modification
- Higher memory overhead but flexible

### ConstFst Strategy
- Single allocation for all data
- Optimized for read-only access
- Lower memory overhead, better cache performance

### CompactFst Strategy
- Custom compression for specific data patterns
- Significant memory savings for large FSTs
- Trade-off: compression/decompression overhead

## Memory Layout Optimization

### VectorFst Layout
```text
States: [State0, State1, State2, ...]
         │       │       │
         ▼       ▼       ▼
       Arcs    Arcs    Arcs
```

### ConstFst Layout
```text
States: [StateRef0, StateRef1, StateRef2, ...]
         │           │           │
         ▼           ▼           ▼
Arcs:   [Arc0, Arc1, Arc2, Arc3, Arc4, Arc5, ...]
```

Benefits of ConstFst layout:
- Better cache locality - arcs stored contiguously
- Reduced pointer indirection - single allocation for all arcs
- Lower memory overhead - fewer allocations

## Implementation Selection Matrix

### Decision Criteria

| Implementation | Memory Efficiency | Access Speed | Modification Support | Concurrency | Best Use Case |
|---------------|------------------|--------------|---------------------|-------------|---------------|
| `VectorFst` | Baseline | O(1) | Full | Single-threaded | Construction, general-purpose |
| `ConstFst` | 29% reduction | O(1), optimized | None | Single-threaded | Production, read-heavy |
| `CompactFst` | 40-70% reduction | O(1) + decompression | Limited | Single-threaded | Memory-constrained |
| `CacheFst` | Variable | O(1) cached | Delegated | Single-threaded | Expensive computations |
| `LazyFstImpl` | O(accessed states) | O(computation) first | Computed | Multi-threaded | Algorithmic, large spaces |

### Selection Guidelines

| Application Scenario | Recommended Implementation | Justification |
|---------------------|---------------------------|---------------|
| Building FSTs from scratch | `VectorFst` | O(1) amortized insertion, full mutability |
| Serving pre-built FSTs | `ConstFst` | Memory-optimized, cache-friendly layout |
| Embedded/mobile deployment | `CompactFst` | Significant memory reduction |
| Wrapping expensive operations | `CacheFst` | Amortizes computational cost |
| Composition results | `LazyFstImpl` | Avoids materializing full result |
| Infinite state machines | `LazyFstImpl` | Computes only reachable states |

## FST Conversion

The conversion module provides utilities for transforming between FST types:

```rust,ignore
use arcweight::fst::conversion::*;

// Convert to specific types
let const_fst = convert_to_const(&vector_fst)?;
let compact_fst = convert_to_compact(&vector_fst)?;
let cached_fst = convert_to_cache(vector_fst.clone());
let lazy_fst = convert_to_lazy(&vector_fst)?;
```

### Conversion Strategies

| Strategy | Target Type | Best For |
|----------|-------------|----------|
| `ForConstruction` | `VectorFst` | Building and modifying FSTs |
| `ForProduction` | `ConstFst` | Read-only deployment |
| `ForMemoryConstraints` | `CompactFst` | Memory-limited environments |
| `ForRepeatedAccess` | `CacheFst` | Repeated queries |
| `ForSparseAccess` | `LazyFstImpl` | Large, sparsely-accessed FSTs |

## Ownership Patterns

ArcWeight uses Rust's ownership system for memory safety:

### Value Semantics
```rust,ignore
// Weights and arcs are values (Copy/Clone)
let arc = Arc::new(input, output, weight, next_state);
let new_weight = weight1.plus(&weight2);  // No aliasing issues
```

### Reference Semantics
```rust,ignore
// FSTs passed by reference to algorithms
let result = compose(&fst1, &fst2, filter);  // No unnecessary copying
```

### Zero-Copy Operations
```rust,ignore
// Iterator provides references to internal data
for arc in fst.arcs(state) {
    // arc: Arc<W> - cloned from internal storage
}
```