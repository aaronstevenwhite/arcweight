# FST Implementations

ArcWeight provides five main FST implementations, each optimized for different use cases:

**API Reference**: See the complete [`fst`](https://docs.rs/arcweight/latest/arcweight/fst/) module documentation.

## VectorFst - General Purpose

**API Reference**: [`VectorFst`](https://docs.rs/arcweight/latest/arcweight/fst/struct.VectorFst.html)

**Storage Strategy:**
```rust
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
- **Memory**: `Vec<Arc<W>>` per state, grows dynamically
- **Access**: \\(O(1)\\) for states, \\(O(1)\\) for arcs within state
- **Modification**: \\(O(1)\\) insertion, \\(O(n)\\) deletion
- **Use case**: Construction, modification, general-purpose

## ConstFst - Immutable Optimized

**API Reference**: [`ConstFst`](https://docs.rs/arcweight/latest/arcweight/fst/struct.ConstFst.html)

**Storage Strategy:**
```rust
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

**Characteristics:**
- **Memory**: ~30% less than `VectorFst`, better cache locality
- **Access**: Direct array indexing, faster iteration
- **Modification**: Immutable (requires conversion from mutable FST)
- **Use case**: Read-only operations, production deployment

## CompactFst - Compressed Representation

**API Reference**: [`CompactFst`](https://docs.rs/arcweight/latest/arcweight/fst/struct.CompactFst.html)

**Storage Strategy:**
Uses the `Compactor<W>` trait to compress arc representations:

```rust
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

**Characteristics:**
- **Memory**: ~40-70% reduction vs `VectorFst` (depends on compactor)
- **Access**: Decompression overhead per access
- **Use case**: Memory-constrained environments, large FSTs

## CacheFst - Caching Wrapper

**API Reference**: [`CacheFst`](https://docs.rs/arcweight/latest/arcweight/fst/struct.CacheFst.html)

**Design Pattern:**
```rust
pub struct CacheFst<F: Fst<W>, W: Semiring> {
    inner: F,
    cache: RefCell<HashMap<StateId, Vec<Arc<W>>>>,
}
```

**Characteristics:**
- **Memory**: Original FST + cache overhead
- **Access**: Fast for cached states, slower for first access
- **Use case**: Wrapping expensive computations, lazy FSTs

## LazyFstImpl - On-Demand Computation

**API Reference**: [`LazyFstImpl`](https://docs.rs/arcweight/latest/arcweight/fst/struct.LazyFstImpl.html)

**Design Pattern:**
```rust
pub struct LazyFstImpl<F, W> 
where
    F: Fn(StateId) -> Result<Vec<Arc<W>>>,
    W: Semiring,
{
    state_fn: F,
    cache: RefCell<HashMap<StateId, LazyState<W>>>,
    start: Option<StateId>,
}
```

**Characteristics:**
- **Memory**: Minimal initial footprint
- **Computation**: On-demand arc generation
- **Use case**: Algorithmically-defined FSTs, composition results

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
```
States: [State0, State1, State2, ...]
         │       │       │
         ▼       ▼       ▼
       Arcs    Arcs    Arcs
```

### ConstFst Layout
```
States: [StateRef0, StateRef1, StateRef2, ...]
         │           │           │
         ▼           ▼           ▼
Arcs:   [Arc0, Arc1, Arc2, Arc3, Arc4, Arc5, ...]
```

**Benefits of ConstFst layout:**
- **Better cache locality** - arcs stored contiguously
- **Reduced pointer indirection** - single allocation for all arcs
- **Lower memory overhead** - fewer allocations

## Choosing the Right Implementation

| Use Case | Recommended Type | Rationale |
|----------|------------------|-----------|
| **Construction & Modification** | `VectorFst` | Mutable, flexible |
| **Production Deployment** | `ConstFst` | Optimized, immutable |
| **Memory-Constrained** | `CompactFst` | Compressed storage |
| **Expensive Computations** | `CacheFst` | Caching wrapper |
| **Algorithmic FSTs** | `LazyFstImpl` | On-demand generation |

## Ownership Patterns

ArcWeight uses Rust's ownership system for memory safety:

### Value Semantics
```rust
// Weights and arcs are values (Copy/Clone)
let arc = Arc::new(input, output, weight, next_state);
let new_weight = weight1.plus(&weight2);  // No aliasing issues
```

### Reference Semantics
```rust
// FSTs passed by reference to algorithms
let result = compose(&fst1, &fst2, filter);  // No unnecessary copying
```

### Zero-Copy Operations
```rust
// Iterator provides references to internal data
for arc in fst.arcs(state) {
    // arc: Arc<W> - cloned from internal storage
}
```