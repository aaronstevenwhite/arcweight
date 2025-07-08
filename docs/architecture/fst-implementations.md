# FST Implementations

ArcWeight provides five main FST implementations, each optimized for different use cases:

**API Reference**: See the complete [`fst`](https://docs.rs/arcweight/latest/arcweight/fst/) module documentation.

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
- **Memory**: `Vec<Arc<W>>` per state, grows dynamically
- **Access**: \\(O(1)\\) for states, \\(O(1)\\) for arcs within state
- **Modification**: \\(O(1)\\) insertion, \\(O(n)\\) deletion
- **Use case**: Construction, modification, general-purpose

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

**Characteristics:**
- **Memory**: ~30% less than `VectorFst`, better cache locality
- **Access**: Direct array indexing, faster iteration
- **Modification**: Immutable (requires conversion from mutable FST)
- **Use case**: Read-only operations, production deployment

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

**Characteristics:**
- **Memory**: ~40-70% reduction vs `VectorFst` (depends on compactor)
- **Access**: Decompression overhead per access
- **Use case**: Memory-constrained environments, large FSTs

## CacheFst - Caching Wrapper

**API Reference**: [`CacheFst`](https://docs.rs/arcweight/latest/arcweight/fst/struct.CacheFst.html)

**Design Pattern:**
```rust,ignore
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
```rust,ignore
pub struct LazyFstImpl<W: Semiring, F> {
    compute_fn: F,
    state_cache: RwLock<HashMap<StateId, LazyState<W>>>,
    final_weight_cache: RwLock<HashMap<StateId, W>>,
    cache_config: CacheConfig,
    start_state: Option<StateId>,
    properties: FstProperties,
    estimated_states: usize,
}
```

**Characteristics:**
- **Memory**: Minimal initial footprint, grows with accessed states
- **Computation**: On-demand arc generation with configurable caching
- **Thread Safety**: RwLock-based concurrent access with eviction policies
- **Use case**: Algorithmically-defined FSTs, composition results, infinite state spaces

## EvictingCacheFst - Wrapper with Eviction Policies

**API Reference**: [`EvictingCacheFst`](https://docs.rs/arcweight/latest/arcweight/fst/struct.EvictingCacheFst.html)

**Design Pattern:**
```rust,ignore
pub struct EvictingCacheFst<F: Fst<W>, W: Semiring> {
    inner: F,
    arc_cache: RwLock<HashMap<StateId, Vec<Arc<W>>>>,
    final_weight_cache: RwLock<HashMap<StateId, Option<W>>>,
    metadata_cache: RwLock<HashMap<StateId, StateMetadata>>,
    cache_config: CacheConfig,
    access_tracker: RwLock<AccessTracker>,
    stats: RwLock<CachePerformanceStats>,
}
```

**Characteristics:**
- **Memory**: Configurable limits with LRU/LFU/Random eviction policies
- **Performance**: Cache hit/miss tracking with adaptive memory management
- **Thread Safety**: Concurrent access with fine-grained locking
- **Use case**: Memory-bounded caching of expensive FST operations

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
| **Expensive Computations** | `CacheFst` | Simple caching wrapper |
| **Memory-Bounded Caching** | `EvictingCacheFst` | Configurable eviction policies |
| **Algorithmic FSTs** | `LazyFstImpl` | On-demand generation with smart caching |
| **Infinite State Spaces** | `LazyFstImpl` | Streaming support for very large automata |

## FST Conversion Matrix

The conversion module provides comprehensive utilities for transforming between FST types based on performance requirements:

### Automatic Conversion

```rust,ignore
use arcweight::prelude::*;
use arcweight::fst::conversion::{ConversionStrategy, auto_convert};

let vector_fst = VectorFst::<TropicalWeight>::new();

// Automatic conversion based on use case
let strategy = ConversionStrategy::ForProduction;
let converted = auto_convert(&vector_fst, strategy)?;
```

### Manual Conversion Functions

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

### Conversion Matrix

| From / To | VectorFst | ConstFst | CompactFst | CacheFst | LazyFstImpl | EvictingCacheFst |
|-----------|-----------|----------|------------|----------|-------------|------------------|
| VectorFst | Identity  | ✓        | ✓          | ✓        | ✓           | ✓                |
| ConstFst  | ✓         | Identity | ✓          | ✓        | ✓           | ✓                |
| CompactFst| ✓         | ✓        | Identity   | ✓        | ✓           | ✓                |
| CacheFst  | ✓         | ✓        | ✓          | Identity | ✓           | ✓                |
| LazyFstImpl | ✓       | ✓        | ✓          | ✓        | Identity    | ✓                |
| EvictingCacheFst | ✓  | ✓        | ✓          | ✓        | ✓           | Identity         |

### Batch Conversion

```rust,ignore
use arcweight::fst::conversion::BatchConverter;

let fsts = vec![fst1, fst2, fst3];
let const_fsts = BatchConverter::convert_all_to_const(&fsts)?;
let compact_fsts = BatchConverter::convert_all_to_compact(&fsts)?;
```

### Conversion Metrics

```rust,ignore
use arcweight::fst::conversion::estimate_conversion_metrics;

let metrics = estimate_conversion_metrics(&fst, ConversionStrategy::ForProduction);
println!("Estimated memory: {} bytes", metrics.estimated_memory);
println!("Estimated access time: {}x", metrics.estimated_access_time);
```

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