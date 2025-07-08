# Memory Management

## Overview

ArcWeight's memory management strategy leverages Rust's ownership system to provide safe, efficient FST operations without manual memory management. This document describes the key memory patterns and strategies used throughout the library.

## Ownership Patterns

### FST Ownership

FSTs own their internal data structures:

```rust,ignore
pub struct VectorFst<W: Semiring> {
    states: Vec<VectorState<W>>,  // Owned vector of states
    start: Option<StateId>,
    properties: FstProperties,
}

struct VectorState<W: Semiring> {
    arcs: Vec<Arc<W>>,            // Owned vector of arcs
    final_weight: Option<W>,
}
```

### Reference-Based Algorithms

Algorithms operate on FST references to avoid unnecessary copying:

```rust,ignore
pub fn compose<W, F1, F2, M>(
    fst1: &F1,    // Borrowed reference
    fst2: &F2,    // Borrowed reference
    filter: impl ComposeFilter<W>,
) -> Result<M>    // Returns owned result
where
    F1: Fst<W>,
    F2: Fst<W>,
    M: MutableFst<W> + Default,
```

## Memory Strategies by FST Type

### VectorFst - Dynamic Growth

- Uses `Vec<T>` for dynamic growth
- Amortized O(1) insertion
- Higher memory overhead but flexible

```rust,ignore
// Pre-allocate when size is known
let mut fst = VectorFst::new();
fst.reserve_states(expected_states);
```

### ConstFst - Compact Storage

- Uses boxed slices for immutability
- Single allocation for all arcs
- ~30% less memory than VectorFst

```rust,ignore
pub struct ConstFst<W: Semiring> {
    states: Box<[ConstState<W>]>,  // Fixed-size array
    arcs: Box<[Arc<W>]>,           // All arcs in one array
    // ...
}
```

### CompactFst - Compressed Representation

- Custom compression via `Compactor` trait
- Trade memory for computation
- 40-70% reduction possible

```rust,ignore
// User-defined compression strategy
impl<W: Semiring> Compactor<W> for MyCompactor {
    type Element = CompressedArc;
    // Compression logic
}
```

### LazyFstImpl - On-Demand Allocation

- Computes states/arcs only when accessed
- Caches results for reuse
- Minimal initial memory footprint

## Arc Storage Patterns

### Copy Semantics

Arcs use copy semantics for efficiency:

```rust,ignore
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Arc<W> {
    pub ilabel: Label,
    pub olabel: Label,
    pub weight: W,
    pub nextstate: StateId,
}
```

This allows:
- Efficient iteration without references
- Simple arc manipulation
- Predictable memory layout

### Arc Iteration

Different iteration strategies for different FST types:

```rust,ignore
// VectorFst: Direct slice iteration
impl<W: Semiring> ExpandedFst<W> for VectorFst<W> {
    fn arcs_slice(&self, state: StateId) -> &[Arc<W>] {
        &self.states[state as usize].arcs
    }
}

// LazyFst: Computed on demand
impl<W: Semiring> Fst<W> for LazyFstImpl<F, W> {
    fn arcs(&self, state: StateId) -> Self::ArcIter<'_> {
        // Compute or retrieve from cache
    }
}
```

## Weight Memory Management

### Semiring Value Storage

Weights are stored by value in arcs:

```rust,ignore
pub trait Semiring: Clone + Debug + /* other bounds */ {
    type Value: Clone + Debug + PartialEq + PartialOrd;
    
    fn new(value: Self::Value) -> Self;
    fn value(&self) -> &Self::Value;
}
```

Common weight types and their memory footprint:
- `TropicalWeight`: 4-8 bytes (f32/f64)
- `BooleanWeight`: 1 byte
- `StringWeight`: Variable (Vec-based)

## Memory Optimization Techniques

### 1. Capacity Pre-allocation

When building FSTs with known size:

```rust,ignore
let mut fst = VectorFst::new();
fst.reserve_states(num_states);

// For specific states
fst.reserve_arcs(state_id, expected_arcs);
```

### 2. FST Conversion

Convert to more memory-efficient representations:

```rust,ignore
// Build with VectorFst
let mut mutable_fst = VectorFst::new();
// ... construction ...

// Convert to ConstFst for deployment
let const_fst = ConstFst::from(&mutable_fst);
```

### 3. Lazy Evaluation

Use lazy FSTs for large computations:

```rust,ignore
// Instead of materializing large composition
let composed = compose(&fst1, &fst2)?;

// Use lazy composition
let lazy = LazyFstImpl::new(|state| {
    // Compute arcs on demand
});
```

## Cache Efficiency

### Memory Layout

FST implementations optimize for cache locality:

- `VectorFst`: Arcs stored contiguously per state
- `ConstFst`: All arcs in single array for better prefetching
- State data separate from arc data

### Access Patterns

Algorithms designed for sequential access:

```rust,ignore
// Good: Sequential state processing
for state in fst.states() {
    for arc in fst.arcs(state) {
        // Process arc
    }
}
```

## Best Practices

### 1. Choose Appropriate FST Type

- Construction: Use `VectorFst`
- Read-only operations: Convert to `ConstFst`
- Memory-constrained: Use `CompactFst`
- Large FSTs: Consider `LazyFstImpl`

### 2. Minimize Allocations

```rust,ignore
// Reuse buffers when possible
let mut queue = VecDeque::with_capacity(fst.num_states());
let mut visited = HashSet::with_capacity(fst.num_states());
```

### 3. Use Move Semantics

```rust,ignore
// Move large FSTs instead of cloning
let result = expensive_computation();
process_fst(result);  // Move, don't clone
```

### 4. Profile Memory Usage

Use Rust's built-in tools:
- `valgrind` with `massif` for heap profiling
- `heaptrack` for allocation tracking
- Size assertions in tests

## Memory Safety Guarantees

Rust's ownership system provides:

1. **No manual memory management** - RAII handles cleanup
2. **No dangling pointers** - Lifetime checking
3. **No data races** - Send/Sync traits
4. **No buffer overflows** - Bounds checking

These guarantees come with zero runtime cost, making ArcWeight both safe and efficient.