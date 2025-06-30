# Performance Architecture

## Multi-Level Optimization Strategy

ArcWeight employs optimization at multiple levels through architectural design choices. For empirical performance data and benchmarks, see [Performance Benchmarks](../benchmarks.md).

**Architecture Focus**: This section covers the design patterns and implementation strategies that enable high performance, not measured results.

## Compile-Time Optimizations

### Generic Specialization

```rust
// Generic specialization enables optimal code generation
impl<W: Semiring> VectorFst<W> {
    #[inline(always)]  // Force inlining for hot paths
    pub fn add_arc(&mut self, state: StateId, arc: Arc<W>) {
        // Compiles to different optimized code for each semiring
        debug_assert!(self.is_valid_state(state));  // Removed in release builds
        
        // Direct memory access - no bounds checking in release
        unsafe {
            self.states.get_unchecked_mut(state as usize)
                .arcs.push(arc);
        }
    }
}
```

### Monomorphization Benefits

- Each algorithm instance with specific types generates optimized machine code
- No runtime dispatch overhead for type-erased operations
- Compiler can inline and optimize aggressively

### Zero-Cost Abstractions

```rust
// Compiles to different, optimized code for each semiring
impl<W: Semiring> VectorFst<W> {
    pub fn add_arc(&mut self, state: StateId, arc: Arc<W>) {
        // Specialized for W = TropicalWeight, BooleanWeight, etc.
    }
}
```

## Runtime Optimizations

### Property-Aware Algorithms

```rust
// Algorithm selection based on FST properties
pub fn optimize_fst<W, F>(fst: &F) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: DivisibleSemiring,
{
    let props = compute_properties(fst);
    
    match props.properties {
        p if p.contains(PropertyFlags::ACYCLIC) => {
            // Use specialized acyclic algorithm
            minimize_acyclic(fst)
        }
        p if p.contains(PropertyFlags::NO_EPSILONS) => {
            // Skip epsilon removal
            minimize_direct(fst)
        }
        _ => {
            // General case
            minimize(&remove_epsilons(fst)?)
        }
    }
}
```

### Lazy Evaluation

```rust
// Composition can be computed on-demand
let lazy_composed = LazyFstImpl::new(move |state| {
    compute_composed_arcs(state, &fst1, &fst2)
});
```

### Memory Access Patterns

```rust
// Cache-friendly iteration patterns
pub fn iterate_cache_friendly<W, F>(fst: &F) 
where 
    F: ExpandedFst<W>,
    W: Semiring,
{
    // Process states in order for better cache locality
    for state in 0..fst.num_states() {
        let arcs = fst.arcs_slice(state);  // Direct slice access
        
        // Process all arcs for this state before moving to next
        for arc in arcs {
            process_arc(arc);
        }
    }
}
```

## SIMD and Parallel Processing

### Parallel State Processing

```rust
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// Parallel state processing
pub fn parallel_shortest_path<W, F>(fst: &F) -> Result<VectorFst<W>>
where
    F: Fst<W> + Sync,
    W: NaturallyOrderedSemiring + Send + Sync,
{
    let states: Vec<_> = fst.states().collect();
    
    // Process states in parallel batches
    let distances: Vec<_> = states
        .par_chunks(1000)  // Batch size tuned for cache performance
        .map(|chunk| compute_distances_batch(fst, chunk))
        .collect();
        
    combine_results(distances)
}
```

### SIMD Operations

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Vectorized weight computations for tropical semiring
#[target_feature(enable = "avx2")]
unsafe fn tropical_plus_simd(weights1: &[f32], weights2: &[f32], result: &mut [f32]) {
    assert_eq!(weights1.len(), weights2.len());
    assert_eq!(weights1.len(), result.len());
    assert_eq!(weights1.len() % 8, 0);  // Process 8 f32s at once
    
    for i in (0..weights1.len()).step_by(8) {
        let w1 = _mm256_loadu_ps(weights1.as_ptr().add(i));
        let w2 = _mm256_loadu_ps(weights2.as_ptr().add(i));
        let min = _mm256_min_ps(w1, w2);  // Tropical addition = min
        _mm256_storeu_ps(result.as_mut_ptr().add(i), min);
    }
}
```

## Algorithm-Specific Optimizations

### Composition Optimizations

```rust
pub fn compose_optimized<F1, F2, W>(
    fst1: &F1,
    fst2: &F2,
) -> Result<VectorFst<W>>
where
    F1: Fst<W>,
    F2: Fst<W>,
    W: Semiring,
{
    let props1 = fst1.properties();
    let props2 = fst2.properties();
    
    // Choose optimal composition strategy
    match (props1.properties, props2.properties) {
        (p1, p2) if p1.contains(PropertyFlags::STRING) && 
                     p2.contains(PropertyFlags::STRING) => {
            // Linear FSTs: O(n) composition
            compose_linear(fst1, fst2)
        }
        (p1, _) if p1.contains(PropertyFlags::NO_EPSILONS) => {
            // No epsilon matching needed
            compose_no_epsilon(fst1, fst2)
        }
        _ => {
            // General case
            compose_general(fst1, fst2)
        }
    }
}
```

### Shortest Path Optimizations

```rust
pub fn shortest_path_optimized<F, W>(
    fst: &F,
) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: NaturallyOrderedSemiring,
{
    let props = fst.properties();
    
    if props.contains(PropertyFlags::ACYCLIC) {
        // Use topological sort for acyclic FSTs: O(V + E)
        shortest_path_acyclic(fst)
    } else if props.contains(PropertyFlags::NO_EPSILONS) {
        // Use Dijkstra without epsilon handling
        shortest_path_dijkstra(fst)
    } else {
        // General case with epsilon handling
        shortest_path_general(fst)
    }
}
```

## Memory Performance

### Cache Optimization

```rust
// Struct layout optimized for cache lines
#[repr(C)]
pub struct Arc<W: Semiring> {
    pub ilabel: u32,     // 4 bytes
    pub olabel: u32,     // 4 bytes
    pub nextstate: u32,  // 4 bytes
    pub weight: W,       // Variable size
}

// Pack small weights efficiently
#[repr(C, packed)]
pub struct TropicalArc {
    pub ilabel: u32,     // 4 bytes
    pub olabel: u32,     // 4 bytes
    pub nextstate: u32,  // 4 bytes
    pub weight: f32,     // 4 bytes - fits in single cache line
}
```

### Memory Prefetching

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn prefetch_states<W: Semiring>(fst: &VectorFst<W>, states: &[StateId]) {
    for &state in states {
        unsafe {
            let state_ptr = fst.states.as_ptr().add(state as usize);
            _mm_prefetch(state_ptr as *const i8, _MM_HINT_T0);
        }
    }
}
```

### Memory Pool Allocation

```rust
pub struct FstPool<W: Semiring> {
    arc_pool: Vec<Arc<W>>,
    state_pool: Vec<VectorState<W>>,
    free_arcs: Vec<usize>,
    free_states: Vec<usize>,
}

impl<W: Semiring> FstPool<W> {
    pub fn allocate_arc(&mut self) -> &mut Arc<W> {
        if let Some(idx) = self.free_arcs.pop() {
            &mut self.arc_pool[idx]
        } else {
            self.arc_pool.push(Arc::default());
            self.arc_pool.last_mut().unwrap()
        }
    }
    
    pub fn deallocate_arc(&mut self, arc: &Arc<W>) {
        let idx = arc as *const _ as usize - self.arc_pool.as_ptr() as usize;
        self.free_arcs.push(idx / std::mem::size_of::<Arc<W>>());
    }
}
```

## Benchmarking and Profiling

### Micro-benchmarks

```rust
#[cfg(test)]
mod benchmarks {
    use criterion::{black_box, criterion_group, criterion_main, Criterion};
    
    fn bench_arc_iteration(c: &mut Criterion) {
        let fst = create_large_fst();
        
        c.bench_function("arc_iteration", |b| {
            b.iter(|| {
                for state in fst.states() {
                    for arc in fst.arcs(state) {
                        black_box(&arc);
                    }
                }
            })
        });
    }
    
    criterion_group!(benches, bench_arc_iteration);
    criterion_main!(benches);
}
```

### Performance Profiling

```rust
#[cfg(feature = "profiling")]
mod profiling {
    use std::time::Instant;
    
    pub struct Timer {
        start: Instant,
        name: &'static str,
    }
    
    impl Timer {
        pub fn new(name: &'static str) -> Self {
            Self {
                start: Instant::now(),
                name,
            }
        }
    }
    
    impl Drop for Timer {
        fn drop(&mut self) {
            let elapsed = self.start.elapsed();
            println!("{}: {:?}", self.name, elapsed);
        }
    }
    
    macro_rules! time {
        ($name:expr, $block:block) => {
            {
                let _timer = Timer::new($name);
                $block
            }
        };
    }
}
```

## Performance Guidelines

### 1. Choose Appropriate Data Structures

```rust
// For read-heavy workloads
let fst = ConstFst::from(vector_fst);  // Better cache performance

// For memory-constrained environments
let fst = CompactFst::from(vector_fst);  // Reduced memory usage

// For construction and modification
let fst = VectorFst::new();  // Most flexible
```

### 2. Leverage Properties

```rust
// Check properties before expensive operations
if fst.properties().contains(PropertyFlags::ACYCLIC) {
    // Use faster acyclic algorithms
} else {
    // Use general algorithms
}
```

### 3. Use Lazy Evaluation

```rust
// Avoid computing unnecessary results
let lazy_result = compose_lazy(&fst1, &fst2);

// Only compute what's needed
for state in needed_states {
    let arcs = lazy_result.arcs(state);
    // Process only required arcs
}
```

### 4. Profile Before Optimizing

```rust
// Always measure before optimizing
#[cfg(feature = "profiling")]
fn profile_algorithm() {
    let fst = create_test_fst();
    
    time!("shortest_path", {
        let result = shortest_path(&fst);
    });
    
    time!("minimize", {
        let result = minimize(&fst);
    });
}
```

This performance architecture ensures ArcWeight delivers high performance while maintaining correctness and flexibility.