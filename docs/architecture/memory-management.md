# Memory Management

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

## Memory Strategies by FST Type

### VectorFst Strategy
- Dynamic growth with `Vec<T>`
- Good for construction and modification
- Higher memory overhead but flexible

**Memory characteristics:**
- Each state stores its own `Vec<Arc<W>>`
- Arcs allocated independently as vectors grow
- Good cache locality within each state's arcs
- Higher allocation overhead due to multiple vectors

### ConstFst Strategy
- Single allocation for all data
- Optimized for read-only access
- Lower memory overhead, better cache performance

**Memory characteristics:**
- All arcs stored in single contiguous array
- States store indices into arc array
- Minimal allocation overhead
- Excellent cache locality for iteration

### CompactFst Strategy
- Custom compression for specific data patterns
- Significant memory savings for large FSTs
- Trade-off: compression/decompression overhead

**Memory characteristics:**
- Uses `Compactor<W>` trait for compression
- Memory usage depends on compression ratio
- CPU overhead for compression/decompression
- Best for memory-constrained environments

## Memory Allocation Patterns

### Pre-allocation for Performance

```rust
impl<W: Semiring> VectorFst<W> {
    /// Create FST with pre-allocated capacity
    pub fn with_capacity(states: usize, total_arcs: usize) -> Self {
        let mut fst = Self {
            states: Vec::with_capacity(states),
            start: None,
            properties: FstProperties::default(),
        };
        
        // Pre-allocate space for states
        for _ in 0..states {
            fst.states.push(VectorState {
                arcs: Vec::with_capacity(total_arcs / states),
                final_weight: None,
            });
        }
        
        fst
    }
}
```

### Memory Pool for Temporary Objects

```rust
pub struct FstBuilder<W: Semiring> {
    fst: VectorFst<W>,
    arc_pool: Vec<Arc<W>>,  // Reuse arc objects
    state_pool: Vec<StateId>,  // Reuse state IDs
}

impl<W: Semiring> FstBuilder<W> {
    pub fn add_arc_pooled(&mut self, state: StateId, arc: Arc<W>) {
        // Reuse pooled objects when possible
        if let Some(mut pooled_arc) = self.arc_pool.pop() {
            pooled_arc.ilabel = arc.ilabel;
            pooled_arc.olabel = arc.olabel;
            pooled_arc.weight = arc.weight;
            pooled_arc.nextstate = arc.nextstate;
            self.fst.add_arc(state, pooled_arc);
        } else {
            self.fst.add_arc(state, arc);
        }
    }
}
```

## Memory Optimization Techniques

### Cache-Friendly Iteration

```rust
// Process states in order for better cache locality
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

### Memory-Mapped FSTs

For very large FSTs that don't fit in memory:

```rust
pub struct MappedFst<W: Semiring> {
    mmap: Mmap,
    header: *const FstHeader,
    states: *const [ConstState<W>],
    arcs: *const [Arc<W>],
}

impl<W: Semiring> MappedFst<W> {
    pub fn from_file(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        // Parse memory-mapped structure
        let header = mmap.as_ptr() as *const FstHeader;
        let states_offset = unsafe { (*header).states_offset };
        let arcs_offset = unsafe { (*header).arcs_offset };
        
        Ok(MappedFst {
            mmap,
            header,
            states: unsafe { mmap.as_ptr().add(states_offset) as *const [ConstState<W>] },
            arcs: unsafe { mmap.as_ptr().add(arcs_offset) as *const [Arc<W>] },
        })
    }
}
```

### Lazy Memory Allocation

```rust
pub struct LazyFst<W: Semiring> {
    generator: Box<dyn Fn(StateId) -> Vec<Arc<W>>>,
    cache: RefCell<LruCache<StateId, Vec<Arc<W>>>>,
}

impl<W: Semiring> LazyFst<W> {
    fn get_arcs(&self, state: StateId) -> Vec<Arc<W>> {
        let mut cache = self.cache.borrow_mut();
        
        if let Some(arcs) = cache.get(&state) {
            arcs.clone()
        } else {
            let arcs = (self.generator)(state);
            cache.put(state, arcs.clone());
            arcs
        }
    }
}
```

## Memory Profiling and Debugging

### Memory Usage Tracking

```rust
pub struct MemoryTracker {
    allocations: AtomicUsize,
    peak_usage: AtomicUsize,
    current_usage: AtomicUsize,
}

impl MemoryTracker {
    pub fn track_allocation(&self, size: usize) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        let current = self.current_usage.fetch_add(size, Ordering::Relaxed) + size;
        
        // Update peak usage
        let mut peak = self.peak_usage.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_usage.compare_exchange_weak(peak, current, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(new_peak) => peak = new_peak,
            }
        }
    }
}
```

### Memory Leak Detection

```rust
#[cfg(debug_assertions)]
pub struct FstDropGuard<W: Semiring> {
    fst: VectorFst<W>,
    allocated_states: usize,
    allocated_arcs: usize,
}

#[cfg(debug_assertions)]
impl<W: Semiring> Drop for FstDropGuard<W> {
    fn drop(&mut self) {
        // Verify all memory was properly cleaned up
        assert_eq!(self.fst.num_states(), 0, "FST not properly cleared before drop");
        
        // Track memory statistics
        MEMORY_TRACKER.track_deallocation(
            self.allocated_states * size_of::<VectorState<W>>() +
            self.allocated_arcs * size_of::<Arc<W>>()
        );
    }
}
```

## Best Practices

### 1. Choose the Right FST Type

```rust
// For construction and modification
let mut fst = VectorFst::<TropicalWeight>::new();

// For read-only production use
let fst = ConstFst::from(vector_fst);

// For memory-constrained environments
let fst = CompactFst::from(vector_fst);
```

### 2. Pre-allocate When Possible

```rust
// Better: Pre-allocate based on known size
let mut fst = VectorFst::with_capacity(expected_states, expected_arcs);

// Avoid: Growing incrementally
let mut fst = VectorFst::new();
for _ in 0..1000000 {
    fst.add_state();  // May cause multiple reallocations
}
```

### 3. Use Memory-Efficient Algorithms

```rust
// Good: Process in-place when possible
let minimized = minimize_in_place(&mut fst);

// Less efficient: Creates intermediate copies
let reversed = reverse(&fst);
let minimized = minimize(&reversed);
```

### 4. Monitor Memory Usage

```rust
#[cfg(feature = "memory-profiling")]
fn profile_memory_usage() {
    let tracker = MemoryTracker::new();
    
    {
        let _guard = tracker.start_tracking();
        let fst = build_large_fst();
        run_algorithms(&fst);
    } // Memory usage logged when guard drops
    
    println!("Peak memory usage: {} MB", tracker.peak_usage() / 1024 / 1024);
}
```

This memory management approach ensures ArcWeight is both safe and efficient for production use.