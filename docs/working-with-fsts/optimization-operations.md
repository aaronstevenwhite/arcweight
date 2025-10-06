# Optimization Operations

**FST performance and memory optimization**

Optimization operations transform FSTs into equivalent but more efficient forms. These operations are important for production systems where performance and memory usage matter.

## Connection

Connection removes states that don't participate in any successful path. Start optimization with connection — it's fast and can significantly reduce FST size.

### When to Use Connection

- After any FST construction
- Before other optimization operations
- When FSTs have been modified

### Basic Connection

```rust,ignore
use arcweight::prelude::*;

fn optimize_fst(fst: &VectorFst<TropicalWeight>) -> Result<VectorFst<TropicalWeight>> {
    // Remove dead states
    let connected = connect(fst)?;
    
    println!("States: {} → {}", 
        fst.num_states(), 
        connected.num_states());
    
    Ok(connected)
}
```

### Understanding Dead States

Dead states can't reach a final state or can't be reached from the initial state:

```rust,ignore
// Example: Building an FST with dead states
let mut fst = VectorFst::new();
let s0 = fst.add_state();
let s1 = fst.add_state();
let s2 = fst.add_state(); // Dead state - no path to final
let s3 = fst.add_state();

fst.set_start(s0);
fst.set_final(s3, TropicalWeight::one());

// Add transitions
fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::one(), s3));
fst.add_arc(s2, Arc::new(3, 3, TropicalWeight::one(), s2)); // Self loop

// s2 is not connected - will be removed
let connected = connect(&fst)?;
assert_eq!(connected.num_states(), 3); // s2 removed
```

## Epsilon Removal

Epsilon transitions (ε:ε) can make FSTs harder to work with. Removing them simplifies structure and improves performance.

### When to Remove Epsilons

- Before determinization
- When FST has many ε-transitions
- To simplify FST structure

### Basic Epsilon Removal

```rust,ignore
fn remove_epsilons_example(fst: &VectorFst<TropicalWeight>) -> Result<VectorFst<TropicalWeight>> {
    // Remove epsilon transitions
    let no_eps = remove_epsilons(fst)?;
    
    // Epsilon removal might create dead states
    let cleaned = connect(&no_eps)?;
    
    Ok(cleaned)
}
```

### Epsilon Removal in Practice

```rust,ignore
// FST with epsilon transitions
let mut fst = VectorFst::new();
let s0 = fst.add_state();
let s1 = fst.add_state();
let s2 = fst.add_state();

fst.set_start(s0);
fst.set_final(s2, TropicalWeight::one());

// Mix of regular and epsilon transitions
fst.add_arc(s0, Arc::new(0, 0, TropicalWeight::new(1.0), s1)); // ε:ε
fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(2.0), s1)); // a:b
fst.add_arc(s1, Arc::new(0, 0, TropicalWeight::new(1.0), s2)); // ε:ε

// After epsilon removal
let clean = remove_epsilons(&fst)?;
// Direct path from s0 to s2 with combined weights
```

## Determinization

Determinization ensures that for any state and input symbol, there's at most one outgoing transition. This makes FSTs faster to traverse.

### When to Determinize

- Need predictable, fast traversal
- Before minimization
- Building real-time systems

### Basic Determinization

```rust,ignore
fn make_deterministic(fst: &VectorFst<TropicalWeight>) -> Result<VectorFst<TropicalWeight>> {
    // Check if already deterministic
    let props = fst.properties();
    if props.contains(FstProperties::DETERMINISTIC) {
        return Ok(fst.clone());
    }
    
    // Determinize
    let det = determinize(fst)?;
    Ok(det)
}
```

### Determinization Example

```rust,ignore
// Non-deterministic FST: multiple paths for same input
let mut fst = VectorFst::new();
let s0 = fst.add_state();
let s1 = fst.add_state();
let s2 = fst.add_state();

fst.set_start(s0);
fst.set_final(s1, TropicalWeight::one());
fst.set_final(s2, TropicalWeight::one());

// Two transitions with same input label
fst.add_arc(s0, Arc::new(1, 2, TropicalWeight::new(1.0), s1));
fst.add_arc(s0, Arc::new(1, 3, TropicalWeight::new(2.0), s2));

// After determinization: one transition per input
let det = determinize(&fst)?;
// Now only one outgoing arc with label 1 from start state
```

### Handling Determinization Explosion

```rust,ignore
fn safe_determinize(
    fst: &VectorFst<TropicalWeight>,
    max_states: usize
) -> Result<VectorFst<TropicalWeight>> {
    // Monitor determinization to prevent explosion
    let mut det = VectorFst::new();
    let mut state_count = 0;
    
    // Use determinization with state limit
    match determinize_with_limit(fst, max_states) {
        Ok(result) => Ok(result),
        Err(_) => {
            eprintln!("Warning: Determinization would exceed {} states", max_states);
            // Return original or use alternative approach
            Ok(fst.clone())
        }
    }
}
```

## Minimization

Minimization produces the smallest FST that recognizes the same language with the same weights. It requires a deterministic input FST.

### When to Minimize

- Memory is constrained
- Need canonical form
- After determinization

### Basic Minimization

```rust,ignore
fn minimize_fst(fst: &VectorFst<TropicalWeight>) -> Result<VectorFst<TropicalWeight>> {
    // Ensure FST is deterministic
    let det = determinize(fst)?;
    
    // Minimize
    let min = minimize(&det)?;
    
    println!("Minimization: {} → {} states", 
        det.num_states(), 
        min.num_states());
    
    Ok(min)
}
```

### Complete Optimization Pipeline

```rust,ignore
fn full_optimization(
    fst: &VectorFst<TropicalWeight>
) -> Result<VectorFst<TropicalWeight>> {
    // Step 1: Remove dead states
    let connected = connect(fst)?;
    
    // Step 2: Remove epsilon transitions
    let no_eps = remove_epsilons(&connected)?;
    
    // Step 3: Make deterministic
    let det = determinize(&no_eps)?;
    
    // Step 4: Minimize
    let min = minimize(&det)?;
    
    // Report optimization results
    println!("Optimization results:");
    println!("  Original states: {}", fst.num_states());
    println!("  After connection: {}", connected.num_states());
    println!("  After epsilon removal: {}", no_eps.num_states());
    println!("  After determinization: {}", det.num_states());
    println!("  After minimization: {}", min.num_states());
    
    Ok(min)
}
```

### Minimization Example

```rust,ignore
// FST with redundant states
let redundant_fst = build_redundant_fst()?;
// States: 0 -a-> 1 -b-> 3
//         0 -a-> 2 -b-> 3
// States 1 and 2 are equivalent

let minimized = minimize(&determinize(&redundant_fst)?)?;
// After minimization: states 1 and 2 merged
// States: 0 -a-> 1 -b-> 2
```

## Advanced Optimization Techniques

### Weight Pushing

Distribute weights to make FST more efficient:

```rust,ignore
fn push_weights(fst: &VectorFst<TropicalWeight>) -> Result<VectorFst<TropicalWeight>> {
    // Push weights toward initial state
    let pushed = push(fst, PushType::Initial)?;
    
    // Useful for probability FSTs
    Ok(pushed)
}
```

### Pruning

Remove paths with high weights (low probability):

```rust,ignore
fn prune_unlikely_paths(
    fst: &VectorFst<TropicalWeight>,
    threshold: f32
) -> Result<VectorFst<TropicalWeight>> {
    // Remove paths with weight > threshold
    let pruned = prune(fst, TropicalWeight::new(threshold))?;
    
    // Clean up
    let connected = connect(&pruned)?;
    
    Ok(connected)
}
```

### Lazy Operations

For very large FSTs, use lazy operations:

```rust,ignore
use arcweight::fst::{LazyFst, ComposeFst};

fn lazy_composition(
    fst1: &VectorFst<TropicalWeight>,
    fst2: &VectorFst<TropicalWeight>
) -> ComposeFst<TropicalWeight> {
    // Composition computed on-demand
    ComposeFst::new(fst1, fst2)
}
```

## Optimization Strategies

### Strategy 1: Memory-Constrained Systems

```rust,ignore
fn optimize_for_memory(fst: &VectorFst<TropicalWeight>) -> Result<VectorFst<TropicalWeight>> {
    // Aggressive optimization for small size
    let connected = connect(fst)?;
    let no_eps = remove_epsilons(&connected)?;
    let det = determinize(&no_eps)?;
    let min = minimize(&det)?;
    
    // Additional compression
    let compacted = compact(&min)?;
    
    Ok(compacted)
}
```

### Strategy 2: Speed-Critical Systems

```rust,ignore
fn optimize_for_speed(fst: &VectorFst<TropicalWeight>) -> Result<VectorFst<TropicalWeight>> {
    // Optimize for fast traversal
    let connected = connect(fst)?;
    let det = determinize(&connected)?;
    
    // Weight pushing for faster shortest path
    let pushed = push(&det, PushType::Initial)?;
    
    // Don't minimize if it would slow down access
    Ok(pushed)
}
```

### Strategy 3: Balanced Optimization

```rust,ignore
fn balanced_optimization(
    fst: &VectorFst<TropicalWeight>,
    size_threshold: usize
) -> Result<VectorFst<TropicalWeight>> {
    let connected = connect(fst)?;
    
    // Only do expensive operations if FST is large
    if connected.num_states() > size_threshold {
        let no_eps = remove_epsilons(&connected)?;
        let det = determinize(&no_eps)?;
        minimize(&det)
    } else {
        Ok(connected)
    }
}
```

## Performance Benchmarks

Typical optimization impact on real-world FSTs:

| Operation | Time | Size Reduction | Speed Improvement |
|-----------|------|----------------|-------------------|
| Connect | O(V+E) | 10-50% | 5-20% |
| Epsilon Removal | O(V+E) | 0-10% | 10-30% |
| Determinize | O(2^n) worst | Varies | 50-200% |
| Minimize | O(n log n) | 20-60% | 0-10% |

## Best Practices

### 1. Profile Before Optimizing

```rust,ignore
fn should_optimize(fst: &VectorFst<TropicalWeight>) -> bool {
    let num_states = fst.num_states();
    let num_arcs = fst.num_arcs();
    let has_epsilons = fst.has_epsilons();
    
    // Optimize if large or has many epsilons
    num_states > 1000 || num_arcs > 5000 || has_epsilons
}
```

### 2. Test Equivalence

```rust,ignore
fn verify_optimization(
    original: &VectorFst<TropicalWeight>,
    optimized: &VectorFst<TropicalWeight>
) -> Result<bool> {
    // Ensure optimization preserved behavior
    equivalent(original, optimized)
}
```

### 3. Monitor Resources

```rust,ignore
use std::time::Instant;

fn timed_optimization(fst: &VectorFst<TropicalWeight>) -> Result<VectorFst<TropicalWeight>> {
    let start = Instant::now();
    let result = full_optimization(fst)?;
    let duration = start.elapsed();
    
    println!("Optimization took: {:?}", duration);
    Ok(result)
}
```

## Common Issues and Solutions

### Issue: Determinization Explosion
**Solution**: Use approximate determinization or pruning

### Issue: Minimization Changes Behavior
**Solution**: Ensure FST is deterministic first

### Issue: Optimization Takes Too Long
**Solution**: Use incremental optimization or lazy operations

## Next Steps

With optimized FSTs, you can:

1. **Find best paths efficiently** → [Path Operations](path-operations.md)
2. **Analyze FST structure** → [Structural Operations](structural-operations.md)
3. **Learn advanced techniques** → [Advanced Topics](advanced-topics.md)

---

**Ready to find solutions?** Continue to [Path Operations](path-operations.md)