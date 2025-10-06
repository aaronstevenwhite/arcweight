# Operations Overview

**Understanding FST operations and when to use them**

This section introduces FST operations conceptually, explaining what each operation does and when to use it. FST operations are algorithms that transform, combine, or analyze FSTs for different purposes.

## What Are FST Operations?

Finite State Transducer (FST) operations are algorithms that transform, combine, or analyze FSTs. They provide building blocks for creating complex language processing systems from simple components.

### Core Capabilities

FST operations allow you to:
- **Transform** one FST into another (determinization, minimization)
- **Combine** multiple FSTs (composition, union, concatenation)
- **Analyze** FST structure (shortest path, projection)  
- **Optimize** FST performance (connection, epsilon removal)

> **New to FSTs?** Make sure you've completed the [Quick Start Guide](../quick-start.md) first.

## When to Use Each Operation

### Composition: Building Pipelines

**Use when:** You need to chain transformations together

**Examples:**
- Text processing: `lowercase → normalize → tokenize`
- Translation: `source text → phrase extraction → target text`
- Speech: `acoustic model → language model → text`

**Key insight:** Output of first FST becomes input to second FST

### Union: Combining Alternatives

**Use when:** You have multiple valid options

**Examples:**
- Vocabularies: Medical terms ∪ General English
- Dialects: US spelling ∪ UK spelling
- Formats: Email patterns ∪ Phone patterns

**Key insight:** Accepts input if ANY component FST accepts it

### Optimization Operations

**Use when:** Performance or memory is a concern

| Operation | When to Use | Effect |
|-----------|-------------|---------|
| **Connect** | Dead states exist | Removes unreachable states |
| **Determinize** | Non-deterministic FST | One path per input |
| **Minimize** | Redundant states | Smallest equivalent FST |
| **Epsilon Remove** | Has ε-transitions | Cleaner structure |

### Path Operations

**Use when:** You need specific outputs

- **Shortest Path**: Find the best/cheapest solution
- **N-best Paths**: Get top N alternatives
- **Pruning**: Remove unlikely paths

### Structural Operations

**Use when:** You need to analyze or extract

- **Input Projection**: What inputs are accepted?
- **Output Projection**: What outputs are possible?
- **Intersection**: What's common between FSTs?
- **Difference**: What's in A but not B?

## Decision Tree

Use this decision tree to identify which operation you need:

```text
What are you trying to do?
│
├─ Combine FSTs?
│  ├─ Sequential processing? → Composition
│  ├─ Alternative options? → Union
│  └─ One after another? → Concatenation
│
├─ Optimize performance?
│  ├─ First optimization? → Connect (always start here!)
│  ├─ Multiple paths per input? → Determinize
│  ├─ Too many states? → Minimize
│  └─ Has ε-transitions? → Epsilon Removal
│
├─ Find solutions?
│  ├─ Best answer only? → Shortest Path
│  ├─ Multiple good answers? → N-best Paths
│  └─ Remove bad options? → Pruning
│
└─ Analyze structure?
   ├─ Valid inputs? → Input Projection
   ├─ Possible outputs? → Output Projection
   ├─ Common elements? → Intersection
   └─ Unique elements? → Difference
```

## Operation Flow

Some operations work better in specific orders. The recommended optimization pipeline is:

```text
Original FST
    ↓
Connect (remove dead states)
    ↓
Remove Epsilons (clean structure)
    ↓
Determinize (one path per input)
    ↓
Minimize (canonical form)
    ↓
Optimized FST
```

### Why This Order?

1. **Connect first** — Removes dead states, making subsequent operations faster
2. **Remove epsilons before determinize** — Determinization handles epsilon-free FSTs better
3. **Determinize before minimize** — Minimization requires deterministic input
4. **Minimize last** — Produces the canonical minimal form

## Common Patterns

### Building an NLP Pipeline

```rust,ignore
// Tokenizer → Lowercase → Stemmer
let pipeline = compose(&tokenizer, &lowercase)?;
let pipeline = compose(&pipeline, &stemmer)?;
```

### Creating a Spell Checker

```rust,ignore
// Dictionary ∪ Common Misspellings
let vocabulary = union(&dictionary, &common_errors)?;
let spell_checker = compose(&edit_distance, &vocabulary)?;
```

### Optimizing for Production

```rust,ignore
// Full optimization pipeline
let fst = connect(&fst)?;           // Remove dead states
let fst = remove_epsilons(&fst)?;   // Remove epsilons
let fst = determinize(&fst)?;       // Make deterministic
let fst = minimize(&fst)?;          // Minimize size
```

## Performance Considerations

### Operation Complexity

| Operation | Best Case | Average | Worst Case |
|-----------|-----------|---------|------------|
| Composition | O(n₁n₂) | O(n₁n₂\|Σ\|) | O(n₁n₂\|Σ\|²) |
| Union | O(n₁+n₂) | O(n₁+n₂) | O(n₁+n₂) |
| Determinization | O(n) | O(n log n) | O(2ⁿ) |
| Minimization | O(n log n) | O(n log n) | O(n log n) |
| Shortest Path | O(E) | O(E + V log V) | O(E + V log V) |

### Memory Usage

- **Composition**: Can be large (product of sizes)
- **Determinization**: Can explode exponentially
- **Union**: Sum of input sizes
- **Minimization**: Usually reduces size

## Best Practices

### 1. Start Simple
Build complex operations from simple, tested components:
```rust,ignore
// Good: Test each component (helper functions not shown)
let lower = build_lowercase_fst()?;
let norm = build_normalizer_fst()?;
let combined = compose(&lower, &norm)?;
```

### 2. Optimize Lazily
Don't optimize until you need to:
```rust,ignore
// Only optimize if FST is large or slow
if fst.num_states() > 10000 {
    fst = optimize_fst(fst)?;
}
```

### 3. Monitor Size
Track FST growth during composition:
```rust,ignore
println!("States: {} → {}", 
    fst1.num_states() * fst2.num_states(),
    composed.num_states());
```

## Common Mistakes to Avoid

### ❌ Over-optimization
```rust,ignore
// Bad: Optimizing tiny FSTs wastes time
let tiny_fst = /* 10 states */;
let optimized = full_optimization_pipeline(tiny_fst)?; // Unnecessary!
```

### ❌ Wrong Operation Order
```rust,ignore
// Bad: Minimizing before determinizing
let min = minimize(&fst)?;  // Error: needs deterministic input!
let det = determinize(&min)?;
```

### ❌ Ignoring Epsilon Transitions
```rust,ignore
// Bad: Determinizing with epsilons
let det = determinize(&fst_with_epsilons)?; // Suboptimal!
// Good: Remove epsilons first
let clean = remove_epsilons(&fst_with_epsilons)?;
let det = determinize(&clean)?;
```

## Next Steps

Now that you understand FST operations conceptually:

1. **Learn Core Operations** → [Core Operations](core-operations.md)
2. **Master Optimization** → [Optimization Operations](optimization-operations.md)
3. **Find Solutions** → [Path Operations](path-operations.md)

---

**Ready to dive deeper?** Start with [Core Operations](core-operations.md) to learn composition, union, and concatenation in detail.