# Frequently Asked Questions

This document addresses common questions about ArcWeight usage and implementation.

## Getting Started

### Q: What is the difference between ArcWeight and OpenFST?

ArcWeight is a Rust implementation of weighted finite state transducers, while OpenFST is a C++ library. Key differences:

| Aspect | ArcWeight | OpenFST |
|--------|-----------|---------|
| Language | Rust | C++ |
| Memory Safety | Compiler-enforced | Manual |
| Build System | Cargo | Make/CMake |
| API Design | Rust idioms | C++ STL |
| Binary Format | OpenFST-compatible | Native |

### Q: Which semiring should I use?

Select based on your application requirements:

| Semiring | Operations | Applications |
|----------|------------|--------------|
| TropicalWeight | min, + | Shortest path, edit distance |
| ProbabilityWeight | +, × | Probabilistic models |
| LogWeight | ⊕log, + | Numerical stability |
| BooleanWeight | ∨, ∧ | Unweighted automata |

```rust,ignore
// Shortest path problems
let fst = VectorFst::<TropicalWeight>::new();

// Probabilistic applications
let fst = VectorFst::<ProbabilityWeight>::new();
```

### Q: How do I convert between FST types?

```rust,ignore
use arcweight::prelude::*;
use arcweight::fst::ConstFst;

// VectorFst to ConstFst
let vector_fst = VectorFst::<TropicalWeight>::new();
let const_fst = ConstFst::from_fst(&vector_fst)?;

// Weight conversion
let tropical_fst = VectorFst::<TropicalWeight>::new();
let boolean_fst: VectorFst<BooleanWeight> = weight_convert(&tropical_fst)?;
```

## Common Operations

### Q: Why does compose(&fst1, &fst2) give a compiler error?

The compose function requires a filter parameter. Use compose_default for standard composition:

```rust,ignore
// Use compose_default for standard composition
let result: VectorFst<TropicalWeight> = compose_default(&fst1, &fst2)?;

// Or specify filter explicitly
let result = compose(&fst1, &fst2, DefaultComposeFilter::new())?;
```

### Q: How do I count the total number of arcs in an FST?

```rust,ignore
let total_arcs: usize = fst.states().map(|s| fst.num_arcs(s)).sum();
```

### Q: My FST has too many states after composition. What can I do?

Apply optimization techniques:

```rust,ignore
// Minimize before composition
let min_fst1 = minimize(&fst1)?;
let min_fst2 = minimize(&fst2)?;
let composed = compose_default(&min_fst1, &min_fst2)?;

// Remove unreachable states
let connected = connect(&composed)?;
```

### Q: How do I extract paths from an FST?

```rust,ignore
fn extract_paths(fst: &VectorFst<TropicalWeight>) -> Vec<String> {
    let mut paths = Vec::new();
    
    if let Some(start) = fst.start() {
        let mut stack = vec![(start, String::new())];
        
        while let Some((state, path)) = stack.pop() {
            if fst.is_final(state) {
                paths.push(path.clone());
            }
            
            for arc in fst.arcs(state) {
                let mut new_path = path.clone();
                if arc.olabel != 0 {  // Non-epsilon
                    new_path.push(char::from_u32(arc.olabel).unwrap_or('?'));
                }
                stack.push((arc.nextstate, new_path));
            }
        }
    }
    
    paths
}
```

## Error Handling

### Q: What does "FST has no start state" mean?

You must set a start state for every FST:

```rust,ignore
let mut fst = VectorFst::<TropicalWeight>::new();
let s0 = fst.add_state();
fst.set_start(s0);  // Required
```

### Q: Why do I get "Invalid state ID" errors?

State IDs must be created with add_state():

```rust,ignore
let mut fst = VectorFst::<TropicalWeight>::new();
let s0 = fst.add_state();  // Returns 0
let s1 = fst.add_state();  // Returns 1

// Use returned state IDs
fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
```

### Q: My semiring operations produce unexpected results. Why?

Ensure values are within valid ranges:

```rust,ignore
// Probability values must be in [0, 1]
let p = ProbabilityWeight::new(0.8);  // Valid

// Use LogWeight for numerical stability
let l = LogWeight::new(-0.223);  // -ln(0.8)
```


## Performance

### Q: How can I optimize FST operations?

Choose appropriate FST types for your use case:

```rust,ignore
// Construction phase
let mut builder = VectorFst::<TropicalWeight>::new();

// Read-only operations
let const_fst = ConstFst::from_fst(&builder)?;
```

Optimize operation order:

```rust,ignore
// Minimize before composition
let min1 = minimize(&fst1)?;
let min2 = minimize(&fst2)?;
let result = compose_default(&min1, &min2)?;
```

### Q: How much memory will my FST use?

Memory usage for common weight types (TropicalWeight, LogWeight, BooleanWeight):

| FST Type | State Overhead | Arc Overhead | Notes |
|----------|----------------|--------------|-------|
| VectorFst | 32 bytes | 16 bytes | Plus Vec allocation overhead |
| ConstFst | 16 bytes | 16 bytes | Most memory efficient |
| CompactFst | 16 bytes | 8-24 bytes | Depends on compactor used |

The estimates are derived from:
- Arc struct: 16 bytes (4 bytes each for ilabel, olabel, weight, nextstate)
- VectorFst state: 32 bytes (8 bytes Option<Weight> + 24 bytes Vec<Arc>)
- ConstFst state: 16 bytes (8 bytes Option<Weight> + 8 bytes for indices)
- CompactFst state: 16 bytes (8 bytes Option<u32> + 8 bytes for indices)

```rust,ignore
let states = fst.num_states();
let arcs: usize = fst.states().map(|s| fst.num_arcs(s)).sum();

// For VectorFst with allocation overhead (~50% for Vecs)
let memory_bytes = states * 48 + arcs * 24;

// For ConstFst (no allocation overhead)
let memory_bytes = states * 16 + arcs * 16;
```

## Debugging

### Q: How can I visualize my FST for debugging?

Create a text representation:

```rust,ignore
fn debug_fst(fst: &VectorFst<TropicalWeight>) {
    println!("Start: {:?}", fst.start());
    
    for state in fst.states() {
        print!("State {}: ", state);
        
        for arc in fst.arcs(state) {
            let input = if arc.ilabel == 0 { "ε" } 
                       else { &char::from_u32(arc.ilabel).unwrap().to_string() };
            let output = if arc.olabel == 0 { "ε" }
                        else { &char::from_u32(arc.olabel).unwrap().to_string() };
            print!("{}:{}/{:.2}->{} ", input, output, arc.weight.value(), arc.nextstate);
        }
        
        if let Some(weight) = fst.final_weight(state) {
            print!("[Final:{:.2}]", weight.value());
        }
        println!();
    }
}
```

### Q: How do I validate my FST structure?

```rust,ignore
fn validate_fst<W: Semiring>(fst: &VectorFst<W>) -> Result<()> {
    // Verify start state
    if fst.start().is_none() {
        return Err("No start state".into());
    }
    
    // Verify arc targets
    let max_state = fst.num_states() as u32;
    for state in fst.states() {
        for arc in fst.arcs(state) {
            if arc.nextstate >= max_state {
                return Err(format!("Invalid target state: {}", arc.nextstate).into());
            }
        }
    }
    
    Ok(())
}
```

## Advanced Topics

### Q: How do I implement a custom semiring?

Implement the Semiring trait:

```rust,ignore
#[derive(Debug, Clone, PartialEq)]
struct CustomWeight(f32);

impl Semiring for CustomWeight {
    type Value = f32;
    
    fn new(value: Self::Value) -> Self {
        CustomWeight(value)
    }
    
    fn value(&self) -> &Self::Value {
        &self.0
    }
    
    fn plus(&self, other: &Self) -> Self {
        CustomWeight(self.0.min(other.0))
    }
    
    fn times(&self, other: &Self) -> Self {
        CustomWeight(self.0 + other.0)
    }
    
    fn zero() -> Self {
        CustomWeight(f32::INFINITY)
    }
    
    fn one() -> Self {
        CustomWeight(0.0)
    }
}
```

### Q: Can I use Unicode strings with FSTs?

Yes, convert characters to u32:

```rust,ignore
fn add_unicode_path(fst: &mut VectorFst<TropicalWeight>, from: u32, to: u32, text: &str) {
    let chars: Vec<char> = text.chars().collect();
    let mut current = from;
    
    for (i, ch) in chars.iter().enumerate() {
        let next = if i == chars.len() - 1 { to } else { fst.add_state() };
        fst.add_arc(current, Arc::new(
            *ch as u32,
            *ch as u32,
            TropicalWeight::one(),
            next
        ));
        current = next;
    }
}
```

### Q: Which operations require specific semiring traits?

| Operation | Required Trait | Compatible Semirings |
|-----------|---------------|---------------------|
| closure, closure_plus | StarSemiring | BooleanWeight |
| minimize, determinize | DivisibleSemiring | TropicalWeight, LogWeight |
| push_weights | DivisibleSemiring | TropicalWeight, LogWeight |
| shortest_path | Semiring | All |
| compose, union | Semiring | All |

### Q: How do I serialize FSTs?

Enable the serde feature:

```toml
arcweight = { version = "0.1.0", features = ["serde"] }
```

```rust,ignore
#[cfg(feature = "serde")]
use arcweight::io::{write_binary, read_binary};

let fst = VectorFst::<TropicalWeight>::new();
write_binary(&fst, "output.fst")?;
let loaded: VectorFst<TropicalWeight> = read_binary("output.fst")?;
```

## Troubleshooting Checklist

When debugging issues, verify:

- Start state set: `fst.set_start(state)`
- Final states defined: `fst.set_final(state, weight)`
- Valid state IDs: Use values from `add_state()`
- Appropriate semiring selected
- Required features enabled

## Further Resources

- [Quick Start](quick-start.md) — Basic usage
- [Core Concepts](core-concepts/) — Theory
- [Working with FSTs](working-with-fsts/) — Operations
- [Examples](examples/) — Applications
- [GitHub Repository](https://github.com/aaronstevenwhite/arcweight) — Source code and issues