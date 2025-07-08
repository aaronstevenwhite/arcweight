# Frequently Asked Questions

Common questions and solutions for ArcWeight users. If you don't find your answer here, check [GitHub Discussions](https://github.com/aaronstevenwhite/arcweight/discussions) or file an [issue](https://github.com/aaronstevenwhite/arcweight/issues).

## Getting Started

### Q: What's the difference between ArcWeight and OpenFST?

**A:** ArcWeight is a pure Rust implementation inspired by OpenFST:

| Feature | ArcWeight | OpenFST |
|---------|-----------|---------|
| **Language** | Pure Rust | C++ |
| **Memory Safety** | Guaranteed by Rust | Manual management |
| **Dependencies** | Minimal | C++ toolchain |
| **Performance** | Comparable | Highly optimized |
| **API Style** | Rust idioms | C++ patterns |
| **Compatibility** | Read/write OpenFST format | Native format |

Choose ArcWeight for Rust projects needing memory safety and modern tooling.

### Q: Which semiring should I use for my application?

**A:** Choose based on your use case:

- **TropicalWeight** (min-plus) - Shortest path, optimization, most common
- **ProbabilityWeight** - Probabilities, language models  
- **LogWeight** - Numerical stability with small probabilities
- **BooleanWeight** - Simple accept/reject, no numeric weights

```rust,ignore
use arcweight::prelude::*;

// Shortest path / optimization
let fst = VectorFst::<TropicalWeight>::new();

// Probabilistic models
let fst = VectorFst::<ProbabilityWeight>::new();

// Simple recognition
let fst = VectorFst::<BooleanWeight>::new();
```

### Q: How do I convert between different FST types?

**A:** Use conversion functions or rebuild:

```rust,ignore
use arcweight::prelude::*;
use arcweight::fst::ConstFst;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// VectorFst to ConstFst (for read-only operations)
let mut vector_fst = VectorFst::<TropicalWeight>::new();
let s0 = vector_fst.add_state();
vector_fst.set_start(s0);
let const_fst = ConstFst::from_fst(&vector_fst)?;
# Ok(())
# }

// Between different semirings (when mathematically valid)
let tropical_fst = VectorFst::<TropicalWeight>::new();
let boolean_fst: VectorFst<BooleanWeight> = weight_convert(&tropical_fst)?;
```

## Common Operations

### Q: Why does `compose(&fst1, &fst2)` give a compiler error?

**A:** The `compose` function requires a filter parameter. Use `compose_default` for most cases:

```rust,ignore
// ❌ This doesn't work
let result = compose(&fst1, &fst2)?;

// ✅ Use this instead
let result: VectorFst<TropicalWeight> = compose_default(&fst1, &fst2)?;

// ✅ Or specify the filter explicitly
let result = compose(&fst1, &fst2, DefaultComposeFilter::new())?;
```

### Q: How do I count the total number of arcs in an FST?

**A:** Sum across all states:

```rust,ignore
let total_arcs: usize = fst.states().map(|s| fst.num_arcs(s)).sum();
println!("Total arcs: {}", total_arcs);
```

### Q: My FST has way too many states after composition. What can I do?

**A:** Try optimization techniques:

```rust,ignore
// 1. Minimize FSTs before composition
let min_fst1 = minimize(&fst1)?;
let min_fst2 = minimize(&fst2)?;
let composed = compose_default(&min_fst1, &min_fst2)?;

// 2. Use connection to remove unreachable states
let connected = connect(&composed)?;

// 3. Consider different composition order
// Sometimes fst2 ∘ fst1 is smaller than fst1 ∘ fst2
```

### Q: How do I extract the actual strings/paths from an FST?

**A:** Walk through the FST and collect symbols:

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
                if arc.olabel != 0 {  // Not epsilon
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

**A:** You forgot to set a start state:

```rust,ignore
let mut fst = VectorFst::<TropicalWeight>::new();
let s0 = fst.add_state();

// ❌ Missing this:
fst.set_start(s0);

// ✅ Always set a start state
```

### Q: Why do I get "Invalid state ID" errors?

**A:** You're using a state ID that doesn't exist:

```rust,ignore
let mut fst = VectorFst::<TropicalWeight>::new();
let s0 = fst.add_state();  // Returns state ID 0

// ❌ Don't use arbitrary numbers
fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), 999));  // State 999 doesn't exist!

// ✅ Use actual state IDs
let s1 = fst.add_state();  // Returns state ID 1
fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::one(), s1));
```

### Q: My semiring operations don't work as expected. What's wrong?

**A:** Check your semiring choice and operations:

```rust,ignore
// ❌ Wrong: trying to add probabilities that sum > 1
let p1 = ProbabilityWeight::new(0.8);
let p2 = ProbabilityWeight::new(0.7);
let sum = p1.plus(&p2);  // Results in 1.5, invalid probability!

// ✅ Use LogWeight for numerical stability
let l1 = LogWeight::new(-log(0.8));
let l2 = LogWeight::new(-log(0.7));
let sum = l1.plus(&l2);  // Proper log-space addition
```


## Performance Issues

### Q: My FST operations are very slow. How can I speed them up?

**A:** Try these optimization strategies:

1. **Use appropriate FST types:**
   ```rust,ignore
   // For construction
   let mut builder = VectorFst::<TropicalWeight>::new();
   
   // For read-only operations (faster access)
   let readonly = ConstFst::from(&builder)?;
   ```

2. **Optimize operation order:**
   ```rust,ignore
   // ✅ Good: minimize before expensive operations
   let fst1_min = minimize(&fst1)?;
   let fst2_min = minimize(&fst2)?;
   let composed = compose_default(&fst1_min, &fst2_min)?;
   
   // ❌ Bad: compose large FSTs then minimize
   let composed_large = compose_default(&fst1, &fst2)?;
   let result = minimize(&composed_large)?;
   ```

3. **Enable parallel features:**
   ```toml
   [dependencies]
   arcweight = { version = "0.1.0", features = ["parallel"] }
   ```

### Q: How much memory will my FST use?

**A:** Rough estimates:

- **VectorFst**: ~50-100 bytes per state + ~30-50 bytes per arc
- **ConstFst**: ~30-40% less memory than VectorFst
- **CompactFst**: ~50-70% less memory than VectorFst

```rust,ignore
// Monitor memory usage during construction
let states = fst.num_states();
let total_arcs: usize = fst.states().map(|s| fst.num_arcs(s)).sum();
let approx_memory = states * 75 + total_arcs * 40;  // Rough bytes
println!("Estimated memory: {} KB", approx_memory / 1024);
```

## Debugging

### Q: How do I visualize my FST to debug it?

**A:** Create a simple text representation:

```rust,ignore
fn debug_fst(fst: &VectorFst<TropicalWeight>) {
    println!("FST Debug:");
    println!("Start state: {:?}", fst.start());
    
    for state in fst.states() {
        print!("State {}: ", state);
        
        // Show outgoing arcs
        for arc in fst.arcs(state) {
            let input = if arc.ilabel == 0 { "ε".to_string() } 
                       else { char::from_u32(arc.ilabel).unwrap_or('?').to_string() };
            let output = if arc.olabel == 0 { "ε".to_string() }
                        else { char::from_u32(arc.olabel).unwrap_or('?').to_string() };
            print!("{}:{}/{}->{} ", input, output, arc.weight.value(), arc.nextstate);
        }
        
        // Show if final
        if let Some(weight) = fst.final_weight(state) {
            print!("[FINAL:{}]", weight.value());
        }
        println!();
    }
}
```


### Q: How do I check if my FST is valid?

**A:** Basic validation checks:

```rust,ignore
fn validate_fst(fst: &VectorFst<TropicalWeight>) -> Result<()> {
    // Check start state exists
    if fst.start().is_none() {
        return Err(Error::Algorithm("No start state".into()));
    }
    
    // Check all arcs point to valid states
    let max_state = fst.num_states() as u32;
    for state in fst.states() {
        for arc in fst.arcs(state) {
            if arc.nextstate >= max_state {
                return Err(Error::Algorithm(format!(
                    "Arc from {} points to invalid state {}", 
                    state, arc.nextstate
                )));
            }
        }
    }
    
    // Check there's at least one final state
    let has_final = fst.states().any(|s| fst.is_final(s));
    if !has_final {
        println!("Warning: FST has no final states");
    }
    
    Ok(())
}
```

## Advanced Topics

### Q: How do I implement a custom semiring?

**A:** Implement the `Semiring` trait:

```rust,ignore
#[derive(Debug, Clone, PartialEq)]
struct CustomWeight {
    value: f32,
}

impl Semiring for CustomWeight {
    type Value = f32;
    
    fn new(value: Self::Value) -> Self {
        CustomWeight { value }
    }
    
    fn value(&self) -> &Self::Value {
        &self.value
    }
    
    fn plus(&self, other: &Self) -> Self {
        // Define addition operation
        CustomWeight { value: self.value.min(other.value) }
    }
    
    fn times(&self, other: &Self) -> Self {
        // Define multiplication operation  
        CustomWeight { value: self.value + other.value }
    }
    
    fn zero() -> Self {
        CustomWeight { value: f32::INFINITY }
    }
    
    fn one() -> Self {
        CustomWeight { value: 0.0 }
    }
}
```

### Q: Why do I see infinity symbols (∞) in my output instead of numbers?

**A:** Some semirings use infinity to represent their zero element:

```rust,ignore
// TropicalWeight uses ∞ as zero (identity for min operation)
let zero = TropicalWeight::zero();
println!("{}", zero);  // Prints: ∞

// To check if a weight is infinity/zero:
if zero.is_zero() {
    println!("This is the zero element");
}

// Convert for display if needed:
let display_value = if zero.is_zero() { 
    "infinity".to_string() 
} else { 
    zero.value().to_string() 
};
```

### Q: Can I use ArcWeight with Unicode strings?

**A:** Yes, but convert to/from u32 codes:

```rust,ignore
fn add_unicode_arc(fst: &mut VectorFst<TropicalWeight>, from: u32, to: u32, text: &str) {
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

// Usage
let mut fst = VectorFst::new();
let s0 = fst.add_state();
let s1 = fst.add_state();
add_unicode_arc(&mut fst, s0, s1, "Hello 世界");
```

### Q: Which semirings support which operations?

**A:** Different FST operations require specific semiring traits:

| Operation | Required Trait | Supported Semirings |
|-----------|---------------|---------------------|
| `closure`, `closure_plus` | `StarSemiring` | BooleanWeight only |
| `minimize`, `determinize` | `DivisibleSemiring` | TropicalWeight, LogWeight |
| `push_weights` | `DivisibleSemiring` | TropicalWeight, LogWeight |
| `shortest_path` | `Semiring` | All semirings |
| `compose`, `concat`, `union` | `Semiring` | All semirings |

```rust,ignore
// Example: Using the right semiring for each operation
let tropical_fst = VectorFst::<TropicalWeight>::new();
let minimized = minimize(&tropical_fst)?;  // ✓ Works

let boolean_fst = VectorFst::<BooleanWeight>::new();
let closed = closure(&boolean_fst)?;  // ✓ Works
```

### Q: How do I serialize FSTs to disk?

**A:** Enable the serde feature:

```toml
[dependencies]
arcweight = { version = "0.1.0", features = ["serde"] }
```

```rust,ignore
#[cfg(feature = "serde")]
use arcweight::io::{write_binary, read_binary};

#[cfg(feature = "serde")]
fn save_and_load() -> Result<()> {
    let fst = VectorFst::<TropicalWeight>::new();
    
    // Save to binary format
    write_binary(&fst, "my_fst.bin")?;
    
    // Load from binary format
    let loaded_fst: VectorFst<TropicalWeight> = read_binary("my_fst.bin")?;
    
    Ok(())
}
```


## Troubleshooting Checklist

When things go wrong, check:

1. **✅ Set start state** - `fst.set_start(state)`
2. **✅ Set final states** - `fst.set_final(state, weight)`  
3. **✅ Valid state IDs** - Use returned values from `add_state()`
4. **✅ Correct semiring** - Match your problem domain and operations
5. **✅ Feature flags** - Enable needed features in `Cargo.toml`

## Getting Help

### Community Resources

- **[GitHub Discussions](https://github.com/aaronstevenwhite/arcweight/discussions)** - Ask questions, share ideas
- **[Issues](https://github.com/aaronstevenwhite/arcweight/issues)** - Report bugs, request features
- **[Examples](examples/)** - Working code for common tasks

### Documentation

- **[Quick Start](quick-start.md)** - Get started quickly
- **[Core Concepts](core-concepts/)** - Understand the theory
- **[Working with FSTs](working-with-fsts/)** - Master FST operations
- **API Reference** - Complete API documentation (will be available on docs.rs after first crate publication)

### Contributing

Found an issue or have an improvement? We welcome contributions:

1. **Bug reports** - Help us improve reliability
2. **Documentation improvements** - Make ArcWeight more accessible
3. **Performance optimizations** - Help us go faster
4. **New examples** - Show off interesting applications

See the contributing guidelines for details on how to get started.

---

**Didn't find what you're looking for?** Ask in [GitHub Issues](https://github.com/aaronstevenwhite/arcweight/issues) or start a discussion.