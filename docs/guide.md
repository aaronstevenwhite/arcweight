# ArcWeight User Guide

## Introduction

ArcWeight is a high-performance Rust library for weighted finite state transducers (WFSTs). This comprehensive guide covers core concepts, practical usage patterns, and real-world applications of the library.

## What Are Finite State Transducers?

### Basic Concepts
A **Finite State Transducer (FST)** is a finite automaton with input and output symbols on each transition. Unlike regular automata that only accept or reject strings, FSTs transform input strings to output strings.

A **Weighted FST (WFST)** extends this by adding weights to transitions, enabling probabilistic reasoning and optimization problems like finding the lowest-cost path.

### Mathematical Foundation
WFSTs operate over mathematical structures called **semirings**:
- **Addition** combines alternative paths (e.g., min, +, logical OR)
- **Multiplication** combines sequential transitions (e.g., +, ×, logical AND)
- **Zero** represents impossible transitions
- **One** represents neutral (free) transitions

## Installation and Setup

### Basic Installation
Add to your `Cargo.toml`:
```toml
[dependencies]
arcweight = "0.1.0"
```

### Optional Features
```toml
[dependencies]
arcweight = { version = "0.1.0", features = ["parallel", "serde"] }
```

Features available:
- `parallel`: Enable multi-threaded algorithms via Rayon
- `serde`: Serialization support for weights and FSTs
- `openfst`: OpenFST format compatibility

## Quick Start

### Your First FST
```rust
use arcweight::prelude::*;

fn main() -> Result<()> {
    // Create a simple acceptor for the string "hello"
    let mut fst = VectorFst::<TropicalWeight>::new();
    
    // Add states
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    let s3 = fst.add_state();
    let s4 = fst.add_state();
    let s5 = fst.add_state();
    
    // Set start and final states
    fst.set_start(s0);
    fst.set_final(s5, TropicalWeight::one());
    
    // Add arcs for "hello" (using ASCII values)
    fst.add_arc(s0, Arc::new(104, 104, TropicalWeight::one(), s1)); // 'h'
    fst.add_arc(s1, Arc::new(101, 101, TropicalWeight::one(), s2)); // 'e'
    fst.add_arc(s2, Arc::new(108, 108, TropicalWeight::one(), s3)); // 'l'
    fst.add_arc(s3, Arc::new(108, 108, TropicalWeight::one(), s4)); // 'l'
    fst.add_arc(s4, Arc::new(111, 111, TropicalWeight::one(), s5)); // 'o'
    
    println!("Created FST with {} states", fst.num_states());
    Ok(())
}
```

### Working with Symbol Tables
```rust
use arcweight::prelude::*;

fn main() -> Result<()> {
    let mut syms = SymbolTable::new();
    
    // Add symbols
    let hello_id = syms.add_symbol("hello");
    let world_id = syms.add_symbol("world");
    
    // Look up symbols
    println!("Symbol 'hello' has ID: {}", hello_id);
    println!("ID {} maps to: {}", hello_id, syms.find_symbol(hello_id).unwrap());
    
    Ok(())
}
```

## Core FST Operations

### Composition
Composition is fundamental for building complex transducers from simpler ones:

```rust
use arcweight::prelude::*;

fn demonstrate_composition() -> Result<()> {
    // Create two simple FSTs
    let mut fst1 = VectorFst::<TropicalWeight>::new();
    let mut fst2 = VectorFst::<TropicalWeight>::new();
    
    // ... build FSTs ...
    
    // Compose them
    let composed = compose(&fst1, &fst2)?;
    
    // The result accepts input from fst1 and produces output from fst2
    println!("Composed FST has {} states", composed.num_states());
    
    Ok(())
}
```

### Determinization and Minimization
```rust
use arcweight::prelude::*;

fn optimize_fst(fst: &VectorFst<TropicalWeight>) -> Result<VectorFst<TropicalWeight>> {
    // Determinize to ensure unique paths
    let det_fst = determinize(fst)?;
    
    // Minimize to reduce state count
    let min_fst = minimize(&det_fst)?;
    
    println!("Original: {} states, Minimized: {} states", 
             fst.num_states(), min_fst.num_states());
    
    Ok(min_fst)
}
```

### Shortest Path
```rust
use arcweight::prelude::*;

fn find_best_path(fst: &VectorFst<TropicalWeight>) -> Result<()> {
    // Find single shortest path
    let shortest = shortest_path_single(fst)?;
    
    // Or find n shortest paths
    let config = ShortestPathConfig::new().nshortest(5);
    let n_shortest = shortest_path(fst, &config)?;
    
    println!("Found {} paths", n_shortest.num_states());
    Ok(())
}
```

## Semirings and Weights

### Tropical Semiring (Most Common)
```rust
use arcweight::prelude::*;

let w1 = TropicalWeight::new(2.5);  // Cost of 2.5
let w2 = TropicalWeight::new(1.8);  // Cost of 1.8

let sum = w1.plus(&w2);             // min(2.5, 1.8) = 1.8
let product = w1.times(&w2);        // 2.5 + 1.8 = 4.3

println!("Sum: {}, Product: {}", sum.value(), product.value());
```

### Probability Semiring
```rust
use arcweight::prelude::*;

let prob1 = ProbabilityWeight::new(0.7);  // 70% probability
let prob2 = ProbabilityWeight::new(0.3);  // 30% probability

let combined = prob1.plus(&prob2);        // 0.7 + 0.3 = 1.0
let sequential = prob1.times(&prob2);     // 0.7 × 0.3 = 0.21

println!("Combined: {}, Sequential: {}", combined.value(), sequential.value());
```

### Custom Semirings
```rust
use arcweight::prelude::*;

#[derive(Debug, Clone, PartialEq)]
struct FeatureWeight {
    cost: f32,
    features: Vec<f32>,
}

impl Semiring for FeatureWeight {
    fn plus(&self, other: &Self) -> Self {
        // Choose the one with lower cost
        if self.cost <= other.cost {
            self.clone()
        } else {
            other.clone()
        }
    }
    
    fn times(&self, other: &Self) -> Self {
        let mut combined_features = self.features.clone();
        for (i, &feature) in other.features.iter().enumerate() {
            if i < combined_features.len() {
                combined_features[i] += feature;
            } else {
                combined_features.push(feature);
            }
        }
        
        FeatureWeight {
            cost: self.cost + other.cost,
            features: combined_features,
        }
    }
    
    fn zero() -> Self {
        FeatureWeight {
            cost: f32::INFINITY,
            features: Vec::new(),
        }
    }
    
    fn one() -> Self {
        FeatureWeight {
            cost: 0.0,
            features: Vec::new(),
        }
    }
}
```

## Practical Applications

### Spell Checking
```rust
use arcweight::prelude::*;

fn build_dictionary_fst(words: &[&str]) -> Result<VectorFst<TropicalWeight>> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    fst.set_start(start);
    
    for word in words {
        let mut current_state = start;
        
        for ch in word.chars() {
            let next_state = fst.add_state();
            fst.add_arc(current_state, Arc::new(
                ch as u32, ch as u32, 
                TropicalWeight::one(), 
                next_state
            ));
            current_state = next_state;
        }
        
        fst.set_final(current_state, TropicalWeight::one());
    }
    
    Ok(fst)
}

fn spell_check() -> Result<()> {
    let dictionary = build_dictionary_fst(&["hello", "world", "rust"])?;
    let edit_distance_fst = build_edit_distance_fst("helo", 1)?; // Allow 1 error
    
    let corrections = compose(&edit_distance_fst, &dictionary)?;
    let best_corrections = shortest_path_single(&corrections)?;
    
    // Extract corrections from best_corrections
    Ok(())
}
```

### Phonetic Modeling
```rust
use arcweight::prelude::*;

fn build_g2p_model() -> Result<VectorFst<TropicalWeight>> {
    let mut fst = VectorFst::new();
    
    // Add grapheme-to-phoneme mappings
    // This is a simplified example
    let start = fst.add_state();
    let end = fst.add_state();
    fst.set_start(start);
    fst.set_final(end, TropicalWeight::one());
    
    // Map 'ph' -> 'f' sound
    fst.add_arc(start, Arc::new(
        'p' as u32, 'f' as u32,
        TropicalWeight::new(0.1),
        end
    ));
    
    Ok(fst)
}
```

## Performance Optimization

### Algorithm Selection
```rust
use arcweight::prelude::*;

fn optimize_based_on_properties(fst: &VectorFst<TropicalWeight>) -> Result<()> {
    let props = fst.properties();
    
    if props.is_deterministic() {
        println!("FST is already deterministic");
    } else {
        println!("Determinizing FST...");
        let _det_fst = determinize(fst)?;
    }
    
    if props.is_acyclic() {
        println!("Using specialized acyclic algorithms");
    }
    
    Ok(())
}
```

### Memory Management
```rust
use arcweight::prelude::*;

fn memory_efficient_processing() -> Result<()> {
    // Use ConstFst for read-only FSTs
    let readonly_fst = ConstFst::from_vector_fst(&vector_fst)?;
    
    // Use CompactFst for space-efficient storage
    let compact_fst = CompactFst::from_vector_fst(&vector_fst)?;
    
    // Use LazyFst for on-demand computation
    let lazy_composed = LazyFst::new(|| {
        compose(&fst1, &fst2)
    });
    
    Ok(())
}
```

### Parallel Processing
```rust
#[cfg(feature = "parallel")]
use arcweight::prelude::*;
use rayon::prelude::*;

#[cfg(feature = "parallel")]
fn parallel_operations() -> Result<()> {
    let fsts: Vec<VectorFst<TropicalWeight>> = vec![/* ... */];
    
    // Process FSTs in parallel
    let results: Result<Vec<_>> = fsts
        .par_iter()
        .map(|fst| shortest_path_single(fst))
        .collect();
    
    let shortest_paths = results?;
    println!("Processed {} FSTs in parallel", shortest_paths.len());
    
    Ok(())
}
```

## I/O and Serialization

### Text Format
```rust
use arcweight::prelude::*;

fn save_and_load_text() -> Result<()> {
    let fst = VectorFst::<TropicalWeight>::new();
    
    // Save to text format
    write_text(&fst, "output.txt")?;
    
    // Load from text format
    let loaded_fst: VectorFst<TropicalWeight> = read_text("output.txt")?;
    
    Ok(())
}
```

### Binary Format
```rust
use arcweight::prelude::*;

fn save_and_load_binary() -> Result<()> {
    let fst = VectorFst::<TropicalWeight>::new();
    
    // Save to binary format (more efficient)
    write_binary(&fst, "output.fst")?;
    
    // Load from binary format
    let loaded_fst: VectorFst<TropicalWeight> = read_binary("output.fst")?;
    
    Ok(())
}
```

### OpenFST Compatibility
```rust
#[cfg(feature = "openfst")]
use arcweight::prelude::*;

#[cfg(feature = "openfst")]
fn openfst_interop() -> Result<()> {
    // Read OpenFST format
    let fst: VectorFst<TropicalWeight> = read_openfst("model.fst")?;
    
    // Process with ArcWeight
    let optimized = minimize(&fst)?;
    
    // Write back to OpenFST format
    write_openfst(&optimized, "optimized.fst")?;
    
    Ok(())
}
```

## Testing and Validation

### Property Testing
```rust
use arcweight::prelude::*;

fn validate_fst_properties(fst: &VectorFst<TropicalWeight>) -> Result<()> {
    // Check basic properties
    assert!(fst.start().is_some(), "FST must have a start state");
    
    // Verify FST is well-formed
    for state in fst.states() {
        for arc in fst.arcs(state) {
            assert!(arc.nextstate < fst.num_states() as StateId, 
                   "Arc points to invalid state");
        }
    }
    
    // Check semiring properties
    let w1 = TropicalWeight::new(1.0);
    let w2 = TropicalWeight::new(2.0);
    
    // Associativity: (w1 + w2) + w3 = w1 + (w2 + w3)
    let w3 = TropicalWeight::new(3.0);
    let left = w1.plus(&w2).plus(&w3);
    let right = w1.plus(&w2.plus(&w3));
    assert_eq!(left, right, "Addition must be associative");
    
    Ok(())
}
```

## Error Handling

### Comprehensive Error Management
```rust
use arcweight::prelude::*;

fn robust_fst_operations() -> Result<()> {
    let fst1 = VectorFst::<TropicalWeight>::new();
    let fst2 = VectorFst::<TropicalWeight>::new();
    
    match compose(&fst1, &fst2) {
        Ok(result) => {
            println!("Composition successful");
            Ok(())
        }
        Err(Error::InvalidOperation(msg)) => {
            eprintln!("Invalid FST operation: {}", msg);
            Err(Error::InvalidOperation(msg))
        }
        Err(Error::Algorithm(msg)) => {
            eprintln!("Algorithm error: {}", msg);
            Err(Error::Algorithm(msg))
        }
        Err(e) => {
            eprintln!("Unexpected error: {}", e);
            Err(e)
        }
    }
}
```

## Advanced Topics

### Lazy Evaluation
```rust
use arcweight::prelude::*;

struct LazyComposition<F1, F2, W> 
where
    F1: Fst<W>,
    F2: Fst<W>,
    W: Semiring,
{
    fst1: F1,
    fst2: F2,
    cache: std::collections::HashMap<StateId, Vec<Arc<W>>>,
}

impl<F1, F2, W> LazyComposition<F1, F2, W>
where
    F1: Fst<W>,
    F2: Fst<W>,
    W: Semiring,
{
    fn new(fst1: F1, fst2: F2) -> Self {
        Self {
            fst1,
            fst2,
            cache: std::collections::HashMap::new(),
        }
    }
}
```

### Integration Patterns
```rust
use arcweight::prelude::*;

// Builder pattern for complex FST construction
struct FstBuilder<W: Semiring> {
    fst: VectorFst<W>,
    symbol_table: SymbolTable,
}

impl<W: Semiring> FstBuilder<W> {
    fn new() -> Self {
        Self {
            fst: VectorFst::new(),
            symbol_table: SymbolTable::new(),
        }
    }
    
    fn add_word(mut self, word: &str, weight: W) -> Self {
        // Implementation
        self
    }
    
    fn build(self) -> (VectorFst<W>, SymbolTable) {
        (self.fst, self.symbol_table)
    }
}
```

## Best Practices

### Design Guidelines
1. **Start Simple**: Begin with basic FST operations before attempting complex compositions
2. **Use Appropriate Semirings**: Choose semirings that match your problem domain
3. **Optimize Incrementally**: Profile before optimizing; use properties to guide optimization
4. **Handle Errors Gracefully**: Always use `Result` types and handle all error cases
5. **Document Invariants**: Clearly document what properties your FSTs maintain

### Performance Tips
1. **Minimize Before Composition**: Smaller FSTs compose faster
2. **Use Deterministic FSTs**: When possible, as they enable optimizations
3. **Cache Expensive Operations**: Store results of costly computations
4. **Choose Right FST Type**: `VectorFst` for flexibility, `ConstFst` for read-only access
5. **Enable Parallelization**: Use the `parallel` feature for large-scale processing

### Testing Strategies
1. **Property-Based Testing**: Use randomized inputs to verify algebraic properties
2. **Round-Trip Testing**: Serialize and deserialize to verify data integrity
3. **Benchmarking**: Regular performance testing to catch regressions
4. **Integration Testing**: Test realistic use cases end-to-end

## Examples and Tutorials

For complete working examples, see:
- [Edit Distance](examples/edit_distance.md) - Spell checking and fuzzy matching
- [Morphological Analyzer](examples/morphological_analyzer.md) - Linguistic analysis
- [Phonological Rules](examples/phonological_rules.md) - Sound change modeling
- [Advanced Usage](examples/advanced-usage.md) - Complex FST operations
- [Cookbook](examples/cookbook.md) - Common patterns and solutions

## Further Resources

- **API Documentation**: Run `cargo doc --open` for complete API reference
- **Benchmarks**: See `docs/benchmarks.md` for performance characteristics
- **Architecture**: See `docs/architecture.md` for implementation details
- **Contributing**: See `CONTRIBUTING.md` for development guidelines
