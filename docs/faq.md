# Frequently Asked Questions

This document answers common questions about ArcWeight, from basic usage to advanced optimization techniques.

## Getting Started

### What is ArcWeight?
ArcWeight is a high-performance Rust library for weighted finite state transducers (WFSTs). It provides a comprehensive toolkit for constructing, manipulating, optimizing, and analyzing FSTs with support for various semiring types and algorithms.

### What makes ArcWeight different from other FST libraries?
- **Type-safe design**: Leverages Rust's type system for correctness
- **Generic semiring support**: Works with any mathematical semiring
- **Trait-based architecture**: Clean interfaces and easy extensibility
- **High performance**: Zero-cost abstractions and optimized algorithms
- **OpenFST compatibility**: Read/write OpenFST format files
- **Modern Rust patterns**: Uses async, parallel processing, and memory safety

### What are the system requirements?
- **Rust**: 1.70.0 or later
- **Memory**: 4GB RAM recommended (2GB minimum)
- **Storage**: 200MB for library and dependencies
- **Platforms**: Linux, macOS, Windows (x86_64, ARM64)
- **Optional**: GPU support for acceleration (feature flag)

## Installation and Setup

### How do I install ArcWeight?
Add to your `Cargo.toml`:
```toml
[dependencies]
arcweight = "0.1.0"
```

For development features:
```toml
[dependencies]
arcweight = { version = "0.1.0", features = ["parallel", "serde", "openfst"] }
```

### What features are available?
- `parallel`: Multi-threaded algorithms via Rayon
- `serde`: Serialization support for weights and FSTs
- `openfst`: OpenFST format compatibility
- `gpu`: GPU acceleration (experimental)
- `python`: Python bindings
- `std`: Standard library support (enabled by default)

### How do I update ArcWeight?
```bash
cargo update arcweight
```

To update to a specific version:
```bash
cargo update arcweight --precise 0.2.0
```

## Basic Usage

### How do I create my first FST?
```rust
use arcweight::prelude::*;

fn main() -> Result<()> {
    // Create a simple acceptor for "hello"
    let mut fst = VectorFst::<TropicalWeight>::new();
    
    // Add states
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    
    // Set start and final states
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());
    
    // Add arc (input_label, output_label, weight, next_state)
    fst.add_arc(s0, Arc::new(104, 104, TropicalWeight::one(), s1)); // 'h'
    
    println!("Created FST with {} states", fst.num_states());
    Ok(())
}
```

### What semirings are available?
- `TropicalWeight`: Min-plus semiring (most common, for shortest paths)
- `ProbabilityWeight`: Real probabilities [0,1]
- `BooleanWeight`: Boolean semiring for unweighted FSTs
- `LogWeight`: Log semiring for numerical stability
- `MinWeight`/`MaxWeight`: Min/max semirings
- `StringWeight`: String concatenation semiring
- `ProductWeight`: Cartesian product of two semirings

### How do I choose the right semiring?
- **Shortest path problems**: Use `TropicalWeight`
- **Probabilistic models**: Use `ProbabilityWeight` or `LogWeight`
- **Unweighted FSTs**: Use `BooleanWeight`
- **Custom needs**: Implement your own `Semiring` trait

## Core Operations

### How do I compose two FSTs?
```rust
use arcweight::prelude::*;

let composed = compose(&fst1, &fst2)?;
// Or with custom filter
let filtered = compose_with_filter(&fst1, &fst2, ComposeFilter::Sequence)?;
```

### How do I determinize an FST?
```rust
let determinized = determinize(&fst)?;
```

### How do I minimize an FST?
```rust
let minimized = minimize(&fst)?;
```

### How do I find shortest paths?
```rust
// Single shortest path
let shortest = shortest_path_single(&fst)?;

// Multiple shortest paths
let config = ShortestPathConfig::new().nshortest(5);
let paths = shortest_path(&fst, &config)?;
```

### How do I optimize an FST?
```rust
// Basic optimization chain
let optimized = fst
    .determinize()?
    .minimize()?
    .remove_epsilons()?;

// Or use the optimization function
let optimized = optimize_fst(&fst)?;
```

## Performance and Optimization

### How do I improve FST performance?
1. **Choose appropriate FST type**:
   - `VectorFst`: General purpose, mutable
   - `ConstFst`: Read-only, space efficient
   - `CompactFst`: Memory optimized

2. **Optimize your FSTs**:
   ```rust
   let optimized = fst.determinize()?.minimize()?;
   ```

3. **Use properties for smart algorithms**:
   ```rust
   if fst.properties().is_deterministic() {
       // Use faster deterministic algorithms
   }
   ```

4. **Enable parallel processing**:
   ```rust
   use rayon::prelude::*;
   
   let results: Vec<_> = fsts.par_iter()
       .map(|fst| process_fst(fst))
       .collect();
   ```

### How do I profile FST operations?
```rust
use std::time::Instant;

fn profile_operation<F, T>(name: &str, operation: F) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    let start = Instant::now();
    let result = operation()?;
    let duration = start.elapsed();
    
    println!("{}: {:?}", name, duration);
    Ok(result)
}

// Usage
let result = profile_operation("composition", || {
    compose(&fst1, &fst2)
})?;
```

### How do I reduce memory usage?
1. **Use compact FST types**:
   ```rust
   let compact = CompactFst::from_vector_fst(&vector_fst)?;
   ```

2. **Enable compression**:
   ```rust
   let compressed = fst.compress_arcs()?;
   ```

3. **Remove unnecessary states**:
   ```rust
   let cleaned = fst.connect()?.minimize()?;
   ```

4. **Use streaming for large datasets**:
   ```rust
   for chunk in large_fst.chunks(1000) {
       process_chunk(chunk)?;
   }
   ```

## Troubleshooting

### Common Errors and Solutions

#### `InvalidOperation` Error
```rust
Error::InvalidOperation("Cannot compose FSTs with incompatible semirings")
```
**Solution**: Ensure both FSTs use the same semiring type, or convert between them:
```rust
let converted = weight_convert(&fst, |w: &TropicalWeight| ProbabilityWeight::new(w.value()))?;
```

#### `Algorithm` Error
```rust
Error::Algorithm("Determinization failed: infinite loop detected")
```
**Solution**: Check for epsilon cycles or use bounded determinization:
```rust
let det = determinize_with_bounds(&fst, 10000, 100000)?;
```

#### Memory Issues
```rust
Error::OutOfMemory
```
**Solution**: 
1. Use more memory-efficient FST types
2. Process in smaller chunks
3. Enable compression
4. Increase system memory

#### Performance Issues
**Problem**: Operations are too slow
**Solutions**:
1. Check FST properties first:
   ```rust
   let props = fst.properties();
   if !props.is_deterministic() {
       fst = determinize(&fst)?;
   }
   ```

2. Use appropriate algorithms:
   ```rust
   if props.is_acyclic() {
       // Use specialized acyclic algorithms
       let result = acyclic_shortest_path(&fst)?;
   }
   ```

### Debugging Techniques

#### Enable verbose logging:
```bash
RUST_LOG=arcweight=debug cargo run
```

#### Verify FST properties:
```rust
fn debug_fst(fst: &VectorFst<TropicalWeight>) -> Result<()> {
    println!("States: {}", fst.num_states());
    println!("Arcs: {}", fst.num_arcs_total());
    
    let props = fst.properties();
    println!("Deterministic: {}", props.is_deterministic());
    println!("Acyclic: {}", props.is_acyclic());
    println!("Connected: {}", props.is_connected());
    
    Ok(())
}
```

#### Validate FST structure:
```rust
fn validate_fst(fst: &VectorFst<TropicalWeight>) -> Result<()> {
    // Check all arcs point to valid states
    for state in fst.states() {
        for arc in fst.arcs(state) {
            if arc.nextstate >= fst.num_states() as StateId {
                return Err(Error::InvalidOperation("Arc points to invalid state".to_string()));
            }
        }
    }
    Ok(())
}
```

## Advanced Topics

### How do I implement a custom semiring?
```rust
use arcweight::prelude::*;

#[derive(Debug, Clone, PartialEq)]
struct FeatureWeight {
    cost: f32,
    features: Vec<f32>,
}

impl Semiring for FeatureWeight {
    fn plus(&self, other: &Self) -> Self {
        // Take minimum cost
        if self.cost <= other.cost {
            self.clone()
        } else {
            other.clone()
        }
    }
    
    fn times(&self, other: &Self) -> Self {
        FeatureWeight {
            cost: self.cost + other.cost,
            features: self.features.iter()
                .zip(&other.features)
                .map(|(a, b)| a + b)
                .collect(),
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

### How do I implement custom FST types?
```rust
use arcweight::prelude::*;

struct MyCustomFst<W: Semiring> {
    data: Vec<Vec<Arc<W>>>,
    start: Option<StateId>,
    finals: Vec<Option<W>>,
}

impl<W: Semiring> Fst<W> for MyCustomFst<W> {
    type ArcIter<'a> = std::slice::Iter<'a, Arc<W>> where Self: 'a;
    
    fn start(&self) -> Option<StateId> {
        self.start
    }
    
    fn final_weight(&self, state: StateId) -> Option<&W> {
        self.finals.get(state as usize)?.as_ref()
    }
    
    fn num_arcs(&self, state: StateId) -> usize {
        self.data.get(state as usize).map_or(0, |arcs| arcs.len())
    }
    
    fn num_states(&self) -> usize {
        self.data.len()
    }
    
    fn properties(&self) -> FstProperties {
        // Compute and return properties
        FstProperties::default()
    }
    
    fn arcs(&self, state: StateId) -> Self::ArcIter<'_> {
        self.data[state as usize].iter()
    }
}
```

### How do I use FSTs with async code?
```rust
use arcweight::prelude::*;
use tokio;

async fn async_fst_processing(
    fsts: Vec<VectorFst<TropicalWeight>>,
) -> Result<Vec<VectorFst<TropicalWeight>>> {
    let mut results = Vec::new();
    
    for fst in fsts {
        // Process FST asynchronously
        let result = tokio::task::spawn_blocking(move || {
            fst.determinize()?.minimize()
        }).await??;
        
        results.push(result);
    }
    
    Ok(results)
}
```

## Integration and Interoperability

### How do I use ArcWeight with OpenFST?
```rust
#[cfg(feature = "openfst")]
use arcweight::prelude::*;

#[cfg(feature = "openfst")]
fn openfst_integration() -> Result<()> {
    // Read OpenFST file
    let fst: VectorFst<TropicalWeight> = read_openfst("model.fst")?;
    
    // Process with ArcWeight
    let optimized = fst.determinize()?.minimize()?;
    
    // Write back to OpenFST format
    write_openfst(&optimized, "optimized.fst")?;
    
    Ok(())
}
```

### How do I serialize FSTs?
```rust
#[cfg(feature = "serde")]
use arcweight::prelude::*;

#[cfg(feature = "serde")]
fn serialization_example() -> Result<()> {
    let fst = VectorFst::<TropicalWeight>::new();
    
    // Serialize to JSON
    let json = serde_json::to_string(&fst)?;
    
    // Deserialize from JSON
    let loaded_fst: VectorFst<TropicalWeight> = serde_json::from_str(&json)?;
    
    // Binary formats are also supported
    let binary = bincode::serialize(&fst)?;
    let loaded_fst: VectorFst<TropicalWeight> = bincode::deserialize(&binary)?;
    
    Ok(())
}
```

### How do I create Python bindings?
```rust
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyclass]
struct PyVectorFst {
    inner: VectorFst<TropicalWeight>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyVectorFst {
    #[new]
    fn new() -> Self {
        Self {
            inner: VectorFst::new(),
        }
    }
    
    fn add_state(&mut self) -> u32 {
        self.inner.add_state()
    }
    
    fn determinize(&self) -> PyResult<PyVectorFst> {
        Ok(PyVectorFst {
            inner: self.inner.determinize().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e))
            })?,
        })
    }
}
```

## Best Practices

### Design Guidelines
1. **Start simple**: Begin with basic operations before complex compositions
2. **Understand your data**: Know the properties of your FSTs
3. **Profile early**: Measure performance before optimizing
4. **Use appropriate types**: Choose FST and weight types for your use case
5. **Handle errors**: Always use `Result` types and proper error handling

### Code Organization
```rust
// Good: Organize by functionality
mod lexicon {
    use arcweight::prelude::*;
    
    pub fn build_pronunciation_fst() -> Result<VectorFst<TropicalWeight>> {
        // Implementation
    }
}

mod morphology {
    use arcweight::prelude::*;
    
    pub fn build_morphology_fst() -> Result<VectorFst<TropicalWeight>> {
        // Implementation  
    }
}

mod pipeline {
    use super::*;
    
    pub fn build_complete_system() -> Result<VectorFst<TropicalWeight>> {
        let lexicon = lexicon::build_pronunciation_fst()?;
        let morphology = morphology::build_morphology_fst()?;
        compose(&lexicon, &morphology)
    }
}
```

### Testing Strategies
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    #[test]
    fn test_fst_properties() {
        let fst = create_test_fst();
        assert!(fst.properties().is_deterministic());
    }
    
    proptest! {
        #[test]
        fn test_composition_associativity(
            fst1 in any_fst(),
            fst2 in any_fst(), 
            fst3 in any_fst()
        ) {
            let left = compose(&compose(&fst1, &fst2)?, &fst3)?;
            let right = compose(&fst1, &compose(&fst2, &fst3)?)?;
            
            // They should be equivalent (up to state renaming)
            assert_equivalent(&left, &right);
        }
    }
}
```

## Community and Support

### Where can I get help?
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community support  
- **Documentation**: Comprehensive API docs and examples
- **Matrix Chat**: Real-time community discussion

### How do I contribute?
1. **Read the contributing guide**: `CONTRIBUTING.md`
2. **Check existing issues**: Avoid duplicating work
3. **Start small**: Begin with documentation or small features
4. **Follow conventions**: Use `cargo fmt` and `cargo clippy`
5. **Add tests**: Ensure your changes are well-tested
6. **Update docs**: Keep documentation current

### How do I report bugs?
1. **Search existing issues** first
2. **Use the issue template** 
3. **Provide minimal reproduction** case
4. **Include system information**:
   - Rust version (`rustc --version`)
   - ArcWeight version
   - Operating system
   - Error messages and stack traces

### How do I request features?
1. **Check the roadmap** for planned features
2. **Describe the use case** clearly
3. **Provide examples** of how it would be used
4. **Consider implementation** complexity
5. **Offer to help** with implementation

## License and Legal

### What license is ArcWeight under?
ArcWeight is licensed under the Apache License 2.0, which allows:
- Commercial use
- Modification
- Distribution
- Private use
- Patent use

### How do I properly attribute ArcWeight?
Include the license file and copyright notice in your distribution:
```
Copyright (c) 2025 Aaron Steven White
Licensed under the Apache License, Version 2.0
```

### Can I use ArcWeight in proprietary software?
Yes, the Apache License 2.0 permits use in proprietary software without requiring you to open-source your code. However, you must include the license text and any attribution notices.

## Roadmap and Future

### What features are planned?
- **GPU acceleration** for large-scale operations
- **Better parallelization** with work-stealing algorithms
- **Streaming FST operations** for massive datasets
- **Neural FST integration** for hybrid models
- **WebAssembly support** for browser applications
- **Mobile optimization** for iOS/Android

### How can I stay updated?
- **Watch the repository** for releases and updates
- **Follow the project blog** for announcements
- **Join the community chat** for discussions
- **Subscribe to the newsletter** for quarterly updates

### How do I influence the roadmap?
- **Participate in discussions** about new features
- **Contribute code** for features you need
- **Sponsor development** for priority features
- **Provide feedback** on experimental features 