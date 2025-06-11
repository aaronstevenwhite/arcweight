# Getting Started with ArcWeight

This guide will walk you through creating and manipulating a simple FST using ArcWeight.

## Basic FST Creation

Let's create a simple FST that transduces between input and output symbols:

```rust
use arcweight::prelude::*;

fn main() -> Result<()> {
    // Create a new FST
    let mut fst = Fst::new();
    
    // Add states
    let start = fst.add_state();
    let end = fst.add_state();
    
    // Set start and final states
    fst.set_start(start)?;
    fst.set_final(end, Weight::one())?;
    
    // Add arcs
    fst.add_arc(start, Arc::new(1, 2, Weight::one(), end))?;
    fst.add_arc(start, Arc::new(3, 4, Weight::one(), end))?;
    
    // Verify the FST
    assert!(fst.verify()?);
    
    // Print the FST
    println!("FST: {}", fst);
    
    Ok(())
}
```

## Working with Weights

Here's how to work with different weight types:

```rust
use arcweight::prelude::*;
use arcweight::semiring::{TropicalWeight, LogWeight};

fn main() -> Result<()> {
    let mut fst = Fst::new();
    let start = fst.add_state();
    let end = fst.add_state();
    
    // Using Tropical weights
    let tropical_weight = TropicalWeight::new(0.5);
    fst.add_arc(start, Arc::new(1, 2, tropical_weight, end))?;
    
    // Using Log weights
    let log_weight = LogWeight::new(0.5);
    fst.add_arc(start, Arc::new(3, 4, log_weight, end))?;
    
    Ok(())
}
```

## Basic FST Operations

Here's how to perform common FST operations:

```rust
use arcweight::prelude::*;

fn main() -> Result<()> {
    // Create two FSTs
    let mut fst1 = Fst::new();
    let mut fst2 = Fst::new();
    
    // ... populate FSTs ...
    
    // Compose FSTs
    let composed = fst1.compose(&fst2)?;
    
    // Determinize
    let determinized = composed.determinize()?;
    
    // Minimize
    let minimized = determinized.minimize()?;
    
    Ok(())
}
```

## Next Steps

- Try the [Advanced Usage](advanced-usage.md) guide for more complex examples
- Check out the [Cookbook](cookbook.md) for common patterns and solutions
- Explore the [API Reference](../api/README.md) for detailed documentation 