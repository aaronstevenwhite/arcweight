# Advanced Usage Guide

This guide covers advanced topics and complex patterns in ArcWeight.

## Complex FST Operations

### HCLG Decoder Graph Construction
```rust
use arcweight::prelude::*;

fn build_hclg_graph(
    hmm: &Fst,
    context: &Fst,
    lexicon: &Fst,
    grammar: &Fst
) -> Result<Fst> {
    // Compose HMM with context dependency
    let hc = hmm.compose(context)?;
    
    // Add lexicon
    let hcl = hc.compose(lexicon)?;
    
    // Add grammar
    let hclg = hcl.compose(grammar)?;
    
    // Optimize the final graph
    hclg.determinize()?.minimize()
}
```

### Phonological Rule Application
```rust
use arcweight::prelude::*;

fn apply_phonological_rules(fst: &Fst, rules: &[Fst]) -> Result<Fst> {
    let mut result = fst.clone();
    
    for rule in rules {
        // Compose with each rule
        result = result.compose(rule)?;
        
        // Optimize after each rule
        result = result.determinize()?.minimize()?;
    }
    
    Ok(result)
}
```

## Custom Semiring Implementations

### Confidence-Weighted Semiring
```rust
use arcweight::prelude::*;
use arcweight::semiring::Semiring;

#[derive(Clone, Debug)]
struct ConfidenceWeight {
    value: f32,
    confidence: f32,
}

impl Semiring for ConfidenceWeight {
    fn zero() -> Self {
        Self {
            value: f32::INFINITY,
            confidence: 0.0,
        }
    }
    
    fn one() -> Self {
        Self {
            value: 0.0,
            confidence: 1.0,
        }
    }
    
    fn plus(&self, other: &Self) -> Self {
        if self.confidence > other.confidence {
            self.clone()
        } else {
            other.clone()
        }
    }
    
    fn times(&self, other: &Self) -> Self {
        Self {
            value: self.value + other.value,
            confidence: self.confidence * other.confidence,
        }
    }
}
```

## Advanced FST Algorithms

### Lattice Rescoring
```rust
use arcweight::prelude::*;

fn rescore_lattice(lattice: &Fst, new_lm: &Fst) -> Result<Fst> {
    // Project to input labels
    let input = lattice.project(ProjectType::Input)?;
    
    // Compose with new language model
    let rescored = input.compose(new_lm)?;
    
    // Project back to output labels
    rescored.project(ProjectType::Output)
}
```

### FST Pruning
```rust
use arcweight::prelude::*;

fn prune_fst(fst: &Fst, beam: f32) -> Result<Fst> {
    // Get best path weight
    let best_weight = fst.shortest_distance()?;
    
    // Prune arcs with weight difference > beam
    let mut pruned = fst.clone();
    for state in pruned.states() {
        pruned.prune_arcs(state, best_weight, beam)?;
    }
    
    Ok(pruned)
}
```

## Parallel Processing

### Batch FST Processing
```rust
use arcweight::prelude::*;
use rayon::prelude::*;

fn parallel_process_fsts(fsts: &[Fst]) -> Result<Vec<Fst>> {
    fsts.par_iter()
        .map(|fst| {
            fst.determinize()?
                .minimize()?
                .optimize()
        })
        .collect()
}
```

### Parallel Composition
```rust
use arcweight::prelude::*;
use rayon::prelude::*;

fn parallel_compose(fst1: &Fst, fsts: &[Fst]) -> Result<Vec<Fst>> {
    fsts.par_iter()
        .map(|fst2| fst1.compose(fst2))
        .collect()
}
```

## Memory Optimization

### Arc Compression
```rust
use arcweight::prelude::*;

fn compress_arcs(fst: &Fst) -> Result<Fst> {
    let mut compressed = fst.clone();
    
    // Sort arcs by label
    compressed.sort_arcs()?;
    
    // Compress common prefixes
    compressed.compress_arcs()?;
    
    Ok(compressed)
}
```

### State Minimization
```rust
use arcweight::prelude::*;

fn minimize_states(fst: &Fst) -> Result<Fst> {
    // Remove unreachable states
    let reachable = fst.remove_unreachable()?;
    
    // Remove dead states
    let live = reachable.remove_dead()?;
    
    // Minimize
    live.minimize()
}
```

## Error Recovery

### FST Repair
```rust
use arcweight::prelude::*;

fn repair_fst(fst: &Fst) -> Result<Fst> {
    // Check for common issues
    if !fst.verify()? {
        // Remove problematic arcs
        let mut repaired = fst.clone();
        repaired.remove_problematic_arcs()?;
        
        // Rebuild connectivity
        repaired.rebuild_connectivity()?;
        
        // Verify repair
        if !repaired.verify()? {
            return Err(Error::new("FST repair failed"));
        }
        
        Ok(repaired)
    } else {
        Ok(fst.clone())
    }
}
```

## Performance Monitoring

### FST Profiling
```rust
use arcweight::prelude::*;
use std::time::Instant;

fn profile_fst_operation<F, T>(fst: &Fst, operation: F) -> Result<T>
where
    F: FnOnce(&Fst) -> Result<T>,
{
    let start = Instant::now();
    let result = operation(fst)?;
    let duration = start.elapsed();
    
    println!("Operation took: {:?}", duration);
    Ok(result)
}
```

## Integration Patterns

### Pipeline Processing
```rust
use arcweight::prelude::*;

struct FstPipeline {
    steps: Vec<Box<dyn Fn(&Fst) -> Result<Fst>>>,
}

impl FstPipeline {
    fn new() -> Self {
        Self { steps: Vec::new() }
    }
    
    fn add_step<F>(&mut self, step: F)
    where
        F: Fn(&Fst) -> Result<Fst> + 'static,
    {
        self.steps.push(Box::new(step));
    }
    
    fn process(&self, mut fst: Fst) -> Result<Fst> {
        for step in &self.steps {
            fst = step(&fst)?;
        }
        Ok(fst)
    }
}
```

## Testing and Validation

### Property-Based Testing
```rust
use arcweight::prelude::*;
use proptest::prelude::*;

fn test_fst_properties() {
    proptest!(|(states: usize, arcs: usize)| {
        let mut fst = Fst::new();
        
        // Generate random FST
        for _ in 0..states {
            fst.add_state();
        }
        
        // Add random arcs
        for _ in 0..arcs {
            let from = (0..states).choose(&mut thread_rng()).unwrap();
            let to = (0..states).choose(&mut thread_rng()).unwrap();
            fst.add_arc(from, Arc::new(1, 2, Weight::one(), to))?;
        }
        
        // Verify properties
        assert!(fst.verify()?);
        
        // Test determinization
        let det = fst.determinize()?;
        assert!(det.is_deterministic()?);
        
        // Test minimization
        let min = det.minimize()?;
        assert!(min.is_minimal()?);
    });
} 