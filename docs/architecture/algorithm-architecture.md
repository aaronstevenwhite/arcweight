# Algorithm Architecture

## Algorithm Organization

**API Reference**: [`algorithms`](https://docs.rs/arcweight/latest/arcweight/algorithms/)

The algorithms module is organized around several key principles:

**Functional Design**: All algorithms are pure functions that take input FSTs and return new FSTs, avoiding mutation of input data.

**Generic Implementation**: Algorithms work with any FST type implementing the required traits, enabling code reuse across different storage strategies.

**Configuration-Driven**: Complex algorithms accept configuration structs to control behavior without requiring multiple function variants.

**Property-Aware**: Algorithms can optimize based on known FST properties, avoiding unnecessary computation when possible.

## Common Patterns

All algorithms follow consistent patterns:

### 1. **Configuration Structs**
```rust,ignore
#[derive(Debug, Clone)]
pub struct ShortestPathConfig {
    pub nshortest: usize,
    pub unique: bool,
    pub weight_threshold: Option<f64>,
    pub state_threshold: Option<usize>,
}
```

### 2. **Generic Type Parameters**
```rust,ignore
pub fn compose<W, F1, F2>(
    fst1: &F1, 
    fst2: &F2, 
    filter: impl ComposeFilter<W>
) -> Result<VectorFst<W>>
where
    W: Semiring,
    F1: Fst<W>,
    F2: Fst<W>,
```

### 3. **Property-Based Optimization**
```rust,ignore
pub fn shortest_path<F, W>(fst: &F, config: ShortestPathConfig) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: NaturallyOrderedSemiring,  // Requires specific semiring property
```

## Algorithm Categories

### Core Operations

**Composition** - Fundamental FST operation:
```rust,ignore
pub fn compose<W, F1, F2, Filter>(
    fst1: &F1,
    fst2: &F2,
    filter: Filter,
) -> Result<VectorFst<W>>
where
    W: Semiring,
    F1: Fst<W>,
    F2: Fst<W>,
    Filter: ComposeFilter<W>,
```

**Union** - Combines multiple FSTs:
```rust,ignore
pub fn union<W, F1, F2>(fst1: &F1, fst2: &F2) -> Result<VectorFst<W>>
where
    W: Semiring,
    F1: Fst<W>,
    F2: Fst<W>,
```

**Concatenation** - Sequential composition:
```rust,ignore
pub fn concat<W, F1, F2>(fst1: &F1, fst2: &F2) -> Result<VectorFst<W>>
where
    W: Semiring,
    F1: Fst<W>,
    F2: Fst<W>,
```

### Optimization Operations

**Minimization** - Reduces FST size while preserving semantics:
```rust,ignore
pub fn minimize<F, W>(fst: &F) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: DivisibleSemiring,
```

**Determinization** - Makes FST deterministic:
```rust,ignore
pub fn determinize<F, W>(fst: &F) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: Semiring,
```

**Epsilon Removal** - Eliminates epsilon transitions:
```rust,ignore
pub fn remove_epsilons<F, W>(fst: &F) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: Semiring,
```

### Path Operations

**Shortest Path** - Finds optimal paths:
```rust,ignore
pub fn shortest_path<F, W>(
    fst: &F,
    config: ShortestPathConfig,
) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: NaturallyOrderedSemiring,
```

**Random Generation** - Generates random paths:
```rust,ignore
pub fn randgen<F, W>(
    fst: &F,
    config: RandGenConfig,
) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: Semiring,
```

### Structural Operations

**Reversal** - Reverses FST structure:
```rust,ignore
pub fn reverse<F, W>(fst: &F) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: Semiring,
```

**Projection** - Projects input or output:
```rust,ignore
pub fn project<F, W>(fst: &F, project_type: ProjectType) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: Semiring,
```

## Algorithm Implementation Patterns

### State Processing

Most algorithms follow a common state processing pattern:

```rust,ignore
pub fn generic_algorithm<F, W>(fst: &F) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: Semiring,
{
    let mut result = VectorFst::new();
    let mut state_map = HashMap::new();
    let mut queue = VecDeque::new();
    
    // Initialize with start state
    if let Some(start) = fst.start() {
        let new_start = result.add_state();
        result.set_start(new_start);
        state_map.insert(start, new_start);
        queue.push_back(start);
    }
    
    // Process states
    while let Some(state) = queue.pop_front() {
        let new_state = state_map[&state];
        
        // Process arcs
        for arc in fst.arcs(state) {
            let next_state = get_or_create_state(arc.nextstate, &mut result, &mut state_map, &mut queue);
            let new_arc = Arc::new(arc.ilabel, arc.olabel, arc.weight.clone(), next_state);
            result.add_arc(new_state, new_arc);
        }
        
        // Handle final states
        if let Some(weight) = fst.final_weight(state) {
            result.set_final(new_state, weight.clone());
        }
    }
    
    Ok(result)
}
```

### Property-Based Optimization

Algorithms leverage FST properties for optimization:

```rust,ignore
pub fn optimized_operation<F, W>(fst: &F) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: Semiring,
{
    let props = fst.properties();
    
    match props.properties {
        p if p.contains(PropertyFlags::ACYCLIC) => {
            // Use specialized acyclic algorithm
            process_acyclic(fst)
        }
        p if p.contains(PropertyFlags::NO_EPSILONS) => {
            // Skip epsilon handling
            process_no_epsilon(fst)
        }
        p if p.contains(PropertyFlags::DETERMINISTIC) => {
            // Use deterministic-specific optimizations
            process_deterministic(fst)
        }
        _ => {
            // General case
            process_general(fst)
        }
    }
}
```

### Configuration-Driven Behavior

Complex algorithms use configuration structs:

```rust,ignore
#[derive(Debug, Clone)]
pub struct ComposeConfig {
    pub filter: ComposeFilterType,
    pub connect: bool,
    pub sort_arcs: bool,
}

impl Default for ComposeConfig {
    fn default() -> Self {
        Self {
            filter: ComposeFilterType::Auto,
            connect: true,
            sort_arcs: false,
        }
    }
}

pub fn compose_with_config<F1, F2, W>(
    fst1: &F1,
    fst2: &F2,
    config: ComposeConfig,
) -> Result<VectorFst<W>>
where
    F1: Fst<W>,
    F2: Fst<W>,
    W: Semiring,
{
    let filter = create_filter(config.filter, fst1, fst2);
    let mut result = compose_internal(fst1, fst2, filter)?;
    
    if config.connect {
        result = connect(&result)?;
    }
    
    if config.sort_arcs {
        arc_sort(&mut result);
    }
    
    Ok(result)
}
```

## Performance Considerations

### Lazy Evaluation

Some algorithms support lazy evaluation for better performance:

```rust,ignore
pub fn compose_lazy<F1, F2, W>(
    fst1: &F1,
    fst2: &F2,
) -> LazyFstImpl<W>
where
    F1: Fst<W>,
    F2: Fst<W>,
    W: Semiring,
{
    LazyFstImpl::new(move |state| {
        // Compute arcs on-demand
        let (s1, s2) = decode_composed_state(state);
        compute_composed_arcs(s1, s2, fst1, fst2)
    })
}
```

### Memory Efficiency

Algorithms are designed to minimize memory allocations:

```rust,ignore
// Reuse existing state mappings
let mut state_map = HashMap::with_capacity(fst.num_states());

// Pre-allocate result FST
let mut result = VectorFst::with_capacity(estimated_states, estimated_arcs);

// Use iterators to avoid intermediate collections
fst.states()
    .filter(|&state| is_accessible(state))
    .for_each(|state| process_state(state, &mut result));
```

This algorithm architecture ensures consistency, performance, and maintainability across all FST operations.