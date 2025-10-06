# Algorithm Architecture

## Design Philosophy

**API Reference**: [`algorithms`](https://docs.rs/arcweight/latest/arcweight/algorithms/)

The algorithm architecture follows formal principles from automata theory and functional programming:

**Functional Design**: Algorithms are mathematically pure functions f: FST → FST, ensuring referential transparency and preventing side effects on input data.

**Generic Implementation**: Type-parameterized algorithms work with any FST implementation satisfying the required trait bounds, enabling code reuse across storage strategies.

**Configuration-Driven**: Complex algorithms accept configuration structures to parameterize behavior, following the strategy pattern.

**Property-Aware Optimization**: Algorithms leverage formal FST properties to select optimal computational strategies, reducing complexity when mathematical preconditions are satisfied.

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

**Composition** - Fundamental FST operation implementing the mathematical composition T₁ ∘ T₂:

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

**Pseudocode for Composition:**
```text
Algorithm: FST_COMPOSE(T₁, T₂)
Input: FSTs T₁ = (Q₁, Σ, Δ₁, δ₁, λ₁, q₁⁰, F₁), 
            T₂ = (Q₂, Σ, Δ₂, δ₂, λ₂, q₂⁰, F₂)
Output: FST T = (Q, Σ, Δ₂, δ, λ, q⁰, F)

1. Initialize result FST T with empty state set Q
2. Create state mapping: state_map = ∅
3. Initialize queue with start state pair (q₁⁰, q₂⁰)
4. Add initial state to T: q⁰ = ADD_STATE(T)
5. state_map[(q₁⁰, q₂⁰)] = q⁰

6. While queue is not empty:
   a. (s₁, s₂) = DEQUEUE(queue)
   b. current_state = state_map[(s₁, s₂)]
   
   c. For each arc (i, o₁, w₁, t₁) ∈ δ₁(s₁):
      d. For each arc (i', o₂, w₂, t₂) ∈ δ₂(s₂):
         e. If FILTER_MATCH(i, o₁, i', o₂):
            f. If (t₁, t₂) ∉ state_map:
               g. new_state = ADD_STATE(T)
               h. state_map[(t₁, t₂)] = new_state
               i. ENQUEUE(queue, (t₁, t₂))
            j. target = state_map[(t₁, t₂)]
            k. ADD_ARC(T, current_state, (i, o₂, w₁ ⊗ w₂, target))

7. Set final weights: F(state_map[(s₁, s₂)]) = F₁(s₁) ⊗ F₂(s₂)
8. Return T
```

**Union** - Combines FSTs implementing T₁ ∪ T₂:
```rust,ignore
pub fn union<W, F1, F2>(fst1: &F1, fst2: &F2) -> Result<VectorFst<W>>
where
    W: Semiring,
    F1: Fst<W>,
    F2: Fst<W>,
```

**Concatenation** - Sequential composition T₁ · T₂:
```rust,ignore
pub fn concat<W, F1, F2>(fst1: &F1, fst2: &F2) -> Result<VectorFst<W>>
where
    W: Semiring,
    F1: Fst<W>,
    F2: Fst<W>,
```

### Optimization Operations

**Minimization** - Reduces FST size while preserving language equivalence:
```rust,ignore
pub fn minimize<F, W>(fst: &F) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: DivisibleSemiring,
```

**Pseudocode for Minimization (Hopcroft's Algorithm):**
```text
Algorithm: FST_MINIMIZE(T)
Input: FST T = (Q, Σ, Δ, δ, λ, q⁰, F)
Output: Minimal FST T' equivalent to T

1. Initialize partition P = {Q_final, Q_non_final}
2. Initialize work_list W = {(Q_final, a) | a ∈ Σ}

3. While W is not empty:
   a. Remove (S, a) from W
   b. For each state t with arc (t, a, w, s) where s ∈ S:
      c. Find partition class C containing t
      d. Split C into C₁ = {states in C with arc to S} and C₂ = remainder
      e. If |C₁| > 0 and |C₂| > 0:
         f. Replace C with C₁ and C₂ in partition P
         g. Add (C₁, b) and (C₂, b) to W for all b ∈ Σ

4. Construct minimal FST T' with states corresponding to partition classes
5. Return T'
```

**Determinization** - Converts non-deterministic FST to deterministic FST:
```rust,ignore
pub fn determinize<F, W>(fst: &F) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: Semiring,
```

**Pseudocode for Determinization (Subset Construction):**
```text
Algorithm: FST_DETERMINIZE(T)
Input: FST T = (Q, Σ, Δ, δ, λ, q⁰, F)
Output: Deterministic FST T' equivalent to T

1. Initialize result FST T' with empty state set
2. Create state mapping: state_map = ∅
3. Initialize with ε-closure: S₀ = ε-CLOSURE({q⁰})
4. Add initial state: q'⁰ = ADD_STATE(T')
5. state_map[S₀] = q'⁰

6. Initialize work_list with S₀
7. While work_list is not empty:
   a. Remove state set S from work_list
   b. current_state = state_map[S]
   
   c. For each symbol a ∈ Σ:
      d. T = ε-CLOSURE(δ(S, a))  // States reachable from S via a
      e. If T ≠ ∅:
         f. If T ∉ state_map:
            g. new_state = ADD_STATE(T')
            h. state_map[T] = new_state
            i. Add T to work_list
         j. target = state_map[T]
         k. weight = ⊕{w | (s, a, w, t) ∈ δ, s ∈ S, t ∈ T}
         l. ADD_ARC(T', current_state, (a, a, weight, target))

8. Set final weights: F'(state_map[S]) = ⊕{F(s) | s ∈ S}
9. Return T'
```

**Epsilon Removal** - Eliminates ε-transitions:
```rust,ignore
pub fn remove_epsilons<F, W>(fst: &F) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: Semiring,
```

### Path Operations

**Shortest Path** - Finds optimal paths using Dijkstra's algorithm:
```rust,ignore
pub fn shortest_path<F, W>(
    fst: &F,
    config: ShortestPathConfig,
) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: NaturallyOrderedSemiring,
```

**Pseudocode for Shortest Path (Dijkstra's Algorithm):**
```text
Algorithm: FST_SHORTEST_PATH(T, config)
Input: FST T = (Q, Σ, Δ, δ, λ, q⁰, F), configuration config
Output: FST containing n-shortest paths

1. Initialize priority queue PQ with (q⁰, 1̄)
2. Initialize distance array: d[s] = 0̄ for all s ∈ Q
3. Set d[q⁰] = 1̄
4. Initialize result FST T' and state mapping

5. While PQ is not empty and |paths found| < config.nshortest:
   a. (current_state, distance) = EXTRACT_MIN(PQ)
   b. If distance > d[current_state]: continue
   
   c. For each arc (i, o, w, target) ∈ δ(current_state):
      d. new_distance = distance ⊗ w
      e. If new_distance ⊕ d[target] ≠ d[target]:
         f. d[target] = new_distance ⊕ d[target]
         g. INSERT(PQ, (target, new_distance))
         h. Add corresponding arc to T'
   
   i. If current_state ∈ F:
      j. Add path to result with weight distance ⊗ F(current_state)

6. Return T'
```

**Random Generation** - Generates random paths with uniform distribution:
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

Some algorithms support lazy evaluation:

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

Algorithms minimize memory allocations:

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