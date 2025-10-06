# Design Patterns and Best Practices

## Core Design Patterns in ArcWeight

### Trait-Based Architecture

The foundation of ArcWeight is its trait system, which provides flexibility and type safety:

```rust,ignore
// Core FST trait hierarchy
pub trait Fst<W: Semiring>: Debug + Send + Sync {
    // Read-only operations
}

pub trait MutableFst<W: Semiring>: Fst<W> {
    // Mutable operations
}

pub trait ExpandedFst<W: Semiring>: Fst<W> {
    // Direct arc access
}
```

This hierarchy allows algorithms to work with the minimal interface they require.

### Generic Programming

Algorithms are generic over FST and semiring types:

```rust,ignore
pub fn compose<W, F1, F2, M>(
    fst1: &F1,
    fst2: &F2,
    filter: impl ComposeFilter<W>,
) -> Result<M>
where
    W: Semiring,
    F1: Fst<W>,
    F2: Fst<W>,
    M: MutableFst<W> + Default,
```

This enables code reuse across different FST implementations and weight types.

### Iterator Pattern

ArcWeight uses iterators for efficient arc traversal:

```rust,ignore
// Arc iteration without allocating collections
for arc in fst.arcs(state) {
    // Process arc
}

// Functional style with iterator chains
let total_arcs: usize = fst.states()
    .map(|s| fst.num_arcs(s))
    .sum();
```

### Property-Based Optimization

The property system enables optimizations based on FST characteristics:

```rust,ignore
let props = fst.properties();
if props.contains(PropertyFlags::ACYCLIC) {
    // Use optimized algorithm for acyclic FSTs
}
```

## Algorithm Design Patterns

### State Queue Pattern

Many algorithms process states using a queue:

```rust,ignore
let mut queue = VecDeque::new();
let mut visited = HashSet::new();

if let Some(start) = fst.start() {
    queue.push_back(start);
    visited.insert(start);
}

while let Some(state) = queue.pop_front() {
    // Process state
    for arc in fst.arcs(state) {
        if visited.insert(arc.nextstate) {
            queue.push_back(arc.nextstate);
        }
    }
}
```

### Result FST Construction

Algorithms typically build result FSTs incrementally:

```rust,ignore
let mut result = M::default();
let mut state_map = HashMap::new();

// Map states from input to output
let new_state = result.add_state();
state_map.insert(old_state, new_state);
```

### Configuration Structs

Complex algorithms use configuration structs for flexibility:

```rust,ignore
#[derive(Debug, Clone, Default)]
pub struct ShortestPathConfig {
    pub nshortest: usize,
    pub unique: bool,
    pub weight_threshold: Option<f64>,
    pub state_threshold: Option<usize>,
}
```

## Memory Management Patterns

### Reference Semantics

FSTs are passed by reference to avoid unnecessary copying:

```rust,ignore
pub fn minimize<F, W, M>(fst: &F) -> Result<M>
where
    F: Fst<W>,
    W: DivisibleSemiring,
    M: MutableFst<W> + Default,
```

### Lazy Evaluation

The `LazyFstImpl` pattern enables on-demand computation:

```rust,ignore
// States and arcs computed only when accessed
let lazy_fst = LazyFstImpl::new(state_fn);
```

## Best Practices

### 1. Use Appropriate Trait Bounds

Only require the traits you actually need:

```rust,ignore
// Good: Minimal requirements
fn count_states<W: Semiring>(fst: &impl Fst<W>) -> usize {
    fst.num_states()
}

// Avoid: Over-constraining
fn count_states<W: Semiring>(fst: &impl MutableFst<W>) -> usize {
    fst.num_states()  // Doesn't need mutability
}
```

### 2. Leverage Type Safety

Use the type system to enforce correctness:

```rust,ignore
// Weight type ensures valid operations
pub fn shortest_path<F, W, M>(fst: &F, config: ShortestPathConfig) -> Result<M>
where
    F: Fst<W>,
    W: NaturallyOrderedSemiring,  // Type system enforces ordering
    M: MutableFst<W> + Default,
```

### 3. Follow Rust Idioms

- Use `Result<T>` for fallible operations
- Implement standard traits (`Debug`, `Clone`, etc.)
- Use iterators instead of explicit loops when appropriate
- Prefer composition over inheritance

### 4. Document Mathematical Properties

When implementing semirings or algorithms, document the mathematical properties:

```rust,ignore
/// Tropical semiring (min, +, ∞, 0)
/// 
/// # Mathematical Properties
/// - Addition: min(a, b)
/// - Multiplication: a + b
/// - Zero: ∞ (positive infinity)
/// - One: 0.0
```

### 5. Error Handling

Use descriptive error types and provide context:

```rust,ignore
pub enum Error {
    #[error("Invalid FST operation: {0}")]
    InvalidOperation(String),
    
    #[error("Algorithm error: {0}")]
    Algorithm(String),
}
```

## Composition Patterns

The composition algorithm demonstrates several patterns:

### Filter Strategy Pattern

Different composition behaviors through filter implementations:

```rust,ignore
// Built-in filters
pub struct SequenceFilter;
pub struct EpsilonFilter;
pub struct NoEpsilonFilter;

// Each implements ComposeFilter trait with different logic
```

### State Encoding

Composed states are encoded/decoded for efficient storage:

```rust,ignore
// Encode two state IDs into one
fn encode_states(s1: StateId, s2: StateId) -> StateId;

// Decode back to original states
fn decode_states(encoded: StateId) -> (StateId, StateId);
```

These patterns ensure ArcWeight code is efficient, maintainable, and follows Rust best practices while maintaining mathematical correctness.