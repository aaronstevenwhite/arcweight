# Design Patterns and Best Practices

## Builder Pattern for Complex FSTs

ArcWeight encourages the builder pattern for constructing complex FSTs:

```rust
pub struct FstBuilder<W: Semiring> {
    fst: VectorFst<W>,
    symbol_table: Option<SymbolTable>,
}

impl<W: Semiring> FstBuilder<W> {
    pub fn new() -> Self {
        Self {
            fst: VectorFst::new(),
            symbol_table: None,
        }
    }
    
    pub fn with_symbol_table(mut self, table: SymbolTable) -> Self {
        self.symbol_table = Some(table);
        self
    }
    
    pub fn add_word(&mut self, word: &str, weight: W) -> Result<()> {
        // Complex word addition logic
    }
    
    pub fn build(self) -> VectorFst<W> {
        self.fst
    }
}
```

## Strategy Pattern for Algorithms

Many algorithms use the strategy pattern to allow customization:

```rust
// Different composition strategies
pub trait ComposeFilter<W: Semiring> {
    fn filter(&mut self, 
              state1: StateId, state2: StateId, 
              arc1: &Arc<W>, arc2: &Arc<W>) -> bool;
}

pub struct NoEpsilonFilter;
pub struct EpsilonFilter;
pub struct SequenceFilter;

// Algorithm accepts any strategy
// API Reference: https://docs.rs/arcweight/latest/arcweight/algorithms/fn.compose.html
pub fn compose<W, F1, F2, Filter>(
    fst1: &F1, 
    fst2: &F2, 
    filter: Filter
) -> Result<VectorFst<W>>
where
    Filter: ComposeFilter<W>,
```

## Factory Pattern for FST Types

```rust
pub trait FstFactory<W: Semiring> {
    type Fst: MutableFst<W>;
    
    fn create() -> Self::Fst;
    fn create_with_capacity(states: usize, arcs: usize) -> Self::Fst;
}

pub struct VectorFstFactory;
impl<W: Semiring> FstFactory<W> for VectorFstFactory {
    type Fst = VectorFst<W>;
    
    fn create() -> Self::Fst {
        VectorFst::new()
    }
}
```

## Iterator Pattern for Memory Efficiency

ArcWeight extensively uses iterators to avoid unnecessary allocations:

```rust
// Lazy evaluation - no intermediate collections
let total_weight: f32 = fst
    .states()
    .flat_map(|state| fst.arcs(state))
    .map(|arc| arc.weight.value())
    .sum();
```

## Adapter Pattern for Legacy Compatibility

```rust
pub struct OpenFstAdapter<F> {
    inner: F,
}

impl<F, W> Fst<W> for OpenFstAdapter<F> 
where 
    F: OpenFstTrait<W>,
    W: Semiring,
{
    // Adapt OpenFST interface to ArcWeight interface
}
```

## Composition Architecture

**API Reference**: [`algorithms::compose`](https://docs.rs/arcweight/latest/arcweight/algorithms/compose/)

The composition algorithm demonstrates sophisticated design:

```rust
pub struct CompositionState {
    state1: StateId,
    state2: StateId,
    filter_state: FilterState,
}

pub struct CompositionCache {
    state_map: HashMap<CompositionState, StateId>,
    queue: VecDeque<CompositionState>,
}

// Lazy composition with on-demand state creation
pub fn compose_lazy<W, F1, F2>(
    fst1: &F1, 
    fst2: &F2
) -> LazyFstImpl<W> {
    LazyFstImpl::new(move |state| {
        let (s1, s2) = decode_composed_state(state);
        compute_arcs_for_composed_state(s1, s2, fst1, fst2)
    })
}
```

## Error Recovery and Robustness

```rust
pub struct RobustFstBuilder<W: Semiring> {
    fst: VectorFst<W>,
    validation_mode: ValidationMode,
    error_recovery: ErrorRecoveryStrategy,
}

pub enum ValidationMode {
    Strict,        // Fail on any invalid operation
    Permissive,    // Try to recover from errors
    Silent,        // Ignore minor issues
}

impl<W: Semiring> RobustFstBuilder<W> {
    pub fn add_arc_safe(&mut self, 
                       state: StateId, 
                       arc: Arc<W>) -> Result<()> {
        match self.validation_mode {
            ValidationMode::Strict => {
                self.validate_arc(&arc)?;
                self.fst.add_arc(state, arc);
            }
            ValidationMode::Permissive => {
                if let Err(e) = self.validate_arc(&arc) {
                    self.error_recovery.handle_error(e)?;
                    // Try to add a corrected version
                    let corrected_arc = self.correct_arc(arc)?;
                    self.fst.add_arc(state, corrected_arc);
                } else {
                    self.fst.add_arc(state, arc);
                }
            }
            ValidationMode::Silent => {
                // Best effort - ignore errors
                self.fst.add_arc(state, arc);
            }
        }
        Ok(())
    }
}
```

## Best Practices

### 1. Use Type Safety

Always prefer compile-time guarantees over runtime checks:

```rust
// Good: Type system enforces constraints
fn shortest_path<F, W>(fst: &F) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: NaturallyOrderedSemiring,  // Compiler enforces ordering requirement
{
    // Implementation can assume ordering exists
}

// Avoid: Runtime checking of properties
fn shortest_path_unchecked<F, W>(fst: &F) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: Semiring,
{
    if !W::properties().is_naturally_ordered() {
        return Err(Error::InvalidOperation("Semiring must be naturally ordered".to_string()));
    }
    // Implementation
}
```

### 2. Leverage Properties

Use FST properties for optimization:

```rust
pub fn optimize_fst<W, F>(fst: &F) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: Semiring,
{
    let props = fst.properties();
    
    if props.has_property(PropertyFlags::NO_EPSILONS) {
        // Skip epsilon removal
        minimize(fst)
    } else {
        minimize(&remove_epsilons(fst)?)
    }
}
```

### 3. Prefer Immutable Operations

Design algorithms to avoid mutation when possible:

```rust
// Good: Returns new FST
pub fn reverse<F, W>(fst: &F) -> Result<VectorFst<W>>
where
    F: Fst<W>,
    W: Semiring,
{
    let mut result = VectorFst::new();
    // Build reversed FST
    Ok(result)
}

// Avoid: Mutates input
pub fn reverse_in_place<F>(fst: &mut F) -> Result<()>
where
    F: MutableFst<W>,
{
    // Modifies fst directly
}
```

### 4. Use Iterator Chains

Leverage Rust's iterator system for efficient processing:

```rust
// Efficient: No intermediate allocations
let final_states: Vec<StateId> = fst
    .states()
    .filter(|&state| fst.final_weight(state).is_some())
    .collect();

// Less efficient: Multiple passes
let mut final_states = Vec::new();
for state in fst.states() {
    if fst.final_weight(state).is_some() {
        final_states.push(state);
    }
}
```

These patterns ensure ArcWeight code is maintainable, efficient, and follows Rust best practices.