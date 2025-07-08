# Trait System

## FST Trait System

**API Reference**: [`fst::traits`](https://docs.rs/arcweight/latest/arcweight/fst/traits/)

The core trait hierarchy provides different levels of FST functionality:

```rust,ignore
/// Base trait - read-only FST operations
pub trait Fst<W: Semiring>: Debug + Send + Sync {
    type ArcIter<'a>: ArcIterator<W> where Self: 'a;
    
    fn start(&self) -> Option<StateId>;
    fn final_weight(&self, state: StateId) -> Option<&W>;
    fn num_arcs(&self, state: StateId) -> usize;
    fn num_states(&self) -> usize;
    fn properties(&self) -> FstProperties;
    fn arcs(&self, state: StateId) -> Self::ArcIter<'_>;
    fn states(&self) -> impl Iterator<Item = StateId>;
    fn num_arcs_total(&self) -> usize;
    fn is_empty(&self) -> bool;
}

/// Mutable FST operations
pub trait MutableFst<W: Semiring>: Fst<W> {
    fn add_state(&mut self) -> StateId;
    fn add_arc(&mut self, state: StateId, arc: Arc<W>);
    fn set_start(&mut self, state: StateId);
    fn set_final(&mut self, state: StateId, weight: W);
    fn delete_arcs(&mut self, state: StateId);
    fn delete_arc(&mut self, state: StateId, arc_idx: usize);
    fn reserve_states(&mut self, n: usize);
    fn reserve_arcs(&mut self, state: StateId, n: usize);
    fn clear(&mut self);
}

/// Memory-resident FSTs with direct access
pub trait ExpandedFst<W: Semiring>: Fst<W> {
    fn arcs_slice(&self, state: StateId) -> &[Arc<W>];
}

/// FSTs computed on-demand
pub trait LazyFst<W: Semiring>: Fst<W> {
    fn expand(&self, state: StateId) -> Result<()>;
}
```

**Design Benefits:**
- **Modularity**: Algorithms work with any FST implementation
- **Performance**: Different implementations optimize for specific use cases
- **Safety**: Type system prevents invalid operations

## Semiring Trait Hierarchy

**API Reference**: [`semiring::traits`](https://docs.rs/arcweight/latest/arcweight/semiring/traits/)

The semiring system provides the algebraic foundation:

```rust,ignore
/// Core semiring operations
pub trait Semiring:
    Clone + Debug + Display + PartialEq + PartialOrd + 
    Add<Output = Self> + Mul<Output = Self> + Zero + One + 
    Send + Sync + 'static
{
    type Value: Clone + Debug + PartialEq + PartialOrd;
    
    fn new(value: Self::Value) -> Self;
    fn value(&self) -> &Self::Value;
    fn plus(&self, other: &Self) -> Self;      // \\(\oplus\\) operation
    fn times(&self, other: &Self) -> Self;     // \\(\otimes\\) operation
    fn properties() -> SemiringProperties;
    fn approx_eq(&self, other: &Self, epsilon: f64) -> bool;
}

/// Specialized semiring properties
pub trait DivisibleSemiring: Semiring {
    fn divide(&self, other: &Self) -> Option<Self>;
}

pub trait NaturallyOrderedSemiring: Semiring + Ord {}

pub trait StarSemiring: Semiring {
    fn star(&self) -> Self;  // Kleene closure
}

pub trait InvertibleSemiring: Semiring {
    fn inverse(&self) -> Option<Self>;
}
```

## Implementation Strategy

- Each semiring type implements the `Semiring` trait with specialized behavior
- Compile-time specialization generates optimized code for each semiring
- Mathematical properties are encoded in specialized trait bounds
- Generic algorithms work with any conforming semiring implementation

## Trait Design Patterns

### Strategy Pattern for Algorithms

Many algorithms use the strategy pattern to allow customization:

```rust,ignore
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

### Factory Pattern for FST Types

```rust,ignore
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

### Iterator Pattern for Memory Efficiency

ArcWeight extensively uses iterators to avoid unnecessary allocations:

```rust,ignore
// Lazy evaluation - no intermediate collections
let total_weight: f32 = fst
    .states()
    .flat_map(|state| fst.arcs(state))
    .map(|arc| arc.weight.value())
    .sum();
```

### Adapter Pattern for Legacy Compatibility

```rust,ignore
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

This trait system design provides the flexibility and type safety that makes ArcWeight both powerful and safe to use.