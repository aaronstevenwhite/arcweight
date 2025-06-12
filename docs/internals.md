# ArcWeight Internals

This document describes the internal architecture and implementation details of ArcWeight.

## Core Data Structures

### FST Implementation
```rust
pub struct Fst {
    states: Vec<State>,
    start_state: Option<StateId>,
    final_weights: HashMap<StateId, Weight>,
    properties: FstProperties,
}
```

### State Representation
```rust
pub struct State {
    arcs: Vec<Arc>,
    final_weight: Option<Weight>,
}
```

### Arc Implementation
```rust
pub struct Arc {
    ilabel: Label,
    olabel: Label,
    weight: Weight,
    nextstate: StateId,
}
```

## Memory Management

### Arc Storage
- Compact arc representation
- Arc sorting and compression
- Memory pooling for arcs

### State Management
- State table implementation
- State ID allocation
- State property tracking

### Weight Handling
- Weight type specialization
- Weight caching
- Weight normalization

## Algorithms

### Composition
```rust
impl Fst {
    pub fn compose(&self, other: &Fst) -> Result<Fst> {
        // Implementation details
    }
}
```

### Determinization
```rust
impl Fst {
    pub fn determinize(&self) -> Result<Fst> {
        // Implementation details
    }
}
```

### Minimization
```rust
impl Fst {
    pub fn minimize(&self) -> Result<Fst> {
        // Implementation details
    }
}
```

## Property System

### Property Types
```rust
pub struct FstProperties {
    bits: u64,
}
```

### Property Propagation
- Property inference
- Property verification
- Property-based optimization

## Semiring System

### Semiring Trait Hierarchy
```rust
pub trait Semiring: Clone + Debug + Send + Sync {
    fn zero() -> Self;
    fn one() -> Self;
    fn plus(&self, other: &Self) -> Self;
    fn times(&self, other: &Self) -> Self;
}

pub trait DivisibleSemiring: Semiring {
    fn divide(&self, other: &Self) -> Option<Self>;
}

pub trait NaturallyOrderedSemiring: Semiring + Ord {}

pub trait StarSemiring: Semiring {
    fn star(&self) -> Self;
}
```

### Semiring Constraints on Operations

**Determinization** requires `DivisibleSemiring + Ord`:
- Enables weight normalization through division
- Requires ordering for subset construction

**Weight Pushing** requires `DivisibleSemiring`:
- Needs division for potential computation
- Zero-sum-free property enforced at type level

**Shortest Path** requires `NaturallyOrderedSemiring`:
- Natural ordering enables path comparison
- Supports negative cycle detection

**Epsilon Removal** benefits from `StarSemiring`:
- Star operation provides convergence guarantees
- K-closed property for termination

### Common Semirings
- **Tropical semiring**: `DivisibleSemiring + NaturallyOrderedSemiring` (supports most operations)
- **Log semiring**: `DivisibleSemiring + NaturallyOrderedSemiring` (numerical stability)
- **Probability semiring**: `Semiring` only (limited operation support)

## IO System

### Serialization
```rust
impl Fst {
    pub fn write(&self, writer: &mut impl Write) -> Result<()> {
        // Implementation details
    }
}
```

### Deserialization
```rust
impl Fst {
    pub fn read(reader: &mut impl Read) -> Result<Self> {
        // Implementation details
    }
}
```

## Performance Optimizations

### Arc Compression
- Prefix compression
- Label compression
- Weight compression

### State Optimization
- State merging
- Dead state removal
- Unreachable state removal

### Algorithm Optimizations
- Lazy evaluation
- Parallel processing
- Memory mapping

## Error Handling

### Error Types
```rust
pub enum Error {
    InvalidState(StateId),
    InvalidArc(ArcId),
    InvalidWeight(Weight),
    IOError(io::Error),
    // ...
}
```

### Error Recovery
- State repair
- Arc repair
- Property repair

## Testing Infrastructure

### Property Testing
```rust
#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_fst_properties(states: usize, arcs: usize) {
            // Test implementation
        }
    }
}
```

### Benchmarking
```rust
#[bench]
fn bench_composition(b: &mut Bencher) {
    // Benchmark implementation
}
```

## Debugging Support

### Debug Printing
```rust
impl Debug for Fst {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // Debug implementation
    }
}
```

### Logging
```rust
impl Fst {
    pub fn debug_print(&self) {
        // Debug printing implementation
    }
}
```

## Future Improvements

### Planned Features
- Parallel composition
- GPU acceleration
- Streaming support

### Performance Goals
- Reduced memory usage
- Faster algorithms
- Better scalability

## Implementation Notes

### Design Decisions
- Arc-based memory model
- Property-based optimization
- Generic weight system

### Trade-offs
- Memory vs. speed
- Generality vs. specialization
- Safety vs. performance

## Maintenance

### Code Organization
- Module structure
- Dependency management
- Testing strategy

### Documentation
- Internal documentation
- API documentation
- Example documentation 