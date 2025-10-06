# Extension Points

ArcWeight is designed to be extensible. This document describes the main extension points and how to use them.

## Custom Semirings

**API Reference**: [`Semiring` trait](https://docs.rs/arcweight/latest/arcweight/semiring/trait.Semiring.html)

The most common extension is implementing custom semirings for specific problem domains:

```rust,ignore
#[derive(Debug, Clone, PartialEq, PartialOrd)]
struct CustomWeight {
    value: f32,
}

impl Semiring for CustomWeight {
    type Value = f32;
    
    fn new(value: Self::Value) -> Self {
        Self { value }
    }
    
    fn value(&self) -> &Self::Value {
        &self.value
    }
    
    fn plus(&self, other: &Self) -> Self {
        // Define custom addition semantics
        Self { value: self.value.min(other.value) }
    }
    
    fn times(&self, other: &Self) -> Self {
        // Define custom multiplication semantics
        Self { value: self.value + other.value }
    }
    
    // ... implement remaining required methods
}
```

### Specialized Semiring Properties

For advanced semirings, implement additional traits:

```rust,ignore
// For semirings that support division
impl DivisibleSemiring for CustomWeight {
    fn divide(&self, other: &Self) -> Option<Self> {
        if other.value != 0.0 {
            Some(Self { value: self.value - other.value })
        } else {
            None
        }
    }
}

// For semirings with natural ordering
impl Ord for CustomWeight {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.partial_cmp(&other.value).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl NaturallyOrderedSemiring for CustomWeight {}

// For semirings with Kleene closure
impl StarSemiring for CustomWeight {
    fn star(&self) -> Self {
        // Define Kleene closure semantics
        if self.value > 1.0 {
            Self::zero()  // Divergent series
        } else {
            Self { value: 1.0 / (1.0 - self.value) }
        }
    }
}
```

## Custom FST Types

Create custom FST implementations for specialized storage strategies:

```rust,ignore
struct MyCustomFst<W: Semiring> {
    // Custom storage strategy
    compressed_data: Vec<u8>,
    state_offsets: Vec<usize>,
    _phantom: PhantomData<W>,
}

impl<W: Semiring> Fst<W> for MyCustomFst<W> {
    type ArcIter<'a> = MyArcIterator<'a, W> where Self: 'a;
    
    fn start(&self) -> Option<StateId> {
        // Custom implementation
        Some(0)
    }
    
    fn final_weight(&self, state: StateId) -> Option<&W> {
        // Custom implementation
        self.decode_final_weight(state)
    }
    
    fn arcs(&self, state: StateId) -> Self::ArcIter<'_> {
        // Custom implementation
        MyArcIterator::new(self, state)
    }
    
    // ... implement remaining required methods
}
```

### Custom Arc Iterators

```rust,ignore
struct MyArcIterator<'a, W: Semiring> {
    fst: &'a MyCustomFst<W>,
    state: StateId,
    position: usize,
}

impl<'a, W: Semiring> ArcIterator<W> for MyArcIterator<'a, W> {
    fn next(&mut self) -> Option<Arc<W>> {
        // Custom arc iteration logic
        self.fst.decode_arc_at(self.state, self.position)
    }
    
    fn done(&self) -> bool {
        self.position >= self.fst.num_arcs(self.state)
    }
    
    fn reset(&mut self) {
        self.position = 0;
    }
}
```

## Algorithm Extensions

### Composition Filters

The composition algorithm supports custom filters through the compose operation. Several built-in filters are provided:
- `SequenceFilter` - Standard composition filter
- `EpsilonFilter` - Handles epsilon transitions
- `NoEpsilonFilter` - Optimized for epsilon-free FSTs

Custom filters can be implemented to modify composition behavior.

## Custom Compactors

Implement compression strategies for `CompactFst`:

```rust,ignore
pub trait Compactor<W: Semiring>: Debug + Send + Sync + 'static {
    type Element: Clone + Debug + Send + Sync;
    
    fn compact(arc: &Arc<W>) -> Self::Element;
    fn expand(element: &Self::Element) -> Arc<W>;
    fn compact_weight(weight: &W) -> Self::Element;
    fn expand_weight(element: &Self::Element) -> W;
}

// Custom compactor for specific arc patterns
struct PatternCompactor;

impl<W: Semiring> Compactor<W> for PatternCompactor 
where
    W: Clone + Default,
{
    type Element = u32;  // Compact representation
    
    fn compact(arc: &Arc<W>) -> Self::Element {
        // Encode arc as u32 based on patterns
        (arc.ilabel << 16) | arc.olabel
    }
    
    fn expand(element: &Self::Element) -> Arc<W> {
        Arc::new(
            element >> 16,           // ilabel
            element & 0xFFFF,        // olabel
            W::default(),           // default weight
            0,                      // default next state
        )
    }
    
    fn compact_weight(weight: &W) -> Self::Element {
        // Custom weight compression
        0  // Placeholder
    }
    
    fn expand_weight(element: &Self::Element) -> W {
        W::default()
    }
}
```

## I/O Format Extensions

ArcWeight provides several built-in I/O formats in the `io` module:
- Text format (AT&T FSM format)
- Binary format (with serde feature)
- OpenFST compatibility format

Custom formats can be implemented by creating new read/write functions following the patterns in the `io` module.

## Property Extensions

The existing property system in `properties/mod.rs` uses bitflags to track FST characteristics. While custom properties are not directly extensible, the existing system provides comprehensive tracking of FST properties for optimization.

## Symbol Table Extensions

Extend symbol tables with custom functionality:

```rust,ignore
pub trait SymbolMapper {
    fn map_symbol(&self, symbol: &str) -> Option<Label>;
    fn reverse_map(&self, label: Label) -> Option<&str>;
}

struct PhonemeMapper {
    phoneme_to_id: HashMap<String, Label>,
    id_to_phoneme: Vec<String>,
}

impl SymbolMapper for PhonemeMapper {
    fn map_symbol(&self, symbol: &str) -> Option<Label> {
        self.phoneme_to_id.get(symbol).copied()
    }
    
    fn reverse_map(&self, label: Label) -> Option<&str> {
        self.id_to_phoneme.get(label as usize).map(|s| s.as_str())
    }
}
```

## Extension Guidelines

### 1. Follow Trait Conventions

- Implement all required trait methods
- Use consistent naming patterns
- Provide comprehensive documentation
- Include usage examples

### 2. Maintain Mathematical Correctness

- Ensure semiring axioms are satisfied
- Verify algorithm correctness with custom types
- Test edge cases thoroughly
- Document mathematical properties

### 3. Optimize for Performance

- Consider cache locality in data structures
- Use appropriate allocation strategies
- Profile custom implementations
- Benchmark against existing implementations

### 4. Provide Clear Documentation

```rust,ignore
/// Custom semiring for geographic distances
/// 
/// This semiring uses the tropical structure (min, +) but with
/// geographic distance calculations for realistic path costs.
/// 
/// # Mathematical Properties
/// 
/// - Addition: min(a, b) - finds shortest distance
/// - Multiplication: a + b - accumulates distance along path
/// - Zero: ∞ - impossible path
/// - One: 0.0 - no distance cost
/// 
/// # Example
/// 
/// ```rust
/// use arcweight::prelude::*;
/// 
/// let weight1 = GeographicWeight::new(5.2);
/// let weight2 = GeographicWeight::new(3.8);
/// let combined = weight1.plus(&weight2);  // min(5.2, 3.8) = 3.8
/// ```
pub struct GeographicWeight {
    distance_km: f64,
}
```

This extension system allows ArcWeight to adapt to various finite state transducer applications while maintaining performance and correctness guarantees.