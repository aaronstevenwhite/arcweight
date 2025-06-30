# Extension Points

ArcWeight is designed to be extensible. This document describes the main extension points and how to use them.

## Custom Semirings

**API Reference**: [`Semiring` trait](https://docs.rs/arcweight/latest/arcweight/semiring/trait.Semiring.html)

The most common extension is implementing custom semirings for specific problem domains:

```rust
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

```rust
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

```rust
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

```rust
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

## Algorithm Plugins

### Custom Composition Filters

Extend composition behavior with custom filters:

```rust
pub trait ComposeFilter<W: Semiring> {
    fn filter(&mut self, state1: StateId, state2: StateId, arc1: &Arc<W>, arc2: &Arc<W>) -> bool;
}

// Custom composition behavior
struct MyComposeFilter {
    max_weight: f32,
}

impl<W: Semiring> ComposeFilter<W> for MyComposeFilter 
where
    W::Value: Into<f32>,
{
    fn filter(&mut self, _state1: StateId, _state2: StateId, arc1: &Arc<W>, arc2: &Arc<W>) -> bool {
        // Custom filtering logic
        let weight1: f32 = arc1.weight.value().into();
        let weight2: f32 = arc2.weight.value().into();
        weight1 + weight2 <= self.max_weight
    }
}
```

### Custom Distance Functions

Implement custom distance metrics for shortest path algorithms:

```rust
pub trait DistanceFunction<W: Semiring> {
    fn distance(&self, from: StateId, to: StateId, weight: &W) -> f64;
}

struct GeographicDistance {
    coordinates: HashMap<StateId, (f64, f64)>,
}

impl<W: Semiring> DistanceFunction<W> for GeographicDistance {
    fn distance(&self, from: StateId, to: StateId, weight: &W) -> f64 {
        if let (Some(&(x1, y1)), Some(&(x2, y2))) = 
            (self.coordinates.get(&from), self.coordinates.get(&to)) {
            ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
        } else {
            f64::INFINITY
        }
    }
}
```

## Custom Compactors

Implement compression strategies for `CompactFst`:

```rust
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

Add support for custom serialization formats:

```rust
pub trait FstFormat<W: Semiring> {
    fn read<R: Read, F: MutableFst<W>>(reader: R) -> Result<F>;
    fn write<W: Write, F: Fst<W>>(fst: &F, writer: W) -> Result<()>;
}

// Custom binary format
struct MyBinaryFormat;

impl<W: Semiring + Serialize + for<'de> Deserialize<'de>> FstFormat<W> for MyBinaryFormat {
    fn read<R: Read, F: MutableFst<W>>(mut reader: R) -> Result<F> {
        // Custom deserialization logic
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)?;
        
        let mut fst = F::default();
        // Decode from buffer into fst
        
        Ok(fst)
    }
    
    fn write<W: Write, F: Fst<W>>(fst: &F, mut writer: W) -> Result<()> {
        // Custom serialization logic
        let serialized = serialize_custom(fst)?;
        writer.write_all(&serialized)?;
        Ok(())
    }
}
```

## Property Extensions

Add custom FST properties for optimization:

```rust
bitflags! {
    pub struct CustomProperties: u64 {
        const PHONOLOGICALLY_ORDERED = 1 << 32;
        const MORPHOLOGICALLY_SORTED = 1 << 33;
        const GEOGRAPHICALLY_INDEXED = 1 << 34;
    }
}

pub trait PropertyComputer<W: Semiring> {
    fn compute_properties<F: Fst<W>>(fst: &F) -> CustomProperties;
}

struct LinguisticPropertyComputer;

impl<W: Semiring> PropertyComputer<W> for LinguisticPropertyComputer {
    fn compute_properties<F: Fst<W>>(fst: &F) -> CustomProperties {
        let mut props = CustomProperties::empty();
        
        // Check for phonological ordering
        if is_phonologically_ordered(fst) {
            props |= CustomProperties::PHONOLOGICALLY_ORDERED;
        }
        
        // Check for morphological sorting
        if is_morphologically_sorted(fst) {
            props |= CustomProperties::MORPHOLOGICALLY_SORTED;
        }
        
        props
    }
}
```

## Symbol Table Extensions

Extend symbol tables with custom functionality:

```rust
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

## Future Extension Points

### Machine Learning Integration

```rust
// Future: Differentiable FST operations
pub trait DifferentiableFst<W: DifferentiableSemiring> {
    fn forward(&self, input: &[Label]) -> (W, GradientTape);
    fn backward(&self, gradient: &GradientTape) -> ParameterGradients;
}
```

### Quantum Computing Preparation

```rust
// Future: Quantum FST simulation
pub trait QuantumSemiring: Semiring {
    type AmplitudeType: ComplexNumber;
    
    fn superposition(&self, other: &Self, amplitude: Self::AmplitudeType) -> Self;
    fn measure(&self) -> (Self, f64);  // Returns collapsed state and probability
}
```

### Advanced Compression

```rust
// Future: Neural compression for FSTs
pub struct NeuralCompactor<W: Semiring> {
    encoder: NeuralNetwork,
    decoder: NeuralNetwork,
    _phantom: PhantomData<W>,
}

impl<W: Semiring> Compactor<W> for NeuralCompactor<W> {
    // Learn optimal compression from data
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

```rust
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

This extension system allows ArcWeight to adapt to virtually any finite state transducer application while maintaining performance and correctness guarantees.