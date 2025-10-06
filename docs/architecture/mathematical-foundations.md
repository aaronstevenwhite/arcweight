# Mathematical Foundations

This document describes how theoretical concepts from automata theory and semiring algebra map to ArcWeight's implementation. For comprehensive mathematical theory, see [Core Concepts](../core-concepts/).

## Formal Definition and Implementation

### Finite State Transducer

A weighted finite state transducer (FST) is formally defined as a 7-tuple:

**T = (Q, Σ, Δ, δ, λ, q₀, F)**

where:
- Q is a finite set of states
- Σ is the input alphabet
- Δ is the output alphabet
- δ: Q × (Σ ∪ {ε}) → 2^Q is the state transition function
- λ: Q × (Σ ∪ {ε}) × Q → K is the output function mapping to semiring K
- q₀ ∈ Q is the initial state
- F: Q → K assigns final weights to states

### Implementation Correspondence

| Mathematical Component | Implementation | Type Specification |
|----------------------|----------------|-------------------|
| **Q** (state set) | State indices | `StateId = u32` |
| **Σ** (input alphabet) | Input labels | `Label = u32` |
| **Δ** (output alphabet) | Output labels | `Label = u32` |
| **δ** (transition function) | Arc collection | `Arc<W>` where `W: Semiring` |
| **λ** (output function) | Arc weights | `W` in `Arc<W>` |
| **q₀** (initial state) | Start state | `Option<StateId>` |
| **F** (final function) | Final weights | `Option<W>` per state |

### Semiring Algebraic Structure

A semiring (K, ⊕, ⊗, 0̄, 1̄) satisfies the following axioms:

1. **(K, ⊕)** forms a commutative monoid with identity 0̄
2. **(K, ⊗)** forms a monoid with identity 1̄
3. **Distributivity**: a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c) and (a ⊕ b) ⊗ c = (a ⊗ c) ⊕ (b ⊗ c)
4. **Annihilation**: 0̄ ⊗ a = a ⊗ 0̄ = 0̄

#### Example: Tropical Semiring

The tropical semiring (ℝ ∪ {∞}, min, +, ∞, 0) is fundamental for shortest-path problems:

```rust,ignore
// Mathematical: a ⊕ b = min(a, b)
let path1 = TropicalWeight::new(5.0);
let path2 = TropicalWeight::new(3.0);
let shortest = path1.plus(&path2);  // Returns 3.0

// Mathematical: a ⊗ b = a + b
let segment1 = TropicalWeight::new(2.0);
let segment2 = TropicalWeight::new(4.0);
let total = segment1.times(&segment2);  // Returns 6.0
```

## Type System as Mathematical Framework

### Algebraic Type Constraints

Rust's trait system encodes mathematical requirements at the type level, preventing invalid operations at compile time:

```rust,ignore
// Semiring trait encodes algebraic axioms
pub trait Semiring: Clone + Debug + PartialEq + PartialOrd {
    type Value: Clone + Debug + PartialEq + PartialOrd;
    
    // Semiring operations with axiomatic properties
    fn plus(&self, other: &Self) -> Self;      // ⊕: associative, commutative
    fn times(&self, other: &Self) -> Self;     // ⊗: associative
    fn zero() -> Self;                          // 0̄: additive identity
    fn one() -> Self;                           // 1̄: multiplicative identity
}
```

### Algorithm Correctness Through Types

Mathematical preconditions are enforced via trait bounds:

```rust,ignore
// Dijkstra's algorithm requires monotonic semiring
pub fn shortest_path<F, W, M>(fst: &F, config: ShortestPathConfig) -> Result<M>
where
    F: Fst<W>,
    W: NaturallyOrderedSemiring,  // Enforces: a ≤ a ⊕ b
    M: MutableFst<W> + Default,
{
    // Implementation guaranteed correct by type constraints
}

// Kleene closure requires star operation
pub fn closure<F, W, M>(fst: &F) -> Result<M>
where
    F: Fst<W>,
    W: StarSemiring,  // Enforces: a* = 1̄ ⊕ a ⊕ a² ⊕ ...
    M: MutableFst<W> + Default,
```

## Formal Properties and Optimization

### FST Property System

The property system tracks formal characteristics enabling algorithmic optimizations:

```rust,ignore
bitflags! {
    pub struct PropertyFlags: u64 {
        // Structural Properties
        const ACCEPTOR = 1 << 0;           // ∀ arc: ilabel = olabel
        const NO_EPSILONS = 1 << 1;        // ∀ arc: ilabel ≠ ε ∧ olabel ≠ ε
        const NO_INPUT_EPSILONS = 1 << 2;  // ∀ arc: ilabel ≠ ε
        const NO_OUTPUT_EPSILONS = 1 << 3; // ∀ arc: olabel ≠ ε
        
        // Determinism Properties
        const INPUT_DETERMINISTIC = 1 << 7;  // ∀ s,l: |{(s,l,_,_)}| ≤ 1
        const OUTPUT_DETERMINISTIC = 1 << 8; // ∀ s,l: |{(s,_,l,_)}| ≤ 1
        const FUNCTIONAL = 1 << 9;           // Input string → unique output
        
        // Graph Properties
        const ACYCLIC = 1 << 15;             // No directed cycles
        const INITIAL_ACYCLIC = 1 << 16;    // No cycles through q₀
        const TOP_SORTED = 1 << 17;         // States in topological order
        const CONNECTED = 1 << 20;          // All states reachable & coreachable
    }
}
```

### Property-Based Algorithm Selection

```rust,ignore
// Example: Shortest distance computation
fn shortest_distance<W: Semiring>(fst: &impl Fst<W>) -> Vec<W> {
    let props = fst.properties();
    
    if props.contains(PropertyFlags::ACYCLIC) {
        // O(|Q| + |E|) topological algorithm
        acyclic_shortest_distance(fst)
    } else {
        // O(|Q||E|) Bellman-Ford algorithm
        bellman_ford_shortest_distance(fst)
    }
}
```

## Correctness Guarantees

### Algebraic Correctness

Semiring implementations must satisfy algebraic laws, verified through property-based testing:

```rust,ignore
// Associativity: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
quickcheck! {
    fn plus_associative(a: TropicalWeight, b: TropicalWeight, c: TropicalWeight) -> bool {
        (a.plus(&b)).plus(&c) == a.plus(&(b.plus(&c)))
    }
}

// Distributivity: a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)
quickcheck! {
    fn left_distributive(a: TropicalWeight, b: TropicalWeight, c: TropicalWeight) -> bool {
        a.times(&(b.plus(&c))) == (a.times(&b)).plus(&(a.times(&c)))
    }
}
```

### Algorithm Invariants

Key algorithms maintain formal invariants throughout execution:

1. **Dijkstra's Algorithm** (shortest path):
   - Invariant: Distance estimates monotonically decrease
   - Precondition: Semiring satisfies monotonicity (a ≤ a ⊕ b)

2. **Determinization**:
   - Invariant: Subset construction preserves language
   - Postcondition: ∀ s,l: |δ(s,l)| ≤ 1

3. **Minimization** (Hopcroft's algorithm):
   - Invariant: Partition refinement preserves equivalence
   - Postcondition: Minimal automaton is unique up to isomorphism

### Complexity Bounds

Theoretical complexity is preserved in implementation:

| Algorithm | Theoretical Bound | Implementation Guarantee |
|-----------|------------------|------------------------|
| Composition | O(\|Q₁\|\|Q₂\|\|Σ\|) | Lazy evaluation can improve average case |
| Determinization | O(2^\|Q\|) | Subset construction with pruning |
| Minimization | O(\|E\| log \|Q\|) | Hopcroft's partition refinement |
| ε-Removal | O(\|Q\|³) | Kleene closure on ε-transitions |

## References and Further Reading

### Internal Documentation
- [Semirings](../core-concepts/semirings.md) - Comprehensive semiring theory and implementations
- [Trait System](trait-system.md) - Type-level encoding of mathematical abstractions
- [Algorithm Architecture](algorithm-architecture.md) - Detailed algorithm implementations

### Foundational Literature
- Mohri, M. (2009). "Weighted Automata Algorithms." In *Handbook of Weighted Automata*.
- Hopcroft, J., & Ullman, J. (1979). *Introduction to Automata Theory, Languages, and Computation*.
- Kuich, W., & Salomaa, A. (1986). *Semirings, Automata, Languages*. Springer-Verlag.