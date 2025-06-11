# Edit Distance Example

This example demonstrates how to use weighted finite state transducers (WFSTs) to compute edit distance between strings with customizable operation costs. It's particularly useful for spell checking, fuzzy string matching, and demonstrating how different weight schemes affect computed distances.

## Overview

The example shows how to:
1. Build an edit distance transducer that accepts strings within edit distance k of a target
2. Compute standard Levenshtein distance with uniform costs
3. Use custom weights for insertions, deletions, and substitutions
4. Extract the shortest paths to find optimal alignments
5. Demonstrate how weight schemes affect the computed distance

## Edit Operations with Custom Costs

The edit distance FST supports three fundamental operations:
- **Insertion**: Add a character (configurable cost)
- **Deletion**: Remove a character (configurable cost)
- **Substitution**: Replace one character with another (configurable cost)
- **Match**: Characters are identical (zero cost)

## Implementation Details

### Edit Distance Lattice Construction

The `build_edit_distance_fst` function creates a lattice FST where:
- States represent positions `(i,d)` where `i` is the character position and `d` is edit count
- The lattice allows for all possible edit sequences up to distance `k`
- Each state represents having matched `i` characters with `d` edits

```rust
fn build_edit_distance_fst(
    target: &str,
    k: usize,
    insertion_cost: f32,
    deletion_cost: f32,
    substitution_cost: f32,
) -> VectorFst<TropicalWeight>
```

### State Organization

States are organized in a 2D grid:
- Row `i`: Position in target string (0 to n)
- Column `d`: Number of edits used (0 to k)
- State `(i,d)` accessible only if `d ≤ k`

### Transition Types

1. **Match transitions**: `(i,d) → (i+1,d)` when input character matches target[i]
2. **Substitution**: `(i,d) → (i+1,d+1)` for any character ≠ target[i]
3. **Insertion**: `(i,d) → (i,d+1)` for any character
4. **Deletion**: `(i,d) → (i+1,d+1)` with epsilon input

### Computing Different Edit Distances

The example demonstrates three scenarios:

#### 1. Uniform Edit Distance (Standard Levenshtein)
```rust
// All operations cost 1.0
let uniform_fst = build_edit_distance_fst("hello", 2, 1.0, 1.0, 1.0);
```

#### 2. Weighted Edit Distance
```rust
// Custom costs: insertion=1.5, deletion=1.2, substitution=2.0
let weighted_fst = build_edit_distance_fst("hello", 2, 1.5, 1.2, 2.0);
```

#### 3. Biased Edit Distance
```rust
// Heavily penalize substitutions, prefer insertions/deletions
let biased_fst = build_edit_distance_fst("hello", 2, 0.8, 0.8, 3.0);
```

### String-to-String Distance Computation

The `compute_edit_distance` function:
1. Builds an edit distance FST for the target string
2. Creates a linear FST for the input string
3. Composes them to find valid alignment paths
4. Extracts the shortest path weight as the distance

```rust
fn compute_edit_distance(
    input: &str,
    target: &str,
    costs: (f32, f32, f32), // (insertion, deletion, substitution)
) -> f32
```

## Usage

Run the example with:
```bash
cargo run --example edit_distance
```

### Example Output

```
=== Edit Distance with Custom Weights ===

1. Uniform weights (standard Levenshtein):
   Distance from "helo" to "hello": 1.00 (insertion of 'l')
   Distance from "kitten" to "sitting": 3.00

2. Custom weights (insertion=1.5, deletion=1.2, substitution=2.0):
   Distance from "helo" to "hello": 1.50 (insertion preferred)
   Distance from "kitten" to "sitting": 5.20

3. Substitution-heavy weights (insertion=0.8, deletion=0.8, substitution=3.0):
   Distance from "helo" to "hello": 0.80 (insertion strongly preferred)
   Distance from "kitten" to "sitting": 3.20

Weight scheme effects:
- Standard: Treats all operations equally
- Custom: Slight preference for deletions over insertions over substitutions
- Biased: Strong preference for insertions/deletions over substitutions
```

## Key Features Demonstrated

### 1. Lattice-Based Construction
The FST is built as a 2D lattice that efficiently represents all possible edit sequences, following the dynamic programming structure but in FST form.

### 2. Configurable Operation Costs
Unlike standard algorithms that use uniform costs, this shows how FSTs naturally support arbitrary semiring weights for different operations.

### 3. Tropical Semiring Application
Uses the tropical semiring (min-plus) where:
- Addition (`plus`) finds the minimum cost path
- Multiplication (`times`) accumulates costs along paths
- Optimal for shortest path problems

### 4. Composition-Based Distance
Demonstrates the power of FST composition: composing the edit distance lattice with input string FSTs gives all valid alignments.

## Algorithmic Insights

### Time Complexity
- Lattice construction: O(n × k) states and O(|Σ| × n × k) arcs
- String FST creation: O(m) where m is input length
- Composition: O(m × n × k) in the worst case
- Shortest path: O(V + E) where V, E are states and arcs

### Space Complexity
- O(n × k) for the lattice structure
- Memory-efficient compared to full dynamic programming table

## Applications

### Spell Checking
Use with dictionary FSTs to find corrections:
```rust
let dict_fst = build_dictionary_fst(&["hello", "help", "held", "heel"]);
let edit_fst = build_edit_distance_fst("helo", 2, 1.0, 1.0, 1.0);
let corrections = compose(&edit_fst, &dict_fst)?;
```

### Bioinformatics
Sequence alignment with custom gap penalties and substitution matrices.

### Information Retrieval
Fuzzy matching with domain-specific cost models.

## Theoretical Background

This implementation follows the classical edit distance formulation but leverages FST composition for:
- **Modularity**: Separate edit costs from specific strings
- **Reusability**: One lattice FST works with any input string FST
- **Extensibility**: Easy to add new operations or constraints
- **Efficiency**: FST algorithms are highly optimized

## Related Examples

- [Morphological Analyzer](morphological_analyzer.md) - Uses edit distance concepts for morphological alternations
- [Word Correction](../examples/word_correction.rs) - Practical spell checking system
- [Pronunciation Lexicon](../examples/pronunciation_lexicon.rs) - Phonetic distance applications 