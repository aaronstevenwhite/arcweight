# Edit Distance Example

This example demonstrates how to compute edit distance between strings using weighted finite state transducers. It implements the classic dynamic programming algorithm as a WFST, enabling flexible cost schemes and demonstrating core WFST concepts.

## Quick Start

```bash
# Run the edit distance example
cargo run --example edit_distance

# Explore the source code
cat examples/edit_distance.rs
```

## What You'll Learn

### Algorithm Translation
Convert classic edit distance dynamic programming into finite state transducer operations, bridging traditional algorithms with FST concepts.

### Flexible Cost Models
Encode different operation costs as semiring weights, enabling custom distance metrics for specialized applications.

### Path Tracking
Compute distances while preserving optimal transformation sequences between strings for detailed analysis.

### Composition Power
Combine simple FSTs through composition to solve complex string processing problems efficiently.

## Core Concepts

### Algorithm Overview

**Edit Distance (Levenshtein Distance)** measures the minimum number of single-character edits needed to transform one string into another.

> **Historical Context**: Vladimir Levenshtein introduced the concept (1966), and Wagner & Fischer developed the efficient dynamic programming algorithm (1974).

### Allowed Operations

#### Insertion
Add a character to the string  
`"cat" → "cart"`

#### Deletion
Remove a character from the string  
`"cart" → "cat"`

#### Substitution
Replace one character with another  
`"cat" → "bat"`

### Mathematical Foundation

For strings `s₁` of length `m` and `s₂` of length `n`, the edit distance `d(i,j)` is defined recursively:

```text
d(i,j) = min {
    d(i-1,j) + deletion_cost,      // Delete from s₁
    d(i,j-1) + insertion_cost,     // Insert into s₁ 
    d(i-1,j-1) + subst_cost        // Substitute in s₁
}
```

**Base Cases:**
- `d(0,j) = j × insertion_cost` (insert j characters)
- `d(i,0) = i × deletion_cost` (delete i characters)

## Implementation

### FST Encoding

The traditional dynamic programming algorithm employs a two-dimensional table. The finite state transducer approach encodes this computation as a lattice structure.

| Approach | Advantages | Use Cases |
|----------|------------|-----------|
| **Traditional DP** | Simple implementation, well-understood | Single pair comparisons |
| **FST Approach** | Reusable, composable, flexible costs | Multiple queries, complex workflows |

### Understanding the State Space

#### State Representation

Each state `(i,d)` represents:
- **i**: Characters processed from target string
- **d**: Edit operations used so far

This representation tracks algorithmic progress through both strings.

#### Arc Structure and Meaning

| Component | Purpose | Examples |
|-----------|---------|----------|
| **Input Label** | Character from source string | `'a'`, `ε` (epsilon) |
| **Output Label** | Character from target string | `'b'`, `ε` (epsilon) |
| **Weight** | Cost of this operation | `1.0`, `0.5`, `2.0` |
| **Destination** | Next state after operation | `state_42` |

### Transducer Properties

- **Reusability**: Single transducer serves multiple source strings
- **Composability**: Integration with other transducers via composition
- **Path Preservation**: Complete transformation sequences retained
- **Semiring Flexibility**: Support for various weight algebras

## Implementation Details

### Transducer Construction

```rust,ignore
fn build_edit_distance_fst(
    target: &str,                // String we're transforming TO
    k: usize,                    // Maximum allowed edits
    insertion_cost: f32,         // Cost to insert a character
    deletion_cost: f32,          // Cost to delete a character  
    substitution_cost: f32,      // Cost to substitute a character
) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let target_chars: Vec<char> = target.chars().collect();
    let n = target_chars.len();
```text

This function constructs a transducer computing edit distance to a fixed target string.

### State Space Design

The state space forms a two-dimensional lattice corresponding to the dynamic programming table:

```rust,ignore
// Create lattice: states[i][d] = state after matching i chars with d edits
let mut states = vec![vec![]; n + 1];
for state_row in states.iter_mut() {
    for _d in 0..=k {
        state_row.push(fst.add_state());
    }
}
```

- **Rows (i ∈ [0, n])**: Position in target string
- **Columns (d ∈ [0, k])**: Number of edit operations applied

### Initial and Final States

```rust,ignore
fst.set_start(states[0][0]);

// Mark states at target end as final
for d in 0..=k {
    fst.set_final(states[n][d], TropicalWeight::one());
}
```

The start state represents position (0,0) in the lattice. Final states occur at row n with any edit count d ≤ k, representing successful transformation of the entire target string.

## Transition Types

Edit operations are encoded as transducer arcs with specific input/output label patterns:

### 1. Match Operation

Character matches advance both string positions without cost:

```rust,ignore
// Advance both strings when characters match
fst.add_arc(
    current_state,
    Arc::new(
        target_chars[i] as u32,  // Input character
        target_chars[i] as u32,  // Output character (same)
        TropicalWeight::one(),   // No cost (weight = 0.0)
        states[i + 1][d],        // Next position, same edit count
    ),
);
```

| Aspect | Value | Meaning |
|--------|-------|---------|
| **Input Label** | `target_chars[i]` | Character from source string |
| **Output Label** | `target_chars[i]` | Same character (perfect match) |
| **Weight** | `TropicalWeight::one()` | No cost (0.0 in tropical semiring) |
| **State Transition** | `(i,d) → (i+1,d)` | Advance position, same edit count |

### 2. Deletion Operation

Deletion transitions consume no input while outputting target characters:

```rust,ignore
// Skip character in target (epsilon input)
fst.add_arc(
    current_state,
    Arc::new(
        0,                              // Epsilon (no input consumed)
        target_chars[i] as u32,         // Output the target character
        TropicalWeight::new(deletion_cost),
        states[i + 1][d + 1],           // Advance target, increment edits
    ),
);
```

| Aspect | Value | Meaning |
|--------|-------|---------|
| **Input Label** | `0` (epsilon) | No character consumed from source |
| **Output Label** | `target_chars[i]` | Target character being "deleted" |
| **Weight** | `deletion_cost` | Cost of this deletion operation |
| **State Transition** | `(i,d) → (i+1,d+1)` | Advance target, increment edit count |

This models removing a character from the source string.

### 3. Insertion and Substitution Operations

**Insertion Operation**

```rust,ignore
// Insert character (consume from source without producing output)
fst.add_arc(
    current_state,
    Arc::new(
        c as u32,                       // Input character to insert
        0,                              // Epsilon (no output produced)
        TropicalWeight::new(insertion_cost),
        states[i][d + 1],               // Same target position, increment edits
    ),
);
```

**Substitution Operation**

```rust,ignore
// Substitute character (only if different from target)
if c != target_chars[i] {
    fst.add_arc(
        current_state,
        Arc::new(
            c as u32,                   // Input character
            target_chars[i] as u32,     // Output target character
            TropicalWeight::new(substitution_cost),
            states[i + 1][d + 1],       // Advance both, increment edits
        ),
    );
}
```


| Operation | Input | Output | State Transition | Purpose |
|-----------|-------|--------|------------------|---------|
| **Insertion** | `c` | `ε` | `(i,d) → (i,d+1)` | Handle extra source characters |
| **Substitution** | `c` | `target[i]` | `(i,d) → (i+1,d+1)` | Handle mismatched characters |

## Transducer Composition

The edit distance computation employs transducer composition:

```rust,ignore
fn compute_edit_distance(source: &str, target: &str, ...) -> Result<f32> {
    // 1. Build edit distance transducer for target
    let edit_fst = build_edit_distance_fst(target, max_edits, ...);
    
    // 2. Build linear acceptor for source string
    let source_fst = string_acceptor(source);
    
    // 3. Compose to find valid transformations
    let composed: VectorFst<TropicalWeight> = compose_default(&source_fst, &edit_fst)?;
    
    // 4. Find shortest path (minimum cost transformation)
    let shortest: VectorFst<TropicalWeight> = shortest_path(&composed, config)?;
    
    // 5. Extract minimum cost
    // ... traverse shortest path to find final cost ...
}
```

### Composition Steps

1. **Edit Distance Transducer**: Construct lattice encoding all transformations to target
2. **Source Acceptor**: Linear automaton accepting the source string
3. **Composition**: Compute intersection of valid paths
4. **Shortest Path**: Extract minimum-weight path using tropical semiring
5. **Cost Extraction**: Sum arc weights along optimal path

## Example Output

### Uniform Costs (Standard Levenshtein)

```text
'kitten' → 'sitting': 3
'saturday' → 'sunday': 3
'hello' → 'hallo': 1
'abc' → 'abc': 0
'abc' → 'def': 3
```

### Variable Costs (insert=0.5, delete=2.0, substitute=1.0)

```text
'kitten' → 'sitting': 2.5
'saturday' → 'sunday': 3
'hello' → 'hallo': 1
```

### High Substitution Cost (insert=1.0, delete=1.0, substitute=3.0)

```text
'kitten' → 'sitting': 5
'saturday' → 'sunday': 8
'hello' → 'hallo': 2
```

## Analysis

Cost models significantly influence optimal transformation paths.

### Uniform Cost Model

With unit costs, the algorithm computes standard Levenshtein distance. For "kitten" → "sitting", one optimal path involves:
- Substitute 'k' → 's' (cost: 1)
- Substitute 'e' → 'i' (cost: 1)
- Insert 'g' (cost: 1)
- Total: 3

The transducer explores all transformation sequences, selecting the minimum-cost path.

### Variable Cost Models

**Reduced Insertion Cost**: With insertion=0.5, the algorithm favors insertions. For "kitten" → "sitting", the optimal path may utilize the cheaper insertion operation, achieving distance 2.5.

**Increased Substitution Cost**: With substitution=3.0, the algorithm prefers deletion-insertion pairs over direct substitution. A substitution costing 3.0 is replaced by delete (1.0) + insert (1.0) = 2.0.

## Applications

### 1. Spell Checking

Edit distance enables spelling correction through candidate generation and ranking:

```rust,ignore
// Find words within edit distance 2 of user input
let typo = "helo";
let dictionary = vec!["hello", "help", "held", "hero", "helm", "hell"];

// Build a custom spell checker with keyboard-aware costs
fn keyboard_distance(c1: char, c2: char) -> f32 {
    // Adjacent keys on QWERTY have lower substitution cost
    match (c1, c2) {
        ('q', 'w') | ('w', 'q') => 0.5,  // Adjacent
        ('o', 'p') | ('p', 'o') => 0.5,  // Adjacent
        // ... more keyboard mappings ...
        _ => 1.0  // Default cost
    }
}

for word in dictionary {
    let distance = compute_edit_distance(typo, word, 1.0, 1.0, 1.0)?;
    if distance <= 2.0 {
        println!("Suggestion: {} (distance: {})", word, distance);
    }
}
```

Keyboard-aware cost models can improve suggestion quality by assigning lower costs to adjacent key substitutions.

### 2. Approximate String Matching

```rust,ignore
// Find database entries similar to query
let query = "John Smith";
let database = vec!["Jon Smith", "John Smyth", "Jane Smith"];

let threshold = 2.0;
for entry in database {
    let similarity = compute_edit_distance(&query, entry, 1.0, 1.0, 1.0)?;
    if similarity <= threshold {
        println!("Match: {} (distance: {})", entry, similarity);
    }
}
```

### 3. DNA Sequence Alignment

```rust,ignore
// Biological sequence alignment with gap penalties
let seq1 = "ACGTACGT";
let seq2 = "ACTTACGT";

let gap_penalty = 2.0;        // Insertion/deletion cost
let mismatch_penalty = 1.0;   // Substitution cost

let alignment_cost = compute_edit_distance(
    seq1, seq2, gap_penalty, gap_penalty, mismatch_penalty
)?;
```

## Advanced Concepts

### State Space Optimization

The transducer contains (n+1) × (k+1) states. Optimization strategies include:
- Beam search pruning
- Diagonal band constraints
- Lazy state construction

### Extending the Algorithm

**Character-specific costs:**
```rust,ignore
fn char_specific_cost(from: char, to: char) -> f32 {
    match (from, to) {
        // Vowel substitutions are cheaper
        ('a', 'e') | ('e', 'a') => 0.5,
        ('i', 'e') | ('e', 'i') => 0.5,
        // Consonant clusters
        ('f', 'ph') => 0.3,  // Common misspelling
        _ => 1.0,
    }
}
```

**Phonetic similarity:**
```rust,ignore
// Use Soundex or metaphone distance for phonetic matching
let phonetic_weight = if sounds_similar(from_char, to_char) { 0.5 } else { 1.0 };
```

### Performance Considerations

Time complexity is \\(O(n \times k × |\Sigma|)\\) where \\(|\Sigma|\\) is alphabet size. Space complexity is \\(O(n \times k)\\). Optimization uses determinization and minimization for repeated queries.

## Key Takeaways

**Flexible Algorithms**  
FSTs enable the same mathematical framework to handle different cost schemes elegantly.

**Composition Power**  
Complex computations emerge from simple FST combinations through composition.

**Weights Matter**  
Different cost functions dramatically change optimal solutions and transformation strategies.

**Practical Impact**  
Edit distance underlies many real-world string processing tasks and applications.

## Related Examples

**[String Alignment](string_alignment.md)**  
Extract and visualize optimal alignments

**[Spell Checking](spell_checking.md)**  
Practical spell checking with edit distance

**[Pronunciation Lexicon](../practical-applications/pronunciation_lexicon.md)**  
Phonetic edit distance applications

## References

**Levenshtein, V.I. (1966)**  
*Binary codes capable of correcting deletions, insertions, and reversals.*  
Soviet Physics Doklady, 10(8):707-710

**Wagner, R.A. & Fischer, M.J. (1974)**  
*The string-to-string correction problem.*  
Journal of the ACM, 21(1):168-173

**Jurafsky, D. & Martin, J.H. (2023)**  
*Speech and Language Processing (3rd Edition)*  
Chapter 2: Minimum Edit Distance


