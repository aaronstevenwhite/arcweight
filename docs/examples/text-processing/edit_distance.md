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

```
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

### Key Insight

Traditional implementations use a 2D dynamic programming table, while the FST approach offers several advantages. The algorithm creates a **lattice FST** that encodes all possible transformations.

| Approach | Advantages | Use Cases |
|----------|------------|-----------|
| **Traditional DP** | Simple implementation, well-understood | Single pair comparisons |
| **FST Approach** | Reusable, composable, flexible costs | Multiple queries, complex workflows |

### Understanding the State Space

#### State Representation

Each state `(i,d)` represents:
- **i**: Characters processed from target string
- **d**: Edit operations used so far

This encoding allows tracking progress through the transformation process.

#### Arc Structure and Meaning

| Component | Purpose | Examples |
|-----------|---------|----------|
| **Input Label** | Character from source string | `'a'`, `ε` (epsilon) |
| **Output Label** | Character from target string | `'b'`, `ε` (epsilon) |
| **Weight** | Cost of this operation | `1.0`, `0.5`, `2.0` |
| **Destination** | Next state after operation | `state_42` |

### FST Advantages

**Reusability**  
Build once for target, use with multiple source strings

**Composability**  
Combine with other FSTs for complex operations

**Path Preservation**  
Maintains transformation sequences, not just distances

**Flexible Semirings**  
Support probabilistic or tropical computations

## Code Walkthrough

Let's explore each component and design decision in detail.

### Core Function: Building the Edit Distance FST

```rust
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
```

> **Function Purpose**: Creates an FST that can compute edit distance from any source string to the specified target string.

### State Space Design

**Key Insight: 2D Lattice Organization**

Organize states in a systematic grid structure that mirrors dynamic programming tables.

```rust
// Create lattice: states[i][d] = state after matching i chars with d edits
let mut states = vec![vec![]; n + 1];
for state_row in states.iter_mut() {
    for _d in 0..=k {
        state_row.push(fst.add_state());
    }
}
```

**Rows (i = 0 to n)**  
Position in target string
- Row 0: Start of target
- Row n: End of target

**Columns (d = 0 to k)**  
Number of edits used
- Column 0: No edits
- Column k: Maximum edits

### Setting Start and Final States

```rust
fst.set_start(states[0][0]);  // Start: 0 chars matched, 0 edits

// Final states: reached end of target with ≤ k edits
for d in 0..=k {
    fst.set_final(states[n][d], TropicalWeight::one());
}
```

**Start State Logic**  
Begin at position `(0,0)`
- 0 characters processed
- 0 edit operations used
- Top-left corner of our grid

**Final States Logic**  
Any state at row `n`
- All target characters processed
- Any edit count ≤ k is valid
- Enables finding various solutions

> **Weight Insight**: `TropicalWeight::one()` (value 0.0) on final states means no additional cost for reaching the end - the path weight alone determines the total cost.

## Three Types of Transitions

**Edit Operations as FST Arcs**

Each edit operation translates to a specific FST arc pattern with distinct input/output labels and state transitions.

### 1. Match Operation (No Cost)

**Perfect Character Match**

When characters match, advance in both strings without incrementing the edit count.

```rust
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

**Skip Target Character**

Deletion means we skip a character in the target string without consuming input.

```rust
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

**Semantic Interpretation**: "Produce this target character without consuming any input"

### 3. Insertion and Substitution Operations

**Source Character Operations**

These operations handle characters from the source string that need processing.

**Insertion Operation**

```rust
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

**Logic**: Consume source character, produce nothing, stay at same target position.

**Substitution Operation**

```rust
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

**Logic**: Transform one character into another, advance both positions.

| Operation | Input | Output | State Transition | Purpose |
|-----------|-------|--------|------------------|---------|
| **Insertion** | `c` | `ε` | `(i,d) → (i,d+1)` | Handle extra source characters |
| **Substitution** | `c` | `target[i]` | `(i,d) → (i+1,d+1)` | Handle mismatched characters |

## FST Implementation

**The Power of FST Composition**

Real power emerges when combining simple FSTs to solve complex problems.

```rust
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

### Step-by-Step Composition Process

**1. Build Edit Distance Transducer**  
Create the lattice FST that encodes all possible ways to transform any string into the target.  
**Benefit**: Reusable for multiple source strings!

**2. Create Source String Acceptor**  
Build a simple linear FST that accepts only the source string.  
**Structure**: Each arc = one character, weights = 1.0

**3. Perform Composition**  
Find all paths where source acceptor's output matches edit FST's input.  
**Result**: Only valid transformation paths remain

**4. Find Shortest Path**  
Seek the minimum cost transformation using Dijkstra's algorithm.  
**Semiring**: Tropical ensures minimum cost, not probability

**5. Extract Distance**  
Traverse the optimal path to sum weights and get total edit distance.  
**Output**: Single optimal transformation cost

## Example Output

**Live Demo Results**

When you run the example, you'll see these transformations in action:

### Standard Levenshtein Distance (all operations cost 1.0)

```
  'kitten' → 'sitting': 3  ✅
  'saturday' → 'sunday': 3  ✅  
  'hello' → 'hallo': 1  ✅
  'abc' → 'abc': 0  ✅ (perfect match)
  'abc' → 'def': 3  ✅
```

### Custom Weights (insert=0.5, delete=2.0, substitute=1.0)

```
  'kitten' → 'sitting': 2.5  📉 (cheaper insertions)
  'saturday' → 'sunday': 3  ➡️
  'hello' → 'hallo': 1  ➡️
```

### Expensive Substitutions (insert=1.0, delete=1.0, substitute=3.0)

```
  'kitten' → 'sitting': 5  📈 (avoids substitutions)
  'saturday' → 'sunday': 8  📈
  'hello' → 'hallo': 2  📈 (delete+insert cheaper than substitute)
```

## Understanding the Results

**Deep Analysis**

Let's analyze how different cost models affect optimal transformations.

### Standard Levenshtein Distance

With uniform costs (1.0 for all operations), we get the classic edit distance:

**"kitten" → "sitting" (distance: 3)**

The FST finds one optimal transformation sequence: substitute 'k' → 's' yielding `sitten` (cost: 1.0), substitute 'e' → 'i' yielding `sittin` (cost: 1.0), and insert 'g' yielding `sitting` (cost: 1.0), for a total cost of 3.0.

Alternative paths exist (like deleting "kitten" and inserting "sitting" = 13 operations), but the FST automatically finds the minimum cost path.

**"saturday" → "sunday" (distance: 3)**

This transformation showcases a more complex optimal path: keep 's' yielding `s` (cost: 0), delete 'a' and 't' keeping `s` (cost: 2.0), keep 'u' yielding `su` (cost: 0), delete 'r' keeping `su` (cost: 1.0), and keep 'n', 'd', 'a', 'y' yielding `sunday` (cost: 0), for a total cost of 3.0.

The FST correctly identifies that aligning the common suffix "nday" minimizes the total cost.

### Custom Weight Effects

Different cost models dramatically change the optimal transformation strategy:

**Cheap Insertions (0.5 cost)**

With `insertion=0.5, deletion=2.0, substitution=1.0`, the algorithm now prefers insertions over other operations. "kitten" → "sitting" achieves distance 2.5 with a possible path of substitute 'k'→'s' (1.0) + substitute 'e'→'i' (1.0) + insert 'g' (0.5). This models scenarios like typing where adding characters is easier than correcting.

**Expensive Substitutions (3.0 cost)**

With `insertion=1.0, deletion=1.0, substitution=3.0`, the algorithm avoids substitutions, preferring insert+delete pairs. "kitten" → "sitting" achieves distance 5.0, where instead of substituting 'k'→'s', it might delete 'k' (1.0) + insert 's' (1.0), costing 2.0 instead of 3.0 for direct substitution. This models scenarios where changing a character requires significant effort.

## Applications

Understanding edit distance through FSTs opens up numerous practical use cases.

### 1. Spell Checking

**Most Common Application**

Use edit distance to find spelling corrections with intelligent cost modeling.

```rust
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

**Output:**
```
Suggestion: hello (distance: 1)  // Missing 'l'
Suggestion: help (distance: 1)   // 'o' → 'p'
Suggestion: held (distance: 2)   // 'o' → 'd'
Suggestion: hero (distance: 2)   // 'l' → 'r'
```

Enhancement ideas include weighting common typos lower (teh → the), considering phonetic similarity (nite → night), and using frequency data to rank suggestions.

### 2. Approximate String Matching

```rust
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

```rust
// Compare genetic sequences with biological costs
let seq1 = "ACGTACGT";
let seq2 = "ACTTACGT";

// Biological costs: insertions/deletions more expensive than substitutions
let insertion_cost = 2.0;   // Gap penalty
let deletion_cost = 2.0;    // Gap penalty  
let substitution_cost = 1.0; // Point mutation

let alignment_cost = compute_edit_distance(seq1, seq2, insertion_cost, deletion_cost, substitution_cost)?;
```

## Advanced Concepts

### State Space Optimization

The FST creates `(n+1) × (k+1)` states where `n` equals target string length and `k` equals maximum allowed edits. For large strings or high edit distances, consider pruning (remove states with costs above threshold), band limitation (only consider edits within diagonal band), and incremental construction (build FST on-demand).

### Extending the Algorithm

**Character-specific costs:**
```rust
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
```rust
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

---

