# String Alignment

This example demonstrates computing and visualizing optimal string alignments using FSTs to extract the actual sequence of operations that transform one string into another.

## Overview

While edit distance tells you *how different* two strings are, string alignment shows *exactly how* they differ. This distinction is crucial for many applications: spell checkers can show what changed, version control systems display file differences, bioinformatics applications align DNA sequences to identify mutations, and machine translation systems align parallel texts for training data.

String alignment extends edit distance by preserving the transformation path, allowing you to reconstruct and visualize the exact sequence of insertions, deletions, and substitutions. FSTs make this natural by encoding operation types in the output labels, enabling both distance computation and alignment extraction in a single framework.

The key insight is using FST output labels to encode different operation types (match, substitute, insert, delete), allowing the same transducer to compute both optimal cost and optimal alignments.

## Quick Start

```bash
cargo run --example string_alignment
```text

## What You'll Learn

- **Operation Encoding**: Modify edit distance FSTs to preserve operation types in output labels  
- **Alignment Extraction**: Extract all optimal alignment paths from composed FSTs
- **Visualization Techniques**: Create visual representations of string transformations
- **Multiple Alignments**: Handle cases where multiple optimal paths have the same cost
- **Custom Scoring**: Adapt alignment algorithms for different domains (text, DNA, code)
- **Performance Optimization**: Efficient algorithms for long sequences and real-time processing

## Core Concepts

### The Alignment Problem

Given two strings S and T, an alignment is a way of writing them one above the other, possibly with gaps (represented by dashes), such that every character appears exactly once, no column contains two gaps, and reading left-to-right preserves the original strings (ignoring gaps).

For example, aligning "KITTEN" and "SITTING":
```text
K I T T E - N
S I T T I N G
```text

This alignment shows K → S (substitution), I → I (match), T → T (match), T → T (match), E → I (substitution), - → N (insertion), and N → G (substitution).

### Optimal Alignments

An optimal alignment minimizes the total cost of operations. The cost depends on the scoring scheme. Unit costs assign cost 1 to all operations (standard edit distance). Weighted costs assign different costs for different operations. Position-specific costs vary by position in the string. Content-specific costs depend on the actual characters.

### Mathematical Formulation

An alignment A between strings S(1..m) and T(1..n) can be represented as a sequence of edit operations:

```text
A = (op₁, op₂, ..., opₖ)
```text

Where each operation opᵢ is one of Match(i,j) where S(i) = T(j), Substitute(i,j) where S(i) → T(j), Insert(j) where ε → T(j), or Delete(i) where S(i) → ε.

The cost of an alignment is:
```text
cost(A) = Σ cost(opᵢ)
```text

## Implementation

### Encoding Alignment Operations

The key insight is encoding operation types in the FST's output labels. We use a special encoding scheme:

```rust,ignore
// Operation encoding in output labels:
// 1000 + char_value: Match
// 2000 + char_value: Substitution (target character)
// 3000 + char_value: Insertion 
// 4000 + char_value: Deletion
```text

This allows us to distinguish between different ways of producing the same character. Output label 1097 represents Match 'a' (97 is ASCII for 'a'). Output label 2097 represents Substitute something to 'a'. Output label 3097 represents Insert 'a'.

### Building the Alignment FST

The alignment FST extends the edit distance FST with encoded operations:

```rust,ignore
fn build_alignment_fst(target: &str, max_edits: usize, costs: Costs) -> VectorFst {
    let mut fst = VectorFst::new();
    let target_chars: Vec<char> = target.chars().collect();
    
    // State lattice: states[position][edits_used]
    // ... create states ...
    
    // Add transitions with encoded operations
    for i in 0..=n {
        for d in 0..states[i].len() {
            // Match transition
            if i < n {
                fst.add_arc(states[i][d], Arc::new(
                    target_chars[i] as u32,          // Input: target char
                    1000 + target_chars[i] as u32,   // Output: encoded match
                    TropicalWeight::one(),            // Cost: 0
                    states[i+1][d],                   // Next state
                ));
            }
            
            // Substitution transitions
            for c in alphabet {
                if c != target_chars[i] {
                    fst.add_arc(states[i][d], Arc::new(
                        c as u32,                     // Input: source char
                        2000 + target_chars[i] as u32,// Output: encoded substitution
                        TropicalWeight::new(subst_cost),
                        states[i+1][d+1],
                    ));
                }
            }
            
            // Similar for insertions and deletions...
        }
    }
}
```text

### Extracting Alignments

After composition, we traverse the FST to extract alignment paths:

```rust,ignore
fn extract_alignment(fst: &VectorFst) -> Vec<Alignment> {
    let mut alignments = Vec::new();
    
    // Traverse all paths from start to final states
    for path in all_paths(fst) {
        let mut operations = Vec::new();
        
        for label in path.output_labels {
            let operation = decode_operation(label);
            operations.push(operation);
        }
        
        alignments.push(Alignment {
            operations,
            cost: path.weight,
        });
    }
    
    alignments
}
```text

## FST Implementation

### Basic Alignment Display

The most common visualization shows three lines:

```text
Source:    K I T T E - N
Alignment: * | | | * + *
Target:    S I T T I N G
```text

Where `|` indicates a match, `*` indicates a substitution, `+` indicates an insertion, and `-` indicates a deletion.

### Implementation

```rust,ignore
fn visualize(alignment: &Alignment) -> String {
    let mut source_line = String::new();
    let mut align_line = String::new();
    let mut target_line = String::new();
    
    for op in &alignment.operations {
        match op {
            AlignmentOp::Match(s, t) => {
                source_line.push(*s);
                align_line.push('|');
                target_line.push(*t);
            }
            AlignmentOp::Substitute(s, t) => {
                source_line.push(*s);
                align_line.push('*');
                target_line.push(*t);
            }
            AlignmentOp::Insert(c) => {
                source_line.push('-');
                align_line.push('+');
                target_line.push(*c);
            }
            AlignmentOp::Delete(c) => {
                source_line.push(*c);
                align_line.push('-');
                target_line.push('-');
            }
        }
    }
}
```text

## Running the Example

```bash
cargo run --example string_alignment
```text

### Sample Output

```text
String Alignment Example
========================

1. Simple Alignment Examples:
-----------------------------

Aligning 'kitten' -> 'sitting':
Cost: 3.0
Visualization:
kitten-
*|||*+*
sitting

Aligning 'saturday' -> 'sunday':
Cost: 3.0
Visualization:
saturday
*|-|-|||
s-un-day

2. Multiple Optimal Alignments:
--------------------------------

Aligning 'abc' -> 'aec':
Found 2 optimal alignment(s) with cost 2.0:

Alignment #1:
abc
|*|
aec

Alignment #2:
a-bc
|+*|
aebc

3. Biological Sequence Alignment:
---------------------------------
Sequence 1: acgtacgt
Sequence 2: acttacgt

Optimal alignment (cost: 2.0):
acgtacgt
||**||||
acttacgt

Alignment statistics:
  Matches: 6
  Substitutions: 2
  Insertions/Deletions: 0
  Similarity: 75.0%
```text

## Understanding Multiple Optimal Alignments

One fascinating aspect of string alignment is that multiple different alignments can have the same optimal cost. This happens when different sequences of operations lead to the same total cost.

### Example: "abc" → "aec"

There are two ways to transform "abc" to "aec" with cost 2:

**Option 1: Direct substitution**
```text
a b c
| * |
a e c
```text
Operations: Match 'a', Substitute 'b'→'e', Match 'c'

**Option 2: Delete and insert**
```text
a - b c
| + - |
a e - c
```text
Operations: Match 'a', Insert 'e', Delete 'b', Match 'c'

Both have the same cost with unit weights, but might differ with custom costs.

### Tie-Breaking Strategies

When multiple alignments exist, we might prefer fewer gaps (minimize insertions/deletions), contiguous operations (group similar operations together), structure preservation (maintain relative positions), or domain-specific preferences (follow biological or linguistic preferences).

## Advanced Features

### 1. Biological Sequence Alignment

DNA and protein alignment requires specialized scoring:

```rust,ignore
struct BiologicalScoring {
    match_score: f32,      // Positive score for matches
    mismatch_cost: f32,    // Substitution penalty
    gap_open: f32,         // Cost to start a gap
    gap_extend: f32,       // Cost to extend a gap
}

// Transitions favor matches in conserved regions
let scoring = BiologicalScoring {
    match_score: -2.0,     // Negative = reward in tropical semiring
    mismatch_cost: 1.0,    
    gap_open: 5.0,         
    gap_extend: 0.5,       
};
```text

Key considerations include conservation (some positions are more important than others), gap penalties (affine gap costs with open + extend are biologically realistic), and substitution matrices (PAM/BLOSUM matrices for amino acids).

### 2. Code Diff Visualization

For version control systems:

```rust,ignore
fn code_diff_alignment(old_code: &str, new_code: &str) -> DiffAlignment {
    // Tokenize by lines instead of characters
    let old_lines = old_code.lines().collect();
    let new_lines = new_code.lines().collect();
    
    // Build FST with line-level operations
    let alignment = compute_alignment(old_lines, new_lines);
    
    // Visualize as unified diff
    for op in alignment.operations {
        match op {
            LineOp::Keep(line) => println!("  {}", line),
            LineOp::Delete(line) => println!("- {}", line),
            LineOp::Insert(line) => println!("+ {}", line),
        }
    }
}
```text

### 3. Phonetic Alignment

For speech processing and pronunciation:

```rust,ignore
// Align phoneme sequences
let pronunciation1 = vec!["K", "IH", "T", "AH", "N"];  // 'kitten'
let pronunciation2 = vec!["S", "IH", "T", "IH", "NG"]; // 'sitting'

// Use phonetic similarity for costs
fn phonetic_distance(p1: &str, p2: &str) -> f32 {
    match (p1, p2) {
        // Similar sounds have lower substitution cost
        ("T", "D") | ("D", "T") => 0.5,  // Voiced/unvoiced pairs
        ("IH", "IY") => 0.3,              // Close vowels
        _ => 1.0,                         // Default
    }
}
```text

### 4. Musical Sequence Alignment

For comparing melodies or chord progressions:

```rust,ignore
struct Note {
    pitch: u8,      // MIDI note number
    duration: f32,  // In beats
}

fn melodic_alignment(melody1: &[Note], melody2: &[Note]) -> MusicAlignment {
    // Cost based on pitch distance and rhythm
    let pitch_weight = 1.0;
    let rhythm_weight = 0.5;
    
    let subst_cost = |n1: &Note, n2: &Note| {
        let pitch_diff = (n1.pitch as i32 - n2.pitch as i32).abs() as f32;
        let rhythm_diff = (n1.duration - n2.duration).abs();
        
        pitch_weight * pitch_diff + rhythm_weight * rhythm_diff
    };
    
    compute_weighted_alignment(melody1, melody2, subst_cost)
}
```text

## Performance Optimization

### 1. Banded Alignment

For similar sequences, restrict edits to a diagonal band:

```rust,ignore
fn banded_alignment(s1: &str, s2: &str, bandwidth: usize) -> Alignment {
    // Only consider states within bandwidth of the main diagonal
    let n = s1.len();
    let m = s2.len();
    
    for i in 0..n {
        for j in 0..m {
            if (i as i32 - j as i32).abs() <= bandwidth as i32 {
                // Process this cell
            }
        }
    }
}
```text

This reduces complexity from O(nm) to O(n×bandwidth).

### 2. Sparse Alignment

For very different sequences, use sparse state representation:

```rust,ignore
struct SparseAlignmentFST {
    states: HashMap<(usize, usize), StateId>,  // (position, edits) -> state
    threshold: f32,  // Prune states with cost > threshold
}
```text

### 3. Incremental Alignment

For real-time applications (e.g., collaborative editing):

```rust,ignore
struct IncrementalAligner {
    previous_alignment: Alignment,
    source_changes: Vec<TextEdit>,
    target_changes: Vec<TextEdit>,
}

impl IncrementalAligner {
    fn update(&mut self, change: TextEdit) -> Alignment {
        // Recompute only affected region
        let affected_range = self.find_affected_range(change);
        let local_alignment = self.align_range(affected_range);
        self.merge_local_alignment(local_alignment)
    }
}
```text

## Extending the Algorithm

### 1. Affine Gap Penalties

Biological sequences often use affine gap penalties where opening a gap costs more than extending it:

```rust,ignore
struct AffineGapFST {
    states: Vec<Vec<Vec<StateId>>>,  // [position][edits][gap_state]
    gap_open: f32,
    gap_extend: f32,
}

// Three state types per position:
// - Match state (can start any operation)
// - Insertion state (can only extend insertions)
// - Deletion state (can only extend deletions)
```text

### 2. Local Alignment

Find the best matching substring rather than full sequences:

```rust,ignore
fn local_alignment(s1: &str, s2: &str) -> LocalAlignment {
    // Allow free insertions/deletions at start/end
    let mut fst = build_alignment_fst(s2, ...);
    
    // Add epsilon transitions from start to any position
    for i in 0..s2.len() {
        fst.add_arc(start, Arc::new(
            0, 0, TropicalWeight::one(), states[i][0]
        ));
    }
    
    // Make all states final with zero cost
    for state in states {
        fst.set_final(state, TropicalWeight::one());
    }
}
```text

### 3. Constrained Alignment

Enforce certain positions to align:

```rust,ignore
struct ConstrainedAlignment {
    anchors: Vec<(usize, usize)>,  // (source_pos, target_pos) pairs
}

fn build_constrained_fst(target: &str, anchors: &[(usize, usize)]) -> VectorFst {
    // Only allow paths that pass through anchor points
    for (s_pos, t_pos) in anchors {
        // Remove transitions that would bypass anchors
        enforce_anchor(fst, s_pos, t_pos);
    }
}
```text

## Common Pitfalls and Solutions

### 1. Memory Usage with Long Sequences

**Problem**: FST size grows with sequence length squared.

**Solution**: Use checkpointing:
```rust,ignore
fn checkpoint_alignment(s1: &str, s2: &str, checkpoint_interval: usize) {
    // Divide into segments and align separately
    let segments = chunk_sequences(s1, s2, checkpoint_interval);
    let partial_alignments = segments.map(|(seg1, seg2)| {
        compute_alignment(seg1, seg2)
    });
    merge_alignments(partial_alignments)
}
```text

### 2. Numerical Precision

**Problem**: Floating-point errors accumulate in long alignments.

**Solution**: Use log-space computation or exact arithmetic:
```rust,ignore
// Use LogWeight instead of TropicalWeight for better precision
type LogFst = VectorFst<LogWeight>;
```text

### 3. Handling Unicode

**Problem**: Multi-byte characters and combining marks.

**Solution**: Use grapheme clusters:
```rust,ignore
use unicode_segmentation::UnicodeSegmentation;

let graphemes: Vec<&str> = text.graphemes(true).collect();
// Align graphemes instead of chars
```text

## Best Practices

### 1. Choose Appropriate Costs

Different applications need different scoring schemes:

```rust,ignore
enum AlignmentDomain {
    Text,        // Unit costs
    DNA,         // Match=0, Mismatch=1, Gap=2
    Protein,     // BLOSUM62 matrix
    Phonetic,    // Articulatory distance
}

fn get_scoring_scheme(domain: AlignmentDomain) -> ScoringScheme {
    match domain {
        AlignmentDomain::Text => ScoringScheme::unit_costs(),
        AlignmentDomain::DNA => ScoringScheme::dna_costs(),
        // etc.
    }
}
```text

### 2. Validate Input

Always check sequence properties:

```rust,ignore
fn validate_sequences(s1: &str, s2: &str) -> Result<()> {
    if s1.is_empty() || s2.is_empty() {
        return Err(anyhow!("Empty sequences not allowed"));
    }
    
    if s1.len() > MAX_SEQUENCE_LENGTH {
        return Err(anyhow!("Sequence too long"));
    }
    
    // Check character set
    if !s1.chars().all(|c| c.is_ascii_lowercase()) {
        return Err(anyhow!("Invalid characters"));
    }
    
    Ok(())
}
```text

### 3. Test Edge Cases

Always test boundary conditions:

```rust,ignore
#[cfg(test)]
mod tests {
    #[test]
    fn test_empty_strings() {
        assert!(align("", "abc").unwrap().cost == 3.0);  // 3 insertions
        assert!(align("abc", "").unwrap().cost == 3.0);  // 3 deletions
        assert!(align("", "").unwrap().cost == 0.0);     // Empty alignment
    }
    
    #[test]
    fn test_identical_strings() {
        let alignment = align("hello", "hello").unwrap();
        assert!(alignment.cost == 0.0);
        assert!(alignment.operations.iter().all(|op| {
            matches!(op, AlignmentOp::Match(_, _))
        }));
    }
}
```text

## Applications

### 1. Spell Checker with Corrections

```rust,ignore
fn spell_check_with_alignment(word: &str, dictionary: &[&str]) -> SpellCheckResult {
    let mut suggestions = Vec::new();
    
    for dict_word in dictionary {
        let alignment = compute_alignment(word, dict_word)?;
        
        if alignment.cost <= 2.0 {
            suggestions.push(SpellingSuggestion {
                word: dict_word.to_string(),
                distance: alignment.cost,
                corrections: describe_corrections(&alignment),
            });
        }
    }
    
    suggestions.sort_by_key(|s| s.distance);
    SpellCheckResult { suggestions }
}

fn describe_corrections(alignment: &Alignment) -> String {
    let mut description = Vec::new();
    
    for (i, op) in alignment.operations.iter().enumerate() {
        match op {
            AlignmentOp::Substitute(from, to) => {
                description.push(format!("Change '{}' to '{}'", from, to));
            }
            AlignmentOp::Insert(c) => {
                description.push(format!("Insert '{}'", c));
            }
            AlignmentOp::Delete(c) => {
                description.push(format!("Remove '{}'", c));
            }
            _ => {} // Matches don't need description
        }
    }
    
    description.join(", ")
}
```text

### 2. Collaborative Editing

```rust,ignore
struct CollaborativeDocument {
    base_text: String,
    user_versions: HashMap<UserId, String>,
}

impl CollaborativeDocument {
    fn merge_changes(&mut self, user_id: UserId, new_text: String) -> MergeResult {
        let base_alignment = compute_alignment(&self.base_text, &new_text)?;
        
        // Check for conflicts with other users
        for (other_user, other_text) in &self.user_versions {
            if other_user != &user_id {
                let other_alignment = compute_alignment(&self.base_text, other_text)?;
                
                if alignments_conflict(&base_alignment, &other_alignment) {
                    return MergeResult::Conflict(describe_conflict());
                }
            }
        }
        
        // Apply non-conflicting changes
        self.user_versions.insert(user_id, new_text);
        MergeResult::Success
    }
}
```text

## Related Examples

Related examples include **[Edit Distance](edit_distance.md)** as the foundation for string alignment, **[Spell Checking](spell_checking.md)** for applying alignment to spelling correction, and **[Transliteration](../practical-applications/transliteration.md)** for cross-script alignment challenges.

## References

- **Needleman, S.B. & Wunsch, C.D. (1970).** A general method applicable to the search for similarities in the amino acid sequence of two proteins. Journal of Molecular Biology, 48(3):443-453

- **Smith, T.F. & Waterman, M.S. (1981).** Identification of common molecular subsequences. Journal of Molecular Biology, 147(1):195-197

- **Gotoh, O. (1982).** An improved algorithm for matching biological sequences. Journal of Molecular Biology, 162(3):705-708

- **Durbin, R., Eddy, S.R., Krogh, A., & Mitchison, G. (1998).** Biological Sequence Analysis: Probabilistic Models of Proteins and Nucleic Acids. Cambridge University Press

- **Gusfield, D. (1997).** Algorithms on Strings, Trees, and Sequences: Computer Science and Computational Biology. Cambridge University Press