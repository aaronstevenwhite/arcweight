# Core Operations

**Essential FST building blocks for combining and transforming**

*Compose • Unite • Concatenate*

Core operations are the fundamental ways to combine FSTs. Master these three operations and you can build almost any FST-based system.

## Composition

Composition is the most powerful FST operation. It chains two FSTs together, where the output of the first becomes the input to the second.

### When to Use Composition

Use composition when you need to:
- Build processing pipelines
- Chain transformations
- Create complex systems from simple components

### Basic Composition

```rust
use arcweight::prelude::*;

// FST1: a:x, b:y (maps 'a' to 'x', 'b' to 'y')
let fst1 = /* ... */;

// FST2: x:1, y:2 (maps 'x' to '1', 'y' to '2')  
let fst2 = /* ... */;

// Composition: a:1, b:2 (maps 'a' to '1', 'b' to '2')
let composed = compose(&fst1, &fst2)?;
```

### How Composition Works

```
Input "a" → FST1 → "x" → FST2 → "1"
         ↘                    ↗
           Composed FST: a:1
```

### Real-World Example: Text Processing Pipeline

```rust
fn build_text_pipeline() -> Result<VectorFst<TropicalWeight>> {
    // Component FSTs
    let lowercaser = build_lowercase_fst()?;
    let normalizer = build_normalizer_fst()?;
    let tokenizer = build_tokenizer_fst()?;
    
    // Chain them together
    let pipeline = compose(&lowercaser, &normalizer)?;
    let pipeline = compose(&pipeline, &tokenizer)?;
    
    Ok(pipeline)
}

// Usage
let pipeline = build_text_pipeline()?;
let input = "Hello, World!";
let output = apply_fst(&pipeline, input)?;
// Output: ["hello", "world"]
```

### Advanced: Composition with Filters

For large FSTs, use composition filters to improve performance:

```rust
use arcweight::algorithms::{compose_with_filter, SequenceComposeFilter};

// Use sequence filter for better performance
let filter = SequenceComposeFilter::new();
let composed = compose_with_filter(&fst1, &fst2, filter)?;
```

### Composition Patterns

#### Pattern 1: Multi-Stage Pipeline
```rust
// Linguistic pipeline: Tokenize → POS Tag → Parse
let pos_tagged = compose(&tokenizer, &pos_tagger)?;
let parsed = compose(&pos_tagged, &parser)?;
```

#### Pattern 2: Rule Application
```rust
// Apply multiple phonological rules
let stage1 = compose(&word_fst, &rule1)?;
let stage2 = compose(&stage1, &rule2)?;
let result = compose(&stage2, &rule3)?;
```

#### Pattern 3: Constraint Satisfaction
```rust
// Spell checker: Error model + Dictionary
let candidates = compose(&error_model, &dictionary)?;
```

## Union

Union combines multiple FSTs into one that accepts any input accepted by any component FST.

### When to Use Union

Use union when you need to:
- Combine alternative vocabularies
- Support multiple formats or patterns
- Build fallback mechanisms

### Basic Union

```rust
// FST1: accepts "cat", "dog"
let animals = /* ... */;

// FST2: accepts "red", "blue"
let colors = /* ... */;

// Union: accepts "cat", "dog", "red", "blue"
let combined = union(&animals, &colors)?;
```

### Building Vocabularies

```rust
fn build_vocabulary() -> Result<VectorFst<TropicalWeight>> {
    let mut vocabulary = VectorFst::new();
    
    // Add multiple word lists
    let medical_terms = load_medical_fst()?;
    let general_english = load_general_fst()?;
    let technical_terms = load_technical_fst()?;
    
    // Combine them all
    vocabulary = union(&vocabulary, &medical_terms)?;
    vocabulary = union(&vocabulary, &general_english)?;
    vocabulary = union(&vocabulary, &technical_terms)?;
    
    // Optimize the result
    optimize(&mut vocabulary)?;
    
    Ok(vocabulary)
}
```

### Union for Pattern Matching

```rust
fn build_pattern_matcher() -> Result<VectorFst<TropicalWeight>> {
    // Different patterns for dates
    let us_dates = build_fst_from_regex(r"\d{2}/\d{2}/\d{4}")?;     // MM/DD/YYYY
    let iso_dates = build_fst_from_regex(r"\d{4}-\d{2}-\d{2}")?;   // YYYY-MM-DD
    let eu_dates = build_fst_from_regex(r"\d{2}\.\d{2}\.\d{4}")?;  // DD.MM.YYYY
    
    // Accept any format
    let date_matcher = union(&us_dates, &iso_dates)?;
    let date_matcher = union(&date_matcher, &eu_dates)?;
    
    Ok(date_matcher)
}
```

### Weighted Union

When using weighted FSTs, union preserves weights:

```rust
// Word frequencies
let common_words = /* FST with high-frequency words (low weights) */;
let rare_words = /* FST with low-frequency words (high weights) */;

// Combined vocabulary preserves frequency information
let full_vocabulary = union(&common_words, &rare_words)?;
```

## Concatenation

Concatenation creates an FST that accepts strings from the first FST followed by strings from the second FST.

### When to Use Concatenation

Use concatenation when you need to:
- Build sequences
- Implement morphological rules
- Create pattern templates

### Basic Concatenation

```rust
// FST1: accepts "hello", "hi"
let greetings = /* ... */;

// FST2: accepts "world", "there"
let targets = /* ... */;

// Concatenation: accepts "hello world", "hello there", "hi world", "hi there"
let phrases = concat(&greetings, &targets)?;
```

### Building Morphology

```rust
fn build_verb_forms() -> Result<VectorFst<TropicalWeight>> {
    // Verb stems
    let stems = build_fst_from_list(&["walk", "talk", "play"])?;
    
    // Suffixes
    let suffixes = build_fst_from_list(&["ing", "ed", "s"])?;
    
    // All verb forms
    let verb_forms = concat(&stems, &suffixes)?;
    // Accepts: "walking", "walked", "walks", "talking", etc.
    
    Ok(verb_forms)
}
```

### Template-Based Generation

```rust
fn build_sentence_templates() -> Result<VectorFst<TropicalWeight>> {
    // Components
    let subjects = build_fst_from_list(&["I", "You", "They"])?;
    let space = build_fst_from_string(" ")?;
    let verbs = build_fst_from_list(&["like", "love", "enjoy"])?;
    let objects = build_fst_from_list(&["coding", "FSTs", "Rust"])?;
    
    // Build template: Subject + " " + Verb + " " + Object
    let subj_space = concat(&subjects, &space)?;
    let subj_verb = concat(&subj_space, &verbs)?;
    let verb_space = concat(&subj_verb, &space)?;
    let full_sentence = concat(&verb_space, &objects)?;
    
    // Generates: "I like coding", "You love FSTs", etc.
    Ok(full_sentence)
}
```

## Combining Operations

The real power comes from combining these operations:

### Example: Spell Checker with Suggestions

```rust
fn build_spell_checker() -> Result<VectorFst<TropicalWeight>> {
    // Build components
    let dictionary = load_dictionary_fst()?;
    let common_typos = load_typos_fst()?;
    let edit_distance_1 = build_edit_distance_fst(1)?;
    let edit_distance_2 = build_edit_distance_fst(2)?;
    
    // Combine valid words and common corrections
    let valid_words = union(&dictionary, &common_typos)?;
    
    // Allow 1 or 2 edits
    let one_edit = compose(&edit_distance_1, &dictionary)?;
    let two_edits = compose(&edit_distance_2, &dictionary)?;
    
    // Combine all suggestions
    let suggestions = union(&valid_words, &one_edit)?;
    let suggestions = union(&suggestions, &two_edits)?;
    
    Ok(suggestions)
}
```

### Example: Linguistic Pipeline

```rust
fn process_text(input: &str) -> Result<Vec<String>> {
    // Build pipeline
    let normalizer = build_normalizer()?;
    let tokenizer = build_tokenizer()?;
    let morph_analyzer = build_morphology()?;
    
    // Compose them
    let pipeline = compose(&normalizer, &tokenizer)?;
    let pipeline = compose(&pipeline, &morph_analyzer)?;
    
    // Apply to input
    apply_fst(&pipeline, input)
}
```

## Performance Tips

### Composition
- Use filters for large FSTs
- Compose smaller FSTs first
- Consider lazy composition for very large FSTs

### Union
- Minimize component FSTs before union
- Use determinization after union if needed
- Consider weight pushing for probability FSTs

### Concatenation
- Determinize components first for better result
- Use epsilon removal after concatenation
- Consider right-to-left vs left-to-right building

## Common Pitfalls

### ❌ Composition Order Matters
```rust
// Not the same!
let result1 = compose(&a, &b)?;
let result2 = compose(&b, &a)?;
// These produce different FSTs!
```

### ❌ Union Weight Conflicts
```rust
// Careful with weights in union
let fst1 = /* "word" with weight 1.0 */;
let fst2 = /* "word" with weight 2.0 */;
let unioned = union(&fst1, &fst2)?;
// "word" now has two paths with different weights
```

### ❌ Concatenation Explosion
```rust
// Can create very large FSTs
let thousand_words = /* 1000 words */;
let thousand_suffixes = /* 1000 suffixes */;
let huge_fst = concat(&thousand_words, &thousand_suffixes)?;
// Results in 1,000,000 combinations!
```

## Next Steps

Now that you understand core operations:

1. **Optimize your FSTs** → [Optimization Operations](optimization-operations.md)
2. **Find best paths** → [Path Operations](path-operations.md)
3. **See real examples** → [Examples Gallery](../examples/)

---

**Ready to optimize?** Continue to [Optimization Operations](optimization-operations.md)