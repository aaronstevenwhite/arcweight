# Structural Operations

**Analyze and extract FST components**

*Project • Intersect • Difference*

Structural operations let you analyze FSTs, extract their components, and perform set operations. These operations are essential for understanding what your FSTs accept and produce.

## Projection

Projection extracts either the input or output language of an FST, creating an FSA (finite state automaton) that accepts only those strings.

### When to Use Projection

- Extract valid inputs or outputs
- Convert transducers to acceptors
- Analyze FST languages
- Validate FST construction

### Input Projection

```rust,ignore
use arcweight::prelude::*;

fn extract_input_language(fst: &VectorFst<TropicalWeight>) -> Result<VectorFst<TropicalWeight>> {
    // Project onto input labels
    let input_fsa = project(fst, ProjectType::Input)?;
    
    // Result accepts same inputs, but output = input
    Ok(input_fsa)
}
```text

### Output Projection

```rust,ignore
fn extract_output_language(fst: &VectorFst<TropicalWeight>) -> Result<VectorFst<TropicalWeight>> {
    // Project onto output labels
    let output_fsa = project(fst, ProjectType::Output)?;
    
    // Result accepts original outputs as inputs
    Ok(output_fsa)
}
```text

### Practical Example: Vocabulary Extraction

```rust,ignore
fn extract_vocabularies(
    translator: &VectorFst<TropicalWeight>
) -> Result<(HashSet<String>, HashSet<String>)> {
    // Get source language vocabulary
    let source_fsa = project(translator, ProjectType::Input)?;
    let source_words = extract_accepted_strings(&source_fsa)?;
    
    // Get target language vocabulary  
    let target_fsa = project(translator, ProjectType::Output)?;
    let target_words = extract_accepted_strings(&target_fsa)?;
    
    Ok((source_words, target_words))
}
```text

### Using Projection for Validation

```rust,ignore
fn validate_translation_coverage(
    translator: &VectorFst<TropicalWeight>,
    required_vocabulary: &VectorFst<TropicalWeight>
) -> Result<bool> {
    // Extract what translator can handle
    let input_language = project(translator, ProjectType::Input)?;
    
    // Check if it covers required vocabulary
    let uncovered = difference(required_vocabulary, &input_language)?;
    
    // If difference is empty, full coverage
    Ok(uncovered.num_states() == 0)
}
```text

## Intersection

Intersection creates an FST that accepts only strings accepted by both input FSTs, with outputs combined appropriately.

### When to Use Intersection

- Find common elements
- Apply constraints
- Filter languages
- Implement "and" operations

### Basic Intersection

```rust,ignore
fn find_common_words(
    vocab1: &VectorFst<TropicalWeight>,
    vocab2: &VectorFst<TropicalWeight>
) -> Result<VectorFst<TropicalWeight>> {
    // Words in both vocabularies
    let common = intersect(vocab1, vocab2)?;
    
    println!("Vocab1: {} words, Vocab2: {} words, Common: {} words",
        count_paths(vocab1)?,
        count_paths(vocab2)?,
        count_paths(&common)?);
    
    Ok(common)
}
```text

### Intersection for Constraint Application

```rust,ignore
fn apply_constraints(
    generator: &VectorFst<TropicalWeight>,
    constraints: Vec<VectorFst<TropicalWeight>>
) -> Result<VectorFst<TropicalWeight>> {
    let mut result = generator.clone();
    
    // Apply each constraint via intersection
    for constraint in constraints {
        result = intersect(&result, &constraint)?;
        
        // Check if we still have valid outputs
        if result.num_states() == 0 {
            return Err("Constraints are too restrictive".into());
        }
    }
    
    Ok(result)
}
```text

### Real-World Example: Password Validation

```rust,ignore
struct PasswordValidator {
    length_constraint: VectorFst<TropicalWeight>,
    char_constraint: VectorFst<TropicalWeight>,
    complexity_constraint: VectorFst<TropicalWeight>,
}

impl PasswordValidator {
    fn validate(&self, password: &str) -> Result<bool> {
        // Build FST for the password
        let password_fst = build_string_fst(password)?;
        
        // Must satisfy all constraints
        let valid = intersect(&password_fst, &self.length_constraint)?;
        let valid = intersect(&valid, &self.char_constraint)?;
        let valid = intersect(&valid, &self.complexity_constraint)?;
        
        // Non-empty intersection means valid
        Ok(valid.num_states() > 0)
    }
}
```text

## Difference

Difference creates an FST accepting strings in the first FST but not in the second.

### When to Use Difference

- Remove unwanted elements
- Find unique elements
- Implement "not" operations
- Compare FST languages

### Basic Difference

```rust,ignore
fn find_unique_words(
    vocab: &VectorFst<TropicalWeight>,
    common_words: &VectorFst<TropicalWeight>
) -> Result<VectorFst<TropicalWeight>> {
    // Words in vocab but not in common_words
    let unique = difference(vocab, common_words)?;
    
    Ok(unique)
}
```text

### Difference for Filtering

```rust,ignore
fn filter_blocklist(
    text: &VectorFst<TropicalWeight>,
    blocklist: &VectorFst<TropicalWeight>
) -> Result<VectorFst<TropicalWeight>> {
    // Remove blocked words
    let filtered = difference(text, blocklist)?;
    
    // Optimize result
    let cleaned = connect(&filtered)?;
    
    Ok(cleaned)
}
```text

### Real-World Example: Spell Checker Candidates

```rust,ignore
fn get_correction_candidates(
    misspelling: &str,
    dictionary: &VectorFst<TropicalWeight>,
    common_errors: &VectorFst<TropicalWeight>
) -> Result<Vec<String>> {
    // Generate possible corrections
    let candidates = generate_edits(misspelling)?;
    
    // Keep only valid words
    let valid_words = intersect(&candidates, dictionary)?;
    
    // Exclude known common errors
    let good_candidates = difference(&valid_words, common_errors)?;
    
    // Extract suggestions
    extract_strings(&good_candidates)
}
```text

## Advanced Structural Operations

### Composition with Projection

```rust,ignore
fn extract_translation_pairs(
    source_text: &VectorFst<TropicalWeight>,
    translator: &VectorFst<TropicalWeight>
) -> Result<Vec<(String, String)>> {
    // Get translations
    let translations = compose(source_text, translator)?;
    
    // Extract input-output pairs
    let mut pairs = Vec::new();
    
    for path in enumerate_paths(&translations)? {
        let source = project_path(&path, ProjectType::Input)?;
        let target = project_path(&path, ProjectType::Output)?;
        pairs.push((source, target));
    }
    
    Ok(pairs)
}
```text

### Set Operations Pipeline

```rust,ignore
fn analyze_vocabulary_overlap(
    vocabs: Vec<VectorFst<TropicalWeight>>
) -> Result<VocabularyAnalysis> {
    // Find intersection of all vocabularies
    let mut common = vocabs[0].clone();
    for vocab in &vocabs[1..] {
        common = intersect(&common, vocab)?;
    }
    
    // Find unique words in each vocabulary
    let mut unique_words = Vec::new();
    for (i, vocab) in vocabs.iter().enumerate() {
        let mut others = vocabs[0].clone();
        for (j, other) in vocabs.iter().enumerate() {
            if i != j {
                others = union(&others, other)?;
            }
        }
        
        let unique = difference(vocab, &others)?;
        unique_words.push(unique);
    }
    
    Ok(VocabularyAnalysis {
        common_words: common,
        unique_words,
    })
}
```text

### Language Comparison

```rust,ignore
fn compare_languages(
    lang1: &VectorFst<TropicalWeight>,
    lang2: &VectorFst<TropicalWeight>
) -> Result<LanguageComparison> {
    // What's in both
    let intersection = intersect(lang1, lang2)?;
    
    // What's only in lang1
    let only_in_1 = difference(lang1, lang2)?;
    
    // What's only in lang2
    let only_in_2 = difference(lang2, lang1)?;
    
    // Union (everything)
    let union = union(lang1, lang2)?;
    
    Ok(LanguageComparison {
        common: count_paths(&intersection)?,
        unique_to_1: count_paths(&only_in_1)?,
        unique_to_2: count_paths(&only_in_2)?,
        total: count_paths(&union)?,
    })
}
```text

## Practical Applications

### FST Debugging

```rust,ignore
fn debug_fst_coverage(
    fst: &VectorFst<TropicalWeight>,
    test_inputs: &[String]
) -> Result<CoverageReport> {
    // Get accepted inputs
    let input_lang = project(fst, ProjectType::Input)?;
    
    let mut accepted = 0;
    let mut rejected = Vec::new();
    
    for input in test_inputs {
        let input_fst = build_string_fst(input)?;
        let result = intersect(&input_fst, &input_lang)?;
        
        if result.num_states() > 0 {
            accepted += 1;
        } else {
            rejected.push(input.clone());
        }
    }
    
    Ok(CoverageReport {
        total: test_inputs.len(),
        accepted,
        rejected,
    })
}
```text

### Language Model Analysis

```rust,ignore
fn analyze_language_model(
    model: &VectorFst<TropicalWeight>,
    corpus: &VectorFst<TropicalWeight>
) -> Result<ModelAnalysis> {
    // What the model can generate
    let model_lang = project(model, ProjectType::Output)?;
    
    // What appears in corpus
    let corpus_lang = project(corpus, ProjectType::Input)?;
    
    // Coverage analysis
    let covered = intersect(&model_lang, &corpus_lang)?;
    let uncovered = difference(&corpus_lang, &model_lang)?;
    let hallucinated = difference(&model_lang, &corpus_lang)?;
    
    Ok(ModelAnalysis {
        coverage: count_paths(&covered)? as f32 / count_paths(&corpus_lang)? as f32,
        uncovered_words: extract_strings(&uncovered)?,
        hallucinated_words: extract_strings(&hallucinated)?,
    })
}
```text

## Performance Considerations

### Operation Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Projection | O(V + E) | O(V + E) |
| Intersection | O(V₁ × V₂) | O(V₁ × V₂) |
| Difference | O(V₁ × V₂) | O(V₁ × V₂) |

### Optimization Strategies

1. **Minimize before set operations**: Smaller FSTs → faster operations
2. **Use lazy evaluation**: For large FSTs, consider lazy intersection/difference
3. **Early termination**: Stop when you find what you need

```rust,ignore
fn early_termination_search(
    fst1: &VectorFst<TropicalWeight>,
    fst2: &VectorFst<TropicalWeight>
) -> Result<bool> {
    // Check if intersection is non-empty without building full result
    let intersection = lazy_intersect(fst1, fst2);
    
    // Just check if start state has any valid paths
    has_any_path(&intersection)
}
```text

## Common Patterns

### Pattern: Multi-Level Filtering

```rust,ignore
fn multi_level_filter(
    input: &VectorFst<TropicalWeight>,
    filters: Vec<(&str, VectorFst<TropicalWeight>)>
) -> Result<VectorFst<TropicalWeight>> {
    let mut result = input.clone();
    
    for (name, filter) in filters {
        println!("Applying filter: {}", name);
        let before = count_paths(&result)?;
        
        result = intersect(&result, &filter)?;
        
        let after = count_paths(&result)?;
        println!("  Reduced from {} to {} paths", before, after);
    }
    
    Ok(result)
}
```text

### Pattern: Incremental Vocabulary Building

```rust,ignore
struct VocabularyBuilder {
    vocabulary: VectorFst<TropicalWeight>,
    excluded: VectorFst<TropicalWeight>,
}

impl VocabularyBuilder {
    fn add_words(&mut self, words: &VectorFst<TropicalWeight>) -> Result<()> {
        // Add new words, excluding blocked ones
        let filtered = difference(words, &self.excluded)?;
        self.vocabulary = union(&self.vocabulary, &filtered)?;
        Ok(())
    }
    
    fn add_exclusions(&mut self, blocked: &VectorFst<TropicalWeight>) -> Result<()> {
        // Add to exclusion list and remove from vocabulary
        self.excluded = union(&self.excluded, blocked)?;
        self.vocabulary = difference(&self.vocabulary, blocked)?;
        Ok(())
    }
}
```text

## Best Practices

### 1. Validate Operations

```rust,ignore
fn safe_intersection(
    fst1: &VectorFst<TropicalWeight>,
    fst2: &VectorFst<TropicalWeight>
) -> Result<VectorFst<TropicalWeight>> {
    // Check if FSTs are compatible
    if !are_compatible(fst1, fst2) {
        return Err("Incompatible FSTs for intersection".into());
    }
    
    let result = intersect(fst1, fst2)?;
    
    // Warn if result is empty
    if result.num_states() == 0 {
        eprintln!("Warning: Intersection produced empty FST");
    }
    
    Ok(result)
}
```text

### 2. Use Type-Safe Wrappers

```rust,ignore
struct InputLanguage(VectorFst<TropicalWeight>);
struct OutputLanguage(VectorFst<TropicalWeight>);

fn project_safe(fst: &VectorFst<TropicalWeight>) -> (InputLanguage, OutputLanguage) {
    let input = project(fst, ProjectType::Input).unwrap();
    let output = project(fst, ProjectType::Output).unwrap();
    
    (InputLanguage(input), OutputLanguage(output))
}
```text

## Next Steps

Now that you understand structural operations:

1. **Learn advanced techniques** → [Advanced Topics](advanced-topics.md)
2. **Apply to real problems** → [Examples Gallery](../examples/)
3. **Understand the theory** → [Core Concepts](../core-concepts/)

---

**Ready for advanced topics?** Continue to [Advanced Topics](advanced-topics.md)