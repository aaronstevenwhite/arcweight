# Word Correction FST Example

This example demonstrates how to build a comprehensive spell checking and word correction system using finite state transducers. It combines edit distance calculations, dictionary lookup, and contextual analysis to provide accurate spelling corrections.

## Overview

The word correction system implements:
1. **Dictionary-based spell checking** with FST lookup
2. **Edit distance correction** for finding similar words
3. **Phonetic similarity** for sound-alike corrections
4. **Context-aware suggestions** using n-gram models
5. **Real-word error detection** (valid words used incorrectly)
6. **Performance optimization** for real-time applications

## Key Components

### Error Types

```rust
#[derive(Debug, Clone, PartialEq)]
enum SpellingErrorType {
    NonWord,     // "teh" → "the" (not in dictionary)
    RealWord,    // "there" → "their" (wrong word in context)
    Phonetic,    // "nite" → "night" (sounds similar)
    Typo,        // "thier" → "their" (common typing error)
    Split,       // "alot" → "a lot" (word boundary error)
    Merge,       // "in to" → "into" (incorrect split)
}
```

### Correction Candidate

```rust
#[derive(Debug, Clone)]
struct CorrectionCandidate {
    word: String,
    original: String,
    edit_distance: f32,
    phonetic_distance: f32,
    frequency_score: f32,
    context_score: f32,
    error_type: SpellingErrorType,
    confidence: f32,
}
```

## Core Components

### Dictionary FST Construction

```rust
fn build_dictionary_fst(
    words: &[String],
    frequencies: Option<&[f32]>,
) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    fst.set_start(start);
    
    for (word, freq) in words.iter().zip(frequencies.unwrap_or(&vec![1.0; words.len()])) {
        let mut current_state = start;
        
        // Build trie structure
        for ch in word.chars() {
            let label = ch as u32;
            
            // Find or create transition
            let next_state = find_or_create_state(&mut fst, current_state, label);
            current_state = next_state;
        }
        
        // Mark final state with frequency-based weight
        let weight = TropicalWeight::new(-freq.ln()); // Higher freq = lower cost
        fst.set_final(current_state, weight);
    }
    
    fst
}
```

### Edit Distance Correction

```rust
fn find_edit_distance_corrections(
    word: &str,
    dictionary: &VectorFst<TropicalWeight>,
    max_distance: usize,
) -> Vec<CorrectionCandidate> {
    // Build edit distance FST for the misspelled word
    let edit_fst = build_edit_distance_fst(word, max_distance, 1.0, 1.0, 1.0);
    
    // Compose with dictionary to find matches
    let corrections = compose(&edit_fst, dictionary)?;
    
    // Extract paths with their costs
    extract_correction_candidates(&corrections, word)
}
```

### Phonetic Similarity

```rust
fn build_phonetic_fst(
    word: &str,
    phonetic_alphabet: &PhoneticAlphabet,
) -> VectorFst<TropicalWeight> {
    // Convert word to phonetic representation
    let phonemes = word_to_phonemes(word, phonetic_alphabet);
    
    // Build FST that accepts phonetically similar words
    build_phonetic_variation_fst(&phonemes)
}

fn phonetic_corrections(
    word: &str,
    dictionary: &VectorFst<TropicalWeight>,
) -> Vec<CorrectionCandidate> {
    let phonetic_fst = build_phonetic_fst(word, &ENGLISH_PHONETIC_ALPHABET);
    let matches = compose(&phonetic_fst, dictionary)?;
    extract_phonetic_candidates(&matches, word)
}
```

### Context-Aware Correction

```rust
struct ContextModel {
    bigram_fst: VectorFst<TropicalWeight>,
    trigram_fst: VectorFst<TropicalWeight>,
    word_frequencies: HashMap<String, f32>,
}

impl ContextModel {
    fn score_correction(
        &self,
        candidate: &str,
        left_context: &[String],
        right_context: &[String],
    ) -> f32 {
        let bigram_score = self.score_bigram(left_context.last(), candidate);
        let trigram_score = self.score_trigram(left_context, candidate);
        let unigram_score = self.word_frequencies.get(candidate).unwrap_or(&0.0001);
        
        // Weighted combination
        0.1 * unigram_score.ln() + 0.4 * bigram_score + 0.5 * trigram_score
    }
}
```

## Advanced Features

### Real-Word Error Detection

```rust
fn detect_real_word_errors(
    sentence: &[String],
    context_model: &ContextModel,
    threshold: f32,
) -> Vec<(usize, Vec<CorrectionCandidate>)> {
    let mut errors = Vec::new();
    
    for (i, word) in sentence.iter().enumerate() {
        let left_context = &sentence[..i];
        let right_context = &sentence[i+1..];
        
        let context_score = context_model.score_in_context(word, left_context, right_context);
        
        if context_score < threshold {
            let confusable_words = find_confusable_words(word);
            let candidates = score_candidates_in_context(
                &confusable_words, left_context, right_context, context_model
            );
            
            if !candidates.is_empty() {
                errors.push((i, candidates));
            }
        }
    }
    
    errors
}
```

### Common Error Patterns

```rust
fn build_common_errors_fst() -> VectorFst<TropicalWeight> {
    let error_patterns = vec![
        // Transposition errors
        ("teh", "the"),
        ("adn", "and"),
        ("taht", "that"),
        
        // Common substitutions
        ("recieve", "receive"),
        ("seperate", "separate"),
        ("occured", "occurred"),
        
        // Phonetic confusions
        ("there", "their"),
        ("your", "you're"),
        ("its", "it's"),
        
        // Keyboard proximity errors
        ("ajd", "and"),   // j/n proximity
        ("tge", "the"),   // g/h proximity
    ];
    
    build_error_pattern_fst(&error_patterns)
}
```

### Compound Word Handling

```rust
fn handle_compound_words(
    word: &str,
    dictionary: &VectorFst<TropicalWeight>,
) -> Vec<CorrectionCandidate> {
    let mut candidates = Vec::new();
    
    // Try splitting at different positions
    for split_pos in 1..word.len() {
        let (left, right) = word.split_at(split_pos);
        
        if is_valid_word(left, dictionary) && is_valid_word(right, dictionary) {
            candidates.push(CorrectionCandidate {
                word: format!("{} {}", left, right),
                original: word.to_string(),
                edit_distance: 0.0,
                error_type: SpellingErrorType::Split,
                confidence: calculate_split_confidence(left, right),
                ..Default::default()
            });
        }
    }
    
    // Try merging with adjacent words
    // (handled at sentence level)
    
    candidates
}
```

## Usage

Run the example with:
```bash
cargo run --example word_correction
```

### Example Output

```
=== Word Correction System Demo ===

Dictionary loaded: 50,000 words
Context model loaded: 2M bigrams, 5M trigrams

Single Word Corrections:
Input: "recieve"
1. receive (edit: 1, conf: 0.95)
2. perceive (edit: 2, conf: 0.12)
3. conceive (edit: 2, conf: 0.08)

Input: "seperate" 
1. separate (edit: 1, conf: 0.98)
2. desperate (edit: 2, conf: 0.05)

Phonetic Corrections:
Input: "nite"
1. night (phonetic match, conf: 0.85)
2. knight (phonetic match, conf: 0.15)

Context-Aware Corrections:
Sentence: "I went their yesterday"
Error at position 2: "their"
1. there (context score: 0.92)
2. they're (context score: 0.08)

Compound Word Suggestions:
Input: "alot"
1. a lot (split suggestion, conf: 0.90)
2. allot (single word, conf: 0.10)

Performance Metrics:
- Dictionary lookup: 0.1ms avg
- Edit distance (k=2): 2.3ms avg  
- Phonetic matching: 1.8ms avg
- Context scoring: 0.5ms avg
- Total pipeline: 4.7ms avg
```

## Performance Optimizations

### Incremental Processing

```rust
struct IncrementalCorrector {
    dictionary_fst: VectorFst<TropicalWeight>,
    cached_edit_fsts: LRUCache<String, VectorFst<TropicalWeight>>,
    context_model: ContextModel,
}

impl IncrementalCorrector {
    fn correct_as_you_type(&mut self, partial_word: &str) -> Vec<String> {
        // Real-time suggestions as user types
        if partial_word.len() < 3 {
            return self.prefix_suggestions(partial_word);
        }
        
        // Use cached edit FSTs for performance
        let edit_fst = self.cached_edit_fsts
            .get_or_insert(partial_word, || {
                build_edit_distance_fst(partial_word, 2, 1.0, 1.0, 1.0)
            });
            
        let matches = compose(edit_fst, &self.dictionary_fst)?;
        extract_top_suggestions(&matches, 5)
    }
}
```

### Parallel Processing

```rust
fn parallel_sentence_correction(
    sentences: &[Vec<String>],
    corrector: &WordCorrector,
) -> Vec<Vec<CorrectionCandidate>> {
    use rayon::prelude::*;
    
    sentences.par_iter()
        .map(|sentence| corrector.correct_sentence(sentence))
        .collect()
}
```

### Memory Optimization

```rust
struct CompactDictionary {
    trie: MinimalTrie,        // Compressed trie representation
    frequencies: Vec<f32>,    // Parallel frequency array
    metadata: DictMetadata,   // Word counts, statistics
}

impl CompactDictionary {
    fn memory_footprint(&self) -> usize {
        self.trie.size_bytes() + 
        self.frequencies.len() * std::mem::size_of::<f32>() +
        std::mem::size_of::<DictMetadata>()
    }
}
```

## Applications

### Text Editors and IDEs
- Real-time spell checking
- Autocomplete and suggestions
- Code comment and documentation checking
- Variable name validation

### Search Engines
- Query spell correction
- "Did you mean" suggestions
- Fuzzy search implementation
- Auto-correction of search terms

### Social Media and Messaging
- Real-time message correction
- Hashtag normalization
- Username suggestion
- Content moderation support

### Educational Tools
- Language learning assistance
- Writing improvement feedback
- Vocabulary enhancement
- Error pattern analysis

## Evaluation Metrics

### Accuracy Metrics

```rust
struct CorrectionMetrics {
    precision: f32,     // Correct corrections / Total corrections
    recall: f32,        // Correct corrections / Total errors
    f1_score: f32,      // Harmonic mean of precision and recall
    
    // Error-type specific metrics
    nonword_accuracy: f32,
    realword_accuracy: f32,
    phonetic_accuracy: f32,
    
    // Performance metrics
    avg_response_time: Duration,
    memory_usage: usize,
    throughput: f32,    // Words/second
}
```

### Benchmark Suite

```rust
fn benchmark_correction_system(
    corrector: &WordCorrector,
    test_corpus: &TestCorpus,
) -> BenchmarkResults {
    let mut results = BenchmarkResults::new();
    
    for test_case in test_corpus.iter() {
        let start_time = Instant::now();
        let corrections = corrector.correct_word(&test_case.misspelled);
        let elapsed = start_time.elapsed();
        
        results.add_result(
            corrections,
            test_case.expected,
            elapsed,
            test_case.error_type,
        );
    }
    
    results.calculate_metrics()
}
```

## Related Examples

- [Edit Distance](edit_distance.md) - Core edit distance algorithm
- [Pronunciation Lexicon](pronunciation_lexicon.md) - Phonetic similarity
- [Number Date Normalizer](number_date_normalizer.md) - Text normalization
- [Morphological Analyzer](morphological_analyzer.md) - Linguistic analysis

## References and Standards

- **Levenshtein Distance**: Edit distance algorithm foundation
- **Soundex/Metaphone**: Phonetic similarity algorithms
- **N-gram Language Models**: Context modeling
- **Unicode Text Segmentation**: Word boundary detection
- **Common Crawl**: Large-scale frequency data