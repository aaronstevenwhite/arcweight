# Spell Checking Example

This example demonstrates building a production-ready spell checking system using FST-based dictionary storage and edit distance algorithms for intelligent error correction.

## Overview

Spell checking is one of the most practical applications of FSTs in everyday computing. Every word processor, search engine, and text input system relies on fast, accurate spelling correction. Traditional approaches use hash tables and manual edit distance calculations, while FSTs provide a unified mathematical framework that elegantly combines dictionary lookup with error correction.

This example shows how to build a complete spell checking system that can handle multiple pronunciations, customizable error models, and real-time suggestions. The FST approach enables sophisticated features like context-aware corrections, phonetic similarity, and keyboard-distance-based costs.

The system combines three key components: FST-based dictionary storage using trie structures for memory efficiency, edit distance transducers for finding words within k edits, and composition techniques for combining dictionary and error constraints.

## Quick Start

```bash
cargo run --example spell_checking
```

## What You'll Learn

- **FST Dictionary Design**: Build efficient trie-based dictionaries using FSTs for fast lookup
- **Edit Distance Integration**: Combine edit distance transducers with dictionary FSTs  
- **Customizable Error Models**: Implement sophisticated cost models for typing patterns
- **Composition Techniques**: Use FST composition to combine dictionary and error constraints
- **Performance Optimization**: Real-time processing and memory-efficient implementations
- **Production Features**: Frequency-based ranking, context awareness, and scalability

## Core Concepts

### Algorithm Overview

**Theoretical Foundation**

Spell checking represents a practical application of FST technology, used in text-based applications from word processors to search engines.

> **Historical Context**: Traditional approaches use hash tables and manual edit distance calculations, while the FST approach offers a unified mathematical model.

### FST Advantages

**Unified Model**  
Dictionary storage and error correction unified within a single mathematical framework

**Composability**  
Multiple constraints like dictionary lookup, edit distance, and context combined through FST composition

**Flexible Costs**  
Optimal corrections according to specified cost models for any alphabet or writing system

**Extensible**  
Phonetic matching, keyboard layouts, and additional scoring methods incorporated systematically

Several key concepts underlie FST-based spell checking implementations that bridge theoretical computer science with practical NLP applications.

### Edit Distance: The Foundation

Edit distance (Levenshtein distance) quantifies string differences by counting the minimum number of single-character edits needed to transform one string into another. The three allowed operations are insertion (add a character, fixing missing letters), deletion (remove a character, fixing extra letters), and substitution (replace a character, fixing wrong letters).

**Real-world examples with explanations:**
```
"helo" → "hello": 1 edit
  - Missing second 'l', insert it at position 3
  - This is why "hello" appears as a suggestion

"wrold" → "world": 1 edit
  - Transpose error: 'o' and 'r' swapped
  - Technically 1 substitution, though it's really a transposition

"frend" → "friend": 1 edit
  - Missing 'i', insert at position 2
  - Common error in English where 'ie' sounds like 'e'

"chekc" → "check": 1 edit
  - Another transpose: final 'k' and 'c' swapped
  - FST finds this as 1 substitution
```

Edit distance works for spell checking because most typos are single-character errors (80%+ of misspellings). Edit distance 1 covers missing letters, extra letters, wrong letters, and adjacent transpositions. Edit distance 2 covers combinations of above and non-adjacent transpositions. Beyond distance 2, suggestions become less relevant.

### Dictionary Representation: The Power of Tries

A trie (prefix tree) is a good data structure for dictionaries because it shares common prefixes between words. When encoded as an FST, we get both space efficiency and composability:

```rust
fn build_dictionary_fst(words: &[&str]) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    fst.set_start(start);

    // Build trie structure with shared prefixes
    let mut state_map: HashMap<Vec<char>, u32> = HashMap::new();
    state_map.insert(vec![], start);

    for word in words {
        let chars: Vec<char> = word.chars().collect();
        let mut prefix = vec![];

        for (i, &ch) in chars.iter().enumerate() {
            let current_state = *state_map.get(&prefix).unwrap();
            prefix.push(ch);

            // Create new state if needed
            if !state_map.contains_key(&prefix) {
                let new_state = fst.add_state();
                state_map.insert(prefix.clone(), new_state);
                
                fst.add_arc(current_state, Arc::new(
                    ch as u32, ch as u32,
                    TropicalWeight::one(),
                    new_state
                ));
            }

            // Mark final state for complete words
            if i == chars.len() - 1 {
                let final_state = *state_map.get(&prefix).unwrap();
                fst.set_final(final_state, TropicalWeight::one());
            }
        }
    }

    fst
}
```

Tries excel for dictionaries for several reasons. Prefix sharing allows words like "test", "testing", "tester" to share the prefix "test", where traditional approaches store each word separately (4+7+6 = 17 characters) while trie approaches store "test" once, then branch (4+3+2 = 9 characters). Fast lookup operates in O(m) time where m is word length, independent of dictionary size. Natural FST structure means each path from start to final state represents a valid word. Composability allows composition with other FSTs (like edit distance) efficiently.

**Example trie structure for `["cat", "cats", "car", "card"]`:**
```
     start
       |
      'c'
       |
      'a'
     /   \
   't'    'r'
   |       |
 (final) (final)
   |       |
  's'     'd'
   |       |
(final) (final)
```

## Implementation

### Edit Distance FST: The Heart of Spell Checking

The edit distance FST is a transducer that accepts any string within k edits of a target word. Understanding this component is important for FST-based spell checking.

### Construction Algorithm: Building the Edit Lattice

The edit distance FST creates a lattice structure that encodes all possible edit sequences:

```rust
fn build_edit_distance_fst(target: &str, k: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let target_chars: Vec<char> = target.chars().collect();
    let n = target_chars.len();

    // Create states: (position in target, edits used)
    // This creates a 2D grid of states where:
    // - First dimension: position in target string (0 to n)
    // - Second dimension: number of edits used (0 to k)
    let mut states = vec![vec![]; n + 1];
    for (i, state_row) in states.iter_mut().enumerate().take(n + 1) {
        for _j in 0..=k.min(i + k) {
            state_row.push(fst.add_state());
        }
    }

    fst.set_start(states[0][0]);  // Start at position 0 with 0 edits

    // Final states: reached end of target with ≤ k edits
    // Weight encodes the actual edit distance
    for j in 0..=k.min(n + k) {
        if j < states[n].len() {
            fst.set_final(states[n][j], TropicalWeight::new(j as f32));
        }
    }

    // Add transitions for each operation type
    add_match_transitions(&mut fst, &states, &target_chars, n, k);
    add_substitution_transitions(&mut fst, &states, &target_chars, n, k);
    add_deletion_transitions(&mut fst, &states, n, k);
    add_insertion_transitions(&mut fst, &states, n, k);

    fst
}
```

Key insights about the state space: State `states[i][j]` means we've processed i characters of the target using j edits. We only create states where edits ≤ k, saving memory. Any state at position n (end of target) is final, with weight = edit distance. Each path through the FST represents a specific edit sequence.

### Transition Types: Encoding Edit Operations

Each type of transition encodes a different edit operation. Understanding these is important for customizing spell checkers.

#### Match Transitions (No Cost)

When the input character matches the target character, we advance without using an edit:

```rust
fn add_match_transitions(
    fst: &mut VectorFst<TropicalWeight>,
    states: &[Vec<u32>],
    target_chars: &[char],
    n: usize,
    k: usize,
) {
    for i in 0..n {
        for j in 0..states[i].len() {
            if j <= i + k && j < states[i + 1].len() {
                // Add arc for matching the target character
                fst.add_arc(states[i][j], Arc::new(
                    target_chars[i] as u32,      // Input must match target[i]
                    target_chars[i] as u32,      // Output is the same
                    TropicalWeight::one(),       // Cost = 0 (no edit needed)
                    states[i + 1][j],           // Move to next position, same edit count
                ));
            }
        }
    }
}
```

For target "cat" and input "cat": at state (0,0) we see 'c' and match, moving to state (1,0); at state (1,0) we see 'a' and match, moving to state (2,0); at state (2,0) we see 't' and match, moving to state (3,0); state (3,0) is final with weight 0 (no edits needed).

#### Substitution Transitions (Cost 1)

Substitutions handle wrong characters by accepting any character different from the target:

```rust
fn add_substitution_transitions(
    fst: &mut VectorFst<TropicalWeight>, 
    states: &[Vec<u32>],
    target_chars: &[char],
    n: usize,
    k: usize,
) {
    for i in 0..n {
        for j in 0..states[i].len() {
            if j < k && j + 1 < states[i + 1].len() {
                // Accept any character except the target character
                // (matching is handled by match transitions)
                for c in b'a'..=b'z' {
                    if c as char != target_chars[i] {
                        fst.add_arc(states[i][j], Arc::new(
                            c as u32, c as u32,
                            TropicalWeight::new(1.0),  // Cost 1 edit
                            states[i + 1][j + 1],     // Advance position, increment edits
                        ));
                    }
                }
            }
        }
    }
}
```

For target "cat" and input "bat": at state (0,0) we see 'b' (not 'c') and substitute, moving to state (1,1) with cost 1; at state (1,1) we see 'a' and match, moving to state (2,1); at state (2,1) we see 't' and match, moving to state (3,1); state (3,1) is final with total weight 1 (one substitution).

**Deletion Transitions (cost 1):**
```rust
fn add_deletion_transitions(
    fst: &mut VectorFst<TropicalWeight>,
    states: &[Vec<u32>],
    n: usize,
    k: usize,
) {
    for i in 0..n {
        for j in 0..states[i].len() {
            if j < k && j + 1 < states[i + 1].len() {
                fst.add_arc(states[i][j], Arc::new(
                    0, 0,  // Epsilon transition (consume target char)
                    TropicalWeight::new(1.0),
                    states[i + 1][j + 1],
                ));
            }
        }
    }
}
```

**Insertion Transitions (cost 1):**
```rust
fn add_insertion_transitions(
    fst: &mut VectorFst<TropicalWeight>,
    states: &[Vec<u32>],
    n: usize,
    k: usize,
) {
    for i in 0..=n {
        for j in 0..states[i].len() {
            if j < k && j + 1 < states[i].len() {
                // Insert any character
                for c in b'a'..=b'z' {
                    fst.add_arc(states[i][j], Arc::new(
                        c as u32, c as u32,
                        TropicalWeight::new(1.0),
                        states[i][j + 1],
                    ));
                }
            }
        }
    }
}
```

### Spelling Correction Discovery

### FST Composition

Find spelling corrections by composing dictionary and edit distance FSTs:

```rust
fn find_spelling_corrections(
    dict_fst: &VectorFst<TropicalWeight>,
    target: &str,
    max_distance: usize,
) -> Result<Vec<(String, f32)>> {
    // Build edit distance FST for the target
    let edit_fst = build_edit_distance_fst(target, max_distance);

    // Compose dictionary with edit distance FST
    let composed: VectorFst<TropicalWeight> = compose_default(dict_fst, &edit_fst)?;

    // Find shortest paths to get best spelling corrections
    let config = ShortestPathConfig {
        nshortest: 10,  // Top 10 spelling corrections
        ..Default::default()
    };
    let shortest: VectorFst<TropicalWeight> = shortest_path(&composed, config)?;

    // Extract spelling corrections and their costs
    let mut results = Vec::new();
    if let Some(start) = shortest.start() {
        extract_paths(&shortest, start, &mut Vec::new(), 0.0, &mut results, &mut HashSet::new());
    }

    // Sort by edit distance (lower is better)
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    Ok(results)
}
```

### Path Extraction

Extract word candidates from the composed FST:

```rust
fn extract_paths(
    fst: &VectorFst<TropicalWeight>,
    state: u32,
    path: &mut Vec<char>,
    cost: f32,
    results: &mut Vec<(String, f32)>,
    visited: &mut HashSet<u32>,
) {
    if visited.contains(&state) {
        return;  // Avoid infinite loops
    }

    // Check if this is a final state (complete word)
    if fst.is_final(state) {
        let word: String = path.iter().collect();
        if let Some(weight) = fst.final_weight(state) {
            results.push((word, cost + weight.value()));
        }
    }

    visited.insert(state);

    // Follow all outgoing arcs
    for arc in fst.arcs(state) {
        if arc.olabel != 0 {  // Non-epsilon output
            path.push(arc.olabel as u8 as char);
            extract_paths(fst, arc.nextstate, path, cost + arc.weight.value(), results, visited);
            path.pop();
        } else {  // Epsilon transition
            extract_paths(fst, arc.nextstate, path, cost + arc.weight.value(), results, visited);
        }
    }

    visited.remove(&state);
}
```

## Running the Example

**Live Demo**

```bash
cargo run --example spell_checking
```

### Sample Output

```
Spell Checking Example
======================

Dictionary contains 41 words

Finding spelling corrections for 'helo' (max edit distance: 2):
--------------------------------------------------
  hello (distance: 1)
  help (distance: 1)
  held (distance: 2)

Finding spelling corrections for 'wrold' (max edit distance: 2):
--------------------------------------------------
  world (distance: 2)

Finding spelling corrections for 'frend' (max edit distance: 2):
--------------------------------------------------
  friend (distance: 1)
  fresh (distance: 2)

Finding spelling corrections for 'chekc' (max edit distance: 2):
--------------------------------------------------
  check (distance: 2)
  
Words within edit distance 1 of 'help':
========================================
  help (distance: 0)
  held (distance: 1)
  helm (distance: 1)
  heap (distance: 1)
```

## Extending the System

**Production-Ready Features**

Beyond basic spell checking, these extensions make the system suitable for real-world deployment.

### Advanced Features

### Weighted Spelling Corrections

Different edit operations can have different costs:

```rust
struct EditCosts {
    insertion: f32,
    deletion: f32,
    substitution: f32,
}

impl EditCosts {
    fn standard() -> Self {
        EditCosts {
            insertion: 1.0,
            deletion: 1.0,
            substitution: 1.0,
        }
    }
    
    fn phonetic_aware() -> Self {
        EditCosts {
            insertion: 1.0,
            deletion: 1.2,        // Slightly more costly
            substitution: 0.8,    // Less costly (common typos)
        }
    }
}
```

### Context-Sensitive Costs

Adjust costs based on character context:

```rust
fn get_substitution_cost(from: char, to: char) -> f32 {
    match (from, to) {
        // Common typos (lower cost)
        ('a', 'e') | ('e', 'a') => 0.5,
        ('i', 'o') | ('o', 'i') => 0.5,
        
        // Keyboard distance-based costs
        _ => keyboard_distance(from, to) * 0.1 + 0.5,
    }
}

fn keyboard_distance(c1: char, c2: char) -> f32 {
    let qwerty_positions = get_qwerty_positions();
    let pos1 = qwerty_positions[&c1];
    let pos2 = qwerty_positions[&c2];
    
    ((pos1.0 - pos2.0).powi(2) + (pos1.1 - pos2.1).powi(2)).sqrt()
}
```

### Frequency-Based Ranking

Incorporate word frequency for better ranking:

```rust
struct FrequencyAwareSpellChecker {
    dictionary_fst: VectorFst<TropicalWeight>,
    word_frequencies: HashMap<String, f32>,
}

impl FrequencyAwareSpellChecker {
    fn find_spelling_corrections_ranked(&self, target: &str, max_distance: usize) -> Vec<SpellingCorrection> {
        let base_corrections = self.find_spelling_corrections(target, max_distance)?;
        
        let mut ranked_corrections: Vec<SpellingCorrection> = base_corrections
            .into_iter()
            .map(|(word, edit_distance)| {
                let frequency_score = self.word_frequencies
                    .get(&word)
                    .map(|f| f.log10())
                    .unwrap_or(-10.0);  // Low score for unknown words
                
                let combined_score = edit_distance - frequency_score * 0.5;
                
                SpellingCorrection {
                    word,
                    edit_distance,
                    frequency_score,
                    combined_score,
                }
            })
            .collect();
        
        ranked_corrections.sort_by(|a, b| a.combined_score.partial_cmp(&b.combined_score).unwrap());
        ranked_corrections
    }
}
```

## Applications

### Real-Time Spell Checking

Implement interactive spell checking:

```rust
struct RealTimeSpellChecker {
    spell_checker: SpellChecker,
    cache: LruCache<String, Vec<String>>,
    min_word_length: usize,
}

impl RealTimeSpellChecker {
    fn check_word(&mut self, word: &str) -> SpellCheckResult {
        if word.len() < self.min_word_length {
            return SpellCheckResult::TooShort;
        }
        
        // Check cache first
        if let Some(cached) = self.cache.get(word) {
            return if cached.is_empty() {
                SpellCheckResult::Correct
            } else {
                SpellCheckResult::Incorrect { suggestions: cached.clone() }
            };
        }
        
        // Check if word is in dictionary
        if self.spell_checker.is_in_dictionary(word) {
            self.cache.put(word.to_string(), vec![]);
            SpellCheckResult::Correct
        } else {
            let suggestions = self.spell_checker.find_spelling_corrections(word, 2);
            self.cache.put(word.to_string(), suggestions.clone());
            SpellCheckResult::Incorrect { suggestions }
        }
    }
}
```

### Document Processing

Process entire documents for spell checking:

```rust
fn spell_check_document(
    text: &str,
    spell_checker: &SpellChecker,
) -> DocumentSpellCheckResult {
    let words = tokenize_text(text);
    let mut corrections = Vec::new();
    let mut statistics = SpellCheckStatistics::new();
    
    for (position, word) in words.iter().enumerate() {
        statistics.total_words += 1;
        
        if !spell_checker.is_in_dictionary(word) {
            statistics.misspelled_words += 1;
            
            let suggestions = spell_checker.find_spelling_corrections(word, 2);
            corrections.push(SpellCheckError {
                position,
                original_word: word.clone(),
                suggestions: suggestions.into_iter().take(5).collect(),
            });
        }
    }
    
    DocumentSpellCheckResult {
        corrections,
        statistics,
    }
}
```

### Search Query Correction

Improve search experience with spell correction:

```rust
fn correct_search_query(
    query: &str,
    search_index: &SearchIndex,
    corrector: &WordCorrector,
) -> CorrectedQuery {
    let words = query.split_whitespace().collect::<Vec<_>>();
    let mut corrected_words = Vec::new();
    let mut any_corrections = false;
    
    for word in words {
        if corrector.is_in_dictionary(word) || search_index.contains_term(word) {
            corrected_words.push(word.to_string());
        } else {
            // Find correction considering search index
            let dictionary_suggestions = corrector.find_corrections(word, 2);
            let search_suggestions = search_index.suggest_similar_terms(word);
            
            let best_suggestion = combine_suggestions(dictionary_suggestions, search_suggestions);
            
            if let Some(suggestion) = best_suggestion {
                corrected_words.push(suggestion);
                any_corrections = true;
            } else {
                corrected_words.push(word.to_string());
            }
        }
    }
    
    CorrectedQuery {
        original: query.to_string(),
        corrected: corrected_words.join(" "),
        has_corrections: any_corrections,
    }
}
```

## Performance Optimization

### Memory-Efficient Dictionary

For large dictionaries, use memory-efficient representations:

```rust
struct CompactDictionary {
    trie: MinimalPerfectHashTrie,
    compressed_suffixes: Vec<u8>,
    state_offsets: Vec<u32>,
}

impl CompactDictionary {
    fn contains(&self, word: &str) -> bool {
        let mut state = 0;
        
        for ch in word.chars() {
            if let Some(next_state) = self.trie.transition(state, ch) {
                state = next_state;
            } else {
                return false;
            }
        }
        
        self.trie.is_final(state)
    }
}
```

### Parallel Processing

Process multiple corrections in parallel:

```rust
use rayon::prelude::*;

fn batch_spell_check(
    words: &[String],
    corrector: &WordCorrector,
) -> Vec<SpellCheckResult> {
    words.par_iter()
        .map(|word| {
            if corrector.is_in_dictionary(word) {
                SpellCheckResult::Correct
            } else {
                let suggestions = corrector.find_corrections(word, 2);
                SpellCheckResult::Incorrect { suggestions }
            }
        })
        .collect()
}
```

### Incremental Correction

For typing applications, use incremental processing:

```rust
struct IncrementalCorrector {
    base_corrector: WordCorrector,
    partial_word: String,
    cached_suggestions: Vec<String>,
}

impl IncrementalCorrector {
    fn add_character(&mut self, ch: char) {
        self.partial_word.push(ch);
        
        // Filter existing suggestions
        self.cached_suggestions.retain(|suggestion| {
            suggestion.starts_with(&self.partial_word) ||
            edit_distance(suggestion, &self.partial_word) <= 2
        });
        
        // Add new suggestions if needed
        if self.cached_suggestions.len() < 5 {
            let new_suggestions = self.base_corrector
                .find_corrections(&self.partial_word, 2);
            self.cached_suggestions.extend(new_suggestions);
            self.cached_suggestions.truncate(10);
        }
    }
    
    fn get_suggestions(&self) -> &[String] {
        &self.cached_suggestions
    }
}
```

## Quality Metrics

### Evaluation Framework

Measure spell checker performance:

```rust
struct SpellCheckerEvaluator {
    test_set: Vec<SpellCheckTestCase>,
}

struct SpellCheckTestCase {
    misspelled_word: String,
    correct_word: String,
    context: Option<String>,
}

impl SpellCheckerEvaluator {
    fn evaluate(&self, corrector: &WordCorrector) -> EvaluationMetrics {
        let mut correct_first = 0;
        let mut correct_in_top_5 = 0;
        let mut total_suggestions = 0;
        
        for test_case in &self.test_set {
            let suggestions = corrector.find_corrections(&test_case.misspelled_word, 3);
            
            if !suggestions.is_empty() {
                total_suggestions += 1;
                
                if suggestions[0].0 == test_case.correct_word {
                    correct_first += 1;
                    correct_in_top_5 += 1;
                } else if suggestions.iter().take(5).any(|(word, _)| word == &test_case.correct_word) {
                    correct_in_top_5 += 1;
                }
            }
        }
        
        EvaluationMetrics {
            accuracy_at_1: correct_first as f32 / total_suggestions as f32,
            accuracy_at_5: correct_in_top_5 as f32 / total_suggestions as f32,
            coverage: total_suggestions as f32 / self.test_set.len() as f32,
        }
    }
}
```

### Error Analysis

Analyze correction system performance:

```rust
fn analyze_correction_errors(
    test_cases: &[SpellCheckTestCase],
    corrector: &WordCorrector,
) -> ErrorAnalysis {
    let mut error_types = HashMap::new();
    let mut missed_corrections = Vec::new();
    
    for test_case in test_cases {
        let suggestions = corrector.find_corrections(&test_case.misspelled_word, 3);
        
        if suggestions.is_empty() {
            error_types.entry("no_suggestions".to_string())
                      .or_insert(0u32)
                      .add_assign(1);
        } else if !suggestions.iter().any(|(word, _)| word == &test_case.correct_word) {
            missed_corrections.push(MissedCorrection {
                input: test_case.misspelled_word.clone(),
                expected: test_case.correct_word.clone(),
                suggestions: suggestions.into_iter().map(|(w, _)| w).collect(),
            });
            
            let error_type = classify_error(&test_case.misspelled_word, &test_case.correct_word);
            error_types.entry(error_type)
                      .or_insert(0u32)
                      .add_assign(1);
        }
    }
    
    ErrorAnalysis {
        error_type_counts: error_types,
        missed_corrections,
    }
}
```

## Advanced Features

### Neural Spell Checking

Integrate neural language models:

```rust
struct NeuralSpellChecker {
    base_corrector: WordCorrector,
    language_model: TransformerModel,
    context_window: usize,
}

impl NeuralSpellChecker {
    fn correct_with_context(
        &self,
        sentence: &str,
        error_position: usize,
    ) -> Vec<ContextualCorrection> {
        let words = sentence.split_whitespace().collect::<Vec<_>>();
        let error_word = words[error_position];
        
        // Get traditional edit distance suggestions
        let base_suggestions = self.base_corrector.find_corrections(error_word, 2);
        
        // Score suggestions using language model
        let mut contextual_corrections = Vec::new();
        for (suggestion, edit_cost) in base_suggestions {
            let context_score = self.score_in_context(&words, error_position, &suggestion);
            
            contextual_corrections.push(ContextualCorrection {
                word: suggestion,
                edit_distance: edit_cost,
                context_score,
                combined_score: edit_cost - context_score * 0.3,
            });
        }
        
        contextual_corrections.sort_by(|a, b| a.combined_score.partial_cmp(&b.combined_score).unwrap());
        contextual_corrections
    }
}
```

### Multilingual Support

Support multiple languages simultaneously:

```rust
struct MultilingualSpellChecker {
    correctors: HashMap<Language, WordCorrector>,
    language_detector: LanguageDetector,
}

impl MultilingualSpellChecker {
    fn correct_multilingual(&self, text: &str) -> MultilingualCorrections {
        let detected_languages = self.language_detector.detect_languages(text);
        let words = tokenize_with_language_hints(text, &detected_languages);
        
        let mut corrections = Vec::new();
        for (word, language_hint) in words {
            if let Some(corrector) = self.correctors.get(&language_hint) {
                if !corrector.is_in_dictionary(&word) {
                    let suggestions = corrector.find_corrections(&word, 2);
                    corrections.push(MultilingualCorrection {
                        word,
                        language: language_hint,
                        suggestions,
                    });
                }
            }
        }
        
        MultilingualCorrections { corrections }
    }
}
```

### Adaptive Learning

Learn from user corrections:

```rust
struct AdaptiveSpellChecker {
    base_corrector: WordCorrector,
    user_dictionary: HashSet<String>,
    correction_feedback: HashMap<String, String>,
    usage_frequency: HashMap<String, u32>,
}

impl AdaptiveSpellChecker {
    fn learn_from_correction(&mut self, misspelled: &str, corrected: &str) {
        self.correction_feedback.insert(misspelled.to_string(), corrected.to_string());
        *self.usage_frequency.entry(corrected.to_string()).or_insert(0) += 1;
    }
    
    fn add_to_user_dictionary(&mut self, word: &str) {
        self.user_dictionary.insert(word.to_string());
    }
    
    fn find_corrections_adaptive(&self, word: &str) -> Vec<String> {
        // Check user feedback first
        if let Some(correction) = self.correction_feedback.get(word) {
            return vec![correction.clone()];
        }
        
        // Check user dictionary
        if self.user_dictionary.contains(word) {
            return vec![];  // Word is correct
        }
        
        // Use base corrector with frequency weighting
        let base_suggestions = self.base_corrector.find_corrections(word, 2);
        let mut weighted_suggestions: Vec<(String, f32)> = base_suggestions
            .into_iter()
            .map(|(suggestion, edit_cost)| {
                let frequency_bonus = self.usage_frequency
                    .get(&suggestion)
                    .map(|f| (*f as f32).log10())
                    .unwrap_or(0.0);
                
                (suggestion, edit_cost - frequency_bonus * 0.2)
            })
            .collect();
        
        weighted_suggestions.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        weighted_suggestions.into_iter().map(|(word, _)| word).collect()
    }
}
```

### Try These Extensions

- **Context-aware correction**: Use surrounding words to improve suggestions
- **Domain-specific dictionaries**: Medical, legal, or technical vocabularies
- **Real-time learning**: Adapt to user vocabulary and patterns
- **Multi-language support**: Handle code-switching and foreign words

### Related Examples

This spell checking example connects with other examples in the collection. **[Edit Distance](edit_distance.md)** provides the foundation for approximate string matching. **[Morphological Analyzer](../linguistic-applications/morphological_analyzer.md)** handles morphological variants. **[Pronunciation Lexicon](../practical-applications/pronunciation_lexicon.md)** provides phonetic similarity for corrections. **[Transliteration](../practical-applications/transliteration.md)** enables cross-script spelling correction.

### Industrial Use Cases

### Text Editors and IDEs

Modern text editors use sophisticated spell checking:

```rust
struct EditorSpellChecker {
    corrector: WordCorrector,
    language_specific_correctors: HashMap<ProgrammingLanguage, WordCorrector>,
    comment_extractor: CommentExtractor,
}

impl EditorSpellChecker {
    fn check_source_code(&self, code: &str, language: ProgrammingLanguage) -> Vec<SpellingError> {
        let comments = self.comment_extractor.extract_comments(code, language);
        let string_literals = self.comment_extractor.extract_string_literals(code, language);
        
        let mut errors = Vec::new();
        
        // Check comments
        for comment in comments {
            errors.extend(self.check_text(&comment.text, comment.position));
        }
        
        // Check string literals (if enabled)
        for literal in string_literals {
            errors.extend(self.check_text(&literal.text, literal.position));
        }
        
        errors
    }
}
```

### Email and Messaging

Email clients integrate spell checking:

```rust
struct EmailSpellChecker {
    corrector: WordCorrector,
    auto_correct_enabled: bool,
    ignore_proper_nouns: bool,
    personal_dictionary: HashSet<String>,
}

impl EmailSpellChecker {
    fn check_email(&self, email: &Email) -> EmailSpellCheckResult {
        let mut errors = Vec::new();
        
        // Check subject line
        errors.extend(self.check_text(&email.subject, TextLocation::Subject));
        
        // Check body
        errors.extend(self.check_text(&email.body, TextLocation::Body));
        
        // Apply auto-correction if enabled
        let corrected_email = if self.auto_correct_enabled {
            self.apply_auto_corrections(email, &errors)
        } else {
            email.clone()
        };
        
        EmailSpellCheckResult {
            original_email: email.clone(),
            corrected_email,
            spelling_errors: errors,
        }
    }
}
```

---

