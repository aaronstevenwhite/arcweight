# Pronunciation Lexicon

This example implements a grapheme-to-phoneme conversion system using finite state transducers for speech synthesis and recognition applications.

## Overview

Grapheme-to-phoneme (G2P) conversion maps orthographic representations to phonetic transcriptions. English orthography presents particular challenges due to irregular spelling-sound correspondences (e.g., "-ough" in "though" /ðoʊ/, "through" /θru/, "tough" /tʌf/, "cough" /kɔf/).

Finite state transducers address these challenges through:
- Efficient lexicon lookup for known words
- Multiple pronunciation handling (e.g., "read" /rid/ vs. /rɛd/)
- Rule-based grapheme-to-phoneme conversion for out-of-vocabulary items
- Context-dependent pronunciation modeling

The implementation provides bidirectional transduction supporting both text-to-speech synthesis and automatic speech recognition.

## Usage

```bash
cargo run --example pronunciation_lexicon
```

## Concepts Demonstrated

- **Phoneme Inventory**: ARPAbet-based phoneme representation
- **Lexicon Architecture**: FST-based pronunciation dictionary
- **Pronunciation Variants**: Homograph disambiguation
- **G2P Conversion**: Rule-based grapheme-to-phoneme mapping
- **Speech System Integration**: TTS and ASR compatibility
- **Optimization Strategies**: Efficient storage and retrieval

## Core Concepts

### Phoneme Representation

The system employs ARPAbet notation for English phonemes:

**Vowels:**
```rust,ignore
AA, AE, AH, AO, AW, AY,  // /ɑ/, /æ/, /ʌ/, /ɔ/, /aʊ/, /aɪ/
EH, ER, EY, IH, IY,      // /ɛ/, /ɝ/, /eɪ/, /ɪ/, /i/
OW, OY, UH, UW           // /oʊ/, /ɔɪ/, /ʊ/, /u/
```

**Consonants:**
```rust,ignore
B, CH, D, DH, F, G, HH,  // /b/, /tʃ/, /d/, /ð/, /f/, /g/, /h/
JH, K, L, M, N, NG,      // /dʒ/, /k/, /l/, /m/, /n/, /ŋ/
P, R, S, SH, T, TH,      // /p/, /r/, /s/, /ʃ/, /t/, /θ/
V, W, Y, Z, ZH           // /v/, /w/, /j/, /z/, /ʒ/
```

**Boundary Markers:**
```rust,ignore
Sil  // Silence/pause marker
```

### Label Encoding

Phoneme labels are offset to prevent collision with character codes:

```rust,ignore
impl Phoneme {
    fn to_label(self) -> u32 {
        (self as u32) + 1000  // Offset prevents character collision
    }
    
    fn from_label(label: u32) -> Option<Self> {
        if (1000..1100).contains(&label) {
            // Decode phoneme from label
        }
    }
}
```

## Lexicon Architecture

### Entry Structure

Lexicon entries accommodate pronunciation variants:

```rust,ignore
struct LexiconEntry {
    word: String,
    pronunciations: Vec<Vec<Phoneme>>,
}
```

Example:
```rust,ignore
LexiconEntry {
    word: "read".to_string(),
    pronunciations: vec![
        vec![R, IY, D], // present: /rid/
        vec![R, EH, D], // past: /rɛd/
    ],
}
```

### Lexicon Contents

The implementation includes representative entries:

```rust,ignore
fn create_sample_lexicon() -> Vec<LexiconEntry> {
    vec![
        // Basic words
        LexiconEntry {
            word: "hello".to_string(),
            pronunciations: vec![
                vec![HH, AH, L, OW],     // /həˈloʊ/
                vec![HH, EH, L, OW],     // variant /hɛˈloʊ/
            ],
        },
        
        // Homophones
        LexiconEntry {
            word: "read".to_string(),
            pronunciations: vec![
                vec![R, IY, D], // present
                vec![R, EH, D], // past  
            ],
        },
        
        // Function words with variants
        LexiconEntry {
            word: "the".to_string(),
            pronunciations: vec![
                vec![DH, AH], // unstressed /ðə/
                vec![DH, IY], // stressed /ðiː/
            ],
        },
    ]
}
```

## FST Implementation

### Simple Lexicon FST

A basic approach directly encodes word-to-phoneme mappings:

```rust,ignore
fn build_simple_lexicon(entries: &[LexiconEntry]) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    fst.set_start(start);

    for entry in entries {
        for pronunciation in &entry.pronunciations {
            let mut states = vec![start];
            
            // Create states for each character
            for _ in 0..entry.word.len() {
                states.push(fst.add_state());
            }
            
            // Add character transitions
            for (i, ch) in entry.word.chars().enumerate() {
                fst.add_arc(states[i], Arc::new(
                    ch as u32, ch as u32,
                    TropicalWeight::one(),
                    states[i + 1]
                ));
            }
            
            // Output phoneme sequence
            let mut current = states[entry.word.len()];
            for phoneme in pronunciation {
                let next = fst.add_state();
                fst.add_arc(current, Arc::new(
                    0, phoneme.to_label(),  // epsilon -> phoneme
                    TropicalWeight::one(), next
                ));
                current = next;
            }
            
            fst.set_final(current, TropicalWeight::one());
        }
    }
    
    fst
}
```

### Word Acceptor

Helper function to create FSTs for individual words:

```rust,ignore
fn word_acceptor(word: &str) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let mut current = fst.add_state();
    fst.set_start(current);

    for ch in word.chars() {
        let next = fst.add_state();
        fst.add_arc(current, Arc::new(
            ch as u32, ch as u32,
            TropicalWeight::one(), next
        ));
        current = next;
    }

    fst.set_final(current, TropicalWeight::one());
    fst
}
```

## Grapheme-to-Phoneme Conversion

### Rule-Based G2P

Out-of-vocabulary words are processed using grapheme-to-phoneme rules:

```rust,ignore
fn build_g2p_rules() -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    fst.set_start(start);
    fst.set_final(start, TropicalWeight::one());

    let rules = vec![
        // Vowel mappings
        ('a', Phoneme::AE),
        ('e', Phoneme::EH),
        ('i', Phoneme::IH),
        ('o', Phoneme::AO),
        ('u', Phoneme::AH),
        
        // Consonant mappings
        ('b', Phoneme::B),   ('c', Phoneme::K),
        ('d', Phoneme::D),   ('f', Phoneme::F),
        ('g', Phoneme::G),   ('h', Phoneme::HH),
    ];

    for (grapheme, phoneme) in rules {
        fst.add_arc(start, Arc::new(
            grapheme as u32,
            phoneme.to_label(),
            TropicalWeight::new(1.0),  // Rule weight
            start,
        ));
    }

    fst
}
```

### Phoneme Extraction

Recover phoneme sequences from composed transducers:

```rust,ignore
fn extract_g2p_phonemes(
    fst: &VectorFst<TropicalWeight>, 
    word: &str
) -> Vec<Phoneme> {
    let word_fst = word_acceptor(word);
    let composed = compose_default(&word_fst, fst)?;
    
    if let Some(start) = composed.start() {
        let mut phonemes = Vec::new();
        extract_phonemes_from_state(&composed, start, &mut phonemes);
        return phonemes;
    }
    
    Vec::new()
}
```

## Text Processing Pipeline

### Integrated System

Complete grapheme-to-phoneme conversion pipeline:

```rust,ignore
fn text_to_phonemes(
    lexicon_entries: &[LexiconEntry],
    g2p_fst: &VectorFst<TropicalWeight>,
    text: &str,
) -> Vec<Phoneme> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut result_phonemes = Vec::new();

    for (i, word) in words.iter().enumerate() {
        // Lexicon lookup with G2P fallback
        if let Some(pronunciations) = lookup_word_in_lexicon(lexicon_entries, word) {
            result_phonemes.extend(pronunciations[0].clone());
        } else {
            let g2p_phonemes = extract_g2p_phonemes(g2p_fst, word);
            result_phonemes.extend(g2p_phonemes);
        }

        // Inter-word silence
        if i != words.len() - 1 {
            result_phonemes.push(Phoneme::Sil);
        }
    }

    result_phonemes
}
```

### Lexicon Lookup

Direct dictionary lookup for efficiency:

```rust,ignore
fn lookup_word_in_lexicon(
    entries: &[LexiconEntry], 
    word: &str
) -> Option<Vec<Vec<Phoneme>>> {
    for entry in entries {
        if entry.word == word {
            return Some(entry.pronunciations.clone());
        }
    }
    None
}
```

## Execution

```bash
cargo run --example pronunciation_lexicon
```

### Sample Output

```text
Word to Phoneme Lookup:
--------------------------
Looking up 'hello':
  Pronunciation 1: HH AH L OW
  Pronunciation 2: HH EH L OW

Looking up 'read':
  Pronunciation 1: R IY D (present tense)
  Pronunciation 2: R EH D (past tense)

Looking up 'the':
  Pronunciation 1: DH AH (unstressed /ðə/)
  Pronunciation 2: DH IY (stressed /ðiː/)

Grapheme-to-Phoneme (G2P) Rules:
-----------------------------------
Applying G2P rules to 'test':
  G2P result: T EH S T

Applying G2P rules to 'book':
  G2P result: B UH K

Text-to-Phoneme Conversion:
------------------------------
Text: "hello world"
Phonemes: HH AH L OW SIL W ER L D

Text: "read the book"  
Phonemes: R IY D SIL DH AH SIL B UH K
```

## Advanced Features

### Pronunciation Variation

The system accommodates multiple pronunciation types:

**Homographs**: Orthographically identical forms with distinct pronunciations (e.g., "read" /rid/ vs. /rɛd/)

**Prosodic Variants**: Stress-conditioned alternations (e.g., "the" /ðə/ vs. /ði/)

**Dialectal Variation**: Regional pronunciation differences (e.g., "tomato" /təˈmeɪtoʊ/ vs. /təˈmɑːtoʊ/)

### Pronunciation Quality Assessment

Different pronunciation sources have different reliability:

```rust,ignore
enum PronunciationSource {
    Dictionary(f32),    // High confidence
    G2P(f32),          // Medium confidence  
    Analogical(f32),   // Lower confidence
}

struct PronunciationResult {
    phonemes: Vec<Phoneme>,
    source: PronunciationSource,
    confidence: f32,
}
```

### Contextual Pronunciation

Context-dependent pronunciation modeling:

```rust,ignore
fn contextual_pronunciation(
    word: &str,
    prev_word: Option<&str>,
    next_word: Option<&str>,
    style: SpeechStyle,
) -> Vec<Phoneme> {
    match style {
        SpeechStyle::Careful => dictionary_pronunciation(word),
        SpeechStyle::Casual => apply_reduction_rules(word),
        SpeechStyle::Fast => apply_heavy_reduction(word),
    }
}
```

## Applications

### Automatic Speech Recognition

Pronunciation lexicons mediate between acoustic and linguistic representations:

```text
Audio Signal → Acoustic Features → Phoneme Sequence → Word Sequence
```

**Implementation:**
```rust,ignore
fn speech_recognition_pipeline(
    audio: &[f32],
    acoustic_model: &AcousticModel,
    lexicon: &PronunciationLexicon,
    language_model: &LanguageModel,
) -> Vec<String> {
    let phonemes = acoustic_model.decode(audio);
    let word_candidates = lexicon.phonemes_to_words(&phonemes);
    let best_words = language_model.score_candidates(word_candidates);
    best_words
}
```

### Text-to-Speech Synthesis

Grapheme-to-phoneme conversion for speech synthesis:

```text
Text → Words → Phonemes → Acoustic Parameters → Audio Signal
```

**Implementation:**
```rust,ignore
fn text_to_speech_pipeline(
    text: &str,
    lexicon: &PronunciationLexicon,
    vocoder: &NeuralVocoder,
) -> Vec<f32> {
    let phonemes = lexicon.text_to_phonemes(text);
    let acoustic_features = vocoder.phonemes_to_features(&phonemes);
    let audio = vocoder.features_to_audio(acoustic_features);
    audio
}
```

### Phonetic Analysis

Study pronunciation patterns in speech corpora:

```rust,ignore
fn analyze_pronunciation_variation(
    corpus: &SpeechCorpus,
    lexicon: &PronunciationLexicon,
) -> VariationReport {
    let mut variations = HashMap::new();
    
    for utterance in corpus.utterances() {
        let expected = lexicon.lookup(&utterance.word);
        let actual = utterance.phonetic_transcription;
        
        if expected != actual {
            variations.entry(utterance.word)
                     .or_insert_with(Vec::new)
                     .push((expected, actual));
        }
    }
    
    VariationReport::new(variations)
}
```

## Speech System Architecture

### Cascaded Transducer Model

Speech systems employ composed transducer hierarchies:

```rust,ignore
// L ∘ G ∘ P ∘ A architecture
let lexicon_fst = build_lexicon_fst(&dictionary);      // L: Lexical transducer
let grammar_fst = build_grammar_fst(&language_model);  // G: Grammar transducer  
let phoneme_fst = build_phoneme_fst(&phoneme_set);     // P: Phonemic constraints
let acoustic_fst = build_acoustic_fst(&acoustic_model); // A: Acoustic model

let complete_system = compose_chain(vec![
    lexicon_fst,
    grammar_fst, 
    phoneme_fst,
    acoustic_fst,
])?;
```

### Search Space Management

FST composition creates large search spaces that require efficient navigation:

```rust,ignore
fn beam_search_decode(
    fst: &VectorFst<TropicalWeight>,
    observations: &[Observation],
    beam_width: usize,
) -> Vec<DecodingPath> {
    let mut beam = vec![DecodingState::initial()];
    
    for observation in observations {
        let mut next_beam = Vec::new();
        
        for state in beam {
            let successors = fst.expand_state(&state, observation);
            next_beam.extend(successors);
        }
        
        // Keep only top beam_width hypotheses
        next_beam.sort_by_key(|state| state.cost);
        next_beam.truncate(beam_width);
        beam = next_beam;
    }
    
    beam.into_iter()
        .filter(|state| fst.is_final(state.fst_state))
        .map(|state| state.path)
        .collect()
}
```

## Performance Considerations

### Memory Optimization

Large lexicons require careful memory management:

```rust,ignore
struct CompactLexicon {
    trie: TrieNode,              // Shared prefix storage
    phoneme_sequences: Vec<Vec<Phoneme>>, // Deduplicated sequences
    word_to_sequence: HashMap<String, usize>, // Index mapping
}

impl CompactLexicon {
    fn lookup(&self, word: &str) -> Option<&[Phoneme]> {
        self.word_to_sequence.get(word)
            .map(|&idx| &self.phoneme_sequences[idx])
    }
}
```

### Lookup Optimization

Fast dictionary access for real-time applications:

```rust,ignore
struct FastLexicon {
    hash_table: HashMap<String, Vec<Vec<Phoneme>>>,
    prefix_tree: Trie<Vec<Phoneme>>,
    bloom_filter: BloomFilter,  // Fast negative lookup
}

impl FastLexicon {
    fn lookup(&self, word: &str) -> Option<&[Vec<Phoneme>]> {
        // Quick negative check
        if !self.bloom_filter.might_contain(word) {
            return None;
        }
        
        // Hash table lookup
        self.hash_table.get(word)
    }
}
```

### Caching Strategies

Cache frequent lookups and G2P results:

```rust,ignore
struct CachedLexicon {
    lexicon: PronunciationLexicon,
    g2p_cache: LruCache<String, Vec<Phoneme>>,
    lookup_cache: LruCache<String, Vec<Vec<Phoneme>>>,
}

impl CachedLexicon {
    fn lookup_with_cache(&mut self, word: &str) -> Vec<Vec<Phoneme>> {
        if let Some(cached) = self.lookup_cache.get(word) {
            return cached.clone();
        }
        
        let result = self.lexicon.lookup(word);
        self.lookup_cache.put(word.to_string(), result.clone());
        result
    }
}
```

## Quality Assurance

### Pronunciation Validation

Ensure lexicon quality through validation:

```rust,ignore
fn validate_lexicon(lexicon: &[LexiconEntry]) -> ValidationReport {
    let mut errors = Vec::new();
    
    for entry in lexicon {
        // Check phoneme validity
        for pronunciation in &entry.pronunciations {
            for phoneme in pronunciation {
                if !is_valid_phoneme(*phoneme) {
                    errors.push(ValidationError::InvalidPhoneme {
                        word: entry.word.clone(),
                        phoneme: *phoneme,
                    });
                }
            }
        }
        
        // Check pronunciation consistency
        if entry.pronunciations.is_empty() {
            errors.push(ValidationError::NoPronunciation {
                word: entry.word.clone(),
            });
        }
    }
    
    ValidationReport::new(errors)
}
```

### Coverage Analysis

Assess lexicon coverage for target domain:

```rust,ignore
fn analyze_coverage(
    lexicon: &PronunciationLexicon,
    corpus: &TextCorpus,
) -> CoverageReport {
    let mut covered = 0;
    let mut total = 0;
    let mut missing_words = HashSet::new();
    
    for word in corpus.words() {
        total += 1;
        if lexicon.contains(word) {
            covered += 1;
        } else {
            missing_words.insert(word.to_string());
        }
    }
    
    CoverageReport {
        coverage_rate: covered as f32 / total as f32,
        missing_words: missing_words.into_iter().collect(),
        total_words: total,
        covered_words: covered,
    }
}
```

## Extensions and Future Work

### Enhanced G2P Models

**Neural G2P:**
```rust,ignore
struct NeuralG2P {
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
    attention: MultiHeadAttention,
}

impl NeuralG2P {
    fn generate_pronunciation(&self, word: &str) -> Vec<Phoneme> {
        let char_embeddings = self.encoder.encode(word);
        let phoneme_sequence = self.decoder.decode(char_embeddings);
        phoneme_sequence
    }
}
```

**Contextual G2P:**
```rust,ignore
fn contextual_g2p(
    word: &str,
    context: &SentenceContext,
    style: PronunciationStyle,
) -> Vec<Phoneme> {
    match style {
        PronunciationStyle::Formal => apply_careful_g2p(word),
        PronunciationStyle::Casual => apply_reduction_rules(word, context),
        PronunciationStyle::Regional(dialect) => apply_dialect_rules(word, dialect),
    }
}
```

### Multilingual Support

**Cross-Language Lexicons:**
```rust,ignore
struct MultilingualLexicon {
    lexicons: HashMap<Language, PronunciationLexicon>,
    language_detector: LanguageDetector,
    code_switching_handler: CodeSwitchingHandler,
}

impl MultilingualLexicon {
    fn lookup_multilingual(&self, word: &str, context: &str) -> Vec<Phoneme> {
        let language = self.language_detector.detect(word, context);
        let lexicon = &self.lexicons[&language];
        
        if let Some(pronunciation) = lexicon.lookup(word) {
            pronunciation[0].clone()
        } else {
            self.handle_unknown_word(word, language)
        }
    }
}
```

### Dynamic Adaptation

**Online Learning:**
```rust,ignore
struct AdaptiveLexicon {
    base_lexicon: PronunciationLexicon,
    user_corrections: HashMap<String, Vec<Phoneme>>,
    confidence_scores: HashMap<String, f32>,
}

impl AdaptiveLexicon {
    fn update_pronunciation(
        &mut self,
        word: &str,
        correct_pronunciation: Vec<Phoneme>,
        confidence: f32,
    ) {
        self.user_corrections.insert(word.to_string(), correct_pronunciation);
        self.confidence_scores.insert(word.to_string(), confidence);
    }
    
    fn lookup_adaptive(&self, word: &str) -> Vec<Phoneme> {
        // Prefer user corrections over base lexicon
        if let Some(user_pron) = self.user_corrections.get(word) {
            user_pron.clone()
        } else {
            self.base_lexicon.lookup(word)
                .map(|prns| prns[0].clone())
                .unwrap_or_else(|| self.g2p_fallback(word))
        }
    }
}
```

## Related Examples

- **[Phonological Rules](../linguistic-applications/phonological_rules.md)**: Systematic pronunciation variation
- **[Morphological Analyzer](../linguistic-applications/morphological_analyzer.md)**: Morphophonological alternations
- **[Spell Checking](../text-processing/spell_checking.md)**: Phonetically informed error correction
- **[Transliteration](transliteration.md)**: Cross-script phonetic mapping

## Practical Applications

### Voice Assistant Systems

Pronunciation lexicons enable bidirectional speech processing in conversational AI systems.

### Language Learning

Pronunciation feedback systems:

```rust,ignore
fn pronunciation_feedback(
    student_audio: &[f32],
    target_word: &str,
    lexicon: &PronunciationLexicon,
) -> FeedbackReport {
    let student_phonemes = extract_phonemes(student_audio);
    let target_phonemes = lexicon.lookup(target_word);
    
    let alignment = align_phoneme_sequences(&student_phonemes, &target_phonemes);
    let errors = identify_pronunciation_errors(&alignment);
    
    FeedbackReport {
        overall_score: calculate_pronunciation_score(&alignment),
        phoneme_errors: errors,
        suggestions: generate_practice_suggestions(&errors),
    }
}
```

### Accessibility Technology

Screen readers and assistive devices:

```rust,ignore
fn screen_reader_pronunciation(
    text: &str,
    user_preferences: &AccessibilityPreferences,
    lexicon: &PronunciationLexicon,
) -> AudioOutput {
    let phonemes = lexicon.text_to_phonemes(text);
    let adjusted_phonemes = apply_accessibility_adjustments(
        phonemes, 
        user_preferences
    );
    synthesize_speech(adjusted_phonemes)
}
```

## Implementation Considerations

### Memory Optimization

Large-scale pronunciation lexicons require efficient memory management:

```rust,ignore
struct MemoryEfficientLexicon {
    // Use a trie for prefix sharing
    trie: PronunciationTrie,
    // Store phoneme sequences once, reference by index
    phoneme_pool: Vec<Vec<Phoneme>>,
    // Map words to phoneme pool indices
    word_to_phoneme_indices: HashMap<String, Vec<usize>>,
}

impl MemoryEfficientLexicon {
    fn add_word(&mut self, word: &str, pronunciations: Vec<Vec<Phoneme>>) {
        let indices: Vec<usize> = pronunciations.into_iter()
            .map(|pron| {
                // Check if this pronunciation already exists
                if let Some(idx) = self.find_phoneme_sequence(&pron) {
                    idx
                } else {
                    // Add new pronunciation to pool
                    self.phoneme_pool.push(pron);
                    self.phoneme_pool.len() - 1
                }
            })
            .collect();
        
        self.word_to_phoneme_indices.insert(word.to_string(), indices);
    }
}
```

### Handling Pronunciation Conflicts

When multiple sources provide different pronunciations:

```rust,ignore
struct PronunciationConflictResolver {
    priority_sources: Vec<PronunciationSource>,
    voting_threshold: f32,
}

impl PronunciationConflictResolver {
    fn resolve_conflicts(
        &self,
        word: &str,
        candidates: Vec<(Vec<Phoneme>, PronunciationSource, f32)>,
    ) -> Vec<Vec<Phoneme>> {
        // Group by pronunciation
        let mut pronunciation_votes: HashMap<Vec<Phoneme>, f32> = HashMap::new();
        
        for (pron, source, confidence) in candidates {
            let weight = self.get_source_weight(&source) * confidence;
            *pronunciation_votes.entry(pron).or_insert(0.0) += weight;
        }
        
        // Select pronunciations above threshold
        let mut selected: Vec<(Vec<Phoneme>, f32)> = pronunciation_votes
            .into_iter()
            .filter(|(_, votes)| *votes >= self.voting_threshold)
            .collect();
        
        // Sort by votes descending
        selected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        selected.into_iter().map(|(pron, _)| pron).collect()
    }
}
```

### G2P Model Training

Training custom G2P models for better coverage:

```rust,ignore
struct G2PTrainer {
    alignment_model: AlignmentModel,
    sequence_model: SequenceModel,
}

impl G2PTrainer {
    fn train_from_lexicon(
        &mut self,
        training_data: &[(String, Vec<Phoneme>)],
        validation_data: &[(String, Vec<Phoneme>)],
    ) -> TrainingResult {
        // Step 1: Learn character-phoneme alignments
        let alignments = self.alignment_model.learn_alignments(training_data);
        
        // Step 2: Extract alignment patterns
        let patterns = extract_g2p_patterns(&alignments);
        
        // Step 3: Train sequence model
        self.sequence_model.train(&patterns);
        
        // Step 4: Evaluate on validation set
        let accuracy = self.evaluate(validation_data);
        
        TrainingResult {
            patterns_learned: patterns.len(),
            validation_accuracy: accuracy,
        }
    }
}
```

## Best Practices

### Quality Assurance

Systematic pronunciation validation:

```rust,ignore
fn validate_pronunciation_quality(
    lexicon: &PronunciationLexicon,
    test_words: &[(String, Vec<Phoneme>)],
) -> QualityReport {
    let mut errors = Vec::new();
    
    for (word, expected) in test_words {
        if let Some(actual) = lexicon.lookup(word) {
            if !actual.contains(expected) {
                errors.push(PronunciationError::Mismatch {
                    word: word.clone(),
                    expected: expected.clone(),
                    actual: actual.clone(),
                });
            }
        } else {
            errors.push(PronunciationError::Missing {
                word: word.clone(),
            });
        }
    }
    
    QualityReport {
        total_tested: test_words.len(),
        errors,
        accuracy: 1.0 - (errors.len() as f32 / test_words.len() as f32),
    }
}
```

### 2. Handling Pronunciation Variants

Design for multiple valid pronunciations from the start:

```rust,ignore
struct VariantAwareLexicon {
    // Primary pronunciations (most common)
    primary: HashMap<String, Vec<Phoneme>>,
    // All variants including regional/style differences
    variants: HashMap<String, Vec<PronunciationVariant>>,
}

struct PronunciationVariant {
    phonemes: Vec<Phoneme>,
    variant_type: VariantType,
    frequency: f32,
    regions: Vec<Region>,
}

enum VariantType {
    Regional,       // Different by geography
    Stylistic,      // Formal vs casual
    Historical,     // Older pronunciations
    Foreign,        // Non-native adaptations
}
```

### 3. Efficient Updates

Support incremental lexicon updates:

```rust,ignore
struct UpdateableLexicon {
    base_lexicon: PronunciationLexicon,
    updates: Vec<LexiconUpdate>,
    update_index: HashMap<String, usize>,
}

impl UpdateableLexicon {
    fn apply_update(&mut self, update: LexiconUpdate) {
        match update.update_type {
            UpdateType::Add => {
                self.updates.push(update.clone());
                self.update_index.insert(update.word.clone(), self.updates.len() - 1);
            }
            UpdateType::Modify => {
                if let Some(&idx) = self.update_index.get(&update.word) {
                    self.updates[idx] = update;
                }
            }
            UpdateType::Remove => {
                self.update_index.remove(&update.word);
            }
        }
    }
    
    fn compile_updates(&mut self) {
        // Periodically merge updates into base lexicon
        let updates_to_apply = std::mem::take(&mut self.updates);
        self.base_lexicon.apply_batch_updates(updates_to_apply);
        self.update_index.clear();
    }
}
```

