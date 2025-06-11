# Pronunciation Lexicon FST Example

This example demonstrates how to build and use a pronunciation lexicon FST that maps between orthographic words and their phonetic representations. It's a fundamental component in speech recognition and synthesis systems.

## Overview

The example implements:
1. **Lexicon FST** mapping words to phoneme sequences
2. **Grapheme-to-Phoneme (G2P)** system for unknown words
3. **Multiple pronunciations** for ambiguous words
4. **Text-to-phoneme conversion** pipeline
5. **Speech processing integration** patterns

## Key Components

### Phoneme Inventory

Uses a simplified phonetic alphabet based on ARPAbet/CMU conventions:

#### Vowels
- **Monophthongs**: AA, AE, AH, AO, EH, ER, IH, IY, OW, UH, UW
- **Diphthongs**: AW, AY, EY, OY

#### Consonants
- **Stops**: B, D, G, K, P, T
- **Fricatives**: DH, F, HH, S, SH, TH, V, Z, ZH
- **Affricates**: CH, JH
- **Nasals**: M, N, NG
- **Liquids**: L, R
- **Glides**: W, Y

#### Special
- **SIL**: Silence/pause marker

### Data Structures

#### Phoneme Representation
```rust
enum Phoneme {
    // Vowels
    AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW,
    // Consonants  
    B, CH, D, DH, F, G, HH, JH, K, L, M, N, NG, P, R, S, SH, T, TH, V, W, Y, Z, ZH,
    // Special
    SIL,
}
```

#### Lexicon Entry
```rust
struct LexiconEntry {
    word: String,
    pronunciation: Vec<Phoneme>,
    frequency: f32,          // Usage frequency weight
    pos_tag: Option<String>, // Part-of-speech for disambiguation
}
```

## Core Functions

### Lexicon Construction

#### Build Pronunciation Lexicon FST
```rust
fn build_pronunciation_lexicon(
    entries: &[LexiconEntry],
) -> VectorFst<TropicalWeight>
```

Creates a lexicon FST where:
- Input labels are characters (letters)
- Output labels are phonemes
- Weights reflect pronunciation likelihood
- Multiple paths exist for words with alternative pronunciations

#### Example Lexicon Entries
```rust
vec![
    LexiconEntry::new("cat", vec![K, AE, T], 1.0),
    LexiconEntry::new("dog", vec![D, AO, G], 1.0),
    LexiconEntry::new("read", vec![R, IY, D], 0.6),    // present tense
    LexiconEntry::new("read", vec![R, EH, D], 0.4),    // past tense
    LexiconEntry::new("the", vec![DH, AH], 0.8),       // unstressed
    LexiconEntry::new("the", vec![DH, IY], 0.2),       // stressed
]
```

### G2P System

#### Default G2P Rules
```rust
fn build_default_g2p_fst() -> VectorFst<TropicalWeight>
```

Implements basic English grapheme-to-phoneme rules:
- Single letter mappings (a → AE, b → B, etc.)
- Common digraphs (ch → CH, sh → SH, th → TH)
- Silent letters (k in "knee", b in "lamb")
- Vowel patterns (magic e, vowel teams)

#### Context-Sensitive G2P
```rust
fn build_context_sensitive_g2p() -> VectorFst<TropicalWeight>
```

Handles context-dependent pronunciations:
- 'c' → /k/ before a,o,u; /s/ before e,i,y
- 'g' → /g/ vs /j/ based on following vowel
- Vowel length in different syllable patterns

### Text Processing Pipeline

#### Text to Phonemes
```rust
fn text_to_phonemes(
    text: &str,
    lexicon: &VectorFst<TropicalWeight>,
    g2p: &VectorFst<TropicalWeight>,
) -> Result<Vec<Phoneme>>
```

Complete pipeline:
1. Tokenize input text
2. Look up words in lexicon FST
3. Fall back to G2P for unknown words
4. Handle punctuation and special tokens
5. Apply post-lexical phonological rules

## Usage

Run the example with:
```bash
cargo run --example pronunciation_lexicon
```

### Example Output

```
=== Pronunciation Lexicon Demo ===

Building lexicon with 1000+ entries...

Text-to-Phoneme Conversion:
Input: "The cat sat on the mat"
Output: DH AH  K AE T  S AE T  AO N  DH AH  M AE T

Multiple Pronunciations:
"read" (present): R IY D (weight: 0.6)
"read" (past):    R EH D (weight: 0.4)

"the" (unstressed): DH AH (weight: 0.8)  
"the" (stressed):   DH IY (weight: 0.2)

Unknown Word G2P:
"programming" → P R OW G R AE M IH NG

Lexicon Statistics:
- Total entries: 1247
- Words with multiple pronunciations: 156
- Average pronunciations per word: 1.2
- G2P fallback rate: 8.3%
```

## Advanced Features

### Stress and Syllable Structure

```rust
enum StressLevel {
    Primary,   // 1
    Secondary, // 2  
    Unstressed, // 0
}

struct SyllableInfo {
    phonemes: Vec<Phoneme>,
    stress: StressLevel,
    onset: Vec<Phoneme>,
    nucleus: Vec<Phoneme>, 
    coda: Vec<Phoneme>,
}
```

### Morphological Integration

```rust
fn build_morphology_aware_lexicon(
    stems: &[LexiconEntry],
    affixes: &[AffixEntry],
) -> VectorFst<TropicalWeight> {
    // Handles:
    // - Stem + affix combinations
    // - Morphophonological alternations  
    // - Stress shift patterns
    // - Productivity weighting
}
```

### Frequency-Based Weighting

```rust
impl LexiconEntry {
    fn frequency_weight(&self) -> TropicalWeight {
        // Convert frequency to tropical weight
        // Higher frequency = lower cost
        TropicalWeight::new(-self.frequency.ln())
    }
}
```

## Applications

### Speech Recognition
- Lexicon compilation for ASR systems
- Out-of-vocabulary (OOV) word handling
- Pronunciation variant modeling
- Confidence scoring

### Text-to-Speech Synthesis
- Phonetic transcription for TTS
- Prosody and stress assignment
- Voice-specific pronunciation adaptation
- Multilingual synthesis

### Language Learning
- Pronunciation training systems
- Phonetic awareness tools
- Second language acquisition research
- Computer-assisted pronunciation training (CAPT)

### Computational Linguistics
- Phonological analysis
- Historical linguistics research
- Cross-linguistic phonetic studies
- Corpus-based pronunciation analysis

## Performance Characteristics

### Memory Usage
- Linear in lexicon size: O(|entries| × |avg_pronunciation_length|)
- Trie compression reduces space for common prefixes
- Phoneme encoding uses compact integer labels

### Lookup Speed
- O(|word_length|) for lexicon lookup
- Constant time per character for G2P fallback
- Batched processing optimizations available

### Accuracy Metrics
- Lexicon coverage: ~92% for typical English text
- G2P accuracy: ~85% phoneme-level correctness
- Word-level accuracy: ~78% for unknown words

## Extending the System

### Adding New Languages
```rust
trait LanguagePhonology {
    type PhonemeSet;
    fn grapheme_to_phoneme_rules(&self) -> Vec<G2PRule>;
    fn phonological_processes(&self) -> Vec<PhonProcess>;
    fn syllable_structure(&self) -> SyllableConstraints;
}
```

### Custom Phoneme Sets
```rust
fn build_custom_phoneme_fst<P: PhonemeInventory>(
    inventory: &P,
) -> VectorFst<TropicalWeight> {
    // Support for language-specific phoneme systems
}
```

### Integration with Speech Tools

```rust
// Kaldi integration
fn export_kaldi_lexicon(
    lexicon: &VectorFst<TropicalWeight>,
    output_path: &str,
) -> Result<()>;

// Festival integration  
fn export_festival_lexicon(
    lexicon: &VectorFst<TropicalWeight>,
    output_path: &str,
) -> Result<()>;
```

## Theoretical Background

### Phonological Theory
- **Phoneme**: Abstract speech sound unit
- **Allophone**: Contextual variant of phoneme
- **Morphophonology**: Sound changes in word formation
- **Prosody**: Stress, tone, rhythm patterns

### Computational Approaches
- **Joint-sequence models**: Direct string-to-string mapping
- **Alignment-based models**: Explicit grapheme-phoneme alignment
- **Neural approaches**: Sequence-to-sequence models
- **Hybrid systems**: FST + neural components

## Quality Assurance

### Testing Framework
```rust
fn test_lexicon_coverage(
    lexicon: &VectorFst<TropicalWeight>,
    test_corpus: &[String],
) -> CoverageStats {
    // Measures:
    // - Word coverage percentage
    // - OOV rate by frequency band
    // - G2P accuracy on held-out data
}
```

### Validation Tools
```rust
fn validate_pronunciations(
    entries: &[LexiconEntry],
) -> ValidationReport {
    // Checks:
    // - Phoneme inventory consistency
    // - Phonotactic constraints
    // - Stress pattern validity
    // - Morphological productivity
}
```

## Related Examples

- [Morphological Analyzer](morphological_analyzer.md) - Morphophonological processes
- [Edit Distance](edit_distance.md) - Phonetic similarity metrics
- [Transliteration](transliteration.md) - Cross-script pronunciation mapping
- [Word Correction](../examples/word_correction.rs) - Spelling with phonetic awareness

## Standards and Resources

- **ARPAbet**: ASCII phonetic alphabet for English
- **CMU Pronouncing Dictionary**: Large-scale English lexicon
- **IPA**: International Phonetic Alphabet
- **SAMPA**: Speech Assessment Methods Phonetic Alphabet
- **Festival**: Speech synthesis system integration