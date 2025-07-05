# Morphological Analyzer

This example demonstrates building a sophisticated morphological analyzer using FSTs to break words into meaningful components and reveal grammatical structure.

## Overview

Morphological analysis sits at the heart of natural language processing, breaking words into their meaningful components (morphemes). While simple for English ("walked" = walk + ed), morphology becomes complex in languages like Finnish, Turkish, or Arabic where a single word can encode what English expresses in an entire sentence.

FSTs provide an elegant solution to morphological complexity through bidirectional processing (analyze or generate words), rule composition (complex phenomena from simple rules), and language independence (works for any morphology). The framework handles rich morphological systems efficiently while maintaining mathematical rigor.

This implementation follows foundational work by Karttunen, Koskenniemi, and other computational morphology pioneers, demonstrating how FSTs can model everything from English derivation to Finnish vowel harmony.

## Quick Start

```bash
cargo run --example morphological_analyzer
```

## What You'll Learn

- **Two-Level Morphology**: Koskenniemi's model for handling sound changes at morpheme boundaries
- **Morphotactics**: Language constraints on morpheme ordering through LEXC-style rules  
- **Bidirectional Processing**: Using the same FST for both analysis and generation
- **Morphophonology**: Modeling sound changes like vowel harmony and consonant gradation
- **Lexicon Building**: Creating FST-based lexicons for stems and affixes
- **Cross-Linguistic Analysis**: Handling English, Finnish, Turkish, and Arabic morphology

## Core Concepts

### Theoretical Background

This implementation is based on foundational work in computational morphology, including **Karttunen, L. (1993)** on finite-state lexicon compiler (LEXC), **Koskenniemi, K. (1983)** on two-level morphology, **Karttunen, L. & Beesley, K. (2001)** on Finite State Morphology, and **Karttunen, L. (1994)** on constructing lexical transducers.

Understanding morphology requires grasping how languages build words from smaller units. Let's explore the key concepts with concrete examples.

### Morphological Categories

Languages encode various types of information through morphology. Our analyzer handles:

#### Inflectional Categories (Grammar)

Part of speech tags include Noun (N) for entities and concepts (cat, happiness, Finland), Verb (V) for actions and states (run, exist, analyze), Adjective (A) for properties (blue, happy, computational), and Adverb (Adv) for manner, time, place (quickly, yesterday, here).

Number marking includes singular for one entity (cat, child), plural for multiple entities (cats, children), and dual for exactly two (found in Arabic, Slovenian).

**Case Systems (Finnish examples):**
```
Nominative (subject):     kissa       "cat" (as subject)
Genitive (possession):    kissan      "cat's"
Partitive (partial):      kissaa      "some cat"
Illative (into):         kissaan     "into the cat"
Inessive (in):           kissassa    "in the cat"
Elative (from):          kissasta    "from the cat"
... and 9 more cases
```

Verbal categories include tense such as present (walk), past (walked), future (will walk); person such as 1st (I walk), 2nd (you walk), 3rd (she walks); and voice such as active (I broke it), passive (it was broken).

#### Derivational Categories (Word Formation)

**Creating New Words:**
```
Verb → Noun (agent):      teach → teacher
Adjective → Noun:         happy → happiness
Noun → Adjective:         nation → national
Verb → Adjective:         break → breakable
```

Semantic changes include diminutives that make smaller/cuter (dog → doggy), augmentatives that make bigger (in Spanish: casa → casona), and causatives that add "cause to" meaning (in Turkish: öl- "die" → öldür- "kill").

### Finite State Lexicon: Organizing Morphological Knowledge

The lexicon is where linguistic knowledge meets computational representation. Our FST-based lexicon organizes morphemes efficiently:

```rust
struct FiniteStateLexicon {
    // Stem classes by category
    noun_stems: HashMap<String, Vec<MorphCategory>>,
    verb_stems: HashMap<String, Vec<MorphCategory>>,
    adjective_stems: HashMap<String, Vec<MorphCategory>>,
    
    // Affix inventories with their features
    noun_suffixes: HashMap<String, Vec<MorphCategory>>,
    verb_suffixes: HashMap<String, Vec<MorphCategory>>,
    derivational_suffixes: HashMap<String, Vec<MorphCategory>>,
    
    // Phonological alternation rules
    phonological_rules: Vec<(String, String, String)>,
}
```

**Key Design Principles:**

The system follows several key design principles. **Stem Classes** group words by inflection patterns, with English distinguishing regular vs. irregular verbs (walk/walked vs. go/went) and Finnish using vowel harmony classes (front vs. back vowels). **Feature Bundles** ensure each morpheme carries grammatical information, such as "+ed" → `[Past, Active]` and "+ssa" → `[Inessive, Singular]`. **Rule-Based Alternations** keep sound changes separate from the lexicon, storing "happy" (stem) and "+ness" (suffix), applying the rule y→i before suffixes to produce the surface form "happiness".

## Implementation

Let's explore how different languages challenge our morphological analyzer with increasing complexity.

### Finnish Morphology: A Case Study in Complexity

Finnish exemplifies agglutinative morphology where words are built by concatenating morphemes, each adding specific meaning. With 15 cases and strict vowel harmony, it's an ideal test for our FST approach.

#### The Finnish Case System

Finnish nouns inflect for 15 cases, each encoding spatial, temporal, or grammatical relationships:

**Basic Cases (Grammatical):**
```
kala (fish) - Nominative (subject)
├── kalan   - Genitive (possession/object)      "fish's" / "of fish"
├── kalaa   - Partitive (partial object)        "some fish"
└── kalaksi - Translative (transformation)      "into a fish"
```

**Local Cases (Spatial Relations):**
```
Interior (inside):
├── kalassa  - Inessive (in)         "in the fish"
├── kalaan   - Illative (into)       "into the fish"
└── kalasta  - Elative (out of)      "out of the fish"

Exterior (surface):
├── kalalla  - Adessive (on/at)      "on the fish"
├── kalalle  - Allative (onto)       "onto the fish"
└── kalalta  - Ablative (off/from)   "off the fish"
```

#### Vowel Harmony: A Phonological Constraint

Finnish vowels are divided into three groups: **back vowels** (a, o, u), **front vowels** (ä, ö, y), and **neutral vowels** (e, i) which can appear with either group.

**Harmony Rule**: Suffixes must match the stem's vowel type:

```
Back vowel stems:
talo (house) + -ssa → talossa    "in the house"
auto (car)   + -lla → autolla    "by car"

Front vowel stems:
kylä (village) + -ssä → kylässä  "in the village"
yö (night)     + -llä → yöllä    "at night"
```

This is where FSTs shine - the same morphological rule has different surface realizations based on phonological context.

### English Morphology

English examples focus on derivational morphology and inflectional patterns:

**Derivational Processes:**
```
work + er  → worker    (agent nominal)
happy + ness → happiness (abstract nominal with y→i rule)
write + er → writer    (with e-deletion)
```

**Inflectional Morphology:**
```
cat + s   → cats     (plural)
walk + ed → walked   (past tense)
work + s  → works    (3rd person singular)
```

### Analysis Pipeline

The morphological analyzer processes words through several stages. **Lexical Lookup** checks if the word exists in stem dictionaries. **Affix Segmentation** tries different affix combinations. **Phonological Processing** applies morphophonological rules. **Feature Assembly** combines stem and affix features. **Result Formatting** generates morphological glosses.

```rust
fn analyze(&self, surface_form: &str) -> Vec<MorphAnalysis> {
    let mut analyses = Vec::new();
    
    // Try different morphological categories
    analyses.extend(self.analyze_as_noun(surface_form));
    analyses.extend(self.analyze_as_verb(surface_form));
    analyses.extend(self.analyze_derivational(surface_form));
    
    analyses
}
```

### Morphophonological Processing

The system handles common sound change patterns:

**Y-to-I Rule (English):**
```rust
// happy + ness → happiness
if stem.ends_with('y') && !suffix.is_empty() {
    let modified_stem = format!("{}{}", &stem[..stem.len() - 1], "i");
    let expected = format!("{}{}", modified_stem, suffix);
    if expected == surface_form {
        return true;
    }
}
```

**E-Deletion Rule (English):**
```rust
// write + er → writer  
if stem.ends_with('e') && !suffix.is_empty() {
    let modified_stem = &stem[..stem.len() - 1];
    let expected = format!("{}{}", modified_stem, suffix);
    if expected == surface_form {
        return true;
    }
}
```

## Running the Example

```bash
cargo run --example morphological_analyzer
```

### Sample Output

```
Finnish Morphological Analysis
---------------------------------
Analyzing 'kalan':
  Analysis 1: kala+n [N+Gen+Sg]
    Gloss: kala.GEN
    Morphemes: kala + n

Analyzing 'kirjassa':
  Analysis 1: kirja+ssa [N+Iness+Sg]
    Gloss: kirja.INESS
    Morphemes: kirja + ssa

English Derivational Morphology
----------------------------------
Analyzing 'worker':
  Analysis 1: work+er [V+Agent+N]
    Gloss: work.AGENT
    Morphemes: work + er

Analyzing 'happiness':
  Analysis 1: happy+ness [A+Abstr+N]
    Gloss: happy.ABSTR
    Morphemes: happy + ness
```

## FST Implementation

### Morphotactic FST

The example includes a simplified FST for morpheme ordering:

```rust
fn build_morphotactic_fst() -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    
    // States for: Start → NounStem → NounInflected → Final
    let start = fst.add_state();
    let noun_stem = fst.add_state();
    let noun_inflected = fst.add_state();
    let final_state = fst.add_state();
    
    // Add morpheme boundary marker
    fst.add_arc(noun_stem, Arc::new(
        '+' as u32,   // morpheme boundary
        0,            // epsilon output
        TropicalWeight::one(),
        noun_inflected,
    ));
    
    fst
}
```

## Applications

This morphological analyzer enables several important applications:

### Natural Language Processing
The morphological analyzer enables several important applications in natural language processing. **Machine Translation** benefits from better handling of morphologically rich languages. **Information Retrieval** improves through lemmatization and morphological normalization. **Text Mining** can extract semantic content from morphological variants more effectively. **Language Modeling** achieves better handling of unseen word forms.

### Linguistic Research
The analyzer supports various linguistic research applications. **Corpus Analysis** enables studying morphological patterns in large corpora. **Language Documentation** helps model morphological systems of understudied languages. **Historical Linguistics** can trace morphological changes over time. **Typological Studies** facilitate comparing morphological systems across languages.

### Language Technology
The system enables various language technology applications. **Spell Checkers** benefit from morphologically-aware error detection and correction. **Grammar Checkers** can detect morphosyntactic errors more accurately. **Text-to-Speech** systems achieve proper pronunciation of morphologically complex words. **Language Learning** applications can generate morphological exercises and provide feedback.

## Advanced Features

### Multiple Pronunciations

The system handles words with multiple valid analyses:

```rust
LexiconEntry {
    word: "read".to_string(),
    pronunciations: vec![
        vec![R, IY, D], // present tense /riːd/
        vec![R, EH, D], // past tense /rɛd/
    ],
}
```

### Contextual Rules

Future extensions could include context-sensitive morphophonological rules:

```rust
PhonologicalRule {
    name: "Finnish Consonant Gradation".to_string(),
    input_context: vec!["k".to_string()],
    output_context: vec!["".to_string()], // deletion
    environment: Some("V_V".to_string()),  // between vowels
}
```

## Performance Considerations

The system incorporates several performance optimizations. **Trie Structure** provides efficient prefix sharing for stem storage. **Rule Caching** memoizes morphophonological rule applications to avoid recomputation. **Lazy Evaluation** generates analyses on demand rather than precomputing all possibilities. **Memory Management** reuses FST states where possible to minimize resource usage.

The system gracefully handles various edge cases including unknown words (no analysis returned), malformed input (character encoding issues), rule conflicts (multiple competing analyses), and resource limits (maximum analysis depth).

## Advanced Features

### Enhanced Phonology
Future work could extend the phonological component with **Two-Level Rules** providing full implementation of Koskenniemi's formalism, **Feature-Based Phonology** using articulatory feature representations, **Optimality Theory** for constraint-based phonological analysis, and **Morphophonemic Alternations** to handle complex stem changes.

### Broader Language Coverage
Expanding language coverage could include **Semitic Languages** with root-and-pattern morphology (Arabic, Hebrew), **Polysynthetic Languages** with complex agglutination (Inuktitut, Mohawk), **Tonal Languages** exploring the interaction of tone and morphology, and **Sign Languages** with spatial morphological processes.

### Machine Learning Integration
Future directions could incorporate **Neural Morphology** using deep learning for morphological analysis, **Unsupervised Learning** to discover morphological patterns from corpora, **Transfer Learning** to adapt analyzers across related languages, and **Active Learning** to iteratively improve with human feedback.

## Related Examples

This morphological analyzer connects with other examples in the collection. **[Edit Distance](../text-processing/edit_distance.md)** provides the foundation for fuzzy morphological matching. **[Phonological Rules](phonological_rules.md)** demonstrates advanced sound change modeling. **[Spell Checking](../text-processing/spell_checking.md)** shows morphologically-aware spell checking. **[Transliteration](../practical-applications/transliteration.md)** explores cross-script morphological analysis.

## References

### Foundational Papers
Key foundational papers include Koskenniemi, K. (1983) "Two-level morphology: A general computational model for word-form recognition and production", Karttunen, L. (1993) "Finite-state lexicon compiler" Technical Report ISTL-NLTT-1993-04-02, Xerox PARC, Kaplan, R. & Kay, M. (1994) "Regular models of phonological rule systems", and Karttunen, L. (1994) "Constructing lexical transducers".

### Modern Applications
Modern applications are described in Beesley, K.R. & Karttunen, L. (2003) "Finite State Morphology" (CSLI Publications), Jurafsky, D. & Martin, J.H. (2023) "Speech and Language Processing" (3rd Edition), and Roark, B. & Sproat, R. (2007) "Computational Approaches to Morphology and Syntax".

### Software Implementations
Relevant software implementations include **XFST** (Xerox Finite State Tool), **HFST** (Helsinki Finite State Technology), **Foma** (Open-source finite state compiler), and **OpenFST** (Google's finite state library).

This morphological analyzer demonstrates the power and elegance of finite state methods for modeling natural language morphology, providing both theoretical insights and practical applications for computational linguistics and NLP systems.