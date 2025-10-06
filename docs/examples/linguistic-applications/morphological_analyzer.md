# Morphological Analyzer

This example implements a two-level morphological analyzer using finite-state transducers for morphological analysis and generation.

## Overview

Morphological analysis decomposes words into constituent morphemes while identifying their grammatical properties. The complexity varies significantly across languages: English exhibits relatively simple morphology (e.g., "walked" → walk + -ed), while agglutinative languages such as Finnish, Turkish, and Arabic encode multiple grammatical categories within single word forms.

Finite state transducers provide a formal framework for morphological processing through:
- Bidirectional transduction between lexical level and surface level
- Two-level rules as parallel constraints
- Language-independent computational architecture

This implementation follows the two-level morphology model {{#cite koskenniemi1983two}} and the Xerox finite-state morphology framework {{#cite beesley2003finite}}, demonstrating applications from English derivational morphology to Finnish case systems with vowel harmony.

## Usage

```bash
cargo run --example morphological_analyzer
```

## Concepts Demonstrated

- **Two-Level Morphology**: Implementation of Koskenniemi's two-level formalism
- **Morphotactics**: Continuation lexicons for morpheme ordering (lexc style)  
- **Bidirectional Processing**: Same transducer for analysis and generation
- **Morphophonological Rules**: Two-level constraints for alternations
- **Lexicon Structure**: Lexical entries with continuation classes
- **Cross-Linguistic Coverage**: Applications to typologically diverse languages

## Core Concepts

### Theoretical Background

The implementation draws upon foundational work in computational morphology: {{#cite koskenniemi1983two}} for the two-level model, {{#cite beesley2003finite}} for the comprehensive Xerox finite-state morphology framework, and the two-level compiler (twolc) developed by Karttunen, Koskenniemi, and Kaplan.

Morphological systems construct words through systematic combination of morphemes according to language-specific morphotactic constraints and morphophonological rules.

### Morphological Categories

Languages encode grammatical and semantic information through morphological processes. The analyzer implements:

#### Inflectional Categories (Grammar)

**Part of Speech Categories:**
- Noun (N): entities and concepts
- Verb (V): actions and states
- Adjective (A): properties and attributes
- Adverb (Adv): manner, time, and place

**Number Marking:**
- Singular: single entity reference
- Plural: multiple entity reference
- Dual: exactly two entities (Arabic, Slovenian)

**Case Systems (Finnish examples):**
```text
Nominative (subject):     kala        "fish" (as subject)
Genitive (possession):    kalan       "of the fish/fish's"
Partitive (partial):      kalaa       "(some) fish"
Illative (into):         kalaan      "into the fish"
Inessive (in):           kalassa     "in the fish"
Elative (from):          kalasta     "from/out of the fish"
... and 9 more cases
```

**Verbal Categories:**
- Tense: present, past, future
- Person: first, second, third
- Voice: active, passive

#### Derivational Categories (Word Formation)

**Creating New Words:**
```text
Verb → Noun (agent):      teach → teacher
Adjective → Noun:         happy → happiness
Noun → Adjective:         nation → national
Verb → Adjective:         break → breakable
```

**Semantic Derivation:**
- Diminutive: semantic diminution (e.g., dog → doggy)
- Augmentative: semantic augmentation (e.g., Spanish: casa → casona)
- Causative: causation semantics (e.g., Turkish: öl- "die" → öldür- "kill")

### Finite State Lexicon Architecture

The lexicon encodes morphological knowledge following the lexc formalism:

```rust,ignore
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

**Design Principles:**

1. **Stem Classification**: Morphemes grouped by inflectional patterns
   - English: regular vs. irregular verb classes
   - Finnish: vowel harmony classes (front/back vowel stems)

2. **Feature Representation**: Grammatical features encoded as attribute sets
   - Example: "+ed" → [+Past, +Active]
   - Example: "+ssa" → [+Inessive, +Singular]

3. **Two-Level Rules**: Morphophonological alternations as parallel constraints
   - Example: y:i ⇔ _ +:0 ness (happy + ness → happiness)

## Implementation

The implementation handles languages of varying morphological complexity.

### Finnish Morphology: A Case Study in Complexity

Finnish exemplifies agglutinative morphology through systematic morpheme concatenation. The language employs 15 grammatical cases, vowel harmony constraints, and consonant gradation, demonstrating the capabilities of finite state morphological analysis.

#### The Finnish Case System

Finnish nouns inflect for 15 cases, each encoding spatial, temporal, or grammatical relationships:

**Basic Cases (Grammatical):**
```text
kala (fish) - Nominative (subject)
├── kalan   - Genitive (possession/object)      "of the fish"
├── kalaa   - Partitive (partial object)        "(some) fish"
└── kalaksi - Translative (transformation)      "(to become) a fish"
```

**Local Cases (Spatial Relations):**
```text
Interior (inside):
├── kalassa  - Inessive (in)         "in the fish"
├── kalaan   - Illative (into)       "into the fish"
└── kalasta  - Elative (from)        "from/out of the fish"

Exterior (surface/possession):
├── kalalla  - Adessive (at/with)    "at/by the fish" or "the fish has"
├── kalalle  - Allative (to)         "to the fish"
└── kalalta  - Ablative (from)       "from the fish"
```

#### Vowel Harmony: A Phonological Constraint

Finnish vowels are divided into three groups: **back vowels** (a, o, u), **front vowels** (ä, ö, y), and **neutral vowels** (e, i) which can appear with either group.

**Harmony Rule**: Suffixes must match the stem's vowel type:

```text
Back vowel stems (a, o, u):
talo (house) + -ssa → talossa    "in the house"
auto (car)   + -lla → autolla    "by car"
katu (street) + -lla → kadulla   "on the street"

Front vowel stems (ä, ö, y):
kylä (village) + -ssä → kylässä  "in the village"
yö (night)     + -llä → yöllä    "at night"
työ (work)     + -ssä → työssä   "at work"
```

Finite-state transducers handle these alternations through two-level rules that define correspondences between the lexical level and surface level based on phonological context. The suffix alternants -ssa/-ssä, -lla/-llä, etc., are selected based on the vowel harmony of the stem.

### English Morphology

English morphology demonstrates both inflectional and derivational processes:

**Derivational Processes:**
```text
work + er  → worker     (agent nominal)
happy + ness → happiness (abstract nominal with y:i correspondence)
write + er → writer     (with e:0 correspondence)
teach + er → teacher    (agent nominal)
kind + ness → kindness  (abstract nominal)
```

**Inflectional Morphology:**
```text
cat + s   → cats     (plural)
walk + ed → walked   (past tense)
work + s  → works    (3rd person singular)
```

### Analysis Pipeline

The morphological analysis proceeds through the following stages:

1. **Surface Form Input**: Receive the word form to analyze
2. **Transducer Application**: Apply the composed lexicon-rules transducer
3. **Lexical Form Recovery**: Extract underlying representations
4. **Feature Assembly**: Combine morphological features from morphemes
5. **Result Formatting**: Generate interlinear morphological glosses

```rust,ignore
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

The implementation includes morphophonological rules for common alternations:

**Y-to-I Rule (English):**
```rust,ignore
// Simplified implementation of two-level rule:
// y:i ⇔ _ +:0 ness
// (y corresponds to i when followed by morpheme boundary and "ness")
if stem.ends_with('y') && suffix == "ness" {
    let modified_stem = format!("{}{}", &stem[..stem.len() - 1], "i");
    let expected = format!("{}{}", modified_stem, suffix);
    if expected == surface_form {
        return true;
    }
}
```

**E-Deletion Rule (English):**
```rust,ignore
// Simplified implementation of two-level rule:
// e:0 ⇔ _ +:0 er  
// (e is deleted when followed by morpheme boundary and "er")
if stem.ends_with('e') && suffix == "er" {
    let modified_stem = &stem[..stem.len() - 1];
    let expected = format!("{}{}", modified_stem, suffix);
    if expected == surface_form {
        return true;
    }
}
```

## Execution

```bash
cargo run --example morphological_analyzer
```

### Sample Output

```text
Finnish Morphological Analysis
---------------------------------
Analyzing 'kalan':
  Analysis 1: kala+n [N+Gen+Sg]
    Gloss: kala.GEN
    Morphemes: kala + n

Analyzing 'kalassa':
  Analysis 1: kala+ssa [N+Iness+Sg]
    Gloss: kala.INESS
    Morphemes: kala + ssa

Analyzing 'taloon':
  Analysis 1: talo+on [N+Ill+Sg]
    Gloss: talo.ILL
    Morphemes: talo + on

Analyzing 'luen':
  Analysis 1: luke+n [V+Pres+1+Sg]
    Gloss: read.PRES.1SG
    Morphemes: luke + n

English Derivational Morphology
----------------------------------
Analyzing 'worker':
  Analysis 1: work+er [V+Ag+N]
    Gloss: work.AGENT
    Morphemes: work + er

Analyzing 'happiness':
  Analysis 1: happy+ness [A+Abstr+N]
    Gloss: happy.ABSTR
    Morphemes: happy + ness
```

## FST Implementation

### Morphotactic FST

A simplified morphotactic transducer encodes morpheme ordering constraints (following lexc continuation class principles):

```rust,ignore
fn build_morphotactic_fst() -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    
    // States for: Start → NounStem → NounInflected → Final
    let start = fst.add_state();
    let noun_stem = fst.add_state();
    let noun_inflected = fst.add_state();
    let final_state = fst.add_state();
    
    // Add morpheme boundary (lexc uses ^, we use + here)
    fst.add_arc(noun_stem, Arc::new(
        '+' as u32,   // morpheme boundary symbol
        0,            // epsilon on surface (boundary deleted)
        TropicalWeight::one(),
        noun_inflected,
    ));
    
    fst
}
```

## Applications

The morphological analyzer supports various computational linguistics applications:

### Natural Language Processing

- **Machine Translation**: Improved handling of morphologically complex languages
- **Information Retrieval**: Morphological normalization and lemmatization
- **Text Mining**: Extraction of semantic content across morphological variants
- **Language Modeling**: Productive handling of unseen word forms

### Linguistic Research

- **Corpus Analysis**: Quantitative study of morphological patterns
- **Language Documentation**: Formal modeling of morphological systems
- **Historical Linguistics**: Tracking morphological change over time
- **Typological Studies**: Cross-linguistic comparison of morphological phenomena

### Language Technology

- **Spell Checking**: Morphologically informed error detection and correction
- **Grammar Checking**: Morphosyntactic agreement verification
- **Text-to-Speech**: Accurate pronunciation of morphologically complex forms
- **Computer-Assisted Language Learning**: Morphological exercise generation

## Advanced Features

### Morphological Ambiguity

The system handles morphologically ambiguous forms:

```rust,ignore
LexiconEntry {
    word: "read".to_string(),
    pronunciations: vec![
        vec![R, IY, D], // present tense /riːd/
        vec![R, EH, D], // past tense /rɛd/
    ],
}
```

### Context-Sensitive Rules

The framework supports context-dependent morphophonological alternations:

```rust,ignore
PhonologicalRule {
    name: "Finnish Consonant Gradation".to_string(),
    // Strong grade → weak grade examples:
    // katu → kadun (t→d)
    // kukka → kukan (kk→k)  
    // kampa → kamman (mp→mm)
    input_context: vec!["t".to_string()],
    output_context: vec!["d".to_string()],
    environment: Some("V_V".to_string()),  // between vowels in closed syllable
}
```

## Performance Considerations

### Optimization Strategies

- **Trie-based Lexicon**: Efficient prefix sharing for stem storage
- **Rule Memoization**: Caching of morphophonological rule applications
- **Lazy Evaluation**: On-demand analysis generation
- **State Reuse**: Minimal FST state allocation

### Edge Case Handling

- Unknown words: Return empty analysis set
- Encoding issues: UTF-8 validation
- Ambiguous analyses: Return all valid interpretations
- Resource constraints: Bounded search depth

## Extensions

### Enhanced Phonological Modeling

- Full two-level rule implementation with proper rule notation
- Feature-based phonological representations
- Context-sensitive alternations with left and right contexts
- Complex morphophonemic phenomena (gradation, harmony, etc.)

### Additional Language Types

- Semitic root-and-pattern morphology (Arabic, Hebrew)
- Polysynthetic morphology (Inuktitut, Mohawk)
- Tonal morphology interactions
- Sign language morphological processes

### Computational Approaches

- Neural morphological analysis models
- Unsupervised morphological segmentation
- Cross-lingual transfer learning
- Active learning for morphological rule acquisition

## Related Examples

- **[Edit Distance](../text-processing/edit_distance.md)**: String similarity for morphological variant matching
- **[Phonological Rules](phonological_rules.md)**: Cascaded phonological rule application
- **[Spell Checking](../text-processing/spell_checking.md)**: Morphologically informed spelling correction
- **[Transliteration](../practical-applications/transliteration.md)**: Cross-script morphological processing

## References

### Foundational Work

- {{#cite koskenniemi1983two}}: Two-level morphology: A general computational model
- {{#cite karttunen1987compiler}}: A compiler for two-level phonological rules
- {{#cite beesley2003finite}}: Finite State Morphology (Xerox tools and theory)

### Related Software

- lexc: Xerox lexicon compiler for morphotactics
- twolc: Two-level rule compiler (Xerox)
- xfst: Xerox finite-state tool for regular expressions
- HFST: Helsinki Finite-State Technology (open-source alternatives)
- Foma: Open-source finite-state morphology compiler