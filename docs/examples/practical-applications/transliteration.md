# Transliteration

This example implements script conversion systems using finite state transducers, supporting multiple transliteration standards and writing systems.

## Overview

Transliteration preserves phonetic representation across writing systems, distinct from translation which preserves semantic content. Key challenges include:
- One-to-many character mappings (e.g., Cyrillic 'х' → 'kh' or 'h')
- Context-dependent forms (e.g., Arabic positional variants)
- Multiple competing standards (BGN/PCGN, ISO, popular conventions)
- Reversibility constraints

Finite state transducers encode transliteration rules as composable transformations. The implementation supports BGN/PCGN and ISO standards for Cyrillic, Arabic, and Greek to Latin conversion.

## Usage

```bash
cargo run --example transliteration
```

## Concepts Demonstrated

- **Multi-Script Support**: Cyrillic, Arabic, and Greek to Latin conversion
- **Standards Compliance**: BGN/PCGN and ISO transliteration schemes
- **Context-Dependent Rules**: Positional variants and digraph handling
- **Bidirectional Transduction**: Reversible transliteration where applicable
- **Ambiguity Management**: Multiple valid transliterations
- **Efficient Implementation**: Optimized transducer composition

## Core Concepts

### Writing Systems

Each script presents distinct challenges for systematic transliteration.

### Cyrillic Script

Cyrillic transliteration addresses the following phenomena:

#### Character Mappings

**Simple Mappings:**
```text
а → a    б → b    в → v    г → g    д → d
е → e    з → z    и → i    к → k    л → l
м → m    н → n    о → o    п → p    р → r
с → s    т → t    у → u    ф → f
```text

**Digraph Mappings:**
```text
ж → zh   (one Cyrillic letter → two Latin letters)
х → kh   (BGN/PCGN) or h (popular)
ц → ts   (represents /ts/ sound)
ч → ch   (represents /tʃ/ sound)
ш → sh   (represents /ʃ/ sound)
щ → shch (most complex: one → four letters!)
```text

**Special Characters:**
```text
ъ → " (hard sign)
ь → ' (soft sign)
ё → yo (BGN/PCGN) or e 
й → y 
ы → y 
э → e 
ю → yu 
я → ya 
```

#### Examples

```text
москва → moskva 
санкт-петербург → sankt-peterburg
большой → bol'shoy 
хорошо → khorosho (BGN/PCGN) / horosho (popular)
щи → shchi
```

### Arabic Script

Arabic to Latin transliteration following BGN/PCGN standards:

**Arabic Letters:**
```text
ا -> a    ب -> b    ت -> t    ث -> th   ج -> j
ح -> h    خ -> kh   د -> d    ذ -> dh   ر -> r
ز -> z    س -> s    ش -> sh   ص -> s    ض -> d
ط -> t    ظ -> z    ع -> '    غ -> gh   ف -> f
ق -> q    ك -> k    ل -> l    م -> m    ن -> n
ه -> h    و -> w    ي -> y
```

**Examples:**
```text
السلام -> assalam
مرحبا -> marhaba
قاهرة -> qahirah
بغداد -> baghdad
مكة -> makkah
```

### Greek Script

Greek to Latin transliteration:

**Greek Alphabet:**
```text
α -> a    β -> v    γ -> g    δ -> d    ε -> e
ζ -> z    η -> i    θ -> th   ι -> i    κ -> k
λ -> l    μ -> m    ν -> n    ξ -> x    ο -> o
π -> p    ρ -> r    σ/ς -> s  τ -> t    υ -> y
φ -> f    χ -> ch   ψ -> ps   ω -> o
```

**Examples:**
```text
αθήνα -> athina
θεσσαλονίκη -> thessaloniki
φιλοσοφία -> filosofia
δημοκρατία -> dimokratia
```

## Transliteration Standards

### BGN/PCGN

Board on Geographic Names/Permanent Committee on Geographical Names standard:

```rust,ignore
TransliterationRule {
    source: "х".to_string(),
    target: "kh".to_string(),
    scheme: TransliterationScheme::BgnPcgn,
}
```

Characteristics:
- Geographic name standardization
- Official document usage
- Diacritical mark preservation

### Popular Conventions

Simplified transliteration schemes:

```rust,ignore
TransliterationRule {
    source: "х".to_string(),
    target: "h".to_string(),    // Simplified vs "kh"
    scheme: TransliterationScheme::Popular,
}
```

Characteristics:
- ASCII-compatible representation
- Keyboard-friendly input
- Informal usage contexts

### ISO Standards

- ISO 9:1995 - Cyrillic transliteration
- ISO 233:1984 - Arabic transliteration
- ISO 843:1997 - Greek transliteration

## Implementation

### TransliterationSystem Structure

The core system organizes rules by script and scheme:

```rust,ignore
struct TransliterationSystem {
    cyrillic_to_latin: HashMap<TransliterationScheme, Vec<TransliterationRule>>,
    arabic_to_latin: HashMap<TransliterationScheme, Vec<TransliterationRule>>,
    greek_to_latin: HashMap<TransliterationScheme, Vec<TransliterationRule>>,
}
```

### TransliterationRule Definition

Individual rules specify context and constraints:

```rust,ignore
struct TransliterationRule {
    source: String,              // Input character/sequence
    target: String,              // Output character/sequence
    context_before: Option<String>, // Left context
    context_after: Option<String>,  // Right context
    scheme: TransliterationScheme,  // Transliteration standard
}
```

### Simple String-Based Processing

For demonstration, the system uses efficient string replacement:

```rust,ignore
fn transliterate_simple(
    &self,
    text: &str,
    source_script: Script,
    scheme: TransliterationScheme,
) -> String {
    let rules = match source_script {
        Script::Cyrillic => self.cyrillic_to_latin.get(&scheme),
        Script::Arabic => self.arabic_to_latin.get(&scheme),
        Script::Greek => self.greek_to_latin.get(&scheme),
        _ => None,
    };

    if let Some(rules) = rules {
        let mut result = text.to_string();
        
        // Sort by length (longest first) for correct digraph handling
        let mut sorted_rules = rules.clone();
        sorted_rules.sort_by(|a, b| b.source.len().cmp(&a.source.len()));

        for rule in &sorted_rules {
            result = result.replace(&rule.source, &rule.target);
        }
        
        result
    } else {
        text.to_string()
    }
}
```

## FST-Based Implementation

### Transliteration FST Construction

Build FSTs for rule-based transliteration:

```rust,ignore
fn build_transliteration_fst(rules: &[TransliterationRule]) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    fst.set_start(start);
    fst.set_final(start, TropicalWeight::one());

    // Sort rules by source length (longest first) for digraphs
    let mut sorted_rules = rules.to_vec();
    sorted_rules.sort_by(|a, b| b.source.len().cmp(&a.source.len()));

    for rule in &sorted_rules {
        let mut current = start;
        let source_chars: Vec<char> = rule.source.chars().collect();
        
        // Process multi-character sequences
        for (i, &ch) in source_chars.iter().enumerate() {
            if i == source_chars.len() - 1 {
                // Last character - output the target
                let target_chars: Vec<char> = rule.target.chars().collect();
                
                for &target_ch in &target_chars {
                    let next = fst.add_state();
                    fst.add_arc(current, Arc::new(
                        ch as u32, target_ch as u32,
                        TropicalWeight::one(), next
                    ));
                    current = next;
                }
                
                // Connect back to start
                fst.add_arc(current, Arc::new(
                    0, 0,  // epsilon transition
                    TropicalWeight::one(), start
                ));
            } else {
                // Intermediate character
                let next = fst.add_state();
                fst.add_arc(current, Arc::new(
                    ch as u32, 0,  // epsilon output
                    TropicalWeight::one(), next
                ));
                current = next;
            }
        }
    }

    // Pass-through for unknown characters
    for ch in 0..=127u8 {
        if ch.is_ascii() {
            fst.add_arc(start, Arc::new(
                ch as u32, ch as u32,
                TropicalWeight::one(), start
            ));
        }
    }

    fst
}
```

### Bidirectional Transliteration

Create reverse mappings for back-transliteration:

```rust,ignore
fn create_reverse_mappings(rules: &[TransliterationRule]) -> Vec<TransliterationRule> {
    rules.iter().map(|rule| TransliterationRule {
        source: rule.target.clone(),
        target: rule.source.clone(),
        context_before: rule.context_before.clone(),
        context_after: rule.context_after.clone(),
        scheme: rule.scheme,
    }).collect()
}
```

**Usage:**
```rust,ignore
// Forward: Cyrillic -> Latin
let forward_fst = build_transliteration_fst(&cyrillic_rules);

// Reverse: Latin -> Cyrillic  
let reverse_rules = create_reverse_mappings(&cyrillic_rules);
let reverse_fst = build_transliteration_fst(&reverse_rules);
```

## Running the Example

```bash
cargo run --example transliteration
```

### Sample Output

```text
Cyrillic to Latin Transliteration:
------------------------------------
Russian: 'москва'
  BGN/PCGN: 'moskva'
  Popular:  'moskva'

Russian: 'матрёшка'
  BGN/PCGN: 'matryoshka'
  Popular:  'matryoshka'

Russian: 'борщ'
  BGN/PCGN: 'borshch'
  Popular:  'borsch'

Arabic to Latin Transliteration:
----------------------------------
Arabic: 'السلام'
  Latin: 'assalam'

Arabic: 'قاهرة'
  Latin: 'qahirah'

Greek to Latin Transliteration:
---------------------------------
Greek: 'αθήνα'
  Latin: 'athina'

Greek: 'φιλοσοφία'
  Latin: 'filosofia'
```

## Advanced Features

### Scheme Comparison

Different schemes handle characters differently:

```text
Character: х (Cyrillic KHA)
BGN/PCGN: 'kh'    (linguistic accuracy)
Popular:  'h'     (simplified)

Character: ё (Cyrillic YO)  
BGN/PCGN: 'ë'     (preserves umlaut)
Popular:  'yo'    (ASCII-only)

Character: щ (Cyrillic SHCHA)
BGN/PCGN: 'shch'  (full representation)
Popular:  'sch'   (simplified)
```

### Context-Sensitive Rules

Handle position-dependent transliteration:

```rust,ignore
struct ContextualRule {
    source: String,
    target: String,
    before_context: Option<String>,  // Left context
    after_context: Option<String>,   // Right context
    position: Position,              // Word position
}

enum Position {
    Initial,    // Word-initial
    Medial,     // Word-medial
    Final,      // Word-final
    Any,        // Any position
}
```

**Example:**
```text
Arabic ة (ta marbuta):
Word-final: ة -> ah
Word-medial: ة -> at
```

### Digraph Handling

Multi-character sequences require careful processing:

```rust,ignore
// Process longest matches first
let digraphs = vec![
    ("дж", "dzh"),  // Cyrillic digraph
    ("тс", "ts"),   // Cyrillic sequence
    ("кх", "kkh"),  // Arabic digraph
];

// Sort by length descending
digraphs.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
```

## Applications

### Information Retrieval

Enable cross-script search:

```rust,ignore
fn cross_script_search(
    query: &str,
    documents: &[Document],
    transliterator: &TransliterationSystem,
) -> Vec<SearchResult> {
    let mut variants = vec![query.to_string()];
    
    // Generate transliteration variants
    for script in [Script::Cyrillic, Script::Arabic, Script::Greek] {
        for scheme in [TransliterationScheme::BgnPcgn, TransliterationScheme::Popular] {
            let variant = transliterator.transliterate_simple(query, script, scheme);
            if variant != query {
                variants.push(variant);
            }
        }
    }
    
    // Search with all variants
    let mut results = Vec::new();
    for variant in variants {
        results.extend(search_documents(&variant, documents));
    }
    
    deduplicate_and_rank(results)
}
```

### Machine Translation

Preprocessing for MT systems:

```rust,ignore
fn preprocess_for_mt(
    text: &str,
    source_lang: Language,
    target_lang: Language,
    transliterator: &TransliterationSystem,
) -> String {
    match (source_lang.script(), target_lang.script()) {
        (Script::Cyrillic, Script::Latin) => {
            transliterator.transliterate_simple(
                text, Script::Cyrillic, TransliterationScheme::Popular
            )
        },
        (Script::Arabic, Script::Latin) => {
            transliterator.transliterate_simple(
                text, Script::Arabic, TransliterationScheme::BgnPcgn
            )
        },
        _ => text.to_string(),
    }
}
```

### Name Matching

Cross-cultural name standardization:

```rust,ignore
fn match_names(
    name1: &str,
    name2: &str,
    transliterator: &TransliterationSystem,
    threshold: f32,
) -> bool {
    // Generate all possible transliterations
    let variants1 = generate_name_variants(name1, transliterator);
    let variants2 = generate_name_variants(name2, transliterator);
    
    // Find best match
    let mut best_similarity = 0.0;
    for v1 in &variants1 {
        for v2 in &variants2 {
            let similarity = string_similarity(v1, v2);
            if similarity > best_similarity {
                best_similarity = similarity;
            }
        }
    }
    
    best_similarity >= threshold
}
```

### Geographic Information Systems

Standardize place names:

```rust,ignore
fn standardize_place_names(
    places: &[PlaceName],
    transliterator: &TransliterationSystem,
) -> Vec<StandardizedPlace> {
    places.iter().map(|place| {
        let standard_name = match place.script {
            Script::Cyrillic => transliterator.transliterate_simple(
                &place.name, Script::Cyrillic, TransliterationScheme::BgnPcgn
            ),
            Script::Arabic => transliterator.transliterate_simple(
                &place.name, Script::Arabic, TransliterationScheme::BgnPcgn
            ),
            _ => place.name.clone(),
        };
        
        StandardizedPlace {
            original_name: place.name.clone(),
            standard_name,
            coordinates: place.coordinates,
            country: place.country.clone(),
        }
    }).collect()
}
```

### Challenges and Solutions

### One-to-Many Mappings

Some characters have multiple valid transliterations:

```rust,ignore
// Arabic ج can be transliterated as:
// - 'j' (common usage)
// - 'g' (Egyptian pronunciation)
// - 'zh' (Persian influence)

struct MultipleMapping {
    source: char,
    targets: Vec<(String, f32)>,  // (transliteration, probability)
}

fn handle_multiple_mappings(
    char: char,
    context: &Context,
    language_variety: LanguageVariety,
) -> String {
    match (char, language_variety) {
        ('ج', LanguageVariety::Egyptian) => "g".to_string(),
        ('ج', LanguageVariety::Persian) => "zh".to_string(),
        ('ج', _) => "j".to_string(),  // default
    }
}
```

### Diacritical Marks

Handle optional diacritics:

```rust,ignore
enum DiacriticPolicy {
    Preserve,   // Keep all diacritics
    Simplified, // Remove or simplify
    Contextual, // Based on target system
}

fn handle_diacritics(
    text: &str,
    policy: DiacriticPolicy,
    target_system: TargetSystem,
) -> String {
    match policy {
        DiacriticPolicy::Preserve => text.to_string(),
        DiacriticPolicy::Simplified => remove_diacritics(text),
        DiacriticPolicy::Contextual => {
            if target_system.supports_diacritics() {
                text.to_string()
            } else {
                remove_diacritics(text)
            }
        }
    }
}
```

### Ambiguous Sequences

Handle ambiguous character combinations:

```rust,ignore
// Russian "сх" could be:
// - с + х -> "skh" 
// - digraph -> "sh" (rare)

fn resolve_ambiguity(
    sequence: &str,
    context: &LinguisticContext,
) -> TransliterationChoice {
    let frequency_score = get_sequence_frequency(sequence, context.language);
    let phonological_score = get_phonological_likelihood(sequence);
    let contextual_score = get_contextual_fit(sequence, context);
    
    let total_score = frequency_score + phonological_score + contextual_score;
    
    if total_score > AMBIGUITY_THRESHOLD {
        TransliterationChoice::Confident(get_best_transliteration(sequence))
    } else {
        TransliterationChoice::Ambiguous(get_all_possibilities(sequence))
    }
}
```

### Quality Assurance

### Validation Framework

Ensure transliteration quality:

```rust,ignore
fn validate_transliteration_rules(
    rules: &[TransliterationRule],
) -> ValidationReport {
    let mut issues = Vec::new();
    
    // Check for conflicts
    for (i, rule1) in rules.iter().enumerate() {
        for rule2 in rules.iter().skip(i + 1) {
            if rule1.source == rule2.source && rule1.target != rule2.target {
                issues.push(ValidationIssue::Conflict {
                    source: rule1.source.clone(),
                    targets: vec![rule1.target.clone(), rule2.target.clone()],
                });
            }
        }
    }
    
    // Check for missing mappings
    let covered_chars: HashSet<char> = rules.iter()
        .flat_map(|rule| rule.source.chars())
        .collect();
    
    for expected_char in get_expected_character_set() {
        if !covered_chars.contains(&expected_char) {
            issues.push(ValidationIssue::MissingMapping {
                character: expected_char,
            });
        }
    }
    
    ValidationReport::new(issues)
}
```

### Round-Trip Testing

Test bidirectional consistency:

```rust,ignore
fn test_round_trip_consistency(
    transliterator: &TransliterationSystem,
    test_words: &[String],
) -> ConsistencyReport {
    let mut inconsistencies = Vec::new();
    
    for word in test_words {
        // Forward transliteration
        let transliterated = transliterator.transliterate_simple(
            word, Script::Cyrillic, TransliterationScheme::BgnPcgn
        );
        
        // Reverse transliteration
        let back_transliterated = transliterator.reverse_transliterate(
            &transliterated, Script::Cyrillic, TransliterationScheme::BgnPcgn
        );
        
        if word != &back_transliterated {
            inconsistencies.push(RoundTripError {
                original: word.clone(),
                forward: transliterated,
                backward: back_transliterated,
            });
        }
    }
    
    ConsistencyReport::new(inconsistencies)
}
```

## Performance Optimization

### Efficient Pattern Matching

Optimize for large-scale processing:

```rust,ignore
struct OptimizedTransliterator {
    trie: AhoCorasick,           // Multi-pattern matching
    longest_rule: usize,         // Maximum rule length
    char_map: HashMap<char, String>, // Single-character rules
}

impl OptimizedTransliterator {
    fn transliterate_fast(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len() * 2);
        let mut i = 0;
        let chars: Vec<char> = text.chars().collect();
        
        while i < chars.len() {
            let mut matched = false;
            
            // Try longest matches first
            for len in (1..=self.longest_rule.min(chars.len() - i)).rev() {
                let substring: String = chars[i..i+len].iter().collect();
                if let Some(replacement) = self.get_replacement(&substring) {
                    result.push_str(replacement);
                    i += len;
                    matched = true;
                    break;
                }
            }
            
            if !matched {
                result.push(chars[i]);
                i += 1;
            }
        }
        
        result
    }
}
```

### Caching Strategies

Cache frequent transliterations:

```rust,ignore
struct CachedTransliterator {
    transliterator: TransliterationSystem,
    word_cache: LruCache<(String, Script, TransliterationScheme), String>,
    phrase_cache: LruCache<String, String>,
}

impl CachedTransliterator {
    fn transliterate_cached(
        &mut self,
        text: &str,
        script: Script,
        scheme: TransliterationScheme,
    ) -> String {
        let cache_key = (text.to_string(), script, scheme);
        
        if let Some(cached) = self.word_cache.get(&cache_key) {
            return cached.clone();
        }
        
        let result = self.transliterator.transliterate_simple(text, script, scheme);
        self.word_cache.put(cache_key, result.clone());
        
        result
    }
}
```

### Standards Compliance

### International Standards

Implement official transliteration standards:

**BGN/PCGN Standards** cover geographic name transliteration, official government usage, and international diplomatic correspondence. **ISO Standards** include ISO 9:1995 (Cyrillic), ISO 233:1984 (Arabic), and ISO 843:1997 (Greek). **Library Standards** encompass ALA-LC (American Library Association), used in bibliographic systems and academic and research applications.

### Quality Metrics

Measure transliteration quality:

```rust,ignore
struct QualityMetrics {
    accuracy: f32,           // Correct vs total
    consistency: f32,        // Same input -> same output
    coverage: f32,           // Handled vs total characters
    reversibility: f32,      // Round-trip success rate
}

fn calculate_quality_metrics(
    transliterator: &TransliterationSystem,
    test_set: &TestSet,
) -> QualityMetrics {
    let mut correct = 0;
    let mut total = 0;
    let mut reversible = 0;
    
    for test_case in &test_set.cases {
        total += 1;
        
        let result = transliterator.transliterate_simple(
            &test_case.input,
            test_case.script,
            test_case.scheme,
        );
        
        if result == test_case.expected_output {
            correct += 1;
        }
        
        // Test reversibility
        let back_result = transliterator.reverse_transliterate(
            &result,
            test_case.script,
            test_case.scheme,
        );
        
        if back_result == test_case.input {
            reversible += 1;
        }
    }
    
    QualityMetrics {
        accuracy: correct as f32 / total as f32,
        consistency: calculate_consistency(transliterator, test_set),
        coverage: calculate_coverage(transliterator, test_set),
        reversibility: reversible as f32 / total as f32,
    }
}
```

### Neural Transliteration

Integrate machine learning approaches:

```rust,ignore
struct NeuralTransliterator {
    encoder: TransformerEncoder,
    decoder: TransformerDecoder,
    attention: CrossAttention,
    character_embeddings: Embedding,
}

impl NeuralTransliterator {
    fn transliterate_neural(&self, text: &str) -> String {
        let char_sequence = text.chars().collect::<Vec<_>>();
        let embeddings = self.character_embeddings.encode(&char_sequence);
        let encoded = self.encoder.forward(embeddings);
        let decoded = self.decoder.forward(encoded);
        self.decode_to_string(decoded)
    }
}
```

### Context-Aware Processing

Use surrounding text for better accuracy:

```rust,ignore
fn contextual_transliteration(
    word: &str,
    context: &SentenceContext,
    transliterator: &ContextualTransliterator,
) -> String {
    let language_hints = extract_language_hints(context);
    let domain_hints = extract_domain_hints(context);
    let style_hints = extract_style_hints(context);
    
    transliterator.transliterate_with_context(
        word,
        &ContextualHints {
            language: language_hints,
            domain: domain_hints,
            style: style_hints,
        }
    )
}
```

### Real-Time Processing

Streaming transliteration for live applications:

```rust,ignore
struct StreamingTransliterator {
    buffer: CharBuffer,
    state_machine: TransliterationStateMachine,
    output_queue: VecDeque<char>,
}

impl StreamingTransliterator {
    fn process_char(&mut self, ch: char) -> Option<String> {
        self.buffer.push(ch);
        
        if let Some(sequence) = self.buffer.get_longest_match() {
            let transliterated = self.state_machine.process(sequence);
            self.buffer.consume(sequence.len());
            Some(transliterated)
        } else {
            None
        }
    }
}
```

## Related Examples

This transliteration example connects with **[Morphological Analyzer](../linguistic-applications/morphological_analyzer.md)** for cross-script morphological analysis, **[Pronunciation Lexicon](pronunciation_lexicon.md)** for phonetic transliteration, **[Edit Distance](../text-processing/edit_distance.md)** for fuzzy transliteration matching, and **[Spell Checking](../text-processing/spell_checking.md)** for transliteration error correction.


## References

### Standards Organizations
Relevant standards organizations include **BGN/PCGN** (Board on Geographic Names / Permanent Committee on Geographical Names), **ISO** (International Organization for Standardization), and **ALA-LC** (American Library Association / Library of Congress).

### Academic Sources
Key academic sources include Wellisch, H. (1978) "The Conversion of Scripts", Sproat, R. (2000) "A Computational Theory of Writing Systems", and Hermjakob, U. et al. (2018) "Transliteration at Scale".

### Software Systems
Relevant software systems include **ICU** (International Components for Unicode), **Transliterator.js** (JavaScript transliteration library), and **polyglot** (Python transliteration toolkit).