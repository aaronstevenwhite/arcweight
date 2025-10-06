# Phonological Rules

This example demonstrates modeling phonological rule systems using FSTs to capture complex sound changes and their interactions in natural language.

## Overview

Every language has unconsciously applied sound rules. When English speakers say "cats" \[kæts\] but "dogs" \[dɔgz\], they're following phonological rules determining when plural -s is voiced. FSTs provide a mathematically precise yet linguistically intuitive way to model these patterns.

Phonological rules transform abstract mental representations ("underlying forms") into actual pronunciations ("surface forms"). The challenge lies in rule interactions: ordering affects outcomes, opacity hides intermediate steps, context sensitivity limits application environments, and variation differs across dialects.

FSTs handle these complexities through composition, where complex behaviors emerge from combining simple transducers. This approach, based on Kaplan and Kay's groundbreaking work, demonstrates how FST composition naturally models rule ordering effects that have puzzled linguists for decades.

## Quick Start

```bash
cargo run --example phonological_rules
```

## What You'll Learn

- **Phonological Features**: Understanding sound systems through feature-based representations
- **Rule Types**: Vowel harmony, consonant assimilation, deletion, and insertion processes  
- **FST Composition**: Modeling rule interactions through transducer composition
- **Rule Ordering**: How different application orders create different outcomes
- **Context Sensitivity**: Rules that apply only in specific phonological environments
- **Cross-Linguistic Patterns**: Examples from Turkish, Finnish, German, and English

## Core Concepts

### Understanding Phonological Features

Before we can model how sounds change, we need to understand what makes sounds different from each other. Linguists discovered that sounds aren't atomic units—they're bundles of features that can be independently manipulated by phonological rules.

Think of features like the controls on a mixing board: you can adjust the bass (voicing), the treble (place of articulation), or the reverb (nasality) independently. This feature-based representation explains why certain sounds pattern together in rules across languages.

The system uses a rich feature representation for phonological segments:

Vowel features include height (High, Mid, Low), backness (Front, Central, Back), and rounding (Rounded, Unrounded). Consonant features include voicing (Voiced, Voiceless), place (Labial, Coronal, Dorsal), and manner (Stop, Fricative, Nasal, Liquid). Prosodic features include stress (Stressed, Unstressed) and boundaries (Word boundary, Syllable boundary).

### Rule Types Implemented

This tutorial implements four classic phonological processes that appear across many of the world's languages. Each demonstrates different aspects of how FSTs model sound changes. **Vowel Harmony** is a long-distance process where vowels share features, found in Turkish, Finnish, Hungarian, and many African languages. FSTs naturally handle these unbounded dependencies. **Consonant Cluster Simplification** uses deletion rules that avoid difficult sound sequences, found in English dialects, child speech, and historical sound changes. FSTs make context-sensitive deletion straightforward. **Vowel Epenthesis** employs insertion rules that break up consonant clusters, found in Japanese loanwords, Arabic dialects, and Hindi-English code-switching. FSTs handle insertion between specific contexts elegantly. **Final Devoicing** neutralizes contrasts in specific positions, found in German, Dutch, Russian, and Catalan. FSTs naturally express position-sensitive rules.

## Implementation

### 1. Vowel Harmony: Long-Distance Feature Agreement

Vowel harmony is one of the most fascinating phonological phenomena, where vowels in a word must agree in certain features. It's like a linguistic version of color coordination—once you pick a color scheme (front or back vowels), you must stick with it throughout the word.

In Turkish, which we model here, all vowels in a word must agree in backness. This affects how suffixes are pronounced with suffix vowels becoming back after back vowels (a, ı, o, u) and front after front vowels (e, i, ö, ü).

**Rule Formalization:** `E → {a/e} depending on stem vowel backness`

The abstract vowel 'E' represents an underspecified vowel that gets its backness feature from the stem. This is similar to how the English plural suffix is underspecified for voicing and gets it from the preceding sound.

**Implementation:**
```rust,ignore
fn build_vowel_harmony_fst() -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    let back_state = fst.add_state();    // After back vowel
    let front_state = fst.add_state();   // After front vowel
    
    // Back vowels trigger back harmony: E $\to$ a
    fst.add_arc(back_state, Arc::new(
        'E' as u32, 'a' as u32, 
        TropicalWeight::one(), back_state
    ));
    
    // Front vowels trigger front harmony: E $\to$ e  
    fst.add_arc(front_state, Arc::new(
        'E' as u32, 'e' as u32,
        TropicalWeight::one(), front_state
    ));
}
```

**Examples:**
```text
kitabE  → kitaba  (back vowel context)
evE     → eve     (front vowel context)
adamE   → adama   (back vowel context)
```

### 2. Consonant Cluster Simplification: Making Speech Easier

Languages tend to avoid difficult consonant sequences, and cluster simplification is one way they deal with this challenge. When two consonants that are hard to pronounce together meet, one often gets deleted.

This process is incredibly common in casual English where "exactly" → \[ɪˈzækli\] (dropping the \[t\]), in AAVE where "desk" → \[dɛs\] (dropping the \[k\]), and historically where Latin "noctem" → Spanish "noche" (cluster simplified).

**Rule Formalization:**
```text
k → ∅ / __t
```

This rule deletes /k/ in the environment before /t/, representing a common cluster simplification pattern.

**Implementation:**
```rust,ignore
fn build_cluster_simplification_fst() -> VectorFst<TropicalWeight> {
    // Delete /k/ when followed by /t/
    fst.add_arc(start, Arc::new(
        'k' as u32, 0,  // epsilon output (deletion)
        TropicalWeight::one(), k_state
    ));
    
    fst.add_arc(k_state, Arc::new(
        't' as u32, 't' as u32,
        TropicalWeight::one(), start
    ));
}
```

**Examples:**
```text
akt     → at      (cluster simplified)
ekte    → ete     (cluster simplified)  
doktor  → dotor   (cluster simplified)
```

### 3. Vowel Epenthesis: Breaking Up the Cluster Party

While some languages delete consonants to avoid clusters, others insert vowels to break them up. This process, called epenthesis, is particularly common when borrowing words from other languages.

Real-world examples include Japanese borrowings where "strike" → \[sutoraiku\] (vowels inserted after each consonant), Hindi-English where "school" → \[iskuːl\] (initial vowel inserted), and Spanish dialectal where "psicología" → \[pisikoloˈxia\] (vowel breaks initial cluster).

**Rule Formalization:** `∅ → i / C_C` (insert /i/ between consonants)

The rule inserts an epenthetic vowel (here, /i/) whenever two consonants would otherwise be adjacent. The choice of /i/ is language-specific—Japanese uses \[u\], Hindi often uses \[i\] or \[ə\], and other languages make different choices.

**Implementation:**
```rust,ignore
fn build_epenthesis_fst() -> VectorFst<TropicalWeight> {
    // After first consonant, insert /i/ before second consonant
    for ch in consonants.chars() {
        fst.add_arc(consonant_state, Arc::new(
            ch as u32, 'i' as u32,  // insert epenthetic vowel
            TropicalWeight::one(), start
        ));
        
        // Then output the consonant
        fst.add_arc(start, Arc::new(
            0, ch as u32,  // epsilon input, output consonant
            TropicalWeight::one(), consonant_state
        ));
    }
}
```

**Examples:**
```text
sport   → sipórt  (cluster broken)
program → pirogirám (clusters broken)
strong  → sitirong (clusters broken)
```

### 4. Final Devoicing

Final devoicing represents positional neutralization wherein the voicing contrast in obstruents is suspended word-finally. This process is widely attested in Germanic languages.

German alternations:
- "Hund" \[hunt\] 'dog' ~ "Hunde" \[ˈhʊndə\] 'dogs'
- "Tag" \[taːk\] 'day' ~ "Tage" \[ˈtaːgə\] 'days'
- "lieb" \[liːp\] 'dear' ~ "liebe" \[ˈliːbə\] 'dear.INFL'

**Rule Formalization:**
```text
[+voice, -sonorant] → [-voice] / __#
```

This rule devoices all obstruents ([-sonorant]) in word-final position. The process affects: /b/→[p], /d/→[t], /g/→[k], /v/→[f], /z/→[s]. German orthography maintains underlying voicing, creating surface-underlying mismatches.

**Implementation:**
```rust,ignore
fn build_final_devoicing_fst() -> VectorFst<TropicalWeight> {
    let voiced_obstruents = ['b', 'd', 'g', 'z', 'v'];
    let voiceless_obstruents = ['p', 't', 'k', 's', 'f'];
    
    // Map voiced to voiceless at word end
    for (&voiced, &voiceless) in voiced_obstruents.iter()
                                 .zip(voiceless_obstruents.iter()) {
        fst.add_arc(start, Arc::new(
            voiced as u32, voiceless as u32,
            TropicalWeight::one(), voiced_state
        ));
    }
}
```

**Examples:**
```text
hund → hunt  (final /d/ devoiced)
tag  → tak   (final /g/ devoiced)  
lieb → liep  (final /b/ devoiced)
```

## FST Implementation

### Rule Ordering and Interaction

Phonological theory recognizes that rules apply in ordered sequences, producing different outputs based on ordering relationships. Finite state transducer composition models this through cascaded application.

Rule interaction types:
- **Feeding**: Rule A creates the structural description for Rule B
- **Bleeding**: Rule A removes the structural description for Rule B
- **Counterfeeding**: Potential feeding relationship blocked by ordering
- **Counterbleeding**: Potential bleeding relationship blocked by ordering

### Sequential Rule Application

Transducer composition implements ordered rule application:

```rust,ignore
fn apply_phonological_rules(
    input: &str, 
    rules: Vec<VectorFst<TropicalWeight>>
) -> Result<String> {
    let mut current_fst = build_word_fst(input);
    
    for (i, rule) in rules.iter().enumerate() {
        println!("Step {}: Applying rule {}", i + 1, i + 1);
        current_fst = compose_default(&current_fst, rule)?;
    }
    
    extract_output_string(&current_fst)
```

### Ordering Effects

Rule ordering produces distinct derivations, supporting serial models of phonology over parallel constraint-based approaches. Consider the following derivation:

**Input:** `aktE` (a hypothetical word with a cluster and a harmony-triggering vowel)

**Order 1: Harmony First, Then Cluster Simplification**
```text
aktE → (harmony looks at 'a', sees back vowel) → akta
     → (cluster simplification deletes 'k') → ata
Final output: ata
```text

**Order 2: Cluster Simplification First, Then Harmony**  
```text
aktE → (cluster simplification deletes 'k') → atE
     → (harmony looks at 'a', sees back vowel) → ata
Final output: ata (same result!)
```

This example demonstrates convergent derivations. However, other rule combinations yield order-dependent outputs, accounting for synchronic irregularities arising from diachronic rule reordering.

## Execution

```bash
cargo run --example phonological_rules
```

### Sample Output

```text
Turkish-style Vowel Harmony
------------------------------
Rule: Suffix vowel 'E' harmonizes with stem vowels
  'kitabE' → 'kitaba'  (back harmony)
  'evE' → 'eve'        (front harmony)
  'adamE' → 'adama'    (back harmony)

Consonant Cluster Simplification
-----------------------------------
Rule: /kt/ → /t/ (cluster reduction)
  'akt' → 'at'         (cluster simplified)
  'doktor' → 'dotor'   (cluster simplified)

Vowel Epenthesis
------------------
Rule: Insert 'i' between consonant clusters
  'sport' → 'siport'   (cluster broken)
  'program' → 'pirogram' (clusters broken)

Final Devoicing (German-style)
---------------------------------
Rule: Voiced obstruents become voiceless word-finally
  'hund' → 'hunt'      (final devoicing)
  'lieb' → 'liep'      (final devoicing)
```

## Advanced Features

### Rule Composition

Multiple phonological rules are modeled through transducer composition:

```rust,ignore
let complex_word = "sportE";
let all_rules = vec![
    epenthesis_fst,    // Break consonant clusters first
    harmony_fst,       // Then apply vowel harmony  
    devoicing_fst,     // Finally apply final devoicing
];

let result = apply_phonological_rules(complex_word, all_rules)?;
// Result: "sipórte" → "sipórte" → "sipórt"
```

### Context-Sensitive Application

Finite state transducers encode phonological environments:

```rust,ignore
// Rule: k → ∅ / _t (only before /t/)
fst.add_arc(start, Arc::new('k' as u32, 0, weight, k_state));
fst.add_arc(k_state, Arc::new('t' as u32, 't' as u32, weight, start));

// Rule: k → k / _V (preserved before vowels)  
for vowel in "aeiou".chars() {
    fst.add_arc(k_state, Arc::new(
        vowel as u32, 'k' as u32,  // output the held k
        weight, start
    ));
}
```

### Weighted Transducers

Variable rule application is modeled through weight assignment:

```rust,ignore
// Preferred rule (lower cost)
fst.add_arc(state1, Arc::new(
    input, output, 
    TropicalWeight::new(1.0),  // Low cost = preferred
    next_state
));

// Dispreferred rule (higher cost)
fst.add_arc(state1, Arc::new(
    input, alt_output,
    TropicalWeight::new(3.0),  // High cost = dispreferred  
    next_state
));
```

### Transducer Construction

Phonological rules are implemented as individual transducers:

```rust,ignore
fn build_rule_fst(rule_type: RuleType) -> VectorFst<TropicalWeight> {
    match rule_type {
        RuleType::VowelHarmony => build_vowel_harmony_fst(),
        RuleType::ClusterSimplification => build_cluster_simplification_fst(),
        RuleType::Epenthesis => build_epenthesis_fst(),
        RuleType::FinalDevoicing => build_final_devoicing_fst(),
    }
}
```

### Path Extraction

Output forms are recovered through path traversal:

```rust,ignore
fn extract_output_string(fst: &VectorFst<TropicalWeight>) -> Option<String> {
    if let Some(start) = fst.start() {
        let mut result = String::new();
        let mut current = start;
        
        loop {
            if fst.is_final(current) {
                return Some(result);
            }
            
            if let Some(arc) = fst.arcs(current).next() {
                if arc.olabel != 0 {
                    result.push(arc.olabel as u8 as char);
                }
                current = arc.nextstate;
            } else {
                break;
            }
        }
    }
    None
}
```

## Applications

Phonological rule systems have both theoretical and practical applications in computational linguistics and speech technology.

### Speech Recognition

Phonological variation modeling improves automatic speech recognition:

```text
Lexical: /ˈæktər/
Surface: \[ˈætər\] (with cluster simplification)
ASR: Must handle both forms
```text

### Text-to-Speech Synthesis

Phonological rules generate contextually appropriate surface forms:

```text
Orthography: "actor"
Lexical: /ˈæktər/  
Surface: \[ˈætər\] (apply phonological rules)
Speech: Generate audio for \[ˈætər\]
```

### Historical Sound Change

Diachronic phonological processes are modeled as rule sequences:

```text
Proto-Germanic: *hund-
Old High German: hunt (final devoicing applied)
Modern German: Hund (final devoicing still active)
```

### Language Documentation

Formal specification of phonological systems:

```text
Language X:
Vowel harmony: \[±back\] spreading
Cluster constraints: No more than 2 consonants
Final neutralization: No voiced obstruents word-finally
```

### Additional Rule Types

**Metathesis**: Segment reordering (e.g., /ask/ → [æks])

**Chain Shifts**: Systematic vowel movements (e.g., Great Vowel Shift)

**Tone Sandhi**: Tonal alternations in context

### Prosodic Phenomena

**Stress Assignment**: Metrical structure and stress placement algorithms

**Syllabification**: Onset maximization and coda constraints

### Constraint-Based Approaches

Optimality Theory implementation via weighted constraints:

```rust,ignore
struct OTConstraint {
    name: String,
    violation_cost: f32,
    constraint_fn: fn(&str) -> usize,  // Count violations
}

fn optimality_theory_fst(
    constraints: &[OTConstraint]
) -> VectorFst<TropicalWeight> {
    // Build FST where path costs = constraint violations
}
```

## Related Examples

- **[Morphological Analyzer](morphological_analyzer.md)**: Morphophonological alternations
- **[Pronunciation Lexicon](../practical-applications/pronunciation_lexicon.md)**: Phonetic representation systems
- **[Transliteration](../practical-applications/transliteration.md)**: Cross-linguistic sound mappings
- **[Edit Distance](../text-processing/edit_distance.md)**: Phonological similarity computation

## Performance Considerations

### Optimization Strategies

- **Minimization**: Reduce transducer state count
- **Determinization**: Ensure unique path outputs
- **Composition Caching**: Memoize rule combinations
- **Lazy Evaluation**: On-demand computation

### Scalability Considerations

- Parallel composition for large rule sets
- Streaming architecture for continuous processing
- Incremental rule updates
- Distributed deployment options

## References

### Theoretical Foundations

- Johnson, C. D. (1972). Formal aspects of phonological description
- {{#cite koskenniemi1983two}}
- {{#cite kaplan1994regular}}

### Computational Implementations

- {{#cite mohri1997finite}}
- {{#cite beesley2003finite}}
- Roark, B., & Sproat, R. (2007). Computational approaches to morphology and syntax

### Software Tools

- XFST: Xerox Finite State Tools
- HFST: Helsinki Finite State Technology
- Foma: Open-source finite state compiler
- Phonological CorpusTools: Quantitative phonological analysis