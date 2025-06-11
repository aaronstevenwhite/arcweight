# Phonological Rules with FSTs

This example demonstrates how to model phonological rule systems using Finite State Transducers, following the foundational work of Kaplan and Kay (1994) and subsequent developments in computational phonology.

## Overview

The example shows how FST composition can model complex phonological systems where multiple rules interact in ordered sequences, demonstrating:

- **Individual phonological processes** as FSTs
- **Rule composition** and interaction effects
- **Classic phonological phenomena**: vowel harmony, cluster simplification, epenthesis, final devoicing
- **Rule ordering effects** and how composition order affects output
- **Bidirectional processing** capabilities

## Academic Foundation

Based on seminal work in computational phonology:

- Kaplan, R. M., & Kay, M. (1994). Regular models of phonological rule systems
- Johnson, C. D. (1972). Formal aspects of phonological description
- Koskenniemi, K. (1983). Two-level morphology

## Running the Example

```bash
cargo run --example phonological_rules
```

## Key Phonological Processes

### 1. Turkish-style Vowel Harmony

Front/back vowel spreading where suffix vowels harmonize with stem vowels:

```
adamE → adama  (back harmony: E → a)
evE   → eve    (front harmony: E → e)
gelE  → gele   (front harmony: E → e)
```

### 2. Consonant Cluster Simplification

/kt/ → /t/ cluster reduction:

```
akt    → at
doktor → dotor
ekte   → ete
```

### 3. Vowel Epenthesis

Insert 'i' to break consonant clusters:

```
sport   → siori
program → piogiam
strong  → sironi
```

### 4. Final Devoicing (German-style)

Voiced obstruents become voiceless word-finally:

```
hund → hunt
tag  → tak
lieb → liep
```

## Rule Interaction and Ordering

The example demonstrates how rule ordering affects output through FST composition:

```
Input: aktE

Order 1: Vowel Harmony → Cluster Simplification
  aktE → akta → ata

Order 2: Cluster Simplification → Vowel Harmony  
  aktE → (no output)
```

This shows **feeding** vs. **bleeding** interactions in phonological rule systems.

## Code Structure

### Core Functions

- `build_vowel_harmony_fst()`: Turkish-style harmony system
- `build_cluster_simplification_fst()`: Consonant cluster reduction
- `build_epenthesis_fst()`: Vowel insertion for cluster breaking
- `build_final_devoicing_fst()`: Word-final obstruent devoicing
- `apply_phonological_rules()`: Sequential rule application via composition

### FST Architecture

Each phonological process is modeled as a separate FST:

```rust
fn build_vowel_harmony_fst() -> VectorFst<TropicalWeight> {
    // State-based harmony tracking
    let start = fst.add_state();
    let back_state = fst.add_state();
    let front_state = fst.add_state();
    
    // Back vowels trigger back harmony state
    // Front vowels trigger front harmony state
    // Harmonizing vowel 'E' realizes according to context
}
```

### Rule Composition

Rules are composed sequentially to model rule interaction:

```rust
fn apply_phonological_rules(
    input: &str,
    rules: Vec<VectorFst<TropicalWeight>>,
) -> Result<String> {
    let mut current_fst = build_word_fst(input);
    
    for rule in rules.iter() {
        current_fst = compose_default(&current_fst, rule)?;
    }
    
    extract_output_string(&current_fst)
}
```

## Theoretical Implications

FST composition models key insights from Kaplan & Kay:

- **Phonological rules as regular relations**
- **Rule application through FST composition**
- **Natural emergence of ordering effects**
- **Modeling of opacity, transparency, and rule interactions**
- **Bidirectional processing**: generation ↔ recognition
- **Connection to two-level morphology**

## Applications

This framework enables:

- **Morphophonological analysis and generation**
- **Text-to-speech synthesis systems**
- **Automatic speech recognition**
- **Historical linguistics** and sound change modeling
- **Language documentation** and endangered language preservation
- **Cross-linguistic phonological typology** studies
- **Psycholinguistic modeling** of phonological processing

## Historical Development

- **Johnson (1972)**: Early formal approaches to phonological rules
- **Koskenniemi (1983)**: Two-level morphology with FSTs
- **Kaplan & Kay (1994)**: Regular models of phonological rule systems
- **Modern applications**: Finite-state phonology in NLP systems

## See Also

- [Morphological Analyzer Example](morphological_analyzer.md) - Two-level morphology with phonological rules
- [Transliteration Example](transliteration.md) - Cross-script phonological mapping
- [FST Composition Documentation](../guide.md#composition) - Technical details on FST composition