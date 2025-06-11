# Finite State Morphology

This example demonstrates finite state morphological analysis following Lauri Karttunen's foundational work on computational morphology. It showcases two-level morphology principles, lexicon construction, and morphophonological alternations.

## Overview

The morphological analyzer implements:

- **Two-level morphology** (Koskenniemi & Karttunen)
- **LEXC approach** for lexicon compilation
- **Finnish morphological analysis** with rich case systems
- **English derivational morphology** with phonological rules
- **Bidirectional processing** (analysis ↔ generation)

## Academic Foundation

This implementation follows theoretical frameworks from:

- Karttunen, L. (1993). Finite-state lexicon compiler
- Karttunen, L. & Beesley, K. (2001). Finite State Morphology
- Koskenniemi, K. (1983). Two-level morphology
- Karttunen, L. (1994). Constructing lexical transducers

## Running the Example

```bash
cargo run --example morphological_analyzer
```

## Key Features Demonstrated

### Finnish Morphology

Classic examples from Karttunen's work on Finnish:

```
kala     → kala+∅     [fish.NOM.SG]
kalan    → kala+n     [fish.GEN.SG]
kalaa    → kala+a     [fish.PART.SG]
kalassa  → kala+ssa   [fish.INESS.SG]
kalasta  → kala+sta   [fish.ELAT.SG]
```

### English Derivational Morphology

Demonstrates morphophonological alternations:

```
worker    → work+er    [work.AGENT]
writer    → write+er   [write.AGENT] (e-deletion)
happiness → happy+ness [happy.ABSTR] (y→i alternation)
```

### Theoretical Framework

- **LEXC**: Lexicon compiler for morphotactics
- **TWOLC**: Two-level rule compiler
- **Compositional architecture**: Lexicon ∘ Phonology
- **Bidirectional processing**: Analysis ↔ Generation

## Code Structure

The example is organized into several key components:

### MorphCategory Enum

Defines morphological categories following Karttunen's taxonomy:

```rust
enum MorphCategory {
    // Major lexical categories
    Noun, Verb, Adjective, Adverb,
    
    // Finnish case system
    Nominative, Genitive, Partitive, Inessive, Elative,
    
    // Verbal features
    Present, Past, FirstPerson, SecondPerson, ThirdPerson,
    
    // Derivational features
    Agent, Abstract, Causative, Frequentative,
}
```

### FiniteStateLexicon

Core morphological analyzer implementing:

- Stem databases organized by category
- Affix inventories with morphological properties
- Morphophonological rule application
- Analysis and generation functions

### Key Functions

- `analyze()`: Surface form → lexical analysis
- `generate()`: Lexical form → surface realization
- `check_with_phonology()`: Apply two-level rules
- `build_morphotactic_fst()`: Demonstrate LEXC approach

## Applications

Karttunen's framework enables:

- Large-scale morphological analyzers
- Spell checkers with morphological awareness
- Machine translation for morphologically rich languages
- Information retrieval with morphological normalization
- Text generation with correct morphological forms

## Historical Impact

- **Xerox finite state tools** (1990s)
- **HFST**: Helsinki Finite State Technology
- **Foma**: Open-source implementation
- **Integration** into major NLP pipelines
- **Foundation** for modern morphological processing

## See Also

- [Phonological Rules Example](phonological_rules.md) - FST composition for phonological processes
- [Transliteration Example](transliteration.md) - Cross-script text conversion
- [API Documentation](https://docs.rs/arcweight) - Full API reference