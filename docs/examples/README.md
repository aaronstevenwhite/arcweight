# Examples

This chapter presents eight examples demonstrating finite state transducer applications across computational linguistics domains. The examples progress from fundamental string algorithms to advanced linguistic analysis systems.

## Organization

The examples are organized into three categories based on their application domain:
1. Text processing fundamentals
2. Practical applications
3. Linguistic applications

Each example includes both theoretical background and implementation details. Source code for all examples resides in the `examples/` directory.

## Available Examples

### Text Processing
- **[Edit Distance](text-processing/edit_distance.md)** - Levenshtein distance computation via weighted transducers
- **[String Alignment](text-processing/string_alignment.md)** - Optimal sequence alignment and path extraction  
- **[Spell Checking](text-processing/spell_checking.md)** - Error model composition with lexicon constraints

### Practical Applications
- **[Number & Date Normalizer](practical-applications/number_date_normalizer.md)** - Text normalization for temporal and numeric expressions
- **[Pronunciation Lexicon](practical-applications/pronunciation_lexicon.md)** - Grapheme-to-phoneme conversion using phonological rules
- **[Transliteration](practical-applications/transliteration.md)** - Script conversion following BGN/PCGN and ISO standards

### Linguistic Applications
- **[Morphological Analyzer](linguistic-applications/morphological_analyzer.md)** - Two-level morphological analysis and synthesis
- **[Phonological Rules](linguistic-applications/phonological_rules.md)** - Phonological rule systems and sound change modeling

The examples build upon each other conceptually. The text processing examples establish fundamental FST operations, while the practical and linguistic applications demonstrate domain-specific implementations.


## Core Concepts Demonstrated

Each example illustrates specific finite state transducer concepts:

- **Edit Distance**: Weighted transducer construction for string metrics
- **String Alignment**: Path extraction algorithms and optimal alignment computation  
- **Spell Checking**: Transducer composition for constraint satisfaction
- **Text Normalization**: Pattern-based transduction for format standardization
- **Pronunciation**: Cascaded transducers for grapheme-to-phoneme conversion
- **Transliteration**: Context-dependent rewrite rules for script conversion
- **Morphology**: Two-level morphological transducers following Koskenniemi (1983)
- **Phonological Rules**: Rule ordering and composition in phonological derivations

## Theoretical Background

These examples implement algorithms from computational linguistics literature, demonstrating the application of finite state methods to natural language processing tasks. The implementations follow established theoretical frameworks while leveraging modern computational techniques.

## Implementation Notes

The examples demonstrate various weight semirings and composition strategies. Users may modify parameters such as operation costs in edit distance computation or rule ordering in phonological systems to explore different theoretical models.

```rust,ignore
// Example: Modifying operation costs
let insertion_cost = 2.0;
let substitution_cost = 3.0;
```

For implementation details, refer to the source files in the `examples/` directory.

## Related Documentation

- **[Core Concepts](../core-concepts/)** - Mathematical foundations of finite state transducers
- **[Working with FSTs](../working-with-fsts/)** - Transducer operations and algorithms
- **[API Reference](https://docs.rs/arcweight)** - Library interface documentation

## Usage

To execute an example:
```bash
cargo run --example <example_name>
```

For instance:
```bash
cargo run --example edit_distance
cargo run --example morphological_analyzer
```