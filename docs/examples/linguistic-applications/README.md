# Linguistic Applications

This section presents finite state transducer implementations of theoretical frameworks in computational linguistics. The examples demonstrate morphological analysis following {{#cite koskenniemi1983two}} and phonological rule systems based on {{#cite kaplan1994regular}}.

## Examples

### [Morphological Analyzer](morphological_analyzer.md)
Implements two-level morphological analysis and synthesis for morphologically complex languages. The system handles morphotactics, morphophonological alternations, and bidirectional processing.

```bash
cargo run --example morphological_analyzer
```

### [Phonological Rules](phonological_rules.md)
Demonstrates phonological rule application using finite state transducer composition. The implementation supports ordered rule systems and morphophonological processes.

```bash
cargo run --example phonological_rules
```

## Theoretical Framework

The implementations follow established computational linguistics methodologies:
- Two-level morphology {{#cite koskenniemi1983two}} for morphophonological alternations
- Finite-state morphology {{#cite beesley2003finite}} for lexical transducers
- Regular models of phonological rules {{#cite kaplan1994regular}}
- Cascaded finite state transducers for complex linguistic phenomena

## Implementation Requirements

These examples assume familiarity with finite state transducer operations and linguistic theory. The text processing examples provide foundational concepts useful for understanding these implementations.

---

[← Examples Overview](../README.md)