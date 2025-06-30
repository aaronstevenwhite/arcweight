# ArcWeight Documentation

**High-performance weighted finite state transducers for Rust**

*Elegant • Efficient • Extensible*

[![Crates.io](https://img.shields.io/crates/v/arcweight.svg)](https://crates.io/crates/arcweight)
[![Documentation](https://docs.rs/arcweight/badge.svg)](https://docs.rs/arcweight)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/aaronstevenwhite/arcweight/blob/main/LICENSE)

This book contains comprehensive documentation for **[ArcWeight](https://github.com/aaronstevenwhite/arcweight)**, a high-performance Rust library for weighted finite state transducers (FSTs). ArcWeight provides a modular and efficient implementation of FST algorithms with extensive support for different semiring structures, making it suitable for applications in computational linguistics, natural language processing, and speech recognition.

## About Finite State Transducers

> **What are FSTs?** Finite state transducers are mathematical structures that map input sequences to output sequences through a finite set of states and weighted transitions.

FSTs provide an elegant framework for modeling sequential transformations and have found widespread applications in language processing tasks such as:

 
- **Spell checking** — Error detection and correction
- **Normalization** - Number and date normalization
- **Transliteration** — Conversion from one script to another


FSTs also support theoretical approaches in computational linguistics:

- **Phonological rule application** — Sound change modeling
- **Morphological analysis** — Word structure decomposition  


The **weighted** nature of FSTs allows for probabilistic modeling and optimization over multiple possible outputs, making them useful for handling uncertainty and ranking alternatives.

## Documentation Structure

### Getting Started
For newcomers and quick setup

- **[Installation](installation.md)** — Setup and dependencies
- **[Quick Start](quick-start.md)** — Your first FST in 15 minutes

### User Guide
In-depth coverage of FST operations

- **[Core Concepts](core-concepts/README.md)** — Mathematical foundations
- **[Working with FSTs](working-with-fsts/README.md)** — Comprehensive operation guide

### Examples & Tutorials
Learn through practical applications

- **[Edit Distance](examples/text-processing/edit_distance.md)** — String similarity with FSTs
- **[String Alignment](examples/text-processing/string_alignment.md)** — String alignment with FSTs
- **[Spell Checking](examples/text-processing/spell_checking.md)** — Building intelligent spell checkers


### Theory & Implementation
Deep dive into the mathematics and design

- **[Core Concepts](core-concepts/)** — Semirings and algorithms
- **[Architecture](architecture/)** — Implementation details
- **[API Reference](api-reference.md)** — Complete technical reference

### Performance & Help
Optimization and troubleshooting

- **[Benchmarks](benchmarks.md)** — Performance analysis
- **[FAQ](faq.md)** — Common questions and tips

## Practical Examples

> **Learn by doing!** Each example includes theory, working code, and real-world applications.

### Text Processing Fundamentals

- **[Edit Distance](examples/text-processing/edit_distance.md)** — Levenshtein distance with FSTs
- **[String Alignment](examples/text-processing/string_alignment.md)** — Visualize transformation sequences
- **[Spell Checking](examples/text-processing/spell_checking.md)** — Dictionary-based error correction

### Practical Applications

- **[Pronunciation Lexicon](examples/practical-applications/pronunciation_lexicon.md)** — Grapheme-to-phoneme conversion
- **[Number & Date Normalizer](examples/practical-applications/number_date_normalizer.md)** — Standardize textual formats
- **[Transliteration](examples/practical-applications/transliteration.md)** — Cross-script character mapping

### Natural Language Processing

- **[Morphological Analyzer](examples/linguistic-applications/morphological_analyzer.md)** — Word structure analysis
- **[Phonological Rules](examples/linguistic-applications/phonological_rules.md)** — Systematic sound changes

## Learning Paths

Choose your adventure based on your background and goals.

### For Beginners
*New to FSTs? Start with hands-on learning!*

1. **[Core Concepts](core-concepts/)** — Theoretical foundations
2. **[Quick Start](quick-start.md)** — Build first FST
3. **[Text Processing Fundamentals](examples/text-processing/)** — Real-world examples

**Perfect for**: Students, researchers new to FSTs, developers wanting practical understanding

### For Experienced Developers
*Ready for the technical deep-dive?*

1. **[Working with FSTs](working-with-fsts/README.md)** — In-depth guide to the package
2. **[Architecture](architecture/README.md)** — Implementation details
3. **[Benchmarks](benchmarks.md)** — Performance analysis
4. **[API Reference](api-reference.md)** — Complete technical docs


**Perfect for**: Systems engineers, library authors, performance-critical applications

> **Tip**: Examples are self-contained but build conceptually. Each includes runnable code, theory, and connections to related topics.