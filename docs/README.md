# ArcWeight Documentation

[![Crates.io](https://img.shields.io/crates/v/arcweight.svg)](https://crates.io/crates/arcweight)
[![Documentation](https://docs.rs/arcweight/badge.svg)](https://docs.rs/arcweight)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/aaronstevenwhite/arcweight/blob/main/LICENSE)

ArcWeight is a Rust library implementing weighted finite state transducers (FSTs) with support for multiple semiring structures. The library provides algorithms for FST construction, composition, and optimization, with applications in computational linguistics, natural language processing, and speech recognition.

## About Finite State Transducers

Finite state transducers are mathematical structures that map input sequences to output sequences through a finite set of states and weighted transitions.

Applications in language processing include:

- Spell checking and error correction
- Text normalization (numbers, dates)
- Script transliteration

Theoretical applications in computational linguistics include:

- Phonological rule modeling
- Morphological analysis

Weighted FSTs extend standard FSTs by associating weights with transitions, enabling probabilistic modeling and optimization over alternative outputs.

## Documentation Structure

### Getting Started

- [Installation](installation.md) — Setup and dependencies
- [Quick Start](quick-start.md) — Basic FST construction

### User Guide

- [Core Concepts](core-concepts/README.md) — Mathematical foundations
- [Working with FSTs](working-with-fsts/README.md) — FST operations

### Examples

- [Edit Distance](examples/text-processing/edit_distance.md) — Levenshtein distance implementation
- [String Alignment](examples/text-processing/string_alignment.md) — Sequence alignment algorithms
- [Spell Checking](examples/text-processing/spell_checking.md) — Dictionary-based correction

### Theory & Implementation

- [Core Concepts](core-concepts/) — Semiring theory and algorithms
- [Architecture](architecture/) — Library design and implementation
- [API Reference](api-reference.md) — Technical reference

### Performance

- [Benchmarks](benchmarks.md) — Performance metrics
- [FAQ](faq.md) — Common questions

## Example Applications

### Text Processing

- [Edit Distance](examples/text-processing/edit_distance.md) — Levenshtein distance computation
- [String Alignment](examples/text-processing/string_alignment.md) — Optimal sequence alignment
- [Spell Checking](examples/text-processing/spell_checking.md) — Error detection and correction

### Applied NLP

- [Pronunciation Lexicon](examples/practical-applications/pronunciation_lexicon.md) — Grapheme-to-phoneme mapping
- [Number & Date Normalizer](examples/practical-applications/number_date_normalizer.md) — Text normalization
- [Transliteration](examples/practical-applications/transliteration.md) — Script conversion

### Computational Linguistics

- [Morphological Analyzer](examples/linguistic-applications/morphological_analyzer.md) — Morphological parsing
- [Phonological Rules](examples/linguistic-applications/phonological_rules.md) — Rule-based phonology

## Reading Guide

### Introduction to FSTs

1. [Core Concepts](core-concepts/) — Mathematical foundations
2. [Quick Start](quick-start.md) — Basic usage
3. [Text Processing Examples](examples/text-processing/) — Applied examples

### Advanced Usage

1. [Working with FSTs](working-with-fsts/README.md) — Advanced operations
2. [Architecture](architecture/README.md) — Internal design
3. [Benchmarks](benchmarks.md) — Performance characteristics
4. [API Reference](api-reference.md) — Complete reference
