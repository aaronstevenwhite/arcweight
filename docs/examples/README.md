# Examples

**Learn FSTs through practical applications** — 8 examples from beginner to expert level.

Each example demonstrates real-world FST usage with clear explanations and working code. Perfect for learning theoretical concepts through hands-on practice.

## Quick Start

```bash
# Run the recommended first example
cargo run --example edit_distance

# Try a real-world application
cargo run --example spell_checking

# View source code
cat examples/edit_distance.rs
```

## Available Examples

### Text Processing
- **[Edit Distance](text-processing/edit_distance.md)** - String similarity using FSTs
- **[String Alignment](text-processing/string_alignment.md)** - Sequence alignment visualization  
- **[Spell Checking](text-processing/spell_checking.md)** - Dictionary-based error correction

### Practical Applications
- **[Number & Date Normalizer](practical-applications/number_date_normalizer.md)** - Text normalization
- **[Pronunciation Lexicon](practical-applications/pronunciation_lexicon.md)** - Grapheme-to-phoneme conversion
- **[Transliteration](practical-applications/transliteration.md)** - Cross-script conversion

### Linguistic Applications
- **[Morphological Analyzer](linguistic-applications/morphological_analyzer.md)** - Word structure analysis
- **[Phonological Rules](linguistic-applications/phonological_rules.md)** - Sound change modeling

**Recommended path**: Start with Edit Distance → String Alignment → Spell Checking


## Key Concepts

**What you'll learn from each example:**

- **Edit Distance**: Dynamic programming with FSTs, cost optimization
- **String Alignment**: Path extraction, transformation visualization  
- **Spell Checking**: FST composition, dictionary integration
- **Text Normalization**: Pattern recognition, format conversion
- **Pronunciation**: Grapheme-to-phoneme mapping, phonological rules
- **Transliteration**: Cross-script conversion, Unicode handling
- **Morphology**: Word structure analysis, morpheme segmentation
- **Phonological Rules**: Sound change modeling, rule composition

## Background

These examples implement classic computational linguistics algorithms using modern FST techniques.

## Extending Examples

Modify examples to suit your needs:

```rust,ignore
// Change operation costs in edit_distance.rs
let insertion_cost = 2.0;
let substitution_cost = 3.0;

// Combine examples
let enhanced = compose_default(&edit_fst, &phonetic_fst)?;
```

**Contributing**: See the CONTRIBUTING.md file in the project root to add new examples.

## Resources

- **[Core Concepts](../core-concepts/)** - Mathematical foundations
- **[Working with FSTs](../working-with-fsts/)** - Essential operations
- **[API Reference](https://docs.rs/arcweight)** - Complete documentation

## Next Steps

**Start here**: [Edit Distance](text-processing/edit_distance.md) - Best introduction to FST patterns.

Then explore other examples based on your interests and build real applications!