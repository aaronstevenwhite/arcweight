# Practical Applications

This section demonstrates finite state transducer applications in natural language processing systems, including text normalization, grapheme-to-phoneme conversion, and script transliteration.

## Examples

### [Number & Date Normalizer](number_date_normalizer.md)
Converts numeric and temporal expressions to standardized textual representations. The implementation handles various numeric formats and date/time conventions.

```bash
cargo run --example number_date_normalizer
```

### [Pronunciation Lexicon](pronunciation_lexicon.md)
Implements grapheme-to-phoneme conversion using finite state transducers. The system combines lexicon lookup with rule-based pronunciation generation.

```bash
cargo run --example pronunciation_lexicon
```

### [Transliteration](transliteration.md)
Performs script conversion following international standards (BGN/PCGN, ISO). The implementation supports Cyrillic, Arabic, and Greek to Latin transliteration.

```bash
cargo run --example transliteration
```

## Application Domains

These implementations address common requirements in:
- Text normalization for consistent data representation
- Speech synthesis requiring accurate pronunciation models
- Cross-linguistic information retrieval and text processing

---

[← Examples Overview](../README.md)