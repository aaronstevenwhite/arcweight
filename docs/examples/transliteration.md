# Transliteration FST Example

This example demonstrates how to build finite state transducers for transliteration between different writing systems. It showcases script conversion, handling of digraphs, context-sensitive rules, and multiple transliteration schemes.

## Overview

The example implements transliteration systems for:
1. **Cyrillic to Latin** (Russian → English) using BGN/PCGN standards
2. **Arabic to Latin** transliteration
3. **Greek to Latin** transliteration  
4. **Bidirectional transliteration** capabilities
5. **Multiple schemes** (BGN/PCGN, ISO, ALA-LC, Scientific, Popular)
6. **Context-sensitive rules** and digraph handling

## Key Features

### Transliteration Schemes

The example supports multiple standardized transliteration schemes:

- **BGN/PCGN**: Board on Geographic Names / Permanent Committee on Geographical Names
- **ISO**: ISO transliteration standards
- **ALA-LC**: American Library Association / Library of Congress
- **Scientific**: Scientific transliteration
- **Popular**: Simplified transliteration for general use

### Script Support

Implemented script conversions:
- Cyrillic ↔ Latin
- Arabic → Latin  
- Greek → Latin
- Hebrew (framework provided)

## Implementation Details

### Transliteration Rule Structure

```rust
struct TransliterationRule {
    source: String,      // Source character/sequence
    target: String,      // Target character/sequence
    scheme: TransliterationScheme,
    context_before: Option<String>,  // Left context
    context_after: Option<String>,   // Right context
}
```

### Core Functions

#### Basic Transliteration FST
```rust
fn build_transliteration_fst(
    rules: &[TransliterationRule],
    scheme: TransliterationScheme,
) -> VectorFst<TropicalWeight>
```

Creates an FST that applies transliteration rules for a specific scheme.

#### Cyrillic to Latin
```rust
fn build_cyrillic_to_latin_fst(
    scheme: TransliterationScheme,
) -> VectorFst<TropicalWeight>
```

Implements Russian Cyrillic to Latin conversion following standard schemes.

#### Bidirectional System
```rust
fn build_bidirectional_fst(
    forward_rules: &[TransliterationRule],
    reverse_rules: &[TransliterationRule],
) -> (VectorFst<TropicalWeight>, VectorFst<TropicalWeight>)
```

Creates paired FSTs for forward and reverse transliteration.

### Character Mappings

#### Cyrillic to Latin (BGN/PCGN)
```
А/а → A/a    К/к → K/k    Х/х → Kh/kh
Б/б → B/b    Л/л → L/l    Ц/ц → Ts/ts
В/в → V/v    М/м → M/m    Ч/ч → Ch/ch
Г/г → G/g    Н/н → N/n    Ш/ш → Sh/sh
Д/д → D/d    О/о → O/o    Щ/щ → Shch/shch
Е/е → E/e    П/п → P/p    Ъ/ъ → "
Ё/ё → Yo/yo  Р/р → R/r    Ы/ы → Y/y
Ж/ж → Zh/zh  С/с → S/s    Ь/ь → '
З/з → Z/z    Т/т → T/t    Э/э → E/e
И/и → I/i    У/у → U/u    Ю/ю → Yu/yu
Й/й → Y/y    Ф/ф → F/f    Я/я → Ya/ya
```

#### Context-Sensitive Rules

Special handling for:
- **Soft/hard signs**: Ъ → ", Ь → '
- **Palatalization**: Before certain vowels
- **Digraphs**: кх → kh, тс → ts, etc.
- **Case preservation**: Maintains original capitalization

### Arabic to Latin

Implements standard Arabic romanization:
```
ا → a    ض → d    ق → q
ب → b    ط → t    ك → k  
ت → t    ظ → z    ل → l
ث → th   ع → '    م → m
ج → j    غ → gh   ن → n
ح → h    ف → f    ه → h
خ → kh   ق → q    و → w
د → d    ك → k    ي → y
ذ → dh   ل → l
ر → r    م → m
ز → z    ن → n
س → s    ه → h
ش → sh   و → w
ص → s    ي → y
```

## Usage

Run the example with:
```bash
cargo run --example transliteration
```

### Example Output

```
=== Transliteration System Demo ===

Cyrillic to Latin (BGN/PCGN):
  Москва → Moskva
  Санкт-Петербург → Sankt-Peterburg  
  Владивосток → Vladivostok
  Екатеринбург → Yekaterinburg

Greek to Latin:
  Αθήνα → Athina
  Θεσσαλονίκη → Thessaloniki
  Πειραιάς → Peiraias

Arabic to Latin:
  القاهرة → al-qahirah
  الرياض → ar-riyadh
  بغداد → baghdad

Bidirectional (Russian):
  Forward:  Россия → Rossiya
  Reverse:  Rossiya → Россия
```

## Advanced Features

### Context-Sensitive Transliteration

Handles complex rules that depend on surrounding characters:

```rust
fn build_context_sensitive_fst(
    rules: &[TransliterationRule],
) -> VectorFst<TropicalWeight> {
    // Implementation handles:
    // - Left context matching
    // - Right context matching  
    // - Multi-character sequences
    // - Priority ordering
}
```

### Multi-Character Mappings

Supports digraphs and trigraphs:
- щ → shch (Cyrillic)
- θ → th (Greek)
- خ → kh (Arabic)

### Quality Assessment

```rust
fn assess_transliteration_quality(
    original: &str,
    transliterated: &str,
    reverse_transliterated: &str,
) -> f32 {
    // Measures round-trip fidelity
    // Accounts for ambiguous mappings
    // Returns quality score 0.0-1.0
}
```

## Applications

### Geographic Information Systems
- Place name standardization
- Map labeling consistency
- Cross-language geographic search

### Digital Libraries
- Catalog standardization
- Cross-script search
- Bibliographic data normalization

### International Communication
- Name romanization
- Address standardization
- Document processing

### Machine Translation
- Preprocessing for translation systems
- Handling proper nouns
- Cross-script information retrieval

## Theoretical Background

### Unicode Normalization
The example properly handles:
- Unicode normalization forms (NFC, NFD)
- Combining character sequences
- Case folding and preservation

### Linguistic Considerations
- **Phonemic accuracy**: Preserving pronunciation
- **Orthographic conventions**: Following official standards
- **Reversibility**: When possible, enable round-trip conversion
- **Ambiguity handling**: Managing one-to-many mappings

### FST Design Patterns
- **Layered architecture**: Separate normalization, transliteration, post-processing
- **Composition**: Combine multiple transformation stages
- **Parameterization**: Support multiple schemes via configuration

## Performance Characteristics

### Memory Usage
- O(|alphabet|²) for simple character mappings
- O(|rules| × |max_context|) for context-sensitive rules
- Efficient trie structure for multi-character sequences

### Processing Speed
- Linear in input length for deterministic FSTs
- Constant-time character lookup after compilation
- Batch processing optimization for large texts

## Extending the System

### Adding New Scripts
```rust
fn build_script_fst(
    script: Script,
    target: Script,
    scheme: TransliterationScheme,
) -> Result<VectorFst<TropicalWeight>> {
    // Framework for adding new script pairs
}
```

### Custom Schemes
```rust
struct CustomScheme {
    rules: Vec<TransliterationRule>,
    name: String,
    source_script: Script,
    target_script: Script,
}
```

## Related Examples

- [Morphological Analyzer](morphological_analyzer.md) - Linguistic analysis with phonological rules
- [Phonological Rules](phonological_rules.md) - Sound change modeling
- [Pronunciation Lexicon](pronunciation_lexicon.md) - Phonetic transcription systems

## Standards and References

- **ISO 9**: Cyrillic transliteration standard
- **BGN/PCGN**: Geographic name romanization
- **ALA-LC**: Library cataloging standards
- **Unicode**: Character encoding and normalization
- **ICU**: International Components for Unicode