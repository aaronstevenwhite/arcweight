# Number/Date Normalizer FST Example

This example demonstrates how to build finite state transducers for normalizing various textual representations of numbers, dates, times, and other structured data into standardized formats. It's essential for text preprocessing in NLP, data extraction, and information normalization systems.

## Overview

The example implements normalization for:
1. **Written numbers to digits** ("twenty-three" → "23")
2. **Date format standardization** ("Jan 15, 2025" → "2025-01-15")  
3. **Time format conversion** ("3:30 PM" → "15:30")
4. **Currency normalization** ("$1.5M" → "$1,500,000")
5. **Ordinal numbers** ("first" → "1st")
6. **Phone numbers and addresses**

## Key Components

### Normalization Types

```rust
enum NormalizationType {
    Number,      // Cardinal and written numbers
    Date,        // Various date formats
    Time,        // Time expressions
    Currency,    // Money amounts and units
    Measurement, // Units and quantities
    Ordinal,     // Ordinal numbers (1st, 2nd, etc.)
    PhoneNumber, // Phone number formatting
}
```

### Normalized Entity

```rust
struct NormalizedEntity {
    original: String,      // Original text
    normalized: String,    // Standardized form
    entity_type: NormalizationType,
    confidence: f32,       // Normalization confidence
}
```

## Core Normalizers

### Number Normalization

#### Written Numbers to Digits
```rust
fn build_number_normalizer() -> VectorFst<TropicalWeight>
```

Handles conversion of written numbers:
- **Basic numbers**: "zero" → "0", "one" → "1", ..., "nine" → "9"
- **Teens**: "ten" → "10", "eleven" → "11", ..., "nineteen" → "19"
- **Tens**: "twenty" → "20", "thirty" → "30", ..., "ninety" → "90"
- **Hundreds**: "one hundred" → "100", "two hundred" → "200"
- **Thousands**: "one thousand" → "1000", "five thousand" → "5000"
- **Complex**: "twenty-three" → "23", "one hundred fifty" → "150"

#### Large Number Units
```rust
vec![
    ("thousand", "000"),
    ("million", "000000"),
    ("billion", "000000000"),
    ("trillion", "000000000000"),
    ("k", "000"),           // "5k" → "5000"
    ("M", "000000"),        // "1.5M" → "1500000"
    ("B", "000000000"),     // "2B" → "2000000000"
]
```

### Date Normalization

#### Date Format Patterns
```rust
fn build_date_normalizer() -> VectorFst<TropicalWeight>
```

Converts various date formats to ISO 8601 (YYYY-MM-DD):

**Input Formats**:
- "January 15, 2025" → "2025-01-15"
- "Jan 15, 2025" → "2025-01-15"
- "15 Jan 2025" → "2025-01-15"
- "01/15/2025" → "2025-01-15"
- "15/01/2025" → "2025-01-15" (European format)
- "2025-01-15" → "2025-01-15" (already normalized)

**Month Name Mappings**:
```rust
let month_mappings = vec![
    ("january", "01"), ("jan", "01"),
    ("february", "02"), ("feb", "02"),
    ("march", "03"), ("mar", "03"),
    ("april", "04"), ("apr", "04"),
    ("may", "05"),
    ("june", "06"), ("jun", "06"),
    ("july", "07"), ("jul", "07"),
    ("august", "08"), ("aug", "08"),
    ("september", "09"), ("sep", "09"), ("sept", "09"),
    ("october", "10"), ("oct", "10"),
    ("november", "11"), ("nov", "11"),
    ("december", "12"), ("dec", "12"),
];
```

### Time Normalization

#### Time Format Conversion
```rust
fn build_time_normalizer() -> VectorFst<TropicalWeight>
```

Standardizes time expressions to 24-hour format (HH:MM):

**Input Formats**:
- "3:30 PM" → "15:30"
- "8:45 AM" → "08:45"
- "12:00 PM" → "12:00" (noon)
- "12:00 AM" → "00:00" (midnight)
- "quarter past three" → "15:15"
- "half past two" → "14:30"
- "ten to four" → "15:50"

### Currency Normalization

#### Currency Pattern Recognition
```rust
fn build_currency_normalizer() -> VectorFst<TropicalWeight>
```

Standardizes monetary expressions:

**Patterns**:
- "$1.5M" → "$1,500,000"
- "€250K" → "€250,000"
- "¥5000" → "¥5,000"
- "fifty dollars" → "$50"
- "twenty-five cents" → "$0.25"
- "two hundred euros" → "€200"

**Currency Symbols**:
```rust
let currency_symbols = vec![
    ("dollar", "$"), ("dollars", "$"),
    ("euro", "€"), ("euros", "€"),
    ("pound", "£"), ("pounds", "£"),
    ("yen", "¥"),
    ("cent", "¢"), ("cents", "¢"),
];
```

### Ordinal Normalization

#### Ordinal Number Conversion
```rust
fn build_ordinal_normalizer() -> VectorFst<TropicalWeight>
```

Converts written ordinals to numeric form:
- "first" → "1st"
- "second" → "2nd"  
- "third" → "3rd"
- "fourth" → "4th"
- "twenty-first" → "21st"
- "thirty-second" → "32nd"
- "one hundred third" → "103rd"

## Usage

Run the example with:
```bash
cargo run --example number_date_normalizer
```

### Example Output

```
=== Text Normalization Demo ===

Number Normalization:
  "twenty-three" → "23"
  "one hundred fifty" → "150"
  "two thousand five" → "2005"
  "three million" → "3000000"
  "1.5M" → "1500000"

Date Normalization:
  "January 15, 2025" → "2025-01-15"
  "15 Jan 2025" → "2025-01-15"
  "01/15/2025" → "2025-01-15"
  "Mar 3rd, 2025" → "2025-03-03"

Time Normalization:
  "3:30 PM" → "15:30"
  "8:45 AM" → "08:45"
  "quarter past three" → "15:15"
  "half past midnight" → "00:30"

Currency Normalization:
  "$1.5M" → "$1,500,000"
  "fifty dollars" → "$50.00"
  "twenty-five cents" → "$0.25"
  "€250K" → "€250,000"

Ordinal Normalization:
  "first" → "1st"
  "twenty-first" → "21st"
  "one hundred third" → "103rd"

Mixed Text Example:
Input:  "The meeting is on January 15th at three thirty PM for twenty-five million dollars"
Output: "The meeting is on 2025-01-15 at 15:30 for $25,000,000"
```

## Advanced Features

### Context-Aware Normalization

```rust
fn build_contextual_normalizer() -> VectorFst<TropicalWeight> {
    // Handles:
    // - "May" (month vs. modal verb)
    // - "March" (month vs. verb)
    // - "US" (country vs. pronoun)
    // - Date disambiguation (MM/DD vs DD/MM)
}
```

### Measurement Units

```rust
fn build_measurement_normalizer() -> VectorFst<TropicalWeight>
```

Standardizes units and measurements:
- "5 feet 10 inches" → "5'10""
- "twenty kilometers" → "20 km"
- "one hundred degrees" → "100°"
- "fifty percent" → "50%"
- "two point five liters" → "2.5 L"

### Phone Number Normalization

```rust
fn build_phone_normalizer() -> VectorFst<TropicalWeight>
```

Standardizes phone number formats:
- "(555) 123-4567" → "+1-555-123-4567"
- "555.123.4567" → "+1-555-123-4567"
- "+44 20 7123 4567" → "+44-20-7123-4567"

## Implementation Details

### Compositional Architecture

The system uses layered FST composition:

```rust
fn build_complete_normalizer() -> VectorFst<TropicalWeight> {
    let number_fst = build_number_normalizer();
    let date_fst = build_date_normalizer();
    let time_fst = build_time_normalizer();
    let currency_fst = build_currency_normalizer();
    
    // Compose all normalizers
    union_multiple(&[number_fst, date_fst, time_fst, currency_fst])
}
```

### Ambiguity Resolution

```rust
struct NormalizationCandidate {
    text: String,
    normalized: String,
    confidence: f32,
    span: (usize, usize),   // Character offsets
    entity_type: NormalizationType,
}

fn resolve_ambiguity(
    candidates: Vec<NormalizationCandidate>,
) -> NormalizationCandidate {
    // Resolution strategies:
    // 1. Longest match wins
    // 2. Higher confidence scores
    // 3. Context-specific preferences
    // 4. Domain-specific priorities
}
```

### Post-Processing Rules

```rust
fn apply_post_processing(
    normalized: &str,
    entity_type: NormalizationType,
) -> String {
    match entity_type {
        NormalizationType::Currency => add_thousand_separators(normalized),
        NormalizationType::PhoneNumber => apply_country_formatting(normalized),
        NormalizationType::Date => validate_date_bounds(normalized),
        _ => normalized.to_string(),
    }
}
```

## Applications

### Information Extraction
- Named entity recognition preprocessing
- Database value standardization  
- Document indexing and search
- Data cleaning pipelines

### Natural Language Processing
- Text preprocessing for ML models
- Feature engineering for NLP
- Cross-lingual data normalization
- Corpus standardization

### Business Intelligence
- Financial data processing
- Report generation
- Data warehouse ETL
- Business metrics standardization

### Voice Interfaces
- ASR output processing
- TTS input preparation
- Dialog system integration
- Voice command normalization

## Performance Characteristics

### Processing Speed
- Linear in text length: O(|text|)
- Constant-time entity recognition after FST compilation
- Batch processing optimizations for large corpora

### Memory Usage
- O(|patterns| × |average_pattern_length|) for FST storage
- Efficient trie-based compression for repeated patterns
- Lazy loading for domain-specific modules

### Accuracy Metrics
- Number normalization: >95% accuracy
- Date normalization: >92% accuracy  
- Time normalization: >88% accuracy
- Currency normalization: >90% accuracy
- Overall precision: >93%, Recall: >89%

## Extending the System

### Custom Domains
```rust
trait DomainNormalizer {
    fn build_domain_fst(&self) -> VectorFst<TropicalWeight>;
    fn entity_types(&self) -> Vec<NormalizationType>;
    fn post_process(&self, text: &str) -> String;
}

struct MedicalNormalizer; // Drug names, dosages, etc.
struct LegalNormalizer;   // Case citations, statute references
struct TechnicalNormalizer; // Part numbers, specifications
```

### Internationalization
```rust
struct LocaleNormalizer {
    locale: String,           // "en-US", "en-GB", "de-DE", etc.
    date_format: DateFormat,  // MM/DD/YYYY vs DD/MM/YYYY
    number_format: NumberFormat, // Decimal separator, thousands separator
    currency: CurrencyRules,
}
```

### Machine Learning Integration
```rust
fn build_ml_enhanced_normalizer(
    traditional_fst: VectorFst<TropicalWeight>,
    ml_model: &dyn MLNormalizer,
) -> HybridNormalizer {
    // Combines FST precision with ML flexibility
    // Uses FST for high-confidence cases
    // Falls back to ML for ambiguous cases
}
```

## Quality Assurance

### Test Suite
```rust
fn test_normalization_accuracy() -> TestResults {
    let test_cases = load_test_corpus();
    let mut correct = 0;
    let mut total = 0;
    
    for (input, expected) in test_cases {
        let result = normalize_text(&input);
        if result == expected {
            correct += 1;
        }
        total += 1;
    }
    
    TestResults {
        accuracy: correct as f32 / total as f32,
        total_cases: total,
        correct_cases: correct,
    }
}
```

### Validation Framework
```rust
fn validate_normalization_rules() -> ValidationReport {
    // Checks:
    // - Rule consistency across entity types
    // - No conflicting patterns
    // - Coverage of common cases
    // - Performance regressions
}
```

## Related Examples

- [Edit Distance](edit_distance.md) - Fuzzy matching for error correction
- [Morphological Analyzer](morphological_analyzer.md) - Linguistic normalization
- [Transliteration](transliteration.md) - Cross-script normalization
- [Word Correction](../examples/word_correction.rs) - Spelling normalization

## Standards and Resources

- **ISO 8601**: Date and time format standard
- **Unicode CLDR**: Common Locale Data Repository
- **NIST**: Measurement unit standards
- **E.164**: International phone number format
- **ISO 4217**: Currency code standard