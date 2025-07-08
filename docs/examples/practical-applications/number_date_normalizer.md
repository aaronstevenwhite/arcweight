# Number & Date Normalizer

This example demonstrates building sophisticated text normalization systems using FSTs to convert varied human input into standardized machine-readable formats.

## Overview

Text normalization is essential for NLP systems, converting the varied ways humans write information into consistent formats. Consider the many representations of the same data: dates can be "Jan 15, 2024", "15/01/24", or "the fifteenth of January"; numbers appear as "23", "twenty-three", or "XXIII"; times include "3:30 PM", "15:30", or "half past three".

Without normalization, search engines miss matches, voice assistants misunderstand commands, and data extraction fails. FSTs provide a systematic framework for handling these variations, ensuring downstream processing receives consistent, predictable input.

This system demonstrates comprehensive normalization covering numbers, dates, times, currency, measurements, and phone numbers, with support for multiple locales and standards.

## Quick Start

```bash
cargo run --example number_date_normalizer
```text

## What You'll Learn

- **Number Normalization**: Convert word forms to digits ("twenty-three" → "23")
- **Date Standardization**: Transform any format to ISO 8601 (YYYY-MM-DD)  
- **Time Processing**: Convert 12-hour to 24-hour and colloquial to standard forms
- **Currency Handling**: Standardize symbols and word forms across locales
- **Measurement Units**: Handle metric/imperial conversions and abbreviations
- **FST Pattern Matching**: Efficient rule-based text transformation techniques

## Core Concepts

### Real-World Applications

Understanding where text normalization fits in the NLP pipeline helps appreciate its importance:

### Text-to-Speech (TTS) Systems

TTS systems need to expand abbreviated forms for natural pronunciation:

```text
Input:  "Meeting at 3:30 PM on Jan 15, 2024"
Step 1: "Meeting at 15:30 on 2024-01-15"          (normalization)
Step 2: "Meeting at fifteen thirty on January     (expansion for speech)
         fifteenth, twenty twenty-four"
```text

**Why FSTs?** The bidirectional nature means the same FST can normalize for processing AND expand for speech.

### Information Extraction & Data Mining

Extract structured data from unstructured text:

```text
Input text: "The company raised $1.5 million in Series A funding"
Extracted:  {
  amount: 1500000.00,
  currency: "USD",
  context: "funding",
  round: "Series A"
}

Input text: "Temperature reached 98.6°F (37°C) at noon"
Extracted:  {
  temp_f: 98.6,
  temp_c: 37.0,
  time: "12:00",
  unit: "fahrenheit/celsius"
}
```text

### Cross-Format Search

Enable users to find information regardless of how it's written:

```text
User query: "flights on March 3rd"
Should match "03/03/2024 departure", "March 3, 2024 flights", "2024-03-03 available seats", and "3rd of March booking".

This requires normalizing BOTH the query and the documents to a common format.
```text

### Core Components

### Number Words to Digits

The system handles a comprehensive range of number representations:

**Basic Numbers (0-19):**
```rust,ignore
("zero", "0"), ("one", "1"), ("two", "2"), ..., ("nineteen", "19")
```text

**Tens:**
```rust,ignore
("twenty", "20"), ("thirty", "30"), ..., ("ninety", "90")
```text

**Compound Numbers:**
```rust,ignore
("twenty-one", "21"), ("thirty-five", "35"), ("forty-seven", "47")
```text

**Large Numbers:**
```rust,ignore
("hundred", "100"), ("thousand", "1000"), ("million", "1000000")
```text

### Date Pattern Recognition

Multiple date formats are normalized to ISO 8601 format:

**Month Abbreviations:**
```rust,ignore
("Jan", "01"), ("Feb", "02"), ("Mar", "03"), ..., ("Dec", "12")
```text

**Full Month Names:**
```rust,ignore
("January", "01"), ("February", "02"), ..., ("December", "12")
```text

**Common Date Formats:**
```rust,ignore
("Jan 15, 2024", "2024-01-15")
("12/25/2023", "2023-12-25")  
("15-Jan-2024", "2024-01-15")
("2024/01/15", "2024-01-15")
```text

### Time Normalization

Convert 12-hour format to 24-hour format:

```rust,ignore
("12:00 AM", "00:00")  // Midnight
("1:00 PM", "13:00")   // 1 PM
("11:59 PM", "23:59")  // Just before midnight
("noon", "12:00")      // Noon
("midnight", "00:00")  // Midnight
```text

### Currency Patterns

Standardize various currency representations:

```rust,ignore
("$10", "USD 10.00")
("ten dollars", "USD 10.00")  
("€100", "EUR 100.00")
("fifty pounds", "GBP 50.00")
("¥1000", "JPY 1000.00")
```text

## Implementation

### NumberNormalizer Structure

The core normalizer contains pattern databases for different entity types:

```rust,ignore
struct NumberNormalizer {
    word_to_digit: Vec<(String, String)>,
    ordinal_to_number: Vec<(String, String)>,
    date_patterns: Vec<(String, String)>,
    time_patterns: Vec<(String, String)>,
    currency_patterns: Vec<(String, String)>,
    measurement_patterns: Vec<(String, String)>,
}
```text

### Text Processing Pipeline

The normalization process follows these steps. **Pattern Matching** identifies normalizable entities in text. **Rule Application** applies transformation rules. **Context Resolution** handles ambiguous cases. **Output Generation** produces normalized text.

```rust,ignore
fn normalize_text(&self, text: &str) -> Vec<NormalizedEntity> {
    let mut results = Vec::new();
    
    // Check each pattern type
    for (pattern, normalized) in &self.word_to_digit {
        if text.contains(pattern) {
            results.push(NormalizedEntity {
                original: pattern.clone(),
                normalized: normalized.clone(),
                entity_type: NormalizationType::Number,
                confidence: 1.0,
            });
        }
    }
    
    // ... (similar for other types)
    results
}
```text

## FST Implementation

### Number Normalization FST

A simple FST demonstrates the concept:

```rust,ignore
fn build_number_normalization_fst() -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    fst.set_start(start);
    fst.set_final(start, TropicalWeight::one());

    let number_rules = vec![
        ("one", "1"), ("two", "2"), ("three", "3"), 
        ("four", "4"), ("five", "5")
    ];

    for (word, digit) in number_rules {
        // Accept the word
        let mut current = start;
        for ch in word.chars() {
            let next = fst.add_state();
            fst.add_arc(current, Arc::new(
                ch as u32, 0,  // epsilon output during input
                TropicalWeight::one(), next
            ));
            current = next;
        }

        // Output the digit
        for ch in digit.chars() {
            let next = fst.add_state();
            fst.add_arc(current, Arc::new(
                0, ch as u32,  // epsilon input, output digit
                TropicalWeight::one(), next
            ));
            current = next;
        }
    }

    fst
}
```text

## Running the Example

```bash
cargo run --example number_date_normalizer
```text

### Sample Output

```text
Number Normalization:
------------------------
'I have twenty-three apples' → 'I have 23 apples'
'The price is fifty dollars' → 'The price is USD 50.00'
'Wait for thirty minutes' → 'Wait for 30 minutes'

Date Normalization:
---------------------
'Meeting on Jan 15, 2024' → 'Meeting on 2024-01-15'
'Born in February 3, 2023' → 'Born in 2023-02-03'
'Deadline is Mar 22, 2024' → 'Deadline is 2024-03-22'

Time Normalization:
---------------------
'Meeting at 2:30 PM' → 'Meeting at 14:30'
'Wake up at 6:15 AM' → 'Wake up at 06:15'
'Lunch at noon' → 'Lunch at 12:00'
```text

## Advanced Features

### Ordinal Number Processing

Convert written ordinals to numeric ordinals:

```rust,ignore
("first", "1st"), ("second", "2nd"), ("third", "3rd"),
("twentieth", "20th"), ("twenty-first", "21st")
```text

**Example Usage:**
```text
"This is the first time" → "This is the 1st time"
"Take the second exit" → "Take the 2nd exit"
"The twenty-first century" → "The 21st century"
```text

### Measurement Normalization

Standardize units and measurements:

```rust,ignore
// Length
("5 feet", "5 ft"), ("10 inches", "10 in"), ("2 miles", "2 mi")

// Weight  
("10 pounds", "10 lbs"), ("2 kilograms", "2 kg")

// Volume
("1 gallon", "1 gal"), ("2 liters", "2 L")

// Temperature
("32 degrees Fahrenheit", "32°F"), ("100 degrees Celsius", "100°C")
```text

### Phone Number Normalization

Standardize various phone number formats:

```rust,ignore
fn normalize_phone_numbers(text: &str) -> Vec<NormalizedEntity> {
    let phone_patterns = vec![
        ("(555) 123-4567", "+1-555-123-4567"),
        ("555-123-4567", "+1-555-123-4567"),
        ("555.123.4567", "+1-555-123-4567"),
        ("5551234567", "+1-555-123-4567"),
    ];
    // ... implementation
}
```text

### Complex Text Processing

### Comprehensive Normalization

The system can handle complex sentences with multiple entities:

**Input:**
```text
"The meeting is on Jan 15, 2024 at 2:30 PM with twenty-three people."
```text

**Output:**
```text
"The meeting is on 2024-01-15 at 14:30 with 23 people."
```text

**Detected Entities:**
```text
Date: 'Jan 15, 2024' → '2024-01-15'
Time: '2:30 PM' → '14:30'  
Number: 'twenty-three' → '23'
```text

### Entity Recognition

The normalizer identifies and categorizes different entity types:

```rust,ignore
#[derive(Debug, Clone, PartialEq)]
enum NormalizationType {
    Number,
    Date,
    Time,
    Currency,
    Measurement,
    Ordinal,
    PhoneNumber,
}

struct NormalizedEntity {
    original: String,
    normalized: String,
    entity_type: NormalizationType,
    confidence: f32,
}
```text

### Localization Support

### Different Locale Formats

The system can be extended to handle locale-specific formats:

**US Format:**
```text
Date: MM/DD/YYYY
Currency: $1,000.00
Measurements: 5'10", 150 lbs
```text

**European Format:**
```text
Date: DD/MM/YYYY  
Currency: €1.000,00
Measurements: 1.78m, 68kg
```text

**Japanese Format:**
```text
Date: YYYY/MM/DD
Currency: ¥1,000
Measurements: 178cm
```text

### Configurable Rules

The FST-based approach enables easy rule switching:

```rust,ignore
impl NumberNormalizer {
    fn with_locale(locale: &str) -> Self {
        match locale {
            "en-US" => Self::new_us_format(),
            "en-GB" => Self::new_uk_format(),
            "de-DE" => Self::new_german_format(),
            _ => Self::new(),
        }
    }
}
```text

## Performance Considerations

### Efficient Pattern Matching

Several techniques optimize pattern matching. **Longest Match First** processing handles longer patterns before shorter ones. **Trie-Based Lookup** provides efficient prefix matching for patterns. **Rule Caching** memoizes frequent transformations. **Parallel Processing** handles different entity types concurrently.

### Memory Optimization

Memory optimization uses several strategies. **Shared Pattern Storage** reuses common subpatterns. **Lazy Loading** loads patterns on demand. **Compression** uses compressed pattern representations. **State Minimization** minimizes FST states.

### Integration

### Speech Recognition

Normalization is crucial for speech recognition accuracy:

```text
Audio: "twenty five dollars"
ASR Output: "twenty five dollars"
Normalized: "USD 25.00"
```text

### Machine Translation

Enable consistent translation of numerical expressions:

```text
English: "January 15th, 2024"
Normalized: "2024-01-15"
German: "15. Januar 2024"
```text

### Information Extraction

Support structured data extraction:

```text
Text: "The company raised $1.5 million in Series A funding."
Extracted: {
  entity: "funding",
  amount: "USD 1500000.00",
  round: "Series A"
}
```text

### Error Handling

### Ambiguous Cases

Handle potentially ambiguous inputs:

```rust,ignore
// "12/01/2024" could be:
// - December 1, 2024 (US format)
// - January 12, 2024 (European format)

fn resolve_date_ambiguity(&self, date_str: &str, locale: &str) -> String {
    match locale {
        "en-US" => self.parse_us_date(date_str),
        "en-GB" => self.parse_uk_date(date_str),
        _ => self.parse_iso_date(date_str),
    }
}
```text

### Validation

Ensure normalized output is valid:

```rust,ignore
fn validate_normalized_date(&self, date: &str) -> bool {
    // Check if date is valid ISO 8601 format
    // Validate month (01-12), day (01-31), year ranges
    true // simplified
}
```text

### Enhanced Pattern Recognition

Enhanced pattern recognition includes **Regular Expression Integration** for complex pattern matching, **Context-Aware Processing** to use surrounding text for disambiguation, **Named Entity Recognition** for integration with NER systems, and **Machine Learning** to learn patterns from data.

### Multilingual Support

Multilingual support includes **Unicode Handling** for non-Latin scripts, **Cultural Variations** to handle culture-specific number formats, **Cross-Language Consistency** to maintain normalization across languages, and **Right-to-Left Scripts** supporting Arabic and Hebrew number formats.

### Advanced Applications

Advanced applications include **Real-Time Processing** with stream processing capabilities, **Batch Processing** to handle large document collections, **API Integration** through web service endpoints, and **Database Integration** for direct database normalization.

## Related Examples

This number normalizer connects with other examples. **[Edit Distance](../text-processing/edit_distance.md)** provides fuzzy matching for inexact patterns. **[Spell Checking](../text-processing/spell_checking.md)** offers spell correction for number words. **[Transliteration](transliteration.md)** handles numbers in different scripts. **[Morphological Analyzer](../linguistic-applications/morphological_analyzer.md)** processes morphological variants of numbers.

## Applications

### Financial Systems
Financial systems benefit from invoice processing and accounting, currency conversion and formatting, financial report generation, and transaction normalization.

### Healthcare Systems  
Healthcare systems use the normalizer for medical record normalization, dosage and measurement standardization, date of birth and appointment processing, and vital sign recording.

### Legal Documents
Legal document processing includes contract date standardization, monetary amount normalization, case number formatting, and legal citation processing.

### Academic Research
Academic research applications include citation date normalization, statistical data processing, historical date standardization, and cross-reference formatting.

## References

### Standards and Specifications
Relevant standards include **ISO 8601** for date and time representation, **Unicode CLDR** for locale data for formatting, **ITU-T E.164** for international telephone numbering, and **ISO 4217** for currency code standards.

### Research Papers
Key research papers include Sproat, R., Black, A.W., Chen, S., Kumar, S., Ostendorf, M., & Richards, C. (2001) "Normalization of non-standard words" Computer Speech & Language, 15:287-333, Taylor, P. (2009) "Text-to-Speech Synthesis", and Bangalore, S. & Rambow, O. (2000) "Exploiting a probabilistic hierarchical model for generation".

