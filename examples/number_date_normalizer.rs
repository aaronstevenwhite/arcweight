//! Number/Date Normalizer FST Example
//!
//! This example demonstrates how to build and use finite state transducers for
//! normalizing various textual representations of numbers, dates, times, and
//! other structured data into standardized formats. It shows:
//! 1. Converting written numbers to digits (e.g., "twenty-three" → "23")
//! 2. Normalizing date formats (e.g., "Jan 15, 2024" → "2024-01-15")
//! 3. Time format standardization (e.g., "3:30 PM" → "15:30")
//! 4. Currency and measurement normalization
//! 5. Ordinal number handling (e.g., "first" → "1st")
//! 6. Phone number and address normalization
//!
//! This is essential for text preprocessing in NLP, data extraction, and
//! information normalization systems.
//!
//! Usage:
//! ```bash
//! cargo run --example number_date_normalizer
//! ```

use arcweight::prelude::*;

/// Types of normalizable entities
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

/// Normalized entity result
#[derive(Debug, Clone)]
struct NormalizedEntity {
    original: String,
    normalized: String,
    entity_type: NormalizationType,
    _confidence: f32,
}

/// Number normalizer database - maps written numbers to digits
struct NumberNormalizer {
    word_to_digit: Vec<(String, String)>,
    ordinal_to_number: Vec<(String, String)>,
    date_patterns: Vec<(String, String)>,
    time_patterns: Vec<(String, String)>,
    currency_patterns: Vec<(String, String)>,
    measurement_patterns: Vec<(String, String)>,
}

impl NumberNormalizer {
    fn new() -> Self {
        let word_to_digit = vec![
            // Basic numbers 0-19
            ("zero".to_string(), "0".to_string()),
            ("one".to_string(), "1".to_string()),
            ("two".to_string(), "2".to_string()),
            ("three".to_string(), "3".to_string()),
            ("four".to_string(), "4".to_string()),
            ("five".to_string(), "5".to_string()),
            ("six".to_string(), "6".to_string()),
            ("seven".to_string(), "7".to_string()),
            ("eight".to_string(), "8".to_string()),
            ("nine".to_string(), "9".to_string()),
            ("ten".to_string(), "10".to_string()),
            ("eleven".to_string(), "11".to_string()),
            ("twelve".to_string(), "12".to_string()),
            ("thirteen".to_string(), "13".to_string()),
            ("fourteen".to_string(), "14".to_string()),
            ("fifteen".to_string(), "15".to_string()),
            ("sixteen".to_string(), "16".to_string()),
            ("seventeen".to_string(), "17".to_string()),
            ("eighteen".to_string(), "18".to_string()),
            ("nineteen".to_string(), "19".to_string()),
            
            // Tens
            ("twenty".to_string(), "20".to_string()),
            ("thirty".to_string(), "30".to_string()),
            ("forty".to_string(), "40".to_string()),
            ("fifty".to_string(), "50".to_string()),
            ("sixty".to_string(), "60".to_string()),
            ("seventy".to_string(), "70".to_string()),
            ("eighty".to_string(), "80".to_string()),
            ("ninety".to_string(), "90".to_string()),
            
            // Compound numbers
            ("twenty-one".to_string(), "21".to_string()),
            ("twenty-two".to_string(), "22".to_string()),
            ("twenty-three".to_string(), "23".to_string()),
            ("thirty-five".to_string(), "35".to_string()),
            ("forty-seven".to_string(), "47".to_string()),
            ("fifty-nine".to_string(), "59".to_string()),
            
            // Hundreds
            ("hundred".to_string(), "100".to_string()),
            ("one hundred".to_string(), "100".to_string()),
            ("two hundred".to_string(), "200".to_string()),
            ("three hundred".to_string(), "300".to_string()),
            
            // Thousands
            ("thousand".to_string(), "1000".to_string()),
            ("one thousand".to_string(), "1000".to_string()),
            ("two thousand".to_string(), "2000".to_string()),
            
            // Millions
            ("million".to_string(), "1000000".to_string()),
            ("one million".to_string(), "1000000".to_string()),
        ];
        
        let ordinal_to_number = vec![
            ("first".to_string(), "1st".to_string()),
            ("second".to_string(), "2nd".to_string()),
            ("third".to_string(), "3rd".to_string()),
            ("fourth".to_string(), "4th".to_string()),
            ("fifth".to_string(), "5th".to_string()),
            ("sixth".to_string(), "6th".to_string()),
            ("seventh".to_string(), "7th".to_string()),
            ("eighth".to_string(), "8th".to_string()),
            ("ninth".to_string(), "9th".to_string()),
            ("tenth".to_string(), "10th".to_string()),
            ("eleventh".to_string(), "11th".to_string()),
            ("twelfth".to_string(), "12th".to_string()),
            ("thirteenth".to_string(), "13th".to_string()),
            ("fourteenth".to_string(), "14th".to_string()),
            ("fifteenth".to_string(), "15th".to_string()),
            ("twentieth".to_string(), "20th".to_string()),
            ("twenty-first".to_string(), "21st".to_string()),
            ("thirtieth".to_string(), "30th".to_string()),
        ];
        
        let date_patterns = vec![
            // Month abbreviations
            ("Jan".to_string(), "01".to_string()),
            ("Feb".to_string(), "02".to_string()),
            ("Mar".to_string(), "03".to_string()),
            ("Apr".to_string(), "04".to_string()),
            ("May".to_string(), "05".to_string()),
            ("Jun".to_string(), "06".to_string()),
            ("Jul".to_string(), "07".to_string()),
            ("Aug".to_string(), "08".to_string()),
            ("Sep".to_string(), "09".to_string()),
            ("Oct".to_string(), "10".to_string()),
            ("Nov".to_string(), "11".to_string()),
            ("Dec".to_string(), "12".to_string()),
            
            // Full month names
            ("January".to_string(), "01".to_string()),
            ("February".to_string(), "02".to_string()),
            ("March".to_string(), "03".to_string()),
            ("April".to_string(), "04".to_string()),
            ("June".to_string(), "06".to_string()),
            ("July".to_string(), "07".to_string()),
            ("August".to_string(), "08".to_string()),
            ("September".to_string(), "09".to_string()),
            ("October".to_string(), "10".to_string()),
            ("November".to_string(), "11".to_string()),
            ("December".to_string(), "12".to_string()),
            
            // Common date formats
            ("Jan 15, 2024".to_string(), "2024-01-15".to_string()),
            ("February 3, 2023".to_string(), "2023-02-03".to_string()),
            ("Mar 22, 2024".to_string(), "2024-03-22".to_string()),
            ("12/25/2023".to_string(), "2023-12-25".to_string()),
            ("01/01/2024".to_string(), "2024-01-01".to_string()),
            ("3/15/24".to_string(), "2024-03-15".to_string()),
            ("15-Jan-2024".to_string(), "2024-01-15".to_string()),
            ("2024/01/15".to_string(), "2024-01-15".to_string()),
        ];
        
        let time_patterns = vec![
            // 12-hour format
            ("12:00 AM".to_string(), "00:00".to_string()),
            ("1:00 AM".to_string(), "01:00".to_string()),
            ("12:00 PM".to_string(), "12:00".to_string()),
            ("1:00 PM".to_string(), "13:00".to_string()),
            ("2:30 PM".to_string(), "14:30".to_string()),
            ("3:45 PM".to_string(), "15:45".to_string()),
            ("6:15 PM".to_string(), "18:15".to_string()),
            ("11:59 PM".to_string(), "23:59".to_string()),
            
            // Common time expressions
            ("noon".to_string(), "12:00".to_string()),
            ("midnight".to_string(), "00:00".to_string()),
            ("quarter past three".to_string(), "15:15".to_string()),
            ("half past four".to_string(), "16:30".to_string()),
            ("quarter to five".to_string(), "16:45".to_string()),
            
            // Informal time
            ("3 o'clock".to_string(), "15:00".to_string()),
            ("8 AM".to_string(), "08:00".to_string()),
            ("5 PM".to_string(), "17:00".to_string()),
        ];
        
        let currency_patterns = vec![
            // Dollar amounts
            ("$10".to_string(), "USD 10.00".to_string()),
            ("$25.50".to_string(), "USD 25.50".to_string()),
            ("$1,000".to_string(), "USD 1000.00".to_string()),
            ("$1.5 million".to_string(), "USD 1500000.00".to_string()),
            ("ten dollars".to_string(), "USD 10.00".to_string()),
            ("fifty cents".to_string(), "USD 0.50".to_string()),
            ("a dollar".to_string(), "USD 1.00".to_string()),
            
            // Other currencies
            ("€100".to_string(), "EUR 100.00".to_string()),
            ("£50".to_string(), "GBP 50.00".to_string()),
            ("¥1000".to_string(), "JPY 1000.00".to_string()),
            ("100 euros".to_string(), "EUR 100.00".to_string()),
            ("fifty pounds".to_string(), "GBP 50.00".to_string()),
        ];
        
        let measurement_patterns = vec![
            // Length
            ("5 feet".to_string(), "5 ft".to_string()),
            ("10 inches".to_string(), "10 in".to_string()),
            ("2 miles".to_string(), "2 mi".to_string()),
            ("100 meters".to_string(), "100 m".to_string()),
            ("5 kilometers".to_string(), "5 km".to_string()),
            ("6 foot 2".to_string(), "6'2\"".to_string()),
            
            // Weight
            ("10 pounds".to_string(), "10 lbs".to_string()),
            ("2 kilograms".to_string(), "2 kg".to_string()),
            ("5 ounces".to_string(), "5 oz".to_string()),
            ("1 ton".to_string(), "1 ton".to_string()),
            
            // Volume
            ("1 gallon".to_string(), "1 gal".to_string()),
            ("2 liters".to_string(), "2 L".to_string()),
            ("8 cups".to_string(), "8 cups".to_string()),
            ("3 tablespoons".to_string(), "3 tbsp".to_string()),
            
            // Temperature
            ("32 degrees Fahrenheit".to_string(), "32°F".to_string()),
            ("100 degrees Celsius".to_string(), "100°C".to_string()),
            ("room temperature".to_string(), "20°C".to_string()),
        ];
        
        NumberNormalizer {
            word_to_digit,
            ordinal_to_number,
            date_patterns,
            time_patterns,
            currency_patterns,
            measurement_patterns,
        }
    }
    
    /// Normalize a text containing various entities
    fn normalize_text(&self, text: &str) -> Vec<NormalizedEntity> {
        let mut results = Vec::new();
        
        // Check for number words
        for (word, digit) in &self.word_to_digit {
            if text.contains(word) {
                results.push(NormalizedEntity {
                    original: word.clone(),
                    normalized: digit.clone(),
                    entity_type: NormalizationType::Number,
                    _confidence: 1.0,
                });
            }
        }
        
        // Check for ordinal numbers
        for (ordinal, number) in &self.ordinal_to_number {
            if text.contains(ordinal) {
                results.push(NormalizedEntity {
                    original: ordinal.clone(),
                    normalized: number.clone(),
                    entity_type: NormalizationType::Ordinal,
                    _confidence: 1.0,
                });
            }
        }
        
        // Check for date patterns
        for (date_text, normalized_date) in &self.date_patterns {
            if text.contains(date_text) {
                results.push(NormalizedEntity {
                    original: date_text.clone(),
                    normalized: normalized_date.clone(),
                    entity_type: NormalizationType::Date,
                    _confidence: 1.0,
                });
            }
        }
        
        // Check for time patterns
        for (time_text, normalized_time) in &self.time_patterns {
            if text.contains(time_text) {
                results.push(NormalizedEntity {
                    original: time_text.clone(),
                    normalized: normalized_time.clone(),
                    entity_type: NormalizationType::Time,
                    _confidence: 1.0,
                });
            }
        }
        
        // Check for currency patterns
        for (currency_text, normalized_currency) in &self.currency_patterns {
            if text.contains(currency_text) {
                results.push(NormalizedEntity {
                    original: currency_text.clone(),
                    normalized: normalized_currency.clone(),
                    entity_type: NormalizationType::Currency,
                    _confidence: 1.0,
                });
            }
        }
        
        // Check for measurement patterns
        for (measurement_text, normalized_measurement) in &self.measurement_patterns {
            if text.contains(measurement_text) {
                results.push(NormalizedEntity {
                    original: measurement_text.clone(),
                    normalized: normalized_measurement.clone(),
                    entity_type: NormalizationType::Measurement,
                    _confidence: 1.0,
                });
            }
        }
        
        results
    }
    
    /// Apply normalizations to a text string
    fn apply_normalizations(&self, text: &str) -> String {
        let mut result = text.to_string();
        
        // Apply number normalizations
        for (word, digit) in &self.word_to_digit {
            result = result.replace(word, digit);
        }
        
        // Apply ordinal normalizations
        for (ordinal, number) in &self.ordinal_to_number {
            result = result.replace(ordinal, number);
        }
        
        // Apply date normalizations
        for (date_text, normalized_date) in &self.date_patterns {
            result = result.replace(date_text, normalized_date);
        }
        
        // Apply time normalizations
        for (time_text, normalized_time) in &self.time_patterns {
            result = result.replace(time_text, normalized_time);
        }
        
        // Apply currency normalizations
        for (currency_text, normalized_currency) in &self.currency_patterns {
            result = result.replace(currency_text, normalized_currency);
        }
        
        // Apply measurement normalizations
        for (measurement_text, normalized_measurement) in &self.measurement_patterns {
            result = result.replace(measurement_text, normalized_measurement);
        }
        
        result
    }
}

/// Build a simple FST for number normalization (demonstration)
fn build_number_normalization_fst() -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    fst.set_start(start);
    fst.set_final(start, TropicalWeight::one());
    
    // Add some simple number transformations
    let number_rules = vec![
        ("one", "1"),
        ("two", "2"),
        ("three", "3"),
        ("four", "4"),
        ("five", "5"),
    ];
    
    for (word, digit) in number_rules {
        let mut current = start;
        
        // Accept the word
        for ch in word.chars() {
            let next = fst.add_state();
            fst.add_arc(current, Arc::new(
                ch as u32,
                0, // epsilon output during word
                TropicalWeight::one(),
                next
            ));
            current = next;
        }
        
        // Output the digit
        for ch in digit.chars() {
            let next = fst.add_state();
            fst.add_arc(current, Arc::new(
                0, // epsilon input
                ch as u32,
                TropicalWeight::one(),
                next
            ));
            current = next;
        }
        
        // Connect back to start for more normalizations
        fst.add_arc(current, Arc::new(
            0, // epsilon
            0, // epsilon
            TropicalWeight::one(),
            start
        ));
    }
    
    fst
}

/// Phone number normalizer patterns
fn normalize_phone_numbers(text: &str) -> Vec<NormalizedEntity> {
    let phone_patterns = vec![
        ("(555) 123-4567", "+1-555-123-4567"),
        ("555-123-4567", "+1-555-123-4567"),
        ("555.123.4567", "+1-555-123-4567"),
        ("5551234567", "+1-555-123-4567"),
        ("+1 555 123 4567", "+1-555-123-4567"),
        ("1-800-FLOWERS", "+1-800-356-9377"),
    ];
    
    let mut results = Vec::new();
    for (pattern, normalized) in phone_patterns {
        if text.contains(pattern) {
            results.push(NormalizedEntity {
                original: pattern.to_string(),
                normalized: normalized.to_string(),
                entity_type: NormalizationType::PhoneNumber,
                _confidence: 0.95,
            });
        }
    }
    results
}

/// Demonstrate FST-based normalization pipeline
fn process_with_fst_pipeline(text: &str, _normalizer_fst: &VectorFst<TropicalWeight>) -> String {
    // This is a simplified demonstration of how an FST could be used
    // In practice, you'd compose the input text FST with the normalizer FST
    
    // For demonstration, we'll show the concept
    let simple_replacements = vec![
        ("one", "1"),
        ("two", "2"),
        ("three", "3"),
    ];
    
    let mut result = text.to_string();
    for (from, to) in simple_replacements {
        result = result.replace(from, to);
    }
    
    result
}

fn main() -> Result<()> {
    println!("Number/Date Normalizer FST Example");
    println!("=================================\n");
    
    // Create normalizer
    let normalizer = NumberNormalizer::new();
    
    // Build demonstration FST
    let _number_fst = build_number_normalization_fst();
    
    println!("Normalization patterns loaded:");
    println!("  {} number word mappings", normalizer.word_to_digit.len());
    println!("  {} ordinal number mappings", normalizer.ordinal_to_number.len());
    println!("  {} date format patterns", normalizer.date_patterns.len());
    println!("  {} time format patterns", normalizer.time_patterns.len());
    println!("  {} currency patterns", normalizer.currency_patterns.len());
    println!("  {} measurement patterns", normalizer.measurement_patterns.len());
    
    // Test number normalization
    println!("\n1. Number Normalization:");
    println!("------------------------");
    let number_tests = vec![
        "I have twenty-three apples",
        "The price is fifty dollars",
        "Wait for thirty minutes",
        "Buy two hundred shares",
        "Population is one million",
        "Temperature is zero degrees",
    ];
    
    for test in number_tests {
        let normalized = normalizer.apply_normalizations(test);
        println!("  '{}' → '{}'", test, normalized);
    }
    
    // Test ordinal normalization
    println!("\n2. Ordinal Number Normalization:");
    println!("--------------------------------");
    let ordinal_tests = vec![
        "This is the first time",
        "Take the second exit",
        "On the third floor",
        "The twentieth century",
        "Twenty-first birthday",
    ];
    
    for test in ordinal_tests {
        let normalized = normalizer.apply_normalizations(test);
        println!("  '{}' → '{}'", test, normalized);
    }
    
    // Test date normalization
    println!("\n3. Date Normalization:");
    println!("---------------------");
    let date_tests = vec![
        "Meeting on Jan 15, 2024",
        "Born in February 3, 2023",
        "Deadline is Mar 22, 2024",
        "Holiday on 12/25/2023",
        "Started on 01/01/2024",
        "Due 3/15/24",
    ];
    
    for test in date_tests {
        let normalized = normalizer.apply_normalizations(test);
        println!("  '{}' → '{}'", test, normalized);
    }
    
    // Test time normalization
    println!("\n4. Time Normalization:");
    println!("---------------------");
    let time_tests = vec![
        "Meeting at 2:30 PM",
        "Wake up at 6:15 AM",
        "Lunch at noon",
        "Deadline at midnight",
        "Call at 3 o'clock",
        "Due at quarter past three",
    ];
    
    for test in time_tests {
        let normalized = normalizer.apply_normalizations(test);
        println!("  '{}' → '{}'", test, normalized);
    }
    
    // Test currency normalization
    println!("\n5. Currency Normalization:");
    println!("-------------------------");
    let currency_tests = vec![
        "Cost is $25.50",
        "Budget of ten dollars",
        "Price €100 euros",
        "Worth fifty pounds",
        "Salary $1.5 million",
        "Change fifty cents",
    ];
    
    for test in currency_tests {
        let normalized = normalizer.apply_normalizations(test);
        println!("  '{}' → '{}'", test, normalized);
    }
    
    // Test measurement normalization
    println!("\n6. Measurement Normalization:");
    println!("-----------------------------");
    let measurement_tests = vec![
        "Height 5 feet 10 inches",
        "Distance 2 miles away",
        "Weight 10 pounds",
        "Volume 2 liters",
        "Temperature 32 degrees Fahrenheit",
        "Length 100 meters",
    ];
    
    for test in measurement_tests {
        let normalized = normalizer.apply_normalizations(test);
        println!("  '{}' → '{}'", test, normalized);
    }
    
    // Test phone number normalization
    println!("\n7. Phone Number Normalization:");
    println!("------------------------------");
    let phone_tests = vec![
        "Call (555) 123-4567",
        "Text 555-123-4567",
        "Fax 555.123.4567",
        "Mobile 5551234567",
        "Office +1 555 123 4567",
    ];
    
    for test in phone_tests {
        let phone_results = normalize_phone_numbers(test);
        if !phone_results.is_empty() {
            let result = &phone_results[0];
            println!("  '{}' → '{}'", test, test.replace(&result.original, &result.normalized));
        } else {
            println!("  '{}' → {} (no normalization)", test, test);
        }
    }
    
    // Comprehensive text normalization
    println!("\n8. Comprehensive Text Normalization:");
    println!("------------------------------------");
    let complex_texts = vec![
        "The meeting is on Jan 15, 2024 at 2:30 PM with twenty-three people.",
        "Budget: ten thousand dollars for the first quarter.",
        "Temperature reached thirty-two degrees Fahrenheit at noon.",
        "Flight duration: two hours and fifteen minutes on February 3, 2023.",
        "Distance: five miles, weight: one hundred pounds, cost: $25.50.",
    ];
    
    for text in complex_texts {
        let normalized = normalizer.apply_normalizations(text);
        println!("\nOriginal:");
        println!("  {}", text);
        println!("Normalized:");
        println!("  {}", normalized);
        
        // Show detected entities
        let entities = normalizer.normalize_text(text);
        if !entities.is_empty() {
            println!("Detected entities:");
            for entity in entities {
                println!("  {:?}: '{}' → '{}'", 
                         entity.entity_type, 
                         entity.original, 
                         entity.normalized);
            }
        }
    }
    
    // FST pipeline demonstration
    println!("\n9. FST Pipeline Processing:");
    println!("--------------------------");
    println!("Demonstrating how FSTs can be used for normalization:");
    
    let fst_test = "I need one apple, two oranges, and three bananas.";
    let fst_result = process_with_fst_pipeline(fst_test, &_number_fst);
    println!("  Input:  {}", fst_test);
    println!("  Output: {}", fst_result);
    
    // Applications and benefits
    println!("\n10. Applications and Benefits:");
    println!("-----------------------------");
    println!("Number/Date normalization is essential for:");
    println!("  • Text-to-Speech systems: consistent pronunciation");
    println!("  • Search engines: matching different number formats");
    println!("  • Data extraction: standardizing structured information");
    println!("  • Machine translation: handling numerical expressions");
    println!("  • Database integration: consistent data formats");
    println!("  • Financial systems: standardizing currency amounts");
    println!("  • Medical records: normalizing measurements and dosages");
    println!("  • Legal documents: standardizing dates and references");
    
    println!("\nFST advantages for normalization:");
    println!("  • Bidirectional: normalization ↔ denormalization");
    println!("  • Compositional: combine multiple normalization rules");
    println!("  • Efficient: linear time processing");
    println!("  • Deterministic: consistent results");
    println!("  • Maintainable: rules are explicit and modifiable");
    println!("  • Language-agnostic: same framework for different locales");
    
    // Localization examples
    println!("\n11. Localization Considerations:");
    println!("--------------------------------");
    println!("Different locales require different normalization rules:");
    println!("  US: MM/DD/YYYY, $1,000.00, 5'10\"");
    println!("  EU: DD/MM/YYYY, €1.000,00, 1.78m");
    println!("  UK: DD/MM/YYYY, £1,000.00, 5ft 10in");
    println!("  JP: YYYY/MM/DD, ¥1,000, 178cm");
    println!();
    println!("FSTs can easily handle locale-specific rules through:");
    println!("  • Separate FSTs for each locale");
    println!("  • Parameterized FST construction");
    println!("  • Runtime rule switching");
    
    Ok(())
}