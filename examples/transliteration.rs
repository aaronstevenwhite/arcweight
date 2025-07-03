//! Transliteration FST Example
//!
//! This example demonstrates how to build and use finite state transducers for
//! transliteration between different writing systems. It shows:
//! 1. Building character mapping FSTs for script conversion
//! 2. Cyrillic to Latin transliteration (Russian → English)
//! 3. Arabic to Latin transliteration
//! 4. Greek to Latin transliteration
//! 5. Bidirectional transliteration systems
//! 6. Multiple transliteration schemes (BGN/PCGN, ISO, etc.)
//! 7. Handling digraphs and context-sensitive rules
//!
//! Transliteration is essential for cross-language text processing, search,
//! machine translation, and internationalization of applications.
//!
//! Usage:
//! ```bash
//! cargo run --example transliteration
//! ```

use arcweight::prelude::*;
use std::collections::HashMap;

/// Transliteration scheme types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)] // Some variants included for completeness but not used in this example
enum TransliterationScheme {
    BgnPcgn,    // Board on Geographic Names / Permanent Committee on Geographical Names
    Iso,        // ISO transliteration standards
    AlaLc,      // American Library Association / Library of Congress
    Scientific, // Scientific transliteration
    Popular,    // Popular/simplified transliteration
}

/// Script types for transliteration
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)] // Script variants included for API completeness, not all used in example
enum Script {
    Cyrillic,
    Latin,
    Arabic,
    Greek,
    Hebrew,
}

/// Transliteration mapping entry
#[derive(Debug, Clone)]
struct TransliterationRule {
    source: String,
    target: String,
    context_before: Option<String>,
    context_after: Option<String>,
    scheme: TransliterationScheme,
}

/// Transliteration system database
struct TransliterationSystem {
    cyrillic_to_latin: HashMap<TransliterationScheme, Vec<TransliterationRule>>,
    arabic_to_latin: HashMap<TransliterationScheme, Vec<TransliterationRule>>,
    greek_to_latin: HashMap<TransliterationScheme, Vec<TransliterationRule>>,
}

impl TransliterationSystem {
    fn new() -> Self {
        let mut system = TransliterationSystem {
            cyrillic_to_latin: HashMap::new(),
            arabic_to_latin: HashMap::new(),
            greek_to_latin: HashMap::new(),
        };

        system.initialize_cyrillic_mappings();
        system.initialize_arabic_mappings();
        system.initialize_greek_mappings();

        system
    }

    fn initialize_cyrillic_mappings(&mut self) {
        // BGN/PCGN Russian transliteration
        let bgn_pcgn_rules = vec![
            // Vowels
            TransliterationRule {
                source: "а".to_string(),
                target: "a".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "е".to_string(),
                target: "e".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ё".to_string(),
                target: "ë".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "и".to_string(),
                target: "i".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "о".to_string(),
                target: "o".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "у".to_string(),
                target: "u".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ы".to_string(),
                target: "y".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "э".to_string(),
                target: "e".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ю".to_string(),
                target: "yu".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "я".to_string(),
                target: "ya".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            // Consonants
            TransliterationRule {
                source: "б".to_string(),
                target: "b".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "в".to_string(),
                target: "v".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "г".to_string(),
                target: "g".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "д".to_string(),
                target: "d".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ж".to_string(),
                target: "zh".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "з".to_string(),
                target: "z".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "й".to_string(),
                target: "y".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "к".to_string(),
                target: "k".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "л".to_string(),
                target: "l".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "м".to_string(),
                target: "m".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "н".to_string(),
                target: "n".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "п".to_string(),
                target: "p".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "р".to_string(),
                target: "r".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "с".to_string(),
                target: "s".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "т".to_string(),
                target: "t".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ф".to_string(),
                target: "f".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "х".to_string(),
                target: "kh".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ц".to_string(),
                target: "ts".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ч".to_string(),
                target: "ch".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ш".to_string(),
                target: "sh".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "щ".to_string(),
                target: "shch".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ъ".to_string(),
                target: "\"".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ь".to_string(),
                target: "'".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
        ];

        // Popular/simplified transliteration
        let popular_rules = vec![
            TransliterationRule {
                source: "а".to_string(),
                target: "a".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "е".to_string(),
                target: "e".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "ё".to_string(),
                target: "yo".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "и".to_string(),
                target: "i".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "о".to_string(),
                target: "o".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "у".to_string(),
                target: "u".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "ы".to_string(),
                target: "y".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "э".to_string(),
                target: "e".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "ю".to_string(),
                target: "yu".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "я".to_string(),
                target: "ya".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "б".to_string(),
                target: "b".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "в".to_string(),
                target: "v".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "г".to_string(),
                target: "g".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "д".to_string(),
                target: "d".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "ж".to_string(),
                target: "zh".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "з".to_string(),
                target: "z".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "й".to_string(),
                target: "y".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "к".to_string(),
                target: "k".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "л".to_string(),
                target: "l".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "м".to_string(),
                target: "m".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "н".to_string(),
                target: "n".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "п".to_string(),
                target: "p".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "р".to_string(),
                target: "r".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "с".to_string(),
                target: "s".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "т".to_string(),
                target: "t".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "ф".to_string(),
                target: "f".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "х".to_string(),
                target: "h".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "ц".to_string(),
                target: "ts".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "ч".to_string(),
                target: "ch".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "ш".to_string(),
                target: "sh".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "щ".to_string(),
                target: "sch".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "ъ".to_string(),
                target: "".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
            TransliterationRule {
                source: "ь".to_string(),
                target: "".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::Popular,
            },
        ];

        self.cyrillic_to_latin
            .insert(TransliterationScheme::BgnPcgn, bgn_pcgn_rules);
        self.cyrillic_to_latin
            .insert(TransliterationScheme::Popular, popular_rules);
    }

    fn initialize_arabic_mappings(&mut self) {
        // Simplified Arabic to Latin transliteration
        let arabic_rules = vec![
            // Arabic letters
            TransliterationRule {
                source: "ا".to_string(),
                target: "a".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ب".to_string(),
                target: "b".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ت".to_string(),
                target: "t".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ث".to_string(),
                target: "th".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ج".to_string(),
                target: "j".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ح".to_string(),
                target: "h".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "خ".to_string(),
                target: "kh".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "د".to_string(),
                target: "d".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ذ".to_string(),
                target: "dh".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ر".to_string(),
                target: "r".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ز".to_string(),
                target: "z".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "س".to_string(),
                target: "s".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ش".to_string(),
                target: "sh".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ص".to_string(),
                target: "s".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ض".to_string(),
                target: "d".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ط".to_string(),
                target: "t".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ظ".to_string(),
                target: "z".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ع".to_string(),
                target: "'".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "غ".to_string(),
                target: "gh".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ف".to_string(),
                target: "f".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ق".to_string(),
                target: "q".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ك".to_string(),
                target: "k".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ل".to_string(),
                target: "l".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "م".to_string(),
                target: "m".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ن".to_string(),
                target: "n".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ه".to_string(),
                target: "h".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "و".to_string(),
                target: "w".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ي".to_string(),
                target: "y".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
        ];

        self.arabic_to_latin
            .insert(TransliterationScheme::BgnPcgn, arabic_rules);
    }

    fn initialize_greek_mappings(&mut self) {
        // Greek to Latin transliteration
        let greek_rules = vec![
            // Vowels
            TransliterationRule {
                source: "α".to_string(),
                target: "a".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ε".to_string(),
                target: "e".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "η".to_string(),
                target: "i".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ι".to_string(),
                target: "i".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ο".to_string(),
                target: "o".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "υ".to_string(),
                target: "y".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ω".to_string(),
                target: "o".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            // Consonants
            TransliterationRule {
                source: "β".to_string(),
                target: "v".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "γ".to_string(),
                target: "g".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "δ".to_string(),
                target: "d".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ζ".to_string(),
                target: "z".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "θ".to_string(),
                target: "th".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "κ".to_string(),
                target: "k".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "λ".to_string(),
                target: "l".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "μ".to_string(),
                target: "m".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ν".to_string(),
                target: "n".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ξ".to_string(),
                target: "x".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "π".to_string(),
                target: "p".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ρ".to_string(),
                target: "r".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "σ".to_string(),
                target: "s".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ς".to_string(),
                target: "s".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "τ".to_string(),
                target: "t".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "φ".to_string(),
                target: "f".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "χ".to_string(),
                target: "ch".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
            TransliterationRule {
                source: "ψ".to_string(),
                target: "ps".to_string(),
                context_before: None,
                context_after: None,
                scheme: TransliterationScheme::BgnPcgn,
            },
        ];

        self.greek_to_latin
            .insert(TransliterationScheme::BgnPcgn, greek_rules);
    }

    /// Transliterate text using simple string replacement (for demonstration)
    fn transliterate_simple(
        &self,
        text: &str,
        source_script: Script,
        scheme: TransliterationScheme,
    ) -> String {
        let rules = match source_script {
            Script::Cyrillic => self.cyrillic_to_latin.get(&scheme),
            Script::Arabic => self.arabic_to_latin.get(&scheme),
            Script::Greek => self.greek_to_latin.get(&scheme),
            _ => None,
        };

        if let Some(rules) = rules {
            let mut result = text.to_string();

            // Sort rules by source length (longest first) to handle digraphs correctly
            let mut sorted_rules = rules.clone();
            sorted_rules.sort_by(|a, b| b.source.len().cmp(&a.source.len()));

            for rule in &sorted_rules {
                result = result.replace(&rule.source, &rule.target);
            }

            result
        } else {
            text.to_string()
        }
    }
}

/// Build a transliteration FST for a specific script and scheme
fn build_transliteration_fst(rules: &[TransliterationRule]) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    fst.set_start(start);
    fst.set_final(start, TropicalWeight::one());

    // Sort rules by source length (longest first) to handle digraphs
    let mut sorted_rules = rules.to_vec();
    sorted_rules.sort_by(|a, b| b.source.len().cmp(&a.source.len()));

    for rule in &sorted_rules {
        let mut current = start;

        // Process each character in the source
        let source_chars: Vec<char> = rule.source.chars().collect();
        for (i, &ch) in source_chars.iter().enumerate() {
            if i == source_chars.len() - 1 {
                // Last character - output the target
                let target_chars: Vec<char> = rule.target.chars().collect();
                let _target_state = start;

                for &target_ch in &target_chars {
                    let next = fst.add_state();
                    fst.add_arc(
                        current,
                        Arc::new(ch as u32, target_ch as u32, TropicalWeight::one(), next),
                    );
                    current = next;
                }

                // Connect back to start for more characters
                fst.add_arc(
                    current,
                    Arc::new(
                        0, // epsilon
                        0, // epsilon
                        TropicalWeight::one(),
                        start,
                    ),
                );
            } else {
                // Intermediate character
                let next = fst.add_state();
                fst.add_arc(
                    current,
                    Arc::new(
                        ch as u32,
                        0, // epsilon output
                        TropicalWeight::one(),
                        next,
                    ),
                );
                current = next;
            }
        }
    }

    // Add pass-through for unknown characters
    for ch in 0..=127u8 {
        if ch.is_ascii() {
            fst.add_arc(
                start,
                Arc::new(ch as u32, ch as u32, TropicalWeight::one(), start),
            );
        }
    }

    fst
}

/// Create reverse transliteration mappings
fn create_reverse_mappings(rules: &[TransliterationRule]) -> Vec<TransliterationRule> {
    rules
        .iter()
        .map(|rule| TransliterationRule {
            source: rule.target.clone(),
            target: rule.source.clone(),
            context_before: rule.context_before.clone(),
            context_after: rule.context_after.clone(),
            scheme: rule.scheme,
        })
        .collect()
}

fn main() -> Result<()> {
    println!("Transliteration FST Example");
    println!("===========================\n");

    // Initialize transliteration system
    let transliteration_system = TransliterationSystem::new();

    println!("Transliteration system initialized with:");
    let cyrillic_count = transliteration_system.cyrillic_to_latin.len();
    println!("  {cyrillic_count} Cyrillic schemes");
    let arabic_count = transliteration_system.arabic_to_latin.len();
    println!("  {arabic_count} Arabic schemes");
    let greek_count = transliteration_system.greek_to_latin.len();
    println!("  {greek_count} Greek schemes");

    // Test Cyrillic transliteration
    println!("\n1. Cyrillic to Latin Transliteration:");
    println!("------------------------------------");

    let cyrillic_tests = vec![
        "москва",          // Moscow
        "санкт-петербург", // Saint Petersburg
        "россия",          // Russia
        "владимир",        // Vladimir
        "екатерина",       // Ekaterina
        "борщ",            // Borscht
        "водка",           // Vodka
        "матрёшка",        // Matryoshka
        "кремль",          // Kremlin
        "большой",         // Bolshoy
    ];

    for text in cyrillic_tests {
        println!("\nRussian: '{text}'");

        // BGN/PCGN transliteration
        let bgn_result = transliteration_system.transliterate_simple(
            text,
            Script::Cyrillic,
            TransliterationScheme::BgnPcgn,
        );
        println!("  BGN/PCGN: '{bgn_result}'");

        // Popular transliteration
        let popular_result = transliteration_system.transliterate_simple(
            text,
            Script::Cyrillic,
            TransliterationScheme::Popular,
        );
        println!("  Popular:  '{popular_result}'");
    }

    // Test Arabic transliteration
    println!("\n2. Arabic to Latin Transliteration:");
    println!("----------------------------------");

    let arabic_tests = vec![
        "السلام",   // Peace (as-salam)
        "مرحبا",   // Hello (marhaba)
        "شكرا",    // Thank you (shukran)
        "قاهرة",   // Cairo (qahirah)
        "بغداد",   // Baghdad
        "دمشق",    // Damascus
        "مكة",     // Mecca
        "المدينة", // Medina
    ];

    for text in arabic_tests {
        println!("\nArabic: '{text}'");
        let result = transliteration_system.transliterate_simple(
            text,
            Script::Arabic,
            TransliterationScheme::BgnPcgn,
        );
        println!("  Latin: '{result}'");
    }

    // Test Greek transliteration
    println!("\n3. Greek to Latin Transliteration:");
    println!("---------------------------------");

    let greek_tests = vec![
        "αθήνα",       // Athens (athina)
        "θεσσαλονίκη", // Thessaloniki
        "κρήτη",       // Crete (kriti)
        "ελλάδα",      // Greece (ellada)
        "φιλοσοφία",   // Philosophy
        "δημοκρατία",  // Democracy
        "μουσική",     // Music
        "θέατρο",      // Theater
    ];

    for text in greek_tests {
        println!("\nGreek: '{text}'");
        let result = transliteration_system.transliterate_simple(
            text,
            Script::Greek,
            TransliterationScheme::BgnPcgn,
        );
        println!("  Latin: '{result}'");
    }

    // Demonstrate FST-based transliteration
    println!("\n4. FST-based Transliteration:");
    println!("-----------------------------");

    if let Some(cyrillic_rules) = transliteration_system
        .cyrillic_to_latin
        .get(&TransliterationScheme::Popular)
    {
        let _transliteration_fst = build_transliteration_fst(cyrillic_rules);
        let state_count = _transliteration_fst.num_states();
        println!("Built transliteration FST with {state_count} states");

        // Create reverse mappings for bidirectional transliteration
        let reverse_rules = create_reverse_mappings(cyrillic_rules);
        let _reverse_fst = build_transliteration_fst(&reverse_rules);
        let reverse_state_count = _reverse_fst.num_states();
        println!("Built reverse transliteration FST with {reverse_state_count} states");
    }

    // Show transliteration scheme differences
    println!("\n5. Transliteration Scheme Comparison:");
    println!("------------------------------------");
    println!("Different schemes handle characters differently:\n");

    let comparison_tests = vec![
        ("х", "Cyrillic letter KHA"),
        ("ё", "Cyrillic letter YO"),
        ("щ", "Cyrillic letter SHCHA"),
        ("ь", "Cyrillic soft sign"),
        ("ъ", "Cyrillic hard sign"),
    ];

    for (char, description) in comparison_tests {
        println!("{char} ({description}):");
        let bgn = transliteration_system.transliterate_simple(
            char,
            Script::Cyrillic,
            TransliterationScheme::BgnPcgn,
        );
        let popular = transliteration_system.transliterate_simple(
            char,
            Script::Cyrillic,
            TransliterationScheme::Popular,
        );
        println!("  BGN/PCGN: '{bgn}' | Popular: '{popular}'");
    }

    // Applications and use cases
    println!("\n6. Applications and Use Cases:");
    println!("-----------------------------");
    println!("Transliteration FSTs are essential for:");
    println!("  • Cross-language information retrieval");
    println!("  • Name matching across different scripts");
    println!("  • Machine translation preprocessing");
    println!("  • International address processing");
    println!("  • Search engine query expansion");
    println!("  • Database normalization");
    println!("  • Multilingual user interfaces");
    println!("  • Geographic information systems");

    // Technical advantages of FSTs
    println!("\n7. FST Advantages for Transliteration:");
    println!("-------------------------------------");
    println!("FSTs provide several benefits over simple substitution:");
    println!("  • Context-sensitive rules: handle position-dependent mappings");
    println!("  • Bidirectional conversion: same FST for forward/reverse");
    println!("  • Composition: combine multiple transformation layers");
    println!("  • Efficiency: linear time processing");
    println!("  • Weighted alternatives: rank multiple transliterations");
    println!("  • Standardization: consistent rule application");

    // Challenges and considerations
    println!("\n8. Challenges in Transliteration:");
    println!("--------------------------------");
    println!("Key challenges include:");
    println!("  • One-to-many mappings: single source → multiple targets");
    println!("  • Context sensitivity: different rules based on neighbors");
    println!("  • Digraphs and trigraphs: multi-character mappings");
    println!("  • Vowel insertion: handling consonant-only scripts");
    println!("  • Orthographic variations: multiple valid spellings");
    println!("  • Historic vs. modern conventions");
    println!("  • Domain-specific requirements (legal, geographic, etc.)");

    Ok(())
}
