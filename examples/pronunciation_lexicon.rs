//! Pronunciation Dictionary / Lexicon FST Example
//!
//! This example demonstrates how to build and use a pronunciation lexicon FST
//! that maps between words and their phonetic representations. It shows:
//! 1. Building a lexicon FST that maps words to phoneme sequences
//! 2. Creating a simple G2P system for unknown words
//! 3. Handling words with multiple pronunciations
//! 4. Text-to-phoneme conversion pipeline
//! 5. Applications in speech processing systems
//!
//! This is a fundamental component in speech recognition and synthesis systems.
//!
//! Usage:
//! ```bash
//! cargo run --example pronunciation_lexicon
//! ```

use arcweight::prelude::*;

/// Phoneme symbols used in our simplified phonetic representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Phoneme {
    // Vowels
    AA,
    AE,
    AH,
    AO,
    AW,
    AY,
    EH,
    ER,
    EY,
    IH,
    IY,
    OW,
    OY,
    UH,
    UW,
    // Consonants
    B,
    CH,
    D,
    DH,
    F,
    G,
    HH,
    JH,
    K,
    L,
    M,
    N,
    NG,
    P,
    R,
    S,
    SH,
    T,
    TH,
    V,
    W,
    Y,
    Z,
    ZH,
    // Special
    Sil,
}

impl Phoneme {
    fn to_label(self) -> u32 {
        (self as u32) + 1000 // Offset to avoid conflicts with character labels
    }

    fn from_label(label: u32) -> Option<Self> {
        if (1000..1100).contains(&label) {
            match label - 1000 {
                0 => Some(Phoneme::AA),
                1 => Some(Phoneme::AE),
                2 => Some(Phoneme::AH),
                3 => Some(Phoneme::AO),
                4 => Some(Phoneme::AW),
                5 => Some(Phoneme::AY),
                6 => Some(Phoneme::EH),
                7 => Some(Phoneme::ER),
                8 => Some(Phoneme::EY),
                9 => Some(Phoneme::IH),
                10 => Some(Phoneme::IY),
                11 => Some(Phoneme::OW),
                12 => Some(Phoneme::OY),
                13 => Some(Phoneme::UH),
                14 => Some(Phoneme::UW),
                15 => Some(Phoneme::B),
                16 => Some(Phoneme::CH),
                17 => Some(Phoneme::D),
                18 => Some(Phoneme::DH),
                19 => Some(Phoneme::F),
                20 => Some(Phoneme::G),
                21 => Some(Phoneme::HH),
                22 => Some(Phoneme::JH),
                23 => Some(Phoneme::K),
                24 => Some(Phoneme::L),
                25 => Some(Phoneme::M),
                26 => Some(Phoneme::N),
                27 => Some(Phoneme::NG),
                28 => Some(Phoneme::P),
                29 => Some(Phoneme::R),
                30 => Some(Phoneme::S),
                31 => Some(Phoneme::SH),
                32 => Some(Phoneme::T),
                33 => Some(Phoneme::TH),
                34 => Some(Phoneme::V),
                35 => Some(Phoneme::W),
                36 => Some(Phoneme::Y),
                37 => Some(Phoneme::Z),
                38 => Some(Phoneme::ZH),
                39 => Some(Phoneme::Sil),
                _ => None,
            }
        } else {
            None
        }
    }

    fn to_string(self) -> &'static str {
        match self {
            Phoneme::AA => "AA",
            Phoneme::AE => "AE",
            Phoneme::AH => "AH",
            Phoneme::AO => "AO",
            Phoneme::AW => "AW",
            Phoneme::AY => "AY",
            Phoneme::EH => "EH",
            Phoneme::ER => "ER",
            Phoneme::EY => "EY",
            Phoneme::IH => "IH",
            Phoneme::IY => "IY",
            Phoneme::OW => "OW",
            Phoneme::OY => "OY",
            Phoneme::UH => "UH",
            Phoneme::UW => "UW",
            Phoneme::B => "B",
            Phoneme::CH => "CH",
            Phoneme::D => "D",
            Phoneme::DH => "DH",
            Phoneme::F => "F",
            Phoneme::G => "G",
            Phoneme::HH => "HH",
            Phoneme::JH => "JH",
            Phoneme::K => "K",
            Phoneme::L => "L",
            Phoneme::M => "M",
            Phoneme::N => "N",
            Phoneme::NG => "NG",
            Phoneme::P => "P",
            Phoneme::R => "R",
            Phoneme::S => "S",
            Phoneme::SH => "SH",
            Phoneme::T => "T",
            Phoneme::TH => "TH",
            Phoneme::V => "V",
            Phoneme::W => "W",
            Phoneme::Y => "Y",
            Phoneme::Z => "Z",
            Phoneme::ZH => "ZH",
            Phoneme::Sil => "SIL",
        }
    }
}

/// Entry in the pronunciation dictionary
struct LexiconEntry {
    word: String,
    pronunciations: Vec<Vec<Phoneme>>,
}

/// Simple approach: directly encode word->phoneme mappings
fn build_simple_lexicon(entries: &[LexiconEntry]) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    fst.set_start(start);

    for entry in entries {
        // For each pronunciation, create a separate path
        for pronunciation in &entry.pronunciations {
            let mut states = vec![start];

            // Create states for each character in the word
            for _ in 0..entry.word.len() {
                states.push(fst.add_state());
            }

            // Add transitions for each character
            for (i, ch) in entry.word.chars().enumerate() {
                fst.add_arc(
                    states[i],
                    Arc::new(
                        ch as u32,
                        ch as u32, // Output the same character for now
                        TropicalWeight::one(),
                        states[i + 1],
                    ),
                );
            }

            // At the end of the word, output the pronunciation
            let mut current = states[entry.word.len()];
            for phoneme in pronunciation {
                let next = fst.add_state();
                fst.add_arc(
                    current,
                    Arc::new(
                        0, // epsilon input
                        phoneme.to_label(),
                        TropicalWeight::one(),
                        next,
                    ),
                );
                current = next;
            }

            fst.set_final(current, TropicalWeight::one());
        }
    }

    fst
}

/// Build an FST that accepts a single word
fn word_acceptor(word: &str) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let mut current = fst.add_state();
    fst.set_start(current);

    for ch in word.chars() {
        let next = fst.add_state();
        fst.add_arc(
            current,
            Arc::new(ch as u32, ch as u32, TropicalWeight::one(), next),
        );
        current = next;
    }

    fst.set_final(current, TropicalWeight::one());
    fst
}

/// Look up pronunciations for a word using the lexicon
fn lookup_word_in_lexicon(entries: &[LexiconEntry], word: &str) -> Option<Vec<Vec<Phoneme>>> {
    for entry in entries {
        if entry.word == word {
            return Some(entry.pronunciations.clone());
        }
    }
    None
}

/// Build a G2P (Grapheme-to-Phoneme) FST for unknown words
fn build_g2p_rules() -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    fst.set_start(start);
    fst.set_final(start, TropicalWeight::one());

    // Simple G2P rules - in practice this would be much more sophisticated
    let rules = vec![
        ('a', Phoneme::AE),
        ('e', Phoneme::EH),
        ('i', Phoneme::IH),
        ('o', Phoneme::AO),
        ('u', Phoneme::AH),
        ('b', Phoneme::B),
        ('c', Phoneme::K),
        ('d', Phoneme::D),
        ('f', Phoneme::F),
        ('g', Phoneme::G),
        ('h', Phoneme::HH),
        ('j', Phoneme::JH),
        ('k', Phoneme::K),
        ('l', Phoneme::L),
        ('m', Phoneme::M),
        ('n', Phoneme::N),
        ('p', Phoneme::P),
        ('r', Phoneme::R),
        ('s', Phoneme::S),
        ('t', Phoneme::T),
        ('v', Phoneme::V),
        ('w', Phoneme::W),
        ('y', Phoneme::Y),
        ('z', Phoneme::Z),
    ];

    for (grapheme, phoneme) in rules {
        fst.add_arc(
            start,
            Arc::new(
                grapheme as u32,
                phoneme.to_label(),
                TropicalWeight::new(1.0),
                start,
            ),
        );
    }

    fst
}

/// Extract phonemes from a G2P result
fn extract_g2p_phonemes(fst: &VectorFst<TropicalWeight>, word: &str) -> Vec<Phoneme> {
    let word_fst = word_acceptor(word);
    let composed: Result<VectorFst<TropicalWeight>> = compose_default(&word_fst, fst);
    if let Ok(composed) = composed {
        if let Some(start) = composed.start() {
            let mut phonemes = Vec::new();
            extract_phonemes_from_state(&composed, start, &mut phonemes);
            return phonemes;
        }
    }
    Vec::new()
}

fn extract_phonemes_from_state(
    fst: &VectorFst<TropicalWeight>,
    state: u32,
    phonemes: &mut Vec<Phoneme>,
) {
    if fst.is_final(state) {
        return;
    }

    if let Some(arc) = fst.arcs(state).next() {
        if arc.olabel != 0 {
            if let Some(phoneme) = Phoneme::from_label(arc.olabel) {
                phonemes.push(phoneme);
            }
        }
        extract_phonemes_from_state(fst, arc.nextstate, phonemes);
    }
}

/// Create a simple pronunciation dictionary
fn create_sample_lexicon() -> Vec<LexiconEntry> {
    use Phoneme::*;

    vec![
        LexiconEntry {
            word: "hello".to_string(),
            pronunciations: vec![
                vec![HH, AH, L, OW],
                vec![HH, EH, L, OW], // Alternative pronunciation
            ],
        },
        LexiconEntry {
            word: "world".to_string(),
            pronunciations: vec![vec![W, ER, L, D]],
        },
        LexiconEntry {
            word: "cat".to_string(),
            pronunciations: vec![vec![K, AE, T]],
        },
        LexiconEntry {
            word: "dog".to_string(),
            pronunciations: vec![vec![D, AO, G]],
        },
        LexiconEntry {
            word: "read".to_string(),
            pronunciations: vec![
                vec![R, IY, D], // present tense
                vec![R, EH, D], // past tense
            ],
        },
        LexiconEntry {
            word: "live".to_string(),
            pronunciations: vec![
                vec![L, IH, V], // verb
                vec![L, AY, V], // adjective
            ],
        },
        LexiconEntry {
            word: "the".to_string(),
            pronunciations: vec![
                vec![DH, AH], // unstressed
                vec![DH, IY], // stressed
            ],
        },
    ]
}

/// Convert text to phonemes using lexicon + G2P fallback
fn text_to_phonemes(
    lexicon_entries: &[LexiconEntry],
    g2p_fst: &VectorFst<TropicalWeight>,
    text: &str,
) -> Vec<Phoneme> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut result_phonemes = Vec::new();

    for (i, word) in words.iter().enumerate() {
        // First try lexicon lookup
        if let Some(pronunciations) = lookup_word_in_lexicon(lexicon_entries, word) {
            // Use first pronunciation
            result_phonemes.extend(pronunciations[0].clone());
        } else {
            // Fallback to G2P rules
            let g2p_phonemes = extract_g2p_phonemes(g2p_fst, word);
            result_phonemes.extend(g2p_phonemes);
        }

        // Add silence between words
        if i != words.len() - 1 {
            result_phonemes.push(Phoneme::Sil);
        }
    }

    result_phonemes
}

fn main() -> Result<()> {
    println!("Pronunciation Dictionary / Lexicon FST Example");
    println!("=============================================\n");

    // Create sample lexicon
    let lexicon_entries = create_sample_lexicon();

    let len = lexicon_entries.len();
    println!("Building lexicon with {len} words...");
    for entry in &lexicon_entries {
        let word = &entry.word;
        let len = entry.pronunciations.len();
        println!("  {word} -> {len} pronunciation(s)");
    }

    // Build the lexicon FST (for demonstration, though we'll use direct lookup)
    let _lexicon_fst = build_simple_lexicon(&lexicon_entries);
    println!("\nLexicon FST built successfully");

    // Test word lookups using direct lookup (simpler and more reliable)
    println!("\n1. Word to Phoneme Lookup:");
    println!("--------------------------");
    let test_words = vec!["hello", "world", "cat", "the", "read", "live", "unknown"];

    for word in test_words {
        println!("\n  Looking up '{word}':");
        if let Some(pronunciations) = lookup_word_in_lexicon(&lexicon_entries, word) {
            for (i, pron) in pronunciations.iter().enumerate() {
                let phoneme_str: Vec<&str> = pron.iter().map(|p| p.to_string()).collect();
                let num = i + 1;
                let phonemes = phoneme_str.join(" ");
                println!("    Pronunciation {num}: {phonemes}");
            }
        } else {
            println!("    Not found in lexicon");
        }
    }

    // Build G2P rules for unknown words
    println!("\n2. Grapheme-to-Phoneme (G2P) Rules:");
    println!("-----------------------------------");
    let g2p_fst = build_g2p_rules();
    let num_states = g2p_fst.num_states();
    println!("G2P FST built with {num_states} states");

    // Test G2P on unknown words
    let unknown_words = vec!["test", "book", "run", "jump"];
    for word in unknown_words {
        println!("\nApplying G2P rules to '{word}':");
        let phonemes = extract_g2p_phonemes(&g2p_fst, word);
        if !phonemes.is_empty() {
            let phoneme_str: Vec<&str> = phonemes.iter().map(|p| p.to_string()).collect();
            let phonemes = phoneme_str.join(" ");
            println!("  G2P result: {phonemes}");
        } else {
            println!("  No G2P result");
        }
    }

    // Demonstrate text-to-phoneme conversion
    println!("\n3. Text-to-Phoneme Conversion:");
    println!("------------------------------");
    let test_sentences = vec![
        "hello world",
        "the cat",
        "read the book",
        "test unknown words",
    ];

    for sentence in test_sentences {
        println!("\nText: \"{sentence}\"");
        let phonemes = text_to_phonemes(&lexicon_entries, &g2p_fst, sentence);
        let phoneme_str: Vec<&str> = phonemes.iter().map(|p| p.to_string()).collect();
        let phonemes = phoneme_str.join(" ");
        println!("Phonemes: {phonemes}");
    }

    // Show applications in speech processing
    println!("\n4. Applications in Speech Processing:");
    println!("------------------------------------");
    println!("Pronunciation lexicons are essential for:");
    println!("  • Speech Recognition: Convert acoustic models to words");
    println!("  • Speech Synthesis: Convert text to pronunciations");
    println!("  • Phonetic Analysis: Study pronunciation variations");
    println!("  • Language Learning: Provide pronunciation guidance");
    println!("  • Dialect Studies: Map regional pronunciation differences");
    println!("  • Voice Assistants: Enable natural language understanding");

    // Demonstrate handling of homophones and multiple pronunciations
    println!("\n5. Homophones and Multiple Pronunciations:");
    println!("-----------------------------------------");
    println!("The lexicon handles words with multiple pronunciations, which occur due to:");
    println!("  • Different meanings (homophones): 'read' (present) vs 'read' (past)");
    println!("  • Grammatical categories: 'live' (verb) vs 'live' (adjective)");
    println!("  • Stress variations: 'the' (unstressed) vs 'the' (emphasized)");
    println!("  • Regional dialects: different pronunciations in different regions");
    println!();

    for entry in &lexicon_entries {
        if entry.pronunciations.len() > 1 {
            let word = &entry.word;
            let len = entry.pronunciations.len();
            println!("  '{word}' has {len} pronunciations:");
            for (i, pron) in entry.pronunciations.iter().enumerate() {
                let phoneme_str: Vec<&str> = pron.iter().map(|p| p.to_string()).collect();
                let num = i + 1;
                let phonemes = phoneme_str.join(" ");
                let desc = match entry.word.as_str() {
                    "hello" => {
                        if i == 0 {
                            "standard"
                        } else {
                            "variant"
                        }
                    }
                    "read" => {
                        if i == 0 {
                            "present tense"
                        } else {
                            "past tense"
                        }
                    }
                    "live" => {
                        if i == 0 {
                            "verb /lɪv/"
                        } else {
                            "adjective /laɪv/"
                        }
                    }
                    "the" => {
                        if i == 0 {
                            "unstressed /ðə/"
                        } else {
                            "stressed /ðiː/"
                        }
                    }
                    _ => "variant",
                };
                println!("    {num}: {phonemes} ({desc})");
            }
            println!();
        }
    }

    // Demonstrate FST composition concept
    println!("6. FST Composition in Speech Processing:");
    println!("---------------------------------------");
    println!("In a complete speech processing system, FSTs are composed:");
    println!("  Text → Words → Phonemes → Acoustic Models → Audio");
    println!("  G2P ∘ Lexicon ∘ Language Model ∘ Acoustic Model");
    println!();
    println!("Each FST handles a specific transformation:");
    println!("  • Lexicon FST: Maps words to phoneme sequences");
    println!("  • G2P FST: Handles out-of-vocabulary words");
    println!("  • Phonetic FST: Models pronunciation variations");
    println!("  • Acoustic FST: Maps phonemes to audio features");

    Ok(())
}
