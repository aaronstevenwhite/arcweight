//! Phonological Rules with FSTs
//!
//! This example demonstrates how to model phonological rule systems using Finite State
//! Transducers (FSTs), following the foundational work of Kaplan and Kay (1994) and
//! subsequent developments in computational phonology. It shows:
//!
//! 1. Building FSTs for individual phonological processes
//! 2. Composing multiple phonological rules to model rule interaction
//! 3. Classic phonological phenomena: vowel harmony, consonant cluster simplification,
//!    vowel epenthesis, and final devoicing
//! 4. Rule ordering effects and how composition order affects output
//! 5. Complex multi-rule phonological systems
//!
//! This demonstrates the theoretical elegance and practical power of using FST
//! composition to model how phonological rules interact in natural language systems.
//!
//! Based on the seminal work:
//! - Kaplan, R. M., & Kay, M. (1994). Regular models of phonological rule systems.
//! - Johnson, C. D. (1972). Formal aspects of phonological description.
//! - Koskenniemi, K. (1983). Two-level morphology.
//!
//! Related examples:
//! - morphological_analyzer.rs: Shows how phonological rules integrate with morphology
//! - transliteration.rs: Demonstrates related phonetic transformation techniques
//!
//! Usage:
//! ```bash
//! cargo run --example phonological_rules
//! ```

use arcweight::prelude::*;

/// Phonological feature representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
enum PhonologicalFeature {
    // Vowel features
    High,
    Mid,
    Low,
    Front,
    Central,
    Back,
    Rounded,
    Unrounded,

    // Consonant features
    Voiced,
    Voiceless,
    Stop,
    Fricative,
    Nasal,
    Liquid,
    Labial,
    Coronal,
    Dorsal,

    // Prosodic features
    Stressed,
    Unstressed,

    // Boundary markers
    WordBoundary,
    SyllableBoundary,
}

/// Phonological segment with features (for future extensions)
#[derive(Debug, Clone)]
#[allow(dead_code)] // Included for API completeness but not used in this example
struct PhonologicalSegment {
    symbol: char,
    features: Vec<PhonologicalFeature>,
}

/// Phonological rule representation (for future extensions)
#[derive(Debug, Clone)]
#[allow(dead_code)] // Included for API completeness but not used in this example
struct PhonologicalRule {
    name: String,
    description: String,
    input_context: Vec<String>,
    output_context: Vec<String>,
    environment: Option<String>,
}

/// Build FST for vowel harmony (front/back spreading)
/// Example: Turkish-style back vowel harmony
fn build_vowel_harmony_fst() -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    let back_state = fst.add_state();
    let front_state = fst.add_state();

    fst.set_start(start);
    fst.set_final(start, TropicalWeight::one());
    fst.set_final(back_state, TropicalWeight::one());
    fst.set_final(front_state, TropicalWeight::one());

    // Back vowels trigger back harmony
    let back_vowels = ['a', 'o', 'u'];
    let front_vowels = ['e', 'i'];

    for &vowel in &back_vowels {
        fst.add_arc(
            start,
            Arc::new(
                vowel as u32,
                vowel as u32,
                TropicalWeight::one(),
                back_state,
            ),
        );
        fst.add_arc(
            back_state,
            Arc::new(
                vowel as u32,
                vowel as u32,
                TropicalWeight::one(),
                back_state,
            ),
        );
    }

    for &vowel in &front_vowels {
        fst.add_arc(
            start,
            Arc::new(
                vowel as u32,
                vowel as u32,
                TropicalWeight::one(),
                front_state,
            ),
        );
        fst.add_arc(
            front_state,
            Arc::new(
                vowel as u32,
                vowel as u32,
                TropicalWeight::one(),
                front_state,
            ),
        );
    }

    // Harmonizing vowel: 'E' becomes 'e' in front context, 'a' in back context
    fst.add_arc(
        back_state,
        Arc::new('E' as u32, 'a' as u32, TropicalWeight::one(), back_state),
    );

    fst.add_arc(
        front_state,
        Arc::new('E' as u32, 'e' as u32, TropicalWeight::one(), front_state),
    );

    // Consonants are transparent
    let consonants = "bcdfghjklmnpqrstvwxyz";
    for ch in consonants.chars() {
        fst.add_arc(
            start,
            Arc::new(ch as u32, ch as u32, TropicalWeight::one(), start),
        );
        fst.add_arc(
            back_state,
            Arc::new(ch as u32, ch as u32, TropicalWeight::one(), back_state),
        );
        fst.add_arc(
            front_state,
            Arc::new(ch as u32, ch as u32, TropicalWeight::one(), front_state),
        );
    }

    fst
}

/// Build FST for consonant cluster simplification
/// Example: /kt/ -> /t/ (cluster reduction)
fn build_cluster_simplification_fst() -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    let k_state = fst.add_state();

    fst.set_start(start);
    fst.set_final(start, TropicalWeight::one());

    // /k/ followed by /t/ becomes just /t/
    fst.add_arc(
        start,
        Arc::new(
            'k' as u32,
            0, // epsilon output (delete k)
            TropicalWeight::one(),
            k_state,
        ),
    );

    fst.add_arc(
        k_state,
        Arc::new('t' as u32, 't' as u32, TropicalWeight::one(), start),
    );

    // All other characters pass through unchanged
    for ch in b'a'..=b'z' {
        let ch = ch as char;
        if ch != 'k' {
            fst.add_arc(
                start,
                Arc::new(ch as u32, ch as u32, TropicalWeight::one(), start),
            );
        }
    }

    // k in other contexts passes through
    for ch in b'a'..=b'z' {
        let ch = ch as char;
        if ch != 't' {
            fst.add_arc(
                k_state,
                Arc::new(
                    ch as u32,
                    'k' as u32, // output the k we held
                    TropicalWeight::one(),
                    start,
                ),
            );
            // Then process the current character
            fst.add_arc(
                start,
                Arc::new(ch as u32, ch as u32, TropicalWeight::one(), start),
            );
        }
    }

    fst
}

/// Build FST for vowel epenthesis (insertion)
/// Example: Insert 'i' to break consonant clusters
fn build_epenthesis_fst() -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    let consonant_state = fst.add_state();

    fst.set_start(start);
    fst.set_final(start, TropicalWeight::one());
    fst.set_final(consonant_state, TropicalWeight::one());

    let vowels = "aeiou";
    let consonants = "bcdfghjklmnpqrstvwxyz";

    // Vowels pass through and reset to start
    for ch in vowels.chars() {
        fst.add_arc(
            start,
            Arc::new(ch as u32, ch as u32, TropicalWeight::one(), start),
        );
        fst.add_arc(
            consonant_state,
            Arc::new(ch as u32, ch as u32, TropicalWeight::one(), start),
        );
    }

    // First consonant goes to consonant state
    for ch in consonants.chars() {
        fst.add_arc(
            start,
            Arc::new(ch as u32, ch as u32, TropicalWeight::one(), consonant_state),
        );
    }

    // Second consonant triggers epenthesis
    for ch in consonants.chars() {
        fst.add_arc(
            consonant_state,
            Arc::new(
                ch as u32,
                'i' as u32, // insert epenthetic vowel
                TropicalWeight::one(),
                start,
            ),
        );
        // Then output the consonant
        fst.add_arc(
            start,
            Arc::new(
                0, // epsilon input
                ch as u32,
                TropicalWeight::one(),
                consonant_state,
            ),
        );
    }

    fst
}

/// Build FST for final devoicing (German-style)
/// Example: /d/ -> /t/ / _#
fn build_final_devoicing_fst() -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    let voiced_state = fst.add_state();

    fst.set_start(start);
    fst.set_final(start, TropicalWeight::one());

    // Final voiced consonants become voiceless
    fst.set_final(voiced_state, TropicalWeight::new(0.0)); // Cost for devoicing

    let voiced_obstruents = ['b', 'd', 'g', 'z', 'v'];
    let voiceless_obstruents = ['p', 't', 'k', 's', 'f'];

    // Map voiced to voiceless at word end
    for (&voiced, &voiceless) in voiced_obstruents.iter().zip(voiceless_obstruents.iter()) {
        fst.add_arc(
            start,
            Arc::new(
                voiced as u32,
                voiceless as u32,
                TropicalWeight::one(),
                voiced_state,
            ),
        );
    }

    // All other characters pass through
    for ch in b'a'..=b'z' {
        let ch = ch as char;
        if !voiced_obstruents.contains(&ch) {
            fst.add_arc(
                start,
                Arc::new(ch as u32, ch as u32, TropicalWeight::one(), start),
            );
        }
    }

    // Non-final voiced consonants pass through unchanged
    for ch in b'a'..=b'z' {
        let ch = ch as char;
        fst.add_arc(
            voiced_state,
            Arc::new(ch as u32, ch as u32, TropicalWeight::one(), start),
        );
    }

    fst
}

/// Build FST that accepts a word
fn build_word_fst(word: &str) -> VectorFst<TropicalWeight> {
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

/// Extract output string from FST path
fn extract_output_string(fst: &VectorFst<TropicalWeight>) -> Option<String> {
    if let Some(start) = fst.start() {
        let mut result = String::new();
        let mut current = start;
        let mut visited = std::collections::HashSet::new();

        loop {
            if visited.contains(&current) {
                break;
            }
            visited.insert(current);

            if fst.is_final(current) {
                return Some(result);
            }

            let mut found = false;
            if let Some(arc) = fst.arcs(current).next() {
                if arc.olabel != 0 {
                    result.push(arc.olabel as u8 as char);
                }
                current = arc.nextstate;
                found = true;
            }

            if !found {
                break;
            }
        }
    }

    None
}

/// Apply a sequence of phonological rules via FST composition
fn apply_phonological_rules(input: &str, rules: Vec<VectorFst<TropicalWeight>>) -> Result<String> {
    let mut current_fst = build_word_fst(input);

    println!("Applying phonological rules in sequence:");
    println!("Input: '{}'", input);

    for (i, rule) in rules.iter().enumerate() {
        println!("\nStep {}: Applying rule {}", i + 1, i + 1);

        // Compose current result with next rule
        current_fst = compose_default(&current_fst, rule)?;

        // Extract intermediate result
        if let Some(intermediate) = extract_output_string(&current_fst) {
            println!("  Result: '{}'", intermediate);
        } else {
            println!("  Result: (no output)");
        }
    }

    // Extract final result
    if let Some(output) = extract_output_string(&current_fst) {
        Ok(output)
    } else {
        Ok("(no output)".to_string())
    }
}

fn main() -> Result<()> {
    println!("Phonological Rules with FSTs");
    println!("============================");
    println!("Modeling phonological processes using Finite State Transducers");
    println!("Based on Kaplan & Kay (1994) and subsequent work\n");

    // Example 1: Turkish vowel harmony
    println!("1. Turkish-style Vowel Harmony");
    println!("------------------------------");
    println!("Rule: Suffix vowel 'E' harmonizes with stem vowels");
    println!("  Front vowels (e, i) → E becomes 'e'");
    println!("  Back vowels (a, o, u) → E becomes 'a'");

    let harmony_fst = build_vowel_harmony_fst();
    let harmony_tests = vec!["kitabE", "evE", "adamE", "gelE"];

    for test in harmony_tests {
        let input_fst = build_word_fst(test);
        let composed: VectorFst<TropicalWeight> = compose_default(&input_fst, &harmony_fst)?;
        if let Some(output) = extract_output_string(&composed) {
            println!("  '{}' → '{}'", test, output);
        }
    }

    // Example 2: Consonant cluster simplification
    println!("\n2. Consonant Cluster Simplification");
    println!("-----------------------------------");
    println!("Rule: /kt/ → /t/ (cluster reduction)");

    let cluster_fst = build_cluster_simplification_fst();
    let cluster_tests = vec!["akt", "ekte", "doktor", "katok"];

    for test in cluster_tests {
        let input_fst = build_word_fst(test);
        let composed: VectorFst<TropicalWeight> = compose_default(&input_fst, &cluster_fst)?;
        if let Some(output) = extract_output_string(&composed) {
            println!("  '{}' → '{}'", test, output);
        }
    }

    // Example 3: Vowel epenthesis
    println!("\n3. Vowel Epenthesis");
    println!("------------------");
    println!("Rule: Insert 'i' between consonant clusters");

    let epenthesis_fst = build_epenthesis_fst();
    let epenthesis_tests = vec!["sport", "program", "strong"];

    for test in epenthesis_tests {
        let input_fst = build_word_fst(test);
        let composed: VectorFst<TropicalWeight> = compose_default(&input_fst, &epenthesis_fst)?;
        if let Some(output) = extract_output_string(&composed) {
            println!("  '{}' → '{}'", test, output);
        }
    }

    // Example 4: Final devoicing
    println!("\n4. Final Devoicing (German-style)");
    println!("---------------------------------");
    println!("Rule: Voiced obstruents become voiceless word-finally");

    let devoicing_fst = build_final_devoicing_fst();
    let devoicing_tests = vec!["hund", "tag", "lieb", "haus"];

    for test in devoicing_tests {
        let input_fst = build_word_fst(test);
        let composed: VectorFst<TropicalWeight> = compose_default(&input_fst, &devoicing_fst)?;
        if let Some(output) = extract_output_string(&composed) {
            println!("  '{}' → '{}'", test, output);
        }
    }

    // Example 5: Rule composition and ordering
    println!("\n5. Rule Interaction and Ordering");
    println!("--------------------------------");
    println!("Demonstrating how rule order affects output");

    let test_word = "aktE";
    println!("Input: '{}'", test_word);

    // Order 1: Harmony before cluster simplification
    println!("\nOrder 1: Vowel Harmony → Cluster Simplification");
    let rules1 = vec![harmony_fst.clone(), cluster_fst.clone()];
    let result1 = apply_phonological_rules(test_word, rules1)?;
    println!("Final result: '{}'", result1);

    // Order 2: Cluster simplification before harmony
    println!("\nOrder 2: Cluster Simplification → Vowel Harmony");
    let rules2 = vec![cluster_fst.clone(), harmony_fst.clone()];
    let result2 = apply_phonological_rules(test_word, rules2)?;
    println!("Final result: '{}'", result2);

    // Example 6: Complex rule interaction
    println!("\n6. Complex Multi-Rule System");
    println!("----------------------------");
    println!("Applying multiple rules in sequence");

    let complex_word = "sportE";
    println!("Input: '{}'", complex_word);

    let all_rules = vec![
        epenthesis_fst, // Break consonant clusters first
        harmony_fst,    // Then apply vowel harmony
        devoicing_fst,  // Finally apply final devoicing
    ];

    let final_result = apply_phonological_rules(complex_word, all_rules)?;
    println!("Final result: '{}'", final_result);

    // Theoretical discussion
    println!("\n7. Theoretical Background and Implications");
    println!("-----------------------------------------");
    println!("This example demonstrates key insights from computational phonology:");
    println!("  • Phonological rules as regular relations (Kaplan & Kay, 1994)");
    println!("  • Rule application through FST composition");
    println!("  • Natural emergence of rule ordering effects from composition order");
    println!("  • Modeling of opacity, transparency, and bleeding/feeding interactions");
    println!("  • Bidirectional processing: generation ↔ recognition");
    println!("  • Connection to two-level morphology (Koskenniemi, 1983)");

    println!("\nHistorical development:");
    println!("  • Johnson (1972): Early formal approaches to phonological rules");
    println!("  • Koskenniemi (1983): Two-level morphology with FSTs");
    println!("  • Kaplan & Kay (1994): Regular models of phonological rule systems");
    println!("  • Modern applications: Finite-state phonology in NLP systems");

    println!("\nApplications in computational linguistics:");
    println!("  • Morphophonological analysis and generation");
    println!("  • Text-to-speech synthesis systems");
    println!("  • Automatic speech recognition");
    println!("  • Historical linguistics and sound change modeling");
    println!("  • Language documentation and endangered language preservation");
    println!("  • Cross-linguistic phonological typology studies");
    println!("  • Psycholinguistic modeling of phonological processing");

    Ok(())
}
