//! Spell Checking Example
//!
//! This example demonstrates how to use Finite State Transducers (FSTs) to find
//! spelling corrections within a specified edit distance. It shows:
//! 1. Building a dictionary FST from a list of words
//! 2. Creating an edit distance FST that accepts strings within distance k
//! 3. Composing FSTs to find matching words
//! 4. Finding the shortest paths to get the best corrections
//! 5. Extracting and ranking results by edit distance
//!
//! This is a practical application showing how FSTs can be used for spell
//! checking and fuzzy string matching in real-world applications.
//!
//! Related examples:
//! - edit_distance.rs: Shows the basic edit distance computation that this builds upon
//! - string_alignment.rs: Shows how to visualize the actual transformations
//!
//! Usage:
//! ```bash
//! cargo run --example spell_checking
//! ```

use arcweight::prelude::*;
use std::collections::HashMap;

/// Creates a dictionary FST from a list of words using a trie structure.
///
/// # Arguments
/// * `words` - A slice of words to include in the dictionary
///
/// # Returns
/// An FST that accepts exactly the words in the input list
fn build_dictionary_fst(words: &[&str]) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    fst.set_start(start);

    // Build a trie structure
    let mut state_map: HashMap<Vec<char>, u32> = HashMap::new();
    state_map.insert(vec![], start);

    for word in words {
        let chars: Vec<char> = word.chars().collect();
        let mut prefix = vec![];

        for (i, &ch) in chars.iter().enumerate() {
            let current_state = *state_map.get(&prefix).unwrap();
            prefix.push(ch);

            if !state_map.contains_key(&prefix) {
                let new_state = fst.add_state();
                state_map.insert(prefix.clone(), new_state);
                fst.add_arc(
                    current_state,
                    Arc::new(ch as u32, ch as u32, TropicalWeight::one(), new_state),
                );
            }

            // If this is the last character, mark as final
            if i == chars.len() - 1 {
                let final_state = *state_map.get(&prefix).unwrap();
                fst.set_final(final_state, TropicalWeight::one());
            }
        }
    }

    fst
}

/// Creates an FST that accepts all words within edit distance k of a target word
///
/// # Arguments
/// * `target` - The target word to match against
/// * `k` - Maximum allowed edit distance
///
/// # Returns
/// An FST that accepts words within edit distance k of target
fn build_edit_distance_fst(target: &str, k: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let target_chars: Vec<char> = target.chars().collect();
    let n = target_chars.len();

    // Create states: (position in target, edits used)
    let mut states = vec![vec![]; n + 1];
    for (i, state_row) in states.iter_mut().enumerate().take(n + 1) {
        for _j in 0..=k.min(i + k) {
            state_row.push(fst.add_state());
        }
    }

    // Start state
    fst.set_start(states[0][0]);

    // Final states - at end of target with <= k edits
    for j in 0..=k.min(n + k) {
        if j < states[n].len() {
            fst.set_final(states[n][j], TropicalWeight::new(j as f32));
        }
    }

    // Add transitions
    for i in 0..n {
        for j in 0..states[i].len() {
            if j > i + k {
                continue; // Skip impossible states
            }

            let current = states[i][j];

            // Match (no cost)
            if j < states[i + 1].len() {
                fst.add_arc(
                    current,
                    Arc::new(
                        target_chars[i] as u32,
                        target_chars[i] as u32,
                        TropicalWeight::one(),
                        states[i + 1][j],
                    ),
                );
            }

            // If we can still make edits
            if j < k {
                // Substitution (cost 1)
                if j + 1 < states[i + 1].len() {
                    for c in b'a'..=b'z' {
                        if c as char != target_chars[i] {
                            fst.add_arc(
                                current,
                                Arc::new(
                                    c as u32,
                                    c as u32,
                                    TropicalWeight::new(1.0),
                                    states[i + 1][j + 1],
                                ),
                            );
                        }
                    }
                }

                // Deletion in target (consume target char with epsilon)
                if j + 1 < states[i + 1].len() {
                    fst.add_arc(
                        current,
                        Arc::new(
                            0, // epsilon
                            0, // epsilon
                            TropicalWeight::new(1.0),
                            states[i + 1][j + 1],
                        ),
                    );
                }

                // Insertion (consume input char)
                if j + 1 < states[i].len() {
                    for c in b'a'..=b'z' {
                        fst.add_arc(
                            current,
                            Arc::new(
                                c as u32,
                                c as u32,
                                TropicalWeight::new(1.0),
                                states[i][j + 1],
                            ),
                        );
                    }
                }
            }
        }
    }

    // Handle insertions at the end
    for j in 0..states[n].len() {
        if j < k && j + 1 < states[n].len() {
            let current = states[n][j];
            for c in b'a'..=b'z' {
                fst.add_arc(
                    current,
                    Arc::new(
                        c as u32,
                        c as u32,
                        TropicalWeight::new(1.0),
                        states[n][j + 1],
                    ),
                );
            }
        }
    }

    fst
}

/// Finds spelling corrections in dictionary within edit distance of target
fn find_spelling_corrections(
    dict_fst: &VectorFst<TropicalWeight>,
    target: &str,
    max_distance: usize,
) -> Result<Vec<(String, f32)>> {
    // Build edit distance FST
    let edit_fst = build_edit_distance_fst(target, max_distance);

    // Compose dictionary with edit distance FST
    let composed: VectorFst<TropicalWeight> = compose_default(dict_fst, &edit_fst)?;

    // Find shortest paths
    let config = ShortestPathConfig {
        nshortest: 10,
        ..Default::default()
    };
    let shortest: VectorFst<TropicalWeight> = shortest_path(&composed, config)?;

    // Extract words and distances
    let mut results = Vec::new();

    if let Some(start) = shortest.start() {
        extract_paths(&shortest, start, &mut Vec::new(), 0.0, &mut results);
    }

    // Deduplicate results and keep the best score for each word
    let mut word_scores: HashMap<String, f32> = HashMap::new();
    for (word, score) in results {
        word_scores
            .entry(word)
            .and_modify(|e| *e = e.min(score))
            .or_insert(score);
    }

    // Convert back to vec and sort by distance
    let mut final_results: Vec<(String, f32)> = word_scores.into_iter().collect();
    final_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    Ok(final_results)
}

/// Helper to extract paths from FST
fn extract_paths(
    fst: &VectorFst<TropicalWeight>,
    state: u32,
    path: &mut Vec<char>,
    cost: f32,
    results: &mut Vec<(String, f32)>,
) {
    if fst.is_final(state) {
        let word: String = path.iter().collect();
        if let Some(weight) = fst.final_weight(state) {
            results.push((word, cost + weight.value()));
        }
    }

    for arc in fst.arcs(state) {
        if arc.olabel != 0 {
            path.push(arc.olabel as u8 as char);
            extract_paths(fst, arc.nextstate, path, cost + arc.weight.value(), results);
            path.pop();
        } else {
            extract_paths(fst, arc.nextstate, path, cost + arc.weight.value(), results);
        }
    }
}

fn main() -> Result<()> {
    println!("Spell Checking Example");
    println!("======================\n");

    // Build a dictionary of common English words
    let dictionary = vec![
        "hello",
        "world",
        "help",
        "held",
        "hell",
        "hold",
        "hero",
        "here",
        "hear",
        "heap",
        "heal",
        "health",
        "helm",
        "helps",
        "friend",
        "friends",
        "friendship",
        "friendly",
        "fresh",
        "spell",
        "spelling",
        "spelled",
        "spells",
        "special",
        "check",
        "checking",
        "checked",
        "checker",
        "checks",
        "correct",
        "correction",
        "corrected",
        "correctly",
        "corrects",
        "example",
        "examples",
        "exemplary",
        "exempt",
        "exemplify",
    ];

    let dictionary_len = dictionary.len();
    println!("Dictionary contains {dictionary_len} words\n");

    // Build the dictionary FST
    let dict_fst = build_dictionary_fst(&dictionary);

    // Test words with typos
    let test_words = vec![
        ("helo", 2),    // "hello" with one deletion
        ("wrold", 2),   // "world" with transposition
        ("frend", 2),   // "friend" with deletion
        ("chekc", 2),   // "check" with transposition
        ("speling", 2), // "spelling" with deletion
        ("corect", 2),  // "correct" with deletion
        ("exmple", 2),  // "example" with deletion
        ("healht", 2),  // "health" with transposition
    ];

    for (misspelled, max_distance) in test_words {
        println!(
            "Finding spelling corrections for '{misspelled}' (max edit distance: {max_distance}):"
        );
        println!("{}", "-".repeat(50));

        let corrections = find_spelling_corrections(&dict_fst, misspelled, max_distance)?;

        if corrections.is_empty() {
            println!("  No spelling corrections found within edit distance {max_distance}");
        } else {
            for (word, distance) in corrections.iter().take(5) {
                println!("  {word} (distance: {distance})");
            }
        }
        println!();
    }

    // Demonstrate finding words within edit distance 1
    println!("\nWords within edit distance 1 of 'help':");
    println!("{}", "=".repeat(40));

    let corrections = find_spelling_corrections(&dict_fst, "help", 1)?;
    for (word, distance) in corrections {
        if distance <= 1.0 {
            println!("  {word} (distance: {distance})");
        }
    }

    // Show how different edit distances affect results
    println!("\n\nEffect of different edit distances for 'wrld':");
    println!("{}", "=".repeat(50));

    for k in 1..=3 {
        println!("\nEdit distance <= {k}:");
        let corrections = find_spelling_corrections(&dict_fst, "wrld", k)?;
        for (word, distance) in corrections.iter().take(5) {
            println!("  {word} (distance: {distance})");
        }
    }

    Ok(())
}
