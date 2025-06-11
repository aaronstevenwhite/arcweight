//! Edit Distance Example
//!
//! This example demonstrates how to use Finite State Transducers (FSTs) to find
//! words within a specified edit distance of a target word. It shows:
//! 1. Building an FST that accepts all strings within edit distance k of a target word
//! 2. Creating a dictionary FST from a list of words
//! 3. Composing FSTs to find matching words
//! 4. Finding the shortest paths to get the best matches
//! 5. Extracting and displaying the results
//!
//! The edit distance FST allows for three types of edits:
//! - Substitution: Replace one character with another (cost: 1)
//! - Insertion: Add a new character (cost: 1)
//! - Deletion: Remove a character (cost: 1)
//!
//! Usage:
//! ```bash
//! cargo run --example edit_distance
//! ```
//!
//! The example will find corrections for the word "helo" within edit distance 2,
//! using a small dictionary of words. The output will show the corrections
//! along with their edit distances.

use arcweight::prelude::*;
use std::collections::HashSet;

/// Builds an FST that accepts all strings within edit distance k of the target word.
/// 
/// The FST is constructed as a grid where:
/// - Each state represents a position in the target word and an edit distance
/// - Transitions represent possible edits (match, substitution, insertion, deletion)
/// - Final states are those that reach the end of the word within the edit distance limit
/// 
/// # Arguments
/// * `target` - The target word to find matches for
/// * `k` - The maximum edit distance allowed
/// 
/// # Returns
/// An FST that accepts all strings within edit distance k of the target word
fn build_edit_distance_fst(target: &str, k: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let target_chars: Vec<char> = target.chars().collect();
    let n = target_chars.len();
    
    // Create states for each position and edit distance
    for _i in 0..=n {
        for _d in 0..=k {
            fst.add_state();
        }
    }
    
    // Set start state
    fst.set_start(0);
    
    // Add transitions for each state
    for i in 0..=n {
        for d in 0..=k {
            let current_state = (i * (k + 1) + d) as u32;
            
            // If we're at the end of the target and within edit distance
            if i == n && d <= k {
                fst.set_final(current_state, TropicalWeight::one());
            }
            
            // If we haven't reached the end
            if i < n {
                // Match transition (cost 0)
                let next_state = ((i + 1) * (k + 1) + d) as u32;
                fst.add_arc(current_state, Arc::new(
                    target_chars[i] as u32,
                    target_chars[i] as u32,
                    TropicalWeight::one(),
                    next_state
                ));
                
                // If we have edits left
                if d < k {
                    // Insertion (cost 1)
                    fst.add_arc(current_state, Arc::new(
                        0,
                        target_chars[i] as u32,
                        TropicalWeight::new(1.0),
                        next_state
                    ));
                    
                    // Deletion (cost 1)
                    fst.add_arc(current_state, Arc::new(
                        target_chars[i] as u32,
                        0,
                        TropicalWeight::new(1.0),
                        next_state
                    ));
                    
                    // Substitution (cost 1)
                    for c in b'a'..=b'z' {
                        if c as char != target_chars[i] {
                            fst.add_arc(current_state, Arc::new(
                                c as u32,
                                target_chars[i] as u32,
                                TropicalWeight::new(1.0),
                                next_state
                            ));
                        }
                    }
                }
            }
        }
    }
    
    fst
}

/// Creates a dictionary FST from a list of words.
/// 
/// The FST is constructed as a trie where:
/// - Each state represents a prefix of one or more words
/// - Transitions represent characters in the words
/// - Final states mark the end of valid words
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
    
    for word in words {
        let mut current_state = start;
        for c in word.chars() {
            let next_state = fst.add_state();
            fst.add_arc(current_state, Arc::new(
                c as u32,
                c as u32,
                TropicalWeight::one(),
                next_state
            ));
            current_state = next_state;
        }
        fst.set_final(current_state, TropicalWeight::one());
    }
    
    fst
}

/// Extracts all paths from an FST along with their weights.
/// 
/// Uses depth-first search to find all paths from the start state to final states.
/// Each path represents a valid word in the FST, and its weight represents the
/// total cost (edit distance) of that word.
/// 
/// # Arguments
/// * `fst` - The FST to extract paths from
/// 
/// # Returns
/// A vector of (word, cost) pairs representing all valid paths in the FST
fn extract_paths(fst: &VectorFst<TropicalWeight>) -> Vec<(String, f32)> {
    let mut paths = Vec::new();
    let mut current_path = Vec::new();
    let mut visited = HashSet::new();
    
    fn dfs(
        fst: &VectorFst<TropicalWeight>,
        state: u32,
        current_path: &mut Vec<char>,
        visited: &mut HashSet<u32>,
        paths: &mut Vec<(String, f32)>,
    ) {
        if visited.contains(&state) {
            return;
        }
        visited.insert(state);
        
        if fst.is_final(state) {
            let word: String = current_path.iter().collect();
            let weight = fst.final_weight(state).unwrap().value();
            paths.push((word, *weight));
        }
        
        for arc in fst.arcs(state) {
            if arc.ilabel != 0 {
                current_path.push(arc.ilabel as u8 as char);
                dfs(fst, arc.nextstate, current_path, visited, paths);
                current_path.pop();
            }
        }
        
        visited.remove(&state);
    }
    
    dfs(fst, fst.start().unwrap(), &mut current_path, &mut visited, &mut paths);
    paths
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create a small dictionary of words to check against
    let dictionary = vec!["hello", "world", "help", "held", "hell"];
    
    // Build the dictionary FST
    let dict_fst = build_dictionary_fst(&dictionary);
    
    // Test word to check for corrections
    let test_word = "helo";
    println!("Finding corrections for '{}':", test_word);
    
    // Create edit distance FST for the test word with maximum edit distance 2
    let edit_fst = build_edit_distance_fst(test_word, 2);
    
    // Compose the edit distance FST with the dictionary to find matching words
    let composed: VectorFst<TropicalWeight> = compose_default(&edit_fst, &dict_fst)?;
    
    // Find the 3 shortest paths in the composed FST
    let mut config = ShortestPathConfig::default();
    config.nshortest = 3;
    let shortest_paths: VectorFst<TropicalWeight> = shortest_path(&composed, config)?;
    
    // Extract and print the corrections with their edit distances
    let corrections = extract_paths(&shortest_paths);
    for (word, cost) in corrections {
        println!("  {} (edit distance: {})", word, cost);
    }
    
    Ok(())
}
