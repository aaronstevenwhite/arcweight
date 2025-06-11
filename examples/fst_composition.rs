//! FST Composition Example
//!
//! This example demonstrates how to compose two Finite State Transducers (FSTs)
//! to create a more complex transducer. It shows:
//! 1. Building FSTs from input strings
//! 2. Composing FSTs using different composition filters
//! 3. Finding the shortest path in the composed FST
//! 4. Extracting and displaying the results
//!
//! Usage:
//! ```bash
//! cargo run --example fst_composition -- "input string" "transformation rules"
//! ```
//!
//! Example:
//! ```bash
//! cargo run --example fst_composition -- "hello" "h->H e->E l->L o->O"
//! ```
//! This will transform "hello" to "HELLO" using the provided rules.

use arcweight::prelude::*;
use std::error::Error;
use std::env;
use std::process;

/// Builds an FST that accepts a single input string
fn build_input_fst(input: &str) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    fst.set_start(start);
    
    let mut current_state = start;
    for c in input.chars() {
        let next_state = fst.add_state();
        // Each arc represents a single character in the input string
        fst.add_arc(current_state, Arc::new(
            c as u32,
            c as u32,  // Output label is now the character itself
            TropicalWeight::one(),
            next_state
        ));
        current_state = next_state;
    }
    fst.set_final(current_state, TropicalWeight::one());
    
    fst
}

/// Builds an FST that implements transformation rules
/// Rules are in the format "a->b c->d" where a->b means "replace a with b"
fn build_rules_fst(rules: &str) -> std::result::Result<VectorFst<TropicalWeight>, Box<dyn Error>> {
    let mut fst = VectorFst::new();
    let start = fst.add_state();
    fst.set_start(start);
    fst.set_final(start, TropicalWeight::one());  // Allow epsilon transitions
    
    // Parse rules
    for rule in rules.split_whitespace() {
        let parts: Vec<&str> = rule.split("->").collect();
        if parts.len() != 2 {
            return Err(format!("Invalid rule format: {}", rule).into());
        }
        
        let input_char = parts[0].chars().next()
            .ok_or_else(|| format!("Empty input in rule: {}", rule))?;
        let output_char = parts[1].chars().next()
            .ok_or_else(|| format!("Empty output in rule: {}", rule))?;
        
        // Add transition for this rule
        fst.add_arc(start, Arc::new(
            input_char as u32,
            output_char as u32,
            TropicalWeight::one(),
            start
        ));
    }
    
    Ok(fst)
}

/// Extracts the output string from a path in an FST
fn extract_output(fst: &VectorFst<TropicalWeight>) -> std::result::Result<String, Box<dyn Error>> {
    let mut output = String::new();
    let mut current_state = fst.start().ok_or_else(|| "FST has no start state")?;
    let mut visited = std::collections::HashSet::new();
    loop {
        if visited.contains(&current_state) {
            break;
        }
        visited.insert(current_state);
        if let Some(weight) = fst.final_weight(current_state) {
            if <TropicalWeight as One>::is_one(weight) {
                // If we're at a final state, return the current output
                return Ok(output);
            }
        }
        let mut found_arc = false;
        for arc in fst.arcs(current_state) {
            if arc.olabel != 0 {
                output.push(arc.olabel as u8 as char);
            }
            current_state = arc.nextstate;
            found_arc = true;
            break;
        }
        if !found_arc {
            break;
        }
    }
    Err("No valid output path found".into())
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <input_string> <transformation_rules>", args[0]);
        eprintln!("Example: {} hello \"h->H e->E l->L o->O\"", args[0]);
        process::exit(1);
    }
    
    let input = &args[1];
    let rules = &args[2];
    
    println!("Input string: {}", input);
    println!("Transformation rules: {}", rules);
    
    // Build the input FST
    let input_fst = build_input_fst(input);
    println!("\nInput FST:");
    println!("Number of states: {}", input_fst.num_states());
    let total_arcs: usize = (0..input_fst.num_states())
        .map(|s| input_fst.arcs(s as u32).count())
        .sum();
    println!("Total number of arcs: {}", total_arcs);
    println!("Start state: {}", input_fst.start().unwrap());
    println!("Final states: {:?}", (0..input_fst.num_states())
        .filter(|&s| input_fst.final_weight(s as u32).is_some())
        .map(|s| s as u32)
        .collect::<Vec<_>>());
    
    // Build the rules FST
    let rules_fst = build_rules_fst(rules)?;
    println!("\nRules FST:");
    println!("Number of states: {}", rules_fst.num_states());
    let total_arcs: usize = (0..rules_fst.num_states())
        .map(|s| rules_fst.arcs(s as u32).count())
        .sum();
    println!("Total number of arcs: {}", total_arcs);
    println!("Start state: {}", rules_fst.start().unwrap());
    println!("Final states: {:?}", (0..rules_fst.num_states())
        .filter(|&s| rules_fst.final_weight(s as u32).is_some())
        .map(|s| s as u32)
        .collect::<Vec<_>>());
    
    // Compose the FSTs
    let composed: VectorFst<TropicalWeight> = compose_default(&input_fst, &rules_fst)?;
    println!("\nComposed FST:");
    println!("Number of states: {}", composed.num_states());
    println!("Start state: {:?}", composed.start());
    let final_states: Vec<u32> = (0..composed.num_states())
        .filter(|&s| composed.final_weight(s as u32).is_some())
        .map(|s| s as u32)
        .collect();
    println!("Final states: {:?}", final_states);
    
    // Find the shortest path
    let mut config = ShortestPathConfig::default();
    config.nshortest = 1;
    let shortest_paths: VectorFst<TropicalWeight> = shortest_path(&composed, config)?;
    println!("\nShortest Path FST:");
    println!("Number of states: {}", shortest_paths.num_states());
    println!("Start state: {:?}", shortest_paths.start());
    let final_states: Vec<u32> = (0..shortest_paths.num_states())
        .filter(|&s| shortest_paths.final_weight(s as u32).is_some())
        .map(|s| s as u32)
        .collect();
    println!("Final states: {:?}", final_states);
    
    // Extract and display the result
    let output = extract_output(&shortest_paths)?;
    println!("\nResult: {}", output);
    
    Ok(())
} 