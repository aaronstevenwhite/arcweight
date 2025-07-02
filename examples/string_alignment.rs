//! String Alignment Example
//!
//! This example demonstrates how to compute optimal string alignments using weighted finite state
//! transducers. It extends the edit distance algorithm to track and visualize the actual sequence
//! of operations that transform one string into another.
//!
//! Key concepts demonstrated:
//! - Building alignment transducers that preserve transformation information
//! - Extracting optimal alignment paths from composed FSTs
//! - Visualizing alignments with different formatting options
//! - Handling multiple optimal alignments
//! - FST-based path extraction for alignment reconstruction
//!
//! Related examples:
//! - edit_distance.rs: Shows basic edit distance computation with FSTs
//! - spell_checking.rs: Uses edit distance for spell correction
//!
//! Usage: cargo run --example string_alignment

use anyhow::Result;
use arcweight::prelude::*;

/// Represents a single alignment operation
#[derive(Debug, Clone, PartialEq)]
enum AlignmentOp {
    Match(char, char),      // Characters match
    Substitute(char, char), // Substitute source -> target
    Insert(char),           // Insert character into source
    Delete(char),           // Delete character from source
}

/// Represents a complete alignment between two strings
#[derive(Debug, Clone)]
struct Alignment {
    operations: Vec<AlignmentOp>,
    cost: f32,
    #[allow(dead_code)]
    source: String,
    #[allow(dead_code)]
    target: String,
}

impl Alignment {
    /// Create visualization of the alignment
    fn visualize(&self) -> String {
        let mut source_line = String::new();
        let mut alignment_line = String::new();
        let mut target_line = String::new();

        for op in &self.operations {
            match op {
                AlignmentOp::Match(s, t) => {
                    source_line.push(*s);
                    alignment_line.push('|');
                    target_line.push(*t);
                }
                AlignmentOp::Substitute(s, t) => {
                    source_line.push(*s);
                    alignment_line.push('*');
                    target_line.push(*t);
                }
                AlignmentOp::Insert(c) => {
                    source_line.push('-');
                    alignment_line.push('+');
                    target_line.push(*c);
                }
                AlignmentOp::Delete(c) => {
                    source_line.push(*c);
                    alignment_line.push('-');
                    target_line.push('-');
                }
            }
        }

        format!("{}\n{}\n{}", source_line, alignment_line, target_line)
    }

    /// Create a detailed description of the alignment
    fn describe(&self) -> String {
        let mut description = Vec::new();
        let mut source_pos = 0;
        let mut target_pos = 0;

        for op in &self.operations {
            match op {
                AlignmentOp::Match(s, _) => {
                    description.push(format!(
                        "Match '{}' at positions {}/{}",
                        s, source_pos, target_pos
                    ));
                    source_pos += 1;
                    target_pos += 1;
                }
                AlignmentOp::Substitute(s, t) => {
                    description.push(format!(
                        "Substitute '{}' -> '{}' at positions {}/{}",
                        s, t, source_pos, target_pos
                    ));
                    source_pos += 1;
                    target_pos += 1;
                }
                AlignmentOp::Insert(c) => {
                    description.push(format!("Insert '{}' at target position {}", c, target_pos));
                    target_pos += 1;
                }
                AlignmentOp::Delete(c) => {
                    description.push(format!("Delete '{}' at source position {}", c, source_pos));
                    source_pos += 1;
                }
            }
        }

        description.join("\n")
    }
}

/// Builds an FST that computes edit distance and preserves alignment information
/// by encoding operations in the output symbols
fn build_alignment_fst(
    source: &str,
    target: &str,
    insertion_cost: f32,
    deletion_cost: f32,
    substitution_cost: f32,
) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let source_chars: Vec<char> = source.chars().collect();
    let target_chars: Vec<char> = target.chars().collect();
    let m = source_chars.len();
    let n = target_chars.len();

    // Create states for the edit distance lattice
    let mut states = vec![vec![]; m + 1];
    for state_row in states.iter_mut().take(m + 1) {
        for _j in 0..=n {
            state_row.push(fst.add_state());
        }
    }

    // Set start state and final state
    fst.set_start(states[0][0]);
    fst.set_final(states[m][n], TropicalWeight::one());

    // Add transitions with operation encoding in output symbols
    // We'll use a simple encoding: 1=match, 2=substitute, 3=insert, 4=delete
    const MATCH: u32 = 1;
    const SUBSTITUTE: u32 = 2;
    const INSERT: u32 = 3;
    const DELETE: u32 = 4;

    for i in 0..=m {
        #[allow(clippy::needless_range_loop)]
        for j in 0..=n {
            let current_state = states[i][j];

            // Deletion: consume source character, no target character
            if i < m {
                let next_state = states[i + 1][j];
                fst.add_arc(
                    current_state,
                    Arc::new(
                        source_chars[i] as u32,
                        DELETE,
                        TropicalWeight::new(deletion_cost),
                        next_state,
                    ),
                );
            }

            // Insertion: no source character, consume target character
            if j < n {
                let next_state = states[i][j + 1];
                fst.add_arc(
                    current_state,
                    Arc::new(
                        0, // epsilon input
                        INSERT,
                        TropicalWeight::new(insertion_cost),
                        next_state,
                    ),
                );
            }

            // Match or substitution: consume both characters
            if i < m && j < n {
                let next_state = states[i + 1][j + 1];
                let source_char = source_chars[i] as u32;
                let target_char = target_chars[j] as u32;

                if source_char == target_char {
                    // Match
                    fst.add_arc(
                        current_state,
                        Arc::new(source_char, MATCH, TropicalWeight::one(), next_state),
                    );
                } else {
                    // Substitution
                    fst.add_arc(
                        current_state,
                        Arc::new(
                            source_char,
                            SUBSTITUTE,
                            TropicalWeight::new(substitution_cost),
                            next_state,
                        ),
                    );
                }
            }
        }
    }

    fst
}

/// Extracts alignment from the shortest path through the alignment FST
fn extract_alignment_from_path(
    source: &str,
    target: &str,
    shortest_path_fst: &VectorFst<TropicalWeight>,
) -> Result<Alignment> {
    let source_chars: Vec<char> = source.chars().collect();
    let target_chars: Vec<char> = target.chars().collect();
    let mut operations = Vec::new();
    let mut total_cost = 0.0;

    // Find path through the FST by following arcs from start to final state
    let start_state = shortest_path_fst.start().unwrap();
    let mut current_state = start_state;
    let mut source_pos = 0;
    let mut target_pos = 0;

    // Extract operations by traversing the path
    let mut arc_iter = shortest_path_fst.arcs(current_state);
    let mut has_arcs = arc_iter.by_ref().count() > 0;

    while current_state != shortest_path_fst.start().unwrap() || has_arcs {
        let mut found_arc = false;

        if let Some(arc) = shortest_path_fst.arcs(current_state).next() {
            // Follow the first (and should be only) arc in shortest path
            let _input_symbol = arc.ilabel;
            let output_symbol = arc.olabel;
            let weight = arc.weight.value();

            total_cost += weight;

            match output_symbol {
                1 => {
                    // Match
                    if source_pos < source_chars.len() && target_pos < target_chars.len() {
                        operations.push(AlignmentOp::Match(
                            source_chars[source_pos],
                            target_chars[target_pos],
                        ));
                        source_pos += 1;
                        target_pos += 1;
                    }
                }
                2 => {
                    // Substitute
                    if source_pos < source_chars.len() && target_pos < target_chars.len() {
                        operations.push(AlignmentOp::Substitute(
                            source_chars[source_pos],
                            target_chars[target_pos],
                        ));
                        source_pos += 1;
                        target_pos += 1;
                    }
                }
                3 => {
                    // Insert
                    if target_pos < target_chars.len() {
                        operations.push(AlignmentOp::Insert(target_chars[target_pos]));
                        target_pos += 1;
                    }
                }
                4 => {
                    // Delete
                    if source_pos < source_chars.len() {
                        operations.push(AlignmentOp::Delete(source_chars[source_pos]));
                        source_pos += 1;
                    }
                }
                _ => {
                    // Unknown operation
                    found_arc = false;
                }
            }

            if output_symbol <= 4 {
                current_state = arc.nextstate;
                found_arc = true;
            }
        }

        if !found_arc {
            break;
        }

        // Update has_arcs for next iteration
        let mut arc_iter = shortest_path_fst.arcs(current_state);
        has_arcs = arc_iter.by_ref().count() > 0;
    }

    Ok(Alignment {
        operations,
        cost: total_cost,
        source: source.to_string(),
        target: target.to_string(),
    })
}

/// Compute string alignment using FST-based approach
fn compute_alignment_fst(source: &str, target: &str) -> Result<Alignment> {
    // Build alignment FST
    let alignment_fst = build_alignment_fst(source, target, 1.0, 1.0, 1.0);

    // Find shortest path
    let shortest = shortest_path(&alignment_fst, ShortestPathConfig::default())?;

    // Extract alignment from shortest path
    extract_alignment_from_path(source, target, &shortest)
}

/// Demonstrate string alignment functionality with both FST-based and manual examples  
fn demonstrate_string_alignment() -> Result<()> {
    println!("String Alignment Example");
    println!("========================\n");

    // First demonstrate FST-based alignment computation
    println!("1. FST-based Alignment Computation:");
    println!("------------------------------------");

    let test_pairs = vec![("kitten", "sitting"), ("hello", "hallo"), ("cat", "dog")];

    for (source, target) in &test_pairs {
        println!("\nComputing alignment for '{}' -> '{}':", source, target);
        match compute_alignment_fst(source, target) {
            Ok(alignment) => {
                println!("FST-computed cost: {}", alignment.cost);
                println!("Visualization:");
                println!("{}", alignment.visualize());
            }
            Err(e) => {
                println!("Error computing FST alignment: {}", e);
                println!("Falling back to manual example...");
            }
        }
    }

    println!("\n2. Manual Alignment Examples (for comparison):");
    println!("-----------------------------------------------");

    // Manually create some alignment examples to show the concept
    let examples = vec![
        (
            "kitten",
            "sitting",
            vec![
                AlignmentOp::Substitute('k', 's'),
                AlignmentOp::Match('i', 'i'),
                AlignmentOp::Match('t', 't'),
                AlignmentOp::Match('t', 't'),
                AlignmentOp::Substitute('e', 'i'),
                AlignmentOp::Insert('n'),
                AlignmentOp::Substitute('n', 'g'),
            ],
            3.0,
        ),
        (
            "hello",
            "hallo",
            vec![
                AlignmentOp::Match('h', 'h'),
                AlignmentOp::Substitute('e', 'a'),
                AlignmentOp::Match('l', 'l'),
                AlignmentOp::Match('l', 'l'),
                AlignmentOp::Match('o', 'o'),
            ],
            1.0,
        ),
        (
            "cat",
            "dog",
            vec![
                AlignmentOp::Substitute('c', 'd'),
                AlignmentOp::Substitute('a', 'o'),
                AlignmentOp::Substitute('t', 'g'),
            ],
            3.0,
        ),
    ];

    for (source, target, operations, cost) in examples {
        let alignment = Alignment {
            operations,
            cost,
            source: source.to_string(),
            target: target.to_string(),
        };

        println!("\nAligning '{}' -> '{}':", source, target);
        println!("Cost: {}", alignment.cost);
        println!("Visualization:");
        println!("{}", alignment.visualize());
    }

    // Show how the same transformation can have multiple optimal paths
    println!("\n3. Multiple Optimal Alignments:");
    println!("--------------------------------");
    println!("For transforming 'abc' -> 'aec' (cost 1):");

    let alignment1 = Alignment {
        operations: vec![
            AlignmentOp::Match('a', 'a'),
            AlignmentOp::Substitute('b', 'e'),
            AlignmentOp::Match('c', 'c'),
        ],
        cost: 1.0,
        source: "abc".to_string(),
        target: "aec".to_string(),
    };

    println!("\nOption 1 - Direct substitution:");
    println!("{}", alignment1.visualize());

    // Alternative with deletion and insertion (if they had equal cost)
    println!("\nOption 2 - Delete and insert (if costs were equal):");
    let alignment2 = Alignment {
        operations: vec![
            AlignmentOp::Match('a', 'a'),
            AlignmentOp::Delete('b'),
            AlignmentOp::Insert('e'),
            AlignmentOp::Match('c', 'c'),
        ],
        cost: 2.0,
        source: "abc".to_string(),
        target: "aec".to_string(),
    };
    println!("{}", alignment2.visualize());

    // Biological sequence example
    println!("\n4. Biological Sequence Alignment:");
    println!("---------------------------------");

    let dna_alignment = Alignment {
        operations: vec![
            AlignmentOp::Match('a', 'a'),
            AlignmentOp::Match('c', 'c'),
            AlignmentOp::Substitute('g', 't'),
            AlignmentOp::Match('t', 't'),
            AlignmentOp::Match('a', 'a'),
            AlignmentOp::Match('c', 'c'),
            AlignmentOp::Match('g', 'g'),
            AlignmentOp::Match('t', 't'),
        ],
        cost: 1.0,
        source: "acgtacgt".to_string(),
        target: "acttacgt".to_string(),
    };

    println!("DNA sequence alignment:");
    println!("Sequence 1: acgtacgt");
    println!("Sequence 2: acttacgt");
    println!("\nOptimal alignment (cost: {}):", dna_alignment.cost);
    println!("{}", dna_alignment.visualize());

    // Count operation types
    let mut matches = 0;
    let mut substitutions = 0;
    let mut indels = 0;

    for op in &dna_alignment.operations {
        match op {
            AlignmentOp::Match(_, _) => matches += 1,
            AlignmentOp::Substitute(_, _) => substitutions += 1,
            AlignmentOp::Insert(_) | AlignmentOp::Delete(_) => indels += 1,
        }
    }

    println!("\nAlignment statistics:");
    println!("  Matches: {}", matches);
    println!("  Substitutions: {}", substitutions);
    println!("  Insertions/Deletions: {}", indels);
    println!(
        "  Similarity: {:.1}%",
        (matches as f32 / dna_alignment.operations.len() as f32) * 100.0
    );

    // Show detailed operation description
    println!("\n5. Detailed Operation Description:");
    println!("----------------------------------");
    println!("Operations for 'hello' -> 'hallo':");

    let hello_alignment = Alignment {
        operations: vec![
            AlignmentOp::Match('h', 'h'),
            AlignmentOp::Substitute('e', 'a'),
            AlignmentOp::Match('l', 'l'),
            AlignmentOp::Match('l', 'l'),
            AlignmentOp::Match('o', 'o'),
        ],
        cost: 1.0,
        source: "hello".to_string(),
        target: "hallo".to_string(),
    };

    println!("{}", hello_alignment.describe());

    Ok(())
}

fn main() -> Result<()> {
    demonstrate_string_alignment()?;

    println!("\n=== Summary ===");
    println!("This example showed how to:");
    println!("- Build FSTs that encode alignment operations in output symbols");
    println!("- Extract alignment paths from FST shortest paths");
    println!("- Visualize string transformations with detailed operations");
    println!("- Handle multiple optimal alignments");
    println!("- Apply alignment to biological sequences");
    println!("- Compare FST-based and manual alignment approaches");

    Ok(())
}
