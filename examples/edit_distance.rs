//! Edit Distance Example
//!
//! This example demonstrates how to use weighted finite state transducers (WFSTs)
//! to compute edit distance between strings. It shows:
//! 1. Building an edit distance transducer with customizable weights for different operations
//! 2. Computing edit distance with uniform weights (standard Levenshtein distance)
//! 3. Computing edit distance with custom weights for insertions, deletions, and substitutions
//! 4. Demonstrating how different weight schemes affect the computed distance
//!
//! Related examples:
//! - string_alignment.rs: Extends this with path extraction and alignment visualization
//! - spell_checking.rs: Uses edit distance for spell correction applications
//!
//! Usage:
//! ```bash
//! cargo run --example edit_distance
//! ```

use arcweight::prelude::*;

/// Builds an FST that computes edit distance between strings using dynamic programming.
/// This creates a simple, correct implementation that handles any characters.
///
/// # Arguments
/// * `source` - The source string
/// * `target` - The target string
/// * `insertion_cost` - Cost of inserting a character
/// * `deletion_cost` - Cost of deleting a character  
/// * `substitution_cost` - Cost of substituting a character
///
/// # Returns
/// An FST that computes the edit distance
fn build_edit_distance_fst(
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
    // State (i,j) represents having processed i chars from source and j chars from target
    let mut states = vec![vec![]; m + 1];
    for state_row in states.iter_mut().take(m + 1) {
        for _j in 0..=n {
            state_row.push(fst.add_state());
        }
    }

    // Start state
    fst.set_start(states[0][0]);

    // Final state - end of both strings
    fst.set_final(states[m][n], TropicalWeight::one());

    // Add transitions
    #[allow(clippy::needless_range_loop)] // Using i,j to index into DP table
    for i in 0..=m {
        for j in 0..=n {
            let current = states[i][j];

            // Match or substitution - advance both strings
            if i < m && j < n {
                let cost = if source_chars[i] == target_chars[j] {
                    0.0 // match
                } else {
                    substitution_cost // substitution
                };
                fst.add_arc(
                    current,
                    Arc::new(
                        source_chars[i] as u32,
                        target_chars[j] as u32,
                        TropicalWeight::new(cost),
                        states[i + 1][j + 1],
                    ),
                );
            }

            // Deletion - advance source only
            if i < m {
                fst.add_arc(
                    current,
                    Arc::new(
                        source_chars[i] as u32,
                        0, // epsilon output
                        TropicalWeight::new(deletion_cost),
                        states[i + 1][j],
                    ),
                );
            }

            // Insertion - advance target only
            if j < n {
                fst.add_arc(
                    current,
                    Arc::new(
                        0, // epsilon input
                        target_chars[j] as u32,
                        TropicalWeight::new(insertion_cost),
                        states[i][j + 1],
                    ),
                );
            }
        }
    }

    fst
}

/// Compute edit distance between two strings with custom weights
fn compute_edit_distance(
    source: &str,
    target: &str,
    insertion_cost: f32,
    deletion_cost: f32,
    substitution_cost: f32,
) -> Result<f32> {
    // Build edit distance FST that directly computes the distance
    let edit_fst = build_edit_distance_fst(
        source,
        target,
        insertion_cost,
        deletion_cost,
        substitution_cost,
    );

    // Find shortest path from start to final state
    let config = ShortestPathConfig::default();
    let shortest: VectorFst<TropicalWeight> = shortest_path(&edit_fst, config)?;

    // Get the cost from the shortest path
    if let Some(start) = shortest.start() {
        // Check if there's a path to a final state
        let mut stack = vec![(start, 0.0)];
        let mut visited = std::collections::HashSet::new();

        while let Some((state, cost)) = stack.pop() {
            if visited.contains(&state) {
                continue;
            }
            visited.insert(state);

            if shortest.is_final(state) {
                return Ok(cost);
            }

            for arc in shortest.arcs(state) {
                stack.push((arc.nextstate, cost + arc.weight.value()));
            }
        }
    }

    // If no path found, return infinity
    Ok(f32::INFINITY)
}

fn main() -> Result<()> {
    println!("Edit Distance Computation Example");
    println!("=================================\n");

    // Test pairs of strings
    let test_pairs = vec![
        ("kitten", "sitting"),
        ("saturday", "sunday"),
        ("hello", "hallo"),
        ("abc", "abc"),
        ("abc", "def"),
        ("", "abc"),
        ("abc", ""),
    ];

    // Example 1: Standard Levenshtein distance (all operations cost 1)
    println!("1. Standard Levenshtein Distance (all operations cost 1.0):");
    println!("-----------------------------------------------------------");
    for (source, target) in &test_pairs {
        let distance = compute_edit_distance(source, target, 1.0, 1.0, 1.0)?;
        if distance.is_finite() {
            println!("  '{source}' -> '{target}': {distance}");
        } else {
            println!("  '{source}' -> '{target}': No transformation found");
        }
    }

    // Example 2: Custom weights - insertions are cheap, deletions are expensive
    println!("\n2. Custom Weights (insert=0.5, delete=2.0, substitute=1.0):");
    println!("------------------------------------------------------------");
    for (source, target) in &test_pairs {
        let distance = compute_edit_distance(source, target, 0.5, 2.0, 1.0)?;
        if distance.is_finite() {
            println!("  '{source}' -> '{target}': {distance}");
        } else {
            println!("  '{source}' -> '{target}': No transformation found");
        }
    }

    // Example 3: Substitutions are very expensive (encourages insertions/deletions)
    println!("\n3. Expensive Substitutions (insert=1.0, delete=1.0, substitute=3.0):");
    println!("--------------------------------------------------------------------");
    for (source, target) in &test_pairs {
        let distance = compute_edit_distance(source, target, 1.0, 1.0, 3.0)?;
        if distance.is_finite() {
            println!("  '{source}' -> '{target}': {distance}");
        } else {
            println!("  '{source}' -> '{target}': No transformation found");
        }
    }

    // Example 4: Detailed analysis of one transformation
    println!("\n4. Detailed Analysis: 'kitten' -> 'sitting'");
    println!("--------------------------------------------");
    println!("This transformation requires:");
    println!("- Substitute 'k' -> 's'");
    println!("- Insert 'i' after 's'");
    println!("- Keep 'itten' -> 'itting' (substitute 'e' -> 'i')");
    println!("- Add 'g' at the end\n");

    let configs = vec![
        ("Uniform weights", 1.0, 1.0, 1.0),
        ("Cheap insertions", 0.5, 1.0, 1.0),
        ("Cheap deletions", 1.0, 0.5, 1.0),
        ("Cheap substitutions", 1.0, 1.0, 0.5),
    ];

    for (name, ins, del, sub) in configs {
        let distance = compute_edit_distance("kitten", "sitting", ins, del, sub)?;
        if distance.is_finite() {
            println!(
                "  {name}: {distance} (ins={ins}, del={del}, sub={sub})"
            );
        } else {
            println!(
                "  {name}: No transformation found (ins={ins}, del={del}, sub={sub})"
            );
        }
    }

    // Example 5: Character-by-character transformations
    println!("\n5. Step-by-step transformations:");
    println!("---------------------------------");
    let examples = vec![
        ("cat", "cut", "Substitute 'a' -> 'u'"),
        ("cat", "cats", "Insert 's' at end"),
        ("cats", "cat", "Delete 's' from end"),
        ("cat", "dog", "Replace all characters"),
    ];

    for (source, target, description) in examples {
        let distance = compute_edit_distance(source, target, 1.0, 1.0, 1.0)?;
        println!(
            "  '{source}' -> '{target}': {description} (distance: {distance})"
        );
    }

    Ok(())
}
