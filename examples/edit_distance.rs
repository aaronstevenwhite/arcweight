//! Edit Distance Example
//!
//! This example demonstrates how to use weighted finite state transducers (WFSTs)
//! to compute edit distance between strings. It shows:
//! 1. Building an edit distance transducer with customizable weights for different operations
//! 2. Computing edit distance with uniform weights (standard Levenshtein distance)
//! 3. Computing edit distance with custom weights for insertions, deletions, and substitutions
//! 4. Demonstrating how different weight schemes affect the computed distance
//!
//! Usage:
//! ```bash
//! cargo run --example edit_distance
//! ```

use arcweight::prelude::*;

/// Builds an FST that accepts all strings within a given edit distance of the target string.
/// Uses a standard edit distance lattice construction.
///
/// # Arguments
/// * `target` - The target string
/// * `k` - Maximum allowed edit distance
/// * `insertion_cost` - Cost of inserting a character
/// * `deletion_cost` - Cost of deleting a character  
/// * `substitution_cost` - Cost of substituting a character
///
/// # Returns
/// An FST that accepts strings within distance k of the target
fn build_edit_distance_fst(
    target: &str,
    k: usize,
    insertion_cost: f32,
    deletion_cost: f32,
    substitution_cost: f32,
) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let target_chars: Vec<char> = target.chars().collect();
    let n = target_chars.len();

    // Create states for the edit distance lattice
    // State (i,d) represents matching i characters with d edits
    let mut states = vec![vec![]; n + 1];
    for state_row in states.iter_mut().take(n + 1) {
        for _d in 0..=k {
            state_row.push(fst.add_state());
        }
    }

    // Start state
    fst.set_start(states[0][0]);

    // Final states - reached end of target with <= k edits
    for d in 0..=k {
        fst.set_final(states[n][d], TropicalWeight::one());
    }

    // Add transitions
    for i in 0..=n {
        for d in 0..=k {
            let current = states[i][d];

            if i < n && d <= k {
                // Match - advance both strings with no cost
                fst.add_arc(
                    current,
                    Arc::new(
                        target_chars[i] as u32,
                        target_chars[i] as u32,
                        TropicalWeight::one(),
                        states[i + 1][d],
                    ),
                );

                if d < k {
                    // Deletion - skip character in target
                    fst.add_arc(
                        current,
                        Arc::new(
                            0, // epsilon
                            target_chars[i] as u32,
                            TropicalWeight::new(deletion_cost),
                            states[i + 1][d + 1],
                        ),
                    );

                    // Substitution - replace character
                    for c in b'a'..=b'z' {
                        if c as char != target_chars[i] {
                            fst.add_arc(
                                current,
                                Arc::new(
                                    c as u32,
                                    target_chars[i] as u32,
                                    TropicalWeight::new(substitution_cost),
                                    states[i + 1][d + 1],
                                ),
                            );
                        }
                    }
                }
            }

            if d < k {
                // Insertion - consume input character
                for c in b'a'..=b'z' {
                    fst.add_arc(
                        current,
                        Arc::new(
                            c as u32,
                            0, // epsilon
                            TropicalWeight::new(insertion_cost),
                            states[i][d + 1],
                        ),
                    );
                }
            }
        }
    }

    fst
}

/// Create a simple linear acceptor for a string
fn string_acceptor(s: &str) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let chars: Vec<char> = s.chars().collect();

    let mut current = fst.add_state();
    fst.set_start(current);

    for &c in &chars {
        let next = fst.add_state();
        fst.add_arc(
            current,
            Arc::new(c as u32, c as u32, TropicalWeight::one(), next),
        );
        current = next;
    }

    fst.set_final(current, TropicalWeight::one());
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
    // Maximum possible edits we'll consider
    let max_edits = (source.len() + target.len()).max(10);

    // Build edit distance transducer
    let edit_fst = build_edit_distance_fst(
        target,
        max_edits,
        insertion_cost,
        deletion_cost,
        substitution_cost,
    );

    // Build source string acceptor
    let source_fst = string_acceptor(source);

    // Compose to find if source can be transformed to target
    let composed: VectorFst<TropicalWeight> = compose_default(&source_fst, &edit_fst)?;

    // Find shortest path
    let config = ShortestPathConfig::default();
    let shortest: VectorFst<TropicalWeight> = shortest_path(&composed, config)?;

    // Get the cost from the shortest path
    let mut min_cost = f32::INFINITY;

    if let Some(start) = shortest.start() {
        // Find all paths and their costs
        let mut stack = vec![(start, 0.0)];
        let mut visited = std::collections::HashSet::new();

        while let Some((state, cost)) = stack.pop() {
            if visited.contains(&state) {
                continue;
            }
            visited.insert(state);

            if shortest.is_final(state) {
                min_cost = min_cost.min(cost);
            }

            for arc in shortest.arcs(state) {
                stack.push((arc.nextstate, cost + arc.weight.value()));
            }
        }
    }

    if min_cost.is_finite() {
        Ok(min_cost)
    } else {
        Ok(f32::INFINITY)
    }
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
            println!("  '{}' -> '{}': {}", source, target, distance);
        } else {
            println!("  '{}' -> '{}': No transformation found", source, target);
        }
    }

    // Example 2: Custom weights - insertions are cheap, deletions are expensive
    println!("\n2. Custom Weights (insert=0.5, delete=2.0, substitute=1.0):");
    println!("------------------------------------------------------------");
    for (source, target) in &test_pairs {
        let distance = compute_edit_distance(source, target, 0.5, 2.0, 1.0)?;
        if distance.is_finite() {
            println!("  '{}' -> '{}': {}", source, target, distance);
        } else {
            println!("  '{}' -> '{}': No transformation found", source, target);
        }
    }

    // Example 3: Substitutions are very expensive (encourages insertions/deletions)
    println!("\n3. Expensive Substitutions (insert=1.0, delete=1.0, substitute=3.0):");
    println!("--------------------------------------------------------------------");
    for (source, target) in &test_pairs {
        let distance = compute_edit_distance(source, target, 1.0, 1.0, 3.0)?;
        if distance.is_finite() {
            println!("  '{}' -> '{}': {}", source, target, distance);
        } else {
            println!("  '{}' -> '{}': No transformation found", source, target);
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
                "  {}: {} (ins={}, del={}, sub={})",
                name, distance, ins, del, sub
            );
        } else {
            println!(
                "  {}: No transformation found (ins={}, del={}, sub={})",
                name, ins, del, sub
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
            "  '{}' -> '{}': {} (distance: {})",
            source, target, description, distance
        );
    }

    Ok(())
}
