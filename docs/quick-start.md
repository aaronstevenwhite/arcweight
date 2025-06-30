# Quick Start Guide

**Build your first FST in 20 minutes!** From zero to working transducer with hands-on, practical examples that give immediate results.

This guide walks you through creating your first weighted finite state transducer (WFST) and performing essential operations. By the end, you'll understand FST fundamentals and be ready for real-world applications.

> **Learning Style**: Hands-on with immediate results  
> **For Theory**: See [Core Concepts](core-concepts/)  
> **For Overview**: See [Main Documentation](README.md)

## Prerequisites

- Rust 1.75.0+ installed ([Installation Guide](installation.md))
- ArcWeight added to your project
- Basic familiarity with Rust syntax

## What You'll Learn

By the end of this guide, you'll master:

| Skill | What You'll Build | Time |
|-------|------------------|------|
| **FST Construction** | States, arcs, and basic structures | 5 min |
| **Weight Types** | Different semiring applications | 5 min |
| **Operations** | Composition, union, shortest path | 3 min |
| **Symbol Tables** | Readable, maintainable code | 2 min |
| **Real Applications** | Working spell checker foundation | 5 min |

## Your First FST: "Hello World"

Let's start with the simplest possible FST - one that accepts the string "hello":

### Step 1: Create Your First FST

Replace `src/main.rs` with:

```rust
use arcweight::prelude::*;

fn main() -> Result<()> {
    println!("Creating your first FST...");
    
    // Create a new FST with tropical weights (cost-based optimization)
    let mut fst = VectorFst::<TropicalWeight>::new();
    
    // Add states for each character in "hello" + start state
    let s0 = fst.add_state();  // Start state
    let s1 = fst.add_state();  // After 'h'
    let s2 = fst.add_state();  // After 'e'
    let s3 = fst.add_state();  // After 'l'
    let s4 = fst.add_state();  // After 'l'
    let s5 = fst.add_state();  // After 'o' (final)
    
    // Configure start and final states
    fst.set_start(s0);
    fst.set_final(s5, TropicalWeight::one()); // Weight 0.0 = no cost
    
    // Add arcs for each character transition
    // Arc::new(input_label, output_label, weight, next_state)
    fst.add_arc(s0, Arc::new('h' as u32, 'h' as u32, TropicalWeight::one(), s1));
    fst.add_arc(s1, Arc::new('e' as u32, 'e' as u32, TropicalWeight::one(), s2));
    fst.add_arc(s2, Arc::new('l' as u32, 'l' as u32, TropicalWeight::one(), s3));
    fst.add_arc(s3, Arc::new('l' as u32, 'l' as u32, TropicalWeight::one(), s4));
    fst.add_arc(s4, Arc::new('o' as u32, 'o' as u32, TropicalWeight::one(), s5));
    
    // Print FST statistics
    // Count total arcs across all states
    let total_arcs: usize = fst.states().map(|s| fst.num_arcs(s)).sum();
    
    println!("✅ FST created successfully!");
    println!("   States: {}", fst.num_states());
    println!("   Arcs: {}", total_arcs);
    println!("   Start state: {:?}", fst.start());
    
    // Examine the structure
    println!("\\nFST Structure:");
    for state in fst.states() {
        print!("State {}: ", state);
        
        // Show arcs from this state
        for arc in fst.arcs(state) {
            let input_char = char::from_u32(arc.ilabel).unwrap_or('?');
            let output_char = char::from_u32(arc.olabel).unwrap_or('?');
            print!("'{}':'{}' -> {} ", input_char, output_char, arc.nextstate);
        }
        
        // Show if this is a final state
        if let Some(weight) = fst.final_weight(state) {
            print!("[FINAL: {}]", weight.value());
        }
        println!();
    }
    
    Ok(())
}
```

Run it:

```bash
cargo run
```

Expected output:
```
Creating your first FST...
✅ FST created successfully!
   States: 6
   Arcs: 5
   Start state: Some(0)

FST Structure:
State 0: 'h':'h' -> 1 
State 1: 'e':'e' -> 2 
State 2: 'l':'l' -> 3 
State 3: 'l':'l' -> 4 
State 4: 'o':'o' -> 5 
State 5: [FINAL: 0]
```

**Congratulations!** You've created your first FST that recognizes the word "hello".

## Understanding FST Components

Master the building blocks that make FSTs powerful: **States** • **Arcs** • **Weights**

Now that you've seen an FST in action, let's understand the fundamental building blocks:

### States

States are the **nodes** in your FST - they track progress through the input:
- **Start**: Where processing begins (one per FST)
- **Final**: Where processing successfully ends (one or more)
- **Intermediate**: Handle transformations between start and final

### Arcs

Arcs are the **edges** connecting states that define transformations:
- **Input Label**: Symbol consumed from input
- **Output Label**: Symbol produced to output  
- **Weight**: Cost or probability of this transition
- **Target State**: Next state to move to

### Weights

Weights define how costs combine using **semiring math** - see the detailed weight types section below.


## Weight Types (Semirings)

Weights define how costs combine in FST operations. ArcWeight supports several types:

**TropicalWeight** - Most common for optimization:
- **Addition**: `min(a, b)` - choose cheapest path
- **Multiplication**: `a + b` - accumulate costs
- **Perfect for**: shortest paths, edit distance, spell checking

**Other semirings**:
- **ProbabilityWeight**: For probabilistic operations
- **BooleanWeight**: Simple accept/reject logic

> **Learn more semirings**: See [Core Concepts](core-concepts/) for mathematical details and [Working with FSTs](working-with-fsts/) for practical examples.

## Basic FST Operations

FSTs become powerful when you combine them. Composition is one of the most important operations. Composition chains two FSTs - the output of the first becomes input to the second:

```rust
use arcweight::prelude::*;

fn composition_example() -> Result<()> {
    // FST 1: "a" -> "b"
    let mut fst1 = VectorFst::<TropicalWeight>::new();
    let s0 = fst1.add_state();
    let s1 = fst1.add_state();
    fst1.set_start(s0);
    fst1.set_final(s1, TropicalWeight::one());
    fst1.add_arc(s0, Arc::new('a' as u32, 'b' as u32, TropicalWeight::one(), s1));
    
    // FST 2: "b" -> "c"  
    let mut fst2 = VectorFst::<TropicalWeight>::new();
    let s0 = fst2.add_state();
    let s1 = fst2.add_state();
    fst2.set_start(s0);
    fst2.set_final(s1, TropicalWeight::one());
    fst2.add_arc(s0, Arc::new('b' as u32, 'c' as u32, TropicalWeight::one(), s1));
    
    // Compose: "a" -> "b" -> "c" becomes "a" -> "c"
    let composed: VectorFst<TropicalWeight> = compose_default(&fst1, &fst2)?;
    
    println!("Composed FST: 'a' -> 'c' directly");
    Ok(())
}
```

> **Learn more operations**: See [Working with FSTs](working-with-fsts/) for union, shortest path, minimization, and advanced techniques.

## Symbol Tables (Optional)

For readable code, use symbol tables to map strings to numeric IDs:

```rust
let mut syms = SymbolTable::new();
let hello_id = syms.add_symbol("hello");
// Use hello_id in FST construction instead of 'h' as u32
```

> **Learn more**: See [Working with FSTs](working-with-fsts/) for detailed symbol table usage.

## Practical Application: Simple Spell Checker

Let's build something useful: a simple spell checker that suggests corrections.

```rust
use arcweight::prelude::*;

fn build_spell_checker() -> Result<()> {
    println!("Building a spell checker...");
    
    // Create dictionary FST
    let mut dictionary = VectorFst::<TropicalWeight>::new();
    
    // Build trie structure for words: "cat", "car", "care"
    let root = dictionary.add_state();
    dictionary.set_start(root);
    
    // Shared "ca" prefix
    let ca_state = dictionary.add_state();
    dictionary.add_arc(root, Arc::new('c' as u32, 'c' as u32, TropicalWeight::one(), ca_state));
    let after_ca = dictionary.add_state();
    dictionary.add_arc(ca_state, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), after_ca));
    
    // Branch for each word ending
    // "cat"
    let cat_final = dictionary.add_state();
    dictionary.add_arc(after_ca, Arc::new('t' as u32, 't' as u32, TropicalWeight::one(), cat_final));
    dictionary.set_final(cat_final, TropicalWeight::one());
    
    // "car"  
    let car_final = dictionary.add_state();
    dictionary.add_arc(after_ca, Arc::new('r' as u32, 'r' as u32, TropicalWeight::one(), car_final));
    dictionary.set_final(car_final, TropicalWeight::one());
    
    // "care"
    let care_mid = dictionary.add_state();
    dictionary.add_arc(after_ca, Arc::new('r' as u32, 'r' as u32, TropicalWeight::one(), care_mid));
    let care_final = dictionary.add_state();
    dictionary.add_arc(care_mid, Arc::new('e' as u32, 'e' as u32, TropicalWeight::one(), care_final));
    dictionary.set_final(care_final, TropicalWeight::one());
    
    println!("Dictionary created with {} states", dictionary.num_states());
    println!("Accepts: 'cat', 'car', 'care'");
    
    // Optimize the FST
    let minimized = minimize(&dictionary)?;
    println!("Minimized to {} states", minimized.num_states());
    
    // In a real spell checker, you would:
    // 1. Create an edit distance FST (allows character errors)
    // 2. Compose edit distance FST with dictionary  
    // 3. Find shortest paths to get correction suggestions
    
    println!("✅ Spell checker foundation complete!");
    
    Ok(())
}

fn main() -> Result<()> {
    build_spell_checker()
}
```  

## Next Steps

**Choose your learning path:**

- **📖 [Examples](examples/README.md)** - Practical applications and use cases
- **🛠️ [Working with FSTs](working-with-fsts/)** - Complete operations guide  
- **📐 [Core Concepts](core-concepts/)** - Mathematical foundations
- **🔧 [Try examples](examples/)**: `cargo run --example edit_distance`

## Quick Reference

```rust
use arcweight::prelude::*;  // Essential import

// Basic FST pattern
let mut fst = VectorFst::<TropicalWeight>::new();
let state = fst.add_state();
fst.set_start(state);
fst.set_final(state, TropicalWeight::one());
fst.add_arc(from_state, Arc::new(input, output, weight, to_state));

// Operations
let result = compose_default(&fst1, &fst2)?;
```