# Quick Start

This guide provides an introduction to constructing weighted finite state transducers (WFSTs) using the ArcWeight library.

Prerequisites: [Core Concepts](core-concepts/) | [Installation](installation.md)

## Requirements

- Rust 1.85.0 or later
- ArcWeight dependency in Cargo.toml
- Basic Rust knowledge

## Overview

This guide covers:
- FST construction (states, arcs, weights)
- Semiring weight types
- Basic operations (composition, minimization)
- Symbol tables
- Example: dictionary-based spell checking

## Basic FST Construction

The following example demonstrates FST construction by implementing an acceptor for the string "hello":

```rust,ignore
use arcweight::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create FST with tropical semiring
    let mut fst = VectorFst::<TropicalWeight>::new();
    
    // Add states
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();
    let s3 = fst.add_state();
    let s4 = fst.add_state();
    let s5 = fst.add_state();
    
    // Set start and final states
    fst.set_start(s0);
    fst.set_final(s5, TropicalWeight::one());
    
    // Add transitions
    fst.add_arc(s0, Arc::new('h' as u32, 'h' as u32, TropicalWeight::one(), s1));
    fst.add_arc(s1, Arc::new('e' as u32, 'e' as u32, TropicalWeight::one(), s2));
    fst.add_arc(s2, Arc::new('l' as u32, 'l' as u32, TropicalWeight::one(), s3));
    fst.add_arc(s3, Arc::new('l' as u32, 'l' as u32, TropicalWeight::one(), s4));
    fst.add_arc(s4, Arc::new('o' as u32, 'o' as u32, TropicalWeight::one(), s5));
    
    // Display structure
    for state in fst.states() {
        print!("State {}: ", state);
        for arc in fst.arcs(state) {
            let input = char::from_u32(arc.ilabel).unwrap_or('?');
            let output = char::from_u32(arc.olabel).unwrap_or('?');
            print!("{}:{} -> {} ", input, output, arc.nextstate);
        }
        if let Some(weight) = fst.final_weight(state) {
            print!("[Final: {}]", weight.value());
        }
        println!();
    }
    
    Ok(())
}
```

Output:
```text
State 0: h:h -> 1 
State 1: e:e -> 2 
State 2: l:l -> 3 
State 3: l:l -> 4 
State 4: o:o -> 5 
State 5: [Final: 0]
```

## FST Components

### States
- Start state: Initial state (unique)
- Final states: Accepting states with associated weights
- Intermediate states: Non-final states

### Arcs
Each arc contains:
- Input label (ilabel): Input symbol
- Output label (olabel): Output symbol  
- Weight: Transition cost/probability
- Next state: Target state ID

### Weights
Weights are elements of a semiring algebraic structure. See [Core Concepts](core-concepts/) for theoretical details.


## Semiring Weight Types

ArcWeight implements several semiring types:

### TropicalWeight
- ⊕ (addition): min(a, b)
- ⊗ (multiplication): a + b
- Zero: ∞
- One: 0
- Applications: Shortest path algorithms, edit distance

### ProbabilityWeight
- ⊕: a + b
- ⊗: a × b
- Zero: 0
- One: 1
- Applications: Probabilistic models

### BooleanWeight
- ⊕: a ∨ b
- ⊗: a ∧ b
- Zero: false
- One: true
- Applications: Unweighted automata

## FST Operations

### Composition
Composition (∘) combines two transducers where the output alphabet of the first matches the input alphabet of the second:

```rust,ignore
// T1: a → b
let mut t1 = VectorFst::<TropicalWeight>::new();
let s0 = t1.add_state();
let s1 = t1.add_state();
t1.set_start(s0);
t1.set_final(s1, TropicalWeight::one());
t1.add_arc(s0, Arc::new('a' as u32, 'b' as u32, TropicalWeight::one(), s1));

// T2: b → c  
let mut t2 = VectorFst::<TropicalWeight>::new();
let s0 = t2.add_state();
let s1 = t2.add_state();
t2.set_start(s0);
t2.set_final(s1, TropicalWeight::one());
t2.add_arc(s0, Arc::new('b' as u32, 'c' as u32, TropicalWeight::one(), s1));

// T1 ∘ T2: a → c
let composed: VectorFst<TropicalWeight> = compose_default(&t1, &t2)?;
```

Additional operations include union, concatenation, closure, inversion, minimization, and determinization. See [Working with FSTs](working-with-fsts/) for details.

## Symbol Tables

Symbol tables provide bidirectional mappings between strings and integer labels:

```rust,ignore
let mut syms = SymbolTable::new();
let hello_id = syms.add_symbol("hello");
let world_id = syms.add_symbol("world");

// Use in FST construction
fst.add_arc(s0, Arc::new(hello_id, world_id, weight, s1));
```

## Example: Dictionary FST for Spell Checking

This example constructs a trie-based dictionary FST:

```rust,ignore
fn build_dictionary() -> Result<VectorFst<TropicalWeight>, Box<dyn std::error::Error>> {
    let mut dict = VectorFst::<TropicalWeight>::new();
    
    // Build trie for: "cat", "car", "care"
    let root = dict.add_state();
    dict.set_start(root);
    
    // Common prefix "ca"
    let c_state = dict.add_state();
    let ca_state = dict.add_state();
    dict.add_arc(root, Arc::new('c' as u32, 'c' as u32, TropicalWeight::one(), c_state));
    dict.add_arc(c_state, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), ca_state));
    
    // "cat"
    let cat_final = dict.add_state();
    dict.add_arc(ca_state, Arc::new('t' as u32, 't' as u32, TropicalWeight::one(), cat_final));
    dict.set_final(cat_final, TropicalWeight::one());
    
    // "car"  
    let car_final = dict.add_state();
    dict.add_arc(ca_state, Arc::new('r' as u32, 'r' as u32, TropicalWeight::one(), car_final));
    dict.set_final(car_final, TropicalWeight::one());
    
    // "care" (sharing 'r' transition with "car")
    let care_final = dict.add_state();
    dict.add_arc(car_final, Arc::new('e' as u32, 'e' as u32, TropicalWeight::one(), care_final));
    dict.set_final(care_final, TropicalWeight::one());
    
    // Minimize for efficiency
    minimize(&dict)
}

// Full spell checker would compose with edit distance transducer
// to find corrections within edit distance k
```  

## Further Reading

- [Working with FSTs](working-with-fsts/) — Complete operations reference
- [Core Concepts](core-concepts/) — Theoretical foundations
- [Examples](examples/) — Applied implementations

## API Summary

```rust,ignore
use arcweight::prelude::*;

// FST construction
let mut fst = VectorFst::<W>::new();
let s = fst.add_state();
fst.set_start(s);
fst.set_final(s, weight);
fst.add_arc(from, Arc::new(ilabel, olabel, weight, to));

// Operations
compose_default(&fst1, &fst2)
minimize(&fst)
determinize(&fst)
```