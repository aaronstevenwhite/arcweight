# ArcWeight Guide

## Introduction

ArcWeight is a Rust library for working with Finite State Transducers (FSTs) and Weighted Finite State Transducers (WFSTs). This guide will walk you through the core concepts, usage patterns, and practical applications of the library.

## Core Concepts

### Finite State Transducers (FSTs)
- Basic FST operations (composition, determinization, minimization)
- Input/output symbol handling
- State and arc management

### Weighted FSTs (WFSTs)
- Weight semirings and their properties
- Weight operations and propagation
- Common weight types (Tropical, Log, Probability)

### Key Components
- Arc representation and manipulation
- State management
- Symbol table handling
- FST operations and algorithms

## Getting Started

### Installation
```toml
[dependencies]
arcweight = "0.1.0"
```

### Basic Usage
```rust
use arcweight::prelude::*;

// Create a simple FST
let mut fst = Fst::new();
// Add states and arcs
// Perform operations
```

## Common Use Cases

### Text Processing
- Spell checking
- Morphological analysis
- Text normalization

### Speech Recognition
- Pronunciation modeling
- Language model integration
- Decoder graph construction

### Natural Language Processing
- Tokenization
- Stemming/Lemmatization
- Grammar checking

## Advanced Topics

### Performance Optimization
- Memory management
- Operation efficiency
- Large-scale FST handling

### Custom Extensions
- Implementing custom semirings
- Extending FST operations
- Integration with other systems

## Best Practices

### FST Design
- State minimization
- Arc optimization
- Symbol table management

### Error Handling
- Input validation
- Operation safety
- Resource management

## Examples

The `examples/` directory contains complete, runnable examples demonstrating various features and use cases:

### Edit Distance
The [Edit Distance example](examples/edit_distance.md) shows how to use FSTs for spell checking and fuzzy string matching. It demonstrates:
- Building an FST that accepts strings within a specified edit distance
- Creating a dictionary FST from a list of words
- Composing FSTs to find matching words
- Finding the best corrections using shortest paths

### FST Composition
The [FST Composition example](examples/fst_composition.md) demonstrates how to compose FSTs for text transformation. It shows:
- Building FSTs from input strings and transformation rules
- Composing FSTs to apply transformations
- Finding the shortest path in the composed FST
- Extracting and displaying the results

For more examples, see the [Examples Directory](examples/).

## API Reference

For detailed API documentation, see the [API Reference](api/README.md) or run `cargo doc --open` to view the generated documentation.
