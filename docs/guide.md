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

See the `examples/` directory for complete, runnable examples demonstrating various features and use cases.

## API Reference

For detailed API documentation, see the [API Reference](api/README.md) or run `cargo doc --open` to view the generated documentation.
