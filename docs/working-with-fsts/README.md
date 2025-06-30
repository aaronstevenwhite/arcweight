# Working with FSTs

**Master FST operations through practical, hands-on examples**

*Transform • Optimize • Analyze*

This comprehensive guide teaches you how to work with FST operations in ArcWeight. While the [Quick Start](../quick-start.md) gives you a fast introduction to basic FST concepts, this guide provides deep coverage of all operations you'll need for production applications.

## How This Guide Differs from Quick Start

| Quick Start | Working with FSTs |
|-------------|-------------------|
| **15-minute introduction** | **Comprehensive operation reference** |
| First FST, basic concepts | All FST operations in depth |
| Single example workflow | Multiple real-world patterns |
| Get started quickly | Master production techniques |

**When to use this guide:**
- You've completed the Quick Start and want to go deeper
- You need to optimize FST performance
- You're building complex FST pipelines
- You want to understand all available operations

## Guide Structure

### [Operations Overview](operations-overview.md)
**Start here** — Understanding FST operations and when to use each one

- What are FST operations?
- Decision tree for choosing operations
- Operation flow and dependencies

### [Core Operations](core-operations.md)
**Essential building blocks** — Fundamental FST transformations

- **Composition** — Chain FSTs together
- **Union** — Combine alternative paths
- **Concatenation** — Sequential combination

### [Optimization Operations](optimization-operations.md)
**Make FSTs fast and efficient** — Performance and memory optimization

- **Determinization** — Remove non-determinism
- **Minimization** — Reduce to canonical form
- **Connection** — Remove dead states
- **Epsilon Removal** — Eliminate ε-transitions

### [Path Operations](path-operations.md)
**Find solutions in FSTs** — Extract best paths and rankings

- **Shortest Path** — Find optimal solutions
- **N-best Paths** — Multiple solutions
- **Pruning** — Remove unlikely paths

### [Structural Operations](structural-operations.md)
**Analyze and transform** — Extract FST components

- **Projection** — Extract input/output languages
- **Difference** — Set operations on FSTs
- **Intersection** — Common accepted strings

### [Advanced Topics](advanced-topics.md)
**Production-ready techniques** — Performance and best practices

- **Closure Operations** — Kleene star and plus
- **Performance Guidelines** — Operation ordering
- **Memory Management** — Large FST handling

## Learning Path

### For Beginners (After Quick Start)
1. Read [Operations Overview](operations-overview.md)
2. Master [Core Operations](core-operations.md)
3. Try [Path Operations](path-operations.md) for practical results

### For Performance Optimization
1. Start with [Optimization Operations](optimization-operations.md)
2. Study [Advanced Topics](advanced-topics.md) performance section
3. Apply to your specific use case

### For Complex Applications
1. Master all [Core Operations](core-operations.md)
2. Understand [Structural Operations](structural-operations.md)
3. Combine with [Advanced Topics](advanced-topics.md)

## Quick Reference

### Most Common Operations

| Operation | Use When | Example |
|-----------|----------|---------|
| **Composition** | Building pipelines | Tokenizer → POS tagger |
| **Union** | Multiple options | Combining vocabularies |
| **Shortest Path** | Best answer | Spell correction |
| **Determinize** | Speed needed | Real-time systems |
| **Minimize** | Memory constrained | Mobile devices |

### Operation Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Composition | O(\|Q₁\| × \|Q₂\| × \|Σ\|) | O(\|Q₁\| × \|Q₂\|) |
| Union | O(\|Q₁\| + \|Q₂\|) | O(\|Q₁\| + \|Q₂\|) |
| Determinization | O(2^n) worst case | O(2^n) worst case |
| Minimization | O(n log n) | O(n) |
| Shortest Path | O(\|E\| + \|V\| log \|V\|) | O(\|V\|) |

## Prerequisites

Before diving into FST operations, ensure you have:

- ✅ Completed the [Quick Start Guide](../quick-start.md)
- ✅ Basic understanding of FST concepts
- ✅ ArcWeight installed and working
- ✅ Familiarity with Rust basics

> **Need Theory?** See [Core Concepts](../core-concepts/) for mathematical foundations

## Real-World Applications

This guide's operations power many applications:

- **Natural Language Processing** — Text normalization, tokenization
- **Speech Recognition** — Acoustic model composition
- **Machine Translation** — Phrase-based translation
- **Spell Checking** — Error correction models
- **Morphological Analysis** — Word decomposition

## Getting Help

- **Examples**: See [Examples Gallery](../examples/) for complete applications
- **Theory**: Consult [Core Concepts](../core-concepts/) for proofs
- **API**: Check the [API Reference](../api-reference.md) for details

---

**Ready to master FST operations?** → Start with [Operations Overview](operations-overview.md)