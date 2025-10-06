# Core Concepts

This section provides a formal treatment of Weighted Finite State Transducers (WFSTs) and semirings as implemented in ArcWeight. Following the established theoretical foundations of finite-state methods {{#cite mohri1997finite}} {{#cite kuich1986semirings}}, we present formal definitions, implementation details, and practical considerations for computational linguists and engineers.

## Table of Contents

- **[FSTs](fsts.md)** — Comprehensive theory of finite state transducers, rational relations, and theoretical foundations
- **[Semirings](semirings.md)** — Mathematical foundations of weight computation, semiring algebra, and implementation details
- **[Algorithms](algorithms.md)** — Core FST algorithms, rational operations, optimization procedures, and complexity analysis

## Overview

This documentation covers theoretical foundations and practical implementation aspects of weighted finite state transducers:

### Theoretical Foundations
- **Formal Definitions**: Mathematical definitions of FSTs, semirings, and rational relations
- **Closure Properties**: Operations that preserve rationality
- **Complexity Analysis**: Time and space complexity of core algorithms
- **Correctness**: Mathematical guarantees for algorithmic implementations

### Practical Implementation
- **Semiring Selection**: Appropriate algebraic structures for specific problems
- **Optimization Strategies**: Determinization, minimization, and performance considerations
- **Memory Management**: Efficient representations and lazy evaluation
- **Symbol Management**: Practices for large-scale symbol handling

### Applications
- **Natural Language Processing**: Machine translation {{#cite kumar2003weighted}}, speech recognition {{#cite mohri2002weighted}}, morphological analysis {{#cite beesley2003finite}}
- **Computational Biology**: Sequence alignment, motif discovery, phylogenetic analysis
- **Signal Processing**: Temporal pattern recognition, time-series analysis
- **Information Extraction**: Named entity recognition, text normalization {{#cite karttunen2001applications}}

## See Also

- **[API Reference](../api-reference.md)** — Function and type documentation
- **[Working with FSTs](../working-with-fsts/)** — Core FST manipulation algorithms
- **[Examples](../examples/README.md)** — WFST applications

## Theoretical Foundations

The mathematical framework of weighted finite-state transducers builds on established theoretical contributions:

- **Rational Relations**: {{#cite elgot1965relations}} established mathematical foundations of relations defined by finite automata
- **Semiring Theory**: {{#cite kuich1986semirings}} provided algebraic framework for weighted computation
- **WFST Algorithms**: {{#cite mohri2002semiring}} developed algorithmic toolkit for shortest-distance problems
- **Practical Implementation**: {{#cite allauzen2007openfst}} demonstrated efficient implementation strategies
- **Morphological Applications**: {{#cite beesley2003finite}} demonstrated finite-state methods in computational linguistics