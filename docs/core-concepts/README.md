# Core Concepts

This section provides a mathematically rigorous introduction to Weighted Finite State Transducers (WFSTs) and semirings as implemented in ArcWeight. We assume familiarity with basic automata theory and focus on formal definitions, implementation details, and practical applications. All content has been carefully organized to eliminate redundancy while preserving comprehensive coverage.

## Table of Contents

- **[FSTs](fsts.md)** — Comprehensive theory of finite state transducers, rational relations, and theoretical foundations
- **[Semirings](semirings.md)** — Mathematical foundations of weight computation, semiring algebra, and implementation details
- **[Algorithms](algorithms.md)** — Core FST algorithms, rational operations, optimization procedures, and complexity analysis

## Overview

This documentation provides comprehensive coverage of the theoretical and practical aspects of weighted finite state transducers:

### Theoretical Foundations
- **Formal Definitions**: Precise mathematical definitions of FSTs, semirings, and rational relations
- **Closure Properties**: Understanding which operations preserve rationality
- **Complexity Analysis**: Time and space complexity for all core algorithms
- **Correctness Proofs**: Mathematical guarantees for algorithmic implementations

### Practical Implementation
- **Semiring Selection**: Choosing appropriate algebraic structures for specific problems
- **Optimization Strategies**: Determinization, minimization, and performance tuning
- **Memory Management**: Efficient representations and lazy evaluation techniques
- **Symbol Management**: Best practices for large-scale symbol handling

### Real-World Applications
- **Natural Language Processing**: Machine translation, speech recognition, morphological analysis
- **Computational Biology**: Sequence alignment, motif discovery, phylogenetic analysis
- **Signal Processing**: Temporal pattern recognition, time-series analysis
- **Information Extraction**: Named entity recognition, text normalization

## See Also

- **[API Reference](../api-reference.md)** - Complete function and type documentation
- **[Working with FSTs](../working-with-fsts/)** - Core FST manipulation algorithms
- **[Examples](../examples/README.md)** - Real-world WFST applications

## References

- Elgot, C.C. & Mezei, J.E. (1965). On relations defined by generalized finite automata. *IBM Journal of Research and Development*, 9:47-65
- Kuich, W. & Salomaa, A. (1986). *Semirings, Automata, Languages*. Springer-Verlag
- Mohri, M. (2002). Semiring frameworks and algorithms for shortest-distance problems. *Journal of Automata, Languages and Combinatorics*, 7(3):321-350
- Allauzen, C., Riley, M., Schalkwyk, J., Skut, W., & Mohri, M. (2007). OpenFst: A general and efficient weighted finite-state transducer library. *CIAA 2007*, LNCS vol 4783, Springer
- Beesley, K.R. & Karttunen, L. (2003). *Finite State Morphology*. CSLI Publications