# ArcWeight Architecture

**Internal Design and Implementation Guide**

This section provides a comprehensive overview of ArcWeight's internal architecture, design decisions, and implementation patterns. It's intended for:

- **Contributors** who want to understand the codebase structure
- **Advanced users** implementing custom FST types or algorithms  
- **Library developers** interested in FST implementation techniques
- **Researchers** studying practical automata theory implementations

> **Note**: This section focuses on internal architecture. For user-facing documentation, see:
> - [Quick Start](../quick-start.md) for getting started
> - [Core Concepts](../core-concepts/) for theoretical foundations  
> - [Working with FSTs](../working-with-fsts/) for practical usage patterns

## Architecture Overview

- **[Design Philosophy](design-philosophy.md)** - Core principles and design decisions
- **[System Architecture](system-architecture.md)** - Layered architecture and component relationships
- **[Mathematical Foundations](mathematical-foundations.md)** - How theory maps to implementation
- **[Trait System](trait-system.md)** - FST and semiring trait hierarchies
- **[FST Implementations](fst-implementations.md)** - Storage strategies and trade-offs
- **[Design Patterns](design-patterns.md)** - Common patterns and best practices
- **[Algorithm Architecture](algorithm-architecture.md)** - Algorithm organization and patterns
- **[Memory Management](memory-management.md)** - Ownership patterns and optimization
- **[Extension Points](extension-points.md)** - How to extend and customize ArcWeight

## Quick Navigation

### For Contributors
1. Start with [Design Philosophy](design-philosophy.md) to understand the principles
2. Review [System Architecture](system-architecture.md) for the big picture  
3. Dive into [Trait System](trait-system.md) for implementation details

### For Advanced Users
1. [FST Implementations](fst-implementations.md) - Choose the right FST type
2. [Algorithm Architecture](algorithm-architecture.md) - Algorithm design patterns
3. [Extension Points](extension-points.md) - Custom implementations

### For Library Developers
1. [Mathematical Foundations](mathematical-foundations.md) - Theory-to-code mapping
2. [Design Patterns](design-patterns.md) - Proven implementation patterns
3. [Algorithm Architecture](algorithm-architecture.md) - How algorithms are structured

---

**For Contributors**: This documentation serves as a guide for understanding the codebase structure and making architectural decisions consistent with the library's design principles.

**For Advanced Users**: Use this documentation to understand implementation trade-offs when choosing between different FST types or when implementing custom extensions.

**For Researchers**: This documentation demonstrates how automata theory concepts can be efficiently implemented in modern systems programming languages while maintaining mathematical rigor.