# Contributing to ArcWeight

Thank you for your interest in contributing to ArcWeight! This document provides guidelines and instructions for contributing to the project.

## Getting Started

### Prerequisites

- Rust 1.85.0 or later (the project's MSRV)
- Git
- Basic familiarity with weighted finite-state transducers (see [documentation](https://docs.rs/arcweight))

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/arcweight.git
   cd arcweight
   ```
3. Build the project:
   ```bash
   cargo build
   ```
4. Run the test suite:
   ```bash
   cargo test --all-features
   ```

## Development Workflow

### Creating a Branch

Create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### Making Changes

1. Make your changes following the code style guidelines below
2. Add tests for new functionality
3. Update documentation for any public API changes
4. Run the test suite: `cargo test --all-features`
5. Run the formatter: `cargo fmt --all`
6. Run the linter: `cargo clippy --all-features --all-targets -- -D warnings`

### Code Style

ArcWeight follows standard Rust conventions:

- Use `rustfmt` for formatting (configuration in `rustfmt.toml`)
- Follow Clippy suggestions (configuration in `clippy.toml`)
- Write idiomatic Rust code using iterators, `Result` types, and zero-cost abstractions
- Prefer trait-based abstractions for extensibility
- Use descriptive variable names and add comments for complex algorithms

### Documentation Standards

All public APIs must be documented:

- **Module-level documentation**: Overview, main concepts, and examples
- **Function-level documentation**: Purpose, parameters, return values, examples, and edge cases
- **Examples**: All public functions should have runnable doc examples
- **Complexity analysis**: Include time and space complexity for algorithms
- **Mathematical notation**: Use Unicode symbols (⊕, ⊗, etc.) for semiring operations

Example:

```rust
/// Computes the shortest distance from the start state to all states.
///
/// Uses a generic single-source shortest-distance algorithm that works
/// for any semiring supporting the required operations.
///
/// # Complexity
///
/// - Time: O(|V| + |E|) for acyclic FSTs
/// - Space: O(|V|) for distance storage
///
/// where |V| is the number of states and |E| is the number of arcs.
///
/// # Examples
///
/// ```
/// use arcweight::prelude::*;
///
/// let mut fst = VectorFst::<TropicalWeight>::new();
/// // ... build FST ...
/// let distances = shortest_distance(&fst)?;
/// # Ok::<(), arcweight::Error>(())
/// ```
pub fn shortest_distance<W, F>(fst: &F) -> Result<Vec<W>>
where
    W: Semiring,
    F: Fst<W>,
{
    // implementation
}
```

### Testing

ArcWeight uses multiple testing strategies:

1. **Unit tests**: Test individual functions in isolation
2. **Integration tests**: Test end-to-end workflows in `tests/`
3. **Doc tests**: Examples in documentation are automatically tested
4. **Property-based tests**: Use `proptest` for algorithmic properties

All new functionality must include tests:

```bash
# Run all tests
cargo test --all-features

# Run specific test
cargo test test_name

# Run with output
cargo test -- --nocapture

# Run benchmarks (compile only)
cargo bench --no-run
```

### Benchmarking

When adding new algorithms or optimizations:

1. Add benchmarks in `benches/` directory
2. Use `criterion` for statistical analysis
3. Document performance characteristics
4. Run benchmarks: `cargo bench`

### Submitting a Pull Request

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Open a pull request on GitHub
3. Fill out the pull request template
4. Wait for CI checks to pass
5. Address any review feedback

### Pull Request Guidelines

- Keep pull requests focused on a single feature or fix
- Write clear, descriptive commit messages
- Include tests and documentation
- Ensure all CI checks pass
- Respond to review feedback promptly

## Types of Contributions

### Bug Fixes

Bug fixes are always welcome! When submitting a bug fix:

- Include a test that reproduces the bug
- Reference any related issues
- Explain the root cause and your fix

### New Features

Before implementing a significant new feature:

1. Open an issue to discuss the feature
2. Ensure it aligns with the project's goals
3. Discuss the API design
4. Implement with tests and documentation

Good candidates for new features:
- Additional FST algorithms from the literature
- New semiring types
- Performance optimizations
- Additional FST storage formats

### Documentation Improvements

Documentation improvements are highly valued:

- Fix typos and unclear explanations
- Add examples for complex functionality
- Improve API documentation
- Add usage guides

### Performance Improvements

When submitting performance improvements:

- Include before/after benchmark results
- Explain the optimization technique
- Ensure correctness is maintained
- Consider trade-offs (speed vs. memory)

## Code Review Process

All contributions go through code review:

1. Automated checks (CI) must pass
2. At least one maintainer will review the code
3. Address feedback and iterate
4. Once approved, a maintainer will merge

## Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Focus on constructive feedback
- Follow the Rust Code of Conduct

## Getting Help

- **Questions**: Open a discussion on GitHub
- **Bugs**: Open an issue with reproduction steps
- **Features**: Open an issue to discuss before implementing
- **Documentation**: [docs.rs/arcweight](https://docs.rs/arcweight)

## Project Structure

```
arcweight/
├── src/
│   ├── algorithms/     # FST algorithms (compose, minimize, etc.)
│   ├── fst/           # FST trait definitions and implementations
│   ├── semiring/      # Semiring implementations
│   └── lib.rs         # Library root
├── examples/          # Real-world examples
├── tests/             # Integration tests
└── benches/           # Performance benchmarks
```

## License

By contributing to ArcWeight, you agree that your contributions will be licensed under the Apache License, Version 2.0.

## Recognition

Contributors are recognized in:
- Git commit history
- GitHub contributors list
- Release notes for significant contributions

Thank you for contributing to ArcWeight!
