# Contributing to ArcWeight

Thank you for your interest in contributing to ArcWeight! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/arcweight.git`
3. Create a feature branch: `git checkout -b feature-name`
4. Make your changes
5. Run tests: `cargo test`
6. Run benchmarks: `cargo bench`
7. Submit a pull request

## Development Setup

### Prerequisites

- Rust 1.75 or later
- Cargo
- rustfmt
- clippy

### Building

```bash
cargo build --all-features
```

### Testing

```bash
# run all tests
cargo test

# run with all features
cargo test --all-features

# run specific test
cargo test test_name

# run property tests
cargo test --test property_tests
```

### Code Quality

Before submitting, ensure:

```bash
# format code
cargo fmt

# run linter
cargo clippy -- -D warnings

# check documentation
cargo doc --no-deps --open
```

## Code Style

- Follow Rust naming conventions
- Use rustfmt for formatting
- Keep functions focused and small
- Document all public APIs
- Add examples for complex functionality
- Use descriptive variable names

## Adding Features

### New Semiring

1. Add implementation in `src/semiring/`
2. Implement required traits
3. Add tests
4. Update documentation

### New Algorithm

1. Add implementation in `src/algorithms/`
2. Follow existing patterns
3. Add comprehensive tests
4. Add benchmarks if applicable
5. Update examples

### New FST Type

1. Add implementation in `src/fst/`
2. Implement required traits
3. Ensure compatibility with algorithms
4. Add tests and documentation

## Testing Guidelines

- Unit test individual components
- Integration test algorithm combinations
- Property test mathematical properties
- Benchmark performance-critical code
- Test edge cases and error conditions

## Documentation

- Document all public items
- Include examples in doc comments
- Update README for significant changes
- Keep CHANGELOG up to date

## Performance

- Benchmark before and after changes
- Avoid unnecessary allocations
- Use iterators over collecting
- Consider cache efficiency
- Profile hot paths

## Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `perf:` Performance improvement
- `docs:` Documentation change
- `test:` Test addition/modification
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

## Pull Request Process

1. Update documentation
2. Add/update tests
3. Ensure CI passes
4. Request review
5. Address feedback
6. Squash commits if requested

## Questions?

Feel free to:
- Open an issue for questions
- Join discussions
- Ask in pull request comments

Thank you for contributing!