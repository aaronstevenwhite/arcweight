# ArcWeight Developer Guide

This is the comprehensive developer guide for ArcWeight contributors. For a quick start, see the [Contributing Quick Start](../CONTRIBUTING.md).

This guide covers in-depth development practices, architecture details, and advanced contribution workflows.

## Development Setup

### Prerequisites
- Rust 1.75 or later (MSRV defined in `clippy.toml`)
- Cargo
- Git
- rustfmt and clippy (usually included with Rust)

### Development Environment Setup

1. **Fork and clone** (see [Quick Start](../CONTRIBUTING.md) for basics)

2. **Set up upstream remote**:
   ```bash
   git remote add upstream https://github.com/aaronstevenwhite/arcweight.git
   ```

3. **Install development tools**:
   ```bash
   # Install additional tools for development
   cargo install cargo-watch    # Watch for changes
   cargo install cargo-expand   # Expand macros
   cargo install cargo-udeps    # Find unused dependencies
   ```

4. **Verify setup**:
   ```bash
   cargo test --all-features
   cargo clippy -- -D warnings
   cargo bench --dry-run
   ```

### Building
```bash
cargo build
```

### Testing
```bash
cargo test
```

## Development Workflow

### Branching Strategy
- **Feature branches**: `feature/description` (e.g., `feature/add-string-semiring`)
- **Bug fixes**: `fix/description` (e.g., `fix/memory-leak-in-determinize`)
- **Documentation**: `docs/description` (e.g., `docs/improve-algorithm-examples`)
- **Performance**: `perf/description` (e.g., `perf/optimize-composition`)

### Development Cycle
1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature
   ```

3. **Iterative development**:
   ```bash
   # Watch for changes during development
   cargo watch -x 'test --lib'
   
   # Run focused tests
   cargo test specific_test_name
   
   # Check performance impact
   cargo bench --baseline baseline_name
   ```

4. **Pre-commit validation**:
   ```bash
   cargo test --all-features
   cargo clippy -- -D warnings
   cargo fmt --check
   cargo doc --no-deps
   ```

### Code Style

ArcWeight uses comprehensive code quality tooling:

#### Configuration
- **`rustfmt.toml`** - Code formatting standards:
  - 100 character line width
  - 4-space indentation
  - Unix line endings
  - Edition 2021 features
- **`clippy.toml`** - Linting configuration:
  - Performance-optimized thresholds
  - Documentation requirements
  - FST-specific allowances
  - Cognitive complexity limits

#### Quality Checks
```bash
# Format code
cargo fmt

# Run linter
cargo clippy -- -D warnings

# Check documentation
cargo doc --no-deps --open
```

### Testing

ArcWeight has a multi-layered testing strategy:

#### Test Types
- **Unit tests** - Individual components (`src/` modules)
- **Integration tests** - Algorithm combinations (`tests/` directory)
- **Property tests** - Mathematical properties using `proptest`
- **Benchmarks** - Performance validation (`benches/` directory)

#### Test Organization
```
tests/
├── algorithms_tests.rs    # Algorithm correctness
├── arc_tests.rs          # Arc implementation
├── fst_tests.rs          # FST implementations
├── integration_tests.rs   # End-to-end scenarios
├── properties.rs         # FST properties
├── proptests.rs          # Property-based tests
└── utils_tests.rs        # Utility functions
```

#### Running Tests
```bash
# All tests
cargo test

# Specific test suites
cargo test --integration
cargo test property_tests
cargo test --test algorithms_tests

# With all features
cargo test --all-features
```

### Documentation
- Add documentation comments for public APIs
- Update relevant documentation files
- Include examples in documentation

## Pull Request Process

1. Update documentation
2. Add tests for new functionality
3. Ensure all tests pass
4. Submit PR with clear description
5. Address review comments

## Code Review Guidelines

### Review Checklist

#### Correctness
- [ ] Algorithm implementation matches mathematical definitions
- [ ] Edge cases are handled (empty FSTs, epsilon transitions, etc.)
- [ ] Error conditions are properly propagated
- [ ] Semiring properties are maintained

#### Performance
- [ ] No unnecessary allocations in hot paths
- [ ] Iterator usage is efficient (avoid collecting when possible)
- [ ] Cache-friendly access patterns
- [ ] Benchmark results show no regression

#### Memory Safety
- [ ] No unsafe code without thorough justification
- [ ] Resource cleanup in error paths
- [ ] No memory leaks in long-running operations

#### Documentation
- [ ] Public APIs have comprehensive documentation
- [ ] Examples are provided for complex functionality
- [ ] Mathematical notation is clear and consistent
- [ ] Performance characteristics are documented

#### Testing
- [ ] Unit tests for individual components
- [ ] Integration tests for algorithm combinations
- [ ] Property tests for mathematical properties
- [ ] Edge case coverage

### Review Process
1. **Automated checks**: Ensure CI passes (tests, clippy, formatting)
2. **Code inspection**: Manual review of logic and patterns
3. **Performance analysis**: Check benchmark results
4. **Documentation review**: Verify clarity and completeness
5. **Testing validation**: Confirm adequate test coverage

## Architecture Deep Dive

### Core Design Principles

#### Trait-Based Architecture
ArcWeight uses traits to define contracts:
- `Semiring` - Weight operations with mathematical guarantees
- `Fst` - Read-only FST operations
- `MutableFst` - Mutable FST operations
- `ExpandedFst` - Random access to states and arcs

#### Generic Programming
```rust
// Algorithms work with any semiring
fn compose<W: Semiring>(
    fst1: &impl Fst<W>,
    fst2: &impl Fst<W>
) -> Result<VectorFst<W>>
```

#### Zero-Copy Iteration
Arc iterators avoid allocations:
```rust
// Iterator returns references, not owned values
for arc in fst.arcs(state_id)? {
    // Process arc without allocation
}
```

### Performance Optimization Patterns

#### Hot Path Optimization
- Minimize allocations in `compose()`, `determinize()`, `shortest_path()`
- Use `SmallVec` for small collections
- Cache-friendly data layout

#### Memory Management
- State pooling in algorithms
- Lazy computation where possible
- Efficient representation choices (CompactFst vs VectorFst)

### Testing Strategy

#### Property-Based Testing
Use `proptest` to verify mathematical properties:
```rust
proptest! {
    #[test]
    fn compose_associative(
        a in any_fst(),
        b in any_fst(),
        c in any_fst()
    ) {
        // (a ∘ b) ∘ c ≡ a ∘ (b ∘ c)
    }
}
```

#### Benchmark-Driven Development
Performance tests guide optimization:
```bash
# Baseline before changes
cargo bench --save-baseline before

# After optimization
cargo bench --baseline before
```

## Release Process

### Version Management
- **Semantic versioning** (MAJOR.MINOR.PATCH)
- **MSRV policy**: Update minimum supported Rust version conservatively
- **Feature flags**: Use for optional dependencies

### Release Checklist
1. [ ] Update version in `Cargo.toml`
2. [ ] Update `CHANGELOG.md` with changes
3. [ ] Verify all tests pass on CI
4. [ ] Run full benchmark suite
5. [ ] Update documentation
6. [ ] Create and push git tag
7. [ ] Publish to crates.io
8. [ ] Update GitHub release notes

## Project Structure

### Key Directories
- `src/` - Source code organized by functionality
  - `algorithms/` - FST algorithms (composition, determinization, etc.)
  - `fst/` - FST implementations (Vector, Const, Compact, etc.)
  - `semiring/` - Weight types (Tropical, Probability, Boolean, etc.)
  - `arc/` - Arc representation and iteration
  - `io/` - File I/O including OpenFST compatibility
  - `utils/` - Supporting utilities
- `tests/` - Comprehensive test suite
- `examples/` - Real-world applications (7 complete examples)
- `docs/` - Documentation and guides
- `benches/` - Performance benchmarks organized by category
  - `core/` - Basic operations
  - `memory/` - Memory usage
  - `io/` - Serialization performance
  - `optimization/` - Advanced algorithms
  - `parallel/` - Parallel processing

### Important Files
- `Cargo.toml` - Project configuration and dependencies
- `README.md` - Project overview and quick start
- `CONTRIBUTING.md` - This contributing guide
- `LICENSE` - Apache 2.0 license
- `clippy.toml` - Linting configuration
- `rustfmt.toml` - Code formatting configuration
- `CLAUDE.md` - Development guidance for AI assistants

## Advanced Topics

### Adding New Semirings

1. **Implement the trait**:
   ```rust
   impl Semiring for CustomWeight {
       fn plus(&self, other: &Self) -> Self { /* ... */ }
       fn times(&self, other: &Self) -> Self { /* ... */ }
       fn zero() -> Self { /* ... */ }
       fn one() -> Self { /* ... */ }
   }
   ```

2. **Add comprehensive tests**:
   - Semiring axioms (associativity, commutativity, distributivity)
   - Identity elements
   - Edge cases (overflow, underflow)

3. **Performance considerations**:
   - Benchmark against existing semirings
   - Consider SIMD optimization opportunities
   - Evaluate memory footprint

### Algorithm Implementation Guidelines

#### State Management
- Use state pools to avoid allocations
- Maintain state ID consistency
- Handle epsilon transitions carefully

#### Error Handling
- Use `Result<T>` for fallible operations
- Provide meaningful error messages
- Ensure resource cleanup on errors

#### Optimization Strategies
- Profile before optimizing
- Use `cargo bench` to measure improvements
- Consider parallel algorithms for large FSTs

### Parallel Algorithm Development

When adding parallel algorithms:
1. **Feature gating**: Use `#[cfg(feature = "parallel")]`
2. **Chunking strategy**: Divide work appropriately
3. **Synchronization**: Minimize shared mutable state
4. **Benchmarking**: Compare against sequential versions

## Communication & Support

### Getting Help
- **Quick questions**: [GitHub Discussions](https://github.com/aaronstevenwhite/arcweight/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/aaronstevenwhite/arcweight/issues)
- **Feature requests**: [GitHub Issues](https://github.com/aaronstevenwhite/arcweight/issues) with enhancement label
- **Development chat**: Pull request comments for specific code discussions

### Issue Templates
When reporting issues:
- **Bug report**: Include minimal reproduction case
- **Performance issue**: Include benchmark results
- **Feature request**: Describe use case and API preferences
- **Documentation**: Specify what's unclear or missing

## Code of Conduct

### Guidelines
- Be respectful
- Be constructive
- Be inclusive
- Be patient

### Enforcement
- Report violations
- Maintain professionalism
- Follow project guidelines

---

## Quick Links

- [Contributing Quick Start](../CONTRIBUTING.md) - Get started quickly
- [Project README](../README.md) - Project overview
- [User Guide](guide.md) - Using ArcWeight
- [Examples](examples/) - Real-world applications
- [Architecture Documentation](internals.md) - Deep dive into internals

## License

By contributing to ArcWeight, you agree that your contributions will be licensed under the Apache 2.0 license. 