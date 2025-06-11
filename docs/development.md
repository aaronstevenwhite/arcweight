# Development Guide

This guide provides detailed information for developers working on ArcWeight.

## Development Environment

### Required Tools
- Rust 1.75.0 or later (MSRV - Minimum Supported Rust Version)
- Cargo
- Git
- A code editor (VS Code, IntelliJ, etc.)
- Rust analyzer (for IDE support)

### Optional Tools
- cargo-watch (for development)
- cargo-expand (for macro debugging)
- cargo-udeps (for unused dependencies)
- cargo-audit (for security checks)

## Project Setup

### Initial Setup
```bash
# Clone the repository
git clone https://github.com/aaronstevenwhite/arcweight.git
cd arcweight

# Install development dependencies
cargo install cargo-watch
cargo install cargo-expand
cargo install cargo-udeps
cargo install cargo-audit
```

### Development Dependencies
Add to your `~/.cargo/config.toml`:
```toml
[alias]
dev = "run --package arcweight --bin arcweight"
test = "test --package arcweight --lib"
watch = "watch -x 'run --package arcweight --bin arcweight'"
```

## Building

### Basic Build
```bash
cargo build
```

### Release Build
```bash
cargo build --release
```

### Specific Features
```bash
cargo build --features "feature1 feature2"
```

## Minimum Supported Rust Version (MSRV)

ArcWeight has a Minimum Supported Rust Version (MSRV) of **1.75.0**. This version is:
- Specified in `Cargo.toml` as `rust-version = "1.75.0"`
- Configured in `clippy.toml` as `msrv = "1.75.0"`
- Tested in CI with the specific version

### Verifying MSRV

The MSRV was determined using `cargo msrv`:

```bash
# Install cargo-msrv
cargo install cargo-msrv

# Find the minimum supported Rust version
cargo msrv find

# Verify a specific version works
cargo msrv verify 1.75.0
```

### Updating MSRV

When updating the MSRV:
1. Run `cargo msrv find` to determine the new minimum version
2. Update `rust-version` in `Cargo.toml`
3. Update `msrv` in `clippy.toml`
4. Update the CI workflow (`.github/workflows/ci.yml`)
5. Update documentation references
6. Add a note in the changelog

## Testing

### Running Tests
```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run tests with output
cargo test -- --nocapture

# Run tests in parallel
cargo test -- --test-threads=1
```

### Test Coverage
```bash
# Install cargo-tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin
```

## Development Workflow

### Code Style
```bash
# Format code
cargo fmt

# Check style
cargo clippy
```

### Continuous Integration
- GitHub Actions workflow
- Automated testing
- Code coverage reporting
- Documentation generation

### Debugging
```bash
# Run with debug logging
RUST_LOG=debug cargo run

# Run with specific module logging
RUST_LOG=arcweight::fst=debug cargo run
```

## Performance

### Benchmarking
```bash
# Run benchmarks
cargo bench

# Run specific benchmark
cargo bench benchmark_name
```

### Profiling
```bash
# Install cargo-flamegraph
cargo install flamegraph

# Generate flamegraph
cargo flamegraph
```

## Documentation

### Generating Documentation
```bash
# Generate docs
cargo doc

# Generate docs with private items
cargo doc --document-private-items

# Serve docs locally
cargo doc --open
```

### Documentation Style
- Use rustdoc comments
- Include examples
- Document error cases
- Link to related items

## Release Process

### Version Management
```bash
# Update version
cargo set-version 0.1.0

# Check outdated dependencies
cargo outdated
```

### Release Checklist
1. Update version numbers
2. Update CHANGELOG.md
3. Run all tests
4. Generate documentation
5. Create release tag
6. Publish to crates.io

## Project Structure

### Source Organization
```
src/
├── lib.rs          # Library entry point
├── prelude.rs      # Common imports
├── fst/            # FST implementation
├── semiring/       # Semiring implementations
├── arc/            # Arc implementation
├── algorithms/     # FST algorithms
├── properties/     # FST properties
└── io/            # Input/output operations
```

### Test Organization
```
tests/
├── integration/    # Integration tests
├── property/       # Property-based tests
└── benchmarks/     # Benchmark tests
```

## Common Tasks

### Adding Dependencies
1. Add to Cargo.toml
2. Run `cargo update`
3. Update documentation if needed

### Creating New Modules
1. Create module file
2. Add to lib.rs
3. Add tests
4. Update documentation

### Debugging Issues
1. Enable debug logging
2. Use debugger
3. Add test cases
4. Document findings

## Best Practices

### Code Quality
- Write tests first
- Document public APIs
- Handle errors properly
- Use appropriate abstractions

### Performance
- Profile before optimizing
- Use appropriate data structures
- Minimize allocations
- Consider parallelization

### Security
- Audit dependencies
- Handle sensitive data
- Validate inputs
- Follow security guidelines

## Troubleshooting

### Common Issues
- Build failures
- Test failures
- Performance issues
- Memory leaks

### Solutions
- Check dependencies
- Update Rust toolchain
- Clear cargo cache
- Check system resources 