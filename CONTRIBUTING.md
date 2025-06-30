# Contributing to ArcWeight - Quick Start

Thank you for your interest in contributing to ArcWeight! This guide provides everything you need to get started with contributing to the project.

## Quick Setup

1. **Fork and clone**:
   ```bash
   git clone https://github.com/yourusername/arcweight.git
   cd arcweight
   ```

2. **Create a branch**:
   ```bash
   git checkout -b feature-name
   ```

3. **Make changes and test**:
   ```bash
   cargo test
   cargo fmt
   cargo clippy
   ```

4. **Submit a pull request**

## Prerequisites

- Rust 1.75+ (MSRV defined in `clippy.toml`)
- Standard Rust toolchain (`cargo`, `rustfmt`, `clippy`)

## Essential Commands

```bash
# Build and test
cargo build
cargo test

# Code quality (uses rustfmt.toml and clippy.toml)
cargo fmt
cargo clippy -- -D warnings

# Run examples
cargo run --example edit_distance

# Performance testing
cargo bench

# Build documentation (requires mdbook and mdbook-katex)
cargo install mdbook mdbook-katex
mdbook build
# View locally at http://localhost:3000
mdbook serve
```

## What to Contribute

- 🐛 **Bug fixes** - Fix issues in algorithms or implementations
- ✨ **New features** - Add semirings, algorithms, or FST types
- 📚 **Examples** - Real-world applications (see `examples/`)
- 🚀 **Performance** - Optimizations and benchmarks
- 📖 **Documentation** - API docs, guides, and tutorials

## Quick Guidelines

- Follow existing code patterns
- Add tests for new functionality
- Update documentation for public APIs
- Run `cargo test && cargo fmt && cargo clippy` before submitting
- Use descriptive commit messages

## Need Help?

- 🐛 [Open an issue](https://github.com/aaronstevenwhite/arcweight/issues)
- 💬 Ask questions in pull request comments
- 📖 [Documentation](docs/README.md) - Browse our comprehensive guides