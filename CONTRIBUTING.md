# Contributing to ArcWeight - Quick Start

Thank you for your interest in contributing to ArcWeight! This is a quick start guide to get you up and running. For comprehensive documentation, see our [Developer Guide](docs/contributing.md).

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

- 📖 [Comprehensive Developer Guide](docs/contributing.md)
- 🐛 [Open an issue](https://github.com/aaronstevenwhite/arcweight/issues)
- 💬 Ask questions in pull request comments

Ready to dive deeper? Check out our [detailed contributor documentation](docs/contributing.md)!