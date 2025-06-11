# Contributing to ArcWeight

Thank you for your interest in contributing to ArcWeight! This document provides guidelines and instructions for contributing to the project.

## Development Setup

### Prerequisites
- Rust (latest stable version)
- Cargo
- Git

### Getting Started
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/aaronstevenwhite/arcweight.git
   cd arcweight
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/aaronstevenwhite/arcweight.git
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

### Branching
- Create a new branch for each feature or bugfix
- Use descriptive branch names (e.g., `feature/add-fst-composition`, `fix/memory-leak`)
- Keep branches up to date with main

### Code Style
- Follow Rust style guidelines
- Run `cargo fmt` before committing
- Run `cargo clippy` to check for common issues

### Testing
- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Add integration tests for complex features

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

### What to Review
- Code correctness
- Performance implications
- Memory usage
- Error handling
- Documentation
- Test coverage

### Review Process
1. Check code style and formatting
2. Verify tests and documentation
3. Review for potential issues
4. Provide constructive feedback

## Release Process

### Versioning
- Follow semantic versioning
- Update version in Cargo.toml
- Update CHANGELOG.md

### Release Steps
1. Update version numbers
2. Update documentation
3. Create release tag
4. Publish to crates.io

## Project Structure

### Key Directories
- `src/` - Source code
- `tests/` - Test files
- `examples/` - Example code
- `docs/` - Documentation
- `benches/` - Benchmarks

### Important Files
- `Cargo.toml` - Project configuration
- `README.md` - Project overview
- `CHANGELOG.md` - Version history
- `LICENSE` - License information

## Communication

### Issue Tracking
- Use GitHub Issues
- Provide clear descriptions
- Include reproduction steps
- Label issues appropriately

### Discussion
- Use GitHub Discussions
- Join community chat
- Participate in code reviews

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

## License

By contributing to ArcWeight, you agree that your contributions will be licensed under the project's license. 