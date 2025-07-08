# Scripts Directory

This directory contains utility scripts for development and CI processes.

## ci-check.sh

A comprehensive script that simulates the exact same checks run by our GitHub CI pipeline. This helps catch issues locally before pushing to GitHub.

### Usage

```bash
# From the project root
./scripts/ci-check.sh
```

### What it checks

1. **Unit tests** with all features enabled
2. **Unit tests** with no default features
3. **Clippy linting** with all features and strict warnings
4. **Clippy linting** with no default features
5. **Code formatting** validation
6. **Documentation** compilation
7. **Benchmark** compilation
8. **Example programs** execution

### Prerequisites

- Rust toolchain with `clippy` and `rustfmt` components
- All project dependencies installed
- Examples should compile without errors

### Exit behavior

The script will exit immediately on the first error encountered, helping you identify and fix issues quickly.

### Recommended workflow

Run this script before committing changes to ensure your code will pass CI:

```bash
# Make changes
git add .
./scripts/ci-check.sh  # Ensure everything passes
git commit -m "Your commit message"
git push
```

This prevents CI failures and saves time in the development cycle.