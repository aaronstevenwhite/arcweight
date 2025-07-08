# Installation & Setup

*Setup guide for all platforms*

This guide covers installing ArcWeight and setting up your development environment for working with weighted finite state transducers (FSTs).

**What's Covered**: 

- System requirements
- Dependency management
- Feature configuration
- Development environment setup

**See Also**: [Main Documentation](README.md) for ArcWeight overview and capabilities

## Prerequisites

### Rust Requirements
- **Version**: 1.75.0 or later
- **Toolchain**: Standard Rust installation

### System Requirements
- **CPU**: x86_64 or aarch64
- **OS**: Linux, macOS, Windows
- **Memory**: 4GB+ RAM recommended

### Quick Check
```bash
# Verify your setup
rustc --version
# Should show: rustc 1.75.0+

cargo --version  
# Should show: cargo 1.75.0+
```

**Rust Installation**: Install from [rustup.rs](https://rustup.rs/)

### Verify Rust Installation

Check your Rust version:

```bash
rustc --version
# Should show: rustc 1.75.0 or higher

cargo --version
# Should show: cargo 1.75.0 or higher
```

If you need to install or update Rust:

```bash
# Install Rust (if not installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Update Rust (if already installed)
rustup update
```

## Installing ArcWeight

### Basic Installation

Add ArcWeight to your `Cargo.toml`:

```toml
[dependencies]
arcweight = "0.1.0"
```

Then run:

```bash
cargo build
```

### With All Features

To enable all optional features:

```toml
[dependencies]
arcweight = { version = "0.1.0", features = ["parallel", "serde"] }
```

### Development Dependencies

For development, testing, and benchmarking:

```toml
[dev-dependencies]
arcweight = { version = "0.1.0", features = ["parallel", "serde"] }
criterion = "0.5"
proptest = "1.4"
```

## Feature Flags

ArcWeight provides several optional features:

### Default Features

By default, ArcWeight enables:

- `parallel` - Multi-threaded algorithms via Rayon
- `serde` - Serialization support for FSTs and weights

### Available Features

| Feature | Description | Dependencies | Use Cases |
|---------|-------------|--------------|-----------|
| `parallel` | Enable parallel processing algorithms | `rayon` | Large FSTs, batch processing, multi-core optimization |
| `serde` | Serialization/deserialization support | `serde`, `bincode` | Saving FSTs to disk, network transmission, caching |

**Feature Details:**

- **Parallel Processing**: The `parallel` feature enables multi-threaded implementations of computationally intensive algorithms. This is particularly beneficial for operations on large FSTs or when processing multiple FSTs concurrently. Algorithms like composition, shortest path, and minimization can see significant speedups on multi-core systems.

- **Serialization**: The `serde` feature allows you to save FSTs to various formats (binary, JSON, etc.) and load them later. This is essential for production systems where you want to pre-compute FSTs and reuse them across sessions, or for distributing trained models.

### Custom Feature Selection

To use only specific features:

```toml
[dependencies]
# Only parallel processing, no serialization
arcweight = { version = "0.1.0", default-features = false, features = ["parallel"] }

# Only serialization, no parallel processing
arcweight = { version = "0.1.0", default-features = false, features = ["serde"] }

# Minimal installation (no optional features)
arcweight = { version = "0.1.0", default-features = false }
```

## Verification

### Test Your Installation

Create a simple test file to verify ArcWeight is working:

```rust,ignore
// test_installation.rs
use arcweight::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple FST
    let mut fst = VectorFst::<TropicalWeight>::new();
    
    // Add states
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    
    // Set start and final states
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());
    
    // Add an arc
    fst.add_arc(s0, Arc::new(
        'a' as u32, 
        'a' as u32, 
        TropicalWeight::one(), 
        s1
    ));
    
    println!("ArcWeight is working");
    println!("Created FST with {} states and {} arcs", 
             fst.num_states(), fst.num_arcs());
    
    Ok(())
}
```

Run the test:

```bash
rustc test_installation.rs --extern arcweight
./test_installation
```

Expected output:
```text
ArcWeight is working
Created FST with 2 states and 1 arcs
```

### Run Examples

Test with the provided examples:

```bash
# Clone the repository for examples
git clone https://github.com/aaronstevenwhite/arcweight.git
cd arcweight

# Run a simple example
cargo run --example edit_distance

# Run with features
cargo run --example spell_checking --features "parallel,serde"
```

## IDE Setup

### VS Code

For the best development experience with VS Code:

1. Install the Rust extension:
   ```bash
   code --install-extension rust-lang.rust-analyzer
   ```

2. Add to your `settings.json`:
   ```json
   {
     "rust-analyzer.cargo.features": ["parallel", "serde"],
     "rust-analyzer.checkOnSave.command": "clippy"
   }
   ```

### Other IDEs

ArcWeight works with any IDE that supports Rust:

- **IntelliJ IDEA** — Install the Rust plugin
- **Vim/Neovim** — Use rust-analyzer with your LSP client
- **Emacs** — Use rustic-mode or rust-mode with lsp-mode

## Troubleshooting

### Common Issues

#### 1. Compilation Errors

**Error**: `error: package 'arcweight' cannot be built because it requires rustc 1.75.0 or newer`

**Solution**: Update Rust:
```bash
rustup update
```

#### 2. Feature Resolution Issues

**Error**: `feature 'parallel' not found`

**Solution**: Check your feature specification:
```toml
# Correct
arcweight = { version = "0.1.0", features = ["parallel"] }

# Incorrect
arcweight = { version = "0.1.0", feature = ["parallel"] }  # Note: 'feature' not 'features'
```

#### 3. Dependency Conflicts

**Error**: Version conflicts with other crates

**Solution**: Use `cargo tree` to investigate:
```bash
cargo tree -d  # Show duplicate dependencies
```

#### 4. Memory Issues During Compilation

**Error**: Compilation runs out of memory

**Solutions**:
```bash
# Reduce parallel compilation
export CARGO_BUILD_JOBS=1

# Or use release mode (less memory intensive)
cargo build --release
```

### Getting Help

If you encounter issues:

1. **Check the documentation**: [docs.rs/arcweight](https://docs.rs/arcweight)
2. **Search existing issues**: [GitHub Issues](https://github.com/aaronstevenwhite/arcweight/issues)
3. **Ask for help**: [GitHub Discussions](https://github.com/aaronstevenwhite/arcweight/discussions)
4. **Report bugs**: [New Issue](https://github.com/aaronstevenwhite/arcweight/issues/new)

## Performance Optimization

### Build Settings

For maximum performance in production:

```toml
[profile.release]
lto = true
codegen-units = 1
panic = "abort"
```

### Runtime Optimization

```bash
# Use release builds for performance-critical applications
cargo build --release

# Enable all CPU features
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Next Steps

Once ArcWeight is installed, choose your learning path based on your background and goals:

**Recommended path**: Start with the **[Quick Start Guide](quick-start.md)** to build your first FST, then explore the **[Examples Overview](examples/README.md)** for practical applications. For comprehensive learning paths tailored to different backgrounds and goals, see the [main documentation](README.md).

## Development Setup

For contributors and advanced users:

### Clone and Build

```bash
git clone https://github.com/aaronstevenwhite/arcweight.git
cd arcweight

# Build with all features
cargo build --all-features

# Run tests
cargo test --all-features

# Run benchmarks
cargo bench

# Check code quality
cargo fmt
cargo clippy --all-features
```

### Documentation

Build documentation locally:

```bash
cargo doc --all-features --open
```

This opens comprehensive API documentation in your browser.