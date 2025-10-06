# Installation

This guide covers the installation and configuration of the ArcWeight library.

Prerequisites: [Core Concepts](core-concepts/) | [Quick Start](quick-start.md)

## System Requirements

- Rust 1.85.0 or later
- Standard Rust toolchain (cargo, rustc)
- Supported platforms: Linux, macOS, Windows (x86_64, aarch64)

### Rust Installation

Verify existing installation:
```bash
rustc --version
cargo --version
```

Install or update Rust:
```bash
# Install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Update existing installation
rustup update
```

## Library Installation

Add to `Cargo.toml`:

```toml
[dependencies]
arcweight = "0.1.0"
```

### Feature Configuration

Default installation includes all features:

```toml
[dependencies]
arcweight = { version = "0.1.0", features = ["parallel", "serde"] }
```

### Minimal Installation

```toml
[dependencies]
arcweight = { version = "0.1.0", default-features = false }
```

## Optional Features

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `parallel` | Parallel algorithms via Rayon | `rayon` |
| `serde` | Serialization support | `serde`, `bincode` |

Both features are enabled by default.

### Feature Selection

```toml
# Parallel processing only
arcweight = { version = "0.1.0", default-features = false, features = ["parallel"] }

# Serialization only
arcweight = { version = "0.1.0", default-features = false, features = ["serde"] }

# No optional features
arcweight = { version = "0.1.0", default-features = false }
```

## Verification

Test installation with a minimal example:

```rust,ignore
use arcweight::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut fst = VectorFst::<TropicalWeight>::new();
    
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    
    fst.set_start(s0);
    fst.set_final(s1, TropicalWeight::one());
    
    fst.add_arc(s0, Arc::new('a' as u32, 'a' as u32, TropicalWeight::one(), s1));
    
    println!("States: {}", fst.num_states());
    Ok(())
}
```

### Running Examples

```bash
# Clone repository
git clone https://github.com/aaronstevenwhite/arcweight.git
cd arcweight

# Run example
cargo run --example edit_distance
```

## Development Environment

### VS Code Configuration

```json
{
  "rust-analyzer.cargo.features": ["parallel", "serde"],
  "rust-analyzer.checkOnSave.command": "clippy"
}
```

### IDE Support

- IntelliJ IDEA with Rust plugin
- Vim/Neovim with rust-analyzer
- Emacs with rustic-mode

## Common Issues

### Rust Version Error

Error: package 'arcweight' cannot be built because it requires rustc 1.85.0 or newer

Solution: `rustup update`

### Feature Resolution

Ensure correct syntax:
```toml
# Correct
arcweight = { version = "0.1.0", features = ["parallel"] }
```

### Memory Constraints

For limited memory systems:
```bash
export CARGO_BUILD_JOBS=1
cargo build --release
```

## Build Optimization

### Release Profile

```toml
[profile.release]
lto = true
codegen-units = 1
panic = "abort"
```

### CPU-Specific Optimization

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Development Setup

```bash
git clone https://github.com/aaronstevenwhite/arcweight.git
cd arcweight

# Build and test
cargo build --all-features
cargo test --all-features
cargo bench

# Generate documentation
cargo doc --all-features --open
```

## Further Reading

- [Quick Start](quick-start.md) — Basic usage
- [Core Concepts](core-concepts/) — Theoretical foundations
- [Examples](examples/) — Applied implementations