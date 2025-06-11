# ArcWeight Compatibility

This document outlines platform support, MSRV policy, and feature compatibility for ArcWeight.

## Platform Support

### Operating Systems
- **Linux** (x86_64, aarch64)
- **macOS** (x86_64, aarch64) 
- **Windows** (x86_64)

### Minimum Requirements
- **Rust**: 1.75.0 or later (MSRV)
- **Memory**: 2GB RAM minimum, 4GB recommended
- **Storage**: 200MB for library and dependencies

## MSRV Policy

### Current MSRV
- **Rust 1.75.0**

### MSRV Updates
The project follows a conservative MSRV policy:
- MSRV updates occur only when necessary for new language features or dependency requirements
- At least 3 months notice before MSRV bumps
- MSRV changes are considered breaking changes and will increment the minor version

## Dependencies

### Core Dependencies
Based on `Cargo.toml`:
```toml
[dependencies]
thiserror = "1.0"          # Error handling
anyhow = "1.0"             # Error context
num-traits = "0.2"         # Numeric traits
ordered-float = "4.2"      # Ordered floating point
indexmap = "2.2"           # Ordered hash maps
byteorder = "1.5"          # Byte order utilities
bitflags = "2.4"           # Bit flag macros
rand = "0.8"               # Random number generation
```

### Optional Dependencies
```toml
rayon = { version = "1.8", optional = true }    # Parallel processing
serde = { version = "1.0", optional = true }    # Serialization
bincode = { version = "1.3", optional = true }  # Binary serialization
```

### Development Dependencies
```toml
[dev-dependencies]
criterion = "0.5"          # Benchmarking
proptest = "1.4"           # Property-based testing
pretty_assertions = "1.4"  # Better test assertions
approx = "0.5"             # Floating point comparisons
rayon = "1.8"              # Parallel testing
```

## Feature Flags

### Available Features
```toml
[features]
default = ["std", "parallel", "serde"]
std = []                   # Standard library support
parallel = ["rayon"]       # Multi-threaded algorithms
serde = ["dep:serde", "bincode", "ordered-float/serde"]  # Serialization
```

### Feature Compatibility
| Feature | Linux | macOS | Windows | Description |
|---------|-------|-------|---------|-------------|
| `std` | ✅ | ✅ | ✅ | Standard library support (default) |
| `parallel` | ✅ | ✅ | ✅ | Rayon-based parallel algorithms |
| `serde` | ✅ | ✅ | ✅ | Serialization support |

### Core Functionality
All core FST operations work across all supported platforms:
- FST construction and manipulation
- All semiring types
- Algorithms (compose, determinize, minimize, etc.)
- I/O operations (text and binary formats)
- Property system

## API Stability

### Current Status (v0.1.x)
- **Pre-1.0**: APIs may change between minor versions
- **Breaking changes**: Will be documented in CHANGELOG.md
- **Deprecation**: Deprecated APIs will be marked and documented

### Stable Core APIs
These APIs are unlikely to change significantly:
- Core FST traits (`Fst`, `MutableFst`, `ExpandedFst`)
- Semiring trait and implementations
- Basic algorithms (compose, determinize, minimize)
- Common weight types

### Potentially Unstable APIs
These may change before 1.0:
- Internal implementation details
- Experimental algorithms
- Performance optimization interfaces
- Error types and messages

## Testing

### Test Coverage
- **Unit tests**: Core functionality and algorithms
- **Integration tests**: End-to-end workflows
- **Property tests**: Mathematical properties using proptest
- **Benchmarks**: Performance regression testing

### Current Testing Status
- **macOS**: Actively tested during development
- **Linux**: Not yet tested (planned)
- **Windows**: Not yet tested (planned)

### Planned CI/CD Setup
To set up cross-platform testing, create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    name: Test
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        rust: [stable, 1.75.0]  # MSRV testing

    steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@master
      with:
        toolchain: ${{ matrix.rust }}
    - run: cargo test --all-features
    - run: cargo test --no-default-features
    - run: cargo bench --no-run  # Compile benchmarks

  clippy:
    name: Clippy
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
      with:
        components: clippy
    - run: cargo clippy --all-features -- -D warnings

  fmt:
    name: Format
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
      with:
        components: rustfmt
    - run: cargo fmt --all -- --check
```

## Performance

### Optimization Levels
- **Debug builds**: No optimizations, debug symbols
- **Release builds**: Standard Rust optimizations
- **Benchmark builds**: LTO and additional optimizations

### Memory Safety
- All code is memory-safe Rust
- No unsafe code in public APIs
- Comprehensive bounds checking

## Version History

### Current Version
- **v0.1.0**: Initial release with core FST functionality

### Planned Releases
Future versions will focus on:
- API stabilization
- Performance improvements  
- Additional algorithms
- Enhanced documentation

## Migration Guide

### Breaking Changes
When breaking changes occur, migration guides will be provided in:
- `CHANGELOG.md` 
- Release notes
- Documentation updates

### Deprecation Process
1. Mark APIs as deprecated with `#[deprecated]`
2. Provide migration path in documentation
3. Keep deprecated APIs for at least one minor version
4. Remove in next major version

## Platform-Specific Notes

### macOS (Currently Tested)
- **Tested on**: macOS during development (Intel and Apple Silicon)
- **Toolchain**: Xcode command line tools
- **Status**: Full feature support confirmed

### Linux (Planned Testing)
- **Target**: Ubuntu 20.04+ and common distributions
- **Toolchain**: Standard GNU toolchain
- **Status**: Expected to work but not yet verified

### Windows (Planned Testing)
- **Target**: Windows 10/11
- **Toolchain**: MSVC toolchain via rust-msvc
- **Status**: Expected to work but not yet verified

### Other Platforms
The library should theoretically work on:
- FreeBSD, NetBSD, OpenBSD
- Other Unix-like systems
- Embedded targets with `std` feature disabled

**Note**: Platform support claims are conservative until CI testing is established.

## Security

### Security Policy
- Regular dependency updates
- Security advisories through GitHub
- Prompt response to security issues

### Safe Defaults
- Memory safety guaranteed by Rust
- Input validation on public APIs
- Proper error handling throughout

## Support Channels

### Getting Help
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community support
- **Documentation**: Comprehensive guides and API docs

### Contributing
See `CONTRIBUTING.md` for:
- Development setup
- Coding standards
- Testing requirements
- Pull request process

## Future Compatibility

### Planned Improvements
- Additional semiring types
- More FST algorithms
- Performance optimizations
- Better error messages

### Long-term Goals
- Stable 1.0 API
- Expanded platform support
- Integration with other libraries
- Advanced optimization features

The ArcWeight project is committed to maintaining backward compatibility and providing clear migration paths for any necessary breaking changes.