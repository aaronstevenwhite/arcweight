# ArcWeight Compatibility

This document outlines platform support, MSRV policy, and feature compatibility for ArcWeight.

## Platform Support

### Operating Systems
- Linux (x86_64, aarch64)
- macOS (x86_64, aarch64)
- Windows (x86_64)

### Minimum Requirements
- Rust 1.70.0 or later
- 2GB RAM minimum
- 100MB disk space

## MSRV Policy

### Current MSRV
- Rust 1.70.0

### MSRV Updates
- Updated every 6 months
- 3 months notice before updates
- Support for current stable and previous version

## Feature Compatibility

### Core Features
| Feature | Linux | macOS | Windows |
|---------|-------|-------|---------|
| FST Operations | ✅ | ✅ | ✅ |
| Weight Semirings | ✅ | ✅ | ✅ |
| IO Operations | ✅ | ✅ | ✅ |
| Parallel Processing | ✅ | ✅ | ✅ |

### Optional Features
| Feature | Linux | macOS | Windows |
|---------|-------|-------|---------|
| GPU Acceleration | ✅ | ✅ | ❌ |
| Memory Mapping | ✅ | ✅ | ✅ |
| Custom Allocators | ✅ | ✅ | ✅ |

## Dependencies

### Required Dependencies
```toml
[dependencies]
serde = "1.0"
thiserror = "1.0"
```

### Optional Dependencies
```toml
[dependencies]
rayon = { version = "1.7", optional = true }
cuda = { version = "0.1", optional = true }
```

## Build Configuration

### Feature Flags
```toml
[features]
default = ["std"]
std = []
parallel = ["rayon"]
gpu = ["cuda"]
```

### Build Options
```toml
[build-dependencies]
cc = "1.0"
```

## API Compatibility

### Stable APIs
- Core FST operations
- Weight semiring traits
- IO operations
- Property system

### Unstable APIs
- Experimental algorithms
- Custom optimizations
- Internal utilities

## Version Compatibility

### Major Versions
- 0.1.x: Initial release
- 0.2.x: Performance improvements
- 0.3.x: API stabilization

### Breaking Changes
- Documented in CHANGELOG.md
- Migration guides provided
- Deprecation notices

## Testing Compatibility

### Test Environments
- Linux (CI)
- macOS (CI)
- Windows (CI)

### Test Coverage
- Unit tests
- Integration tests
- Property tests
- Benchmarks

## Performance Compatibility

### Optimization Levels
- Debug: No optimizations
- Release: Standard optimizations
- Release with LTO: Full optimizations

### Memory Models
- Default allocator
- Custom allocator
- Pool allocator

## Security Compatibility

### Security Features
- Safe memory management
- Input validation
- Error handling

### Security Requirements
- Rust security updates
- Dependency updates
- Security audits

## Documentation Compatibility

### Documentation Formats
- rustdoc
- mdBook
- GitHub Pages

### Documentation Tools
- cargo doc
- mdbook
- rustdoc-stripper

## Community Support

### Communication Channels
- GitHub Issues
- GitHub Discussions
- Discord

### Contribution Guidelines
- Code of Conduct
- Contributing Guide
- Development Guide

## Future Compatibility

### Planned Support
- Additional platforms
- New architectures
- Enhanced features

### Deprecation Policy
- 6 months notice
- Migration guides
- Alternative solutions 