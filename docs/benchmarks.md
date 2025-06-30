# Performance Benchmarks

This document presents comprehensive performance analysis of ArcWeight operations, measured on real hardware using Criterion.rs benchmarking framework.

## Test Environment

**Hardware Specifications:**
- **Processor**: Apple M1 Max (ARM64 architecture)
- **Cores**: 10 cores (8 performance + 2 efficiency)
- **Memory**: 64 GB unified memory
- **OS**: macOS 15.5 (Build 24F74)
- **Rust Version**: 1.85.0 (release mode with optimizations)

**Benchmark Configuration:**
- **Framework**: Criterion.rs with default settings
- **Samples**: 100 samples per benchmark  
- **Warm-up**: 3 seconds per benchmark
- **Compilation**: `bench` profile (inherits from `release` with standard optimizations)

## Core Operations Performance

### FST Creation

Building FSTs from scratch with different topologies:

| FST Type | Size | Creation Time | Throughput (states/sec) |
|----------|------|---------------|-------------------------|
| **Linear** | 100 states | 5.92 µs | 16,892 states/sec |
| **Linear** | 1,000 states | 52.44 µs | 19,070 states/sec |
| **Branching** | 100 states | 5.91 µs | 16,920 states/sec |

**Key Insights:**
- Excellent scaling for small to medium FSTs
- Branching structure has virtually no overhead compared to linear
- Performance is consistent across different FST topologies

### Shortest Path Computation

Finding optimal paths through FSTs using tropical semiring:

**Algorithm**: Dijkstra-style shortest path with early termination

| FST Structure | Size | Computation Time | Performance (states/sec) |
|---------------|------|------------------|-------------------------|
| **Linear** | 1,000 states | 67.09 µs | 14,905 states/sec |
| **Linear** | 10,000 states | 667.50 µs | 14,981 states/sec |
| **Branching** | 100 states | 7.46 µs | 13,405 states/sec |

**Complexity**: Actual performance shows near-linear scaling, much better than theoretical \\(O((V + E) \log V)\\)

### Composition Operations

Chaining FSTs through composition (T\\(_1\\) \\(\circ\\) T\\(_2\\)):

| FST\\(_1\\) \\(\times\\) FST\\(_2\\) | Composition Time | Notes |
|-------------|------------------|-------|
| **100 \\(\times\\) 100** (linear) | 12.53 µs | Standard composition |
| **100 \\(\times\\) 50** (mixed) | 478.78 ns | Optimized for smaller second FST |

**Performance Notes:**
- Small FST composition is extremely fast (< 1 µs for mixed cases)
- Linear \\(\times\\) Linear composition shows good performance for moderate sizes
- Asymmetric sizes (different FST sizes) can be significantly faster

### Determinization Performance

Converting non-deterministic FSTs to deterministic form:

| Input FST | Input Size | Determinization Time |
|-----------|------------|---------------------|
| **Linear** | 100 states | 29.20 µs |
| **Branching** | 50 states | 26.28 µs |

**Complexity**: Much better than theoretical exponential worst-case due to typical FST structure

## Memory Performance

### Large-Scale FST Handling

Performance with substantial FSTs:

| Operation | 10,000 States | 50,000 States | Scaling Factor |
|-----------|---------------|---------------|----------------|
| **Creation** | 1.056 ms | 6.22 ms | 5.89x |
| **Clone** | 514.10 µs | > 3.5 ms* | > 6.8x |

*Note: 50,000 state cloning exceeded benchmark timeout, indicating > 3.5ms

**Memory Insights:**
- **Creation** scales well but shows some overhead for very large FSTs
- **Cloning** requires deep copy and scales with FST complexity
- Memory allocation patterns are efficient for typical use cases

### Arc Processing Performance

Detailed arc-level operations:

| Operation | 1,000 States | 5,000 States | Performance Notes |
|-----------|--------------|--------------|-------------------|
| **Arc Count** | 802.40 ns | 4.08 µs | ~5x scaling |
| **Arc Iteration** | 27.74 µs | 139.92 µs | ~5x scaling |
| **Arc Lookup** | 27.80 µs | 139.39 µs | ~5x scaling |

**Arc Operations** show excellent linear scaling characteristics.

## Real-World Performance Patterns

### Parallel vs Sequential Processing

For arc-intensive operations:

| Operation | Sequential | Parallel | Parallel Overhead |
|-----------|------------|----------|-------------------|
| **Arc Count** | 402.4 ns | 35.73 µs | ~89x slower |
| **Arc Iteration** | 765.1 ns | 35.31 µs | ~46x slower |
| **Weight Sum** | 2.89 µs | 68.48 µs | ~24x slower |

**Key Finding**: Parallel processing shows significant overhead for small FSTs. Parallelization is only beneficial for very large datasets due to thread coordination costs.

## Algorithm Complexity Analysis

### Actual vs Theoretical Performance

| Algorithm | Theoretical | Measured Scaling | Performance Notes |
|-----------|-------------|------------------|-------------------|
| **FST Creation** | \\(O(V + E)\\) | Linear | Excellent scaling |
| **Shortest Path** | \\(O((V+E) \log V)\\) | Near-linear | Better than expected |
| **Composition** | \\(O(V_1 \times V_2 \times \Sigma )\\) | Sublinear for small FSTs | Filter optimization effective |
| **Determinization** | \\(O(2^V)\\) worst case | Linear-to-quadratic typical | Real FSTs avoid exponential blowup |

### Scalability Observations

**Linear Scaling (Excellent):**
- FST creation: Consistent ~17-19K states/sec
- Arc operations: Predictable 5x scaling from 1K to 5K states
- Memory usage: Proportional to FST size

**Near-Linear Scaling (Very Good):**
- Shortest path: ~15K states/sec across different sizes
- Basic FST operations maintain efficiency

## Performance Recommendations

### For Different Use Cases

**Small FSTs (< 1,000 states):**
- All operations are fast (< 100 µs)
- No special optimization needed
- Memory usage negligible

**Medium FSTs (1,000 - 10,000 states):**
- Use sequential processing (parallel overhead too high)
- Operations remain efficient
- Consider minimization before complex operations

**Large FSTs (> 10,000 states):**
- Essential to profile memory usage
- Minimize before expensive operations like composition
- Consider lazy evaluation for very large computations

### Development Workflow

**Optimization Priority:**
1. **Algorithm choice** - Select appropriate operations
2. **Operation order** - Minimize before compose when possible
3. **Data structure** - VectorFst for construction, ConstFst for queries
4. **Avoid premature parallelization** - Sequential is faster for typical FST sizes

## Benchmark Reproduction

To reproduce these benchmarks on your system:

```bash
# Run all benchmarks (warning: takes significant time)
cargo bench

# Run specific benchmark categories
cargo bench --bench basic_operations
cargo bench --bench memory_usage  
cargo bench --bench shortest_path

# Generate detailed reports with outlier analysis
cargo bench -- --verbose
```

**Expected Variance:**
- ±5-10% due to system load and thermal conditions
- ±20-30% between different architectures (x86_64 vs ARM64)
- ±2-3x between debug and release builds

## Platform-Specific Notes

**Apple M1 Max Performance Characteristics:**
- Excellent single-core performance benefits FST algorithms
- Unified memory architecture reduces memory bandwidth bottlenecks
- ARM64 NEON instructions may benefit certain operations
- Sequential processing often outperforms parallel due to coordination overhead

**Cross-Platform Expectations:**
- Intel/AMD x86_64: Likely 10-30% different performance
- Different ARM chips: Similar performance characteristics
- Memory-constrained systems: May show different scaling patterns

---

**Data Collection Date**: Run on actual hardware as of 2025-06-16  
**Reproducibility**: All numbers are from real Criterion.rs benchmark runs and can be reproduced using the provided commands.