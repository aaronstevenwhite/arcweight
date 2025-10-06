# Performance Benchmarks

This document presents performance measurements of ArcWeight operations using the Criterion.rs benchmarking framework.

## Test Environment

- Architecture: ARM64
- Operating System: macOS 15.5
- Rust Version: 1.85.0
- Benchmark Framework: Criterion.rs
- Compilation Profile: Release with optimizations

## Core Operations

### FST Creation

Time complexity: O(V + E) where V = states, E = arcs

| FST Type | States | Time | States/sec |
|----------|--------|------|------------|
| Linear | 100 | 5.80 µs | 17,241 |
| Linear | 1,000 | 51.85 µs | 19,286 |
| Branching | 100 | 5.93 µs | 16,863 |

Creation time scales linearly with FST size. Branching topology shows minimal overhead.

### Shortest Path

Time complexity: O((V + E) log V) using priority queue

| FST Type | States | Time | States/sec |
|----------|--------|------|------------|
| Linear | 1,000 | 66.44 µs | 15,051 |
| Linear | 10,000 | 707.13 µs | 14,142 |
| Branching | 100 | 7.32 µs | 13,661 |
| Branching | 500 | 33.37 µs | 14,978 |

Observed scaling is near-linear for typical FST structures, better than theoretical worst-case.

### Composition

Time complexity: O(V₁ × V₂ × D₁ × D₂) worst case, where D = out-degree

| FST₁ × FST₂ | Time | 
|-------------|------|
| 100 × 100 (linear) | 12.41 µs |
| 100 × 50 (mixed) | 487.0 ns |

Composition benefits from on-the-fly construction and filter optimization.

### Determinization

Time complexity: O(2^V) worst case (exponential in subset construction)

| FST Type | States | Time |
|----------|--------|------|
| Linear | 100 | 29.11 µs |
| Branching | 50 | 26.04 µs |

Practical performance avoids exponential blowup for typical FST structures.

## Memory Operations

### Large FST Performance

| Operation | 10,000 States | 50,000 States | Scaling |
|-----------|---------------|---------------|---------|
| Creation | 1.07 ms | 5.78 ms | 5.4× |
| Clone | 497.1 µs | 3.47 ms | 7.0× |
| Clear | 700.5 ps | 708.1 ps | ~1× |

Memory operations scale linearly with FST size. Clear operation is O(1) as it only resets internal state.

## Parallel Processing Analysis

### Sequential vs Parallel Performance

| Operation | Sequential | Parallel | Overhead Factor |
|-----------|------------|----------|-----------------|
| Arc Count | 417.7 ns | 33.41 µs | 80× |
| Arc Iteration | 729.4 ns | 30.42 µs | 42× |
| Weight Sum | 2.79 µs | 54.43 µs | 19× |

Parallel processing incurs significant overhead for small FSTs due to thread coordination costs. Sequential processing is recommended for FSTs with fewer than 100,000 arcs.

## Complexity Analysis

### Theoretical vs Observed Complexity

| Algorithm | Theoretical Complexity | Observed Behavior |
|-----------|----------------------|-------------------|
| Creation | O(V + E) | Linear |
| Shortest Path | O((V + E) log V) | Near-linear |
| Composition | O(V₁ × V₂ × D₁ × D₂) | Sublinear for small FSTs |
| Determinization | O(2^V) worst case | Linear to quadratic typical |
| Union | O(V₁ + V₂ + E₁ + E₂) | Linear |
| Minimization | O(2^V) worst case* | Polynomial typical |

*Minimization uses Brzozowski's algorithm with double reversal and determinization.

## Performance Guidelines

### FST Size Categories

| Size | States | Recommendations |
|------|--------|-----------------|
| Small | < 1,000 | All operations < 100 µs |
| Medium | 1,000-10,000 | Use sequential processing |
| Large | > 10,000 | Consider optimization strategies |

### Optimization Strategies

1. Minimize FSTs before composition or determinization
2. Use ConstFst for read-only operations
3. Prefer sequential processing for typical workloads
4. Profile memory usage for large FSTs

## Reproducing Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific categories
cargo bench --bench basic_operations
cargo bench --bench memory_usage
cargo bench --bench shortest_path

# Quick benchmarks
cargo bench -- --quick
```

Expected variance: ±5-10% due to system load and thermal throttling.

## Further Reading

- [Core Concepts](core-concepts/) — Theoretical foundations
- [Working with FSTs](working-with-fsts/) — Operations guide
- [Architecture](architecture/) — Implementation details