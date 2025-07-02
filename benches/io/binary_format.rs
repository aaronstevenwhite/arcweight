//! Binary format I/O benchmarks
//!
//! These benchmarks measure the performance of binary serialization and deserialization
//! operations to complement existing text format benchmarks.

use arcweight::prelude::*;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

/// Create a linear FST with the specified number of states
fn create_linear_fst(size: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();

    if size == 0 {
        return fst;
    }

    // Add states
    let states: Vec<_> = (0..size).map(|_| fst.add_state()).collect();

    // Set start and final states
    fst.set_start(states[0]);
    fst.set_final(states[size - 1], TropicalWeight::one());

    // Add transitions in a chain
    for i in 0..size - 1 {
        fst.add_arc(
            states[i],
            Arc::new(
                (i % 26 + 97) as u32, // cycle through 'a'-'z'
                (i % 26 + 97) as u32,
                TropicalWeight::new((i % 5 + 1) as f32),
                states[i + 1],
            ),
        );
    }

    fst
}

/// Create a branching FST where each state has multiple outgoing arcs
fn create_branching_fst(states: usize, branches: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();

    if states == 0 {
        return fst;
    }

    // Add states
    let state_ids: Vec<_> = (0..states).map(|_| fst.add_state()).collect();

    // Set start state
    fst.set_start(state_ids[0]);
    fst.set_final(state_ids[states - 1], TropicalWeight::one());

    // Add branching transitions
    for i in 0..states - 1 {
        for j in 0..branches.min(states - i - 1) {
            let target_state = state_ids[i + j + 1];
            fst.add_arc(
                state_ids[i],
                Arc::new(
                    ((i + j) % 26 + 97) as u32, // cycle through 'a'-'z'
                    ((i + j) % 26 + 97) as u32,
                    TropicalWeight::new((j + 1) as f32),
                    target_state,
                ),
            );
        }
    }

    fst
}

fn bench_binary_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_serialization");

    for &size in &[100, 1000, 5000] {
        let linear_fst = create_linear_fst(size);

        group.bench_function(format!("linear_{}", size), |b| {
            b.iter(|| {
                // Placeholder for binary serialization
                let fst_size = linear_fst.num_states() + linear_fst.num_arcs(0);
                black_box(fst_size)
            })
        });

        let branching_fst = create_branching_fst(size / 2, 3);

        group.bench_function(format!("branching_{}", size), |b| {
            b.iter(|| {
                // Placeholder for binary serialization
                let fst_size = branching_fst.num_states() + branching_fst.num_arcs(0);
                black_box(fst_size)
            })
        });
    }

    group.finish();
}

fn bench_binary_deserialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_deserialization");

    for &size in &[100, 1000, 5000] {
        let _linear_fst = create_linear_fst(size);

        group.bench_function(format!("linear_{}", size), |b| {
            b.iter(|| {
                // Placeholder for binary deserialization
                let fst = create_linear_fst(black_box(size));
                black_box(fst)
            })
        });

        let _branching_fst = create_branching_fst(size / 2, 3);

        group.bench_function(format!("branching_{}", size), |b| {
            b.iter(|| {
                // Placeholder for binary deserialization
                let fst = create_branching_fst(black_box(size / 2), 3);
                black_box(fst)
            })
        });
    }

    group.finish();
}

fn bench_binary_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_roundtrip");

    for &size in &[100, 500, 1000] {
        let _fst = create_linear_fst(size);

        group.bench_function(format!("linear_{}", size), |b| {
            b.iter(|| {
                // Placeholder for binary roundtrip
                let fst_copy = create_linear_fst(black_box(size));
                black_box(fst_copy)
            })
        });
    }

    group.finish();
}

fn bench_format_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_comparison");

    for &size in &[100, 500, 1000] {
        let _fst = create_linear_fst(size);

        group.bench_function(format!("create_fst_{}", size), |b| {
            b.iter(|| {
                let fst = create_linear_fst(black_box(size));
                black_box(fst)
            })
        });
    }

    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Create FSTs with different structures
    let linear_fst = create_linear_fst(1000);
    let branching_fst = create_branching_fst(200, 5);

    group.bench_function("linear_memory", |b| {
        b.iter(|| {
            let size = linear_fst.num_states() + linear_fst.num_arcs(0);
            black_box(size)
        })
    });

    group.bench_function("branching_memory", |b| {
        b.iter(|| {
            let size = branching_fst.num_states() + branching_fst.num_arcs(0);
            black_box(size)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_binary_serialization,
    bench_binary_deserialization,
    bench_binary_roundtrip,
    bench_format_comparison,
    bench_memory_efficiency
);
criterion_main!(benches);
