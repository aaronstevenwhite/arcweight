//! Union operation benchmarks
//!
//! These benchmarks measure the performance of FST union operations across
//! different FST sizes and structures.

use arcweight::algorithms::union;
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
                TropicalWeight::new(1.0),
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

fn bench_union_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("union_linear");

    for size in [10, 100, 1000].iter() {
        let fst1 = create_linear_fst(*size);
        let fst2 = create_linear_fst(*size);

        group.bench_function(format!("linear_{}", size), |b| {
            b.iter(|| {
                let result: VectorFst<TropicalWeight> =
                    union(black_box(&fst1), black_box(&fst2)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_union_branching(c: &mut Criterion) {
    let mut group = c.benchmark_group("union_branching");

    for size in [10, 100, 500].iter() {
        let fst1 = create_branching_fst(*size, 3);
        let fst2 = create_branching_fst(*size, 3);

        group.bench_function(format!("branching_{}", size), |b| {
            b.iter(|| {
                let result: VectorFst<TropicalWeight> =
                    union(black_box(&fst1), black_box(&fst2)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_union_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("union_mixed");

    for size in [50, 200, 500].iter() {
        let linear_fst = create_linear_fst(*size);
        let branching_fst = create_branching_fst(*size / 2, 4);

        group.bench_function(format!("linear_x_branching_{}", size), |b| {
            b.iter(|| {
                let result: VectorFst<TropicalWeight> =
                    union(black_box(&linear_fst), black_box(&branching_fst)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_union_empty(c: &mut Criterion) {
    let mut group = c.benchmark_group("union_empty");

    let empty_fst = VectorFst::<TropicalWeight>::new();
    let linear_fst = create_linear_fst(100);

    group.bench_function("empty_x_linear", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> =
                union(black_box(&empty_fst), black_box(&linear_fst)).unwrap();
            black_box(result)
        })
    });

    group.bench_function("linear_x_empty", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> =
                union(black_box(&linear_fst), black_box(&empty_fst)).unwrap();
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_union_linear,
    bench_union_branching,
    bench_union_mixed,
    bench_union_empty
);
criterion_main!(benches);
