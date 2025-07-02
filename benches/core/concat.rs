//! Concatenation operation benchmarks
//!
//! These benchmarks measure the performance of FST concatenation operations across
//! different FST sizes and structures.

use arcweight::algorithms::concat;
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

/// Create an FST with multiple final states
fn create_multi_final_fst(size: usize, final_count: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();

    if size == 0 {
        return fst;
    }

    // Add states
    let states: Vec<_> = (0..size).map(|_| fst.add_state()).collect();

    // Set start state
    fst.set_start(states[0]);

    // Set multiple final states
    let final_states = final_count.min(size);
    for i in 0..final_states {
        let final_idx = size - 1 - i;
        fst.set_final(states[final_idx], TropicalWeight::new((i + 1) as f32));
    }

    // Add transitions
    for i in 0..size - 1 {
        fst.add_arc(
            states[i],
            Arc::new(
                (i % 26 + 97) as u32,
                (i % 26 + 97) as u32,
                TropicalWeight::new(1.0),
                states[i + 1],
            ),
        );

        // Add some branching
        if i < size - 2 {
            fst.add_arc(
                states[i],
                Arc::new(
                    ((i + 13) % 26 + 97) as u32,
                    ((i + 13) % 26 + 97) as u32,
                    TropicalWeight::new(2.0),
                    states[i + 2],
                ),
            );
        }
    }

    fst
}

fn bench_concat_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("concat_linear");

    for size in [10, 100, 1000].iter() {
        let fst1 = create_linear_fst(*size);
        let fst2 = create_linear_fst(*size);

        group.bench_function(format!("linear_{}", size), |b| {
            b.iter(|| {
                let result: VectorFst<TropicalWeight> =
                    concat(black_box(&fst1), black_box(&fst2)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_concat_branching(c: &mut Criterion) {
    let mut group = c.benchmark_group("concat_branching");

    for size in [10, 100, 500].iter() {
        let fst1 = create_branching_fst(*size, 3);
        let fst2 = create_branching_fst(*size, 3);

        group.bench_function(format!("branching_{}", size), |b| {
            b.iter(|| {
                let result: VectorFst<TropicalWeight> =
                    concat(black_box(&fst1), black_box(&fst2)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_concat_multi_final(c: &mut Criterion) {
    let mut group = c.benchmark_group("concat_multi_final");

    for size in [50, 200, 500].iter() {
        let fst1 = create_multi_final_fst(*size, 5); // 5 final states
        let fst2 = create_linear_fst(*size);

        group.bench_function(format!("multi_final_{}", size), |b| {
            b.iter(|| {
                let result: VectorFst<TropicalWeight> =
                    concat(black_box(&fst1), black_box(&fst2)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_concat_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("concat_mixed");

    for size in [50, 200, 500].iter() {
        let linear_fst = create_linear_fst(*size);
        let branching_fst = create_branching_fst(*size / 2, 4);

        group.bench_function(format!("linear_x_branching_{}", size), |b| {
            b.iter(|| {
                let result: VectorFst<TropicalWeight> =
                    concat(black_box(&linear_fst), black_box(&branching_fst)).unwrap();
                black_box(result)
            })
        });

        group.bench_function(format!("branching_x_linear_{}", size), |b| {
            b.iter(|| {
                let result: VectorFst<TropicalWeight> =
                    concat(black_box(&branching_fst), black_box(&linear_fst)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_concat_empty(c: &mut Criterion) {
    let mut group = c.benchmark_group("concat_empty");

    let empty_fst = VectorFst::<TropicalWeight>::new();
    let linear_fst = create_linear_fst(100);

    group.bench_function("empty_x_linear", |b| {
        b.iter(|| {
            match concat::<TropicalWeight, _, _, VectorFst<TropicalWeight>>(
                black_box(&empty_fst),
                black_box(&linear_fst),
            ) {
                Ok(result) => black_box(result),
                Err(_) => black_box(VectorFst::<TropicalWeight>::new()),
            }
        })
    });

    group.bench_function("linear_x_empty", |b| {
        b.iter(|| {
            match concat::<TropicalWeight, _, _, VectorFst<TropicalWeight>>(
                black_box(&linear_fst),
                black_box(&empty_fst),
            ) {
                Ok(result) => black_box(result),
                Err(_) => black_box(VectorFst::<TropicalWeight>::new()),
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_concat_linear,
    bench_concat_branching,
    bench_concat_multi_final,
    bench_concat_mixed,
    bench_concat_empty
);
criterion_main!(benches);
