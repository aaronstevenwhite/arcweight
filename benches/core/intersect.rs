//! Intersection operation benchmarks
//!
//! These benchmarks measure the performance of FST intersection operations across
//! different FST sizes and structures.

use arcweight::algorithms::intersect;
use arcweight::prelude::*;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

/// Create a linear FST with the specified number of states
fn create_linear_fst(size: usize, symbol_offset: u32) -> VectorFst<TropicalWeight> {
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
                (i % 26 + 97 + symbol_offset as usize) as u32, // cycle through symbols with offset
                (i % 26 + 97 + symbol_offset as usize) as u32,
                TropicalWeight::new(1.0),
                states[i + 1],
            ),
        );
    }

    fst
}

/// Create a branching FST where each state has multiple outgoing arcs
fn create_branching_fst(
    states: usize,
    branches: usize,
    symbol_offset: u32,
) -> VectorFst<TropicalWeight> {
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
            let symbol = ((i + j) % 26 + 97 + symbol_offset as usize) as u32;
            fst.add_arc(
                state_ids[i],
                Arc::new(
                    symbol,
                    symbol,
                    TropicalWeight::new((j + 1) as f32),
                    target_state,
                ),
            );
        }
    }

    fst
}

/// Create an FST that accepts overlapping patterns with another FST
fn create_overlapping_fst(size: usize, pattern_length: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();

    if size == 0 || pattern_length == 0 {
        return fst;
    }

    // Add states
    let states: Vec<_> = (0..size).map(|_| fst.add_state()).collect();

    // Set start and final states
    fst.set_start(states[0]);
    fst.set_final(states[size - 1], TropicalWeight::one());

    // Create repeating pattern
    let pattern: Vec<u32> = (0..pattern_length).map(|i| (i % 26 + 97) as u32).collect();

    // Add transitions following the pattern
    for i in 0..size - 1 {
        let symbol = pattern[i % pattern_length];
        fst.add_arc(
            states[i],
            Arc::new(symbol, symbol, TropicalWeight::new(1.0), states[i + 1]),
        );

        // Add some alternative paths that break the pattern occasionally
        if i % 7 == 0 && i < size - 2 {
            let alt_symbol = (symbol - 97 + 13) % 26 + 97; // rotate by 13
            fst.add_arc(
                states[i],
                Arc::new(
                    alt_symbol,
                    alt_symbol,
                    TropicalWeight::new(2.0),
                    states[i + 2],
                ),
            );
        }
    }

    fst
}

fn bench_intersect_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersect_linear");

    for size in [10, 100, 500].iter() {
        // Create FSTs with overlapping symbol sets
        let fst1 = create_linear_fst(*size, 0);
        let fst2 = create_linear_fst(*size, 0); // Same symbols

        group.bench_function(format!("identical_linear_{}", size), |b| {
            b.iter(|| {
                let result: VectorFst<TropicalWeight> =
                    intersect(black_box(&fst1), black_box(&fst2)).unwrap();
                black_box(result)
            })
        });

        // Create FSTs with partially overlapping symbols
        let fst3 = create_linear_fst(*size, 5); // Offset symbols

        group.bench_function(format!("partial_overlap_linear_{}", size), |b| {
            b.iter(|| {
                let result: VectorFst<TropicalWeight> =
                    intersect(black_box(&fst1), black_box(&fst3)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_intersect_branching(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersect_branching");

    for size in [10, 50, 200].iter() {
        let fst1 = create_branching_fst(*size, 3, 0);
        let fst2 = create_branching_fst(*size, 3, 0);

        group.bench_function(format!("branching_{}", size), |b| {
            b.iter(|| {
                let result: VectorFst<TropicalWeight> =
                    intersect(black_box(&fst1), black_box(&fst2)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_intersect_overlapping_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersect_overlapping_patterns");

    for size in [20, 100, 300].iter() {
        // FSTs with overlapping but different patterns
        let fst1 = create_overlapping_fst(*size, 5); // pattern length 5
        let fst2 = create_overlapping_fst(*size, 7); // pattern length 7

        group.bench_function(format!("pattern_overlap_{}", size), |b| {
            b.iter(|| {
                let result: VectorFst<TropicalWeight> =
                    intersect(black_box(&fst1), black_box(&fst2)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_intersect_disjoint(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersect_disjoint");

    for size in [50, 200, 500].iter() {
        // Create FSTs with completely disjoint symbol sets
        let fst1 = create_linear_fst(*size, 0); // symbols a-z
        let fst2 = create_linear_fst(*size, 26); // symbols A-Z (shifted by 26)

        group.bench_function(format!("disjoint_{}", size), |b| {
            b.iter(|| {
                let result: VectorFst<TropicalWeight> =
                    intersect(black_box(&fst1), black_box(&fst2)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_intersect_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersect_mixed");

    for size in [50, 200, 400].iter() {
        let linear_fst = create_linear_fst(*size, 0);
        let branching_fst = create_branching_fst(*size / 2, 4, 0);

        group.bench_function(format!("linear_x_branching_{}", size), |b| {
            b.iter(|| {
                let result: VectorFst<TropicalWeight> =
                    intersect(black_box(&linear_fst), black_box(&branching_fst)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_intersect_empty(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersect_empty");

    let empty_fst = VectorFst::<TropicalWeight>::new();
    let linear_fst = create_linear_fst(100, 0);

    group.bench_function("empty_x_linear", |b| {
        b.iter(|| {
            match intersect::<TropicalWeight, _, _, VectorFst<TropicalWeight>>(
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
            match intersect::<TropicalWeight, _, _, VectorFst<TropicalWeight>>(
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
    bench_intersect_linear,
    bench_intersect_branching,
    bench_intersect_overlapping_patterns,
    bench_intersect_disjoint,
    bench_intersect_mixed,
    bench_intersect_empty
);
criterion_main!(benches);
