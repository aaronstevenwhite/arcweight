//! Closure operation benchmarks
//!
//! These benchmarks measure the performance of FST closure operations (Kleene star
//! and plus) across different FST sizes and structures.

use arcweight::algorithms::{closure, closure_plus};
use arcweight::prelude::*;
use arcweight::semiring::BooleanWeight;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

/// Create a linear FST with the specified number of states
fn create_linear_fst(size: usize) -> VectorFst<BooleanWeight> {
    let mut fst = VectorFst::new();

    if size == 0 {
        return fst;
    }

    // Add states
    let states: Vec<_> = (0..size).map(|_| fst.add_state()).collect();

    // Set start and final states
    fst.set_start(states[0]);
    fst.set_final(states[size - 1], BooleanWeight::one());

    // Add transitions in a chain
    for i in 0..size - 1 {
        fst.add_arc(
            states[i],
            Arc::new(
                (i % 26 + 97) as u32, // cycle through 'a'-'z'
                (i % 26 + 97) as u32,
                BooleanWeight::one(),
                states[i + 1],
            ),
        );
    }

    fst
}

/// Create a branching FST where each state has multiple outgoing arcs
fn create_branching_fst(states: usize, branches: usize) -> VectorFst<BooleanWeight> {
    let mut fst = VectorFst::new();

    if states == 0 {
        return fst;
    }

    // Add states
    let state_ids: Vec<_> = (0..states).map(|_| fst.add_state()).collect();

    // Set start state
    fst.set_start(state_ids[0]);
    fst.set_final(state_ids[states - 1], BooleanWeight::one());

    // Add branching transitions
    for i in 0..states - 1 {
        for j in 0..branches.min(states - i - 1) {
            let target_state = state_ids[i + j + 1];
            fst.add_arc(
                state_ids[i],
                Arc::new(
                    ((i + j) % 26 + 97) as u32, // cycle through 'a'-'z'
                    ((i + j) % 26 + 97) as u32,
                    BooleanWeight::one(),
                    target_state,
                ),
            );
        }
    }

    fst
}

/// Create a cyclic FST with self-loops and cycles
fn create_cyclic_fst(size: usize) -> VectorFst<BooleanWeight> {
    let mut fst = VectorFst::new();

    if size == 0 {
        return fst;
    }

    // Add states
    let states: Vec<_> = (0..size).map(|_| fst.add_state()).collect();

    // Set start and final states
    fst.set_start(states[0]);
    fst.set_final(states[size - 1], BooleanWeight::one());

    // Add linear chain
    for i in 0..size - 1 {
        fst.add_arc(
            states[i],
            Arc::new(
                (i % 26 + 97) as u32,
                (i % 26 + 97) as u32,
                BooleanWeight::one(),
                states[i + 1],
            ),
        );
    }

    // Add self-loops on some states
    for i in (0..size).step_by(3) {
        fst.add_arc(
            states[i],
            Arc::new(
                ((i + 26) % 26 + 97) as u32,
                ((i + 26) % 26 + 97) as u32,
                BooleanWeight::one(),
                states[i],
            ),
        );
    }

    // Add some back-cycles
    if size > 3 {
        for i in 2..size {
            if i % 4 == 0 {
                fst.add_arc(
                    states[i],
                    Arc::new(
                        ((i + 10) % 26 + 97) as u32,
                        ((i + 10) % 26 + 97) as u32,
                        BooleanWeight::one(),
                        states[i - 2],
                    ),
                );
            }
        }
    }

    fst
}

fn bench_closure_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("closure_linear");

    for size in [5, 20, 50, 100].iter() {
        let fst = create_linear_fst(*size);

        group.bench_function(format!("kleene_star_{size}"), |b| {
            b.iter(|| {
                let result: VectorFst<BooleanWeight> = closure(black_box(&fst)).unwrap();
                black_box(result)
            })
        });

        group.bench_function(format!("kleene_plus_{size}"), |b| {
            b.iter(|| {
                let result: VectorFst<BooleanWeight> = closure_plus(black_box(&fst)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_closure_branching(c: &mut Criterion) {
    let mut group = c.benchmark_group("closure_branching");

    for size in [5, 20, 50].iter() {
        let fst = create_branching_fst(*size, 3);

        group.bench_function(format!("kleene_star_branching_{size}"), |b| {
            b.iter(|| {
                let result: VectorFst<BooleanWeight> = closure(black_box(&fst)).unwrap();
                black_box(result)
            })
        });

        group.bench_function(format!("kleene_plus_branching_{size}"), |b| {
            b.iter(|| {
                let result: VectorFst<BooleanWeight> = closure_plus(black_box(&fst)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_closure_cyclic(c: &mut Criterion) {
    let mut group = c.benchmark_group("closure_cyclic");

    for size in [5, 15, 30].iter() {
        let fst = create_cyclic_fst(*size);

        group.bench_function(format!("kleene_star_cyclic_{size}"), |b| {
            b.iter(|| {
                let result: VectorFst<BooleanWeight> = closure(black_box(&fst)).unwrap();
                black_box(result)
            })
        });

        group.bench_function(format!("kleene_plus_cyclic_{size}"), |b| {
            b.iter(|| {
                let result: VectorFst<BooleanWeight> = closure_plus(black_box(&fst)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_closure_single_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("closure_single_state");

    // Create single-state FST with self-loop
    let mut fst = VectorFst::new();
    let state = fst.add_state();
    fst.set_start(state);
    fst.set_final(state, BooleanWeight::one());
    fst.add_arc(
        state,
        Arc::new(97, 97, BooleanWeight::one(), state), // 'a' -> 'a'
    );

    group.bench_function("single_state_kleene_star", |b| {
        b.iter(|| {
            let result: VectorFst<BooleanWeight> = closure(black_box(&fst)).unwrap();
            black_box(result)
        })
    });

    group.bench_function("single_state_kleene_plus", |b| {
        b.iter(|| {
            let result: VectorFst<BooleanWeight> = closure_plus(black_box(&fst)).unwrap();
            black_box(result)
        })
    });

    group.finish();
}

fn bench_closure_empty(c: &mut Criterion) {
    let mut group = c.benchmark_group("closure_empty");

    let empty_fst = VectorFst::<BooleanWeight>::new();

    group.bench_function("empty_kleene_star", |b| {
        b.iter(|| {
            let result: VectorFst<BooleanWeight> = closure(black_box(&empty_fst)).unwrap();
            black_box(result)
        })
    });

    group.bench_function("empty_kleene_plus", |b| {
        b.iter(|| {
            let result: VectorFst<BooleanWeight> = closure_plus(black_box(&empty_fst)).unwrap();
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_closure_linear,
    bench_closure_branching,
    bench_closure_cyclic,
    bench_closure_single_state,
    bench_closure_empty
);
criterion_main!(benches);
