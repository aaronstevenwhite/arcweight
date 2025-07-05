//! Reverse operation benchmarks
//!
//! These benchmarks measure the performance of FST reversal operations across
//! different FST sizes and structures.

use arcweight::algorithms::reverse;
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
                (i % 26 + 97) as u32,       // cycle through 'a'-'z'
                ((i + 5) % 26 + 97) as u32, // different output symbols
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
                    ((i + j) % 26 + 97) as u32,
                    ((i + j + 10) % 26 + 97) as u32, // different output
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

    // Set multiple final states with different weights
    let final_states = final_count.min(size);
    for i in 0..final_states {
        let final_idx = size - 1 - i;
        fst.set_final(states[final_idx], TropicalWeight::new((i + 1) as f32));
    }

    // Add transitions creating a more complex structure
    for i in 0..size - 1 {
        fst.add_arc(
            states[i],
            Arc::new(
                (i % 26 + 97) as u32,
                ((i + 3) % 26 + 97) as u32,
                TropicalWeight::new(1.0),
                states[i + 1],
            ),
        );

        // Add some branching to later states
        if i < size - 2 {
            fst.add_arc(
                states[i],
                Arc::new(
                    ((i + 13) % 26 + 97) as u32,
                    ((i + 16) % 26 + 97) as u32,
                    TropicalWeight::new(2.0),
                    states[i + 2],
                ),
            );
        }
    }

    fst
}

/// Create an FST with cycles and self-loops
fn create_cyclic_fst(size: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();

    if size == 0 {
        return fst;
    }

    // Add states
    let states: Vec<_> = (0..size).map(|_| fst.add_state()).collect();

    // Set start and final states
    fst.set_start(states[0]);
    fst.set_final(states[size - 1], TropicalWeight::one());

    // Add main path
    for i in 0..size - 1 {
        fst.add_arc(
            states[i],
            Arc::new(
                (i % 26 + 97) as u32,
                ((i + 7) % 26 + 97) as u32,
                TropicalWeight::new(1.0),
                states[i + 1],
            ),
        );
    }

    // Add self-loops
    for i in (0..size).step_by(3) {
        fst.add_arc(
            states[i],
            Arc::new(
                ((i + 26) % 26 + 97) as u32,
                ((i + 33) % 26 + 97) as u32,
                TropicalWeight::new(0.5),
                states[i],
            ),
        );
    }

    // Add back edges (cycles)
    if size > 3 {
        for i in 2..size {
            if i % 4 == 0 {
                fst.add_arc(
                    states[i],
                    Arc::new(
                        ((i + 20) % 26 + 97) as u32,
                        ((i + 23) % 26 + 97) as u32,
                        TropicalWeight::new(3.0),
                        states[i - 2],
                    ),
                );
            }
        }
    }

    fst
}

fn bench_reverse_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_linear");

    for size in [10, 100, 1000, 5000].iter() {
        let fst = create_linear_fst(*size);

        group.bench_function(format!("linear_{size}"), |b| {
            b.iter(|| {
                let result: VectorFst<TropicalWeight> = reverse(black_box(&fst)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_reverse_branching(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_branching");

    for size in [10, 100, 500, 1000].iter() {
        let fst = create_branching_fst(*size, 3);

        group.bench_function(format!("branching_{size}"), |b| {
            b.iter(|| {
                let result: VectorFst<TropicalWeight> = reverse(black_box(&fst)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_reverse_multi_final(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_multi_final");

    for size in [50, 200, 500, 1000].iter() {
        let fst = create_multi_final_fst(*size, 5);

        group.bench_function(format!("multi_final_{size}"), |b| {
            b.iter(|| {
                let result: VectorFst<TropicalWeight> = reverse(black_box(&fst)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_reverse_cyclic(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_cyclic");

    for size in [10, 50, 200, 500].iter() {
        let fst = create_cyclic_fst(*size);

        group.bench_function(format!("cyclic_{size}"), |b| {
            b.iter(|| {
                let result: VectorFst<TropicalWeight> = reverse(black_box(&fst)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_reverse_transducer_vs_acceptor(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_transducer_vs_acceptor");

    let size = 200;

    // Create transducer (different input/output symbols)
    let transducer = create_linear_fst(size);

    // Create acceptor (same input/output symbols)
    let mut acceptor = VectorFst::new();
    let states: Vec<_> = (0..size).map(|_| acceptor.add_state()).collect();
    acceptor.set_start(states[0]);
    acceptor.set_final(states[size - 1], TropicalWeight::one());

    for i in 0..size - 1 {
        let symbol = (i % 26 + 97) as u32;
        acceptor.add_arc(
            states[i],
            Arc::new(symbol, symbol, TropicalWeight::new(1.0), states[i + 1]),
        );
    }

    group.bench_function("transducer", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> = reverse(black_box(&transducer)).unwrap();
            black_box(result)
        })
    });

    group.bench_function("acceptor", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> = reverse(black_box(&acceptor)).unwrap();
            black_box(result)
        })
    });

    group.finish();
}

fn bench_reverse_empty_and_trivial(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_empty_and_trivial");

    let empty_fst = VectorFst::<TropicalWeight>::new();

    group.bench_function("empty", |b| {
        b.iter(|| {
            match reverse::<TropicalWeight, _, VectorFst<TropicalWeight>>(black_box(&empty_fst)) {
                Ok(result) => black_box(result),
                Err(_) => black_box(VectorFst::<TropicalWeight>::new()),
            }
        })
    });

    // Single state FST
    let mut single_state = VectorFst::new();
    let state = single_state.add_state();
    single_state.set_start(state);
    single_state.set_final(state, TropicalWeight::one());

    group.bench_function("single_state", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> = reverse(black_box(&single_state)).unwrap();
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_reverse_linear,
    bench_reverse_branching,
    bench_reverse_multi_final,
    bench_reverse_cyclic,
    bench_reverse_transducer_vs_acceptor,
    bench_reverse_empty_and_trivial
);
criterion_main!(benches);
