//! FST type comparison benchmarks
//!
//! These benchmarks compare the performance of basic operations (arc iteration,
//! state access, etc.) across different FST implementations.

use arcweight::fst::ConstFst;
use arcweight::prelude::*;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

/// Create a linear VectorFst with the specified number of states
fn create_vector_fst(size: usize) -> VectorFst<TropicalWeight> {
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

fn bench_arc_iteration_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("arc_iteration_comparison");
    let size = 1000;

    let vector_fst = create_vector_fst(size);
    let const_fst = ConstFst::from_fst(&vector_fst).unwrap();

    group.bench_function("vector_fst", |b| {
        b.iter(|| {
            let mut arc_count = 0;
            for state_id in 0..vector_fst.num_states() as u32 {
                for _arc in vector_fst.arcs(state_id) {
                    arc_count += 1;
                }
            }
            black_box(arc_count)
        })
    });

    group.bench_function("const_fst", |b| {
        b.iter(|| {
            let mut arc_count = 0;
            for state_id in 0..const_fst.num_states() as u32 {
                for _arc in const_fst.arcs(state_id) {
                    arc_count += 1;
                }
            }
            black_box(arc_count)
        })
    });

    group.finish();
}

fn bench_state_access_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_access_comparison");
    let size = 1000;

    let vector_fst = create_vector_fst(size);
    let const_fst = ConstFst::from_fst(&vector_fst).unwrap();

    group.bench_function("vector_fst_num_arcs", |b| {
        b.iter(|| {
            let mut total_arcs = 0;
            for state_id in 0..vector_fst.num_states() as u32 {
                total_arcs += vector_fst.num_arcs(state_id);
            }
            black_box(total_arcs)
        })
    });

    group.bench_function("const_fst_num_arcs", |b| {
        b.iter(|| {
            let mut total_arcs = 0;
            for state_id in 0..const_fst.num_states() as u32 {
                total_arcs += const_fst.num_arcs(state_id);
            }
            black_box(total_arcs)
        })
    });

    group.finish();
}

fn bench_memory_usage_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage_comparison");

    for &size in &[100, 500, 1000, 2000] {
        group.bench_function(format!("vector_fst_creation_{}", size), |b| {
            b.iter(|| {
                let fst = create_vector_fst(black_box(size));
                black_box(fst)
            })
        });

        group.bench_function(format!("const_fst_conversion_{}", size), |b| {
            let vector_fst = create_vector_fst(size);
            b.iter(|| {
                let const_fst = ConstFst::from_fst(black_box(&vector_fst)).unwrap();
                black_box(const_fst)
            })
        });
    }

    group.finish();
}

fn bench_final_weight_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("final_weight_access");
    let size = 1000;

    let vector_fst = create_vector_fst(size);
    let const_fst = ConstFst::from_fst(&vector_fst).unwrap();

    group.bench_function("vector_fst", |b| {
        b.iter(|| {
            let mut final_count = 0;
            for state_id in 0..vector_fst.num_states() as u32 {
                if vector_fst.final_weight(state_id).is_some() {
                    final_count += 1;
                }
            }
            black_box(final_count)
        })
    });

    group.bench_function("const_fst", |b| {
        b.iter(|| {
            let mut final_count = 0;
            for state_id in 0..const_fst.num_states() as u32 {
                if const_fst.final_weight(state_id).is_some() {
                    final_count += 1;
                }
            }
            black_box(final_count)
        })
    });

    group.finish();
}

fn bench_start_state_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("start_state_access");
    let size = 1000;

    let vector_fst = create_vector_fst(size);
    let const_fst = ConstFst::from_fst(&vector_fst).unwrap();

    group.bench_function("vector_fst", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                let start = vector_fst.start();
                black_box(start);
            }
        })
    });

    group.bench_function("const_fst", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                let start = const_fst.start();
                black_box(start);
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_arc_iteration_comparison,
    bench_state_access_comparison,
    bench_memory_usage_comparison,
    bench_final_weight_access,
    bench_start_state_access
);
criterion_main!(benches);
