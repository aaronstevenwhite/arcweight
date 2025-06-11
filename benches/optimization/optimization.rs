use arcweight::algorithms::push_weights;
use arcweight::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;

fn create_optimization_test_fst(n: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let mut states = Vec::new();

    for _ in 0..=n {
        states.push(fst.add_state());
    }

    fst.set_start(states[0]);
    fst.set_final(states[n], TropicalWeight::one());

    // Create a complex FST with multiple paths and weights
    for i in 0..n {
        // Add multiple arcs with different weights
        for j in 1..=3 {
            fst.add_arc(
                states[i],
                Arc::new(
                    j as u32,
                    j as u32,
                    TropicalWeight::new(j as f32),
                    states[i + 1],
                ),
            );
        }

        // Add some epsilon transitions
        if i < n - 1 {
            fst.add_arc(
                states[i],
                Arc::new(0, 0, TropicalWeight::new(0.5), states[i + 1]),
            );
        }
    }

    fst
}

pub fn bench_weight_pushing(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_pushing");
    group.measurement_time(Duration::from_secs(10));

    let small_fst = create_optimization_test_fst(100);
    let medium_fst = create_optimization_test_fst(500);
    let large_fst = create_optimization_test_fst(1000);

    group.bench_function("small_fst_push", |b| {
        b.iter(|| {
            let fst = black_box(&small_fst);
            let result: VectorFst<TropicalWeight> = push_weights(fst).unwrap();
            black_box(result);
        })
    });

    group.bench_function("medium_fst_push", |b| {
        b.iter(|| {
            let fst = black_box(&medium_fst);
            let result: VectorFst<TropicalWeight> = push_weights(fst).unwrap();
            black_box(result);
        })
    });

    group.bench_function("large_fst_push", |b| {
        b.iter(|| {
            let fst = black_box(&large_fst);
            let result: VectorFst<TropicalWeight> = push_weights(fst).unwrap();
            black_box(result);
        })
    });

    group.finish();
}

pub fn bench_minimize(c: &mut Criterion) {
    let mut group = c.benchmark_group("minimize");
    group.measurement_time(Duration::from_secs(10));

    let small_fst = create_optimization_test_fst(100);
    let medium_fst = create_optimization_test_fst(500);
    let large_fst = create_optimization_test_fst(1000);

    group.bench_function("small_fst_minimize", |b| {
        b.iter(|| {
            let fst = black_box(&small_fst);
            let result: VectorFst<TropicalWeight> = minimize(fst).unwrap();
            black_box(result);
        })
    });

    group.bench_function("medium_fst_minimize", |b| {
        b.iter(|| {
            let fst = black_box(&medium_fst);
            let result: VectorFst<TropicalWeight> = minimize(fst).unwrap();
            black_box(result);
        })
    });

    group.bench_function("large_fst_minimize", |b| {
        b.iter(|| {
            let fst = black_box(&large_fst);
            let result: VectorFst<TropicalWeight> = minimize(fst).unwrap();
            black_box(result);
        })
    });

    group.finish();
}

pub fn bench_optimize(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimize");
    group.measurement_time(Duration::from_secs(10));

    let small_fst = create_optimization_test_fst(100);
    let medium_fst = create_optimization_test_fst(500);
    let large_fst = create_optimization_test_fst(1000);

    group.bench_function("small_fst_optimize", |b| {
        b.iter(|| {
            let fst = black_box(&small_fst);
            let minimized: VectorFst<TropicalWeight> = minimize(fst).unwrap();
            let result: VectorFst<TropicalWeight> = push_weights(&minimized).unwrap();
            black_box(result);
        })
    });

    group.bench_function("medium_fst_optimize", |b| {
        b.iter(|| {
            let fst = black_box(&medium_fst);
            let minimized: VectorFst<TropicalWeight> = minimize(fst).unwrap();
            let result: VectorFst<TropicalWeight> = push_weights(&minimized).unwrap();
            black_box(result);
        })
    });

    group.bench_function("large_fst_optimize", |b| {
        b.iter(|| {
            let fst = black_box(&large_fst);
            let minimized: VectorFst<TropicalWeight> = minimize(fst).unwrap();
            let result: VectorFst<TropicalWeight> = push_weights(&minimized).unwrap();
            black_box(result);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_weight_pushing,
    bench_minimize,
    bench_optimize
);
criterion_main!(benches);
