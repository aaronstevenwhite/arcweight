use arcweight::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;

fn create_minimization_test_fst(n: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let mut states = Vec::new();

    for _ in 0..=n {
        states.push(fst.add_state());
    }

    fst.set_start(states[0]);
    fst.set_final(states[n], TropicalWeight::one());

    // Create a complex FST with redundant states
    for i in 0..n {
        // Add multiple arcs with same weights to create equivalent states
        for j in 1..=3 {
            fst.add_arc(
                states[i],
                Arc::new(j as u32, j as u32, TropicalWeight::new(1.0), states[i + 1]),
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

pub fn bench_minimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("minimization");
    group.measurement_time(Duration::from_secs(10));

    let small_fst = create_minimization_test_fst(100);
    let medium_fst = create_minimization_test_fst(500);
    let large_fst = create_minimization_test_fst(1000);

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

pub fn bench_minimization_with_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("minimization_with_weights");
    group.measurement_time(Duration::from_secs(10));

    let small_fst = create_minimization_test_fst(100);
    let medium_fst = create_minimization_test_fst(500);
    let large_fst = create_minimization_test_fst(1000);

    group.bench_function("small_fst_minimize_with_weights", |b| {
        b.iter(|| {
            let fst = black_box(&small_fst);
            let result: VectorFst<TropicalWeight> = minimize(fst).unwrap();
            black_box(result);
        })
    });

    group.bench_function("medium_fst_minimize_with_weights", |b| {
        b.iter(|| {
            let fst = black_box(&medium_fst);
            let result: VectorFst<TropicalWeight> = minimize(fst).unwrap();
            black_box(result);
        })
    });

    group.bench_function("large_fst_minimize_with_weights", |b| {
        b.iter(|| {
            let fst = black_box(&large_fst);
            let result: VectorFst<TropicalWeight> = minimize(fst).unwrap();
            black_box(result);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_minimization, bench_minimization_with_weights);
criterion_main!(benches);
