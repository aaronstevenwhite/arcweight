use arcweight::prelude::*;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use std::time::Duration;

fn create_epsilon_test_fst(n: usize) -> VectorFst<BooleanWeight> {
    let mut fst = VectorFst::new();
    let mut states = Vec::new();

    for _ in 0..=n {
        states.push(fst.add_state());
    }

    fst.set_start(states[0]);
    fst.set_final(states[n], BooleanWeight::one());

    // Create a complex FST with many epsilon transitions
    for i in 0..n {
        // Add regular arcs
        fst.add_arc(
            states[i],
            Arc::new(1, 1, BooleanWeight::new(true), states[i + 1]),
        );

        // Add epsilon transitions
        if i < n - 1 {
            // Forward epsilon
            fst.add_arc(
                states[i],
                Arc::new(0, 0, BooleanWeight::new(true), states[i + 1]),
            );

            // Self-loop epsilon
            fst.add_arc(
                states[i],
                Arc::new(0, 0, BooleanWeight::new(true), states[i]),
            );

            // Skip-ahead epsilon
            if i < n - 2 {
                fst.add_arc(
                    states[i],
                    Arc::new(0, 0, BooleanWeight::new(true), states[i + 2]),
                );
            }
        }
    }

    fst
}

pub fn bench_epsilon_removal(c: &mut Criterion) {
    let mut group = c.benchmark_group("epsilon_removal");
    group.measurement_time(Duration::from_secs(10));

    let small_fst = create_epsilon_test_fst(100);
    let medium_fst = create_epsilon_test_fst(500);
    let large_fst = create_epsilon_test_fst(1000);

    group.bench_function("small_fst_remove_epsilons", |b| {
        b.iter(|| {
            let fst = black_box(&small_fst);
            let result: VectorFst<BooleanWeight> = remove_epsilons(fst).unwrap();
            black_box(result);
        })
    });

    group.bench_function("medium_fst_remove_epsilons", |b| {
        b.iter(|| {
            let fst = black_box(&medium_fst);
            let result: VectorFst<BooleanWeight> = remove_epsilons(fst).unwrap();
            black_box(result);
        })
    });

    group.bench_function("large_fst_remove_epsilons", |b| {
        b.iter(|| {
            let fst = black_box(&large_fst);
            let result: VectorFst<BooleanWeight> = remove_epsilons(fst).unwrap();
            black_box(result);
        })
    });

    group.finish();
}

pub fn bench_epsilon_removal_with_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("epsilon_removal_with_weights");
    group.measurement_time(Duration::from_secs(10));

    let small_fst = create_epsilon_test_fst(100);
    let medium_fst = create_epsilon_test_fst(500);
    let large_fst = create_epsilon_test_fst(1000);

    group.bench_function("small_fst_remove_epsilons_with_weights", |b| {
        b.iter(|| {
            let fst = black_box(&small_fst);
            let result: VectorFst<BooleanWeight> = remove_epsilons(fst).unwrap();
            black_box(result);
        })
    });

    group.bench_function("medium_fst_remove_epsilons_with_weights", |b| {
        b.iter(|| {
            let fst = black_box(&medium_fst);
            let result: VectorFst<BooleanWeight> = remove_epsilons(fst).unwrap();
            black_box(result);
        })
    });

    group.bench_function("large_fst_remove_epsilons_with_weights", |b| {
        b.iter(|| {
            let fst = black_box(&large_fst);
            let result: VectorFst<BooleanWeight> = remove_epsilons(fst).unwrap();
            black_box(result);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_epsilon_removal,
    bench_epsilon_removal_with_weights
);
criterion_main!(benches);
