use arcweight::prelude::*;
use criterion::{criterion_group, criterion_main, Criterion};
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Duration;

fn create_parallel_state_test_fst(n: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let mut states = Vec::new();

    for _ in 0..=n {
        states.push(fst.add_state());
    }

    fst.set_start(states[0]);
    fst.set_final(states[n], TropicalWeight::one());

    // Create a complex FST with many states and transitions
    for i in 0..n {
        // Add multiple arcs from each state
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

        // Add some self-loops
        fst.add_arc(
            states[i],
            Arc::new(4, 4, TropicalWeight::new(1.0), states[i]),
        );
    }

    fst
}

pub fn bench_parallel_state_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_state_processing");
    group.measurement_time(Duration::from_secs(10));

    let small_fst = create_parallel_state_test_fst(100);
    let medium_fst = create_parallel_state_test_fst(500);
    let large_fst = create_parallel_state_test_fst(1000);

    group.bench_function("small_fst_parallel_state_count", |b| {
        b.iter(|| {
            let fst = black_box(&small_fst);
            let states: Vec<_> = fst.states().collect();
            let count = states.par_iter().count();
            black_box(count);
        })
    });

    group.bench_function("medium_fst_parallel_state_count", |b| {
        b.iter(|| {
            let fst = black_box(&medium_fst);
            let states: Vec<_> = fst.states().collect();
            let count = states.par_iter().count();
            black_box(count);
        })
    });

    group.bench_function("large_fst_parallel_state_count", |b| {
        b.iter(|| {
            let fst = black_box(&large_fst);
            let states: Vec<_> = fst.states().collect();
            let count = states.par_iter().count();
            black_box(count);
        })
    });

    group.finish();
}

pub fn bench_parallel_state_arc_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_state_arc_count");
    group.measurement_time(Duration::from_secs(10));

    let small_fst = create_parallel_state_test_fst(100);
    let medium_fst = create_parallel_state_test_fst(500);
    let large_fst = create_parallel_state_test_fst(1000);

    group.bench_function("small_fst_parallel_state_arc_count", |b| {
        b.iter(|| {
            let fst = black_box(&small_fst);
            let states: Vec<_> = fst.states().collect();
            let counts: Vec<usize> = states
                .par_iter()
                .map(|state| fst.num_arcs(*state))
                .collect();
            black_box(counts);
        })
    });

    group.bench_function("medium_fst_parallel_state_arc_count", |b| {
        b.iter(|| {
            let fst = black_box(&medium_fst);
            let states: Vec<_> = fst.states().collect();
            let counts: Vec<usize> = states
                .par_iter()
                .map(|state| fst.num_arcs(*state))
                .collect();
            black_box(counts);
        })
    });

    group.bench_function("large_fst_parallel_state_arc_count", |b| {
        b.iter(|| {
            let fst = black_box(&large_fst);
            let states: Vec<_> = fst.states().collect();
            let counts: Vec<usize> = states
                .par_iter()
                .map(|state| fst.num_arcs(*state))
                .collect();
            black_box(counts);
        })
    });

    group.finish();
}

pub fn bench_parallel_state_weight_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_state_weight_sum");
    group.measurement_time(Duration::from_secs(10));

    let small_fst = create_parallel_state_test_fst(100);
    let medium_fst = create_parallel_state_test_fst(500);
    let large_fst = create_parallel_state_test_fst(1000);

    group.bench_function("small_fst_parallel_state_weight_sum", |b| {
        b.iter(|| {
            let fst = black_box(&small_fst);
            let states: Vec<_> = fst.states().collect();
            let sums: Vec<f32> = states
                .par_iter()
                .map(|state| fst.arcs(*state).map(|arc| *arc.weight.value()).sum())
                .collect();
            black_box(sums);
        })
    });

    group.bench_function("medium_fst_parallel_state_weight_sum", |b| {
        b.iter(|| {
            let fst = black_box(&medium_fst);
            let states: Vec<_> = fst.states().collect();
            let sums: Vec<f32> = states
                .par_iter()
                .map(|state| fst.arcs(*state).map(|arc| *arc.weight.value()).sum())
                .collect();
            black_box(sums);
        })
    });

    group.bench_function("large_fst_parallel_state_weight_sum", |b| {
        b.iter(|| {
            let fst = black_box(&large_fst);
            let states: Vec<_> = fst.states().collect();
            let sums: Vec<f32> = states
                .par_iter()
                .map(|state| fst.arcs(*state).map(|arc| *arc.weight.value()).sum())
                .collect();
            black_box(sums);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_parallel_state_processing,
    bench_parallel_state_arc_count,
    bench_parallel_state_weight_sum
);
criterion_main!(benches);
