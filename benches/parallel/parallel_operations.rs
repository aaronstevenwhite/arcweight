use arcweight::prelude::*;
use criterion::{criterion_group, criterion_main, Criterion};
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Duration;

fn create_parallel_test_fst(n: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let mut states = Vec::new();

    for _ in 0..=n {
        states.push(fst.add_state());
    }

    fst.set_start(states[0]);
    fst.set_final(states[n], TropicalWeight::one());

    // Create a complex FST with many parallel paths
    for i in 0..n {
        // Add multiple parallel arcs
        for j in 1..=5 {
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
    }

    fst
}

pub fn bench_parallel_composition(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_composition");
    group.measurement_time(Duration::from_secs(10));

    let small_fst = create_parallel_test_fst(100);
    let medium_fst = create_parallel_test_fst(500);
    let large_fst = create_parallel_test_fst(1000);

    group.bench_function("small_fst_parallel_compose", |b| {
        b.iter(|| {
            let fst1 = black_box(&small_fst);
            let fst2 = black_box(&small_fst);
            let result: VectorFst<TropicalWeight> = compose_default(fst1, fst2).unwrap();
            black_box(result);
        })
    });

    group.bench_function("medium_fst_parallel_compose", |b| {
        b.iter(|| {
            let fst1 = black_box(&medium_fst);
            let fst2 = black_box(&medium_fst);
            let result: VectorFst<TropicalWeight> = compose_default(fst1, fst2).unwrap();
            black_box(result);
        })
    });

    group.bench_function("large_fst_parallel_compose", |b| {
        b.iter(|| {
            let fst1 = black_box(&large_fst);
            let fst2 = black_box(&large_fst);
            let result: VectorFst<TropicalWeight> = compose_default(fst1, fst2).unwrap();
            black_box(result);
        })
    });

    group.finish();
}

pub fn bench_parallel_shortest_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_shortest_path");
    group.measurement_time(Duration::from_secs(10));

    let small_fst = create_parallel_test_fst(100);
    let medium_fst = create_parallel_test_fst(500);
    let large_fst = create_parallel_test_fst(1000);

    group.bench_function("small_fst_parallel_shortest_path", |b| {
        b.iter(|| {
            let fst = black_box(&small_fst);
            let result: VectorFst<TropicalWeight> = shortest_path_single(fst).unwrap();
            black_box(result);
        })
    });

    group.bench_function("medium_fst_parallel_shortest_path", |b| {
        b.iter(|| {
            let fst = black_box(&medium_fst);
            let result: VectorFst<TropicalWeight> = shortest_path_single(fst).unwrap();
            black_box(result);
        })
    });

    group.bench_function("large_fst_parallel_shortest_path", |b| {
        b.iter(|| {
            let fst = black_box(&large_fst);
            let result: VectorFst<TropicalWeight> = shortest_path_single(fst).unwrap();
            black_box(result);
        })
    });

    group.finish();
}

pub fn bench_parallel_arc_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_arc_processing");
    group.measurement_time(Duration::from_secs(10));

    let small_fst = create_parallel_test_fst(100);
    let medium_fst = create_parallel_test_fst(500);
    let large_fst = create_parallel_test_fst(1000);

    group.bench_function("small_fst_parallel_arc_sum", |b| {
        b.iter(|| {
            let fst = black_box(&small_fst);
            let states: Vec<_> = fst.states().collect();
            let sum: f32 = states
                .par_iter()
                .flat_map(|state| fst.arcs(*state).collect::<Vec<_>>())
                .map(|arc| *arc.weight.value())
                .sum();
            black_box(sum);
        })
    });

    group.bench_function("medium_fst_parallel_arc_sum", |b| {
        b.iter(|| {
            let fst = black_box(&medium_fst);
            let states: Vec<_> = fst.states().collect();
            let sum: f32 = states
                .par_iter()
                .flat_map(|state| fst.arcs(*state).collect::<Vec<_>>())
                .map(|arc| *arc.weight.value())
                .sum();
            black_box(sum);
        })
    });

    group.bench_function("large_fst_parallel_arc_sum", |b| {
        b.iter(|| {
            let fst = black_box(&large_fst);
            let states: Vec<_> = fst.states().collect();
            let sum: f32 = states
                .par_iter()
                .flat_map(|state| fst.arcs(*state).collect::<Vec<_>>())
                .map(|arc| *arc.weight.value())
                .sum();
            black_box(sum);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_parallel_composition,
    bench_parallel_shortest_path,
    bench_parallel_arc_processing
);
criterion_main!(benches);
