use arcweight::prelude::*;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

pub fn create_linear_fst(n: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let mut states = Vec::new();

    for _ in 0..=n {
        states.push(fst.add_state());
    }

    fst.set_start(states[0]);
    fst.set_final(states[n], TropicalWeight::one());

    for i in 0..n {
        fst.add_arc(
            states[i],
            Arc::new(
                (i + 1) as u32,
                (i + 1) as u32,
                TropicalWeight::new(1.0),
                states[i + 1],
            ),
        );
    }

    fst
}

pub fn create_branching_fst(n: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let mut states = Vec::new();

    for _ in 0..=n {
        states.push(fst.add_state());
    }

    fst.set_start(states[0]);
    fst.set_final(states[n], TropicalWeight::one());

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
    }

    fst
}

pub fn create_non_deterministic_fst(n: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let mut states = Vec::new();

    for _ in 0..=n {
        states.push(fst.add_state());
    }

    fst.set_start(states[0]);
    fst.set_final(states[n], TropicalWeight::one());

    for i in 0..n {
        // Add multiple arcs with same input label
        for j in 1..=3 {
            fst.add_arc(
                states[i],
                Arc::new(
                    1, // Same input label
                    j as u32,
                    TropicalWeight::new(j as f32),
                    states[i + 1],
                ),
            );
        }
    }

    fst
}

pub fn bench_linear_determinization(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_determinization");
    group.measurement_time(std::time::Duration::from_secs(10));

    let fst = create_linear_fst(100);
    group.bench_function("100_states", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> = determinize(black_box(&fst)).unwrap();
            black_box(result);
        })
    });

    let fst = create_linear_fst(500);
    group.bench_function("500_states", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> = determinize(black_box(&fst)).unwrap();
            black_box(result);
        })
    });

    group.finish();
}

pub fn bench_branching_determinization(c: &mut Criterion) {
    let mut group = c.benchmark_group("branching_determinization");
    group.measurement_time(std::time::Duration::from_secs(10));

    let fst = create_branching_fst(50);
    group.bench_function("50_states", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> = determinize(black_box(&fst)).unwrap();
            black_box(result);
        })
    });

    let fst = create_branching_fst(100);
    group.bench_function("100_states", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> = determinize(black_box(&fst)).unwrap();
            black_box(result);
        })
    });

    group.finish();
}

pub fn bench_non_deterministic_determinization(c: &mut Criterion) {
    let mut group = c.benchmark_group("non_deterministic_determinization");
    group.measurement_time(std::time::Duration::from_secs(10));

    let fst = create_non_deterministic_fst(50);
    group.bench_function("50_states", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> = determinize(black_box(&fst)).unwrap();
            black_box(result);
        })
    });

    let fst = create_non_deterministic_fst(100);
    group.bench_function("100_states", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> = determinize(black_box(&fst)).unwrap();
            black_box(result);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_linear_determinization,
    bench_branching_determinization,
    bench_non_deterministic_determinization
);
criterion_main!(benches);
