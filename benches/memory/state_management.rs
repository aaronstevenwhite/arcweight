use arcweight::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn create_test_fst(n: usize) -> VectorFst<TropicalWeight> {
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

pub fn bench_state_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_creation");
    group.measurement_time(std::time::Duration::from_secs(10));

    group.bench_function("1000_states", |b| {
        b.iter(|| {
            let mut fst: VectorFst<TropicalWeight> = VectorFst::new();
            for _ in 0..1000 {
                black_box(fst.add_state());
            }
        })
    });

    group.bench_function("10000_states", |b| {
        b.iter(|| {
            let mut fst: VectorFst<TropicalWeight> = VectorFst::new();
            for _ in 0..10000 {
                black_box(fst.add_state());
            }
        })
    });

    group.finish();
}

pub fn bench_state_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_lookup");
    group.measurement_time(std::time::Duration::from_secs(10));

    let fst = create_test_fst(1000);
    group.bench_function("1000_states", |b| {
        b.iter(|| {
            for state in 0..fst.num_states() {
                black_box(fst.arcs(state as u32));
            }
        })
    });

    let fst = create_test_fst(10000);
    group.bench_function("10000_states", |b| {
        b.iter(|| {
            for state in 0..fst.num_states() {
                black_box(fst.arcs(state as u32));
            }
        })
    });

    group.finish();
}

pub fn bench_state_modification(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_modification");
    group.measurement_time(std::time::Duration::from_secs(10));

    let mut fst = create_test_fst(1000);
    group.bench_function("1000_states", |b| {
        b.iter(|| {
            for state in 0..fst.num_states() {
                fst.add_arc(
                    state as u32,
                    Arc::new(1, 1, TropicalWeight::new(1.0), (state + 1) as u32),
                );
            }
        })
    });

    let mut fst = create_test_fst(10000);
    group.bench_function("10000_states", |b| {
        b.iter(|| {
            for state in 0..fst.num_states() {
                fst.add_arc(
                    state as u32,
                    Arc::new(1, 1, TropicalWeight::new(1.0), (state + 1) as u32),
                );
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_state_creation,
    bench_state_lookup,
    bench_state_modification
);
criterion_main!(benches);
