use arcweight::prelude::*;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use std::time::Duration;

fn create_large_fst(n: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let mut states = Vec::new();

    for _ in 0..=n {
        states.push(fst.add_state());
    }

    fst.set_start(states[0]);
    fst.set_final(states[n], TropicalWeight::one());

    // Create a dense FST with many arcs
    for i in 0..n {
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

fn bench_arc_storage(c: &mut Criterion) {
    let mut group = c.benchmark_group("arc_storage");
    group.measurement_time(Duration::from_secs(10));

    let fst = create_large_fst(1000);

    group.bench_function("arc_count_5000", |b| {
        b.iter(|| {
            let count = black_box(&fst).num_arcs_total();
            black_box(count);
        })
    });

    group.finish();
}

fn bench_state_management(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_management");

    group.bench_function("state_creation_1000", |b| {
        b.iter(|| {
            let mut fst = VectorFst::<TropicalWeight>::new();
            for _ in 0..1000 {
                black_box(fst.add_state());
            }
        })
    });

    let fst = create_large_fst(1000);

    group.bench_function("state_lookup_1000", |b| {
        b.iter(|| {
            for i in 0..1000 {
                black_box(fst.num_arcs(i as u32));
            }
        })
    });

    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("large_fst_creation_10000", |b| {
        b.iter(|| {
            let fst = create_large_fst(10000);
            black_box(fst);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_arc_storage,
    bench_state_management,
    bench_memory_usage
);
criterion_main!(benches);
