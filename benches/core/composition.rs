use arcweight::prelude::*;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use std::time::Duration;

fn create_linear_fst(n: usize) -> VectorFst<TropicalWeight> {
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
            Arc::new(1, 1, TropicalWeight::new(1.0), states[i + 1]),
        );
    }

    fst
}

fn create_branching_fst(n: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let mut states = Vec::new();

    for _ in 0..=n {
        states.push(fst.add_state());
    }

    fst.set_start(states[0]);
    fst.set_final(states[n], TropicalWeight::one());

    for i in 0..n {
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

pub fn bench_linear_composition(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_composition");
    group.measurement_time(Duration::from_secs(10));

    let fst1 = create_linear_fst(100);
    let fst2 = create_linear_fst(100);
    let fst3 = create_linear_fst(500);
    let fst4 = create_linear_fst(500);

    group.bench_function("compose_linear_100", |b| {
        b.iter(|| {
            let fst1 = black_box(&fst1).clone();
            let fst2 = black_box(&fst2).clone();
            compose_default::<_, _, _, VectorFst<TropicalWeight>>(&fst1, &fst2).unwrap();
        })
    });

    group.bench_function("compose_linear_500", |b| {
        b.iter(|| {
            let fst3 = black_box(&fst3).clone();
            let fst4 = black_box(&fst4).clone();
            compose_default::<_, _, _, VectorFst<TropicalWeight>>(&fst3, &fst4).unwrap();
        })
    });

    group.finish();
}

pub fn bench_mixed_composition(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_composition");
    group.measurement_time(Duration::from_secs(10));

    let linear_fst1 = create_linear_fst(100);
    let branching_fst1 = create_branching_fst(100);
    let linear_fst2 = create_linear_fst(500);
    let branching_fst2 = create_branching_fst(500);

    group.bench_function("compose_mixed_100", |b| {
        b.iter(|| {
            let linear_fst = black_box(&linear_fst1).clone();
            let branching_fst = black_box(&branching_fst1).clone();
            compose_default::<_, _, _, VectorFst<TropicalWeight>>(&linear_fst, &branching_fst)
                .unwrap();
        })
    });

    group.bench_function("compose_mixed_500", |b| {
        b.iter(|| {
            let linear_fst = black_box(&linear_fst2).clone();
            let branching_fst = black_box(&branching_fst2).clone();
            compose_default::<_, _, _, VectorFst<TropicalWeight>>(&linear_fst, &branching_fst)
                .unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, bench_linear_composition, bench_mixed_composition);
criterion_main!(benches);
