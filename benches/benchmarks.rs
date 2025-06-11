//! Performance benchmarks

use arcweight::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

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

fn bench_shortest_path(c: &mut Criterion) {
    let fst = create_linear_fst(1000);

    c.bench_function("shortest_path_1000", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> = shortest_path_single(black_box(&fst)).unwrap();
            black_box(result);
        })
    });
}

fn bench_compose(c: &mut Criterion) {
    let fst1 = create_linear_fst(100);
    let fst2 = create_linear_fst(100);

    c.bench_function("compose_100x100", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> =
                compose_default(black_box(&fst1), black_box(&fst2)).unwrap();
            black_box(result);
        })
    });
}

fn bench_determinize(c: &mut Criterion) {
    let mut fst = VectorFst::<TropicalWeight>::new();
    let s0 = fst.add_state();
    let s1 = fst.add_state();
    let s2 = fst.add_state();

    fst.set_start(s0);
    fst.set_final(s2, TropicalWeight::one());

    // create non-deterministic paths
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(1.0), s1));
    fst.add_arc(s0, Arc::new(1, 1, TropicalWeight::new(2.0), s2));
    fst.add_arc(s1, Arc::new(2, 2, TropicalWeight::new(1.0), s2));

    c.bench_function("determinize_small", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> = determinize(black_box(&fst)).unwrap();
            black_box(result);
        })
    });
}

criterion_group!(
    benches,
    bench_shortest_path,
    bench_compose,
    bench_determinize
);
criterion_main!(benches);
