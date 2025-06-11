use criterion::{black_box, criterion_group, criterion_main, Criterion};
use arcweight::prelude::*;
use arcweight::algorithms::push_weights;
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
        for j in 1..=10 {
            fst.add_arc(states[i], Arc::new(
                j as u32,
                j as u32,
                TropicalWeight::new(j as f32),
                states[i + 1],
            ));
        }
    }
    
    fst
}

pub fn bench_parallel_compose(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_compose");
    group.measurement_time(Duration::from_secs(10));
    
    let fst1 = create_parallel_test_fst(100);
    let fst2 = create_parallel_test_fst(100);
    
    group.bench_function("parallel_compose_100x100", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> = 
                compose_default(black_box(&fst1), black_box(&fst2)).unwrap();
            black_box(result);
        })
    });
    
    group.finish();
}

pub fn bench_parallel_optimize(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_optimize");
    group.measurement_time(Duration::from_secs(10));
    
    let fst = create_parallel_test_fst(1000);
    
    group.bench_function("parallel_optimize_1000", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let minimized: VectorFst<TropicalWeight> = minimize(fst).unwrap();
            let result: VectorFst<TropicalWeight> = push_weights(&minimized).unwrap();
            black_box(result);
        })
    });
    
    group.finish();
}

pub fn bench_parallel_weight_pushing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_weight_pushing");
    group.measurement_time(Duration::from_secs(10));
    
    let fst = create_parallel_test_fst(1000);
    
    group.bench_function("parallel_push_weights_1000", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let result: VectorFst<TropicalWeight> = push_weights(fst).unwrap();
            black_box(result);
        })
    });
    
    group.finish();
}

pub fn bench_parallel_minimize(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_minimize");
    group.measurement_time(Duration::from_secs(10));
    
    let fst = create_parallel_test_fst(1000);
    
    group.bench_function("parallel_minimize_1000", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let result: VectorFst<TropicalWeight> = minimize(fst).unwrap();
            black_box(result);
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_parallel_compose,
    bench_parallel_optimize,
    bench_parallel_weight_pushing,
    bench_parallel_minimize
);
criterion_main!(benches); 