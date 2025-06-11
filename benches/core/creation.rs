use criterion::{black_box, criterion_group, criterion_main, Criterion};
use arcweight::prelude::*;
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
        fst.add_arc(states[i], Arc::new(
            1, 1,
            TropicalWeight::new(1.0),
            states[i + 1],
        ));
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

pub fn bench_linear_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_creation");
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function("create_linear_100", |b| {
        b.iter(|| create_linear_fst(black_box(100)))
    });
    
    group.bench_function("create_linear_1000", |b| {
        b.iter(|| create_linear_fst(black_box(1000)))
    });
    
    group.bench_function("create_linear_10000", |b| {
        b.iter(|| create_linear_fst(black_box(10000)))
    });
    
    group.finish();
}

pub fn bench_branching_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("branching_creation");
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function("create_branching_100", |b| {
        b.iter(|| create_branching_fst(black_box(100)))
    });
    
    group.bench_function("create_branching_1000", |b| {
        b.iter(|| create_branching_fst(black_box(1000)))
    });
    
    group.bench_function("create_branching_10000", |b| {
        b.iter(|| create_branching_fst(black_box(10000)))
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_linear_creation,
    bench_branching_creation
);
criterion_main!(benches); 