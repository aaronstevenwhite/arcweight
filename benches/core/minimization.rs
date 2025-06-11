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

fn create_redundant_fst(n: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let mut states = Vec::new();
    
    for _ in 0..=n {
        states.push(fst.add_state());
    }
    
    fst.set_start(states[0]);
    fst.set_final(states[n], TropicalWeight::one());
    
    for i in 0..n {
        // Add redundant paths
        for _ in 1..=2 {
            fst.add_arc(states[i], Arc::new(
                1, 1,
                TropicalWeight::new(1.0),
                states[i + 1],
            ));
        }
    }
    
    fst
}

pub fn bench_linear_minimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_minimization");
    group.measurement_time(Duration::from_secs(10));
    
    let fst1 = create_linear_fst(100);
    let fst2 = create_linear_fst(500);
    
    group.bench_function("minimize_linear_100", |b| {
        b.iter(|| {
            let mut fst = black_box(&fst1).clone();
            let result: VectorFst<TropicalWeight> = minimize(&mut fst).unwrap();
            black_box(result);
        })
    });
    
    group.bench_function("minimize_linear_500", |b| {
        b.iter(|| {
            let mut fst = black_box(&fst2).clone();
            let result: VectorFst<TropicalWeight> = minimize(&mut fst).unwrap();
            black_box(result);
        })
    });
    
    group.finish();
}

pub fn bench_branching_minimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("branching_minimization");
    group.measurement_time(Duration::from_secs(10));
    
    let fst1 = create_branching_fst(50);
    let fst2 = create_branching_fst(100);
    
    group.bench_function("minimize_branching_50", |b| {
        b.iter(|| {
            let mut fst = black_box(&fst1).clone();
            let result: VectorFst<TropicalWeight> = minimize(&mut fst).unwrap();
            black_box(result);
        })
    });
    
    group.bench_function("minimize_branching_100", |b| {
        b.iter(|| {
            let mut fst = black_box(&fst2).clone();
            let result: VectorFst<TropicalWeight> = minimize(&mut fst).unwrap();
            black_box(result);
        })
    });
    
    group.finish();
}

pub fn bench_redundant_minimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("redundant_minimization");
    group.measurement_time(Duration::from_secs(10));
    
    let fst1 = create_redundant_fst(50);
    let fst2 = create_redundant_fst(100);
    
    group.bench_function("minimize_redundant_50", |b| {
        b.iter(|| {
            let mut fst = black_box(&fst1).clone();
            let result: VectorFst<TropicalWeight> = minimize(&mut fst).unwrap();
            black_box(result);
        })
    });
    
    group.bench_function("minimize_redundant_100", |b| {
        b.iter(|| {
            let mut fst = black_box(&fst2).clone();
            let result: VectorFst<TropicalWeight> = minimize(&mut fst).unwrap();
            black_box(result);
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_linear_minimization,
    bench_branching_minimization,
    bench_redundant_minimization
);
criterion_main!(benches); 