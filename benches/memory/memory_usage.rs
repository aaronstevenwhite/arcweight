use criterion::{black_box, criterion_group, criterion_main, Criterion};
use arcweight::prelude::*;

pub fn create_large_fst(n: usize) -> VectorFst<TropicalWeight> {
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

pub fn bench_large_fst_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_fst_creation");
    group.measurement_time(std::time::Duration::from_secs(10));
    
    group.bench_function("10000_states", |b| {
        b.iter(|| {
            let fst = create_large_fst(10000);
            black_box(fst);
        })
    });
    
    group.bench_function("50000_states", |b| {
        b.iter(|| {
            let fst = create_large_fst(50000);
            black_box(fst);
        })
    });
    
    group.finish();
}

pub fn bench_fst_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("fst_clone");
    group.measurement_time(std::time::Duration::from_secs(10));
    
    let fst = create_large_fst(10000);
    
    group.bench_function("clone_10000_states", |b| {
        b.iter(|| {
            let cloned = black_box(&fst).clone();
            black_box(cloned);
        })
    });
    
    let fst = create_large_fst(50000);
    
    group.bench_function("clone_50000_states", |b| {
        b.iter(|| {
            let cloned = black_box(&fst).clone();
            black_box(cloned);
        })
    });
    
    group.finish();
}

pub fn bench_fst_clear(c: &mut Criterion) {
    let mut group = c.benchmark_group("fst_clear");
    group.measurement_time(std::time::Duration::from_secs(10));
    
    let mut fst = create_large_fst(10000);
    
    group.bench_function("clear_10000_states", |b| {
        b.iter(|| {
            fst.clear();
            black_box(&fst);
        })
    });
    
    let mut fst = create_large_fst(50000);
    
    group.bench_function("clear_50000_states", |b| {
        b.iter(|| {
            fst.clear();
            black_box(&fst);
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_large_fst_creation,
    bench_fst_clone,
    bench_fst_clear
);
criterion_main!(benches); 