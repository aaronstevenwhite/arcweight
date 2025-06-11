use criterion::{black_box, criterion_group, criterion_main, Criterion};
use arcweight::prelude::*;

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
            (i + 1) as u32,
            (i + 1) as u32,
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
        // Add multiple arcs from each state
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

fn bench_fst_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("fst_creation");
    
    group.bench_function("linear_100", |b| {
        b.iter(|| create_linear_fst(black_box(100)))
    });
    
    group.bench_function("linear_1000", |b| {
        b.iter(|| create_linear_fst(black_box(1000)))
    });
    
    group.bench_function("branching_100", |b| {
        b.iter(|| create_branching_fst(black_box(100)))
    });
    
    group.finish();
}

fn bench_shortest_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("shortest_path");
    
    let linear_fst = create_linear_fst(1000);
    let branching_fst = create_branching_fst(100);
    
    group.bench_function("linear_1000", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> = 
                shortest_path_single(black_box(&linear_fst)).unwrap();
            black_box(result);
        })
    });
    
    group.bench_function("branching_100", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> = 
                shortest_path_single(black_box(&branching_fst)).unwrap();
            black_box(result);
        })
    });
    
    group.finish();
}

fn bench_compose(c: &mut Criterion) {
    let mut group = c.benchmark_group("compose");
    
    let fst1 = create_linear_fst(100);
    let fst2 = create_linear_fst(100);
    let fst3 = create_branching_fst(50);
    
    group.bench_function("linear_100x100", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> = 
                compose_default(black_box(&fst1), black_box(&fst2)).unwrap();
            black_box(result);
        })
    });
    
    group.bench_function("linear_branching_100x50", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> = 
                compose_default(black_box(&fst1), black_box(&fst3)).unwrap();
            black_box(result);
        })
    });
    
    group.finish();
}

fn bench_determinize(c: &mut Criterion) {
    let mut group = c.benchmark_group("determinize");
    
    let linear_fst = create_linear_fst(100);
    let branching_fst = create_branching_fst(50);
    
    group.bench_function("linear_100", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> = 
                determinize(black_box(&linear_fst)).unwrap();
            black_box(result);
        })
    });
    
    group.bench_function("branching_50", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> = 
                determinize(black_box(&branching_fst)).unwrap();
            black_box(result);
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_fst_creation,
    bench_shortest_path,
    bench_compose,
    bench_determinize
);
criterion_main!(benches); 