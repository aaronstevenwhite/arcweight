use criterion::{black_box, criterion_group, criterion_main, Criterion};
use arcweight::prelude::*;

pub fn create_dense_fst(n: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let mut states = Vec::new();
    
    for _ in 0..=n {
        states.push(fst.add_state());
    }
    
    fst.set_start(states[0]);
    fst.set_final(states[n], TropicalWeight::one());
    
    for i in 0..n {
        // Add multiple arcs between each pair of states
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

pub fn bench_arc_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("arc_count");
    group.measurement_time(std::time::Duration::from_secs(10));
    
    let fst = create_dense_fst(1000);
    group.bench_function("1000_states", |b| {
        b.iter(|| {
            let mut count = 0;
            for state in 0..fst.num_states() {
                count += fst.arcs(state as u32).count();
            }
            black_box(count);
        })
    });
    
    let fst = create_dense_fst(5000);
    group.bench_function("5000_states", |b| {
        b.iter(|| {
            let mut count = 0;
            for state in 0..fst.num_states() {
                count += fst.arcs(state as u32).count();
            }
            black_box(count);
        })
    });
    
    group.finish();
}

pub fn bench_arc_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("arc_iteration");
    group.measurement_time(std::time::Duration::from_secs(10));
    
    let fst = create_dense_fst(1000);
    group.bench_function("1000_states", |b| {
        b.iter(|| {
            let mut sum = TropicalWeight::zero();
            for state in 0..fst.num_states() {
                for arc in fst.arcs(state as u32) {
                    sum = sum.plus(&arc.weight);
                }
            }
            black_box(sum);
        })
    });
    
    let fst = create_dense_fst(5000);
    group.bench_function("5000_states", |b| {
        b.iter(|| {
            let mut sum = TropicalWeight::zero();
            for state in 0..fst.num_states() {
                for arc in fst.arcs(state as u32) {
                    sum = sum.plus(&arc.weight);
                }
            }
            black_box(sum);
        })
    });
    
    group.finish();
}

pub fn bench_arc_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("arc_lookup");
    group.measurement_time(std::time::Duration::from_secs(10));
    
    let fst = create_dense_fst(1000);
    group.bench_function("1000_states", |b| {
        b.iter(|| {
            let mut sum = TropicalWeight::zero();
            for state in 0..fst.num_states() {
                for arc in fst.arcs(state as u32) {
                    sum = sum.plus(&arc.weight);
                }
            }
            black_box(sum);
        })
    });
    
    let fst = create_dense_fst(5000);
    group.bench_function("5000_states", |b| {
        b.iter(|| {
            let mut sum = TropicalWeight::zero();
            for state in 0..fst.num_states() {
                for arc in fst.arcs(state as u32) {
                    sum = sum.plus(&arc.weight);
                }
            }
            black_box(sum);
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_arc_count,
    bench_arc_iteration,
    bench_arc_lookup
);
criterion_main!(benches); 