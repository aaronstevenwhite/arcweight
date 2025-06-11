use criterion::{black_box, criterion_group, criterion_main, Criterion};
use arcweight::prelude::*;
use std::time::Duration;
use rayon::prelude::*;

fn create_large_fst(n: usize) -> VectorFst<TropicalWeight> {
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

pub fn bench_arc_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("arc_count");
    group.measurement_time(Duration::from_secs(10));
    
    let fst = create_large_fst(1000);
    
    group.bench_function("arc_count_sequential", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let count = fst.num_arcs_total();
            black_box(count);
        })
    });
    
    group.bench_function("arc_count_parallel", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let states: Vec<_> = fst.states().collect();
            let count: usize = states.par_iter()
                .map(|state| fst.num_arcs(*state))
                .sum();
            black_box(count);
        })
    });
    
    group.finish();
}

pub fn bench_arc_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("arc_iteration");
    group.measurement_time(Duration::from_secs(10));
    
    let fst = create_large_fst(1000);
    
    group.bench_function("arc_iter_sequential", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let mut count = 0;
            for state in fst.states() {
                for _arc in fst.arcs(state) {
                    count += 1;
                }
            }
            black_box(count);
        })
    });
    
    group.bench_function("arc_iter_parallel", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let states: Vec<_> = fst.states().collect();
            let count: usize = states.par_iter()
                .map(|state| fst.arcs(*state).count())
                .sum();
            black_box(count);
        })
    });
    
    group.finish();
}

pub fn bench_arc_weight_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("arc_weight_sum");
    group.measurement_time(Duration::from_secs(10));
    
    let fst = create_large_fst(1000);
    
    group.bench_function("weight_sum_sequential", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let mut sum = 0.0f32;
            for state in fst.states() {
                for arc in fst.arcs(state) {
                    sum += arc.weight.value();
                }
            }
            black_box(sum);
        })
    });
    
    group.bench_function("weight_sum_parallel", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let states: Vec<_> = fst.states().collect();
            let sum: f32 = states.par_iter()
                .flat_map(|state| fst.arcs(*state).collect::<Vec<_>>())
                .map(|arc| *arc.weight.value())
                .sum();
            black_box(sum);
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_arc_count,
    bench_arc_iteration,
    bench_arc_weight_sum
);
criterion_main!(benches);