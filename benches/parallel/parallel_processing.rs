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

pub fn bench_parallel_arc_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_arc_processing");
    group.measurement_time(Duration::from_secs(10));
    
    let fst = create_large_fst(1000);
    
    group.bench_function("sequential_arc_sum", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let mut total = 0.0f32;
            for state in fst.states() {
                for arc in fst.arcs(state) {
                    total += arc.weight.value();
                }
            }
            black_box(total);
        })
    });
    
    group.bench_function("parallel_arc_sum", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let states: Vec<_> = fst.states().collect();
            let total: f32 = states.par_iter()
                .flat_map(|state| fst.arcs(*state).collect::<Vec<_>>())
                .map(|arc| *arc.weight.value())
                .sum();
            black_box(total);
        })
    });
    
    group.finish();
}

pub fn bench_parallel_state_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_state_processing");
    group.measurement_time(Duration::from_secs(10));
    
    let fst = create_large_fst(1000);
    
    group.bench_function("sequential_state_count", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let count = fst.num_states();
            black_box(count);
        })
    });
    
    group.bench_function("parallel_state_analysis", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let states: Vec<_> = fst.states().collect();
            let analysis: Vec<(u32, usize)> = states.par_iter()
                .map(|&state| (state, fst.num_arcs(state)))
                .collect();
            black_box(analysis);
        })
    });
    
    group.finish();
}

pub fn bench_parallel_final_states(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_final_states");
    group.measurement_time(Duration::from_secs(10));
    
    let fst = create_large_fst(1000);
    
    group.bench_function("sequential_final_check", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let mut final_count = 0usize;
            for state in fst.states() {
                if fst.is_final(state) {
                    final_count += 1;
                }
            }
            black_box(final_count);
        })
    });
    
    group.bench_function("parallel_final_check", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let states: Vec<_> = fst.states().collect();
            let final_count = states.par_iter()
                .filter(|&&state| fst.is_final(state))
                .count();
            black_box(final_count);
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_parallel_arc_processing,
    bench_parallel_state_processing,
    bench_parallel_final_states
);
criterion_main!(benches);