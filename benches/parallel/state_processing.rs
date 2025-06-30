use arcweight::prelude::*;
use criterion::{criterion_group, criterion_main, Criterion};
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Duration;

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

pub fn bench_state_counting(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_counting");
    group.measurement_time(Duration::from_secs(10));

    let fst = create_large_fst(1000);

    group.bench_function("sequential_state_count", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let count = fst.num_states();
            black_box(count);
        })
    });

    group.bench_function("parallel_state_count", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let states: Vec<_> = fst.states().collect();
            let count = states.len();
            black_box(count);
        })
    });

    group.finish();
}

pub fn bench_state_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_analysis");
    group.measurement_time(Duration::from_secs(10));

    let fst = create_large_fst(1000);

    group.bench_function("sequential_state_analysis", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let mut analysis = Vec::new();
            for state in fst.states() {
                let arc_count = fst.num_arcs(state);
                let is_final = fst.is_final(state);
                analysis.push((state, arc_count, is_final));
            }
            black_box(analysis);
        })
    });

    group.bench_function("parallel_state_analysis", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let states: Vec<_> = fst.states().collect();
            let analysis: Vec<_> = states
                .par_iter()
                .map(|&state| {
                    let arc_count = fst.num_arcs(state);
                    let is_final = fst.is_final(state);
                    (state, arc_count, is_final)
                })
                .collect();
            black_box(analysis);
        })
    });

    group.finish();
}

pub fn bench_final_state_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("final_state_detection");
    group.measurement_time(Duration::from_secs(10));

    let fst = create_large_fst(1000);

    group.bench_function("sequential_final_detection", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let mut final_states = Vec::new();
            for state in fst.states() {
                if fst.is_final(state) {
                    final_states.push(state);
                }
            }
            black_box(final_states);
        })
    });

    group.bench_function("parallel_final_detection", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let states: Vec<_> = fst.states().collect();
            let final_states: Vec<_> = states
                .par_iter()
                .filter(|&&state| fst.is_final(state))
                .copied()
                .collect();
            black_box(final_states);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_state_counting,
    bench_state_analysis,
    bench_final_state_detection
);
criterion_main!(benches);
