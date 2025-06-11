use arcweight::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rayon::prelude::*;
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
        for j in 1..=5 {
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

pub fn bench_weight_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_calculation");
    group.measurement_time(Duration::from_secs(10));

    let fst = create_large_fst(1000);

    group.bench_function("sequential_weight_sum", |b| {
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

    group.bench_function("parallel_weight_sum", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let states: Vec<_> = fst.states().collect();
            let total: f32 = states
                .par_iter()
                .flat_map(|state| fst.arcs(*state).collect::<Vec<_>>())
                .map(|arc| *arc.weight.value())
                .sum();
            black_box(total);
        })
    });

    group.finish();
}

pub fn bench_weight_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_analysis");
    group.measurement_time(Duration::from_secs(10));

    let fst = create_large_fst(1000);

    group.bench_function("sequential_weight_stats", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let mut weights = Vec::new();
            for state in fst.states() {
                for arc in fst.arcs(state) {
                    weights.push(*arc.weight.value());
                }
            }
            let min = weights.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f32 = weights.iter().sum();
            black_box((min, max, sum));
        })
    });

    group.bench_function("parallel_weight_stats", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let states: Vec<_> = fst.states().collect();
            let weights: Vec<f32> = states
                .par_iter()
                .flat_map(|state| fst.arcs(*state).collect::<Vec<_>>())
                .map(|arc| *arc.weight.value())
                .collect();
            let min = weights
                .par_iter()
                .cloned()
                .reduce(|| f32::INFINITY, f32::min);
            let max = weights
                .par_iter()
                .cloned()
                .reduce(|| f32::NEG_INFINITY, f32::max);
            let sum: f32 = weights.par_iter().sum();
            black_box((min, max, sum));
        })
    });

    group.finish();
}

pub fn bench_final_weight_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("final_weight_processing");
    group.measurement_time(Duration::from_secs(10));

    let fst = create_large_fst(1000);

    group.bench_function("sequential_final_weights", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let mut final_weights = Vec::new();
            for state in fst.states() {
                if let Some(weight) = fst.final_weight(state) {
                    final_weights.push(*weight.value());
                }
            }
            black_box(final_weights);
        })
    });

    group.bench_function("parallel_final_weights", |b| {
        b.iter(|| {
            let fst = black_box(&fst);
            let states: Vec<_> = fst.states().collect();
            let final_weights: Vec<f32> = states
                .par_iter()
                .filter_map(|&state| fst.final_weight(state).map(|w| *w.value()))
                .collect();
            black_box(final_weights);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_weight_calculation,
    bench_weight_analysis,
    bench_final_weight_processing
);
criterion_main!(benches);
