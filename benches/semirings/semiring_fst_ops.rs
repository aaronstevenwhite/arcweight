//! FST operations with different semirings benchmarks
//!
//! These benchmarks measure the performance of FST operations (composition, shortest path, etc.)
//! when using different semiring types to understand the overhead of different weight types.

use arcweight::algorithms::{compose_default, shortest_path, ShortestPathConfig};
use arcweight::prelude::*;
use arcweight::semiring::{LogWeight, ProbabilityWeight, RealWeight, StringWeight};
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

/// Create a linear FST with the specified semiring type
fn create_linear_fst_tropical(size: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();

    if size == 0 {
        return fst;
    }

    let states: Vec<_> = (0..size).map(|_| fst.add_state()).collect();
    fst.set_start(states[0]);
    fst.set_final(states[size - 1], TropicalWeight::one());

    for i in 0..size - 1 {
        fst.add_arc(
            states[i],
            Arc::new(
                (i % 26 + 97) as u32,
                (i % 26 + 97) as u32,
                TropicalWeight::new((i % 5 + 1) as f32),
                states[i + 1],
            ),
        );
    }

    fst
}

fn create_linear_fst_log(size: usize) -> VectorFst<LogWeight> {
    let mut fst = VectorFst::new();

    if size == 0 {
        return fst;
    }

    let states: Vec<_> = (0..size).map(|_| fst.add_state()).collect();
    fst.set_start(states[0]);
    fst.set_final(states[size - 1], LogWeight::one());

    for i in 0..size - 1 {
        fst.add_arc(
            states[i],
            Arc::new(
                (i % 26 + 97) as u32,
                (i % 26 + 97) as u32,
                LogWeight::new((i % 5 + 1) as f64),
                states[i + 1],
            ),
        );
    }

    fst
}

fn create_linear_fst_probability(size: usize) -> VectorFst<ProbabilityWeight> {
    let mut fst = VectorFst::new();

    if size == 0 {
        return fst;
    }

    let states: Vec<_> = (0..size).map(|_| fst.add_state()).collect();
    fst.set_start(states[0]);
    fst.set_final(states[size - 1], ProbabilityWeight::one());

    for i in 0..size - 1 {
        fst.add_arc(
            states[i],
            Arc::new(
                (i % 26 + 97) as u32,
                (i % 26 + 97) as u32,
                ProbabilityWeight::new(0.5 + (i % 5) as f64 * 0.1),
                states[i + 1],
            ),
        );
    }

    fst
}

fn create_linear_fst_real(size: usize) -> VectorFst<RealWeight> {
    let mut fst = VectorFst::new();

    if size == 0 {
        return fst;
    }

    let states: Vec<_> = (0..size).map(|_| fst.add_state()).collect();
    fst.set_start(states[0]);
    fst.set_final(states[size - 1], RealWeight::one());

    for i in 0..size - 1 {
        fst.add_arc(
            states[i],
            Arc::new(
                (i % 26 + 97) as u32,
                (i % 26 + 97) as u32,
                RealWeight::new((i % 5 + 1) as f64),
                states[i + 1],
            ),
        );
    }

    fst
}

fn create_linear_fst_boolean(size: usize) -> VectorFst<BooleanWeight> {
    let mut fst = VectorFst::new();

    if size == 0 {
        return fst;
    }

    let states: Vec<_> = (0..size).map(|_| fst.add_state()).collect();
    fst.set_start(states[0]);
    fst.set_final(states[size - 1], BooleanWeight::one());

    for i in 0..size - 1 {
        fst.add_arc(
            states[i],
            Arc::new(
                (i % 26 + 97) as u32,
                (i % 26 + 97) as u32,
                BooleanWeight::new(i % 2 == 0),
                states[i + 1],
            ),
        );
    }

    fst
}

fn create_linear_fst_string(size: usize) -> VectorFst<StringWeight> {
    let mut fst = VectorFst::new();

    if size == 0 {
        return fst;
    }

    let states: Vec<_> = (0..size).map(|_| fst.add_state()).collect();
    fst.set_start(states[0]);
    fst.set_final(states[size - 1], StringWeight::one());

    for i in 0..size - 1 {
        let weight_str = format!("w{}", i % 10);
        fst.add_arc(
            states[i],
            Arc::new(
                (i % 26 + 97) as u32,
                (i % 26 + 97) as u32,
                StringWeight::new(weight_str.as_bytes().to_vec()),
                states[i + 1],
            ),
        );
    }

    fst
}

fn bench_shortest_path_by_semiring(c: &mut Criterion) {
    let mut group = c.benchmark_group("shortest_path_by_semiring");
    let size = 100;

    let tropical_fst = create_linear_fst_tropical(size);
    let _log_fst = create_linear_fst_log(size);
    let _prob_fst = create_linear_fst_probability(size);
    let real_fst = create_linear_fst_real(size);
    let bool_fst = create_linear_fst_boolean(size);
    let _string_fst = create_linear_fst_string(size);

    group.bench_function("tropical", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> =
                shortest_path(black_box(&tropical_fst), ShortestPathConfig::default()).unwrap();
            black_box(result)
        })
    });

    // Only test semirings that support shortest path (NaturallyOrderedSemiring)
    group.bench_function("real", |b| {
        b.iter(|| {
            let result: VectorFst<RealWeight> =
                shortest_path(black_box(&real_fst), ShortestPathConfig::default()).unwrap();
            black_box(result)
        })
    });

    group.bench_function("boolean", |b| {
        b.iter(|| {
            let result: VectorFst<BooleanWeight> =
                shortest_path(black_box(&bool_fst), ShortestPathConfig::default()).unwrap();
            black_box(result)
        })
    });

    group.finish();
}

fn bench_composition_by_semiring(c: &mut Criterion) {
    let mut group = c.benchmark_group("composition_by_semiring");
    let size = 50;

    let tropical_fst1 = create_linear_fst_tropical(size);
    let tropical_fst2 = create_linear_fst_tropical(size);

    let log_fst1 = create_linear_fst_log(size);
    let log_fst2 = create_linear_fst_log(size);

    let prob_fst1 = create_linear_fst_probability(size);
    let prob_fst2 = create_linear_fst_probability(size);

    let bool_fst1 = create_linear_fst_boolean(size);
    let bool_fst2 = create_linear_fst_boolean(size);

    group.bench_function("tropical", |b| {
        b.iter(|| {
            let result: VectorFst<TropicalWeight> =
                compose_default(black_box(&tropical_fst1), black_box(&tropical_fst2)).unwrap();
            black_box(result)
        })
    });

    group.bench_function("log", |b| {
        b.iter(|| {
            let result: VectorFst<LogWeight> =
                compose_default(black_box(&log_fst1), black_box(&log_fst2)).unwrap();
            black_box(result)
        })
    });

    group.bench_function("probability", |b| {
        b.iter(|| {
            let result: VectorFst<ProbabilityWeight> =
                compose_default(black_box(&prob_fst1), black_box(&prob_fst2)).unwrap();
            black_box(result)
        })
    });

    group.bench_function("boolean", |b| {
        b.iter(|| {
            let result: VectorFst<BooleanWeight> =
                compose_default(black_box(&bool_fst1), black_box(&bool_fst2)).unwrap();
            black_box(result)
        })
    });

    group.finish();
}

fn bench_fst_creation_by_semiring(c: &mut Criterion) {
    let mut group = c.benchmark_group("fst_creation_by_semiring");
    let size = 1000;

    group.bench_function("tropical", |b| {
        b.iter(|| {
            let fst = create_linear_fst_tropical(black_box(size));
            black_box(fst)
        })
    });

    group.bench_function("log", |b| {
        b.iter(|| {
            let fst = create_linear_fst_log(black_box(size));
            black_box(fst)
        })
    });

    group.bench_function("probability", |b| {
        b.iter(|| {
            let fst = create_linear_fst_probability(black_box(size));
            black_box(fst)
        })
    });

    group.bench_function("real", |b| {
        b.iter(|| {
            let fst = create_linear_fst_real(black_box(size));
            black_box(fst)
        })
    });

    group.bench_function("boolean", |b| {
        b.iter(|| {
            let fst = create_linear_fst_boolean(black_box(size));
            black_box(fst)
        })
    });

    group.bench_function("string", |b| {
        b.iter(|| {
            let fst = create_linear_fst_string(black_box(size));
            black_box(fst)
        })
    });

    group.finish();
}

fn bench_weight_intensive_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_intensive_operations");

    // Test operations that perform many weight computations
    let size = 20;
    let iterations = 100;

    group.bench_function("tropical_weight_accumulation", |b| {
        let fst = create_linear_fst_tropical(size);
        b.iter(|| {
            let mut total = TropicalWeight::zero();
            for _ in 0..iterations {
                if let Some(start) = fst.start() {
                    for arc in fst.arcs(start) {
                        total = total.plus(&arc.weight);
                    }
                }
            }
            black_box(total)
        })
    });

    group.bench_function("log_weight_accumulation", |b| {
        let fst = create_linear_fst_log(size);
        b.iter(|| {
            let mut total = LogWeight::zero();
            for _ in 0..iterations {
                if let Some(start) = fst.start() {
                    for arc in fst.arcs(start) {
                        total = total.plus(&arc.weight);
                    }
                }
            }
            black_box(total)
        })
    });

    group.bench_function("string_weight_accumulation", |b| {
        let fst = create_linear_fst_string(size);
        b.iter(|| {
            let mut total = StringWeight::zero();
            for _ in 0..iterations {
                if let Some(start) = fst.start() {
                    for arc in fst.arcs(start) {
                        total = total.plus(&arc.weight);
                    }
                }
            }
            black_box(total)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_shortest_path_by_semiring,
    bench_composition_by_semiring,
    bench_fst_creation_by_semiring,
    bench_weight_intensive_operations
);
criterion_main!(benches);
