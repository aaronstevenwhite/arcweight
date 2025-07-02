//! Basic semiring operation benchmarks
//!
//! These benchmarks measure the performance of fundamental semiring operations
//! (plus, times, zero, one) across different weight types.

use arcweight::prelude::*;
use arcweight::semiring::{LogWeight, ProbabilityWeight, ProductWeight, RealWeight, StringWeight};
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn bench_tropical_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tropical_operations");

    let w1 = TropicalWeight::new(1.5);
    let w2 = TropicalWeight::new(2.3);
    let w3 = TropicalWeight::new(0.7);

    group.bench_function("plus", |b| {
        b.iter(|| {
            let result = black_box(w1).plus(black_box(&w2));
            black_box(result)
        })
    });

    group.bench_function("times", |b| {
        b.iter(|| {
            let result = black_box(w1).times(black_box(&w2));
            black_box(result)
        })
    });

    group.bench_function("chain_operations", |b| {
        b.iter(|| {
            let result = black_box(w1)
                .times(black_box(&w2))
                .plus(black_box(&w3))
                .times(black_box(&w1));
            black_box(result)
        })
    });

    group.bench_function("power", |b| {
        b.iter(|| {
            let mut result = TropicalWeight::one();
            for _ in 0..10 {
                result = result.times(black_box(&w1));
            }
            black_box(result)
        })
    });

    group.finish();
}

fn bench_log_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("log_operations");

    let w1 = LogWeight::new(1.5);
    let w2 = LogWeight::new(2.3);
    let w3 = LogWeight::new(0.7);

    group.bench_function("plus", |b| {
        b.iter(|| {
            let result = black_box(w1).plus(black_box(&w2));
            black_box(result)
        })
    });

    group.bench_function("times", |b| {
        b.iter(|| {
            let result = black_box(w1).times(black_box(&w2));
            black_box(result)
        })
    });

    group.bench_function("chain_operations", |b| {
        b.iter(|| {
            let result = black_box(w1)
                .times(black_box(&w2))
                .plus(black_box(&w3))
                .times(black_box(&w1));
            black_box(result)
        })
    });

    group.bench_function("power", |b| {
        b.iter(|| {
            let mut result = LogWeight::one();
            for _ in 0..10 {
                result = result.times(black_box(&w1));
            }
            black_box(result)
        })
    });

    group.finish();
}

fn bench_probability_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("probability_operations");

    let w1 = ProbabilityWeight::new(0.6);
    let w2 = ProbabilityWeight::new(0.3);
    let w3 = ProbabilityWeight::new(0.8);

    group.bench_function("plus", |b| {
        b.iter(|| {
            let result = black_box(w1).plus(black_box(&w2));
            black_box(result)
        })
    });

    group.bench_function("times", |b| {
        b.iter(|| {
            let result = black_box(w1).times(black_box(&w2));
            black_box(result)
        })
    });

    group.bench_function("chain_operations", |b| {
        b.iter(|| {
            let result = black_box(w1)
                .times(black_box(&w2))
                .plus(black_box(&w3))
                .times(black_box(&w1));
            black_box(result)
        })
    });

    group.bench_function("power", |b| {
        b.iter(|| {
            let mut result = ProbabilityWeight::one();
            for _ in 0..10 {
                result = result.times(black_box(&w1));
            }
            black_box(result)
        })
    });

    group.finish();
}

fn bench_real_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_operations");

    let w1 = RealWeight::new(1.5);
    let w2 = RealWeight::new(2.3);
    let w3 = RealWeight::new(-0.7);

    group.bench_function("plus", |b| {
        b.iter(|| {
            let result = black_box(w1).plus(black_box(&w2));
            black_box(result)
        })
    });

    group.bench_function("times", |b| {
        b.iter(|| {
            let result = black_box(w1).times(black_box(&w2));
            black_box(result)
        })
    });

    group.bench_function("chain_operations", |b| {
        b.iter(|| {
            let result = black_box(w1)
                .times(black_box(&w2))
                .plus(black_box(&w3))
                .times(black_box(&w1));
            black_box(result)
        })
    });

    group.bench_function("power", |b| {
        b.iter(|| {
            let mut result = RealWeight::one();
            for _ in 0..10 {
                result = result.times(black_box(&w1));
            }
            black_box(result)
        })
    });

    group.finish();
}

fn bench_boolean_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("boolean_operations");

    let w1 = BooleanWeight::new(true);
    let w2 = BooleanWeight::new(false);

    group.bench_function("plus", |b| {
        b.iter(|| {
            let result = black_box(w1).plus(black_box(&w2));
            black_box(result)
        })
    });

    group.bench_function("times", |b| {
        b.iter(|| {
            let result = black_box(w1).times(black_box(&w2));
            black_box(result)
        })
    });

    group.bench_function("chain_operations", |b| {
        b.iter(|| {
            let result = black_box(w1)
                .times(black_box(&w2))
                .plus(black_box(&w1))
                .times(black_box(&w2));
            black_box(result)
        })
    });

    group.finish();
}

fn bench_string_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_operations");

    let w1 = StringWeight::new("hello".as_bytes().to_vec());
    let w2 = StringWeight::new("world".as_bytes().to_vec());
    let w3 = StringWeight::new("!".as_bytes().to_vec());

    group.bench_function("plus", |b| {
        b.iter(|| {
            let result = black_box(w1.clone()).plus(black_box(&w2));
            black_box(result)
        })
    });

    group.bench_function("times", |b| {
        b.iter(|| {
            let result = black_box(w1.clone()).times(black_box(&w2));
            black_box(result)
        })
    });

    group.bench_function("chain_operations", |b| {
        b.iter(|| {
            let result = black_box(w1.clone())
                .times(black_box(&w2))
                .times(black_box(&w3));
            black_box(result)
        })
    });

    group.bench_function("long_concatenation", |b| {
        let short_str = StringWeight::new("ab".as_bytes().to_vec());
        b.iter(|| {
            let mut result = StringWeight::one();
            for _ in 0..20 {
                result = result.times(black_box(&short_str));
            }
            black_box(result)
        })
    });

    group.finish();
}

fn bench_product_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("product_operations");

    let w1 = ProductWeight::new(TropicalWeight::new(1.5), LogWeight::new(0.7));
    let w2 = ProductWeight::new(TropicalWeight::new(2.3), LogWeight::new(1.2));
    let w3 = ProductWeight::new(TropicalWeight::new(0.8), LogWeight::new(0.4));

    group.bench_function("plus", |b| {
        b.iter(|| {
            let result = black_box(w1.clone()).plus(black_box(&w2));
            black_box(result)
        })
    });

    group.bench_function("times", |b| {
        b.iter(|| {
            let result = black_box(w1.clone()).times(black_box(&w2));
            black_box(result)
        })
    });

    group.bench_function("chain_operations", |b| {
        b.iter(|| {
            let result = black_box(w1.clone())
                .times(black_box(&w2))
                .plus(black_box(&w3))
                .times(black_box(&w1));
            black_box(result)
        })
    });

    group.finish();
}

fn bench_semiring_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("semiring_comparison");

    // Compare the same mathematical operation across semirings
    let tropical = TropicalWeight::new(1.5);
    let log = LogWeight::new(1.5);
    let prob = ProbabilityWeight::new(0.6);
    let real = RealWeight::new(1.5);
    let boolean = BooleanWeight::new(true);

    group.bench_function("tropical_plus", |b| {
        b.iter(|| black_box(tropical).plus(black_box(&TropicalWeight::new(2.0))))
    });

    group.bench_function("log_plus", |b| {
        b.iter(|| black_box(log).plus(black_box(&LogWeight::new(2.0))))
    });

    group.bench_function("probability_plus", |b| {
        b.iter(|| black_box(prob).plus(black_box(&ProbabilityWeight::new(0.3))))
    });

    group.bench_function("real_plus", |b| {
        b.iter(|| black_box(real).plus(black_box(&RealWeight::new(2.0))))
    });

    group.bench_function("boolean_plus", |b| {
        b.iter(|| black_box(boolean).plus(black_box(&BooleanWeight::new(false))))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_tropical_operations,
    bench_log_operations,
    bench_probability_operations,
    bench_real_operations,
    bench_boolean_operations,
    bench_string_operations,
    bench_product_operations,
    bench_semiring_comparison
);
criterion_main!(benches);
