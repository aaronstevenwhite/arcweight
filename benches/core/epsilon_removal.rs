use criterion::{black_box, criterion_group, criterion_main, Criterion};
use arcweight::prelude::*;
use std::time::Duration;

fn create_linear_fst(n: usize) -> VectorFst<BooleanWeight> {
    let mut fst = VectorFst::new();
    let mut states = Vec::new();
    
    for _ in 0..=n {
        states.push(fst.add_state());
    }
    
    fst.set_start(states[0]);
    fst.set_final(states[n], BooleanWeight::one());
    
    for i in 0..n {
        fst.add_arc(states[i], Arc::new(
            1, 1,
            BooleanWeight::new(true),
            states[i + 1],
        ));
    }
    
    fst
}

fn create_branching_fst(n: usize) -> VectorFst<BooleanWeight> {
    let mut fst = VectorFst::new();
    let mut states = Vec::new();
    
    for _ in 0..=n {
        states.push(fst.add_state());
    }
    
    fst.set_start(states[0]);
    fst.set_final(states[n], BooleanWeight::one());
    
    for i in 0..n {
        for j in 1..=3 {
            fst.add_arc(states[i], Arc::new(
                j as u32,
                j as u32,
                BooleanWeight::new(true),
                states[i + 1],
            ));
        }
    }
    
    fst
}

fn create_epsilon_fst(n: usize) -> VectorFst<BooleanWeight> {
    let mut fst = VectorFst::new();
    let mut states = Vec::new();
    
    for _ in 0..=n {
        states.push(fst.add_state());
    }
    
    fst.set_start(states[0]);
    fst.set_final(states[n], BooleanWeight::one());
    
    for i in 0..n {
        // Add epsilon arcs
        for _ in 1..=2 {
            fst.add_arc(states[i], Arc::new(
                0, 0,
                BooleanWeight::new(true),
                states[i + 1],
            ));
        }
        // Add non-epsilon arcs
        fst.add_arc(states[i], Arc::new(
            1, 1,
            BooleanWeight::new(true),
            states[i + 1],
        ));
    }
    
    fst
}

pub fn bench_linear_epsilon_removal(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_epsilon_removal");
    group.measurement_time(Duration::from_secs(10));
    
    let fst1 = create_linear_fst(100);
    let fst2 = create_linear_fst(500);
    
    group.bench_function("remove_epsilons_linear_100", |b| {
        b.iter(|| {
            let fst = black_box(&fst1);
            let result: VectorFst<BooleanWeight> = remove_epsilons(fst).unwrap();
            black_box(result);
        })
    });
    
    group.bench_function("remove_epsilons_linear_500", |b| {
        b.iter(|| {
            let fst = black_box(&fst2);
            let result: VectorFst<BooleanWeight> = remove_epsilons(fst).unwrap();
            black_box(result);
        })
    });
    
    group.finish();
}

pub fn bench_branching_epsilon_removal(c: &mut Criterion) {
    let mut group = c.benchmark_group("branching_epsilon_removal");
    group.measurement_time(Duration::from_secs(10));
    
    let fst1 = create_branching_fst(50);
    let fst2 = create_branching_fst(100);
    
    group.bench_function("remove_epsilons_branching_50", |b| {
        b.iter(|| {
            let fst = black_box(&fst1);
            let result: VectorFst<BooleanWeight> = remove_epsilons(fst).unwrap();
            black_box(result);
        })
    });
    
    group.bench_function("remove_epsilons_branching_100", |b| {
        b.iter(|| {
            let fst = black_box(&fst2);
            let result: VectorFst<BooleanWeight> = remove_epsilons(fst).unwrap();
            black_box(result);
        })
    });
    
    group.finish();
}

pub fn bench_epsilon_removal(c: &mut Criterion) {
    let mut group = c.benchmark_group("epsilon_removal");
    group.measurement_time(Duration::from_secs(10));
    
    let fst1 = create_epsilon_fst(50);
    let fst2 = create_epsilon_fst(100);
    
    group.bench_function("remove_epsilons_50", |b| {
        b.iter(|| {
            let fst = black_box(&fst1);
            let result: VectorFst<BooleanWeight> = remove_epsilons(fst).unwrap();
            black_box(result);
        })
    });
    
    group.bench_function("remove_epsilons_100", |b| {
        b.iter(|| {
            let fst = black_box(&fst2);
            let result: VectorFst<BooleanWeight> = remove_epsilons(fst).unwrap();
            black_box(result);
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_linear_epsilon_removal,
    bench_branching_epsilon_removal,
    bench_epsilon_removal
);
criterion_main!(benches); 