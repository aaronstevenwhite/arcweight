#[cfg(feature = "serde")]
use arcweight::prelude::*;
use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(feature = "serde")]
use std::hint::black_box;
#[cfg(feature = "serde")]
use std::io::Cursor;

#[cfg(feature = "serde")]
fn create_test_fst(n: usize) -> VectorFst<TropicalWeight> {
    let mut fst = VectorFst::new();
    let mut states = Vec::new();

    for _ in 0..=n {
        states.push(fst.add_state());
    }

    fst.set_start(states[0]);
    fst.set_final(states[n], TropicalWeight::one());

    for i in 0..n {
        fst.add_arc(
            states[i],
            Arc::new(
                (i + 1) as u32,
                (i + 1) as u32,
                TropicalWeight::new(1.0),
                states[i + 1],
            ),
        );
    }

    fst
}

#[cfg(feature = "serde")]
pub fn bench_deserialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("deserialization");

    let small_fst = create_test_fst(100);
    let medium_fst = create_test_fst(1000);
    let large_fst = create_test_fst(10000);

    let mut small_writer = Cursor::new(Vec::new());
    write_binary(&small_fst, &mut small_writer).unwrap();
    let small_data = small_writer.into_inner();

    let mut medium_writer = Cursor::new(Vec::new());
    write_binary(&medium_fst, &mut medium_writer).unwrap();
    let medium_data = medium_writer.into_inner();

    let mut large_writer = Cursor::new(Vec::new());
    write_binary(&large_fst, &mut large_writer).unwrap();
    let large_data = large_writer.into_inner();

    group.bench_function("small_fst_read", |b| {
        b.iter(|| {
            let mut reader = Cursor::new(&small_data);
            let fst: VectorFst<TropicalWeight> = read_binary(&mut reader).unwrap();
            black_box(fst);
        })
    });

    group.bench_function("medium_fst_read", |b| {
        b.iter(|| {
            let mut reader = Cursor::new(&medium_data);
            let fst: VectorFst<TropicalWeight> = read_binary(&mut reader).unwrap();
            black_box(fst);
        })
    });

    group.bench_function("large_fst_read", |b| {
        b.iter(|| {
            let mut reader = Cursor::new(&large_data);
            let fst: VectorFst<TropicalWeight> = read_binary(&mut reader).unwrap();
            black_box(fst);
        })
    });

    group.finish();
}

#[cfg(feature = "serde")]
criterion_group!(benches, bench_deserialization);

#[cfg(not(feature = "serde"))]
fn bench_dummy(_c: &mut Criterion) {}

#[cfg(not(feature = "serde"))]
criterion_group!(benches, bench_dummy);

criterion_main!(benches);
