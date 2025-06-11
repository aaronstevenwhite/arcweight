use criterion::{black_box, criterion_group, criterion_main, Criterion};
use arcweight::prelude::*;
use std::io::Cursor;

pub fn create_test_fst(n: usize) -> VectorFst<TropicalWeight> {
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

pub fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");
    group.measurement_time(std::time::Duration::from_secs(10));
    
    let fst = create_test_fst(100);
    group.bench_function("small_fst", |b| {
        b.iter(|| {
            let mut writer = Cursor::new(Vec::new());
            write_text(&fst, &mut writer, None, None).unwrap();
            black_box(writer);
        })
    });
    
    let fst = create_test_fst(1000);
    group.bench_function("medium_fst", |b| {
        b.iter(|| {
            let mut writer = Cursor::new(Vec::new());
            write_text(&fst, &mut writer, None, None).unwrap();
            black_box(writer);
        })
    });
    
    let fst = create_test_fst(10000);
    group.bench_function("large_fst", |b| {
        b.iter(|| {
            let mut writer = Cursor::new(Vec::new());
            write_text(&fst, &mut writer, None, None).unwrap();
            black_box(writer);
        })
    });
    
    group.finish();
}

fn bench_deserialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("deserialization");
    
    let small_fst = create_test_fst(100);
    let medium_fst = create_test_fst(1000);
    let large_fst = create_test_fst(10000);
    
    let mut small_writer = Cursor::new(Vec::new());
    write_text(&small_fst, &mut small_writer, None, None).unwrap();
    let small_data = small_writer.into_inner();
    
    let mut medium_writer = Cursor::new(Vec::new());
    write_text(&medium_fst, &mut medium_writer, None, None).unwrap();
    let medium_data = medium_writer.into_inner();
    
    let mut large_writer = Cursor::new(Vec::new());
    write_text(&large_fst, &mut large_writer, None, None).unwrap();
    let large_data = large_writer.into_inner();
    
    group.bench_function("small_fst_read", |b| {
        b.iter(|| {
            let mut reader = Cursor::new(&small_data);
            let fst: VectorFst<TropicalWeight> = read_text(&mut reader, None, None).unwrap();
            black_box(fst);
        })
    });
    
    group.bench_function("medium_fst_read", |b| {
        b.iter(|| {
            let mut reader = Cursor::new(&medium_data);
            let fst: VectorFst<TropicalWeight> = read_text(&mut reader, None, None).unwrap();
            black_box(fst);
        })
    });
    
    group.bench_function("large_fst_read", |b| {
        b.iter(|| {
            let mut reader = Cursor::new(&large_data);
            let fst: VectorFst<TropicalWeight> = read_text(&mut reader, None, None).unwrap();
            black_box(fst);
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_serialization,
    bench_deserialization
);
criterion_main!(benches); 