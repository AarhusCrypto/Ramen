use criterion::{black_box, criterion_group, criterion_main, Criterion};
use utils::hash::{AesHashFunction, HashFunction};

pub fn bench_hash_range(c: &mut Criterion) {
    let n = 100_000;
    let range_size = 10_000;
    let hash_function = AesHashFunction::<u32>::sample(range_size);
    c.bench_function("AesHashFunction.hash_range", |b| {
        b.iter(|| hash_function.hash_range(black_box(0..n)))
    });
}

criterion_group!(benches, bench_hash_range);
criterion_main!(benches);
