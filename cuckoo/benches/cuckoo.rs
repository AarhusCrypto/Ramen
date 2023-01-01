use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cuckoo::cuckoo::{Hasher, Parameters};
use cuckoo::hash::AesHashFunction;

pub fn bench_hash_domain_into_buckets(c: &mut Criterion) {
    let number_inputs = 1_000;
    let parameters = Parameters::<AesHashFunction<u32>, _>::sample(number_inputs);
    let hasher = Hasher::new(parameters);
    let domain_size = 100_000;
    c.bench_function("Hasher<AesHashFunction>.hash_domain_into_buckets", |b| {
        b.iter(|| hasher.hash_domain_into_buckets(black_box(domain_size)))
    });
}

pub fn bench_cuckoo_hash_items(c: &mut Criterion) {
    let number_inputs = 1_000;
    let parameters = Parameters::<AesHashFunction<u32>, _>::sample(number_inputs);
    let hasher = Hasher::new(parameters);
    let items: Vec<u64> = (0..number_inputs as u64).collect();
    c.bench_function("Hasher<AesHashFunction>.cuckoo_hash_items", |b| {
        b.iter(|| hasher.cuckoo_hash_items(black_box(&items)))
    });
}

criterion_group!(
    benches,
    bench_hash_domain_into_buckets,
    bench_cuckoo_hash_items
);
criterion_main!(benches);
