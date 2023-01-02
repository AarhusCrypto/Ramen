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

pub fn bench_position_map(c: &mut Criterion) {
    let number_inputs = 1_000;
    let parameters = Parameters::<AesHashFunction<u32>, _>::sample(number_inputs);
    let hasher = Hasher::new(parameters);
    let domain_size = 100_000;

    let hash_table = hasher.hash_domain_into_buckets(domain_size);
    let lookup_table =
        Hasher::<AesHashFunction<u32>, _>::compute_pos_lookup_table(domain_size, &hash_table);
    let lookup_table2 =
        Hasher::<AesHashFunction<u32>, _>::compute_pos_lookup_table(domain_size, &hash_table);

    let pos = |bucket_i: usize, item: u64| -> u64 {
        let idx = hash_table[bucket_i].partition_point(|x| x < &item);
        assert!(idx != hash_table[bucket_i].len());
        assert_eq!(item, hash_table[bucket_i][idx]);
        assert!(idx == 0 || hash_table[bucket_i][idx - 1] != item);
        idx as u64
    };

    let mut group = c.benchmark_group("position_map");
    group.bench_function("normal", |b| {
        b.iter(|| {
            for item in 0..domain_size {
                for &(bucket_i, _) in lookup_table[item as usize].iter() {
                    let idx = pos(bucket_i, item);
                    black_box(idx);
                }
            }
        })
    });
    group.bench_function("precomputed", |b| {
        b.iter(|| {
            for item in 0..domain_size {
                for &(bucket_i, _) in lookup_table[item as usize].iter() {
                    let idx = Hasher::<AesHashFunction<u32>, _>::pos_lookup(
                        &lookup_table2,
                        bucket_i,
                        item,
                    );
                    black_box(idx);
                }
            }
        })
    });
    group.finish();
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
    bench_position_map,
    bench_cuckoo_hash_items
);
criterion_main!(benches);
