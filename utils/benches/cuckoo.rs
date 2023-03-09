use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use utils::cuckoo::{Hasher, Parameters};
use utils::hash::AesHashFunction;

const LOG_DOMAIN_SIZES: [u32; 4] = [8, 12, 16, 20];

pub fn bench_hash_domain_into_buckets(c: &mut Criterion) {
    for log_domain_size in LOG_DOMAIN_SIZES {
        let log_number_inputs = log_domain_size / 2;
        let parameters = Parameters::<AesHashFunction<u32>, _>::sample(1 << log_number_inputs);
        let hasher = Hasher::new(parameters);
        c.bench_with_input(
            BenchmarkId::new(
                "Hasher<AesHashFunction>.hash_domain_into_buckets",
                log_domain_size,
            ),
            &log_domain_size,
            |b, &log_domain_size| {
                b.iter(|| hasher.hash_domain_into_buckets(black_box(1 << log_domain_size)))
            },
        );
    }
}

pub fn bench_position_map(c: &mut Criterion) {
    let number_inputs = 1_000;
    let parameters = Parameters::<AesHashFunction<u32>, _>::sample(number_inputs);
    let hasher = Hasher::new(parameters);
    let domain_size = 100_000;

    let hash_table = hasher.hash_domain_into_buckets(domain_size);
    // (ab)use one lookup table to obtain the input pairs for pos
    let values =
        Hasher::<AesHashFunction<u32>, _>::compute_pos_lookup_table(domain_size, &hash_table);

    let pos = |bucket_i: usize, item: u64| -> u64 {
        let idx = hash_table[bucket_i].partition_point(|x| x < &item);
        assert!(idx != hash_table[bucket_i].len());
        assert_eq!(item, hash_table[bucket_i][idx]);
        assert!(idx == 0 || hash_table[bucket_i][idx - 1] != item);
        idx as u64
    };

    let mut group = c.benchmark_group("position_map");
    group.bench_function("search", |b| {
        b.iter(|| {
            for item in 0..domain_size {
                for &(bucket_i, _) in values[item as usize].iter() {
                    let idx = pos(bucket_i, item);
                    black_box(idx);
                }
            }
        })
    });
    group.bench_function("lookup", |b| {
        let pos_lookup_table =
            Hasher::<AesHashFunction<u32>, _>::compute_pos_lookup_table(domain_size, &hash_table);
        b.iter(|| {
            for item in 0..domain_size {
                for &(bucket_i, _) in values[item as usize].iter() {
                    let idx = Hasher::<AesHashFunction<u32>, _>::pos_lookup(
                        &pos_lookup_table,
                        bucket_i,
                        item,
                    );
                    black_box(idx);
                }
            }
        })
    });
    group.bench_function("precomputation+lookup", |b| {
        b.iter(|| {
            let pos_lookup_table = Hasher::<AesHashFunction<u32>, _>::compute_pos_lookup_table(
                domain_size,
                &hash_table,
            );
            for item in 0..domain_size {
                for &(bucket_i, _) in values[item as usize].iter() {
                    let idx = Hasher::<AesHashFunction<u32>, _>::pos_lookup(
                        &pos_lookup_table,
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
    for log_domain_size in LOG_DOMAIN_SIZES {
        let log_number_inputs = log_domain_size / 2;
        let parameters = Parameters::<AesHashFunction<u32>, _>::sample(1 << log_number_inputs);
        let hasher = Hasher::new(parameters);
        let items: Vec<u64> = (0..1 << log_number_inputs).collect();
        c.bench_with_input(
            BenchmarkId::new("Hasher<AesHashFunction>.cuckoo_hash_items", log_domain_size),
            &log_domain_size,
            |b, _| b.iter(|| hasher.cuckoo_hash_items(black_box(&items))),
        );
    }
}

criterion_group!(
    benches,
    bench_hash_domain_into_buckets,
    bench_position_map,
    bench_cuckoo_hash_items
);
criterion_main!(benches);
