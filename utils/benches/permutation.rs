use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use utils::permutation::{FisherYatesPermutation, Permutation};

const LOG_DOMAIN_SIZES: [u32; 4] = [8, 12, 16, 20];

pub fn bench_permutation(c: &mut Criterion) {
    for log_domain_size in LOG_DOMAIN_SIZES {
        c.bench_with_input(
            BenchmarkId::new("FisherYates", log_domain_size),
            &log_domain_size,
            |b, &log_domain_size| {
                let key = FisherYatesPermutation::sample(1 << log_domain_size);
                b.iter(|| black_box(FisherYatesPermutation::from_key(key)))
            },
        );
    }
}

criterion_group!(benches, bench_permutation);
criterion_main!(benches);
