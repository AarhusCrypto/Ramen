use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dpf::spdpf::{DummySpDpf, HalfTreeSpDpf, SinglePointDpf};
use utils::field::{Fp, FromHash};

const LOG_DOMAIN_SIZES: [usize; 4] = [8, 12, 16, 20];

fn bench_spdpf_keygen<SPDPF, F>(c: &mut Criterion, dpf_name: &str, field_name: &str)
where
    SPDPF: SinglePointDpf<Value = F>,
    F: Copy + FromHash,
{
    let mut group = c.benchmark_group(format!("{}-{}-keygen", dpf_name, field_name));
    let alpha = 42;
    let beta = F::hash_bytes(&[0x13, 0x37, 0x42, 0x47]);
    for log_domain_size in LOG_DOMAIN_SIZES.iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(log_domain_size),
            log_domain_size,
            |b, &log_domain_size| {
                b.iter(|| {
                    let (_key_0, _key_1) = SPDPF::generate_keys(1 << log_domain_size, alpha, beta);
                });
            },
        );
    }
    group.finish();
}

fn bench_spdpf_evaluate_domain<SPDPF, F>(c: &mut Criterion, dpf_name: &str, field_name: &str)
where
    SPDPF: SinglePointDpf<Value = F>,
    F: Copy + FromHash,
{
    let mut group = c.benchmark_group(format!("{}-{}-evaluate_domain", dpf_name, field_name));
    let alpha = 42;
    let beta = F::hash_bytes(&[0x13, 0x37, 0x42, 0x47]);
    for log_domain_size in LOG_DOMAIN_SIZES.iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(log_domain_size),
            log_domain_size,
            |b, &log_domain_size| {
                let (key_0, _key_1) = SPDPF::generate_keys(1 << log_domain_size, alpha, beta);
                b.iter(|| {
                    SPDPF::evaluate_domain(&key_0);
                });
            },
        );
    }
    group.finish();
}

fn bench_spdpf<SPDPF, F>(c: &mut Criterion, dpf_name: &str, field_name: &str)
where
    SPDPF: SinglePointDpf<Value = F>,
    F: Copy + FromHash,
{
    bench_spdpf_keygen::<SPDPF, F>(c, dpf_name, field_name);
    bench_spdpf_evaluate_domain::<SPDPF, F>(c, dpf_name, field_name);
}

fn bench_all_spdpf(c: &mut Criterion) {
    bench_spdpf::<DummySpDpf<Fp>, _>(c, "DummySpDpf", "Fp");
    bench_spdpf::<HalfTreeSpDpf<Fp>, _>(c, "HalfTreeSpDpf", "Fp");
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_all_spdpf
);
criterion_main!(benches);
