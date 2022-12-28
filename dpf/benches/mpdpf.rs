use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use cuckoo::hash::AesHashFunction;
use dpf::mpdpf::{DummyMpDpf, MultiPointDpf, SmartMpDpf};
use dpf::spdpf::{DummySpDpf, HalfTreeSpDpf};
use rand::{thread_rng, Rng};
use utils::field::{Fp, FromHash};

const LOG_DOMAIN_SIZES: [u32; 4] = [8, 12, 16, 20];

fn setup_points<F: FromHash>(log_domain_size: u32) -> (Vec<u64>, Vec<F>) {
    assert_eq!(log_domain_size % 2, 0);
    let domain_size = 1 << log_domain_size;
    let number_points = 1 << (log_domain_size / 2);
    let alphas = {
        let mut alphas = Vec::<u64>::with_capacity(number_points);
        while alphas.len() < number_points {
            let x = thread_rng().gen_range(0..domain_size);
            match alphas.as_slice().binary_search(&x) {
                Ok(_) => continue,
                Err(i) => alphas.insert(i, x),
            }
        }
        alphas
    };
    let betas: Vec<F> = (0..number_points)
        .map(|x| F::hash_bytes(&x.to_be_bytes()))
        .collect();
    (alphas, betas)
}

fn bench_mpdpf_keygen<MPDPF, F>(c: &mut Criterion, dpf_name: &str, field_name: &str)
where
    MPDPF: MultiPointDpf<Value = F>,
    F: Copy + FromHash,
{
    let mut group = c.benchmark_group(format!("{}-{}-keygen", dpf_name, field_name));
    for log_domain_size in LOG_DOMAIN_SIZES.iter() {
        let (alphas, betas) = setup_points(*log_domain_size);
        group.bench_with_input(
            BenchmarkId::new("without-precomputation", log_domain_size),
            log_domain_size,
            |b, &log_domain_size| {
                let mpdpf = MPDPF::new(1 << log_domain_size, 1 << (log_domain_size / 2));
                b.iter(|| {
                    let (_key_0, _key_1) = mpdpf.generate_keys(&alphas, &betas);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("with-precomputation", log_domain_size),
            log_domain_size,
            |b, &log_domain_size| {
                let mut mpdpf = MPDPF::new(1 << log_domain_size, 1 << (log_domain_size / 2));
                mpdpf.precompute();
                b.iter(|| {
                    let (_key_0, _key_1) = mpdpf.generate_keys(&alphas, &betas);
                });
            },
        );
    }
    group.finish();
}

fn bench_mpdpf_evaluate_domain<MPDPF, F>(c: &mut Criterion, dpf_name: &str, field_name: &str)
where
    MPDPF: MultiPointDpf<Value = F>,
    F: Copy + FromHash,
{
    let mut group = c.benchmark_group(format!("{}-{}-evaluate_domain", dpf_name, field_name));
    for log_domain_size in LOG_DOMAIN_SIZES.iter() {
        let (alphas, betas) = setup_points(*log_domain_size);
        group.bench_with_input(
            BenchmarkId::new("without-precomputation", log_domain_size),
            log_domain_size,
            |b, &log_domain_size| {
                let mpdpf = MPDPF::new(1 << log_domain_size, 1 << (log_domain_size / 2));
                let (key_0, _key_1) = mpdpf.generate_keys(&alphas, &betas);
                b.iter(|| {
                    mpdpf.evaluate_domain(&key_0);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("with-precomputation", log_domain_size),
            log_domain_size,
            |b, &log_domain_size| {
                let mut mpdpf = MPDPF::new(1 << log_domain_size, 1 << (log_domain_size / 2));
                mpdpf.precompute();
                let (key_0, _key_1) = mpdpf.generate_keys(&alphas, &betas);
                b.iter(|| {
                    mpdpf.evaluate_domain(&key_0);
                });
            },
        );
    }
    group.finish();
}

fn bench_mpdpf<MPDPF, F>(c: &mut Criterion, dpf_name: &str, field_name: &str)
where
    MPDPF: MultiPointDpf<Value = F>,
    F: Copy + FromHash,
{
    bench_mpdpf_keygen::<MPDPF, F>(c, dpf_name, field_name);
    bench_mpdpf_evaluate_domain::<MPDPF, F>(c, dpf_name, field_name);
}

fn bench_all_mpdpf(c: &mut Criterion) {
    bench_mpdpf::<DummyMpDpf<Fp>, _>(c, "DummyMpDpf", "Fp");
    bench_mpdpf::<SmartMpDpf<Fp, DummySpDpf<Fp>, AesHashFunction<u16>>, _>(
        c,
        "SmartMpDpf<Dummy,Aes>",
        "Fp",
    );
    bench_mpdpf::<SmartMpDpf<Fp, HalfTreeSpDpf<Fp>, AesHashFunction<u16>>, _>(
        c,
        "SmartMpDpf<HalfTree,Aes>",
        "Fp",
    );
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_all_mpdpf
);
criterion_main!(benches);
