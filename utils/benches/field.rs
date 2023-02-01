use bincode;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ff::Field;
use rand::thread_rng;
use utils::field::Fp;

pub fn bench_encode(c: &mut Criterion) {
    c.bench_function("Fp::encode", |b| {
        let x = Fp::random(thread_rng());
        b.iter(|| {
            black_box(
                bincode::encode_to_vec(x, bincode::config::standard()).expect("encode failed"),
            )
        });
    });
}

pub fn bench_decode(c: &mut Criterion) {
    c.bench_function("Fp::decode", |b| {
        let x = Fp::random(thread_rng());
        let bytes = bincode::encode_to_vec(x, bincode::config::standard()).expect("encode failed");
        b.iter(|| {
            black_box(
                bincode::decode_from_slice::<Fp, _>(&bytes, bincode::config::standard())
                    .expect("decode failed"),
            )
        });
    });
}

criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
