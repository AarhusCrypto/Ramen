use bincode;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ff::Field;
use rand::thread_rng;
use utils::field::Fp;

const VEC_LENS: [usize; 4] = [1, 4, 16, 64];

pub fn bench_encode(c: &mut Criterion) {
    for vec_len in VEC_LENS {
        c.bench_with_input(
            BenchmarkId::new("Fp::encode", vec_len),
            &vec_len,
            |b, &vec_len| {
                let x: Vec<_> = (0..vec_len).map(|_| Fp::random(thread_rng())).collect();
                b.iter(|| {
                    black_box(
                        bincode::encode_to_vec(&x, bincode::config::standard())
                            .expect("encode failed"),
                    )
                });
            },
        );
    }
}

pub fn bench_decode(c: &mut Criterion) {
    for vec_len in VEC_LENS {
        c.bench_with_input(
            BenchmarkId::new("Fp::decode", vec_len),
            &vec_len,
            |b, &vec_len| {
                let x: Vec<_> = (0..vec_len).map(|_| Fp::random(thread_rng())).collect();
                let bytes =
                    bincode::encode_to_vec(&x, bincode::config::standard()).expect("encode failed");
                b.iter(|| {
                    black_box(
                        bincode::decode_from_slice::<Vec<Fp>, _>(&bytes, bincode::config::standard())
                            .expect("decode failed"),
                    )
                });
            },
        );
    }
}

criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
