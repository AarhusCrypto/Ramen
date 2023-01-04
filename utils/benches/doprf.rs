use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ff::Field;
use rand::thread_rng;
use utils::doprf::LegendrePrf;
use utils::doprf::{DOPrfParty1, DOPrfParty2, DOPrfParty3};
use utils::field::Fp;

pub fn bench_legendre_prf(c: &mut Criterion) {
    let mut group = c.benchmark_group("LegendrePrf");
    group.bench_function("keygen", |b| {
        b.iter(|| black_box(LegendrePrf::<Fp>::key_gen()))
    });
    group.bench_function("eval", |b| {
        let key = LegendrePrf::<Fp>::key_gen();
        let x = Fp::random(thread_rng());
        b.iter(|| black_box(LegendrePrf::<Fp>::eval(&key, x)))
    });
    group.finish();
}

const LOG_NUM_EVALUATIONS: [usize; 4] = [4, 6, 8, 10];

pub fn bench_doprf(c: &mut Criterion) {
    let mut group = c.benchmark_group("DOPrf");

    let mut party_1 = DOPrfParty1::<Fp>::new();
    let mut party_2 = DOPrfParty2::<Fp>::new();
    let mut party_3 = DOPrfParty3::<Fp>::new();

    group.bench_function("init", |b| {
        b.iter(|| {
            party_1.reset();
            party_2.reset();
            party_3.reset();
            let (msg_1_2, msg_1_3) = party_1.init_round_0();
            let (msg_2_1, msg_2_3) = party_2.init_round_0();
            let (msg_3_1, msg_3_2) = party_3.init_round_0();
            party_1.init_round_1(msg_2_1, msg_3_1);
            party_2.init_round_1(msg_1_2, msg_3_2);
            party_3.init_round_1(msg_1_3, msg_2_3);
        });
    });

    {
        party_1.reset();
        party_2.reset();
        party_3.reset();
        let (msg_1_2, msg_1_3) = party_1.init_round_0();
        let (msg_2_1, msg_2_3) = party_2.init_round_0();
        let (msg_3_1, msg_3_2) = party_3.init_round_0();
        party_1.init_round_1(msg_2_1, msg_3_1);
        party_2.init_round_1(msg_1_2, msg_3_2);
        party_3.init_round_1(msg_1_3, msg_2_3);
    }

    for log_num_evaluations in LOG_NUM_EVALUATIONS {
        group.bench_with_input(
            BenchmarkId::new("preprocess", log_num_evaluations),
            &log_num_evaluations,
            |b, &log_num_evaluations| {
                let num = 1 << log_num_evaluations;
                b.iter(|| {
                    party_1.reset_preprocessing();
                    party_2.reset_preprocessing();
                    party_3.reset_preprocessing();
                    let (msg_1_2, msg_1_3) = party_1.preprocess_round_0(num);
                    let (msg_2_1, msg_2_3) = party_2.preprocess_round_0(num);
                    let (msg_3_1, msg_3_2) = party_3.preprocess_round_0(num);
                    party_1.preprocess_round_1(num, msg_2_1, msg_3_1);
                    party_2.preprocess_round_1(num, msg_1_2, msg_3_2);
                    party_3.preprocess_round_1(num, msg_1_3, msg_2_3);
                });
            },
        );
    }

    for log_num_evaluations in LOG_NUM_EVALUATIONS {
        group.bench_with_input(
            BenchmarkId::new("preprocess+eval", log_num_evaluations),
            &log_num_evaluations,
            |b, &log_num_evaluations| {
                let num = 1 << log_num_evaluations;
                let shares_1: Vec<Fp> = (0..num).map(|_| Fp::random(thread_rng())).collect();
                let shares_2: Vec<Fp> = (0..num).map(|_| Fp::random(thread_rng())).collect();
                let shares_3: Vec<Fp> = (0..num).map(|_| Fp::random(thread_rng())).collect();
                b.iter(|| {
                    let (msg_1_2, msg_1_3) = party_1.preprocess_round_0(num);
                    let (msg_2_1, msg_2_3) = party_2.preprocess_round_0(num);
                    let (msg_3_1, msg_3_2) = party_3.preprocess_round_0(num);
                    party_1.preprocess_round_1(num, msg_2_1, msg_3_1);
                    party_2.preprocess_round_1(num, msg_1_2, msg_3_2);
                    party_3.preprocess_round_1(num, msg_1_3, msg_2_3);

                    let (msg_2_1, msg_2_3) = party_2.eval_round_0(num, &shares_2);
                    let (msg_3_1, _) = party_3.eval_round_0(num, &shares_3);
                    let (_, msg_1_3) = party_1.eval_round_1(num, &shares_1, &msg_2_1, &msg_3_1);
                    let _output = party_3.eval_round_2(num, &shares_3, msg_1_3, msg_2_3);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_legendre_prf, bench_doprf
);
criterion_main!(benches);
