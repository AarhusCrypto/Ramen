use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oram::p_ot::{POTIndexParty, POTKeyParty, POTReceiverParty};
use rand::{thread_rng, Rng};
use utils::field::Fp;
use utils::permutation::FisherYatesPermutation;

const LOG_DOMAIN_SIZES: [u32; 4] = [8, 12, 16, 20];

pub fn bench_pot(c: &mut Criterion) {
    let mut group = c.benchmark_group("POT");

    for log_domain_size in LOG_DOMAIN_SIZES {
        group.bench_with_input(
            BenchmarkId::new("init", log_domain_size),
            &log_domain_size,
            |b, &log_domain_size| {
                let mut key_party =
                    POTKeyParty::<Fp, FisherYatesPermutation>::new(1 << log_domain_size);
                let mut index_party =
                    POTIndexParty::<Fp, FisherYatesPermutation>::new(1 << log_domain_size);
                let mut receiver_party = POTReceiverParty::<Fp>::new(1 << log_domain_size);
                b.iter(|| {
                    key_party.reset();
                    index_party.reset();
                    receiver_party.reset();
                    let (msg_to_index_party, msg_to_receiver_party) = key_party.init();
                    index_party.init(msg_to_index_party.0, msg_to_index_party.1);
                    receiver_party.init(msg_to_receiver_party);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("expand", log_domain_size),
            &log_domain_size,
            |b, &log_domain_size| {
                let mut key_party =
                    POTKeyParty::<Fp, FisherYatesPermutation>::new(1 << log_domain_size);
                key_party.init();
                b.iter(|| {
                    black_box(key_party.expand());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("access", log_domain_size),
            &log_domain_size,
            |b, &log_domain_size| {
                let mut key_party =
                    POTKeyParty::<Fp, FisherYatesPermutation>::new(1 << log_domain_size);
                let mut index_party =
                    POTIndexParty::<Fp, FisherYatesPermutation>::new(1 << log_domain_size);
                let mut receiver_party = POTReceiverParty::<Fp>::new(1 << log_domain_size);
                let (msg_to_index_party, msg_to_receiver_party) = key_party.init();
                index_party.init(msg_to_index_party.0, msg_to_index_party.1);
                receiver_party.init(msg_to_receiver_party);
                let index = thread_rng().gen_range(0..1 << log_domain_size);
                b.iter(|| {
                    let msg_to_receiver_party = index_party.access(index);
                    let output =
                        receiver_party.access(msg_to_receiver_party.0, msg_to_receiver_party.1);
                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_pot
);
criterion_main!(benches);
