pub trait SinglePointDpfKey {
    fn get_party_id(&self) -> usize;
    fn get_log_domain_size(&self) -> u64;
}

pub trait SinglePointDpf {
    type Key: Copy + SinglePointDpfKey;

    fn generate_keys(log_domain_size: u64, alpha: u64, beta: u64) -> (Self::Key, Self::Key);
    fn evaluate_at(key: &Self::Key, index: u64) -> u64;
    fn evaluate_domain(key: &Self::Key) -> Vec<u64> {
        (0..(1 << key.get_log_domain_size()))
            .map(|x| Self::evaluate_at(&key, x))
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DummySpDpfKey {
    party_id: usize,
    log_domain_size: u64,
    alpha: u64,
    beta: u64,
}

impl SinglePointDpfKey for DummySpDpfKey {
    fn get_party_id(&self) -> usize {
        self.party_id
    }
    fn get_log_domain_size(&self) -> u64 {
        self.log_domain_size
    }
}

pub struct DummySpDpf {}

impl SinglePointDpf for DummySpDpf {
    type Key = DummySpDpfKey;

    fn generate_keys(log_domain_size: u64, alpha: u64, beta: u64) -> (Self::Key, Self::Key) {
        assert!(alpha < (1 << log_domain_size));
        (
            DummySpDpfKey {
                party_id: 0,
                log_domain_size,
                alpha,
                beta,
            },
            DummySpDpfKey {
                party_id: 1,
                log_domain_size,
                alpha,
                beta,
            },
        )
    }

    fn evaluate_at(key: &Self::Key, index: u64) -> u64 {
        if key.get_party_id() == 0 && index == key.alpha {
            key.beta
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};

    fn test_spdpf_with_param<SPDPF: SinglePointDpf>(log_domain_size: u64) {
        let domain_size = 1 << log_domain_size;
        let alpha = thread_rng().gen_range(0..domain_size);
        let beta = thread_rng().gen();
        let (key_0, key_1) = SPDPF::generate_keys(log_domain_size, alpha, beta);

        let out_0 = SPDPF::evaluate_domain(&key_0);
        let out_1 = SPDPF::evaluate_domain(&key_1);
        for i in 0..domain_size {
            let value = SPDPF::evaluate_at(&key_0, i) + SPDPF::evaluate_at(&key_1, i);
            assert_eq!(value, out_0[i as usize] + out_1[i as usize]);
            if i == alpha {
                assert_eq!(value, beta);
            } else {
                assert_eq!(value, 0);
            }
        }
    }

    #[test]
    fn test_spdpf() {
        for log_domain_size in 5..10 {
            test_spdpf_with_param::<DummySpDpf>(log_domain_size);
        }
    }
}
