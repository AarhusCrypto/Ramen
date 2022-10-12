use core::fmt::Debug;
use core::marker::PhantomData;
use core::ops::Add;
use num::traits::Zero;

pub trait SinglePointDpfKey {
    fn get_party_id(&self) -> usize;
    fn get_log_domain_size(&self) -> u64;
}

pub trait SinglePointDpf {
    type Key: Copy + SinglePointDpfKey;
    type Value: Add<Output = Self::Value> + Copy + Debug + Eq + Zero;

    fn generate_keys(log_domain_size: u64, alpha: u64, beta: Self::Value)
        -> (Self::Key, Self::Key);
    fn evaluate_at(key: &Self::Key, index: u64) -> Self::Value;
    fn evaluate_domain(key: &Self::Key) -> Vec<Self::Value> {
        (0..(1 << key.get_log_domain_size()))
            .map(|x| Self::evaluate_at(&key, x))
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DummySpDpfKey<V: Copy> {
    party_id: usize,
    log_domain_size: u64,
    alpha: u64,
    beta: V,
}

impl<V> SinglePointDpfKey for DummySpDpfKey<V>
where
    V: Copy,
{
    fn get_party_id(&self) -> usize {
        self.party_id
    }
    fn get_log_domain_size(&self) -> u64 {
        self.log_domain_size
    }
}

pub struct DummySpDpf<V>
where
    V: Add<Output = V> + Copy + Debug + Eq + Zero,
{
    phantom: PhantomData<V>,
}

impl<V> SinglePointDpf for DummySpDpf<V>
where
    V: Add<Output = V> + Copy + Debug + Eq + Zero,
{
    type Key = DummySpDpfKey<V>;
    type Value = V;

    fn generate_keys(log_domain_size: u64, alpha: u64, beta: V) -> (Self::Key, Self::Key) {
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

    fn evaluate_at(key: &Self::Key, index: u64) -> V {
        if key.get_party_id() == 0 && index == key.alpha {
            key.beta
        } else {
            V::zero()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::{Distribution, Standard};
    use rand::{thread_rng, Rng};

    fn test_spdpf_with_param<SPDPF: SinglePointDpf>(log_domain_size: u64)
    where
        Standard: Distribution<SPDPF::Value>,
    {
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
                assert_eq!(value, SPDPF::Value::zero());
            }
        }
    }

    #[test]
    fn test_spdpf() {
        for log_domain_size in 5..10 {
            test_spdpf_with_param::<DummySpDpf<u64>>(log_domain_size);
        }
    }
}
