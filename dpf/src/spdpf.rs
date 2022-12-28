use core::fmt::Debug;
use core::marker::PhantomData;
use core::ops::{Add, Sub};
use num::traits::Zero;
use rand::{thread_rng, Rng};

use utils::bit_decompose::bit_decompose;
use utils::fixed_key_aes::FixedKeyAes;
use utils::pseudorandom_conversion::{PRConvertTo, PRConverter};

pub trait SinglePointDpfKey: Clone + Debug {
    fn get_party_id(&self) -> usize;
    fn get_log_domain_size(&self) -> u64;
}

pub trait SinglePointDpf {
    type Key: SinglePointDpfKey;
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
pub struct DummySpDpfKey<V: Copy + Debug> {
    party_id: usize,
    log_domain_size: u64,
    alpha: u64,
    beta: V,
}

impl<V> SinglePointDpfKey for DummySpDpfKey<V>
where
    V: Copy + Debug,
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

/// Implementation of the Half-Tree DPF scheme from Guo et al. (ePrint 2022/1431, Figure 8)
#[derive(Clone, Debug)]
pub struct HalfTreeSpDpfKey<V: Copy + Debug> {
    /// party id b
    party_id: usize,
    /// n where domain size is N := 2^n
    log_domain_size: u64,
    /// (s_b^0 || t_b^0) and t_b^0 is the LSB
    party_seed: u128,
    /// vector of length n: CW_1, ..., CW_(n-1)
    correction_words: Vec<u128>,
    /// high part of CW_n = (HCW, [LCW[0], LCW[1]])
    hcw: u128,
    /// low parts of CW_n = (HCW, [LCW[0], LCW[1]])
    lcw: [bool; 2],
    /// CW_(n+1)
    correction_word_np1: V,
}

impl<V> SinglePointDpfKey for HalfTreeSpDpfKey<V>
where
    V: Copy + Debug,
{
    fn get_party_id(&self) -> usize {
        self.party_id
    }
    fn get_log_domain_size(&self) -> u64 {
        self.log_domain_size
    }
}

pub struct HalfTreeSpDpf<V>
where
    V: Add<Output = V> + Copy + Debug + Eq + Zero,
{
    phantom: PhantomData<V>,
}

impl<V> HalfTreeSpDpf<V>
where
    V: Add<Output = V> + Sub<Output = V> + Copy + Debug + Eq + Zero,
{
    const FIXED_KEY_AES_KEY: [u8; 16] =
        0xdead_beef_1337_4247_dead_beef_1337_4247_u128.to_le_bytes();
    const HASH_KEY: u128 = 0xc000ffee_c0ffffee_c0ffeeee_c00ffeee_u128;
}

impl<V> SinglePointDpf for HalfTreeSpDpf<V>
where
    V: Add<Output = V> + Sub<Output = V> + Copy + Debug + Eq + Zero,
    PRConverter: PRConvertTo<V>,
{
    type Key = HalfTreeSpDpfKey<V>;
    type Value = V;

    fn generate_keys(log_domain_size: u64, alpha: u64, beta: V) -> (Self::Key, Self::Key) {
        assert!(alpha < (1 << log_domain_size));

        let mut rng = thread_rng();

        if log_domain_size == 0 {
            // simply secret-share beta
            let beta_0: V = PRConverter::convert(rng.gen::<u128>());
            let beta_1: V = beta - beta_0;
            return (
                HalfTreeSpDpfKey {
                    party_id: 0,
                    log_domain_size,
                    party_seed: Default::default(),
                    correction_words: Default::default(),
                    hcw: Default::default(),
                    lcw: Default::default(),
                    correction_word_np1: beta_0,
                },
                HalfTreeSpDpfKey {
                    party_id: 1,
                    log_domain_size,
                    party_seed: Default::default(),
                    correction_words: Default::default(),
                    hcw: Default::default(),
                    lcw: Default::default(),
                    correction_word_np1: beta_1,
                },
            );
        }

        let fkaes = FixedKeyAes::new(Self::FIXED_KEY_AES_KEY);
        let hash = |x: u128| fkaes.hash_ccr(Self::HASH_KEY ^ x);
        let convert = |x: u128| -> V { PRConverter::convert(x) };

        let n = log_domain_size as usize;
        let alpha_bits: Vec<bool> = bit_decompose(alpha, n);

        let delta = rng.gen::<u128>() | 1u128;

        let mut correction_words = Vec::<u128>::with_capacity(n - 1);
        let mut st_0 = rng.gen::<u128>();
        let mut st_1 = st_0 ^ delta;
        let party_seeds = (st_0, st_1);

        for i in 0..(n - 1) as usize {
            let cw_i = hash(st_0) ^ hash(st_1) ^ (1 - alpha_bits[i] as u128) * delta;
            st_0 = hash(st_0) ^ alpha_bits[i] as u128 * (st_0) ^ (st_0 & 1) * cw_i;
            st_1 = hash(st_1) ^ alpha_bits[i] as u128 * (st_1) ^ (st_1 & 1) * cw_i;
            correction_words.push(cw_i);
        }

        let high_low = [[hash(st_0), hash(st_0 ^ 1)], [hash(st_1), hash(st_1 ^ 1)]];
        const HIGH_MASK: u128 = u128::MAX - 1;
        const LOW_MASK: u128 = 1u128;
        let a_n = alpha_bits[n - 1];
        let hcw = (high_low[0][1 - a_n as usize] ^ high_low[1][1 - a_n as usize]) & HIGH_MASK;
        let lcw = [
            ((high_low[0][0] ^ high_low[1][0] ^ (1 - a_n as u128)) & LOW_MASK) != 0,
            ((high_low[0][1] ^ high_low[1][1] ^ a_n as u128) & LOW_MASK) != 0,
        ];

        st_0 = high_low[0][a_n as usize] ^ (st_0 & 1) * (hcw | lcw[a_n as usize] as u128);
        st_1 = high_low[1][a_n as usize] ^ (st_1 & 1) * (hcw | lcw[a_n as usize] as u128);
        let correction_word_np1: V = match (st_0 & 1).wrapping_sub(st_1 & 1) {
            u128::MAX => convert(st_0 >> 1) - convert(st_1 >> 1) - beta,
            0 => V::zero(),
            1 => convert(st_1 >> 1) - convert(st_0 >> 1) + beta,
            _ => panic!("should not happend, since matching a difference of two bits"),
        };

        (
            HalfTreeSpDpfKey {
                party_id: 0,
                log_domain_size,
                party_seed: party_seeds.0,
                correction_words: correction_words.clone(),
                hcw,
                lcw,
                correction_word_np1,
            },
            HalfTreeSpDpfKey {
                party_id: 1,
                log_domain_size,
                party_seed: party_seeds.1,
                correction_words,
                hcw,
                lcw,
                correction_word_np1,
            },
        )
    }

    fn evaluate_at(key: &Self::Key, index: u64) -> V {
        assert!(index < (1 << key.log_domain_size));

        if key.log_domain_size == 0 {
            // beta is simply secret-shared
            return key.correction_word_np1;
        }

        let fkaes = FixedKeyAes::new(Self::FIXED_KEY_AES_KEY);
        let hash = |x: u128| fkaes.hash_ccr(Self::HASH_KEY ^ x);
        let convert = |x: u128| -> V { PRConverter::convert(x) };

        let n = key.log_domain_size as usize;
        let index_bits: Vec<bool> = bit_decompose(index, n);

        let mut st_b = key.party_seed;
        for i in 0..n - 1 {
            st_b = hash(st_b) ^ index_bits[i] as u128 * st_b ^ (st_b & 1) * key.correction_words[i];
        }
        let x_n = index_bits[n - 1];
        let high_low_b_xn = hash(st_b ^ x_n as u128);
        st_b = high_low_b_xn ^ (st_b & 1) * (key.hcw | key.lcw[x_n as usize] as u128);

        let value = convert(st_b >> 1)
            + if st_b & 1 == 0 {
                V::zero()
            } else {
                key.correction_word_np1
            };
        if key.party_id == 0 {
            value
        } else {
            V::zero() - value
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::num::Wrapping;
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
        for log_domain_size in 0..10 {
            test_spdpf_with_param::<DummySpDpf<u64>>(log_domain_size);
            test_spdpf_with_param::<HalfTreeSpDpf<Wrapping<u64>>>(log_domain_size);
        }
    }
}
