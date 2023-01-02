use core::fmt::Debug;
use core::marker::PhantomData;
use core::ops::{Add, Neg, Sub};
use num::traits::Zero;
use rand::{thread_rng, Rng};

use utils::bit_decompose::bit_decompose;
use utils::fixed_key_aes::FixedKeyAes;
use utils::pseudorandom_conversion::{PRConvertTo, PRConverter};

pub trait SinglePointDpfKey: Clone + Debug {
    fn get_party_id(&self) -> usize;
    fn get_domain_size(&self) -> usize;
}

pub trait SinglePointDpf {
    type Key: SinglePointDpfKey;
    type Value: Add<Output = Self::Value> + Copy + Debug + Eq + Zero;

    fn generate_keys(domain_size: usize, alpha: u64, beta: Self::Value) -> (Self::Key, Self::Key);
    fn evaluate_at(key: &Self::Key, index: u64) -> Self::Value;
    fn evaluate_domain(key: &Self::Key) -> Vec<Self::Value> {
        (0..key.get_domain_size())
            .map(|x| Self::evaluate_at(&key, x as u64))
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DummySpDpfKey<V: Copy + Debug> {
    party_id: usize,
    domain_size: usize,
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
    fn get_domain_size(&self) -> usize {
        self.domain_size
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

    fn generate_keys(domain_size: usize, alpha: u64, beta: V) -> (Self::Key, Self::Key) {
        assert!(alpha < domain_size as u64);
        (
            DummySpDpfKey {
                party_id: 0,
                domain_size,
                alpha,
                beta,
            },
            DummySpDpfKey {
                party_id: 1,
                domain_size,
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
    /// size n of the DPF's domain [n]
    domain_size: usize,
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
    fn get_domain_size(&self) -> usize {
        self.domain_size
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
    V: Add<Output = V> + Sub<Output = V> + Neg<Output = V> + Copy + Debug + Eq + Zero,
    PRConverter: PRConvertTo<V>,
{
    type Key = HalfTreeSpDpfKey<V>;
    type Value = V;

    fn generate_keys(domain_size: usize, alpha: u64, beta: V) -> (Self::Key, Self::Key) {
        assert!(alpha < domain_size as u64);

        let mut rng = thread_rng();

        if domain_size == 1 {
            // simply secret-share beta
            let beta_0: V = PRConverter::convert(rng.gen::<u128>());
            let beta_1: V = beta - beta_0;
            return (
                HalfTreeSpDpfKey {
                    party_id: 0,
                    domain_size,
                    party_seed: Default::default(),
                    correction_words: Default::default(),
                    hcw: Default::default(),
                    lcw: Default::default(),
                    correction_word_np1: beta_0,
                },
                HalfTreeSpDpfKey {
                    party_id: 1,
                    domain_size,
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

        let tree_height = (domain_size as f64).log2().ceil() as usize;
        let alpha_bits: Vec<bool> = bit_decompose(alpha, tree_height);

        let delta = rng.gen::<u128>() | 1u128;

        let mut correction_words = Vec::<u128>::with_capacity(tree_height - 1);
        let mut st_0 = rng.gen::<u128>();
        let mut st_1 = st_0 ^ delta;
        let party_seeds = (st_0, st_1);

        for i in 0..(tree_height - 1) as usize {
            let cw_i = hash(st_0) ^ hash(st_1) ^ (1 - alpha_bits[i] as u128) * delta;
            st_0 = hash(st_0) ^ alpha_bits[i] as u128 * (st_0) ^ (st_0 & 1) * cw_i;
            st_1 = hash(st_1) ^ alpha_bits[i] as u128 * (st_1) ^ (st_1 & 1) * cw_i;
            correction_words.push(cw_i);
        }

        let high_low = [[hash(st_0), hash(st_0 ^ 1)], [hash(st_1), hash(st_1 ^ 1)]];
        const HIGH_MASK: u128 = u128::MAX - 1;
        const LOW_MASK: u128 = 1u128;
        let a_n = alpha_bits[tree_height - 1];
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
                domain_size,
                party_seed: party_seeds.0,
                correction_words: correction_words.clone(),
                hcw,
                lcw,
                correction_word_np1,
            },
            HalfTreeSpDpfKey {
                party_id: 1,
                domain_size,
                party_seed: party_seeds.1,
                correction_words,
                hcw,
                lcw,
                correction_word_np1,
            },
        )
    }

    fn evaluate_at(key: &Self::Key, index: u64) -> V {
        assert!(key.domain_size > 0);
        assert!(index < key.domain_size as u64);

        if key.domain_size == 1 {
            // beta is simply secret-shared
            return key.correction_word_np1;
        }

        let fkaes = FixedKeyAes::new(Self::FIXED_KEY_AES_KEY);
        let hash = |x: u128| fkaes.hash_ccr(Self::HASH_KEY ^ x);
        let convert = |x: u128| -> V { PRConverter::convert(x) };

        let tree_height = (key.domain_size as f64).log2().ceil() as usize;
        let index_bits: Vec<bool> = bit_decompose(index, tree_height);

        let mut st_b = key.party_seed;
        for i in 0..tree_height - 1 {
            st_b = hash(st_b) ^ index_bits[i] as u128 * st_b ^ (st_b & 1) * key.correction_words[i];
        }
        let x_n = index_bits[tree_height - 1];
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

    fn evaluate_domain(key: &Self::Key) -> Vec<V> {
        assert!(key.domain_size > 0);
        let fkaes = FixedKeyAes::new(Self::FIXED_KEY_AES_KEY);
        let hash = |x: u128| fkaes.hash_ccr(Self::HASH_KEY ^ x);
        let convert = |x: u128| -> V { PRConverter::convert(x) };

        if key.domain_size == 1 {
            // beta is simply secret-shared
            return vec![key.correction_word_np1];
        }

        let tree_height = (key.domain_size as f64).log2().ceil() as usize;
        let last_index = key.domain_size - 1;

        let mut seeds = vec![0u128; key.domain_size];
        seeds[0] = key.party_seed;

        // since the last layer is handled separately, we only need the following block if we have
        // more than one layer
        if tree_height > 1 {
            // iterate over the tree layer by layer
            for i in 0..(tree_height - 1) {
                // expand each node in this layer;
                // we need to iterate from right to left, since we reuse the same buffer
                for j in (0..(last_index >> (tree_height - i)) + 1).rev() {
                    // for j in (0..(1 << i)).rev() {
                    let st = seeds[j];
                    let st_0 = hash(st) ^ (st & 1) * key.correction_words[i];
                    let st_1 = hash(st) ^ st ^ (st & 1) * key.correction_words[i];
                    seeds[2 * j] = st_0;
                    seeds[2 * j + 1] = st_1;
                }
            }
        }

        // expand last layer
        {
            // handle the last expansion separately, since we might not need both outputs
            let j = (key.domain_size >> 1) - 1;
            let st = seeds[j];
            let st_0 = hash(st) ^ (st & 1) * (key.hcw | key.lcw[0] as u128);
            seeds[2 * j] = st_0;
            // check if we need both outputs
            if key.domain_size & 1 == 0 {
                let st_1 = hash(st ^ 1 as u128) ^ (st & 1) * (key.hcw | key.lcw[1] as u128);
                seeds[2 * j + 1] = st_1;
            }

            // handle the other expansions as usual
            for j in (0..(key.domain_size >> 1) - 1).rev() {
                let st = seeds[j];
                let st_0 = hash(st) ^ (st & 1) * (key.hcw | key.lcw[0] as u128);
                let st_1 = hash(st ^ 1 as u128) ^ (st & 1) * (key.hcw | key.lcw[1] as u128);
                seeds[2 * j] = st_0;
                seeds[2 * j + 1] = st_1;
            }
        }

        // convert leaves into V elements
        if key.party_id == 0 {
            seeds
                .iter()
                .map(|st_b| {
                    let mut tmp = convert(st_b >> 1);
                    if st_b & 1 == 1 {
                        tmp = tmp + key.correction_word_np1;
                    }
                    tmp
                })
                .collect()
        } else {
            seeds
                .iter()
                .map(|st_b| {
                    let mut tmp = convert(st_b >> 1);
                    if st_b & 1 == 1 {
                        tmp = tmp + key.correction_word_np1;
                    }
                    -tmp
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::num::Wrapping;
    use rand::distributions::{Distribution, Standard};
    use rand::{thread_rng, Rng};

    fn test_spdpf_with_param<SPDPF: SinglePointDpf>(domain_size: usize)
    where
        Standard: Distribution<SPDPF::Value>,
    {
        let alpha = thread_rng().gen_range(0..domain_size as u64);
        let beta = thread_rng().gen();
        let (key_0, key_1) = SPDPF::generate_keys(domain_size, alpha, beta);

        let out_0 = SPDPF::evaluate_domain(&key_0);
        let out_1 = SPDPF::evaluate_domain(&key_1);
        for i in 0..domain_size as u64 {
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
    fn test_spdpf_dummy() {
        for log_domain_size in 0..10 {
            test_spdpf_with_param::<DummySpDpf<u64>>(1 << log_domain_size);
        }
    }

    #[test]
    fn test_spdpf_half_tree_power_of_two_domain() {
        for log_domain_size in 0..10 {
            test_spdpf_with_param::<HalfTreeSpDpf<Wrapping<u64>>>(1 << log_domain_size);
        }
    }

    #[test]
    fn test_spdpf_half_tree_random_domain() {
        for _ in 0..10 {
            let domain_size = thread_rng().gen_range(1..(1 << 10));
            test_spdpf_with_param::<HalfTreeSpDpf<Wrapping<u64>>>(domain_size);
        }
    }
}
