//! Trait definitions and implementations of single-point distributed point functions (SP-DPFs).

use bincode;
use core::fmt::Debug;
use core::marker::PhantomData;
use core::ops::{Add, Neg, Sub};
use num::traits::Zero;
use rand::{thread_rng, Rng};
use utils::bit_decompose::bit_decompose;
use utils::fixed_key_aes::FixedKeyAes;
use utils::pseudorandom_conversion::{PRConvertTo, PRConverter};

/// Trait for the keys of a single-point DPF scheme.
pub trait SinglePointDpfKey: Clone + Debug {
    /// Return the party ID, 0 or 1, corresponding to this key.
    fn get_party_id(&self) -> usize;

    /// Return the domain size of the shared function.
    fn get_domain_size(&self) -> u128;
}

/// Trait for a single-point DPF scheme.
pub trait SinglePointDpf {
    /// The key type of the scheme.
    type Key: SinglePointDpfKey;

    /// The value type of the scheme.
    type Value: Add<Output = Self::Value> + Copy + Debug + Eq + Zero;

    /// Key generation for a given `domain_size`, an index `alpha` and a value `beta`.
    ///
    /// The shared point function is `f: {0, ..., domain_size - 1} -> Self::Value` such that
    /// `f(alpha) = beta` and `f(x) = 0` for `x != alpha`.
    fn generate_keys(domain_size: u128, alpha: u128, beta: Self::Value) -> (Self::Key, Self::Key);

    /// Evaluation using a DPF key on a single `index` from `{0, ..., domain_size - 1}`.
    fn evaluate_at(key: &Self::Key, index: u128) -> Self::Value;

    /// Evaluation using a DPF key on the whole domain.
    ///
    /// This might be implemented more efficiently than just repeatedly calling
    /// [`Self::evaluate_at`].
    fn evaluate_domain(key: &Self::Key) -> Vec<Self::Value> {
        (0..key.get_domain_size())
            .map(|x| Self::evaluate_at(key, x as u128))
            .collect()
    }
}

/// Key type for the insecure [DummySpDpf] scheme, which trivially contains the defining parameters
/// `alpha` and `beta`.
#[derive(Clone, Copy, Debug, bincode::Encode, bincode::Decode)]
pub struct DummySpDpfKey<V: Copy + Debug> {
    party_id: usize,
    domain_size: u128,
    alpha: u128,
    beta: V,
}

impl<V> SinglePointDpfKey for DummySpDpfKey<V>
where
    V: Copy + Debug,
{
    fn get_party_id(&self) -> usize {
        self.party_id
    }
    fn get_domain_size(&self) -> u128 {
        self.domain_size
    }
}

/// Insecure SP-DPF scheme for testing purposes.
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

    fn generate_keys(domain_size: u128, alpha: u128, beta: V) -> (Self::Key, Self::Key) {
        assert!(alpha < domain_size);
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

    fn evaluate_at(key: &Self::Key, index: u128) -> V {
        if key.get_party_id() == 0 && index == key.alpha {
            key.beta
        } else {
            V::zero()
        }
    }

    fn evaluate_domain(key: &Self::Key) -> Vec<Self::Value> {
        debug_assert!(key.domain_size <= usize::MAX as u128);
        let mut output = vec![V::zero(); key.domain_size as usize];
        if key.get_party_id() == 0 {
            output[key.alpha as usize] = key.beta;
        }
        output
    }
}

/// Key type for the [HalfTreeSpDpf] scheme.
#[derive(Clone, Debug, bincode::Encode, bincode::Decode)]
pub struct HalfTreeSpDpfKey<V: Copy + Debug> {
    /// party id `b`
    party_id: usize,
    /// size `n` of the DPF's domain `[n]`
    domain_size: u128,
    /// `(s_b^0 || t_b^0)` and `t_b^0` is the LSB
    party_seed: u128,
    /// vector of length `n`: `CW_1, ..., CW_(n-1)`
    correction_words: Vec<u128>,
    /// high part of `CW_n = (HCW, [LCW[0], LCW[1]])`
    hcw: u128,
    /// low parts of `CW_n = (HCW, [LCW[0], LCW[1]])`
    lcw: [bool; 2],
    /// `CW_(n+1)`
    correction_word_np1: V,
}

impl<V> SinglePointDpfKey for HalfTreeSpDpfKey<V>
where
    V: Copy + Debug,
{
    fn get_party_id(&self) -> usize {
        self.party_id
    }
    fn get_domain_size(&self) -> u128 {
        self.domain_size
    }
}

/// Implementation of the Half-Tree DPF scheme from Guo et al. ([ePrint 2022/1431, Figure
/// 8](https://eprint.iacr.org/2022/1431.pdf#page=18)).
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
    const HASH_KEY: u128 = 0xc000_ffee_c0ff_ffee_c0ff_eeee_c00f_feee_u128;
}

impl<V> SinglePointDpf for HalfTreeSpDpf<V>
where
    V: Add<Output = V> + Sub<Output = V> + Neg<Output = V> + Copy + Debug + Eq + Zero,
    PRConverter: PRConvertTo<V>,
{
    type Key = HalfTreeSpDpfKey<V>;
    type Value = V;

    fn generate_keys(domain_size: u128, alpha: u128, beta: V) -> (Self::Key, Self::Key) {
        assert!(alpha < domain_size);

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

        debug_assert_eq!(alpha_bits.len(), tree_height);

        for alpha_i in alpha_bits.iter().copied().take(tree_height - 1) {
            let cw_i = hash(st_0) ^ hash(st_1) ^ ((1 - alpha_i as u128) * delta);
            st_0 = hash(st_0) ^ (alpha_i as u128 * st_0) ^ ((st_0 & 1) * cw_i);
            st_1 = hash(st_1) ^ (alpha_i as u128 * st_1) ^ ((st_1 & 1) * cw_i);
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

        st_0 = high_low[0][a_n as usize] ^ ((st_0 & 1) * (hcw | lcw[a_n as usize] as u128));
        st_1 = high_low[1][a_n as usize] ^ ((st_1 & 1) * (hcw | lcw[a_n as usize] as u128));
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

    fn evaluate_at(key: &Self::Key, index: u128) -> V {
        assert!(key.domain_size > 0);
        assert!(index < key.domain_size);

        if key.domain_size == 1 {
            // beta is simply secret-shared
            return key.correction_word_np1;
        }

        let fkaes = FixedKeyAes::new(Self::FIXED_KEY_AES_KEY);
        let hash = |x: u128| fkaes.hash_ccr(Self::HASH_KEY ^ x);
        let convert = |x: u128| -> V { PRConverter::convert(x) };

        let tree_height = (key.domain_size as f64).log2().ceil() as usize;
        let index_bits: Vec<bool> = bit_decompose(index, tree_height);

        debug_assert_eq!(index_bits.len(), tree_height);

        let mut st_b = key.party_seed;
        for (index_bit_i, correction_word_i) in index_bits
            .iter()
            .copied()
            .zip(key.correction_words.iter())
            .take(tree_height - 1)
        {
            st_b = hash(st_b) ^ (index_bit_i as u128 * st_b) ^ ((st_b & 1) * correction_word_i);
        }
        let x_n = index_bits[tree_height - 1];
        let high_low_b_xn = hash(st_b ^ x_n as u128);
        st_b = high_low_b_xn ^ ((st_b & 1) * (key.hcw | key.lcw[x_n as usize] as u128));

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
        assert!(key.domain_size <= usize::MAX as u128);

        if key.domain_size == 1 {
            // beta is simply secret-shared
            return vec![key.correction_word_np1];
        }

        let fkaes = FixedKeyAes::new(Self::FIXED_KEY_AES_KEY);
        let hash = |x: u128| fkaes.hash_ccr(Self::HASH_KEY ^ x);
        let convert = |x: u128| -> V { PRConverter::convert(x) };

        let tree_height = (key.domain_size as f64).log2().ceil() as usize;
        let last_index = (key.domain_size - 1) as usize;

        let mut seeds = vec![0u128; key.domain_size as usize];
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
                    let st_0 = hash(st) ^ ((st & 1) * key.correction_words[i]);
                    let st_1 = hash(st) ^ st ^ ((st & 1) * key.correction_words[i]);
                    seeds[2 * j] = st_0;
                    seeds[2 * j + 1] = st_1;
                }
            }
        }

        // expand last layer
        {
            // handle the last expansion separately, since we might not need both outputs
            let j = last_index >> 1;
            let st = seeds[j];
            let st_0 = hash(st) ^ ((st & 1) * (key.hcw | key.lcw[0] as u128));
            seeds[2 * j] = st_0;
            // check if we need both outputs
            if key.domain_size & 1 == 0 {
                let st_1 = hash(st ^ 1) ^ ((st & 1) * (key.hcw | key.lcw[1] as u128));
                seeds[2 * j + 1] = st_1;
            }

            // handle the other expansions as usual
            for j in (0..(last_index >> 1)).rev() {
                let st = seeds[j];
                let st_0 = hash(st) ^ ((st & 1) * (key.hcw | key.lcw[0] as u128));
                let st_1 = hash(st ^ 1) ^ ((st & 1) * (key.hcw | key.lcw[1] as u128));
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

    fn test_spdpf_with_param<SPDPF: SinglePointDpf>(domain_size: u128, alpha: Option<u128>)
    where
        Standard: Distribution<SPDPF::Value>,
    {
        let alpha = if alpha.is_some() {
            alpha.unwrap()
        } else {
            thread_rng().gen_range(0..domain_size)
        };
        let beta = thread_rng().gen();
        let (key_0, key_1) = SPDPF::generate_keys(domain_size, alpha, beta);

        let out_0 = SPDPF::evaluate_domain(&key_0);
        let out_1 = SPDPF::evaluate_domain(&key_1);
        assert_eq!(out_0.len() as u128, domain_size);
        assert_eq!(out_1.len() as u128, domain_size);
        for i in 0..domain_size {
            let value = SPDPF::evaluate_at(&key_0, i) + SPDPF::evaluate_at(&key_1, i);
            assert_eq!(
                value,
                out_0[i as usize] + out_1[i as usize],
                "evaluate_at/domain mismatch at position {i}"
            );
            if i == alpha {
                assert_eq!(
                    value, beta,
                    "incorrect value != beta at position alpha = {i}"
                );
            } else {
                assert_eq!(
                    value,
                    SPDPF::Value::zero(),
                    "incorrect value != 0 at position {i}"
                );
            }
        }
    }

    #[test]
    fn test_spdpf_dummy() {
        for log_domain_size in 0..10 {
            test_spdpf_with_param::<DummySpDpf<u64>>(1 << log_domain_size, None);
        }
    }

    #[test]
    fn test_spdpf_half_tree_power_of_two_domain() {
        for log_domain_size in 0..10 {
            test_spdpf_with_param::<HalfTreeSpDpf<Wrapping<u64>>>(1 << log_domain_size, None);
        }
    }

    #[test]
    fn test_spdpf_half_tree_random_domain() {
        for _ in 0..10 {
            let domain_size = thread_rng().gen_range(1..(1 << 10));
            test_spdpf_with_param::<HalfTreeSpDpf<Wrapping<u64>>>(domain_size, None);
        }
    }

    #[test]
    fn test_spdpf_half_tree_exhaustive_params() {
        for domain_size in 1..=32 {
            for alpha in 0..domain_size as u128 {
                test_spdpf_with_param::<HalfTreeSpDpf<Wrapping<u64>>>(domain_size, Some(alpha));
            }
        }
    }
}
