use core::fmt;
use core::fmt::Debug;
use core::marker::PhantomData;
use core::ops::{Add, AddAssign};
use num::traits::Zero;

use crate::spdpf::SinglePointDpf;
use cuckoo::{
    cuckoo::Hasher as CuckooHasher, cuckoo::Parameters as CuckooParameters,
    cuckoo::NUMBER_HASH_FUNCTIONS as CUCKOO_NUMBER_HASH_FUNCTIONS, hash::HashFunction,
};

pub trait MultiPointDpfKey: Clone + Debug {
    fn get_party_id(&self) -> usize;
    fn get_log_domain_size(&self) -> u64;
    fn get_number_points(&self) -> usize;
}

pub trait MultiPointDpf {
    type Key: MultiPointDpfKey;
    type Value: Add<Output = Self::Value> + Copy + Debug + Eq + Zero;

    fn generate_keys(
        log_domain_size: u64,
        alphas: &[u64],
        betas: &[Self::Value],
    ) -> (Self::Key, Self::Key);
    fn evaluate_at(key: &Self::Key, index: u64) -> Self::Value;
    fn evaluate_domain(key: &Self::Key) -> Vec<Self::Value> {
        (0..(1 << key.get_log_domain_size()))
            .map(|x| Self::evaluate_at(&key, x))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct DummyMpDpfKey<V: Copy + Debug> {
    party_id: usize,
    log_domain_size: u64,
    number_points: usize,
    alphas: Vec<u64>,
    betas: Vec<V>,
}

impl<V> MultiPointDpfKey for DummyMpDpfKey<V>
where
    V: Copy + Debug,
{
    fn get_party_id(&self) -> usize {
        self.party_id
    }
    fn get_log_domain_size(&self) -> u64 {
        self.log_domain_size
    }
    fn get_number_points(&self) -> usize {
        self.number_points
    }
}

pub struct DummyMpDpf<V>
where
    V: Add<Output = V> + Copy + Debug + Eq + Zero,
{
    phantom: PhantomData<V>,
}

impl<V> MultiPointDpf for DummyMpDpf<V>
where
    V: Add<Output = V> + Copy + Debug + Eq + Zero,
{
    type Key = DummyMpDpfKey<V>;
    type Value = V;

    fn generate_keys(log_domain_size: u64, alphas: &[u64], betas: &[V]) -> (Self::Key, Self::Key) {
        assert_eq!(
            alphas.len(),
            betas.len(),
            "alphas and betas must be the same size"
        );
        assert!(
            alphas.iter().all(|alpha| alpha < &(1 << log_domain_size)),
            "all alphas must be in the domain"
        );
        assert!(alphas.windows(2).all(|w| w[0] <= w[1]));
        let number_points = alphas.len();
        (
            DummyMpDpfKey {
                party_id: 0,
                log_domain_size,
                number_points,
                alphas: alphas.iter().copied().collect(),
                betas: betas.iter().copied().collect(),
            },
            DummyMpDpfKey {
                party_id: 1,
                log_domain_size,
                number_points,
                alphas: alphas.iter().copied().collect(),
                betas: betas.iter().copied().collect(),
            },
        )
    }

    fn evaluate_at(key: &Self::Key, index: u64) -> V {
        if key.get_party_id() == 0 {
            match key.alphas.binary_search(&index) {
                Ok(i) => key.betas[i],
                Err(_) => V::zero(),
            }
        } else {
            V::zero()
        }
    }
}

pub struct SmartMpDpfKey<SPDPF, H>
where
    SPDPF: SinglePointDpf,
    H: HashFunction,
{
    party_id: usize,
    log_domain_size: u64,
    number_points: usize,
    spdpf_keys: Vec<Option<SPDPF::Key>>,
    cuckoo_parameters: CuckooParameters<H>,
}

impl<SPDPF, H> Debug for SmartMpDpfKey<SPDPF, H>
where
    SPDPF: SinglePointDpf,
    H: HashFunction,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let (newline, indentation) = if f.alternate() {
            ("\n", "    ")
        } else {
            (" ", "")
        };
        write!(f, "SmartMpDpfKey<SPDPF, H>{{{}", newline)?;
        write!(
            f,
            "{}party_id: {:?},{}",
            indentation, self.party_id, newline
        )?;
        write!(
            f,
            "{}log_domain_size: {:?},{}",
            indentation, self.log_domain_size, newline
        )?;
        write!(
            f,
            "{}number_points: {:?},{}",
            indentation, self.number_points, newline
        )?;
        if f.alternate() {
            write!(f, "    spdpf_keys:\n")?;
            for (i, k) in self.spdpf_keys.iter().enumerate() {
                write!(f, "        spdpf_keys[{}]: {:?}\n", i, k)?;
            }
        } else {
            write!(f, " spdpf_keys: {:?},", self.spdpf_keys)?;
        }
        write!(
            f,
            "{}cuckoo_parameters: {:?}{}",
            indentation, self.cuckoo_parameters, newline
        )?;
        write!(f, "}}")?;
        Ok(())
    }
}

impl<SPDPF, H> Clone for SmartMpDpfKey<SPDPF, H>
where
    SPDPF: SinglePointDpf,
    H: HashFunction,
{
    fn clone(&self) -> Self {
        Self {
            party_id: self.party_id,
            log_domain_size: self.log_domain_size,
            number_points: self.number_points,
            spdpf_keys: self.spdpf_keys.clone(),
            cuckoo_parameters: self.cuckoo_parameters.clone(),
        }
    }
}

impl<SPDPF, H> MultiPointDpfKey for SmartMpDpfKey<SPDPF, H>
where
    SPDPF: SinglePointDpf,
    H: HashFunction,
{
    fn get_party_id(&self) -> usize {
        self.party_id
    }
    fn get_log_domain_size(&self) -> u64 {
        self.log_domain_size
    }
    fn get_number_points(&self) -> usize {
        self.number_points
    }
}

pub struct SmartMpDpf<V, SPDPF, H>
where
    V: Add<Output = V> + AddAssign + Copy + Debug + Eq + Zero,
    SPDPF: SinglePointDpf<Value = V>,
    H: HashFunction,
{
    phantom_v: PhantomData<V>,
    phantom_s: PhantomData<SPDPF>,
    phantom_h: PhantomData<H>,
}

impl<V, SPDPF, H> MultiPointDpf for SmartMpDpf<V, SPDPF, H>
where
    V: Add<Output = V> + AddAssign + Copy + Debug + Eq + Zero,
    SPDPF: SinglePointDpf<Value = V>,
    H: HashFunction,
{
    type Key = SmartMpDpfKey<SPDPF, H>;
    type Value = V;

    fn generate_keys(
        log_domain_size: u64,
        alphas: &[u64],
        betas: &[Self::Value],
    ) -> (Self::Key, Self::Key) {
        assert_eq!(alphas.len(), betas.len());
        assert!(alphas.windows(2).all(|w| w[0] < w[1]));
        assert!(alphas.iter().all(|&alpha| alpha < (1 << log_domain_size)));
        let number_points = alphas.len();

        let cuckoo_parameters = CuckooParameters::<H>::sample(number_points);
        let hasher = CuckooHasher::<H>::new(cuckoo_parameters);
        let (cuckoo_table_items, cuckoo_table_indices) = hasher.cuckoo_hash_items(alphas);
        let simple_htable = hasher.hash_domain_into_buckets(1 << log_domain_size);

        let pos = |bucket_i: usize, item: u64| -> u64 {
            let idx = simple_htable[bucket_i].partition_point(|x| x < &item);
            assert!(idx != simple_htable[bucket_i].len());
            assert_eq!(item, simple_htable[bucket_i][idx]);
            assert!(idx == 0 || simple_htable[bucket_i][idx - 1] != item);
            idx as u64
        };

        let number_buckets = hasher.get_parameters().get_number_buckets();

        let mut keys_0 = Vec::<Option<SPDPF::Key>>::with_capacity(number_buckets);
        let mut keys_1 = Vec::<Option<SPDPF::Key>>::with_capacity(number_buckets);
        let mut bucket_sizes = vec![0u64; number_buckets];

        for bucket_i in 0..number_buckets {
            let bucket_size = simple_htable[bucket_i].len() as u64;

            // remember the bucket size
            bucket_sizes[bucket_i] = bucket_size;

            // if bucket is empty, add invalid dummy keys to the arrays to make the
            // indices work
            if bucket_size == 0 {
                keys_0.push(None);
                keys_1.push(None);
                continue;
            }

            let sp_log_domain_size = (bucket_size as f64).log2().ceil() as u64;

            let (alpha, beta) = if cuckoo_table_items[bucket_i] != CuckooHasher::<H>::UNOCCUPIED {
                let alpha = pos(bucket_i, cuckoo_table_items[bucket_i]);
                let beta = betas[cuckoo_table_indices[bucket_i]];
                (alpha, beta)
            } else {
                (0, V::zero())
            };
            let (key_0, key_1) = SPDPF::generate_keys(sp_log_domain_size, alpha, beta);
            keys_0.push(Some(key_0));
            keys_1.push(Some(key_1));
        }

        (
            SmartMpDpfKey::<SPDPF, H> {
                party_id: 0,
                log_domain_size,
                number_points,
                spdpf_keys: keys_0,
                cuckoo_parameters,
            },
            SmartMpDpfKey::<SPDPF, H> {
                party_id: 1,
                log_domain_size,
                number_points,
                spdpf_keys: keys_1,
                cuckoo_parameters,
            },
        )
    }

    fn evaluate_at(key: &Self::Key, index: u64) -> Self::Value {
        let domain_size = 1 << key.log_domain_size;
        assert!(index < domain_size);

        let hasher = CuckooHasher::<H>::new(key.cuckoo_parameters);

        let hashes = hasher.hash_items(&[index]);
        let simple_htable = hasher.hash_domain_into_buckets(domain_size);

        let pos = |bucket_i: usize, item: u64| -> u64 {
            let idx = simple_htable[bucket_i].partition_point(|x| x < &item);
            assert!(idx != simple_htable[bucket_i].len());
            assert_eq!(item, simple_htable[bucket_i][idx]);
            assert!(idx == 0 || simple_htable[bucket_i][idx - 1] != item);
            idx as u64
        };
        eprintln!(
            "hashes({}) = ({:#?}, {:#?}, {:#?})",
            index, hashes[0][0], hashes[1][0], hashes[2][0]
        );
        let mut output = {
            let hash = hashes[0][0] as usize;
            assert!(key.spdpf_keys[hash].is_some());
            let sp_key = key.spdpf_keys[hash].as_ref().unwrap();
            assert_eq!(simple_htable[hash][pos(hash, index) as usize], index);
            SPDPF::evaluate_at(&sp_key, pos(hash, index))
        };

        // prevent adding the same term multiple times when we have collisions
        let mut hash_bit_map = [0u8; 2];
        if hashes[0][0] != hashes[1][0] {
            // hash_bit_map[i] |= 1;
            hash_bit_map[0] = 1;
        }
        if hashes[0][0] != hashes[2][0] && hashes[1][0] != hashes[2][0] {
            // hash_bit_map[i] |= 2;
            hash_bit_map[1] = 1;
        }

        for j in 1..CUCKOO_NUMBER_HASH_FUNCTIONS {
            if hash_bit_map[j - 1] == 0 {
                continue;
            }
            let hash = hashes[j][0] as usize;
            assert!(key.spdpf_keys[hash].is_some());
            let sp_key = key.spdpf_keys[hash].as_ref().unwrap();
            assert_eq!(simple_htable[hash][pos(hash, index) as usize], index);
            output += SPDPF::evaluate_at(&sp_key, pos(hash, index));
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spdpf::DummySpDpf;
    use cuckoo::hash::AesHashFunction;
    use rand::distributions::{Distribution, Standard};
    use rand::{thread_rng, Rng};
    use std::num::Wrapping;

    fn test_mpdpf_with_param<MPDPF: MultiPointDpf>(log_domain_size: u64, number_points: usize)
    where
        Standard: Distribution<MPDPF::Value>,
    {
        let domain_size = (1 << log_domain_size) as u64;
        assert!(number_points <= domain_size as usize);
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
        let betas: Vec<MPDPF::Value> = (0..number_points).map(|_| thread_rng().gen()).collect();
        let (key_0, key_1) = MPDPF::generate_keys(log_domain_size, &alphas, &betas);

        let out_0 = MPDPF::evaluate_domain(&key_0);
        let out_1 = MPDPF::evaluate_domain(&key_1);
        for i in 0..domain_size {
            let value = MPDPF::evaluate_at(&key_0, i) + MPDPF::evaluate_at(&key_1, i);
            assert_eq!(value, out_0[i as usize] + out_1[i as usize]);
            let expected_result = match alphas.binary_search(&i) {
                Ok(i) => betas[i],
                Err(_) => MPDPF::Value::zero(),
            };
            assert_eq!(value, expected_result, "wrong value at index {}", i);
        }
    }

    #[test]
    fn test_mpdpf() {
        type Value = Wrapping<u64>;
        for log_domain_size in 5..10 {
            for log_number_points in 0..5 {
                test_mpdpf_with_param::<DummyMpDpf<Value>>(log_domain_size, 1 << log_number_points);
                test_mpdpf_with_param::<SmartMpDpf<Value, DummySpDpf<Value>, AesHashFunction>>(
                    log_domain_size,
                    1 << log_number_points,
                );
            }
        }
    }
}
