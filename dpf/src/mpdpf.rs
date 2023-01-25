use bincode;
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
    fn get_domain_size(&self) -> usize;
    fn get_number_points(&self) -> usize;
}

pub trait MultiPointDpf {
    type Key: MultiPointDpfKey;
    type Value: Add<Output = Self::Value> + Copy + Debug + Eq + Zero;

    fn new(domain_size: usize, number_points: usize) -> Self;
    fn get_domain_size(&self) -> usize;
    fn get_number_points(&self) -> usize;
    fn precompute(&mut self) {}
    fn generate_keys(&self, alphas: &[u64], betas: &[Self::Value]) -> (Self::Key, Self::Key);
    fn evaluate_at(&self, key: &Self::Key, index: u64) -> Self::Value;
    fn evaluate_domain(&self, key: &Self::Key) -> Vec<Self::Value> {
        (0..key.get_domain_size())
            .map(|x| self.evaluate_at(&key, x as u64))
            .collect()
    }
}

#[derive(Clone, Debug, bincode::Encode, bincode::Decode)]
pub struct DummyMpDpfKey<V: Copy + Debug> {
    party_id: usize,
    domain_size: usize,
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
    fn get_domain_size(&self) -> usize {
        self.domain_size
    }
    fn get_number_points(&self) -> usize {
        self.number_points
    }
}

pub struct DummyMpDpf<V>
where
    V: Add<Output = V> + Copy + Debug + Eq + Zero,
{
    domain_size: usize,
    number_points: usize,
    phantom: PhantomData<V>,
}

impl<V> MultiPointDpf for DummyMpDpf<V>
where
    V: Add<Output = V> + Copy + Debug + Eq + Zero,
{
    type Key = DummyMpDpfKey<V>;
    type Value = V;

    fn new(domain_size: usize, number_points: usize) -> Self {
        Self {
            domain_size,
            number_points,
            phantom: PhantomData,
        }
    }

    fn get_domain_size(&self) -> usize {
        self.domain_size
    }

    fn get_number_points(&self) -> usize {
        self.number_points
    }

    fn generate_keys(&self, alphas: &[u64], betas: &[V]) -> (Self::Key, Self::Key) {
        assert_eq!(
            alphas.len(),
            self.number_points,
            "number of points does not match constructor argument"
        );
        assert_eq!(
            alphas.len(),
            betas.len(),
            "alphas and betas must be the same size"
        );
        assert!(
            alphas
                .iter()
                .all(|&alpha| alpha < (self.domain_size as u64)),
            "all alphas must be in the domain"
        );
        assert!(alphas.windows(2).all(|w| w[0] <= w[1]));
        (
            DummyMpDpfKey {
                party_id: 0,
                domain_size: self.domain_size,
                number_points: self.number_points,
                alphas: alphas.iter().copied().collect(),
                betas: betas.iter().copied().collect(),
            },
            DummyMpDpfKey {
                party_id: 1,
                domain_size: self.domain_size,
                number_points: self.number_points,
                alphas: alphas.iter().copied().collect(),
                betas: betas.iter().copied().collect(),
            },
        )
    }

    fn evaluate_at(&self, key: &Self::Key, index: u64) -> V {
        assert_eq!(self.domain_size, key.domain_size);
        assert_eq!(self.number_points, key.number_points);
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
    H: HashFunction<u16>,
{
    party_id: usize,
    domain_size: usize,
    number_points: usize,
    spdpf_keys: Vec<Option<SPDPF::Key>>,
    cuckoo_parameters: CuckooParameters<H, u16>,
}

impl<SPDPF, H> Debug for SmartMpDpfKey<SPDPF, H>
where
    SPDPF: SinglePointDpf,
    H: HashFunction<u16>,
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
            "{}domain_size: {:?},{}",
            indentation, self.domain_size, newline
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
    H: HashFunction<u16>,
{
    fn clone(&self) -> Self {
        Self {
            party_id: self.party_id,
            domain_size: self.domain_size,
            number_points: self.number_points,
            spdpf_keys: self.spdpf_keys.clone(),
            cuckoo_parameters: self.cuckoo_parameters.clone(),
        }
    }
}

impl<SPDPF, H> MultiPointDpfKey for SmartMpDpfKey<SPDPF, H>
where
    SPDPF: SinglePointDpf,
    H: HashFunction<u16>,
{
    fn get_party_id(&self) -> usize {
        self.party_id
    }
    fn get_domain_size(&self) -> usize {
        self.domain_size
    }
    fn get_number_points(&self) -> usize {
        self.number_points
    }
}

struct SmartMpDpfPrecomputationData<H: HashFunction<u16>> {
    pub cuckoo_parameters: CuckooParameters<H, u16>,
    pub hasher: CuckooHasher<H, u16>,
    pub hashes: [Vec<u16>; CUCKOO_NUMBER_HASH_FUNCTIONS],
    pub simple_htable: Vec<Vec<u64>>,
    pub bucket_sizes: Vec<usize>,
    pub position_map_lookup_table: Vec<[(usize, usize); 3]>,
}

pub struct SmartMpDpf<V, SPDPF, H>
where
    V: Add<Output = V> + AddAssign + Copy + Debug + Eq + Zero,
    SPDPF: SinglePointDpf<Value = V>,
    H: HashFunction<u16>,
{
    domain_size: usize,
    number_points: usize,
    precomputation_data: Option<SmartMpDpfPrecomputationData<H>>,
    phantom_v: PhantomData<V>,
    phantom_s: PhantomData<SPDPF>,
    phantom_h: PhantomData<H>,
}

impl<V, SPDPF, H> SmartMpDpf<V, SPDPF, H>
where
    V: Add<Output = V> + AddAssign + Copy + Debug + Eq + Zero,
    SPDPF: SinglePointDpf<Value = V>,
    H: HashFunction<u16>,
{
    fn precompute_hashes(
        domain_size: usize,
        number_points: usize,
    ) -> SmartMpDpfPrecomputationData<H> {
        let seed: [u8; 32] = [
            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
            42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
        ];
        let cuckoo_parameters = CuckooParameters::from_seed(number_points, seed);
        assert!(
            cuckoo_parameters.get_number_buckets() < (1 << u16::BITS),
            "too many buckets, use larger type for hash values"
        );
        let hasher = CuckooHasher::<H, u16>::new(cuckoo_parameters);
        let hashes = hasher.hash_domain(domain_size as u64);
        let simple_htable =
            hasher.hash_domain_into_buckets_given_hashes(domain_size as u64, &hashes);
        let bucket_sizes = CuckooHasher::<H, u16>::compute_bucket_sizes(&simple_htable);
        let position_map_lookup_table =
            CuckooHasher::<H, u16>::compute_pos_lookup_table(domain_size as u64, &simple_htable);
        SmartMpDpfPrecomputationData {
            cuckoo_parameters,
            hasher,
            hashes,
            simple_htable,
            bucket_sizes,
            position_map_lookup_table,
        }
    }
}

impl<V, SPDPF, H> MultiPointDpf for SmartMpDpf<V, SPDPF, H>
where
    V: Add<Output = V> + AddAssign + Copy + Debug + Eq + Zero,
    SPDPF: SinglePointDpf<Value = V>,
    H: HashFunction<u16>,
{
    type Key = SmartMpDpfKey<SPDPF, H>;
    type Value = V;

    fn new(domain_size: usize, number_points: usize) -> Self {
        assert!(domain_size < (1 << u32::BITS));
        Self {
            domain_size,
            number_points,
            precomputation_data: None,
            phantom_v: PhantomData,
            phantom_s: PhantomData,
            phantom_h: PhantomData,
        }
    }

    fn get_domain_size(&self) -> usize {
        self.domain_size
    }

    fn get_number_points(&self) -> usize {
        self.domain_size
    }

    fn precompute(&mut self) {
        self.precomputation_data = Some(Self::precompute_hashes(
            self.domain_size,
            self.number_points,
        ));
    }

    fn generate_keys(&self, alphas: &[u64], betas: &[Self::Value]) -> (Self::Key, Self::Key) {
        assert_eq!(alphas.len(), betas.len());
        debug_assert!(alphas.windows(2).all(|w| w[0] < w[1]));
        debug_assert!(alphas.iter().all(|&alpha| alpha < self.domain_size as u64));
        let number_points = alphas.len();

        // if not data is precomputed, do it now
        let mut precomputation_data_fresh: Option<SmartMpDpfPrecomputationData<H>> = None;
        if self.precomputation_data.is_none() {
            precomputation_data_fresh = Some(Self::precompute_hashes(
                self.domain_size,
                self.number_points,
            ));
        }
        // select either the precomputed or the freshly computed data
        let precomputation_data = self
            .precomputation_data
            .as_ref()
            .unwrap_or_else(|| precomputation_data_fresh.as_ref().unwrap());
        let cuckoo_parameters = &precomputation_data.cuckoo_parameters;
        let hasher = &precomputation_data.hasher;
        let (cuckoo_table_items, cuckoo_table_indices) = hasher.cuckoo_hash_items(alphas);
        let position_map_lookup_table = &precomputation_data.position_map_lookup_table;
        let pos = |bucket_i: usize, item: u64| -> u64 {
            CuckooHasher::<H, u16>::pos_lookup(position_map_lookup_table, bucket_i, item)
        };

        let number_buckets = hasher.get_parameters().get_number_buckets();
        let bucket_sizes = &precomputation_data.bucket_sizes;

        let mut keys_0 = Vec::<Option<SPDPF::Key>>::with_capacity(number_buckets);
        let mut keys_1 = Vec::<Option<SPDPF::Key>>::with_capacity(number_buckets);

        for bucket_i in 0..number_buckets {
            // if bucket is empty, add invalid dummy keys to the arrays to make the
            // indices work
            if bucket_sizes[bucket_i] == 0 {
                keys_0.push(None);
                keys_1.push(None);
                continue;
            }

            let (alpha, beta) =
                if cuckoo_table_items[bucket_i] != CuckooHasher::<H, u16>::UNOCCUPIED {
                    let alpha = pos(bucket_i, cuckoo_table_items[bucket_i]);
                    let beta = betas[cuckoo_table_indices[bucket_i]];
                    (alpha, beta)
                } else {
                    (0, V::zero())
                };
            let (key_0, key_1) = SPDPF::generate_keys(bucket_sizes[bucket_i], alpha, beta);
            keys_0.push(Some(key_0));
            keys_1.push(Some(key_1));
        }

        (
            SmartMpDpfKey::<SPDPF, H> {
                party_id: 0,
                domain_size: self.domain_size,
                number_points,
                spdpf_keys: keys_0,
                cuckoo_parameters: cuckoo_parameters.clone(),
            },
            SmartMpDpfKey::<SPDPF, H> {
                party_id: 1,
                domain_size: self.domain_size,
                number_points,
                spdpf_keys: keys_1,
                cuckoo_parameters: cuckoo_parameters.clone(),
            },
        )
    }

    fn evaluate_at(&self, key: &Self::Key, index: u64) -> Self::Value {
        assert_eq!(self.domain_size, key.domain_size);
        assert_eq!(self.number_points, key.number_points);
        assert_eq!(key.domain_size, self.domain_size);
        assert!(index < self.domain_size as u64);

        let hasher = CuckooHasher::<H, u16>::new(key.cuckoo_parameters);

        let hashes = hasher.hash_items(&[index]);
        let simple_htable = hasher.hash_domain_into_buckets(self.domain_size as u64);

        let pos = |bucket_i: usize, item: u64| -> u64 {
            let idx = simple_htable[bucket_i].partition_point(|x| x < &item);
            debug_assert!(idx != simple_htable[bucket_i].len());
            debug_assert_eq!(item, simple_htable[bucket_i][idx]);
            debug_assert!(idx == 0 || simple_htable[bucket_i][idx - 1] != item);
            idx as u64
        };
        let mut output = {
            let hash = H::hash_value_as_usize(hashes[0][0]);
            debug_assert!(key.spdpf_keys[hash].is_some());
            let sp_key = key.spdpf_keys[hash].as_ref().unwrap();
            debug_assert_eq!(simple_htable[hash][pos(hash, index) as usize], index);
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
            let hash = H::hash_value_as_usize(hashes[j][0]);
            debug_assert!(key.spdpf_keys[hash].is_some());
            let sp_key = key.spdpf_keys[hash].as_ref().unwrap();
            debug_assert_eq!(simple_htable[hash][pos(hash, index) as usize], index);
            output += SPDPF::evaluate_at(&sp_key, pos(hash, index));
        }

        output
    }

    fn evaluate_domain(&self, key: &Self::Key) -> Vec<Self::Value> {
        assert_eq!(self.domain_size, key.domain_size);
        assert_eq!(self.number_points, key.number_points);
        let domain_size = self.domain_size as u64;

        // if not data is precomputed, do it now
        let mut precomputation_data_fresh: Option<SmartMpDpfPrecomputationData<H>> = None;
        if self.precomputation_data.is_none() {
            precomputation_data_fresh = Some(Self::precompute_hashes(
                self.domain_size,
                self.number_points,
            ));
        }
        // select either the precomputed or the freshly computed data
        let precomputation_data = self
            .precomputation_data
            .as_ref()
            .unwrap_or_else(|| precomputation_data_fresh.as_ref().unwrap());
        let hashes = &precomputation_data.hashes;
        let simple_htable = &precomputation_data.simple_htable;
        let position_map_lookup_table = &precomputation_data.position_map_lookup_table;
        let pos = |bucket_i: usize, item: u64| -> u64 {
            CuckooHasher::<H, u16>::pos_lookup(position_map_lookup_table, bucket_i, item)
        };

        let mut outputs = Vec::<Self::Value>::with_capacity(domain_size as usize);

        let sp_dpf_full_domain_evaluations: Vec<Vec<V>> = key
            .spdpf_keys
            .iter()
            .map(|sp_key_opt| {
                sp_key_opt
                    .as_ref()
                    .map_or(vec![], |sp_key| SPDPF::evaluate_domain(&sp_key))
            })
            .collect();

        let spdpf_evaluate_at =
            |hash: usize, index| sp_dpf_full_domain_evaluations[hash][pos(hash, index) as usize];

        for index in 0..domain_size {
            outputs.push({
                let hash = H::hash_value_as_usize(hashes[0][index as usize]);
                debug_assert!(key.spdpf_keys[hash].is_some());
                debug_assert_eq!(simple_htable[hash][pos(hash, index) as usize], index);
                spdpf_evaluate_at(hash, index)
            });

            // prevent adding the same term multiple times when we have collisions
            let mut hash_bit_map = [0u8; 2];
            if hashes[0][index as usize] != hashes[1][index as usize] {
                hash_bit_map[0] = 1;
            }
            if hashes[0][index as usize] != hashes[2][index as usize]
                && hashes[1][index as usize] != hashes[2][index as usize]
            {
                hash_bit_map[1] = 1;
            }

            for j in 1..CUCKOO_NUMBER_HASH_FUNCTIONS {
                if hash_bit_map[j - 1] == 0 {
                    continue;
                }
                let hash = H::hash_value_as_usize(hashes[j][index as usize]);
                debug_assert!(key.spdpf_keys[hash].is_some());
                debug_assert_eq!(simple_htable[hash][pos(hash, index) as usize], index);
                outputs[index as usize] += spdpf_evaluate_at(hash, index)
            }
        }

        outputs
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

    fn test_mpdpf_with_param<MPDPF: MultiPointDpf>(
        log_domain_size: u32,
        number_points: usize,
        precomputation: bool,
    ) where
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
        let mut mpdpf = MPDPF::new(domain_size as usize, number_points);
        if precomputation {
            mpdpf.precompute();
        }
        let (key_0, key_1) = mpdpf.generate_keys(&alphas, &betas);

        let out_0 = mpdpf.evaluate_domain(&key_0);
        let out_1 = mpdpf.evaluate_domain(&key_1);
        for i in 0..domain_size {
            let value = mpdpf.evaluate_at(&key_0, i) + mpdpf.evaluate_at(&key_1, i);
            assert_eq!(value, out_0[i as usize] + out_1[i as usize]);
            let expected_result = match alphas.binary_search(&i) {
                Ok(i) => betas[i],
                Err(_) => MPDPF::Value::zero(),
            };
            assert_eq!(value, expected_result, "wrong value at index {}", i);
        }
    }

    #[test]
    fn test_dummy_mpdpf() {
        type Value = Wrapping<u64>;
        for log_domain_size in 5..10 {
            for log_number_points in 0..5 {
                test_mpdpf_with_param::<DummyMpDpf<Value>>(
                    log_domain_size,
                    1 << log_number_points,
                    false,
                );
            }
        }
    }

    #[test]
    fn test_smart_mpdpf() {
        type Value = Wrapping<u64>;
        for log_domain_size in 5..7 {
            for log_number_points in 0..5 {
                for precomputation in [false, true] {
                    test_mpdpf_with_param::<
                        SmartMpDpf<Value, DummySpDpf<Value>, AesHashFunction<u16>>,
                    >(log_domain_size, 1 << log_number_points, precomputation);
                }
            }
        }
    }
}
