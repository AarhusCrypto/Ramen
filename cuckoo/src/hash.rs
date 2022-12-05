use core::fmt::Debug;
use core::ops::Range;
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use utils::fixed_key_aes::FixedKeyAes;

pub trait HashFunctionParameters {}

pub trait HashFunction {
    // type Domain;
    // type Range;
    type Description: Copy + Debug;

    /// Sample a random hash function.
    fn sample(range_size: u64) -> Self;

    /// Sample a hash function using a given seed.
    fn from_seed(range_size: u64, seed: [u8; 32]) -> Self;

    fn from_description(description: Self::Description) -> Self;
    fn to_description(&self) -> Self::Description;

    /// Return the number of elements n in the range [0, n).
    fn get_range_size(&self) -> u64;

    /// Hash a single item.
    fn hash_single(&self, item: u64) -> u64;

    /// Hash a slice of items.
    fn hash_slice(&self, items: &[u64]) -> Vec<u64> {
        items.iter().map(|x| self.hash_single(*x)).collect()
    }

    /// Hash a range [a,b) of items.
    fn hash_range(&self, items: Range<u64>) -> Vec<u64> {
        items.map(|x| self.hash_single(x)).collect()
    }
}

/// Fixed-key AES hashing using a circular correlation robust hash function
#[derive(Clone, Debug)]
pub struct AesHashFunction {
    description: AesHashFunctionDescription,
    /// FixedKeyAes object including expanded key.
    aes: FixedKeyAes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AesHashFunctionDescription {
    /// Size of the range.
    range_size: u64,
    /// Raw AES key.
    key: [u8; 16],
}

impl HashFunction for AesHashFunction {
    type Description = AesHashFunctionDescription;

    fn get_range_size(&self) -> u64 {
        self.description.range_size
    }

    fn from_seed(range_size: u64, seed: [u8; 32]) -> Self {
        let mut rng = ChaCha12Rng::from_seed(seed);
        let key = rng.gen();
        Self::from_description(AesHashFunctionDescription { range_size, key })
    }

    fn sample(range_size: u64) -> Self {
        let key: [u8; 16] = thread_rng().gen();
        Self::from_description(AesHashFunctionDescription { range_size, key })
    }

    fn from_description(description: Self::Description) -> Self {
        let aes = FixedKeyAes::new(description.key);
        Self { description, aes }
    }
    fn to_description(&self) -> Self::Description {
        self.description
    }

    fn hash_single(&self, item: u64) -> u64 {
        let h = self.aes.hash_ccr(item as u128);
        (h % self.description.range_size as u128) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_hash_function<H: HashFunction>() {
        let range_size = 42;
        let h = AesHashFunction::sample(range_size);
        let h2 = AesHashFunction::from_description(h.to_description());
        assert_eq!(range_size, h.get_range_size());
        assert_eq!(h.to_description(), h2.to_description());

        for _ in 0..100 {
            let x: u64 = thread_rng().gen();
            let hx = h.hash_single(x);
            assert!(hx < range_size);
            assert_eq!(hx, h2.hash_single(x));
        }

        let a = 1337;
        let b = 1427;
        let vec: Vec<u64> = (a..b).collect();
        let hashes_range = h.hash_range(a..b);
        let hashes_slice = h.hash_slice(vec.as_slice());
        for (i, x) in (a..b).enumerate() {
            assert_eq!(hashes_range[i], h.hash_single(x));
        }
        assert_eq!(hashes_range, hashes_slice);
    }

    #[test]
    fn test_aes_hash_function() {
        test_hash_function::<AesHashFunction>();
    }
}
