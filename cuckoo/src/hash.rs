use aes::cipher::crypto_common::Block;
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use rand::{thread_rng, Rng};
use std::ops::Range;

pub trait HashFunctionParameters {}

pub trait HashFunction {
    // type Domain;
    // type Range;
    type Description: Copy;

    /// Sample hash function.
    fn sample(range_size: u64) -> Self;

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

/// Fixed-key AES hashing using the MMO construction from  Guo et al.
/// (https://eprint.iacr.org/2019/074.pdf):
///
///   y = AES.Encrypt(key, sigma(x)) ^ sigma(x),
///
/// with sigma(x) = (x.high64 ^ x.low64, x.high64).
#[derive(Clone, Debug)]
pub struct AesHashFunction {
    description: AesHashFunctionDescription,
    /// AES object including expanded key.
    aes: Aes128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AesHashFunctionDescription {
    /// Size of the range.
    range_size: u64,
    /// Raw AES key.
    key: [u8; 16],
}

/// Permutation sigma(x) = (x.high64 ^ x.low64, x.high64).
fn sigma(x: u128) -> u128 {
    let low = x & 0xffffffffffffffff;
    let high = x >> 64;
    ((high ^ low) << 64) | high
}

impl HashFunction for AesHashFunction {
    type Description = AesHashFunctionDescription;

    fn get_range_size(&self) -> u64 {
        self.description.range_size
    }

    fn sample(range_size: u64) -> Self {
        let key: [u8; 16] = thread_rng().gen();
        Self::from_description(AesHashFunctionDescription { range_size, key })
    }

    fn from_description(description: Self::Description) -> Self {
        let aes = Aes128::new_from_slice(&description.key)
            .expect("does not fail since key has the right size");
        Self { description, aes }
    }
    fn to_description(&self) -> Self::Description {
        self.description
    }

    fn hash_single(&self, item: u64) -> u64 {
        let sigma_x = sigma(item as u128).to_le_bytes();
        let mut block = Block::<Aes128>::clone_from_slice(&sigma_x);
        self.aes.encrypt_block(&mut block);
        for (x, y) in block.iter_mut().zip(sigma_x.iter()) {
            *x ^= y;
        }
        let high: &[u8; 8] = block.as_slice()[..8]
            .try_into()
            .expect("does not fail since block is 16 bytes long");
        u64::from_le_bytes(*high) % self.description.range_size
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
