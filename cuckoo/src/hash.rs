use core::fmt::Debug;
use core::ops::Range;
use funty::Integral;
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use std::marker::PhantomData;
use utils::fixed_key_aes::FixedKeyAes;

pub trait HashFunctionParameters {}
pub trait HashFunctionValue: Integral + TryInto<usize>
where
    <Self as TryInto<usize>>::Error: Debug,
{
}

impl HashFunctionValue for u32 {}
impl HashFunctionValue for u64 {}

pub trait HashFunction<Value: HashFunctionValue>
where
    <Value as TryInto<usize>>::Error: Debug,
{
    // type Domain;

    type Description: Copy + Debug + PartialEq + Eq;

    /// Sample a random hash function.
    fn sample(range_size: usize) -> Self;

    /// Sample a hash function using a given seed.
    fn from_seed(range_size: usize, seed: [u8; 32]) -> Self;

    fn from_description(description: Self::Description) -> Self;
    fn to_description(&self) -> Self::Description;

    /// Return the number of elements n in the range [0, n).
    fn get_range_size(&self) -> usize;

    /// Hash a single item.
    fn hash_single(&self, item: u64) -> Value;

    /// Hash a slice of items.
    fn hash_slice(&self, items: &[u64]) -> Vec<Value> {
        items.iter().map(|x| self.hash_single(*x)).collect()
    }

    /// Hash a range [a,b) of items.
    fn hash_range(&self, items: Range<u64>) -> Vec<Value> {
        items.map(|x| self.hash_single(x)).collect()
    }

    /// Convert a hash value into a usize. Useful when hashes are used as indices.
    /// Might panic if Value is not convertible to usize.
    #[inline(always)]
    fn hash_value_as_usize(value: Value) -> usize
    where
        <Value as TryInto<usize>>::Error: Debug,
    {
        <Value as TryInto<usize>>::try_into(value).unwrap()
    }
}

/// Fixed-key AES hashing using a circular correlation robust hash function
#[derive(Clone, Debug)]
pub struct AesHashFunction<Value> {
    description: AesHashFunctionDescription,
    /// FixedKeyAes object including expanded key.
    aes: FixedKeyAes,
    _phantom: PhantomData<Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AesHashFunctionDescription {
    /// Size of the range.
    range_size: usize,
    /// Raw AES key.
    key: [u8; 16],
}

impl<Value: HashFunctionValue> HashFunction<Value> for AesHashFunction<Value>
where
    <Value as TryInto<usize>>::Error: Debug,
    <Value as TryFrom<u128>>::Error: Debug,
{
    type Description = AesHashFunctionDescription;

    fn get_range_size(&self) -> usize {
        self.description.range_size
    }

    fn from_seed(range_size: usize, seed: [u8; 32]) -> Self {
        let mut rng = ChaCha12Rng::from_seed(seed);
        let key = rng.gen();
        Self::from_description(AesHashFunctionDescription { range_size, key })
    }

    fn sample(range_size: usize) -> Self {
        let key: [u8; 16] = thread_rng().gen();
        Self::from_description(AesHashFunctionDescription { range_size, key })
    }

    fn from_description(description: Self::Description) -> Self {
        let aes = FixedKeyAes::new(description.key);
        Self {
            description,
            aes,
            _phantom: PhantomData,
        }
    }
    fn to_description(&self) -> Self::Description {
        self.description
    }

    fn hash_single(&self, item: u64) -> Value {
        let h = self.aes.hash_ccr(item as u128);
        (h % self.description.range_size as u128)
            .try_into()
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_hash_function<Value: HashFunctionValue, H: HashFunction<Value>>()
    where
        <Value as TryInto<usize>>::Error: Debug,
    {
        let range_size = 42;
        let h = H::sample(range_size);
        let h2 = H::from_description(h.to_description());
        assert_eq!(range_size, h.get_range_size());
        assert_eq!(h.to_description(), h2.to_description());

        for _ in 0..100 {
            let x: u64 = thread_rng().gen();
            let hx = h.hash_single(x);
            assert!(<Value as TryInto<usize>>::try_into(hx).unwrap() < range_size);
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
        test_hash_function::<u32, AesHashFunction<u32>>();
        test_hash_function::<u64, AesHashFunction<u64>>();
    }
}
