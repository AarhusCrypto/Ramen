use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

pub trait Permutation {
    type Key: Copy;

    fn sample(log_domain_size: u32) -> Self::Key;
    fn from_key(key: Self::Key) -> Self;
    fn get_key(&self) -> Self::Key;
    fn get_log_domain_size(&self) -> u32;
    fn permute(&self, x: usize) -> usize;
    // fn inverse(&self, x: usize) -> usize;
    // fn permuted_vector() -> Vec<usize>;
}

#[derive(Clone, Copy, Debug)]
pub struct FisherYatesPermutationKey {
    log_domain_size: u32,
    prg_seed: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct FisherYatesPermutation {
    key: FisherYatesPermutationKey,
    permuted_vector: Vec<usize>,
}

impl Permutation for FisherYatesPermutation {
    type Key = FisherYatesPermutationKey;

    fn sample(log_domain_size: u32) -> Self::Key {
        Self::Key {
            log_domain_size,
            prg_seed: thread_rng().gen(),
        }
    }

    fn from_key(key: Self::Key) -> Self {
        let mut rng = ChaCha20Rng::from_seed(key.prg_seed);
        let mut permuted_vector: Vec<usize> = (0..(1 << key.log_domain_size)).collect();
        // To shuffle an array a of n elements (indices 0..n-1):
        let n = 1 << key.log_domain_size;
        for i in (1..n).rev() {
            let j: usize = rng.gen_range(0..=i);
            permuted_vector.swap(j, i);
        }
        Self {
            key,
            permuted_vector,
        }
    }

    fn get_key(&self) -> Self::Key {
        self.key
    }

    fn get_log_domain_size(&self) -> u32 {
        self.key.log_domain_size
    }

    fn permute(&self, x: usize) -> usize {
        assert!(x < self.permuted_vector.len());
        self.permuted_vector[x]
    }
}
