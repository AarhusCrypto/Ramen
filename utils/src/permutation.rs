use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// Trait that models a random permutation.
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

/// Random permutation based on a Fisher-Yates shuffle of [0, N) with a seeded PRG.
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
        // rng seeded by the key
        let mut rng = ChaCha20Rng::from_seed(key.prg_seed);
        // size of the domain
        let n = 1 << key.log_domain_size;
        // vector to store permutation explicitly
        let mut permuted_vector: Vec<usize> = (0..n).collect();
        // run Fisher-Yates
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_permutation<Perm: Permutation>(log_domain_size: u32) {
        let n: usize = 1 << log_domain_size;
        let key = Perm::sample(log_domain_size);
        let perm = Perm::from_key(key);
        let mut buffer = vec![0usize; n];
        for i in 0..n {
            buffer[i] = perm.permute(i);
        }
        buffer.sort();
        for i in 0..n {
            assert_eq!(buffer[i], i);
        }
    }

    #[test]
    fn test_all_permutations() {
        let log_domain_size = 10;
        test_permutation::<FisherYatesPermutation>(log_domain_size);
    }
}
