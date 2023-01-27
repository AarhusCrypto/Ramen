use bincode;
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// Trait that models a random permutation.
pub trait Permutation {
    type Key: Copy;

    fn sample(domain_size: usize) -> Self::Key;
    fn from_key(key: Self::Key) -> Self;
    fn get_key(&self) -> Self::Key;
    fn get_domain_size(&self) -> usize;
    fn permute(&self, x: usize) -> usize;
    // fn inverse(&self, x: usize) -> usize;
    // fn permuted_vector() -> Vec<usize>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, bincode::Encode, bincode::Decode)]
pub struct FisherYatesPermutationKey {
    domain_size: usize,
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

    fn sample(domain_size: usize) -> Self::Key {
        Self::Key {
            domain_size,
            prg_seed: thread_rng().gen(),
        }
    }

    fn from_key(key: Self::Key) -> Self {
        // rng seeded by the key
        let mut rng = ChaCha20Rng::from_seed(key.prg_seed);
        // size of the domain
        let n = key.domain_size;
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

    fn get_domain_size(&self) -> usize {
        self.key.domain_size
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
        let key = Perm::sample(n);
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

    #[test]
    fn test_serialization() {
        for _ in 0..100 {
            let log_domain_size = thread_rng().gen_range(1..30);
            let key = FisherYatesPermutation::sample(log_domain_size);
            let bytes = bincode::encode_to_vec(key, bincode::config::standard()).unwrap();
            let (new_key, bytes_read): (FisherYatesPermutationKey, usize) =
                bincode::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
            assert_eq!(bytes_read, bytes.len());
            assert_eq!(new_key, key);
        }
    }
}
