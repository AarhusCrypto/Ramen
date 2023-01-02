use aes::cipher::crypto_common::Block;
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use rand::{thread_rng, Rng};

/// Fixed-key AES implementation.  Implements the ((tweakable) circular) correlation robust hash
/// functions from Guo et al. eprint 2019/074
#[derive(Clone, Debug)]
pub struct FixedKeyAes {
    /// AES object including expanded key.
    aes: Aes128,
}

impl FixedKeyAes {
    pub fn new(key: [u8; 16]) -> Self {
        Self {
            aes: Aes128::new_from_slice(&key).expect("does not fail since key has the right size"),
        }
    }

    pub fn sample() -> Self {
        let key: [u8; 16] = thread_rng().gen();
        Self::new(key)
    }

    /// Permutation sigma(x) = (x.high64 ^ x.low64, x.high64).
    fn sigma(x: u128) -> u128 {
        let low = x & 0xffffffffffffffff;
        let high = x >> 64;
        ((high ^ low) << 64) | high
    }

    /// Random permutation pi(x) = AES(k, x)
    pub fn pi(&self, x: u128) -> u128 {
        let mut block = Block::<Aes128>::clone_from_slice(&x.to_le_bytes());
        self.aes.encrypt_block(&mut block);
        u128::from_le_bytes(
            block
                .as_slice()
                .try_into()
                .expect("does not fail since block is 16 bytes long"),
        )
    }

    /// MMO function pi(x) ^ x
    pub fn hash_cr(&self, x: u128) -> u128 {
        self.pi(x) ^ x
    }

    /// MMO-hat function pi(sigma(x)) ^ sigma(x)
    pub fn hash_ccr(&self, x: u128) -> u128 {
        let sigma_x = Self::sigma(x);
        self.pi(sigma_x) ^ sigma_x
    }

    /// TMMO function pi(pi(x) ^ i) ^ pi(x)
    pub fn hash_tccr(&self, x: u128, tweak: u128) -> u128 {
        let pi_x = self.pi(x);
        self.pi(pi_x ^ tweak) ^ pi_x
    }
}
