use blake3;
use ff::FromUniformBytes;
use rand::{thread_rng, Rng};

#[derive(Clone, Copy, Debug)]
pub struct PrfKey([u8; blake3::KEY_LEN]);

pub struct Prf {}

impl Prf {
    pub const OUT_LEN: usize = blake3::OUT_LEN;

    pub fn key_gen() -> PrfKey {
        PrfKey(thread_rng().gen())
    }

    pub fn eval<F: FromUniformBytes<{ blake3::OUT_LEN }>>(key: &PrfKey, index: u64) -> F {
        let hash = blake3::keyed_hash(&key.0, &index.to_be_bytes());
        F::from_uniform_bytes(hash.as_bytes())
    }
}
