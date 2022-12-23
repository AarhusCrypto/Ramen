use blake3;
use ff::PrimeField;
use rand::{thread_rng, Rng};

/// Prime field with modulus
/// p = 340282366920938462946865773367900766209.
#[derive(PrimeField)]
#[PrimeFieldModulus = "340282366920938462946865773367900766209"]
#[PrimeFieldGenerator = "7"]
#[PrimeFieldReprEndianness = "little"]
pub struct Fp([u64; 3]);

pub trait FromPrf {
    type PrfKey: Copy;
    /// PRF key generation
    fn prf_key_gen() -> Self::PrfKey;
    /// PRF into Fp
    fn prf(key: &Self::PrfKey, input: u64) -> Self;
}

pub trait FromHash {
    /// Hash into Fp
    fn hash(input: u64) -> Self;
}

impl Fp {
    fn from_xof(mut xof: blake3::OutputReader) -> Self {
        assert_eq!(Self::NUM_BITS, 128);
        loop {
            let tmp = {
                let mut repr = [0u64; 3];
                for i in 0..2 {
                    let mut bytes = [0u8; 8];
                    xof.fill(&mut bytes);
                    repr[i] = u64::from_le_bytes(bytes);
                }
                Self(repr)
            };

            if tmp.is_valid() {
                return tmp;
            }
        }
    }
}

impl FromPrf for Fp {
    type PrfKey = [u8; blake3::KEY_LEN];

    /// PRF key generation
    fn prf_key_gen() -> Self::PrfKey {
        thread_rng().gen()
    }

    /// PRF into Fp
    fn prf(key: &Self::PrfKey, input: u64) -> Self {
        let mut hasher = blake3::Hasher::new_keyed(&key);
        hasher.update(&input.to_be_bytes());
        let xof = hasher.finalize_xof();
        Self::from_xof(xof)
    }
}

impl FromHash for Fp {
    /// Hash into Fp
    fn hash(input: u64) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&input.to_be_bytes());
        let xof = hasher.finalize_xof();
        Self::from_xof(xof)
    }
}
