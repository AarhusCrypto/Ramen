use crate::fixed_key_aes::FixedKeyAes;
use bincode;
use blake3;
use ff::{Field, PrimeField};
use num;
use rand::{thread_rng, Rng};

#[allow(non_upper_case_globals)]
pub const p: u128 = 340282366920938462946865773367900766209;

/// Prime field with modulus
/// p = 340282366920938462946865773367900766209.
#[derive(PrimeField, bincode::Encode, bincode::Decode)]
#[PrimeFieldModulus = "340282366920938462946865773367900766209"]
#[PrimeFieldGenerator = "7"]
#[PrimeFieldReprEndianness = "little"]
pub struct Fp([u64; 3]);

impl num::traits::Zero for Fp {
    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }
}

pub trait FromPrf {
    type PrfKey: Copy;
    /// PRF key generation
    fn prf_key_gen() -> Self::PrfKey;
    /// PRF into Fp
    fn prf(key: &Self::PrfKey, input: u64) -> Self;
    /// PRF into vector of Fp
    fn prf_vector(key: &Self::PrfKey, input: u64, size: usize) -> Vec<Self>
    where
        Self: Sized;
}

pub trait FromPrg {
    fn expand(input: u128) -> Self;
    fn expand_bytes(input: &[u8]) -> Self;
}

pub trait Modulus128 {
    /// Modulus of the prime field
    const MOD: u128;
}
impl Modulus128 for Fp {
    const MOD: u128 = p;
}

pub trait FromHash {
    /// Hash into Fp
    fn hash(input: u64) -> Self;
    fn hash_bytes(input: &[u8]) -> Self;
}

pub trait LegendreSymbol: PrimeField {
    /// Return an arbitrary QNR.
    fn get_non_random_qnr() -> Self;
    /// Compute the Legendre Symbol (p/a)
    fn legendre_symbol(a: Self) -> Self;
}

impl LegendreSymbol for Fp {
    // (p-1)/ 2 = 0b11111111111111111111111111111111111111111111111111111111111 00 1
    // 00000000000000000000000000000000000000000000000000000000000000000
    // (59x '1', 2x '9', 1x '1', 65x '0')

    /// 7 is not a square mod p.
    fn get_non_random_qnr() -> Self {
        Self::from_u128(7)
    }

    /// Compute the Legendre Symbol (p/a)
    fn legendre_symbol(a: Self) -> Self {
        // handle 65x even
        let mut x = a;
        for _ in 0..65 {
            x = x.square();
        }

        // handle 1x odd
        let mut y = x;
        x = x.square();

        // handle 2x even
        x = x.square();
        x = x.square();

        // handle 59x odd
        for _ in 0..58 {
            y = x * y;
            x = x.square();
        }
        let z = x * y;

        assert!(
            (z == -Fp::ONE || z == Fp::ONE || z == Fp::ZERO) && (z != Fp::ZERO || a == Fp::ZERO)
        );

        z
    }
}

impl Fp {
    fn from_xof(xof: &mut blake3::OutputReader) -> Self {
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

    pub fn to_le_bytes(&self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        let repr = self.to_repr();
        debug_assert_eq!(&repr.as_ref()[16..], &[0u8; 8]);
        bytes.copy_from_slice(&repr.as_ref()[0..16]);
        bytes
    }

    pub fn from_le_bytes_vartime(bytes: &[u8; 16]) -> Option<Self> {
        let mut repr = <Self as PrimeField>::Repr::default();
        debug_assert_eq!(repr.as_ref(), &[0u8; 24]);
        repr.as_mut()[0..16].copy_from_slice(bytes);
        Self::from_repr_vartime(repr)
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
        let mut xof = hasher.finalize_xof();
        Self::from_xof(&mut xof)
    }

    /// PRF into vector of Fp
    fn prf_vector(key: &Self::PrfKey, input: u64, size: usize) -> Vec<Self> {
        let mut hasher = blake3::Hasher::new_keyed(&key);
        hasher.update(&input.to_be_bytes());
        let mut xof = hasher.finalize_xof();
        (0..size).map(|_| Self::from_xof(&mut xof)).collect()
    }
}

impl FromPrg for Fp {
    fn expand(input: u128) -> Self {
        Self::expand_bytes(&input.to_be_bytes())
    }

    fn expand_bytes(input: &[u8]) -> Self {
        assert_eq!(input.len(), 16);
        // not really "fixed-key"
        let aes = FixedKeyAes::new(input.try_into().unwrap());
        let mut i = 0;
        loop {
            let val = aes.pi(i);
            if val < Fp::MOD {
                return Fp::from_u128(val);
            }
            i += 1;
        }
    }
}

impl FromHash for Fp {
    /// Hash into Fp
    fn hash(input: u64) -> Self {
        Self::hash_bytes(&input.to_be_bytes())
    }

    fn hash_bytes(input: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(input);
        let mut xof = hasher.finalize_xof();
        Self::from_xof(&mut xof)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lagrange_symbol() {
        const INPUTS: [u128; 20] = [
            0,
            1,
            2,
            35122421919063474048031845924062067909,
            61212839083344548205786527436063227216,
            108886203898319005744174078164860101674,
            112160854746794802432264095652132979488,
            142714630766673706362679167860844911107,
            144328356835331043954321695814395383527,
            149714699338443771695584577213555322897,
            162837698983268132975860461485836731565,
            185920817468766357617527011469055960606,
            207479253861772423381237297118907360324,
            220976947578297059190439898234224764278,
            225624737240143795963751467909724695007,
            230022448309092634504744292546382561960,
            284649339713848098295138218361935151979,
            293856596737329296797721884860187281734,
            315840344961299616831836711745928570660,
            340282366920938462946865773367900766208,
        ];
        const OUTPUTS: [u128; 20] = [
            0,
            1,
            1,
            1,
            1,
            1,
            340282366920938462946865773367900766208,
            1,
            1,
            340282366920938462946865773367900766208,
            1,
            340282366920938462946865773367900766208,
            1,
            1,
            1,
            340282366920938462946865773367900766208,
            340282366920938462946865773367900766208,
            1,
            1,
            1,
        ];
        for (&x, &y) in INPUTS.iter().zip(OUTPUTS.iter()) {
            assert_eq!(Fp::legendre_symbol(Fp::from_u128(x)), Fp::from_u128(y));
        }
        assert_eq!(Fp::legendre_symbol(Fp::get_non_random_qnr()), -Fp::ONE);
    }

    #[test]
    fn test_serialization() {
        for _ in 0..100 {
            let x = Fp::random(thread_rng());
            let x_bytes = bincode::encode_to_vec(x, bincode::config::standard()).unwrap();
            let (y, bytes_read): (Fp, usize) =
                bincode::decode_from_slice(&x_bytes, bincode::config::standard()).unwrap();
            assert_eq!(bytes_read, x_bytes.len());
            assert_eq!(y, x);
        }
    }

    #[test]
    fn test_to_bytes() {
        for _ in 0..100 {
            let x = Fp::random(thread_rng());
            let x_bytes = x.to_le_bytes();
            let y = Fp::from_le_bytes_vartime(&x_bytes).expect("from_le_bytes_vartime failed");
            assert_eq!(x, y);
        }
    }
}
