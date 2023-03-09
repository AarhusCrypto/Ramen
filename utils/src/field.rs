//! Implementation of the prime field used in Ramen.

#![allow(missing_docs)] // Otherwise, there will be a warning originating from the PrimeField
                        // derive macro ...

use crate::fixed_key_aes::FixedKeyAes;
use bincode;
use blake3;
use ff::{Field, PrimeField};
use num;
use rand::{thread_rng, Rng};
use rug;

/// Prime number `p` defining [`Fp`].
#[allow(non_upper_case_globals)]
pub const p: u128 = 340282366920938462946865773367900766209;

/// Prime field with modulus [`p`].
#[derive(PrimeField)]
#[PrimeFieldModulus = "340282366920938462946865773367900766209"]
#[PrimeFieldGenerator = "7"]
#[PrimeFieldReprEndianness = "little"]
pub struct Fp([u64; 3]);

impl bincode::Encode for Fp {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.to_le_bytes(), encoder)?;
        Ok(())
    }
}

impl bincode::Decode for Fp {
    fn decode<D: bincode::de::Decoder>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        let bytes: [u8; 16] = bincode::Decode::decode(decoder)?;
        Self::from_le_bytes_vartime(&bytes).ok_or_else(|| {
            bincode::error::DecodeError::OtherString(format!(
                "{bytes:?} does not encode a valid Fp element"
            ))
        })
    }
}

impl num::traits::Zero for Fp {
    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }
}

/// Specifies that Self is the range of a PRF.
pub trait FromPrf {
    /// Key type of the PRF.
    type PrfKey: Copy;

    /// PRF key generation.
    fn prf_key_gen() -> Self::PrfKey;

    /// PRF: `[u64] -> Self`.
    fn prf(key: &Self::PrfKey, input: u64) -> Self;

    /// PRF into vector of Self.
    fn prf_vector(key: &Self::PrfKey, input: u64, size: usize) -> Vec<Self>
    where
        Self: Sized;
}

/// Specifies that Self can be obtained from a PRG.
pub trait FromPrg {
    /// Expand a seed given as 128 bit integer.
    fn expand(input: u128) -> Self;

    /// Expand a seed given as byte slice of length 16.
    fn expand_bytes(input: &[u8]) -> Self;
}

/// Trait for prime fields where the modulus can be provided as a 128 bit integer.
pub trait Modulus128 {
    /// Modulus of the prime field
    const MOD: u128;
}

impl Modulus128 for Fp {
    const MOD: u128 = p;
}

/// Specifies that Self can be hashed into.
pub trait FromHash {
    /// Hash a 64 bit integer into Self.
    fn hash(input: u64) -> Self;

    /// Hash a byte slice into Self.
    fn hash_bytes(input: &[u8]) -> Self;
}

/// Definies the Legendre symbol in a prime field.
pub trait LegendreSymbol: PrimeField {
    /// Return an arbitrary QNR.
    fn get_non_random_qnr() -> Self;

    /// Compute the Legendre Symbol (p/a)
    fn legendre_symbol(a: Self) -> i8;
}

/// Simple implementation of the legendre symbol using exponentiation.
pub fn legendre_symbol_exp(a: Fp) -> i8 {
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

    debug_assert!(
        (z == -Fp::ONE || z == Fp::ONE || z == Fp::ZERO) && (z != Fp::ZERO || a == Fp::ZERO)
    );

    if z == Fp::ONE {
        1
    } else if z == -Fp::ONE {
        -1
    } else if z == Fp::ZERO {
        0
    } else {
        panic!("something went wrong during Legendre Symbol computation")
    }
}

/// Faster implementation of the legendre symbol using the `rug` library.
pub fn legendre_symbol_rug(a: Fp) -> i8 {
    let bytes = a.to_le_bytes();
    let a_int = rug::Integer::from_digits(&bytes, rug::integer::Order::LsfLe);
    let p_int = rug::Integer::from(p);
    a_int.legendre(&p_int) as i8
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
    fn legendre_symbol(a: Self) -> i8 {
        legendre_symbol_rug(a)
    }
}

impl Fp {
    fn from_xof(xof: &mut blake3::OutputReader) -> Self {
        assert_eq!(Self::NUM_BITS, 128);
        loop {
            let tmp = {
                let mut repr = [0u64; 3];
                for limb in repr.iter_mut().take(2) {
                    let mut bytes = [0u8; 8];
                    xof.fill(&mut bytes);
                    *limb = u64::from_le_bytes(bytes);
                }
                Self(repr)
            };

            if tmp.is_valid() {
                return tmp;
            }
        }
    }

    /// Convert a field element into 16 bytes using little endian byte order.
    pub fn to_le_bytes(&self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        let repr = self.to_repr();
        debug_assert_eq!(&repr.as_ref()[16..], &[0u8; 8]);
        bytes.copy_from_slice(&repr.as_ref()[0..16]);
        bytes
    }

    /// Create a field element from 16 bytes using little endian byte order.
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
        let mut hasher = blake3::Hasher::new_keyed(key);
        hasher.update(&input.to_be_bytes());
        let mut xof = hasher.finalize_xof();
        Self::from_xof(&mut xof)
    }

    /// PRF into vector of Fp
    fn prf_vector(key: &Self::PrfKey, input: u64, size: usize) -> Vec<Self> {
        let mut hasher = blake3::Hasher::new_keyed(key);
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
    fn test_legendre_symbol() {
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
        const OUTPUTS: [i8; 20] = [
            0, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1,
        ];
        for (&x, &y) in INPUTS.iter().zip(OUTPUTS.iter()) {
            assert_eq!(Fp::legendre_symbol(Fp::from_u128(x)), y);
            assert_eq!(legendre_symbol_exp(Fp::from_u128(x)), y);
            assert_eq!(legendre_symbol_rug(Fp::from_u128(x)), y);
        }
        assert_eq!(Fp::legendre_symbol(Fp::get_non_random_qnr()), -1);
        assert_eq!(legendre_symbol_exp(Fp::get_non_random_qnr()), -1);
        assert_eq!(legendre_symbol_rug(Fp::get_non_random_qnr()), -1);
    }

    #[test]
    fn test_serialization() {
        for _ in 0..100 {
            let x = Fp::random(thread_rng());
            let x_bytes =
                bincode::encode_to_vec(x, bincode::config::standard().skip_fixed_array_length())
                    .unwrap();
            assert_eq!(x_bytes.len(), 16);
            let (y, bytes_read): (Fp, usize) = bincode::decode_from_slice(
                &x_bytes,
                bincode::config::standard().skip_fixed_array_length(),
            )
            .unwrap();
            assert_eq!(bytes_read, x_bytes.len());
            assert_eq!(y, x);
        }
    }

    #[test]
    fn test_to_bytes() {
        for _ in 0..100 {
            let x = Fp::random(thread_rng());
            let x_bytes = x.to_le_bytes();
            assert_eq!(x_bytes.len(), 16);
            let y = Fp::from_le_bytes_vartime(&x_bytes).expect("from_le_bytes_vartime failed");
            assert_eq!(x, y);
        }
    }
}
