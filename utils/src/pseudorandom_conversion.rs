//! Functionality to convert random seeds into a pseudorandom value.
//!
//! Used for the Half-Tree implementation.

use crate::field::FromPrg;
use core::num::Wrapping;

/// Trait describing a functionality to convert a 128 bit random seed into a pseudorandom value of
/// type `T`.
pub trait PRConvertTo<T> {
    /// Convert random seed into an instance of type `T`.
    fn convert(randomness: u128) -> T;
}

/// Helper struct for the pseudorandom conversion.
pub struct PRConverter {}

impl PRConvertTo<u8> for PRConverter {
    fn convert(randomness: u128) -> u8 {
        (randomness & 0xff) as u8
    }
}
impl PRConvertTo<u16> for PRConverter {
    fn convert(randomness: u128) -> u16 {
        (randomness & 0xffff) as u16
    }
}
impl PRConvertTo<u32> for PRConverter {
    fn convert(randomness: u128) -> u32 {
        (randomness & 0xffffffff) as u32
    }
}
impl PRConvertTo<u64> for PRConverter {
    fn convert(randomness: u128) -> u64 {
        (randomness & 0xffffffffffffffff) as u64
    }
}

impl<T> PRConvertTo<Wrapping<T>> for PRConverter
where
    PRConverter: PRConvertTo<T>,
{
    fn convert(randomness: u128) -> Wrapping<T> {
        Wrapping(<Self as PRConvertTo<T>>::convert(randomness))
    }
}

impl<F: FromPrg> PRConvertTo<F> for PRConverter {
    fn convert(randomness: u128) -> F {
        F::expand(randomness)
    }
}
