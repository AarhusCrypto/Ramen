//! Functionality to compute the bit decomposition of integers.

use num::PrimInt;

/// Decompose an integer `x` into a vector of its bits.
pub fn bit_decompose<T: PrimInt, U: From<bool>>(x: T, n_bits: usize) -> Vec<U> {
    assert!(n_bits as u32 == T::zero().count_zeros() || x < T::one() << n_bits);
    (0..n_bits)
        .map(|i| (x & (T::one() << (n_bits - i - 1)) != T::zero()).into())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_decompose() {
        assert_eq!(
            bit_decompose::<u32, u32>(0x42, 8),
            vec![0, 1, 0, 0, 0, 0, 1, 0]
        );
        assert_eq!(
            bit_decompose::<u32, u32>(0x42, 10),
            vec![0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        );
        assert_eq!(
            bit_decompose::<u32, u32>(0x46015ced, 32),
            vec![
                0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0,
                1, 1, 0, 1,
            ]
        );
    }
}
