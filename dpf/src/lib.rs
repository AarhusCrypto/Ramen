//! Implementation of distributed point functions (DPFs).
//!
//! A (single-)point function is a function `f` that is specified by two values `(a, b)` such that
//! `f(a) = b` and `f(x) = 0` for all other values `x != 0`.
//! A multi-point function generalizes this concept to `n` points `(a_i, b_i)` for `i = 1, ..., n`,
//! such that `f(a_i) = b_i` and `f(x) = 0` whenever `x` is not one of the `a_i`.
//!
//! A distributed point function (DPF) scheme allows to take the description of a point function
//! `f` and output two keys `k_0, k_1`. These keys can be used with an evaluation algorithm `Eval`
//! to obtain an additive share of `f`'s value such that `Eval(k_0, x) + Eval(k_1, x) = f(x)` for
//! all `x`.

#![warn(missing_docs)]

pub mod mpdpf;
pub mod spdpf;
