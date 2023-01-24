use bitvec::{slice::BitSlice, vec::BitVec};
use core::marker::PhantomData;
use itertools::izip;
use rand::{thread_rng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaChaRng;
use std::iter::repeat;
use utils::field::{FromLimbs, FromPrf, LegendreSymbol, Modulus128};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegendrePrfKey<F: LegendreSymbol> {
    pub keys: Vec<F>,
}

impl<F: LegendreSymbol> LegendrePrfKey<F> {
    pub fn get_output_bitsize(&self) -> usize {
        self.keys.len()
    }
}

/// Legendre PRF: F x F -> F
pub struct LegendrePrf<F> {
    _phantom: PhantomData<F>,
}

impl<F: LegendreSymbol> LegendrePrf<F> {
    pub fn key_gen(output_bitsize: usize) -> LegendrePrfKey<F> {
        LegendrePrfKey {
            keys: (0..output_bitsize)
                .map(|_| F::random(thread_rng()))
                .collect(),
        }
    }

    pub fn eval(key: &LegendrePrfKey<F>, input: F) -> BitVec {
        let mut output = BitVec::with_capacity(key.keys.len());
        for &k in key.keys.iter() {
            let ls = F::legendre_symbol(k + input);
            assert!(ls != F::ZERO, "unlikely");
            output.push(ls == F::ONE);
        }
        output
    }
}

struct SharedPrf<F: FromPrf> {
    key: F::PrfKey,
    counter: u64,
}

impl<F: FromPrf> SharedPrf<F> {
    pub fn key_gen() -> Self {
        Self {
            key: F::prf_key_gen(),
            counter: 0,
        }
    }

    pub fn from_key(key: F::PrfKey) -> Self {
        Self { key, counter: 0 }
    }

    pub fn get_key(&self) -> F::PrfKey {
        self.key
    }

    pub fn eval(&mut self) -> F {
        let output = F::prf(&self.key, self.counter);
        self.counter += 1;
        output
    }
}

pub struct DOPrfParty1<F: LegendreSymbol + FromPrf> {
    _phantom: PhantomData<F>,
    output_bitsize: usize,
    shared_prf_1_2: Option<SharedPrf<F>>,
    shared_prf_1_3: Option<SharedPrf<F>>,
    legendre_prf_key: Option<LegendrePrfKey<F>>,
    is_initialized: bool,
    num_preprocessed_invocations: usize,
    preprocessed_squares: Vec<F>,
    preprocessed_mt_c1: Vec<F>,
}

impl<F> DOPrfParty1<F>
where
    F: LegendreSymbol + FromPrf,
{
    pub fn new(output_bitsize: usize) -> Self {
        assert!(output_bitsize > 0);
        Self {
            _phantom: PhantomData,
            output_bitsize,
            shared_prf_1_2: None,
            shared_prf_1_3: None,
            legendre_prf_key: None,
            is_initialized: false,
            num_preprocessed_invocations: 0,
            preprocessed_squares: Default::default(),
            preprocessed_mt_c1: Default::default(),
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.output_bitsize)
    }

    pub fn reset_preprocessing(&mut self) {
        self.num_preprocessed_invocations = 0;
        self.preprocessed_squares = Default::default();
        self.preprocessed_mt_c1 = Default::default();
    }

    pub fn init_round_0(&mut self) -> (F::PrfKey, ()) {
        assert!(!self.is_initialized);
        // sample and share a PRF key with Party 2
        self.shared_prf_1_2 = Some(SharedPrf::key_gen());
        (self.shared_prf_1_2.as_ref().unwrap().get_key(), ())
    }

    pub fn init_round_1(&mut self, _: (), shared_prf_key_1_3: F::PrfKey) {
        assert!(!self.is_initialized);
        // receive shared PRF key from Party 3
        self.shared_prf_1_3 = Some(SharedPrf::from_key(shared_prf_key_1_3));
        // generate Legendre PRF key
        self.legendre_prf_key = Some(LegendrePrf::key_gen(self.output_bitsize));
        self.is_initialized = true;
    }

    pub fn get_legendre_prf_key(&self) -> LegendrePrfKey<F> {
        assert!(self.is_initialized);
        self.legendre_prf_key.as_ref().unwrap().clone()
    }

    pub fn preprocess_round_0(&mut self, num: usize) -> ((), ()) {
        assert!(self.is_initialized);
        let n = num * self.output_bitsize;
        self.preprocessed_squares
            .extend((0..n).map(|_| self.shared_prf_1_2.as_mut().unwrap().eval().square()));
        ((), ())
    }

    pub fn preprocess_round_1(&mut self, num: usize, preprocessed_mt_c1: Vec<F>, _: ()) {
        assert!(self.is_initialized);
        let n = num * self.output_bitsize;
        assert_eq!(preprocessed_mt_c1.len(), n);
        self.preprocessed_mt_c1.extend(preprocessed_mt_c1);
        self.num_preprocessed_invocations += num;
    }

    pub fn get_num_preprocessed_invocations(&self) -> usize {
        self.num_preprocessed_invocations
    }

    pub fn get_preprocessed_data(&self) -> (&[F], &[F]) {
        (&self.preprocessed_squares, &self.preprocessed_mt_c1)
    }

    pub fn check_preprocessing(&self) {
        let num = self.num_preprocessed_invocations;
        let n = num * self.output_bitsize;
        assert_eq!(self.preprocessed_squares.len(), n);
        assert_eq!(self.preprocessed_mt_c1.len(), n);
    }

    pub fn eval_round_1(
        &mut self,
        num: usize,
        shares1: &[F],
        masked_shares2: &[F],
        mult_e: &[F],
    ) -> ((), Vec<F>) {
        assert!(num <= self.num_preprocessed_invocations);
        let n = num * self.output_bitsize;
        assert_eq!(shares1.len(), num);
        assert_eq!(masked_shares2.len(), num);
        assert_eq!(mult_e.len(), num);
        let k = &self.legendre_prf_key.as_ref().unwrap().keys;
        assert_eq!(k.len(), self.output_bitsize);
        let output_shares_z1: Vec<F> = izip!(
            shares1
                .iter()
                .flat_map(|s1i| repeat(s1i).take(self.output_bitsize)),
            masked_shares2
                .iter()
                .flat_map(|ms2i| repeat(ms2i).take(self.output_bitsize)),
            k.iter().cycle(),
            self.preprocessed_squares.drain(0..n),
            self.preprocessed_mt_c1.drain(0..n),
            mult_e
                .iter()
                .flat_map(|e| repeat(e).take(self.output_bitsize)),
        )
        .map(|(&s1_i, &ms2_i, &k_j, sq_ij, c1_ij, &e_ij)| {
            sq_ij * (k_j + s1_i + ms2_i) + e_ij * sq_ij + c1_ij
        })
        .collect();
        self.num_preprocessed_invocations -= num;
        ((), output_shares_z1)
    }
}

pub struct DOPrfParty2<F: LegendreSymbol + FromPrf> {
    _phantom: PhantomData<F>,
    output_bitsize: usize,
    shared_prf_1_2: Option<SharedPrf<F>>,
    shared_prf_2_3: Option<SharedPrf<F>>,
    is_initialized: bool,
    num_preprocessed_invocations: usize,
    preprocessed_rerand_m2: Vec<F>,
}

impl<F> DOPrfParty2<F>
where
    F: LegendreSymbol + FromPrf,
{
    pub fn new(output_bitsize: usize) -> Self {
        assert!(output_bitsize > 0);
        Self {
            _phantom: PhantomData,
            output_bitsize,
            shared_prf_1_2: None,
            shared_prf_2_3: None,
            is_initialized: false,
            num_preprocessed_invocations: 0,
            preprocessed_rerand_m2: Default::default(),
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.output_bitsize)
    }

    pub fn reset_preprocessing(&mut self) {
        self.num_preprocessed_invocations = 0;
        self.preprocessed_rerand_m2 = Default::default();
    }

    pub fn init_round_0(&mut self) -> ((), F::PrfKey) {
        assert!(!self.is_initialized);
        self.shared_prf_2_3 = Some(SharedPrf::key_gen());
        ((), self.shared_prf_2_3.as_ref().unwrap().get_key())
    }

    pub fn init_round_1(&mut self, shared_prf_key_1_2: F::PrfKey, _: ()) {
        assert!(!self.is_initialized);
        // receive shared PRF key from Party 1
        self.shared_prf_1_2 = Some(SharedPrf::from_key(shared_prf_key_1_2));
        self.is_initialized = true;
    }

    pub fn preprocess_round_0(&mut self, num: usize) -> (Vec<F>, ()) {
        assert!(self.is_initialized);
        let n = num * self.output_bitsize;

        let preprocessed_squares: Vec<F> = (0..n)
            .map(|_| self.shared_prf_1_2.as_mut().unwrap().eval().square())
            .collect();
        self.preprocessed_rerand_m2
            .extend((0..num).map(|_| self.shared_prf_2_3.as_mut().unwrap().eval()));
        let preprocessed_mult_d: Vec<F> = (0..n)
            .map(|_| self.shared_prf_2_3.as_mut().unwrap().eval())
            .collect();
        let preprocessed_mt_b: Vec<F> = (0..num)
            .map(|_| self.shared_prf_2_3.as_mut().unwrap().eval())
            .collect();
        let preprocessed_mt_c3: Vec<F> = (0..n)
            .map(|_| self.shared_prf_2_3.as_mut().unwrap().eval())
            .collect();
        let preprocessed_c1: Vec<F> = izip!(
            preprocessed_squares.iter(),
            preprocessed_mult_d.iter(),
            preprocessed_mt_b
                .iter()
                .flat_map(|b| repeat(b).take(self.output_bitsize)),
            preprocessed_mt_c3.iter(),
        )
        .map(|(&s, &d, &b, &c3)| (s - d) * b - c3)
        .collect();
        self.num_preprocessed_invocations += num;
        (preprocessed_c1, ())
    }

    pub fn preprocess_round_1(&mut self, _: usize, _: (), _: ()) {
        assert!(self.is_initialized);
    }

    pub fn get_num_preprocessed_invocations(&self) -> usize {
        self.num_preprocessed_invocations
    }

    pub fn get_preprocessed_data(&self) -> &[F] {
        &self.preprocessed_rerand_m2
    }

    pub fn check_preprocessing(&self) {
        let num = self.num_preprocessed_invocations;
        assert_eq!(self.preprocessed_rerand_m2.len(), num);
    }

    pub fn eval_round_0(&mut self, num: usize, shares2: &[F]) -> (Vec<F>, ()) {
        assert!(num <= self.num_preprocessed_invocations);
        assert_eq!(shares2.len(), num);
        let masked_shares2: Vec<F> =
            izip!(shares2.iter(), self.preprocessed_rerand_m2.drain(0..num),)
                .map(|(&s2i, m2i)| s2i + m2i)
                .collect();
        self.num_preprocessed_invocations -= num;
        (masked_shares2, ())
    }
}

pub struct DOPrfParty3<F: LegendreSymbol + FromPrf> {
    _phantom: PhantomData<F>,
    output_bitsize: usize,
    shared_prf_1_3: Option<SharedPrf<F>>,
    shared_prf_2_3: Option<SharedPrf<F>>,
    is_initialized: bool,
    num_preprocessed_invocations: usize,
    preprocessed_rerand_m3: Vec<F>,
    preprocessed_mt_b: Vec<F>,
    preprocessed_mt_c3: Vec<F>,
    preprocessed_mult_d: Vec<F>,
    mult_e: Vec<F>,
}

impl<F> DOPrfParty3<F>
where
    F: LegendreSymbol + FromPrf + FromLimbs + Modulus128,
{
    pub fn new(output_bitsize: usize) -> Self {
        assert!(output_bitsize > 0);
        Self {
            _phantom: PhantomData,
            output_bitsize,
            shared_prf_1_3: None,
            shared_prf_2_3: None,
            is_initialized: false,
            num_preprocessed_invocations: 0,
            preprocessed_rerand_m3: Default::default(),
            preprocessed_mt_b: Default::default(),
            preprocessed_mt_c3: Default::default(),
            preprocessed_mult_d: Default::default(),
            mult_e: Default::default(),
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.output_bitsize)
    }

    pub fn reset_preprocessing(&mut self) {
        self.num_preprocessed_invocations = 0;
        self.preprocessed_rerand_m3 = Default::default();
        self.preprocessed_mt_b = Default::default();
        self.preprocessed_mt_c3 = Default::default();
        self.preprocessed_mult_d = Default::default();
        self.mult_e = Default::default();
    }

    pub fn init_round_0(&mut self) -> (F::PrfKey, ()) {
        assert!(!self.is_initialized);
        self.shared_prf_1_3 = Some(SharedPrf::key_gen());
        (self.shared_prf_1_3.as_ref().unwrap().get_key(), ())
    }

    pub fn init_round_1(&mut self, _: (), shared_prf_key_2_3: F::PrfKey) {
        self.shared_prf_2_3 = Some(SharedPrf::from_key(shared_prf_key_2_3));
        self.is_initialized = true;
    }

    pub fn preprocess_round_0(&mut self, num: usize) -> ((), ()) {
        assert!(self.is_initialized);
        let n = num * self.output_bitsize;

        self.preprocessed_rerand_m3
            .extend((0..num).map(|_| -self.shared_prf_2_3.as_mut().unwrap().eval()));
        self.preprocessed_mult_d
            .extend((0..n).map(|_| self.shared_prf_2_3.as_mut().unwrap().eval()));
        self.preprocessed_mt_b
            .extend((0..num).map(|_| self.shared_prf_2_3.as_mut().unwrap().eval()));
        self.preprocessed_mt_c3
            .extend((0..n).map(|_| self.shared_prf_2_3.as_mut().unwrap().eval()));
        ((), ())
    }

    pub fn preprocess_round_1(&mut self, num: usize, _: (), _: ()) {
        assert!(self.is_initialized);
        self.num_preprocessed_invocations += num;
    }

    pub fn get_num_preprocessed_invocations(&self) -> usize {
        self.num_preprocessed_invocations
    }

    pub fn get_preprocessed_data(&self) -> (&[F], &[F], &[F], &[F]) {
        (
            &self.preprocessed_rerand_m3,
            &self.preprocessed_mt_b,
            &self.preprocessed_mt_c3,
            &self.preprocessed_mult_d,
        )
    }

    pub fn check_preprocessing(&self) {
        let num = self.num_preprocessed_invocations;
        let n = num * self.output_bitsize;
        assert_eq!(self.preprocessed_rerand_m3.len(), num);
        assert_eq!(self.preprocessed_mt_b.len(), num);
        assert_eq!(self.preprocessed_mt_c3.len(), n);
        assert_eq!(self.preprocessed_mult_d.len(), n);
    }

    pub fn eval_round_0(&mut self, num: usize, shares3: &[F]) -> (Vec<F>, ()) {
        assert!(num <= self.num_preprocessed_invocations);
        assert_eq!(shares3.len(), num);
        self.mult_e = izip!(
            shares3.iter(),
            &self.preprocessed_rerand_m3[0..num],
            self.preprocessed_mt_b.drain(0..num),
        )
        .map(|(&s3_i, m3_i, b_i)| s3_i + m3_i - b_i)
        .collect();
        (self.mult_e.clone(), ())
    }

    pub fn eval_round_2(
        &mut self,
        num: usize,
        shares3: &[F],
        output_shares_z1: Vec<F>,
        _: (),
    ) -> Vec<BitVec> {
        assert!(num <= self.num_preprocessed_invocations);
        let n = num * self.output_bitsize;
        assert_eq!(shares3.len(), num);
        assert_eq!(output_shares_z1.len(), n);
        let lprf_inputs: Vec<F> = izip!(
            shares3
                .iter()
                .flat_map(|s3| repeat(s3).take(self.output_bitsize)),
            self.preprocessed_rerand_m3
                .drain(0..num)
                .flat_map(|m3| repeat(m3).take(self.output_bitsize)),
            self.preprocessed_mult_d.drain(0..n),
            self.mult_e
                .drain(0..num)
                .flat_map(|e| repeat(e).take(self.output_bitsize)),
            self.preprocessed_mt_c3.drain(0..n),
            output_shares_z1.iter(),
        )
        .map(|(&s3_i, m3_i, d_ij, e_i, c3_ij, &z1_ij)| {
            d_ij * (s3_i + m3_i) + c3_ij + z1_ij - d_ij * e_i
        })
        .collect();
        assert_eq!(lprf_inputs.len(), n);
        let output: Vec<BitVec> = lprf_inputs
            .chunks_exact(self.output_bitsize)
            .map(|chunk| {
                let mut bv = BitVec::with_capacity(self.output_bitsize);
                for &x in chunk.iter() {
                    let ls = F::legendre_symbol(x);
                    debug_assert!(ls != F::ZERO, "unlikely");
                    bv.push(ls == F::ONE);
                }
                bv
            })
            .collect();
        self.num_preprocessed_invocations -= num;
        output
    }
}

pub struct MaskedDOPrfParty1<F: LegendreSymbol + FromPrf> {
    _phantom: PhantomData<F>,
    output_bitsize: usize,
    shared_prf_1_2: Option<SharedPrf<F>>,
    shared_prf_1_3: Option<SharedPrf<F>>,
    legendre_prf_key: Option<LegendrePrfKey<F>>,
    is_initialized: bool,
    num_preprocessed_invocations: usize,
    preprocessed_rerand_m1: Vec<F>,
    preprocessed_mt_a: Vec<F>,
    preprocessed_mt_c1: Vec<F>,
    preprocessed_mult_e: Vec<F>,
    mult_d: Vec<F>,
}

impl<F> MaskedDOPrfParty1<F>
where
    F: LegendreSymbol + FromPrf,
{
    pub fn new(output_bitsize: usize) -> Self {
        assert!(output_bitsize > 0);
        Self {
            _phantom: PhantomData,
            output_bitsize,
            shared_prf_1_2: None,
            shared_prf_1_3: None,
            legendre_prf_key: None,
            is_initialized: false,
            num_preprocessed_invocations: 0,
            preprocessed_rerand_m1: Default::default(),
            preprocessed_mt_a: Default::default(),
            preprocessed_mt_c1: Default::default(),
            preprocessed_mult_e: Default::default(),
            mult_d: Default::default(),
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.output_bitsize)
    }

    pub fn reset_preprocessing(&mut self) {
        self.num_preprocessed_invocations = 0;
        self.preprocessed_rerand_m1 = Default::default();
        self.preprocessed_mt_a = Default::default();
        self.preprocessed_mt_c1 = Default::default();
        self.preprocessed_mult_e = Default::default();
    }

    pub fn init_round_0(&mut self) -> (F::PrfKey, ()) {
        assert!(!self.is_initialized);
        // sample and share a PRF key with Party 2
        self.shared_prf_1_2 = Some(SharedPrf::key_gen());
        (self.shared_prf_1_2.as_ref().unwrap().get_key(), ())
    }

    pub fn init_round_1(&mut self, _: (), shared_prf_key_1_3: F::PrfKey) {
        assert!(!self.is_initialized);
        // receive shared PRF key from Party 3
        self.shared_prf_1_3 = Some(SharedPrf::from_key(shared_prf_key_1_3));
        // generate Legendre PRF key
        self.legendre_prf_key = Some(LegendrePrf::key_gen(self.output_bitsize));
        self.is_initialized = true;
    }

    pub fn get_legendre_prf_key(&self) -> LegendrePrfKey<F> {
        assert!(self.is_initialized);
        self.legendre_prf_key.as_ref().unwrap().clone()
    }

    pub fn preprocess_round_0(&mut self, num: usize) -> ((), ()) {
        assert!(self.is_initialized);
        let n = num * self.output_bitsize;
        self.preprocessed_rerand_m1
            .extend((0..num).map(|_| self.shared_prf_1_2.as_mut().unwrap().eval()));
        self.preprocessed_mt_a
            .extend((0..n).map(|_| self.shared_prf_1_2.as_mut().unwrap().eval()));
        self.preprocessed_mt_c1
            .extend((0..n).map(|_| self.shared_prf_1_2.as_mut().unwrap().eval()));
        self.preprocessed_mult_e
            .extend((0..n).map(|_| self.shared_prf_1_2.as_mut().unwrap().eval()));
        ((), ())
    }

    pub fn preprocess_round_1(&mut self, num: usize, _: (), _: ()) {
        assert!(self.is_initialized);
        self.num_preprocessed_invocations += num;
    }

    pub fn get_num_preprocessed_invocations(&self) -> usize {
        self.num_preprocessed_invocations
    }

    pub fn get_preprocessed_data(&self) -> (&[F], &[F], &[F], &[F]) {
        (
            &self.preprocessed_rerand_m1,
            &self.preprocessed_mt_a,
            &self.preprocessed_mt_c1,
            &self.preprocessed_mult_e,
        )
    }

    pub fn check_preprocessing(&self) {
        let num = self.num_preprocessed_invocations;
        let n = num * self.output_bitsize;
        assert_eq!(self.preprocessed_rerand_m1.len(), num);
        assert_eq!(self.preprocessed_mt_a.len(), n);
        assert_eq!(self.preprocessed_mt_c1.len(), n);
        assert_eq!(self.preprocessed_mult_e.len(), n);
    }

    pub fn eval_round_0(&mut self, num: usize, shares1: &[F]) -> ((), Vec<F>) {
        assert!(num <= self.num_preprocessed_invocations);
        assert_eq!(shares1.len(), num);
        let n = num * self.output_bitsize;
        let k = &self.legendre_prf_key.as_ref().unwrap().keys;
        self.mult_d = izip!(
            k.iter().cycle(),
            shares1
                .iter()
                .flat_map(|s1| repeat(s1).take(self.output_bitsize)),
            self.preprocessed_rerand_m1
                .iter()
                .take(num)
                .flat_map(|m1| repeat(m1).take(self.output_bitsize)),
            self.preprocessed_mt_a.drain(0..n),
        )
        .map(|(&k_i, &s1_i, m1_i, a_i)| k_i + s1_i + m1_i - a_i)
        .collect();
        assert_eq!(self.mult_d.len(), n);
        ((), self.mult_d.clone())
    }

    pub fn eval_round_2(
        &mut self,
        num: usize,
        shares1: &[F],
        _: (),
        output_shares_z3: Vec<F>,
    ) -> Vec<BitVec> {
        assert!(num <= self.num_preprocessed_invocations);
        let n = num * self.output_bitsize;
        assert_eq!(shares1.len(), num);
        assert_eq!(output_shares_z3.len(), n);
        let k = &self.legendre_prf_key.as_ref().unwrap().keys;
        let lprf_inputs: Vec<F> = izip!(
            k.iter().cycle(),
            shares1
                .iter()
                .flat_map(|s1| repeat(s1).take(self.output_bitsize)),
            self.preprocessed_rerand_m1
                .drain(0..num)
                .flat_map(|m1| repeat(m1).take(self.output_bitsize)),
            self.preprocessed_mult_e.drain(0..n),
            self.mult_d.drain(..),
            self.preprocessed_mt_c1.drain(0..n),
            output_shares_z3.iter(),
        )
        .map(|(&k_j, &s1_i, m1_i, e_ij, d_ij, c1_ij, &z3_ij)| {
            e_ij * (k_j + s1_i + m1_i) + c1_ij + z3_ij - d_ij * e_ij
        })
        .collect();
        assert_eq!(lprf_inputs.len(), n);
        let output: Vec<BitVec> = lprf_inputs
            .chunks_exact(self.output_bitsize)
            .map(|chunk| {
                let mut bv = BitVec::with_capacity(self.output_bitsize);
                for &x in chunk.iter() {
                    let ls = F::legendre_symbol(x);
                    debug_assert!(ls != F::ZERO, "unlikely");
                    bv.push(ls == F::ONE);
                }
                bv
            })
            .collect();
        self.num_preprocessed_invocations -= num;
        output
    }
}

pub struct MaskedDOPrfParty2<F: LegendreSymbol + FromPrf> {
    _phantom: PhantomData<F>,
    output_bitsize: usize,
    shared_prf_1_2: Option<SharedPrf<F>>,
    shared_prf_2_3: Option<SharedPrf<F>>,
    shared_prg_2_3: Option<ChaChaRng>,
    is_initialized: bool,
    num_preprocessed_invocations: usize,
    preprocessed_rerand_m2: Vec<F>,
    preprocessed_r: BitVec,
}

impl<F> MaskedDOPrfParty2<F>
where
    F: LegendreSymbol + FromPrf,
{
    pub fn new(output_bitsize: usize) -> Self {
        assert!(output_bitsize > 0);
        Self {
            _phantom: PhantomData,
            output_bitsize,
            shared_prf_1_2: None,
            shared_prf_2_3: None,
            shared_prg_2_3: None,
            is_initialized: false,
            num_preprocessed_invocations: 0,
            preprocessed_rerand_m2: Default::default(),
            preprocessed_r: Default::default(),
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.output_bitsize)
    }

    pub fn reset_preprocessing(&mut self) {
        self.num_preprocessed_invocations = 0;
        self.preprocessed_rerand_m2 = Default::default();
    }

    pub fn init_round_0(&mut self) -> ((), (F::PrfKey, <ChaChaRng as SeedableRng>::Seed)) {
        assert!(!self.is_initialized);
        self.shared_prf_2_3 = Some(SharedPrf::key_gen());
        self.shared_prg_2_3 = Some(ChaChaRng::from_seed(thread_rng().gen()));
        (
            (),
            (
                self.shared_prf_2_3.as_ref().unwrap().get_key(),
                self.shared_prg_2_3.as_ref().unwrap().get_seed(),
            ),
        )
    }

    pub fn init_round_1(&mut self, shared_prf_key_1_2: F::PrfKey, _: ()) {
        assert!(!self.is_initialized);
        // receive shared PRF key from Party 1
        self.shared_prf_1_2 = Some(SharedPrf::from_key(shared_prf_key_1_2));
        self.is_initialized = true;
    }

    pub fn preprocess_round_0(&mut self, num: usize) -> ((), Vec<F>) {
        assert!(self.is_initialized);
        let n = num * self.output_bitsize;

        let mut preprocessed_t: Vec<_> = (0..n)
            .map(|_| self.shared_prf_2_3.as_mut().unwrap().eval().square())
            .collect();
        debug_assert!(!preprocessed_t.contains(&F::ZERO));
        {
            let mut random_bytes = vec![0u8; (n + 7) / 8];
            self.shared_prg_2_3
                .as_mut()
                .unwrap()
                .fill_bytes(&mut random_bytes);
            let new_r_slice = BitSlice::<u8>::from_slice(&random_bytes);
            self.preprocessed_r.extend(&new_r_slice[..n]);
            for (i, r_i) in new_r_slice.iter().by_vals().take(n).enumerate() {
                if r_i {
                    preprocessed_t[i] *= F::get_non_random_qnr();
                }
            }
        }
        self.preprocessed_rerand_m2
            .extend((0..num).map(|_| -self.shared_prf_1_2.as_mut().unwrap().eval()));
        let preprocessed_mt_a: Vec<F> = (0..n)
            .map(|_| self.shared_prf_1_2.as_mut().unwrap().eval())
            .collect();
        let preprocessed_mt_c1: Vec<F> = (0..n)
            .map(|_| self.shared_prf_1_2.as_mut().unwrap().eval())
            .collect();
        let preprocessed_mult_e: Vec<F> = (0..n)
            .map(|_| self.shared_prf_1_2.as_mut().unwrap().eval())
            .collect();
        let preprocessed_c3: Vec<F> = izip!(
            preprocessed_t.iter(),
            preprocessed_mult_e.iter(),
            preprocessed_mt_a.iter(),
            preprocessed_mt_c1.iter(),
        )
        .map(|(&t, &e, &a, &c1)| a * (t - e) - c1)
        .collect();
        self.num_preprocessed_invocations += num;
        ((), preprocessed_c3)
    }

    pub fn preprocess_round_1(&mut self, _: usize, _: (), _: ()) {
        assert!(self.is_initialized);
    }

    pub fn get_num_preprocessed_invocations(&self) -> usize {
        self.num_preprocessed_invocations
    }

    pub fn get_preprocessed_data(&self) -> (&BitSlice, &[F]) {
        (&self.preprocessed_r, &self.preprocessed_rerand_m2)
    }

    pub fn check_preprocessing(&self) {
        let num = self.num_preprocessed_invocations;
        assert_eq!(self.preprocessed_rerand_m2.len(), num);
    }

    pub fn eval_round_0(&mut self, num: usize, shares2: &[F]) -> ((), Vec<F>) {
        assert!(num <= self.num_preprocessed_invocations);
        assert_eq!(shares2.len(), num);
        let masked_shares2: Vec<F> =
            izip!(shares2.iter(), self.preprocessed_rerand_m2.drain(0..num),)
                .map(|(&s2i, m2i)| s2i + m2i)
                .collect();
        assert_eq!(masked_shares2.len(), num);
        ((), masked_shares2)
    }

    pub fn eval_get_output(&mut self, num: usize) -> Vec<BitVec> {
        assert!(num <= self.num_preprocessed_invocations);
        let n = num * self.output_bitsize;
        let mut output = Vec::with_capacity(num);
        for chunk in self
            .preprocessed_r
            .chunks_exact(self.output_bitsize)
            .take(num)
        {
            output.push(chunk.to_bitvec());
        }
        let (_, last_r) = self.preprocessed_r.split_at(n);
        self.preprocessed_r = last_r.to_bitvec();
        self.num_preprocessed_invocations -= num;
        output
    }
}

pub struct MaskedDOPrfParty3<F: LegendreSymbol + FromPrf> {
    _phantom: PhantomData<F>,
    output_bitsize: usize,
    shared_prf_1_3: Option<SharedPrf<F>>,
    shared_prf_2_3: Option<SharedPrf<F>>,
    shared_prg_2_3: Option<ChaChaRng>,
    is_initialized: bool,
    num_preprocessed_invocations: usize,
    preprocessed_r: BitVec,
    preprocessed_t: Vec<F>,
    preprocessed_mt_c3: Vec<F>,
}

impl<F> MaskedDOPrfParty3<F>
where
    F: LegendreSymbol + FromPrf + FromLimbs + Modulus128,
{
    pub fn new(output_bitsize: usize) -> Self {
        assert!(output_bitsize > 0);
        Self {
            _phantom: PhantomData,
            output_bitsize,
            shared_prf_1_3: None,
            shared_prf_2_3: None,
            shared_prg_2_3: None,
            is_initialized: false,
            num_preprocessed_invocations: 0,
            preprocessed_r: Default::default(),
            preprocessed_t: Default::default(),
            preprocessed_mt_c3: Default::default(),
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.output_bitsize)
    }

    pub fn reset_preprocessing(&mut self) {
        self.num_preprocessed_invocations = 0;
        self.preprocessed_t = Default::default();
        self.preprocessed_mt_c3 = Default::default();
    }

    pub fn init_round_0(&mut self) -> (F::PrfKey, ()) {
        assert!(!self.is_initialized);
        self.shared_prf_1_3 = Some(SharedPrf::key_gen());
        (self.shared_prf_1_3.as_ref().unwrap().get_key(), ())
    }

    pub fn init_round_1(
        &mut self,
        _: (),
        (shared_prf_key_2_3, shared_prg_seed_2_3): (F::PrfKey, <ChaChaRng as SeedableRng>::Seed),
    ) {
        self.shared_prf_2_3 = Some(SharedPrf::from_key(shared_prf_key_2_3));
        self.shared_prg_2_3 = Some(ChaChaRng::from_seed(shared_prg_seed_2_3));
        self.is_initialized = true;
    }

    pub fn preprocess_round_0(&mut self, num: usize) -> ((), ()) {
        assert!(self.is_initialized);
        let n = num * self.output_bitsize;
        let start_index = self.num_preprocessed_invocations * self.output_bitsize;

        self.preprocessed_t
            .extend((0..n).map(|_| self.shared_prf_2_3.as_mut().unwrap().eval().square()));
        debug_assert!(!self.preprocessed_t[start_index..].contains(&F::ZERO));
        {
            let mut random_bytes = vec![0u8; (n + 7) / 8];
            self.shared_prg_2_3
                .as_mut()
                .unwrap()
                .fill_bytes(&mut random_bytes);
            let new_r_slice = BitSlice::<u8>::from_slice(&random_bytes);
            self.preprocessed_r.extend(&new_r_slice[..n]);
            for (i, r_i) in new_r_slice.iter().by_vals().take(n).enumerate() {
                if r_i {
                    self.preprocessed_t[start_index + i] *= F::get_non_random_qnr();
                }
            }
        }
        ((), ())
    }

    pub fn preprocess_round_1(&mut self, num: usize, _: (), preprocessed_mt_c3: Vec<F>) {
        assert!(self.is_initialized);
        let n = num * self.output_bitsize;
        assert_eq!(preprocessed_mt_c3.len(), n);
        self.preprocessed_mt_c3.extend(preprocessed_mt_c3);
        self.num_preprocessed_invocations += num;
    }

    pub fn get_num_preprocessed_invocations(&self) -> usize {
        self.num_preprocessed_invocations
    }

    pub fn get_preprocessed_data(&self) -> (&BitSlice, &[F], &[F]) {
        (
            &self.preprocessed_r,
            &self.preprocessed_t,
            &self.preprocessed_mt_c3,
        )
    }

    pub fn check_preprocessing(&self) {
        let num = self.num_preprocessed_invocations;
        let n = num * self.output_bitsize;
        assert_eq!(self.preprocessed_t.len(), n);
        assert_eq!(self.preprocessed_mt_c3.len(), n);
    }

    pub fn eval_round_1(
        &mut self,
        num: usize,
        shares3: &[F],
        mult_d: &[F],
        masked_shares2: &[F],
    ) -> (Vec<F>, ()) {
        assert!(num <= self.num_preprocessed_invocations);
        let n = num * self.output_bitsize;
        assert_eq!(shares3.len(), num);
        assert_eq!(masked_shares2.len(), num);
        assert_eq!(mult_d.len(), n);
        let output_shares_z3: Vec<F> = izip!(
            shares3
                .iter()
                .flat_map(|s1i| repeat(s1i).take(self.output_bitsize)),
            masked_shares2
                .iter()
                .flat_map(|ms2i| repeat(ms2i).take(self.output_bitsize)),
            self.preprocessed_t.drain(0..n),
            self.preprocessed_mt_c3.drain(0..n),
            mult_d,
        )
        .map(|(&s3_i, &ms2_i, t_ij, c3_ij, &d_ij)| t_ij * (s3_i + ms2_i) + d_ij * t_ij + c3_ij)
        .collect();
        (output_shares_z3, ())
    }

    pub fn eval_get_output(&mut self, num: usize) -> Vec<BitVec> {
        assert!(num <= self.num_preprocessed_invocations);
        let n = num * self.output_bitsize;
        let mut output = Vec::with_capacity(num);
        for chunk in self
            .preprocessed_r
            .chunks_exact(self.output_bitsize)
            .take(num)
        {
            output.push(chunk.to_bitvec());
        }
        let (_, last_r) = self.preprocessed_r.split_at(n);
        self.preprocessed_r = last_r.to_bitvec();
        self.num_preprocessed_invocations -= num;
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::Field;
    use utils::field::Fp;

    #[test]
    fn test_doprf() {
        let output_bitsize = 42;

        let mut party_1 = DOPrfParty1::<Fp>::new(output_bitsize);
        let mut party_2 = DOPrfParty2::<Fp>::new(output_bitsize);
        let mut party_3 = DOPrfParty3::<Fp>::new(output_bitsize);

        let (msg_1_2, msg_1_3) = party_1.init_round_0();
        let (msg_2_1, msg_2_3) = party_2.init_round_0();
        let (msg_3_1, msg_3_2) = party_3.init_round_0();
        party_1.init_round_1(msg_2_1, msg_3_1);
        party_2.init_round_1(msg_1_2, msg_3_2);
        party_3.init_round_1(msg_1_3, msg_2_3);

        // preprocess num invocations
        let num = 20;

        let (msg_1_2, msg_1_3) = party_1.preprocess_round_0(num);
        let (msg_2_1, msg_2_3) = party_2.preprocess_round_0(num);
        let (msg_3_1, msg_3_2) = party_3.preprocess_round_0(num);
        party_1.preprocess_round_1(num, msg_2_1, msg_3_1);
        party_2.preprocess_round_1(num, msg_1_2, msg_3_2);
        party_3.preprocess_round_1(num, msg_1_3, msg_2_3);

        assert_eq!(party_1.get_num_preprocessed_invocations(), num);
        assert_eq!(party_2.get_num_preprocessed_invocations(), num);
        assert_eq!(party_3.get_num_preprocessed_invocations(), num);

        party_1.check_preprocessing();
        party_2.check_preprocessing();
        party_3.check_preprocessing();

        // preprocess another n invocations
        let (msg_1_2, msg_1_3) = party_1.preprocess_round_0(num);
        let (msg_2_1, msg_2_3) = party_2.preprocess_round_0(num);
        let (msg_3_1, msg_3_2) = party_3.preprocess_round_0(num);
        party_1.preprocess_round_1(num, msg_2_1, msg_3_1);
        party_2.preprocess_round_1(num, msg_1_2, msg_3_2);
        party_3.preprocess_round_1(num, msg_1_3, msg_2_3);

        let num = 2 * num;

        assert_eq!(party_1.get_num_preprocessed_invocations(), num);
        assert_eq!(party_2.get_num_preprocessed_invocations(), num);
        assert_eq!(party_3.get_num_preprocessed_invocations(), num);

        party_1.check_preprocessing();
        party_2.check_preprocessing();
        party_3.check_preprocessing();

        // verify preprocessed data
        {
            let n = num * output_bitsize;
            let (squares, mt_c1) = party_1.get_preprocessed_data();
            let rerand_m2 = party_2.get_preprocessed_data();
            let (rerand_m3, mt_b, mt_c3, mult_d) = party_3.get_preprocessed_data();

            assert_eq!(squares.len(), n);
            assert!(squares.iter().all(|&x| Fp::legendre_symbol(x) == Fp::ONE));

            assert_eq!(rerand_m2.len(), num);
            assert_eq!(rerand_m3.len(), num);
            assert!(izip!(rerand_m2.iter(), rerand_m3.iter()).all(|(&m2, &m3)| m2 + m3 == Fp::ZERO));

            let mt_a: Vec<Fp> = squares
                .iter()
                .zip(mult_d.iter())
                .map(|(&s, &d)| s - d)
                .collect();
            assert_eq!(mult_d.len(), n);
            // assert!(
            //     izip!(squares.iter(), mt_a.iter(), mult_d.iter()).all(|(&s, &a, &d)| d == s - a)
            // );

            assert_eq!(mt_a.len(), n);
            assert_eq!(mt_b.len(), num);
            assert_eq!(mt_c1.len(), n);
            assert_eq!(mt_c3.len(), n);
            let mut triple_it = izip!(
                mt_a.iter(),
                mt_b.iter().flat_map(|b| repeat(b).take(output_bitsize)),
                mt_c1.iter(),
                mt_c3.iter()
            );
            assert_eq!(triple_it.clone().count(), n);
            assert!(triple_it.all(|(&a, &b, &c1, &c3)| a * b == c1 + c3));
        }

        // perform n evaluations
        let num = 15;

        let shares_1: Vec<Fp> = (0..num).map(|_| Fp::random(thread_rng())).collect();
        let shares_2: Vec<Fp> = (0..num).map(|_| Fp::random(thread_rng())).collect();
        let shares_3: Vec<Fp> = (0..num).map(|_| Fp::random(thread_rng())).collect();

        let (msg_2_1, msg_2_3) = party_2.eval_round_0(num, &shares_2);
        let (msg_3_1, _) = party_3.eval_round_0(num, &shares_3);
        let (_, msg_1_3) = party_1.eval_round_1(num, &shares_1, &msg_2_1, &msg_3_1);
        let output = party_3.eval_round_2(num, &shares_3, msg_1_3, msg_2_3);

        assert_eq!(party_1.get_num_preprocessed_invocations(), 25);
        assert_eq!(party_2.get_num_preprocessed_invocations(), 25);
        assert_eq!(party_3.get_num_preprocessed_invocations(), 25);
        party_1.check_preprocessing();
        party_2.check_preprocessing();
        party_3.check_preprocessing();

        assert_eq!(output.len(), num);
        assert!(output.iter().all(|bv| bv.len() == output_bitsize));

        // check that the output matches the non-distributed version
        let legendre_prf_key = party_1.get_legendre_prf_key();
        for i in 0..num {
            let input_i = shares_1[i] + shares_2[i] + shares_3[i];
            let output_i = LegendrePrf::<Fp>::eval(&legendre_prf_key, input_i);
            assert_eq!(output[i], output_i);
        }
    }

    #[test]
    fn test_masked_doprf() {
        let output_bitsize = 42;

        let mut party_1 = MaskedDOPrfParty1::<Fp>::new(output_bitsize);
        let mut party_2 = MaskedDOPrfParty2::<Fp>::new(output_bitsize);
        let mut party_3 = MaskedDOPrfParty3::<Fp>::new(output_bitsize);

        let (msg_1_2, msg_1_3) = party_1.init_round_0();
        let (msg_2_1, msg_2_3) = party_2.init_round_0();
        let (msg_3_1, msg_3_2) = party_3.init_round_0();
        party_1.init_round_1(msg_2_1, msg_3_1);
        party_2.init_round_1(msg_1_2, msg_3_2);
        party_3.init_round_1(msg_1_3, msg_2_3);

        // preprocess num invocations
        let num = 20;

        let (msg_1_2, msg_1_3) = party_1.preprocess_round_0(num);
        let (msg_2_1, msg_2_3) = party_2.preprocess_round_0(num);
        let (msg_3_1, msg_3_2) = party_3.preprocess_round_0(num);
        party_1.preprocess_round_1(num, msg_2_1, msg_3_1);
        party_2.preprocess_round_1(num, msg_1_2, msg_3_2);
        party_3.preprocess_round_1(num, msg_1_3, msg_2_3);

        assert_eq!(party_1.get_num_preprocessed_invocations(), num);
        assert_eq!(party_2.get_num_preprocessed_invocations(), num);
        assert_eq!(party_3.get_num_preprocessed_invocations(), num);

        party_1.check_preprocessing();
        party_2.check_preprocessing();
        party_3.check_preprocessing();

        // preprocess another n invocations
        let (msg_1_2, msg_1_3) = party_1.preprocess_round_0(num);
        let (msg_2_1, msg_2_3) = party_2.preprocess_round_0(num);
        let (msg_3_1, msg_3_2) = party_3.preprocess_round_0(num);
        party_1.preprocess_round_1(num, msg_2_1, msg_3_1);
        party_2.preprocess_round_1(num, msg_1_2, msg_3_2);
        party_3.preprocess_round_1(num, msg_1_3, msg_2_3);

        let num = 2 * num;

        assert_eq!(party_1.get_num_preprocessed_invocations(), num);
        assert_eq!(party_2.get_num_preprocessed_invocations(), num);
        assert_eq!(party_3.get_num_preprocessed_invocations(), num);

        party_1.check_preprocessing();
        party_2.check_preprocessing();
        party_3.check_preprocessing();

        // verify preprocessed data
        {
            let n = num * output_bitsize;
            let (rerand_m1, mt_a, mt_c1, mult_e) = party_1.get_preprocessed_data();
            let (r2, rerand_m2) = party_2.get_preprocessed_data();
            let (r3, ts, mt_c3) = party_3.get_preprocessed_data();

            assert_eq!(r2.len(), n);
            assert_eq!(r2, r3);
            assert_eq!(ts.len(), n);
            assert!(r2.iter().by_vals().zip(ts.iter()).all(|(r_i, &t_i)| {
                if r_i {
                    Fp::legendre_symbol(t_i) == -Fp::ONE
                } else {
                    Fp::legendre_symbol(t_i) == Fp::ONE
                }
            }));

            assert_eq!(rerand_m1.len(), num);
            assert_eq!(rerand_m2.len(), num);
            assert!(izip!(rerand_m1.iter(), rerand_m2.iter()).all(|(&m1, &m2)| m1 + m2 == Fp::ZERO));

            let mt_b: Vec<Fp> = ts.iter().zip(mult_e.iter()).map(|(&t, &e)| t - e).collect();
            assert_eq!(mult_e.len(), n);

            assert_eq!(mt_a.len(), n);
            assert_eq!(mt_b.len(), n);
            assert_eq!(mt_c1.len(), n);
            assert_eq!(mt_c3.len(), n);
            let mut triple_it = izip!(mt_a.iter(), mt_b.iter(), mt_c1.iter(), mt_c3.iter());
            assert_eq!(triple_it.clone().count(), n);
            assert!(triple_it.all(|(&a, &b, &c1, &c3)| a * b == c1 + c3));
        }

        // perform n evaluations
        let num = 15;

        let shares_1: Vec<Fp> = (0..num).map(|_| Fp::random(thread_rng())).collect();
        let shares_2: Vec<Fp> = (0..num).map(|_| Fp::random(thread_rng())).collect();
        let shares_3: Vec<Fp> = (0..num).map(|_| Fp::random(thread_rng())).collect();

        let (_, msg_1_3) = party_1.eval_round_0(num, &shares_1);
        let (_, msg_2_3) = party_2.eval_round_0(num, &shares_2);
        let (msg_3_1, _) = party_3.eval_round_1(num, &shares_3, &msg_1_3, &msg_2_3);
        let masked_output = party_1.eval_round_2(num, &shares_1, (), msg_3_1);
        let mask2 = party_2.eval_get_output(num);
        let mask3 = party_3.eval_get_output(num);

        assert_eq!(party_1.get_num_preprocessed_invocations(), 25);
        assert_eq!(party_2.get_num_preprocessed_invocations(), 25);
        assert_eq!(party_3.get_num_preprocessed_invocations(), 25);
        party_1.check_preprocessing();
        party_2.check_preprocessing();
        party_3.check_preprocessing();

        assert_eq!(masked_output.len(), num);
        assert!(masked_output.iter().all(|bv| bv.len() == output_bitsize));
        assert_eq!(mask2.len(), num);
        assert_eq!(mask2, mask3);
        assert!(mask2.iter().all(|bv| bv.len() == output_bitsize));

        // check that the output matches the non-distributed version
        let legendre_prf_key = party_1.get_legendre_prf_key();
        for i in 0..num {
            let input_i = shares_1[i] + shares_2[i] + shares_3[i];
            let expected_output_i = LegendrePrf::<Fp>::eval(&legendre_prf_key, input_i);
            let output_i = masked_output[i].clone() ^ &mask2[i];
            assert_eq!(output_i, expected_output_i);
        }
    }
}
