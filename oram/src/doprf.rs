use crate::common::Error;
use bincode;
use bitvec;
use communicator::{AbstractCommunicator, Fut, Serializable};
use core::marker::PhantomData;
use funty::Unsigned;
use itertools::izip;
use rand::{thread_rng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaChaRng;
use std::iter::repeat;
use utils::field::LegendreSymbol;

pub type BitVec = bitvec::vec::BitVec<u8>;
type BitSlice = bitvec::slice::BitSlice<u8>;

#[derive(Clone, Debug, Eq, PartialEq, bincode::Encode, bincode::Decode)]
pub struct LegendrePrfKey<F: LegendreSymbol> {
    pub keys: Vec<F>,
}

impl<F: LegendreSymbol> LegendrePrfKey<F> {
    pub fn get_output_bitsize(&self) -> usize {
        self.keys.len()
    }
}

/// Legendre PRF: F x F -> {0,1}^k
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

    pub fn eval<'a>(key: &'a LegendrePrfKey<F>, input: F) -> impl Iterator<Item = bool> + 'a {
        key.keys.iter().map(move |&k| {
            let ls = F::legendre_symbol(k + input);
            debug_assert!(ls != 0, "unlikely");
            ls == 1
        })
    }

    pub fn eval_bits(key: &LegendrePrfKey<F>, input: F) -> BitVec {
        let mut output = BitVec::with_capacity(key.keys.len());
        output.extend(Self::eval(key, input));
        output
    }

    pub fn eval_to_uint<T: Unsigned>(key: &LegendrePrfKey<F>, input: F) -> T {
        assert!(key.keys.len() <= T::BITS as usize);
        let mut output = T::ZERO;
        for (i, b) in Self::eval(key, input).enumerate() {
            if b {
                output |= T::ONE << i;
            }
        }
        output
    }
}

fn to_uint<T: Unsigned>(vs: impl IntoIterator<Item = impl IntoIterator<Item = bool>>) -> Vec<T> {
    vs.into_iter()
        .map(|v| {
            let mut output = T::ZERO;
            for (i, b) in v.into_iter().enumerate() {
                if b {
                    output |= T::ONE << i;
                }
            }
            output
        })
        .collect()
}

type SharedSeed = [u8; 32];

pub struct DOPrfParty1<F: LegendreSymbol> {
    _phantom: PhantomData<F>,
    output_bitsize: usize,
    shared_prg_1_2: Option<ChaChaRng>,
    shared_prg_1_3: Option<ChaChaRng>,
    legendre_prf_key: Option<LegendrePrfKey<F>>,
    is_initialized: bool,
    num_preprocessed_invocations: usize,
    preprocessed_squares: Vec<F>,
    preprocessed_mt_c1: Vec<F>,
}

impl<F> DOPrfParty1<F>
where
    F: LegendreSymbol,
{
    pub fn new(output_bitsize: usize) -> Self {
        assert!(output_bitsize > 0);
        Self {
            _phantom: PhantomData,
            output_bitsize,
            shared_prg_1_2: None,
            shared_prg_1_3: None,
            legendre_prf_key: None,
            is_initialized: false,
            num_preprocessed_invocations: 0,
            preprocessed_squares: Default::default(),
            preprocessed_mt_c1: Default::default(),
        }
    }

    pub fn from_legendre_prf_key(legendre_prf_key: LegendrePrfKey<F>) -> Self {
        let mut new = Self::new(legendre_prf_key.keys.len());
        new.legendre_prf_key = Some(legendre_prf_key);
        new
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.output_bitsize)
    }

    pub fn reset_preprocessing(&mut self) {
        self.num_preprocessed_invocations = 0;
        self.preprocessed_squares = Default::default();
        self.preprocessed_mt_c1 = Default::default();
    }

    pub fn init_round_0(&mut self) -> (SharedSeed, ()) {
        assert!(!self.is_initialized);
        // sample and share a PRF key with Party 2
        self.shared_prg_1_2 = Some(ChaChaRng::from_seed(thread_rng().gen()));
        (self.shared_prg_1_2.as_ref().unwrap().get_seed(), ())
    }

    pub fn init_round_1(&mut self, _: (), shared_prg_seed_1_3: SharedSeed) {
        assert!(!self.is_initialized);
        // receive shared PRF key from Party 3
        self.shared_prg_1_3 = Some(ChaChaRng::from_seed(shared_prg_seed_1_3));
        if self.legendre_prf_key.is_none() {
            // generate Legendre PRF key
            self.legendre_prf_key = Some(LegendrePrf::key_gen(self.output_bitsize));
        }
        self.is_initialized = true;
    }

    pub fn init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error> {
        let fut_3_1 = comm.receive_previous()?;
        let (msg_1_2, _) = self.init_round_0();
        comm.send_next(msg_1_2)?;
        self.init_round_1((), fut_3_1.get()?);
        Ok(())
    }

    pub fn get_legendre_prf_key(&self) -> LegendrePrfKey<F> {
        assert!(self.legendre_prf_key.is_some());
        self.legendre_prf_key.as_ref().unwrap().clone()
    }

    pub fn preprocess_round_0(&mut self, num: usize) -> ((), ()) {
        assert!(self.is_initialized);
        let n = num * self.output_bitsize;
        self.preprocessed_squares
            .extend((0..n).map(|_| F::random(self.shared_prg_1_2.as_mut().unwrap()).square()));
        ((), ())
    }

    pub fn preprocess_round_1(&mut self, num: usize, preprocessed_mt_c1: Vec<F>, _: ()) {
        assert!(self.is_initialized);
        let n = num * self.output_bitsize;
        assert_eq!(preprocessed_mt_c1.len(), n);
        self.preprocessed_mt_c1.extend(preprocessed_mt_c1);
        self.num_preprocessed_invocations += num;
    }

    pub fn preprocess<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        num: usize,
    ) -> Result<(), Error>
    where
        F: Serializable,
    {
        let fut_2_1 = comm.receive_next()?;
        self.preprocess_round_0(num);
        self.preprocess_round_1(num, fut_2_1.get()?, ());
        Ok(())
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

    pub fn eval<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        num: usize,
        shares1: &[F],
    ) -> Result<(), Error>
    where
        F: Serializable,
    {
        assert_eq!(shares1.len(), num);
        let fut_2_1 = comm.receive_next::<Vec<_>>()?;
        let fut_3_1 = comm.receive_previous::<Vec<_>>()?;
        let (_, msg_1_3) = self.eval_round_1(num, shares1, &fut_2_1.get()?, &fut_3_1.get()?);
        comm.send_previous(msg_1_3)?;
        Ok(())
    }
}

pub struct DOPrfParty2<F: LegendreSymbol> {
    _phantom: PhantomData<F>,
    output_bitsize: usize,
    shared_prg_1_2: Option<ChaChaRng>,
    shared_prg_2_3: Option<ChaChaRng>,
    is_initialized: bool,
    num_preprocessed_invocations: usize,
    preprocessed_rerand_m2: Vec<F>,
}

impl<F> DOPrfParty2<F>
where
    F: LegendreSymbol,
{
    pub fn new(output_bitsize: usize) -> Self {
        assert!(output_bitsize > 0);
        Self {
            _phantom: PhantomData,
            output_bitsize,
            shared_prg_1_2: None,
            shared_prg_2_3: None,
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

    pub fn init_round_0(&mut self) -> ((), SharedSeed) {
        assert!(!self.is_initialized);
        self.shared_prg_2_3 = Some(ChaChaRng::from_seed(thread_rng().gen()));
        ((), self.shared_prg_2_3.as_ref().unwrap().get_seed())
    }

    pub fn init_round_1(&mut self, shared_prg_seed_1_2: SharedSeed, _: ()) {
        assert!(!self.is_initialized);
        // receive shared PRF key from Party 1
        self.shared_prg_1_2 = Some(ChaChaRng::from_seed(shared_prg_seed_1_2));
        self.is_initialized = true;
    }

    pub fn init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error> {
        let fut_1_2 = comm.receive_previous()?;
        let (_, msg_2_3) = self.init_round_0();
        comm.send_next(msg_2_3)?;
        self.init_round_1(fut_1_2.get()?, ());
        Ok(())
    }

    pub fn preprocess_round_0(&mut self, num: usize) -> (Vec<F>, ()) {
        assert!(self.is_initialized);
        let n = num * self.output_bitsize;

        let preprocessed_squares: Vec<F> = (0..n)
            .map(|_| F::random(self.shared_prg_1_2.as_mut().unwrap()).square())
            .collect();
        self.preprocessed_rerand_m2
            .extend((0..num).map(|_| F::random(self.shared_prg_2_3.as_mut().unwrap())));
        let preprocessed_mult_d: Vec<F> = (0..n)
            .map(|_| F::random(self.shared_prg_2_3.as_mut().unwrap()))
            .collect();
        let preprocessed_mt_b: Vec<F> = (0..num)
            .map(|_| F::random(self.shared_prg_2_3.as_mut().unwrap()))
            .collect();
        let preprocessed_mt_c3: Vec<F> = (0..n)
            .map(|_| F::random(self.shared_prg_2_3.as_mut().unwrap()))
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

    pub fn preprocess<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        num: usize,
    ) -> Result<(), Error>
    where
        F: Serializable,
    {
        let (msg_2_1, _) = self.preprocess_round_0(num);
        comm.send_previous(msg_2_1)?;
        self.preprocess_round_1(num, (), ());
        Ok(())
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

    pub fn eval<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        num: usize,
        shares2: &[F],
    ) -> Result<(), Error>
    where
        F: Serializable,
    {
        assert_eq!(shares2.len(), num);
        let (msg_2_1, _) = self.eval_round_0(1, shares2);
        comm.send_previous(msg_2_1)?;
        Ok(())
    }
}

pub struct DOPrfParty3<F: LegendreSymbol> {
    _phantom: PhantomData<F>,
    output_bitsize: usize,
    shared_prg_1_3: Option<ChaChaRng>,
    shared_prg_2_3: Option<ChaChaRng>,
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
    F: LegendreSymbol,
{
    pub fn new(output_bitsize: usize) -> Self {
        assert!(output_bitsize > 0);
        Self {
            _phantom: PhantomData,
            output_bitsize,
            shared_prg_1_3: None,
            shared_prg_2_3: None,
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

    pub fn init_round_0(&mut self) -> (SharedSeed, ()) {
        assert!(!self.is_initialized);
        self.shared_prg_1_3 = Some(ChaChaRng::from_seed(thread_rng().gen()));
        (self.shared_prg_1_3.as_ref().unwrap().get_seed(), ())
    }

    pub fn init_round_1(&mut self, _: (), shared_prg_seed_2_3: SharedSeed) {
        self.shared_prg_2_3 = Some(ChaChaRng::from_seed(shared_prg_seed_2_3));
        self.is_initialized = true;
    }

    pub fn init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error> {
        let fut_2_3 = comm.receive_previous()?;
        let (msg_3_1, _) = self.init_round_0();
        comm.send_next(msg_3_1)?;
        self.init_round_1((), fut_2_3.get()?);
        Ok(())
    }

    pub fn preprocess_round_0(&mut self, num: usize) -> ((), ()) {
        assert!(self.is_initialized);
        let n = num * self.output_bitsize;

        self.preprocessed_rerand_m3
            .extend((0..num).map(|_| -F::random(self.shared_prg_2_3.as_mut().unwrap())));
        self.preprocessed_mult_d
            .extend((0..n).map(|_| F::random(self.shared_prg_2_3.as_mut().unwrap())));
        self.preprocessed_mt_b
            .extend((0..num).map(|_| F::random(self.shared_prg_2_3.as_mut().unwrap())));
        self.preprocessed_mt_c3
            .extend((0..n).map(|_| F::random(self.shared_prg_2_3.as_mut().unwrap())));
        ((), ())
    }

    pub fn preprocess_round_1(&mut self, num: usize, _: (), _: ()) {
        assert!(self.is_initialized);
        self.num_preprocessed_invocations += num;
    }

    pub fn preprocess<C: AbstractCommunicator>(
        &mut self,
        _comm: &mut C,
        num: usize,
    ) -> Result<(), Error>
    where
        F: Serializable,
    {
        self.preprocess_round_0(num);
        self.preprocess_round_1(num, (), ());
        Ok(())
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
                    debug_assert!(ls != 0, "unlikely");
                    bv.push(ls == 1);
                }
                bv
            })
            .collect();
        self.num_preprocessed_invocations -= num;
        output
    }

    pub fn eval<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        num: usize,
        shares3: &[F],
    ) -> Result<Vec<BitVec>, Error>
    where
        F: Serializable,
    {
        assert_eq!(shares3.len(), num);
        let fut_1_3 = comm.receive_next()?;
        let (msg_3_1, _) = self.eval_round_0(num, shares3);
        comm.send_next(msg_3_1)?;
        let output = self.eval_round_2(num, shares3, fut_1_3.get()?, ());
        Ok(output)
    }

    pub fn eval_to_uint<C: AbstractCommunicator, T: Unsigned>(
        &mut self,
        comm: &mut C,
        num: usize,
        shares3: &[F],
    ) -> Result<Vec<T>, Error>
    where
        F: Serializable,
    {
        assert!(self.output_bitsize <= T::BITS as usize);
        Ok(to_uint(self.eval(comm, num, shares3)?))
    }
}

pub struct JointDOPrf<F: LegendreSymbol> {
    output_bitsize: usize,
    doprf_p1_prev: DOPrfParty1<F>,
    doprf_p2_next: DOPrfParty2<F>,
    doprf_p3_mine: DOPrfParty3<F>,
}

impl<F: LegendreSymbol + Serializable> JointDOPrf<F> {
    pub fn new(output_bitsize: usize) -> Self {
        Self {
            output_bitsize,
            doprf_p1_prev: DOPrfParty1::new(output_bitsize),
            doprf_p2_next: DOPrfParty2::new(output_bitsize),
            doprf_p3_mine: DOPrfParty3::new(output_bitsize),
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.output_bitsize);
    }

    pub fn get_legendre_prf_key_prev(&self) -> LegendrePrfKey<F> {
        self.doprf_p1_prev.get_legendre_prf_key()
    }

    pub fn init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error> {
        let fut_prev = comm.receive_previous()?;
        let (msg_1_2, _) = self.doprf_p1_prev.init_round_0();
        let (_, msg_2_3) = self.doprf_p2_next.init_round_0();
        let (msg_3_1, _) = self.doprf_p3_mine.init_round_0();
        comm.send_next((msg_1_2, msg_2_3, msg_3_1))?;
        let (msg_1_2, msg_2_3, msg_3_1) = fut_prev.get()?;
        self.doprf_p1_prev.init_round_1((), msg_3_1);
        self.doprf_p2_next.init_round_1(msg_1_2, ());
        self.doprf_p3_mine.init_round_1((), msg_2_3);
        Ok(())
    }

    pub fn preprocess<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        num: usize,
    ) -> Result<(), Error> {
        let fut_2_1 = comm.receive_next()?;
        let (msg_2_1, _) = self.doprf_p2_next.preprocess_round_0(num);
        comm.send_previous(msg_2_1)?;
        self.doprf_p2_next.preprocess_round_1(num, (), ());
        self.doprf_p3_mine.preprocess_round_0(num);
        self.doprf_p3_mine.preprocess_round_1(num, (), ());
        self.doprf_p1_prev.preprocess_round_0(num);
        self.doprf_p1_prev
            .preprocess_round_1(num, fut_2_1.get()?, ());
        Ok(())
    }

    pub fn eval_to_uint<C: AbstractCommunicator, T: Unsigned>(
        &mut self,
        comm: &mut C,
        shares: &[F],
    ) -> Result<Vec<T>, Error> {
        let num = shares.len();

        let fut_2_1 = comm.receive_next::<Vec<_>>()?; // round 0
        let fut_3_1 = comm.receive_previous::<Vec<_>>()?; // round 0
        let fut_1_3 = comm.receive_next()?; // round 1

        let (msg_2_1, _) = self.doprf_p2_next.eval_round_0(num, shares);
        comm.send_previous(msg_2_1)?;

        let (msg_3_1, _) = self.doprf_p3_mine.eval_round_0(num, shares);
        comm.send_next(msg_3_1)?;

        let (_, msg_1_3) =
            self.doprf_p1_prev
                .eval_round_1(num, shares, &fut_2_1.get()?, &fut_3_1.get()?);
        comm.send_previous(msg_1_3)?;

        let output = self
            .doprf_p3_mine
            .eval_round_2(num, shares, fut_1_3.get()?, ());

        Ok(to_uint(output))
    }
}

pub struct MaskedDOPrfParty1<F: LegendreSymbol> {
    _phantom: PhantomData<F>,
    output_bitsize: usize,
    shared_prg_1_2: Option<ChaChaRng>,
    shared_prg_1_3: Option<ChaChaRng>,
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
    F: LegendreSymbol,
{
    pub fn new(output_bitsize: usize) -> Self {
        assert!(output_bitsize > 0);
        Self {
            _phantom: PhantomData,
            output_bitsize,
            shared_prg_1_2: None,
            shared_prg_1_3: None,
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

    pub fn from_legendre_prf_key(legendre_prf_key: LegendrePrfKey<F>) -> Self {
        let mut new = Self::new(legendre_prf_key.keys.len());
        new.legendre_prf_key = Some(legendre_prf_key);
        new
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

    pub fn init_round_0(&mut self) -> (SharedSeed, ()) {
        assert!(!self.is_initialized);
        // sample and share a PRF key with Party 2
        self.shared_prg_1_2 = Some(ChaChaRng::from_seed(thread_rng().gen()));
        (self.shared_prg_1_2.as_ref().unwrap().get_seed(), ())
    }

    pub fn init_round_1(&mut self, _: (), shared_prg_seed_1_3: SharedSeed) {
        assert!(!self.is_initialized);
        // receive shared PRF key from Party 3
        self.shared_prg_1_3 = Some(ChaChaRng::from_seed(shared_prg_seed_1_3));
        if self.legendre_prf_key.is_none() {
            // generate Legendre PRF key
            self.legendre_prf_key = Some(LegendrePrf::key_gen(self.output_bitsize));
        }
        self.is_initialized = true;
    }

    pub fn init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error> {
        let fut_3_1 = comm.receive_previous()?;
        let (msg_1_2, _) = self.init_round_0();
        comm.send_next(msg_1_2)?;
        self.init_round_1((), fut_3_1.get()?);
        Ok(())
    }

    pub fn get_legendre_prf_key(&self) -> LegendrePrfKey<F> {
        assert!(self.is_initialized);
        self.legendre_prf_key.as_ref().unwrap().clone()
    }

    pub fn preprocess_round_0(&mut self, num: usize) -> ((), ()) {
        assert!(self.is_initialized);
        let n = num * self.output_bitsize;
        self.preprocessed_rerand_m1
            .extend((0..num).map(|_| F::random(self.shared_prg_1_2.as_mut().unwrap())));
        self.preprocessed_mt_a
            .extend((0..n).map(|_| F::random(self.shared_prg_1_2.as_mut().unwrap())));
        self.preprocessed_mt_c1
            .extend((0..n).map(|_| F::random(self.shared_prg_1_2.as_mut().unwrap())));
        self.preprocessed_mult_e
            .extend((0..n).map(|_| F::random(self.shared_prg_1_2.as_mut().unwrap())));
        ((), ())
    }

    pub fn preprocess_round_1(&mut self, num: usize, _: (), _: ()) {
        assert!(self.is_initialized);
        self.num_preprocessed_invocations += num;
    }

    pub fn preprocess<C: AbstractCommunicator>(
        &mut self,
        _comm: &mut C,
        num: usize,
    ) -> Result<(), Error> {
        self.preprocess_round_0(num);
        self.preprocess_round_1(num, (), ());
        Ok(())
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
                    debug_assert!(ls != 0, "unlikely");
                    bv.push(ls == 1);
                }
                bv
            })
            .collect();
        self.num_preprocessed_invocations -= num;
        output
    }

    pub fn eval<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        num: usize,
        shares1: &[F],
    ) -> Result<Vec<BitVec>, Error>
    where
        F: Serializable,
    {
        assert_eq!(shares1.len(), num);
        let fut_3_1 = comm.receive_previous()?;
        let (_, msg_1_3) = self.eval_round_0(num, shares1);
        comm.send_previous(msg_1_3)?;
        let output = self.eval_round_2(1, shares1, (), fut_3_1.get()?);
        Ok(output)
    }

    pub fn eval_to_uint<C: AbstractCommunicator, T: Unsigned>(
        &mut self,
        comm: &mut C,
        num: usize,
        shares1: &[F],
    ) -> Result<Vec<T>, Error>
    where
        F: Serializable,
    {
        assert!(self.output_bitsize <= T::BITS as usize);
        Ok(to_uint(self.eval(comm, num, shares1)?))
    }
}

pub struct MaskedDOPrfParty2<F: LegendreSymbol> {
    _phantom: PhantomData<F>,
    output_bitsize: usize,
    shared_prg_1_2: Option<ChaChaRng>,
    shared_prg_2_3: Option<ChaChaRng>,
    is_initialized: bool,
    num_preprocessed_invocations: usize,
    preprocessed_rerand_m2: Vec<F>,
    preprocessed_r: BitVec,
}

impl<F> MaskedDOPrfParty2<F>
where
    F: LegendreSymbol,
{
    pub fn new(output_bitsize: usize) -> Self {
        assert!(output_bitsize > 0);
        Self {
            _phantom: PhantomData,
            output_bitsize,
            shared_prg_1_2: None,
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

    pub fn init_round_0(&mut self) -> ((), SharedSeed) {
        assert!(!self.is_initialized);
        self.shared_prg_2_3 = Some(ChaChaRng::from_seed(thread_rng().gen()));
        ((), self.shared_prg_2_3.as_ref().unwrap().get_seed())
    }

    pub fn init_round_1(&mut self, shared_prg_seed_1_2: SharedSeed, _: ()) {
        assert!(!self.is_initialized);
        // receive shared PRF key from Party 1
        self.shared_prg_1_2 = Some(ChaChaRng::from_seed(shared_prg_seed_1_2));
        self.is_initialized = true;
    }

    pub fn init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error> {
        let fut_1_2 = comm.receive_previous()?;
        let (_, msg_2_3) = self.init_round_0();
        comm.send_next(msg_2_3)?;
        self.init_round_1(fut_1_2.get()?, ());
        Ok(())
    }

    pub fn preprocess_round_0(&mut self, num: usize) -> ((), Vec<F>) {
        assert!(self.is_initialized);
        let n = num * self.output_bitsize;

        let mut preprocessed_t: Vec<_> = (0..n)
            .map(|_| F::random(self.shared_prg_2_3.as_mut().unwrap()).square())
            .collect();
        debug_assert!(!preprocessed_t.contains(&F::ZERO));
        {
            let mut random_bytes = vec![0u8; (n + 7) / 8];
            self.shared_prg_2_3
                .as_mut()
                .unwrap()
                .fill_bytes(&mut random_bytes);
            let new_r_slice = BitSlice::from_slice(&random_bytes);
            self.preprocessed_r.extend(&new_r_slice[..n]);
            for (i, r_i) in new_r_slice.iter().by_vals().take(n).enumerate() {
                if r_i {
                    preprocessed_t[i] *= F::get_non_random_qnr();
                }
            }
        }
        self.preprocessed_rerand_m2
            .extend((0..num).map(|_| -F::random(self.shared_prg_1_2.as_mut().unwrap())));
        let preprocessed_mt_a: Vec<F> = (0..n)
            .map(|_| F::random(self.shared_prg_1_2.as_mut().unwrap()))
            .collect();
        let preprocessed_mt_c1: Vec<F> = (0..n)
            .map(|_| F::random(self.shared_prg_1_2.as_mut().unwrap()))
            .collect();
        let preprocessed_mult_e: Vec<F> = (0..n)
            .map(|_| F::random(self.shared_prg_1_2.as_mut().unwrap()))
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

    pub fn preprocess<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        num: usize,
    ) -> Result<(), Error>
    where
        F: Serializable,
    {
        let (_, msg_2_3) = self.preprocess_round_0(num);
        comm.send_next(msg_2_3)?;
        self.preprocess_round_1(num, (), ());
        Ok(())
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

    pub fn eval<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        num: usize,
        shares2: &[F],
    ) -> Result<Vec<BitVec>, Error>
    where
        F: Serializable,
    {
        assert_eq!(shares2.len(), num);
        let (_, msg_2_3) = self.eval_round_0(num, shares2);
        comm.send_next(msg_2_3)?;
        let output = self.eval_get_output(num);
        Ok(output)
    }

    pub fn eval_to_uint<C: AbstractCommunicator, T: Unsigned>(
        &mut self,
        comm: &mut C,
        num: usize,
        shares2: &[F],
    ) -> Result<Vec<T>, Error>
    where
        F: Serializable,
    {
        assert!(self.output_bitsize <= T::BITS as usize);
        Ok(to_uint(self.eval(comm, num, shares2)?))
    }
}

pub struct MaskedDOPrfParty3<F: LegendreSymbol> {
    _phantom: PhantomData<F>,
    output_bitsize: usize,
    shared_prg_1_3: Option<ChaChaRng>,
    shared_prg_2_3: Option<ChaChaRng>,
    is_initialized: bool,
    num_preprocessed_invocations: usize,
    preprocessed_r: BitVec,
    preprocessed_t: Vec<F>,
    preprocessed_mt_c3: Vec<F>,
}

impl<F> MaskedDOPrfParty3<F>
where
    F: LegendreSymbol,
{
    pub fn new(output_bitsize: usize) -> Self {
        assert!(output_bitsize > 0);
        Self {
            _phantom: PhantomData,
            output_bitsize,
            shared_prg_1_3: None,
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

    pub fn init_round_0(&mut self) -> (SharedSeed, ()) {
        assert!(!self.is_initialized);
        self.shared_prg_1_3 = Some(ChaChaRng::from_seed(thread_rng().gen()));
        (self.shared_prg_1_3.as_ref().unwrap().get_seed(), ())
    }

    pub fn init_round_1(&mut self, _: (), shared_prg_seed_2_3: SharedSeed) {
        self.shared_prg_2_3 = Some(ChaChaRng::from_seed(shared_prg_seed_2_3));
        self.is_initialized = true;
    }

    pub fn init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error> {
        let fut_2_3 = comm.receive_previous()?;
        let (msg_3_1, _) = self.init_round_0();
        comm.send_next(msg_3_1)?;
        self.init_round_1((), fut_2_3.get()?);
        Ok(())
    }

    pub fn preprocess_round_0(&mut self, num: usize) -> ((), ()) {
        assert!(self.is_initialized);
        let n = num * self.output_bitsize;
        let start_index = self.num_preprocessed_invocations * self.output_bitsize;

        self.preprocessed_t
            .extend((0..n).map(|_| F::random(self.shared_prg_2_3.as_mut().unwrap()).square()));
        debug_assert!(!self.preprocessed_t[start_index..].contains(&F::ZERO));
        {
            let mut random_bytes = vec![0u8; (n + 7) / 8];
            self.shared_prg_2_3
                .as_mut()
                .unwrap()
                .fill_bytes(&mut random_bytes);
            let new_r_slice = BitSlice::from_slice(&random_bytes);
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

    pub fn preprocess<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        num: usize,
    ) -> Result<(), Error>
    where
        F: Serializable,
    {
        let fut_2_3 = comm.receive_previous()?;
        self.preprocess_round_0(num);
        self.preprocess_round_1(num, (), fut_2_3.get()?);
        Ok(())
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

    pub fn eval<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        num: usize,
        shares3: &[F],
    ) -> Result<Vec<BitVec>, Error>
    where
        F: Serializable,
    {
        assert_eq!(shares3.len(), num);
        let fut_1_3 = comm.receive_next::<Vec<_>>()?;
        let fut_2_3 = comm.receive_previous::<Vec<_>>()?;
        let (msg_3_1, _) = self.eval_round_1(1, shares3, &fut_1_3.get()?, &fut_2_3.get()?);
        comm.send_next(msg_3_1)?;
        let output = self.eval_get_output(num);
        Ok(output)
    }

    pub fn eval_to_uint<C: AbstractCommunicator, T: Unsigned>(
        &mut self,
        comm: &mut C,
        num: usize,
        shares3: &[F],
    ) -> Result<Vec<T>, Error>
    where
        F: Serializable,
    {
        assert!(self.output_bitsize <= T::BITS as usize);
        Ok(to_uint(self.eval(comm, num, shares3)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bincode;
    use ff::Field;
    use utils::field::Fp;

    fn doprf_init(
        party_1: &mut DOPrfParty1<Fp>,
        party_2: &mut DOPrfParty2<Fp>,
        party_3: &mut DOPrfParty3<Fp>,
    ) {
        let (msg_1_2, msg_1_3) = party_1.init_round_0();
        let (msg_2_1, msg_2_3) = party_2.init_round_0();
        let (msg_3_1, msg_3_2) = party_3.init_round_0();
        party_1.init_round_1(msg_2_1, msg_3_1);
        party_2.init_round_1(msg_1_2, msg_3_2);
        party_3.init_round_1(msg_1_3, msg_2_3);
    }

    fn doprf_preprocess(
        party_1: &mut DOPrfParty1<Fp>,
        party_2: &mut DOPrfParty2<Fp>,
        party_3: &mut DOPrfParty3<Fp>,
        num: usize,
    ) {
        let (msg_1_2, msg_1_3) = party_1.preprocess_round_0(num);
        let (msg_2_1, msg_2_3) = party_2.preprocess_round_0(num);
        let (msg_3_1, msg_3_2) = party_3.preprocess_round_0(num);
        party_1.preprocess_round_1(num, msg_2_1, msg_3_1);
        party_2.preprocess_round_1(num, msg_1_2, msg_3_2);
        party_3.preprocess_round_1(num, msg_1_3, msg_2_3);
    }

    fn doprf_eval(
        party_1: &mut DOPrfParty1<Fp>,
        party_2: &mut DOPrfParty2<Fp>,
        party_3: &mut DOPrfParty3<Fp>,
        shares_1: &[Fp],
        shares_2: &[Fp],
        shares_3: &[Fp],
        num: usize,
    ) -> Vec<BitVec> {
        assert_eq!(shares_1.len(), num);
        assert_eq!(shares_2.len(), num);
        assert_eq!(shares_3.len(), num);

        let (msg_2_1, msg_2_3) = party_2.eval_round_0(num, &shares_2);
        let (msg_3_1, _) = party_3.eval_round_0(num, &shares_3);
        let (_, msg_1_3) = party_1.eval_round_1(num, &shares_1, &msg_2_1, &msg_3_1);
        let output = party_3.eval_round_2(num, &shares_3, msg_1_3, msg_2_3);
        output
    }

    #[test]
    fn test_doprf() {
        let output_bitsize = 42;

        let mut party_1 = DOPrfParty1::<Fp>::new(output_bitsize);
        let mut party_2 = DOPrfParty2::<Fp>::new(output_bitsize);
        let mut party_3 = DOPrfParty3::<Fp>::new(output_bitsize);

        doprf_init(&mut party_1, &mut party_2, &mut party_3);

        // preprocess num invocations
        let num = 20;
        doprf_preprocess(&mut party_1, &mut party_2, &mut party_3, num);

        assert_eq!(party_1.get_num_preprocessed_invocations(), num);
        assert_eq!(party_2.get_num_preprocessed_invocations(), num);
        assert_eq!(party_3.get_num_preprocessed_invocations(), num);

        party_1.check_preprocessing();
        party_2.check_preprocessing();
        party_3.check_preprocessing();

        // preprocess another n invocations
        doprf_preprocess(&mut party_1, &mut party_2, &mut party_3, num);

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
            assert!(squares.iter().all(|&x| Fp::legendre_symbol(x) == 1));

            assert_eq!(rerand_m2.len(), num);
            assert_eq!(rerand_m3.len(), num);
            assert!(izip!(rerand_m2.iter(), rerand_m3.iter()).all(|(&m2, &m3)| m2 + m3 == Fp::ZERO));

            let mt_a: Vec<Fp> = squares
                .iter()
                .zip(mult_d.iter())
                .map(|(&s, &d)| s - d)
                .collect();
            assert_eq!(mult_d.len(), n);

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

        let output = doprf_eval(
            &mut party_1,
            &mut party_2,
            &mut party_3,
            &shares_1,
            &shares_2,
            &shares_3,
            num,
        );

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
            let output_i = LegendrePrf::<Fp>::eval_bits(&legendre_prf_key, input_i);
            assert_eq!(output[i], output_i);
        }
    }

    fn mdoprf_init(
        party_1: &mut MaskedDOPrfParty1<Fp>,
        party_2: &mut MaskedDOPrfParty2<Fp>,
        party_3: &mut MaskedDOPrfParty3<Fp>,
    ) {
        let (msg_1_2, msg_1_3) = party_1.init_round_0();
        let (msg_2_1, msg_2_3) = party_2.init_round_0();
        let (msg_3_1, msg_3_2) = party_3.init_round_0();
        party_1.init_round_1(msg_2_1, msg_3_1);
        party_2.init_round_1(msg_1_2, msg_3_2);
        party_3.init_round_1(msg_1_3, msg_2_3);
    }

    fn mdoprf_preprocess(
        party_1: &mut MaskedDOPrfParty1<Fp>,
        party_2: &mut MaskedDOPrfParty2<Fp>,
        party_3: &mut MaskedDOPrfParty3<Fp>,
        num: usize,
    ) {
        let (msg_1_2, msg_1_3) = party_1.preprocess_round_0(num);
        let (msg_2_1, msg_2_3) = party_2.preprocess_round_0(num);
        let (msg_3_1, msg_3_2) = party_3.preprocess_round_0(num);
        party_1.preprocess_round_1(num, msg_2_1, msg_3_1);
        party_2.preprocess_round_1(num, msg_1_2, msg_3_2);
        party_3.preprocess_round_1(num, msg_1_3, msg_2_3);
    }

    fn mdoprf_eval(
        party_1: &mut MaskedDOPrfParty1<Fp>,
        party_2: &mut MaskedDOPrfParty2<Fp>,
        party_3: &mut MaskedDOPrfParty3<Fp>,
        shares_1: &[Fp],
        shares_2: &[Fp],
        shares_3: &[Fp],
        num: usize,
    ) -> (Vec<BitVec>, Vec<BitVec>, Vec<BitVec>) {
        assert_eq!(shares_1.len(), num);
        assert_eq!(shares_2.len(), num);
        assert_eq!(shares_3.len(), num);

        let (_, msg_1_3) = party_1.eval_round_0(num, &shares_1);
        let (_, msg_2_3) = party_2.eval_round_0(num, &shares_2);
        let (msg_3_1, _) = party_3.eval_round_1(num, &shares_3, &msg_1_3, &msg_2_3);
        let masked_output = party_1.eval_round_2(num, &shares_1, (), msg_3_1);
        let mask2 = party_2.eval_get_output(num);
        let mask3 = party_3.eval_get_output(num);
        (masked_output, mask2, mask3)
    }

    #[test]
    fn test_masked_doprf() {
        let output_bitsize = 42;

        let mut party_1 = MaskedDOPrfParty1::<Fp>::new(output_bitsize);
        let mut party_2 = MaskedDOPrfParty2::<Fp>::new(output_bitsize);
        let mut party_3 = MaskedDOPrfParty3::<Fp>::new(output_bitsize);

        mdoprf_init(&mut party_1, &mut party_2, &mut party_3);

        // preprocess num invocations
        let num = 20;
        mdoprf_preprocess(&mut party_1, &mut party_2, &mut party_3, num);

        assert_eq!(party_1.get_num_preprocessed_invocations(), num);
        assert_eq!(party_2.get_num_preprocessed_invocations(), num);
        assert_eq!(party_3.get_num_preprocessed_invocations(), num);

        party_1.check_preprocessing();
        party_2.check_preprocessing();
        party_3.check_preprocessing();

        // preprocess another n invocations
        mdoprf_preprocess(&mut party_1, &mut party_2, &mut party_3, num);

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
                    Fp::legendre_symbol(t_i) == -1
                } else {
                    Fp::legendre_symbol(t_i) == 1
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
        let (masked_output, mask2, mask3) = mdoprf_eval(
            &mut party_1,
            &mut party_2,
            &mut party_3,
            &shares_1,
            &shares_2,
            &shares_3,
            num,
        );

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
            let expected_output_i = LegendrePrf::<Fp>::eval_bits(&legendre_prf_key, input_i);
            let output_i = masked_output[i].clone() ^ &mask2[i];
            assert_eq!(output_i, expected_output_i);
        }

        // preprocess another n invocations
        mdoprf_preprocess(&mut party_1, &mut party_2, &mut party_3, num);

        // perform another n evaluations on the same inputs
        let num = 15;

        let (new_masked_output, new_mask2, new_mask3) = mdoprf_eval(
            &mut party_1,
            &mut party_2,
            &mut party_3,
            &shares_1,
            &shares_2,
            &shares_3,
            num,
        );

        assert_eq!(party_1.get_num_preprocessed_invocations(), 25);
        assert_eq!(party_2.get_num_preprocessed_invocations(), 25);
        assert_eq!(party_3.get_num_preprocessed_invocations(), 25);
        party_1.check_preprocessing();
        party_2.check_preprocessing();
        party_3.check_preprocessing();

        assert_eq!(new_masked_output.len(), num);
        assert!(new_masked_output
            .iter()
            .all(|bv| bv.len() == output_bitsize));
        assert_eq!(new_mask2.len(), num);
        assert_eq!(new_mask2, new_mask3);
        assert!(new_mask2.iter().all(|bv| bv.len() == output_bitsize));

        // check that the new output matches the previous one
        for i in 0..num {
            let expected_output_i = masked_output[i].clone() ^ &mask2[i];
            let output_i = new_masked_output[i].clone() ^ &new_mask2[i];
            assert_eq!(output_i, expected_output_i);
        }
    }

    #[test]
    fn test_masked_doprf_single() {
        let output_bitsize = 42;

        let mut party_1 = MaskedDOPrfParty1::<Fp>::new(output_bitsize);
        let mut party_2 = MaskedDOPrfParty2::<Fp>::new(output_bitsize);
        let mut party_3 = MaskedDOPrfParty3::<Fp>::new(output_bitsize);

        mdoprf_init(&mut party_1, &mut party_2, &mut party_3);

        let share_1 = Fp::random(thread_rng());
        let share_2 = Fp::random(thread_rng());
        let share_3 = Fp::random(thread_rng());

        mdoprf_preprocess(&mut party_1, &mut party_2, &mut party_3, 1);
        let (masked_output_1, mask2_1, mask3_1) = mdoprf_eval(
            &mut party_1,
            &mut party_2,
            &mut party_3,
            &[share_1],
            &[share_2],
            &[share_3],
            1,
        );
        mdoprf_preprocess(&mut party_1, &mut party_2, &mut party_3, 1);
        let (masked_output_2, mask2_2, mask3_2) = mdoprf_eval(
            &mut party_1,
            &mut party_2,
            &mut party_3,
            &[share_1],
            &[share_2],
            &[share_3],
            1,
        );
        mdoprf_preprocess(&mut party_1, &mut party_2, &mut party_3, 1);
        let (masked_output_3, mask2_3, mask3_3) = mdoprf_eval(
            &mut party_1,
            &mut party_2,
            &mut party_3,
            &[share_1],
            &[share_2],
            &[share_3],
            1,
        );

        assert_eq!(mask2_1, mask3_1);
        assert_eq!(mask2_2, mask3_2);
        assert_eq!(mask2_3, mask3_3);
        let plain_output = masked_output_1[0].clone() ^ mask2_1[0].clone();
        assert_eq!(
            masked_output_2[0].clone() ^ mask2_2[0].clone(),
            plain_output
        );
        assert_eq!(
            masked_output_3[0].clone() ^ mask2_3[0].clone(),
            plain_output
        );
    }

    #[test]
    fn test_serialization() {
        let original_key = LegendrePrf::<Fp>::key_gen(42);
        let encoded_key =
            bincode::encode_to_vec(&original_key, bincode::config::standard()).unwrap();
        let (decoded_key, _size): (LegendrePrfKey<Fp>, usize) =
            bincode::decode_from_slice(&encoded_key, bincode::config::standard()).unwrap();
        assert_eq!(decoded_key, original_key);
    }
}
