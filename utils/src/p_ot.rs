use crate::permutation::Permutation;
use crate::prf::{Prf, PrfKey};
use core::marker::PhantomData;
use ff::{Field, FromUniformBytes};

pub struct POTKeyParty<F, Perm> {
    /// log of the database size
    log_domain_size: u32,
    /// if init was run
    is_initialized: bool,
    /// PRF key of the Index Party
    prf_key_i: Option<PrfKey>,
    /// PRF key of the Receiver Party
    prf_key_r: Option<PrfKey>,
    /// Permutation
    permutation: Option<Perm>,
    _phantom: PhantomData<F>,
}

impl<F: Field + FromUniformBytes<{ Prf::OUT_LEN }>, Perm: Permutation> POTKeyParty<F, Perm> {
    pub fn new(log_domain_size: u32) -> Self {
        Self {
            log_domain_size,
            is_initialized: false,
            prf_key_i: None,
            prf_key_r: None,
            permutation: None,
            _phantom: PhantomData,
        }
    }

    pub fn init(&mut self) -> ((PrfKey, Perm::Key), PrfKey) {
        assert!(!self.is_initialized);
        self.prf_key_i = Some(Prf::key_gen());
        self.prf_key_r = Some(Prf::key_gen());
        let permutation_key = Perm::sample(self.log_domain_size);
        self.permutation = Some(Perm::from_key(permutation_key));
        self.is_initialized = true;
        (
            (self.prf_key_i.unwrap(), permutation_key),
            self.prf_key_r.unwrap(),
        )
    }

    pub fn expand(&self) -> Vec<F> {
        assert!(self.is_initialized);
        let n = 1 << self.log_domain_size;
        (0..n)
            .map(|x| {
                let pi_x = self.permutation.as_ref().unwrap().permute(x) as u64;
                Prf::eval::<F>(&self.prf_key_i.unwrap(), pi_x)
                    + Prf::eval::<F>(&self.prf_key_r.unwrap(), pi_x)
            })
            .collect()
    }
}

pub struct POTIndexParty<F, Perm> {
    /// log of the database size
    log_domain_size: u32,
    /// if init was run
    is_initialized: bool,
    /// PRF key of the Index Party
    prf_key_i: Option<PrfKey>,
    /// Permutation
    permutation: Option<Perm>,
    _phantom: PhantomData<F>,
}

impl<F: Field + FromUniformBytes<{ Prf::OUT_LEN }>, Perm: Permutation> POTIndexParty<F, Perm> {
    pub fn new(log_domain_size: u32) -> Self {
        Self {
            log_domain_size,
            is_initialized: false,
            prf_key_i: None,
            permutation: None,
            _phantom: PhantomData,
        }
    }

    pub fn init(&mut self, prf_key_i: PrfKey, permutation_key: Perm::Key) {
        assert!(!self.is_initialized);
        self.prf_key_i = Some(prf_key_i);
        self.permutation = Some(Perm::from_key(permutation_key));
        self.is_initialized = true;
    }

    pub fn access(&self, index: u64) -> (F, u64) {
        let pi_x = self.permutation.as_ref().unwrap().permute(index as usize) as u64;
        (Prf::eval(&self.prf_key_i.unwrap(), pi_x), pi_x)
    }
}

pub struct POTReceiverParty<F> {
    /// log of the database size
    log_domain_size: u32,
    /// if init was run
    is_initialized: bool,
    /// PRF key of the Receiver Party
    prf_key_r: Option<PrfKey>,
    _phantom: PhantomData<F>,
}

impl<F: Field + FromUniformBytes<{ Prf::OUT_LEN }>> POTReceiverParty<F> {
    pub fn new(log_domain_size: u32) -> Self {
        Self {
            log_domain_size,
            is_initialized: false,
            prf_key_r: None,
            _phantom: PhantomData,
        }
    }

    pub fn init(&mut self, prf_key_r: PrfKey) {
        assert!(!self.is_initialized);
        self.prf_key_r = Some(prf_key_r);
        self.is_initialized = true;
    }

    pub fn access(&self, permuted_index: u64, output_share: F) -> F {
        Prf::eval::<F>(&self.prf_key_r.unwrap(), permuted_index) + output_share
    }
}
