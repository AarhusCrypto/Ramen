use crate::common::Error;
use communicator::{AbstractCommunicator, Fut, Serializable};
use core::marker::PhantomData;
use ff::Field;
use rayon::prelude::*;
use utils::field::FromPrf;
use utils::permutation::Permutation;

pub struct POTKeyParty<F: FromPrf, Perm> {
    /// log of the database size
    domain_size: usize,
    /// if init was run
    is_initialized: bool,
    /// PRF key of the Index Party
    prf_key_i: Option<<F as FromPrf>::PrfKey>,
    /// PRF key of the Receiver Party
    prf_key_r: Option<<F as FromPrf>::PrfKey>,
    /// Permutation
    permutation: Option<Perm>,
    _phantom: PhantomData<F>,
}

impl<F, Perm> POTKeyParty<F, Perm>
where
    F: Field + FromPrf,
    F::PrfKey: Sync,
    Perm: Permutation + Sync,
{
    pub fn new(domain_size: usize) -> Self {
        Self {
            domain_size,
            is_initialized: false,
            prf_key_i: None,
            prf_key_r: None,
            permutation: None,
            _phantom: PhantomData,
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.domain_size);
    }

    pub fn init(&mut self) -> ((F::PrfKey, Perm::Key), F::PrfKey) {
        assert!(!self.is_initialized);
        self.prf_key_i = Some(F::prf_key_gen());
        self.prf_key_r = Some(F::prf_key_gen());
        let permutation_key = Perm::sample(self.domain_size);
        self.permutation = Some(Perm::from_key(permutation_key));
        self.is_initialized = true;
        (
            (self.prf_key_i.unwrap(), permutation_key),
            self.prf_key_r.unwrap(),
        )
    }

    pub fn run_init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error>
    where
        <F as FromPrf>::PrfKey: Serializable,
        Perm::Key: Serializable,
    {
        let (msg_to_index_party, msg_to_receiver_party) = self.init();
        comm.send_next(msg_to_index_party)?;
        comm.send_previous(msg_to_receiver_party)?;
        Ok(())
    }

    pub fn expand(&self) -> Vec<F> {
        assert!(self.is_initialized);
        (0..self.domain_size)
            .into_par_iter()
            .map(|x| {
                let pi_x = self.permutation.as_ref().unwrap().permute(x);
                F::prf(&self.prf_key_i.unwrap(), pi_x as u64)
                    + F::prf(&self.prf_key_r.unwrap(), pi_x as u64)
            })
            .collect()
    }
}

pub struct POTIndexParty<F: FromPrf, Perm> {
    /// log of the database size
    domain_size: usize,
    /// if init was run
    is_initialized: bool,
    /// PRF key of the Index Party
    prf_key_i: Option<<F as FromPrf>::PrfKey>,
    /// Permutation
    permutation: Option<Perm>,
    _phantom: PhantomData<F>,
}

impl<F: Field + FromPrf, Perm: Permutation> POTIndexParty<F, Perm> {
    pub fn new(domain_size: usize) -> Self {
        Self {
            domain_size,
            is_initialized: false,
            prf_key_i: None,
            permutation: None,
            _phantom: PhantomData,
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.domain_size);
    }

    pub fn init(&mut self, prf_key_i: F::PrfKey, permutation_key: Perm::Key) {
        assert!(!self.is_initialized);
        self.prf_key_i = Some(prf_key_i);
        self.permutation = Some(Perm::from_key(permutation_key));
        self.is_initialized = true;
    }

    pub fn run_init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error>
    where
        <F as FromPrf>::PrfKey: Serializable,
        Perm::Key: Serializable,
    {
        let msg_from_key_party: (F::PrfKey, Perm::Key) = comm.receive_previous()?.get()?;
        self.init(msg_from_key_party.0, msg_from_key_party.1);
        Ok(())
    }

    pub fn access(&self, index: usize) -> (usize, F) {
        assert!(index < self.domain_size);
        let pi_x = self.permutation.as_ref().unwrap().permute(index);
        (pi_x, F::prf(&self.prf_key_i.unwrap(), pi_x as u64))
    }

    pub fn run_access<C: AbstractCommunicator>(
        &self,
        comm: &mut C,
        index: usize,
    ) -> Result<(), Error>
    where
        F: Serializable,
    {
        let msg_to_receiver_party = self.access(index);
        comm.send_next(msg_to_receiver_party)?;
        Ok(())
    }
}

pub struct POTReceiverParty<F: FromPrf> {
    /// log of the database size
    domain_size: usize,
    /// if init was run
    is_initialized: bool,
    /// PRF key of the Receiver Party
    prf_key_r: Option<<F as FromPrf>::PrfKey>,
    _phantom: PhantomData<F>,
}

impl<F: Field + FromPrf> POTReceiverParty<F> {
    pub fn new(domain_size: usize) -> Self {
        Self {
            domain_size,
            is_initialized: false,
            prf_key_r: None,
            _phantom: PhantomData,
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.domain_size);
    }

    pub fn init(&mut self, prf_key_r: F::PrfKey) {
        assert!(!self.is_initialized);
        self.prf_key_r = Some(prf_key_r);
        self.is_initialized = true;
    }

    pub fn run_init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error>
    where
        <F as FromPrf>::PrfKey: Serializable,
    {
        let msg_from_key_party: F::PrfKey = comm.receive_next()?.get()?;
        self.init(msg_from_key_party);
        Ok(())
    }

    pub fn access(&self, permuted_index: usize, output_share: F) -> F {
        assert!(permuted_index < self.domain_size);
        F::prf(&self.prf_key_r.unwrap(), permuted_index as u64) + output_share
    }

    pub fn run_access<C: AbstractCommunicator>(&self, comm: &mut C) -> Result<F, Error>
    where
        F: Serializable,
    {
        let msg_from_index_party: (usize, F) = comm.receive_previous()?.get()?;
        let output = self.access(msg_from_index_party.0, msg_from_index_party.1);
        Ok(output)
    }
}

pub struct JointPOTParties<F: FromPrf, Perm> {
    key_party: POTKeyParty<F, Perm>,
    index_party: POTIndexParty<F, Perm>,
    receiver_party: POTReceiverParty<F>,
}

impl<F, Perm> JointPOTParties<F, Perm>
where
    F: Field + FromPrf,
    F::PrfKey: Sync,
    Perm: Permutation + Sync,
{
    pub fn new(domain_size: usize) -> Self {
        Self {
            key_party: POTKeyParty::new(domain_size),
            index_party: POTIndexParty::new(domain_size),
            receiver_party: POTReceiverParty::new(domain_size),
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.key_party.domain_size);
    }

    pub fn init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error>
    where
        <F as FromPrf>::PrfKey: Serializable,
        Perm::Key: Serializable,
    {
        self.key_party.run_init(comm)?;
        self.index_party.run_init(comm)?;
        self.receiver_party.run_init(comm)
    }

    pub fn access<C: AbstractCommunicator>(&self, comm: &mut C, my_index: usize) -> Result<F, Error>
    where
        F: Serializable,
    {
        self.index_party.run_access(comm, my_index)?;
        self.receiver_party.run_access(comm)
    }

    pub fn expand(&self) -> Vec<F> {
        self.key_party.expand()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use utils::field::Fp;
    use utils::permutation::FisherYatesPermutation;

    fn test_pot<F, Perm>(log_domain_size: u32)
    where
        F: Field + FromPrf,
        F::PrfKey: Sync,
        Perm: Permutation + Sync,
    {
        let domain_size = 1 << log_domain_size;

        // creation
        let mut key_party = POTKeyParty::<F, Perm>::new(domain_size);
        let mut index_party = POTIndexParty::<F, Perm>::new(domain_size);
        let mut receiver_party = POTReceiverParty::<F>::new(domain_size);
        assert!(!key_party.is_initialized());
        assert!(!index_party.is_initialized());
        assert!(!receiver_party.is_initialized());

        // init
        let (msg_to_index_party, msg_to_receiver_party) = key_party.init();
        index_party.init(msg_to_index_party.0, msg_to_index_party.1);
        receiver_party.init(msg_to_receiver_party);
        assert!(key_party.is_initialized());
        assert!(index_party.is_initialized());
        assert!(receiver_party.is_initialized());

        // expand to the key party's output
        let output_k = key_party.expand();
        assert_eq!(output_k.len(), domain_size);

        // access each index and verify consistency with key party's output
        for i in 0..domain_size {
            let msg_to_receiver_party = index_party.access(i);
            let output = receiver_party.access(msg_to_receiver_party.0, msg_to_receiver_party.1);
            assert_eq!(output, output_k[i]);
        }
    }

    #[test]
    fn test_all_pot() {
        let log_domain_size = 10;
        test_pot::<Fp, FisherYatesPermutation>(log_domain_size);
    }
}
