use crate::stash::Stash;
use ff::PrimeField;
use std::marker::PhantomData;

type Address = usize;

pub struct DistributedOram<F: PrimeField> {
    party_id: usize,
    log_db_size: usize,
    stash_size: usize,
    memory_size: usize,
    memory_share: Vec<F>,
    is_initialized: bool,
    access_counter: usize,
    addresses_read: Vec<Address>,
    stash: Stash<F>,
    _phantom: PhantomData<F>,
}

impl<F: PrimeField> DistributedOram<F> {
    pub fn new(party_id: usize, log_db_size: usize) -> Self {
        assert!(party_id < 3);
        assert_eq!(log_db_size % 1, 0);
        let stash_size = 1 << (log_db_size / 2);
        let memory_size = (1 << log_db_size) + stash_size;

        Self {
            party_id,
            log_db_size,
            stash_size,
            memory_size,
            memory_share: Default::default(),
            is_initialized: false,
            access_counter: 0,
            addresses_read: Default::default(),
            stash: Stash::new(stash_size),
            _phantom: PhantomData,
        }
    }

    pub fn init(&mut self, share: &[F]) {
        // - 3x DOPRF key generation
        // - 3x p-OT initialization
        // - randomize DB
        self.is_initialized = true;
        panic!("not implemented");
    }

    pub fn read_from_database(&mut self) -> F {
        panic!("not implemented");
    }

    pub fn refresh(&mut self) {
        panic!("not implemented");
        assert!(self.is_initialized);
    }

    pub fn access(&mut self) {
        panic!("not implemented");
        assert!(self.is_initialized);
    }

    pub fn get_db(&self) -> Vec<F> {
        panic!("not implemented");
        assert!(self.is_initialized);
    }
}
