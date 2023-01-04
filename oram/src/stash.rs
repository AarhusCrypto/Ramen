use ff::PrimeField;
use std::marker::PhantomData;

pub struct Stash<F: PrimeField> {
    stash_size: usize,
    stash_share: Vec<StashEntryShare<F>>,
    _phantom: PhantomData<F>,
}

#[derive(Clone, Copy, Debug, Default)]
struct StashEntryShare<F: PrimeField> {
    pub address_share: F,
    pub value_share: F,
    pub old_value_share: F,
}

impl<F: PrimeField> Stash<F> {

    pub fn new(stash_size: usize) -> Self {

        Self {
            stash_size,
            stash_share: Default::default(),
            _phantom: PhantomData,
        }
    }

    pub fn init(&mut self, stash_share: &[F]) {
        panic!("not implemented");
    }

    pub fn read(&self, counter: usize) {
        panic!("not implemented");
    }

    pub fn write(&self) {
        panic!("not implemented");
    }
}
