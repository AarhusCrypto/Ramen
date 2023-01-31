use crate::common::{Error, InstructionShare};
use crate::doprf::{DOPrfParty1, DOPrfParty2, DOPrfParty3, LegendrePrf, LegendrePrfKey};
use crate::p_ot::{POTIndexParty, POTKeyParty, POTReceiverParty};
use crate::select::{Select, SelectProtocol};
use crate::stash::{Stash, StashProtocol};
use communicator::{AbstractCommunicator, Fut, Serializable};
use dpf::{mpdpf::MultiPointDpf, spdpf::SinglePointDpf};
use ff::PrimeField;
use itertools::{izip, Itertools};
use rand::thread_rng;
use std::iter::repeat;
use std::marker::PhantomData;
use utils::field::{FromPrf, LegendreSymbol};
use utils::permutation::FisherYatesPermutation;

pub trait DistributedOram<F>
where
    F: PrimeField,
{
    fn get_party_id(&self) -> usize;

    fn get_log_db_size(&self) -> u32;

    fn init<C: AbstractCommunicator>(&mut self, comm: &mut C, db_share: &[F]) -> Result<(), Error>;

    fn access<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        instruction: InstructionShare<F>,
    ) -> Result<F, Error>;

    fn get_db<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        rerandomize_shares: bool,
    ) -> Result<Vec<F>, Error>;
}

const PARTY_1: usize = 0;
const PARTY_2: usize = 1;
const PARTY_3: usize = 2;

fn compute_oram_prf_output_bitsize(memory_size: usize) -> usize {
    (usize::BITS - memory_size.leading_zeros()) as usize + 40
}

pub struct DistributedOramProtocol<F, MPDPF, SPDPF>
where
    F: FromPrf + LegendreSymbol + Serializable,
    F::PrfKey: Serializable,
    MPDPF: MultiPointDpf<Value = F>,
    MPDPF::Key: Serializable,
    SPDPF: SinglePointDpf<Value = F>,
    SPDPF::Key: Serializable,
{
    party_id: usize,
    log_db_size: u32,
    stash_size: usize,
    memory_size: usize,
    memory_share: Vec<F>,
    memory_index_tags_prev: Vec<u128>,
    memory_index_tags_next: Vec<u128>,
    memory_index_tags_prev_sorted: Vec<u128>,
    memory_index_tags_next_sorted: Vec<u128>,
    garbled_memory_share: Vec<(u128, F)>,
    is_initialized: bool,
    address_tags_read: Vec<u128>,
    stash: StashProtocol<F, SPDPF>,
    doprf_prev: DOPrfParty1<F>,
    doprf_next: DOPrfParty2<F>,
    doprf_mine: DOPrfParty3<F>,
    legendre_prf_key_next: Option<LegendrePrfKey<F>>,
    legendre_prf_key_prev: Option<LegendrePrfKey<F>>,
    pot_key_party: POTKeyParty<F, FisherYatesPermutation>,
    pot_index_party: POTIndexParty<F, FisherYatesPermutation>,
    pot_receiver_party: POTReceiverParty<F>,
    _phantom: PhantomData<MPDPF>,
}

impl<F, MPDPF, SPDPF> DistributedOramProtocol<F, MPDPF, SPDPF>
where
    F: FromPrf + LegendreSymbol + Serializable,
    F::PrfKey: Serializable,
    MPDPF: MultiPointDpf<Value = F>,
    MPDPF::Key: Serializable,
    SPDPF: SinglePointDpf<Value = F>,
    SPDPF::Key: Serializable,
{
    pub fn new(party_id: usize, log_db_size: u32) -> Self {
        assert!(party_id < 3);
        assert_eq!(log_db_size % 1, 0);
        let stash_size = 1 << (log_db_size / 2);
        let memory_size = (1 << log_db_size) + stash_size;
        let prf_output_bitsize = compute_oram_prf_output_bitsize(memory_size);

        Self {
            party_id,
            log_db_size,
            stash_size,
            memory_size,
            memory_share: Default::default(),
            memory_index_tags_prev: Default::default(),
            memory_index_tags_next: Default::default(),
            memory_index_tags_prev_sorted: Default::default(),
            memory_index_tags_next_sorted: Default::default(),
            garbled_memory_share: Default::default(),
            is_initialized: false,
            address_tags_read: Default::default(),
            stash: StashProtocol::new(party_id, stash_size),
            doprf_prev: DOPrfParty1::new(prf_output_bitsize),
            doprf_next: DOPrfParty2::new(prf_output_bitsize),
            doprf_mine: DOPrfParty3::new(prf_output_bitsize),
            legendre_prf_key_next: None,
            legendre_prf_key_prev: None,
            pot_key_party: POTKeyParty::new(memory_size),
            pot_index_party: POTIndexParty::new(memory_size),
            pot_receiver_party: POTReceiverParty::new(memory_size),
            _phantom: PhantomData,
        }
    }

    pub fn get_access_counter(&self) -> usize {
        self.stash.get_access_counter()
    }

    pub fn get_stash(&self) -> &StashProtocol<F, SPDPF> {
        &self.stash
    }

    fn pos_prev(&self, tag: u128) -> usize {
        debug_assert_eq!(self.memory_index_tags_prev_sorted.len(), self.memory_size);
        self.memory_index_tags_prev_sorted
            .binary_search(&tag)
            .expect("tag not found")
    }

    fn pos_next(&self, tag: u128) -> usize {
        debug_assert_eq!(self.memory_index_tags_next_sorted.len(), self.memory_size);
        self.memory_index_tags_next_sorted
            .binary_search(&tag)
            .expect("tag not found")
    }

    fn pos_mine(&self, tag: u128) -> usize {
        debug_assert_eq!(self.garbled_memory_share.len(), self.memory_size);
        self.garbled_memory_share
            .binary_search_by_key(&tag, |x| x.0)
            .expect("tag not found")
    }

    fn read_from_database<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        address_share: F,
    ) -> Result<F, Error> {
        let mut value_share = F::ZERO;

        // 1. Compute address tag
        let address_tag: u128 = match self.party_id {
            PARTY_1 => {
                self.doprf_mine.preprocess(comm, 1)?;
                let address_tag = self.doprf_mine.eval_to_uint(comm, 1, &[address_share])?[0];
                self.doprf_next.preprocess(comm, 1)?;
                self.doprf_next.eval(comm, 1, &[address_share])?;
                self.doprf_prev.preprocess(comm, 1)?;
                self.doprf_prev.eval(comm, 1, &[address_share])?;
                address_tag
            }
            PARTY_2 => {
                self.doprf_prev.preprocess(comm, 1)?;
                self.doprf_prev.eval(comm, 1, &[address_share])?;
                self.doprf_mine.preprocess(comm, 1)?;
                let address_tag = self.doprf_mine.eval_to_uint(comm, 1, &[address_share])?[0];
                self.doprf_next.preprocess(comm, 1)?;
                self.doprf_next.eval(comm, 1, &[address_share])?;
                address_tag
            }
            PARTY_3 => {
                self.doprf_next.preprocess(comm, 1)?;
                self.doprf_next.eval(comm, 1, &[address_share])?;
                self.doprf_prev.preprocess(comm, 1)?;
                self.doprf_prev.eval(comm, 1, &[address_share])?;
                self.doprf_mine.preprocess(comm, 1)?;
                self.doprf_mine.eval_to_uint(comm, 1, &[address_share])?[0]
            }
            _ => panic!("invalid party id"),
        };

        // 2. Update tags read list
        self.address_tags_read.push(address_tag);

        // 3. Compute index in garbled memory and retrieve share
        let garbled_index = self.pos_mine(address_tag);
        value_share += self.garbled_memory_share[garbled_index].1;

        // 4. Run p-OT.Access
        self.pot_index_party.run_access(comm, garbled_index)?;
        value_share -= self.pot_receiver_party.run_access(comm)?;

        Ok(value_share)
    }

    fn update_database_from_stash<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
    ) -> Result<(), Error> {
        let mpdpf = MPDPF::new(self.memory_size, self.stash_size);
        let mut points = Vec::with_capacity(self.stash_size);
        let mut values = Vec::with_capacity(self.stash_size);
        let (_, stash_values_share, stash_old_values_share) = self.stash.get_stash_share();
        assert_eq!(stash_values_share.len(), self.get_access_counter());
        assert_eq!(stash_old_values_share.len(), self.get_access_counter());
        assert_eq!(self.address_tags_read.len(), self.get_access_counter());
        for (tag, val, old_val) in izip!(
            self.address_tags_read.iter().copied(),
            stash_values_share.iter().copied(),
            stash_old_values_share.iter().copied()
        ) {
            points.push(self.pos_mine(tag) as u64);
            values.push(val - old_val);
        }
        self.address_tags_read.truncate(0);

        // sort point, value pairs
        let (points, values): (Vec<u64>, Vec<F>) = {
            let mut indices: Vec<usize> = (0..points.len()).collect();
            indices.sort_by_key(|&i| points[i]);
            points.sort();
            let new_values = indices.iter().map(|&i| values[i]).collect();
            (points, new_values)
        };

        let fut_dpf_key_from_prev = comm.receive_previous()?;
        let fut_dpf_key_from_next = comm.receive_next()?;
        let (dpf_key_prev, dpf_key_next) = mpdpf.generate_keys(&points, &values);
        comm.send_previous(dpf_key_prev)?;
        comm.send_next(dpf_key_next)?;
        let dpf_key_from_prev = fut_dpf_key_from_prev.get()?;
        let dpf_key_from_next = fut_dpf_key_from_next.get()?;

        let new_memory_share_from_prev = mpdpf.evaluate_domain(&dpf_key_from_prev);
        let new_memory_share_from_next = mpdpf.evaluate_domain(&dpf_key_from_next);

        {
            let mut memory_share = Vec::new();
            std::mem::swap(&mut self.memory_share, &mut memory_share);
            for j in 0..self.memory_size {
                memory_share[j] += new_memory_share_from_prev
                    [self.pos_prev(self.memory_index_tags_prev[j])]
                    + new_memory_share_from_next[self.pos_next(self.memory_index_tags_next[j])];
            }
            std::mem::swap(&mut self.memory_share, &mut memory_share);
        }

        Ok(())
    }

    fn refresh<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error> {
        // 0. Reset the functionalities
        self.stash.reset();
        self.doprf_prev.reset();
        self.doprf_mine.reset();
        self.doprf_next.reset();
        self.pot_key_party.reset();
        self.pot_index_party.reset();
        self.pot_receiver_party.reset();

        // 1. Initialize the stash
        self.stash.init(comm)?;

        // 2. Run r-DB init protocol
        // a) Initialize DOPRF
        {
            match self.party_id {
                PARTY_1 => {
                    self.doprf_mine.init(comm)?;
                    self.doprf_next.init(comm)?;
                    self.doprf_prev.init(comm)?;
                }
                PARTY_2 => {
                    self.doprf_prev.init(comm)?;
                    self.doprf_mine.init(comm)?;
                    self.doprf_next.init(comm)?;
                }
                PARTY_3 => {
                    self.doprf_next.init(comm)?;
                    self.doprf_prev.init(comm)?;
                    self.doprf_mine.init(comm)?;
                }
                _ => panic!("invalid party id"),
            };
            let fut_lpk_next = comm.receive_previous::<LegendrePrfKey<F>>()?;
            comm.send_next(self.doprf_prev.get_legendre_prf_key())?;
            self.legendre_prf_key_prev = Some(self.doprf_prev.get_legendre_prf_key());
            self.legendre_prf_key_next = Some(fut_lpk_next.get()?);
        }

        // b) Initialize p-OT
        {
            match self.party_id {
                PARTY_1 => {
                    self.pot_key_party.run_init(comm)?;
                    self.pot_receiver_party.run_init(comm)?;
                    self.pot_index_party.run_init(comm)?;
                }
                PARTY_2 => {
                    self.pot_index_party.run_init(comm)?;
                    self.pot_key_party.run_init(comm)?;
                    self.pot_receiver_party.run_init(comm)?;
                }
                PARTY_3 => {
                    self.pot_receiver_party.run_init(comm)?;
                    self.pot_index_party.run_init(comm)?;
                    self.pot_key_party.run_init(comm)?;
                }
                _ => panic!("invalid party id"),
            };
        }

        // c) Compute index tags and garble the memory share for the next party
        self.memory_index_tags_prev = Vec::with_capacity(self.memory_size);
        self.memory_index_tags_prev
            .extend((0..self.memory_size).map(|j| {
                LegendrePrf::eval_to_uint::<u128>(
                    &self.legendre_prf_key_prev.as_ref().unwrap(),
                    F::from_u128(j as u128),
                )
            }));
        self.memory_index_tags_prev_sorted = self
            .memory_index_tags_prev
            .iter()
            .copied()
            .sorted_unstable()
            .collect();
        debug_assert!(
            self.memory_index_tags_prev_sorted
                .windows(2)
                .all(|w| w[0] < w[1]),
            "index tags not sorted or colliding"
        );
        let fut_garbled_memory_share = comm.receive_previous()?;
        let mut garbled_memory_share_next: Vec<_> = self
            .memory_share
            .iter()
            .copied()
            .enumerate()
            .map(|(j, x)| {
                (
                    LegendrePrf::eval_to_uint::<u128>(
                        &self.legendre_prf_key_next.as_ref().unwrap(),
                        F::from_u128(j as u128),
                    ),
                    x,
                )
            })
            .collect();
        self.memory_index_tags_next = garbled_memory_share_next.iter().map(|x| x.0).collect();
        garbled_memory_share_next.sort_unstable_by_key(|x| x.0);
        self.memory_index_tags_next_sorted =
            garbled_memory_share_next.iter().map(|x| x.0).collect();
        debug_assert!(
            self.memory_index_tags_next_sorted
                .windows(2)
                .all(|w| w[0] < w[1]),
            "index tags not sorted or colliding"
        );
        // the memory_index_tags_{prev,next} now define the pos_{prev,next} maps
        // - pos_(i-1)(tag) -> index of tag in mem_idx_tags_prev
        // - pos_(i+1)(tag) -> index of tag in mem_idx_tags_next

        let mask = self.pot_key_party.expand();
        for j in 0..self.memory_size {
            let (tag, val) = garbled_memory_share_next[j];
            let masked_val = val + mask[self.pos_next(tag)];
            garbled_memory_share_next[j] = (tag, masked_val);
        }
        comm.send_next(garbled_memory_share_next)?;
        self.garbled_memory_share = fut_garbled_memory_share.get()?;
        // the garbled_memory_share now defines the pos_mine map:
        // - pos_i(tag) -> index of tag in garbled_memory_share

        Ok(())
    }
}

impl<F, MPDPF, SPDPF> DistributedOram<F> for DistributedOramProtocol<F, MPDPF, SPDPF>
where
    F: FromPrf + LegendreSymbol + Serializable,
    F::PrfKey: Serializable,
    MPDPF: MultiPointDpf<Value = F>,
    MPDPF::Key: Serializable,
    SPDPF: SinglePointDpf<Value = F>,
    SPDPF::Key: Serializable,
{
    fn get_party_id(&self) -> usize {
        self.party_id
    }

    fn get_log_db_size(&self) -> u32 {
        self.log_db_size
    }

    fn init<C: AbstractCommunicator>(&mut self, comm: &mut C, db_share: &[F]) -> Result<(), Error> {
        let db_size = 1 << self.log_db_size;
        assert_eq!(db_share.len(), db_size);

        // 1. Initialize memory share with given db share and pad with dummy values
        self.memory_share = Vec::with_capacity(self.memory_size);
        self.memory_share.extend_from_slice(db_share);
        self.memory_share
            .extend(repeat(F::ZERO).take(self.stash_size));

        // 2. Run the refresh protocol to initialize everything.
        self.refresh(comm)?;

        self.is_initialized = true;
        Ok(())
    }

    fn access<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        instruction: InstructionShare<F>,
    ) -> Result<F, Error> {
        assert!(self.is_initialized);

        // 1. Read from the stash
        let stash_state = self.stash.read(comm, instruction)?;

        // 2. If the value was found in a stash, we read from the dummy address
        let dummy_address_share = match self.party_id {
            PARTY_1 => F::from_u128((1 << self.log_db_size) + self.get_access_counter() as u128),
            _ => F::ZERO,
        };
        let db_address_share = SelectProtocol::select(
            comm,
            stash_state.flag,
            dummy_address_share,
            instruction.address,
        )?;

        // 3. Read a (dummy or real) value from the database
        let db_value_share = self.read_from_database(comm, db_address_share)?;

        // 4. Write the read value into the stash
        self.stash.write(
            comm,
            instruction,
            stash_state,
            db_address_share,
            db_value_share,
        )?;

        // 5. Select the right value to return
        let read_value =
            SelectProtocol::select(comm, stash_state.flag, stash_state.value, db_value_share)?;

        // 6. If the stash is full, write the value back into the database
        if self.get_access_counter() == self.stash_size {
            self.update_database_from_stash(comm)?;
            self.refresh(comm)?;
        }

        Ok(read_value)
    }

    fn get_db<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        rerandomize_shares: bool,
    ) -> Result<Vec<F>, Error> {
        assert!(self.is_initialized);

        if self.get_access_counter() > 0 {
            self.refresh(comm)?;
        }

        if rerandomize_shares {
            let fut = comm.receive_previous()?;
            let mut rng = thread_rng();
            let mask: Vec<_> = (0..1 << self.log_db_size)
                .map(|_| F::random(&mut rng))
                .collect();
            let mut masked_share: Vec<_> = self.memory_share[0..1 << self.log_db_size]
                .iter()
                .zip(mask.iter())
                .map(|(&x, &m)| x + m)
                .collect();
            comm.send_next(mask)?;
            let mask_prev: Vec<F> = fut.get()?;
            masked_share
                .iter_mut()
                .zip(mask_prev.iter())
                .for_each(|(x, &mp)| *x -= mp);
            Ok(masked_share)
        } else {
            Ok(self.memory_share[0..1 << self.log_db_size].to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Operation;
    use communicator::unix::make_unix_communicators;
    use dpf::mpdpf::DummyMpDpf;
    use dpf::spdpf::DummySpDpf;
    use ff::Field;
    use std::thread;
    use utils::field::Fp;

    fn run_init<F, C, P>(
        mut doram_party: P,
        mut comm: C,
        db_share: &[F],
    ) -> thread::JoinHandle<(P, C)>
    where
        F: PrimeField,
        C: AbstractCommunicator + Send + 'static,
        P: DistributedOram<F> + Send + 'static,
    {
        let db_share = db_share.to_vec();
        thread::Builder::new()
            .name(format!("Party {}", doram_party.get_party_id()))
            .spawn(move || {
                doram_party.init(&mut comm, &db_share).unwrap();
                (doram_party, comm)
            })
            .unwrap()
    }

    fn run_access<F, C, P>(
        mut doram_party: P,
        mut comm: C,
        instruction: InstructionShare<F>,
    ) -> thread::JoinHandle<(P, C, F)>
    where
        F: PrimeField,
        C: AbstractCommunicator + Send + 'static,
        P: DistributedOram<F> + Send + 'static,
    {
        thread::Builder::new()
            .name(format!("Party {}", doram_party.get_party_id()))
            .spawn(move || {
                let output = doram_party.access(&mut comm, instruction).unwrap();
                (doram_party, comm, output)
            })
            .unwrap()
    }

    fn run_get_db<F, C, P>(mut doram_party: P, mut comm: C) -> thread::JoinHandle<(P, C, Vec<F>)>
    where
        F: PrimeField,
        C: AbstractCommunicator + Send + 'static,
        P: DistributedOram<F> + Send + 'static,
    {
        thread::Builder::new()
            .name(format!("Party {}", doram_party.get_party_id()))
            .spawn(move || {
                let output = doram_party.get_db(&mut comm, false).unwrap();
                (doram_party, comm, output)
            })
            .unwrap()
    }

    #[test]
    fn test_oram() {
        type SPDPF = DummySpDpf<Fp>;
        type MPDPF = DummyMpDpf<Fp>;

        let log_db_size = 4;
        let db_size = 1 << log_db_size;
        let stash_size = 1 << (4 >> 1);

        let party_1 = DistributedOramProtocol::<Fp, MPDPF, SPDPF>::new(PARTY_1, log_db_size);
        let party_2 = DistributedOramProtocol::<Fp, MPDPF, SPDPF>::new(PARTY_2, log_db_size);
        let party_3 = DistributedOramProtocol::<Fp, MPDPF, SPDPF>::new(PARTY_3, log_db_size);
        assert_eq!(party_1.get_party_id(), PARTY_1);
        assert_eq!(party_2.get_party_id(), PARTY_2);
        assert_eq!(party_3.get_party_id(), PARTY_3);
        assert_eq!(party_1.get_log_db_size(), log_db_size);
        assert_eq!(party_2.get_log_db_size(), log_db_size);
        assert_eq!(party_3.get_log_db_size(), log_db_size);

        let (comm_3, comm_2, comm_1) = {
            let mut comms = make_unix_communicators(3);
            (
                comms.pop().unwrap(),
                comms.pop().unwrap(),
                comms.pop().unwrap(),
            )
        };

        // Initialize DB with zeros
        let db_share_1: Vec<_> = repeat(Fp::ZERO).take(db_size).collect();
        let db_share_2: Vec<_> = repeat(Fp::ZERO).take(db_size).collect();
        let db_share_3: Vec<_> = repeat(Fp::ZERO).take(db_size).collect();

        let h1 = run_init(party_1, comm_1, &db_share_1);
        let h2 = run_init(party_2, comm_2, &db_share_2);
        let h3 = run_init(party_3, comm_3, &db_share_3);
        let (mut party_1, mut comm_1) = h1.join().unwrap();
        let (mut party_2, mut comm_2) = h2.join().unwrap();
        let (mut party_3, mut comm_3) = h3.join().unwrap();

        fn mk_read(address: u128, value: u128) -> InstructionShare<Fp> {
            InstructionShare {
                operation: Operation::Read.encode(),
                address: Fp::from_u128(address),
                value: Fp::from_u128(value),
            }
        }

        fn mk_write(address: u128, value: u128) -> InstructionShare<Fp> {
            InstructionShare {
                operation: Operation::Write.encode(),
                address: Fp::from_u128(address),
                value: Fp::from_u128(value),
            }
        }

        let inst_zero_share = InstructionShare {
            operation: Fp::ZERO,
            address: Fp::ZERO,
            value: Fp::ZERO,
        };

        let number_cycles = 8;
        let instructions = [
            mk_write(12, 18),
            mk_read(12, 899),
            mk_write(13, 457),
            mk_write(0, 77),
            mk_write(13, 515),
            mk_write(15, 421),
            mk_write(13, 895),
            mk_write(4, 941),
            mk_write(1, 358),
            mk_read(9, 894),
            mk_read(7, 678),
            mk_write(3, 110),
            mk_read(15, 691),
            mk_read(13, 335),
            mk_write(9, 286),
            mk_read(13, 217),
            mk_write(10, 167),
            mk_read(3, 909),
            mk_write(2, 949),
            mk_read(14, 245),
            mk_write(3, 334),
            mk_write(0, 378),
            mk_write(2, 129),
            mk_write(5, 191),
            mk_write(15, 662),
            mk_write(4, 724),
            mk_write(1, 190),
            mk_write(6, 887),
            mk_write(9, 271),
            mk_read(12, 666),
            mk_write(0, 57),
            mk_write(2, 185),
        ];
        let expected_values = [
            Fp::from_u128(0),
            Fp::from_u128(18),
            Fp::from_u128(0),
            Fp::from_u128(0),
            Fp::from_u128(457),
            Fp::from_u128(0),
            Fp::from_u128(515),
            Fp::from_u128(0),
            Fp::from_u128(0),
            Fp::from_u128(0),
            Fp::from_u128(0),
            Fp::from_u128(0),
            Fp::from_u128(421),
            Fp::from_u128(895),
            Fp::from_u128(0),
            Fp::from_u128(895),
            Fp::from_u128(0),
            Fp::from_u128(110),
            Fp::from_u128(0),
            Fp::from_u128(0),
            Fp::from_u128(110),
            Fp::from_u128(77),
            Fp::from_u128(949),
            Fp::from_u128(0),
            Fp::from_u128(421),
            Fp::from_u128(941),
            Fp::from_u128(358),
            Fp::from_u128(0),
            Fp::from_u128(286),
            Fp::from_u128(18),
            Fp::from_u128(378),
            Fp::from_u128(129),
        ];
        let expected_db_contents = [
            [77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 457, 0, 0],
            [77, 0, 0, 0, 941, 0, 0, 0, 0, 0, 0, 0, 18, 895, 0, 421],
            [77, 358, 0, 110, 941, 0, 0, 0, 0, 0, 0, 0, 18, 895, 0, 421],
            [77, 358, 0, 110, 941, 0, 0, 0, 0, 286, 0, 0, 18, 895, 0, 421],
            [
                77, 358, 949, 110, 941, 0, 0, 0, 0, 286, 167, 0, 18, 895, 0, 421,
            ],
            [
                378, 358, 129, 334, 941, 191, 0, 0, 0, 286, 167, 0, 18, 895, 0, 421,
            ],
            [
                378, 190, 129, 334, 724, 191, 887, 0, 0, 286, 167, 0, 18, 895, 0, 662,
            ],
            [
                57, 190, 185, 334, 724, 191, 887, 0, 0, 271, 167, 0, 18, 895, 0, 662,
            ],
        ];

        // fn print_stash(
        //     p1: &DistributedOramProtocol<Fp, MPDPF, SPDPF>,
        //     p2: &DistributedOramProtocol<Fp, MPDPF, SPDPF>,
        //     p3: &DistributedOramProtocol<Fp, MPDPF, SPDPF>,
        // ) {
        //     let st1 = p1.get_stash().get_stash_share();
        //     let st2 = p2.get_stash().get_stash_share();
        //     let st3 = p3.get_stash().get_stash_share();
        //     let adrs: Vec<_> = izip!(st1.0.iter(), st2.0.iter(), st3.0.iter(),)
        //         .map(|(&x, &y, &z)| x + y + z)
        //         .collect();
        //     let vals: Vec<_> = izip!(st1.1.iter(), st2.1.iter(), st3.1.iter(),)
        //         .map(|(&x, &y, &z)| x + y + z)
        //         .collect();
        //     let olds: Vec<_> = izip!(st1.2.iter(), st2.2.iter(), st3.2.iter(),)
        //         .map(|(&x, &y, &z)| x + y + z)
        //         .collect();
        //     eprintln!("STASH: =======================");
        //     eprintln!("adrs = {adrs:?}");
        //     eprintln!("vals = {vals:?}");
        //     eprintln!("olds = {olds:?}");
        //     eprintln!("==============================");
        // }

        for i in 0..number_cycles {
            for j in 0..stash_size {
                let inst = instructions[i * stash_size + j];
                // eprintln!("Running {inst:?}");
                let expected_value = expected_values[i * stash_size + j];
                let h1 = run_access(party_1, comm_1, inst);
                let h2 = run_access(party_2, comm_2, inst_zero_share);
                let h3 = run_access(party_3, comm_3, inst_zero_share);
                let (p1, c1, value_1) = h1.join().unwrap();
                let (p2, c2, value_2) = h2.join().unwrap();
                let (p3, c3, value_3) = h3.join().unwrap();
                (party_1, party_2, party_3) = (p1, p2, p3);
                (comm_1, comm_2, comm_3) = (c1, c2, c3);
                assert_eq!(value_1 + value_2 + value_3, expected_value);
                // print_stash(&party_1, &party_2, &party_3);
            }
            let h1 = run_get_db(party_1, comm_1);
            let h2 = run_get_db(party_2, comm_2);
            let h3 = run_get_db(party_3, comm_3);
            let (p1, c1, db_share_1) = h1.join().unwrap();
            let (p2, c2, db_share_2) = h2.join().unwrap();
            let (p3, c3, db_share_3) = h3.join().unwrap();
            (party_1, party_2, party_3) = (p1, p2, p3);
            (comm_1, comm_2, comm_3) = (c1, c2, c3);
            let db: Vec<_> = izip!(db_share_1.iter(), db_share_2.iter(), db_share_3.iter())
                .map(|(&x, &y, &z)| x + y + z)
                .collect();
            // eprintln!("expected = {:#x?}", expected_db_contents[i]);
            // eprintln!("db = {:#?}", db);
            for k in 0..db_size {
                assert_eq!(db[k], Fp::from_u128(expected_db_contents[i][k]));
            }
        }
    }
}
