use crate::common::{Error, InstructionShare};
use crate::doprf::{JointDOPrf, LegendrePrf, LegendrePrfKey};
use crate::p_ot::JointPOTParties;
use crate::select::{Select, SelectProtocol};
use crate::stash::{
    ProtocolStep as StashProtocolStep, Runtimes as StashRuntimes, Stash, StashProtocol,
};
use communicator::{AbstractCommunicator, Fut, Serializable};
use dpf::{mpdpf::MultiPointDpf, spdpf::SinglePointDpf};
use ff::PrimeField;
use itertools::Itertools;
use rand::thread_rng;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::iter::repeat;
use std::marker::PhantomData;
use std::time::{Duration, Instant};
use strum::IntoEnumIterator;
use utils::field::{FromPrf, LegendreSymbol};
use utils::permutation::FisherYatesPermutation;

pub trait DistributedOram<F>
where
    F: PrimeField,
{
    fn get_party_id(&self) -> usize;

    fn get_log_db_size(&self) -> u32;

    fn init<C: AbstractCommunicator>(&mut self, comm: &mut C, db_share: &[F]) -> Result<(), Error>;

    fn preprocess<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        number_epochs: usize,
    ) -> Result<(), Error>;

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
// const PARTY_2: usize = 1;
// const PARTY_3: usize = 2;

fn compute_oram_prf_output_bitsize(memory_size: usize) -> usize {
    (usize::BITS - memory_size.leading_zeros()) as usize + 40
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, strum_macros::EnumIter, strum_macros::Display)]
pub enum ProtocolStep {
    Preprocess = 0,
    PreprocessLPRFKeyGenPrev,
    PreprocessLPRFEvalSortPrev,
    PreprocessLPRFKeyRecvNext,
    PreprocessLPRFEvalSortNext,
    PreprocessMpDpdfPrecomp,
    PreprocessRecvTagsMine,
    PreprocessStash,
    PreprocessDOPrf,
    PreprocessPOt,
    PreprocessSelect,
    Access,
    AccessStashRead,
    AccessAddressSelection,
    AccessDatabaseRead,
    AccessStashWrite,
    AccessValueSelection,
    AccessRefresh,
    DbReadAddressTag,
    DbReadGarbledIndex,
    DbReadPotAccess,
    DbWriteMpDpfKeyExchange,
    DbWriteMpDpfEvaluations,
    DbWriteUpdateMemory,
    RefreshJitPreprocess,
    RefreshResetFuncs,
    RefreshGetPreproc,
    RefreshSorting,
    RefreshPOtExpandMasking,
    RefreshReceivingShare,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Runtimes {
    durations: [Duration; 30],
    stash_runtimes: StashRuntimes,
}

impl Runtimes {
    #[inline(always)]
    pub fn record(&mut self, id: ProtocolStep, duration: Duration) {
        self.durations[id as usize] += duration;
    }

    pub fn get_stash_runtimes(&self) -> StashRuntimes {
        self.stash_runtimes
    }

    pub fn set_stash_runtimes(&mut self, stash_runtimes: StashRuntimes) {
        self.stash_runtimes = stash_runtimes;
    }

    pub fn get(&self, id: ProtocolStep) -> Duration {
        self.durations[id as usize]
    }

    pub fn print(&self, party_id: usize, num_accesses: usize) {
        println!("==================== Party {party_id} ====================");
        println!("- times per access over {num_accesses} accesses in total");
        println!(
            "{:30}    {:7.3} ms",
            ProtocolStep::Preprocess,
            self.get(ProtocolStep::Preprocess).as_secs_f64() * 1000.0 / num_accesses as f64
        );
        for step in ProtocolStep::iter()
            .filter(|x| x.to_string().starts_with("Preprocess") && *x != ProtocolStep::Preprocess)
        {
            println!(
                "    {:26}    {:7.3} ms",
                step,
                self.get(step).as_secs_f64() * 1000.0 / num_accesses as f64
            );
        }
        for step in ProtocolStep::iter().filter(|x| x.to_string().starts_with("Access")) {
            println!(
                "{:30}    {:7.3} ms",
                step,
                self.get(step).as_secs_f64() * 1000.0 / num_accesses as f64
            );
            match step {
                ProtocolStep::AccessDatabaseRead => {
                    for step in ProtocolStep::iter().filter(|x| x.to_string().starts_with("DbRead"))
                    {
                        println!(
                            "    {:26}    {:7.3} ms",
                            step,
                            self.get(step).as_secs_f64() * 1000.0 / num_accesses as f64
                        );
                    }
                }
                ProtocolStep::AccessRefresh => {
                    for step in ProtocolStep::iter().filter(|x| {
                        x.to_string().starts_with("DbWrite") || x.to_string().starts_with("Refresh")
                    }) {
                        println!(
                            "    {:26}    {:7.3} ms",
                            step,
                            self.get(step).as_secs_f64() * 1000.0 / num_accesses as f64
                        );
                    }
                }
                ProtocolStep::AccessStashRead => {
                    for step in
                        StashProtocolStep::iter().filter(|x| x.to_string().starts_with("Read"))
                    {
                        println!(
                            "    {:26}    {:7.3} ms",
                            step,
                            self.stash_runtimes.get(step).as_secs_f64() * 1000.0
                                / num_accesses as f64
                        );
                    }
                }
                ProtocolStep::AccessStashWrite => {
                    for step in
                        StashProtocolStep::iter().filter(|x| x.to_string().starts_with("Write"))
                    {
                        println!(
                            "    {:26}    {:7.3} ms",
                            step,
                            self.stash_runtimes.get(step).as_secs_f64() * 1000.0
                                / num_accesses as f64
                        );
                    }
                }
                _ => {}
            }
        }
        println!("==================================================");
    }
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
    prf_output_bitsize: usize,
    number_preprocessed_epochs: usize,
    preprocessed_legendre_prf_key_next: VecDeque<LegendrePrfKey<F>>,
    preprocessed_legendre_prf_key_prev: VecDeque<LegendrePrfKey<F>>,
    preprocessed_memory_index_tags_prev: VecDeque<Vec<u128>>,
    preprocessed_memory_index_tags_next: VecDeque<Vec<u128>>,
    preprocessed_memory_index_tags_mine_sorted: VecDeque<Vec<u128>>,
    preprocessed_memory_index_tags_prev_sorted: VecDeque<Vec<u128>>,
    preprocessed_memory_index_tags_next_sorted: VecDeque<Vec<u128>>,
    preprocessed_stash: VecDeque<StashProtocol<F, SPDPF>>,
    preprocessed_select: VecDeque<SelectProtocol<F>>,
    preprocessed_doprf: VecDeque<JointDOPrf<F>>,
    preprocessed_pot: VecDeque<JointPOTParties<F, FisherYatesPermutation>>,
    preprocessed_pot_expands: VecDeque<Vec<F>>,
    memory_index_tags_prev: Vec<u128>,
    memory_index_tags_next: Vec<u128>,
    memory_index_tags_prev_sorted: Vec<u128>,
    memory_index_tags_next_sorted: Vec<u128>,
    memory_index_tags_mine_sorted: Vec<u128>,
    garbled_memory_share: Vec<F>,
    is_initialized: bool,
    address_tags_read: Vec<u128>,
    stash: Option<StashProtocol<F, SPDPF>>,
    select_party: Option<SelectProtocol<F>>,
    joint_doprf: Option<JointDOPrf<F>>,
    legendre_prf_key_next: Option<LegendrePrfKey<F>>,
    legendre_prf_key_prev: Option<LegendrePrfKey<F>>,
    joint_pot: Option<JointPOTParties<F, FisherYatesPermutation>>,
    mpdpf: MPDPF,
    _phantom: PhantomData<MPDPF>,
}

impl<F, MPDPF, SPDPF> DistributedOramProtocol<F, MPDPF, SPDPF>
where
    F: FromPrf + LegendreSymbol + Serializable,
    F::PrfKey: Serializable + Sync,
    MPDPF: MultiPointDpf<Value = F> + Sync,
    MPDPF::Key: Serializable,
    SPDPF: SinglePointDpf<Value = F> + Sync,
    SPDPF::Key: Serializable + Sync,
{
    pub fn new(party_id: usize, log_db_size: u32) -> Self {
        assert!(party_id < 3);
        assert_eq!(log_db_size & 1, 0);
        let stash_size = 1 << (log_db_size / 2);
        let memory_size = (1 << log_db_size) + stash_size;
        let prf_output_bitsize = compute_oram_prf_output_bitsize(memory_size);

        Self {
            party_id,
            log_db_size,
            stash_size,
            memory_size,
            memory_share: Default::default(),
            number_preprocessed_epochs: 0,
            prf_output_bitsize,
            preprocessed_legendre_prf_key_next: Default::default(),
            preprocessed_legendre_prf_key_prev: Default::default(),
            preprocessed_memory_index_tags_prev: Default::default(),
            preprocessed_memory_index_tags_next: Default::default(),
            preprocessed_memory_index_tags_mine_sorted: Default::default(),
            preprocessed_memory_index_tags_prev_sorted: Default::default(),
            preprocessed_memory_index_tags_next_sorted: Default::default(),
            preprocessed_stash: Default::default(),
            preprocessed_select: Default::default(),
            preprocessed_doprf: Default::default(),
            preprocessed_pot: Default::default(),
            preprocessed_pot_expands: Default::default(),
            memory_index_tags_prev: Default::default(),
            memory_index_tags_next: Default::default(),
            memory_index_tags_prev_sorted: Default::default(),
            memory_index_tags_next_sorted: Default::default(),
            memory_index_tags_mine_sorted: Default::default(),
            garbled_memory_share: Default::default(),
            is_initialized: false,
            address_tags_read: Default::default(),
            stash: None,
            select_party: None,
            joint_doprf: None,
            legendre_prf_key_next: None,
            legendre_prf_key_prev: None,
            joint_pot: None,
            mpdpf: MPDPF::new(memory_size, stash_size),
            _phantom: PhantomData,
        }
    }

    pub fn get_access_counter(&self) -> usize {
        self.stash.as_ref().unwrap().get_access_counter()
    }

    pub fn get_stash(&self) -> &StashProtocol<F, SPDPF> {
        self.stash.as_ref().unwrap()
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
        debug_assert_eq!(self.memory_index_tags_mine_sorted.len(), self.memory_size);
        self.memory_index_tags_mine_sorted
            .binary_search(&tag)
            .expect("tag not found")
    }

    fn read_from_database<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        address_share: F,
        runtimes: Option<Runtimes>,
    ) -> Result<(F, Option<Runtimes>), Error> {
        let mut value_share = F::ZERO;

        let t_start = Instant::now();

        // 1. Compute address tag
        let address_tag: u128 = self
            .joint_doprf
            .as_mut()
            .unwrap()
            .eval_to_uint(comm, &[address_share])?[0];

        // 2. Update tags read list
        self.address_tags_read.push(address_tag);

        let t_after_address_tag = Instant::now();

        // 3. Compute index in garbled memory and retrieve share
        let garbled_index = self.pos_mine(address_tag);
        value_share += self.garbled_memory_share[garbled_index];

        let t_after_index_computation = Instant::now();

        // 4. Run p-OT.Access
        value_share -= self
            .joint_pot
            .as_ref()
            .unwrap()
            .access(comm, garbled_index)?;

        let t_after_pot_access = Instant::now();

        let runtimes = runtimes.map(|mut r| {
            r.record(
                ProtocolStep::DbReadAddressTag,
                t_after_address_tag - t_start,
            );
            r.record(
                ProtocolStep::DbReadAddressTag,
                t_after_index_computation - t_after_address_tag,
            );
            r.record(
                ProtocolStep::DbReadAddressTag,
                t_after_pot_access - t_after_index_computation,
            );
            r
        });

        Ok((value_share, runtimes))
    }

    fn update_database_from_stash<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        runtimes: Option<Runtimes>,
    ) -> Result<Option<Runtimes>, Error> {
        let t_start = Instant::now();

        let fut_dpf_key_from_prev = comm.receive_previous()?;
        let fut_dpf_key_from_next = comm.receive_next()?;

        let (_, stash_values_share, stash_old_values_share) =
            self.stash.as_ref().unwrap().get_stash_share();
        assert_eq!(stash_values_share.len(), self.get_access_counter());
        assert_eq!(stash_old_values_share.len(), self.get_access_counter());
        assert_eq!(self.address_tags_read.len(), self.get_access_counter());
        let mut points: Vec<_> = self
            .address_tags_read
            .par_iter()
            .copied()
            .map(|tag| self.pos_mine(tag) as u64)
            .collect();
        let values: Vec<_> = stash_values_share
            .par_iter()
            .copied()
            .zip(stash_old_values_share.par_iter().copied())
            .map(|(val, old_val)| val - old_val)
            .collect();
        self.address_tags_read.truncate(0);

        // sort point, value pairs
        let (points, values): (Vec<u64>, Vec<F>) = {
            let mut indices: Vec<usize> = (0..points.len()).collect();
            indices.par_sort_unstable_by_key(|&i| points[i]);
            points.par_sort();
            let new_values = indices.par_iter().map(|&i| values[i]).collect();
            (points, new_values)
        };

        let (dpf_key_prev, dpf_key_next) = self.mpdpf.generate_keys(&points, &values);
        comm.send_previous(dpf_key_prev)?;
        comm.send_next(dpf_key_next)?;
        let dpf_key_from_prev = fut_dpf_key_from_prev.get()?;
        let dpf_key_from_next = fut_dpf_key_from_next.get()?;

        let t_after_mpdpf_key_exchange = Instant::now();

        let new_memory_share_from_prev = self.mpdpf.evaluate_domain(&dpf_key_from_prev);
        let new_memory_share_from_next = self.mpdpf.evaluate_domain(&dpf_key_from_next);

        let t_after_mpdpf_evaluations = Instant::now();

        {
            let mut memory_share = Vec::new();
            std::mem::swap(&mut self.memory_share, &mut memory_share);
            memory_share
                .par_iter_mut()
                .enumerate()
                .for_each(|(j, mem_cell)| {
                    *mem_cell += new_memory_share_from_prev
                        [self.pos_prev(self.memory_index_tags_prev[j])]
                        + new_memory_share_from_next[self.pos_next(self.memory_index_tags_next[j])];
                });
            std::mem::swap(&mut self.memory_share, &mut memory_share);
        }

        let t_after_memory_update = Instant::now();

        let runtimes = runtimes.map(|mut r| {
            r.record(
                ProtocolStep::DbWriteMpDpfKeyExchange,
                t_after_mpdpf_key_exchange - t_start,
            );
            r.record(
                ProtocolStep::DbWriteMpDpfEvaluations,
                t_after_mpdpf_evaluations - t_after_mpdpf_key_exchange,
            );
            r.record(
                ProtocolStep::DbWriteUpdateMemory,
                t_after_memory_update - t_after_mpdpf_evaluations,
            );
            r
        });

        Ok(runtimes)
    }

    pub fn preprocess_with_runtimes<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        number_epochs: usize,
        runtimes: Option<Runtimes>,
    ) -> Result<Option<Runtimes>, Error> {
        let already_preprocessed = self.number_preprocessed_epochs;

        // Reserve some space
        self.preprocessed_legendre_prf_key_prev
            .reserve(number_epochs);
        self.preprocessed_legendre_prf_key_next
            .reserve(number_epochs);
        self.preprocessed_memory_index_tags_prev
            .reserve(number_epochs);
        self.preprocessed_memory_index_tags_next
            .reserve(number_epochs);
        self.preprocessed_memory_index_tags_prev_sorted
            .reserve(number_epochs);
        self.preprocessed_memory_index_tags_next_sorted
            .reserve(number_epochs);
        self.preprocessed_memory_index_tags_mine_sorted
            .reserve(number_epochs);
        self.preprocessed_stash.reserve(number_epochs);
        self.preprocessed_select.reserve(number_epochs);
        self.preprocessed_doprf.reserve(number_epochs);
        self.preprocessed_pot.reserve(number_epochs);
        self.preprocessed_pot_expands.reserve(number_epochs);

        let t_start = Instant::now();

        // Generate Legendre PRF keys
        let fut_lpks_next = comm.receive_previous::<Vec<LegendrePrfKey<F>>>()?;
        let fut_tags_mine_sorted = comm.receive_previous::<Vec<Vec<u128>>>()?;
        self.preprocessed_legendre_prf_key_prev
            .extend((0..number_epochs).map(|_| LegendrePrf::key_gen(self.prf_output_bitsize)));
        let new_lpks_prev =
            &self.preprocessed_legendre_prf_key_prev.make_contiguous()[already_preprocessed..];
        comm.send_slice_next(new_lpks_prev.as_ref())?;

        let t_after_gen_lpks_prev = Instant::now();

        // Compute memory index tags
        for lpk_prev in new_lpks_prev {
            let memory_index_tags_prev: Vec<_> = (0..self.memory_size)
                .into_par_iter()
                .map(|j| LegendrePrf::eval_to_uint::<u128>(lpk_prev, F::from_u128(j as u128)))
                .collect();
            let mut memory_index_tags_prev_sorted = memory_index_tags_prev.clone();
            memory_index_tags_prev_sorted.par_sort_unstable();
            self.preprocessed_memory_index_tags_prev
                .push_back(memory_index_tags_prev);
            self.preprocessed_memory_index_tags_prev_sorted
                .push_back(memory_index_tags_prev_sorted);
        }

        let t_after_computing_index_tags_prev = Instant::now();

        self.preprocessed_legendre_prf_key_next
            .extend(fut_lpks_next.get()?.into_iter());
        let new_lpks_next =
            &self.preprocessed_legendre_prf_key_next.make_contiguous()[already_preprocessed..];

        let t_after_receiving_lpks_next = Instant::now();

        for lpk_next in new_lpks_next {
            let memory_index_tags_next: Vec<_> = (0..self.memory_size)
                .into_par_iter()
                .map(|j| LegendrePrf::eval_to_uint::<u128>(lpk_next, F::from_u128(j as u128)))
                .collect();
            let memory_index_tags_next_with_index_sorted: Vec<_> = memory_index_tags_next
                .iter()
                .copied()
                .enumerate()
                .sorted_unstable_by_key(|(_, x)| *x)
                .collect();
            self.preprocessed_memory_index_tags_next
                .push_back(memory_index_tags_next);
            self.preprocessed_memory_index_tags_next_sorted.push_back(
                memory_index_tags_next_with_index_sorted
                    .par_iter()
                    .map(|(_, x)| *x)
                    .collect(),
            );
        }
        comm.send_next(
            self.preprocessed_memory_index_tags_next_sorted
                .make_contiguous()[already_preprocessed..]
                .to_vec(),
        )?;

        let t_after_computing_index_tags_next = Instant::now();

        self.mpdpf.precompute();

        let t_after_mpdpf_precomp = Instant::now();

        self.preprocessed_memory_index_tags_mine_sorted
            .extend(fut_tags_mine_sorted.get()?);

        let t_after_receiving_index_tags_mine = Instant::now();

        // Initialize Stash instances
        self.preprocessed_stash
            .extend((0..number_epochs).map(|_| StashProtocol::new(self.party_id, self.stash_size)));
        for stash in self
            .preprocessed_stash
            .iter_mut()
            .skip(already_preprocessed)
        {
            stash.init(comm)?;
        }

        let t_after_init_stash = Instant::now();

        // Initialize DOPRF instances
        self.preprocessed_doprf
            .extend((0..number_epochs).map(|_| JointDOPrf::new(self.prf_output_bitsize)));
        for (doprf, lpk_prev) in self
            .preprocessed_doprf
            .iter_mut()
            .skip(already_preprocessed)
            .zip(
                self.preprocessed_legendre_prf_key_prev
                    .iter()
                    .skip(already_preprocessed),
            )
        {
            doprf.set_legendre_prf_key_prev(lpk_prev.clone());
            doprf.init(comm)?;
            doprf.preprocess(comm, self.stash_size)?;
        }

        let t_after_init_doprf = Instant::now();

        // Precompute p-OTs and expand the mask
        self.preprocessed_pot
            .extend((0..number_epochs).map(|_| JointPOTParties::new(self.memory_size)));
        for pot in self.preprocessed_pot.iter_mut().skip(already_preprocessed) {
            pot.init(comm)?;
        }
        self.preprocessed_pot_expands.extend(
            self.preprocessed_pot.make_contiguous()[already_preprocessed..]
                .iter()
                .map(|pot| pot.expand()),
        );

        let t_after_preprocess_pot = Instant::now();

        self.preprocessed_select
            .extend((0..number_epochs).map(|_| SelectProtocol::default()));
        for select in self
            .preprocessed_select
            .iter_mut()
            .skip(already_preprocessed)
        {
            select.init(comm)?;
            select.preprocess(comm, 2 * self.stash_size)?;
        }

        let t_after_preprocess_select = Instant::now();

        self.number_preprocessed_epochs += number_epochs;

        debug_assert_eq!(
            self.preprocessed_legendre_prf_key_prev.len(),
            self.number_preprocessed_epochs
        );
        debug_assert_eq!(
            self.preprocessed_legendre_prf_key_next.len(),
            self.number_preprocessed_epochs
        );
        debug_assert_eq!(
            self.preprocessed_memory_index_tags_prev.len(),
            self.number_preprocessed_epochs
        );
        debug_assert_eq!(
            self.preprocessed_memory_index_tags_prev_sorted.len(),
            self.number_preprocessed_epochs
        );
        debug_assert_eq!(
            self.preprocessed_memory_index_tags_next.len(),
            self.number_preprocessed_epochs
        );
        debug_assert_eq!(
            self.preprocessed_memory_index_tags_next_sorted.len(),
            self.number_preprocessed_epochs
        );
        debug_assert_eq!(
            self.preprocessed_memory_index_tags_mine_sorted.len(),
            self.number_preprocessed_epochs
        );
        debug_assert_eq!(
            self.preprocessed_stash.len(),
            self.number_preprocessed_epochs
        );
        debug_assert_eq!(
            self.preprocessed_doprf.len(),
            self.number_preprocessed_epochs
        );
        debug_assert_eq!(self.preprocessed_pot.len(), self.number_preprocessed_epochs);
        debug_assert_eq!(
            self.preprocessed_pot_expands.len(),
            self.number_preprocessed_epochs
        );
        debug_assert_eq!(
            self.preprocessed_select.len(),
            self.number_preprocessed_epochs
        );

        let runtimes = runtimes.map(|mut r| {
            r.record(
                ProtocolStep::PreprocessLPRFKeyGenPrev,
                t_after_gen_lpks_prev - t_start,
            );
            r.record(
                ProtocolStep::PreprocessLPRFEvalSortPrev,
                t_after_computing_index_tags_prev - t_after_gen_lpks_prev,
            );
            r.record(
                ProtocolStep::PreprocessLPRFKeyRecvNext,
                t_after_receiving_lpks_next - t_after_computing_index_tags_prev,
            );
            r.record(
                ProtocolStep::PreprocessLPRFEvalSortNext,
                t_after_computing_index_tags_next - t_after_receiving_lpks_next,
            );
            r.record(
                ProtocolStep::PreprocessMpDpdfPrecomp,
                t_after_mpdpf_precomp - t_after_computing_index_tags_next,
            );
            r.record(
                ProtocolStep::PreprocessRecvTagsMine,
                t_after_receiving_index_tags_mine - t_after_mpdpf_precomp,
            );
            r.record(
                ProtocolStep::PreprocessStash,
                t_after_init_stash - t_after_receiving_index_tags_mine,
            );
            r.record(
                ProtocolStep::PreprocessDOPrf,
                t_after_init_doprf - t_after_init_stash,
            );
            r.record(
                ProtocolStep::PreprocessPOt,
                t_after_preprocess_pot - t_after_init_doprf,
            );
            r.record(
                ProtocolStep::PreprocessSelect,
                t_after_preprocess_select - t_after_preprocess_pot,
            );
            r.record(
                ProtocolStep::Preprocess,
                t_after_preprocess_select - t_start,
            );
            r
        });

        Ok(runtimes)
    }

    fn refresh<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        runtimes: Option<Runtimes>,
    ) -> Result<Option<Runtimes>, Error> {
        let t_start = Instant::now();

        // 0. Do preprocessing if not already done

        let runtimes = if self.number_preprocessed_epochs == 0 {
            self.preprocess_with_runtimes(comm, 1, runtimes)?
        } else {
            runtimes
        };

        let t_after_jit_preprocessing = Instant::now();

        // 1. Expect to receive garbled memory share
        let fut_garbled_memory_share = comm.receive_previous::<Vec<F>>()?;

        // 2. Get fresh (initialized) instances of the functionalities

        // a) Stash
        self.stash = self.preprocessed_stash.pop_front();
        debug_assert!(self.stash.is_some());

        // b) DOPRF
        self.legendre_prf_key_prev = self.preprocessed_legendre_prf_key_prev.pop_front();
        self.legendre_prf_key_next = self.preprocessed_legendre_prf_key_next.pop_front();
        self.joint_doprf = self.preprocessed_doprf.pop_front();
        debug_assert!(self.legendre_prf_key_prev.is_some());
        debug_assert!(self.legendre_prf_key_next.is_some());
        debug_assert!(self.joint_doprf.is_some());

        // c) p-OT
        self.joint_pot = self.preprocessed_pot.pop_front();
        debug_assert!(self.joint_pot.is_some());

        // d) select
        self.select_party = self.preprocessed_select.pop_front();
        debug_assert!(self.joint_pot.is_some());

        // e) Retrieve preprocessed index tags
        self.memory_index_tags_prev = self
            .preprocessed_memory_index_tags_prev
            .pop_front()
            .unwrap();
        self.memory_index_tags_prev_sorted = self
            .preprocessed_memory_index_tags_prev_sorted
            .pop_front()
            .unwrap();
        self.memory_index_tags_next = self
            .preprocessed_memory_index_tags_next
            .pop_front()
            .unwrap();
        self.memory_index_tags_next_sorted = self
            .preprocessed_memory_index_tags_next_sorted
            .pop_front()
            .unwrap();
        self.memory_index_tags_mine_sorted = self
            .preprocessed_memory_index_tags_mine_sorted
            .pop_front()
            .unwrap();
        debug_assert!(
            self.memory_index_tags_prev_sorted
                .windows(2)
                .all(|w| w[0] < w[1]),
            "index tags not sorted or colliding"
        );
        debug_assert!(
            self.memory_index_tags_next_sorted
                .windows(2)
                .all(|w| w[0] < w[1]),
            "index tags not sorted or colliding"
        );
        debug_assert!(
            self.memory_index_tags_mine_sorted
                .windows(2)
                .all(|w| w[0] < w[1]),
            "index tags not sorted or colliding"
        );

        let t_after_get_preprocessed_data = Instant::now();

        // 2.) Garble the memory share for the next party
        let mut garbled_memory_share_next: Vec<_> = self
            .memory_share
            .iter()
            .copied()
            .zip(self.memory_index_tags_next.iter().copied())
            .sorted_unstable_by_key(|(_, i)| *i)
            .map(|(x, _)| x)
            .collect();

        let t_after_sort = Instant::now();

        // the memory_index_tags_{prev,next} now define the pos_{prev,next} maps
        // - pos_(i-1)(tag) -> index of tag in mem_idx_tags_prev
        // - pos_(i+1)(tag) -> index of tag in mem_idx_tags_next

        let mask = self.preprocessed_pot_expands.pop_front().unwrap();
        self.memory_index_tags_next_sorted
            .par_iter()
            .zip(garbled_memory_share_next.par_iter_mut())
            .for_each(|(&tag, val)| {
                *val += mask[self.pos_next(tag)];
            });
        comm.send_next(garbled_memory_share_next)?;

        let t_after_pot_expand = Instant::now();

        self.garbled_memory_share = fut_garbled_memory_share.get()?;
        // the garbled_memory_share now defines the pos_mine map:
        // - pos_i(tag) -> index of tag in garbled_memory_share

        let t_after_receiving = Instant::now();

        // account that we used one set of preprocessing material
        self.number_preprocessed_epochs -= 1;

        let runtimes = runtimes.map(|mut r| {
            r.record(
                ProtocolStep::RefreshJitPreprocess,
                t_after_jit_preprocessing - t_start,
            );
            r.record(
                ProtocolStep::RefreshGetPreproc,
                t_after_get_preprocessed_data - t_after_jit_preprocessing,
            );
            r.record(
                ProtocolStep::RefreshSorting,
                t_after_sort - t_after_get_preprocessed_data,
            );
            r.record(
                ProtocolStep::RefreshPOtExpandMasking,
                t_after_pot_expand - t_after_sort,
            );
            r.record(
                ProtocolStep::RefreshReceivingShare,
                t_after_receiving - t_after_pot_expand,
            );
            r
        });

        Ok(runtimes)
    }

    pub fn access_with_runtimes<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        instruction: InstructionShare<F>,
        runtimes: Option<Runtimes>,
    ) -> Result<(F, Option<Runtimes>), Error> {
        assert!(self.is_initialized);

        // 1. Read from the stash
        let t_start = Instant::now();
        let (stash_state, stash_runtimes) = self.stash.as_mut().unwrap().read_with_runtimes(
            comm,
            instruction,
            runtimes.map(|r| r.get_stash_runtimes()),
        )?;
        let t_after_stash_read = Instant::now();

        // 2. If the value was found in a stash, we read from the dummy address
        let dummy_address_share = match self.party_id {
            PARTY_1 => F::from_u128((1 << self.log_db_size) + self.get_access_counter() as u128),
            _ => F::ZERO,
        };
        let db_address_share = self.select_party.as_mut().unwrap().select(
            comm,
            stash_state.flag,
            dummy_address_share,
            instruction.address,
        )?;
        let t_after_address_selection = Instant::now();

        // 3. Read a (dummy or real) value from the database
        let (db_value_share, runtimes) =
            self.read_from_database(comm, db_address_share, runtimes)?;
        let t_after_db_read = Instant::now();

        // 4. Write the read value into the stash
        let stash_runtime = self.stash.as_mut().unwrap().write_with_runtimes(
            comm,
            instruction,
            stash_state,
            db_address_share,
            db_value_share,
            stash_runtimes,
        )?;
        let t_after_stash_write = Instant::now();

        // 5. Select the right value to return
        let read_value = self.select_party.as_mut().unwrap().select(
            comm,
            stash_state.flag,
            stash_state.value,
            db_value_share,
        )?;
        let t_after_value_selection = Instant::now();

        // 6. If the stash is full, write the value back into the database
        let runtimes = if self.get_access_counter() == self.stash_size {
            let runtimes = self.update_database_from_stash(comm, runtimes)?;
            self.refresh(comm, runtimes)?
        } else {
            runtimes
        };
        let t_after_refresh = Instant::now();

        let runtimes = runtimes.map(|mut r| {
            r.set_stash_runtimes(stash_runtime.unwrap());
            r.record(ProtocolStep::AccessStashRead, t_after_stash_read - t_start);
            r.record(
                ProtocolStep::AccessAddressSelection,
                t_after_address_selection - t_after_stash_read,
            );
            r.record(
                ProtocolStep::AccessDatabaseRead,
                t_after_db_read - t_after_address_selection,
            );
            r.record(
                ProtocolStep::AccessStashWrite,
                t_after_stash_write - t_after_db_read,
            );
            r.record(
                ProtocolStep::AccessValueSelection,
                t_after_value_selection - t_after_stash_write,
            );
            r.record(
                ProtocolStep::AccessRefresh,
                t_after_refresh - t_after_value_selection,
            );
            r.record(ProtocolStep::Access, t_after_refresh - t_start);
            r
        });

        Ok((read_value, runtimes))
    }
}

impl<F, MPDPF, SPDPF> DistributedOram<F> for DistributedOramProtocol<F, MPDPF, SPDPF>
where
    F: FromPrf + LegendreSymbol + Serializable,
    F::PrfKey: Serializable + Sync,
    MPDPF: MultiPointDpf<Value = F> + Sync,
    MPDPF::Key: Serializable,
    SPDPF: SinglePointDpf<Value = F> + Sync,
    SPDPF::Key: Serializable + Sync,
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
        self.refresh(comm, None)?;

        self.is_initialized = true;
        Ok(())
    }

    fn preprocess<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        number_epochs: usize,
    ) -> Result<(), Error> {
        self.preprocess_with_runtimes(comm, number_epochs, None)
            .map(|_| ())
    }

    fn access<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        instruction: InstructionShare<F>,
    ) -> Result<F, Error> {
        self.access_with_runtimes(comm, instruction, None)
            .map(|x| x.0)
    }

    fn get_db<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        rerandomize_shares: bool,
    ) -> Result<Vec<F>, Error> {
        assert!(self.is_initialized);

        if self.get_access_counter() > 0 {
            self.refresh(comm, None)?;
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
    use itertools::izip;
    use std::thread;
    use utils::field::Fp;

    const PARTY_1: usize = 0;
    const PARTY_2: usize = 1;
    const PARTY_3: usize = 2;

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
