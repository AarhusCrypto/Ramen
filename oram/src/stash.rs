//! Stash protocol implementation.

use crate::common::{Error, InstructionShare};
use crate::doprf::{
    DOPrfParty1, DOPrfParty2, DOPrfParty3, LegendrePrf, MaskedDOPrfParty1, MaskedDOPrfParty2,
    MaskedDOPrfParty3,
};
use crate::mask_index::{MaskIndex, MaskIndexProtocol};
use crate::select::{Select, SelectProtocol};
use communicator::{AbstractCommunicator, Fut, Serializable};
use dpf::spdpf::SinglePointDpf;
use ff::PrimeField;
use rand::thread_rng;
use rayon::prelude::*;
use std::marker::PhantomData;
use std::time::{Duration, Instant};
use utils::field::LegendreSymbol;

/// Result of a stash read.
///
/// All values are shared.
#[derive(Clone, Copy, Debug, Default)]
pub struct StashStateShare<F: PrimeField> {
    /// Share of 1 if the searched address was present in the stash, and share of 0 otherwise.
    pub flag: F,
    /// Possible location of the found entry in the stash.
    pub location: F,
    /// Possible value of the found entry.
    pub value: F,
}

/// State of the stash protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    New,
    AwaitingRead,
    AwaitingWrite,
    AccessesExhausted,
}

const PARTY_1: usize = 0;
const PARTY_2: usize = 1;
const PARTY_3: usize = 2;

/// Definition of the stash interface.
pub trait Stash<F: PrimeField> {
    /// Return ID of the current party.
    fn get_party_id(&self) -> usize;

    /// Return capacity of the stash.
    fn get_stash_size(&self) -> usize;

    /// Return current access counter.
    fn get_access_counter(&self) -> usize;

    /// Reset the data structure to be used again.
    fn reset(&mut self);

    /// Initialize the stash.
    fn init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error>;

    /// Perform a read from the stash.
    fn read<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        instruction: InstructionShare<F>,
    ) -> Result<StashStateShare<F>, Error>;

    /// Perform a write into the stash.
    fn write<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        instruction: InstructionShare<F>,
        stash_state: StashStateShare<F>,
        db_address_share: F,
        db_value_share: F,
    ) -> Result<(), Error>;

    /// Get an additive share of the stash.
    fn get_stash_share(&self) -> (&[F], &[F], &[F]);
}

fn compute_stash_prf_output_bitsize(stash_size: usize) -> usize {
    2 * (usize::BITS - (stash_size - 1).leading_zeros()) as usize + 40
}

/// Protocol steps of the stash initialization, read, and write.
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum_macros::EnumIter, strum_macros::Display)]
pub enum ProtocolStep {
    Init = 0,
    ReadMaskedAddressTag,
    ReadDpfKeyGen,
    ReadLookupFlagLocation,
    ReadComputeLocation,
    ReadReshareFlag,
    ReadConvertToReplicated,
    ReadComputeMaskedIndex,
    ReadDpfKeyDistribution,
    ReadDpfEvaluations,
    WriteAddressTag,
    WriteStoreTriple,
    WriteSelectPreviousValue,
    WriteSelectValue,
    WriteComputeMaskedIndex,
    WriteDpfKeyDistribution,
    WriteDpfEvaluations,
}

/// Collection of accumulated runtimes for the protocol steps.
#[derive(Debug, Default, Clone, Copy)]
pub struct Runtimes {
    durations: [Duration; 17],
}

impl Runtimes {
    /// Add another duration to the accumulated runtimes for a protocol step.
    #[inline(always)]
    pub fn record(&mut self, id: ProtocolStep, duration: Duration) {
        self.durations[id as usize] += duration;
    }

    /// Get the accumulated durations for a protocol step.
    pub fn get(&self, id: ProtocolStep) -> Duration {
        self.durations[id as usize]
    }
}

/// Implementation of the stash protocol.
pub struct StashProtocol<F, SPDPF>
where
    F: PrimeField + LegendreSymbol + Serializable,
    SPDPF: SinglePointDpf<Value = F>,
{
    party_id: usize,
    stash_size: usize,
    access_counter: usize,
    state: State,
    stash_addresses_share: Vec<F>,
    stash_values_share: Vec<F>,
    stash_old_values_share: Vec<F>,
    address_tag_list: Vec<u128>,
    select_party: Option<SelectProtocol<F>>,
    doprf_party_1: Option<DOPrfParty1<F>>,
    doprf_party_2: Option<DOPrfParty2<F>>,
    doprf_party_3: Option<DOPrfParty3<F>>,
    masked_doprf_party_1: Option<MaskedDOPrfParty1<F>>,
    masked_doprf_party_2: Option<MaskedDOPrfParty2<F>>,
    masked_doprf_party_3: Option<MaskedDOPrfParty3<F>>,
    _phantom: PhantomData<SPDPF>,
}

impl<F, SPDPF> StashProtocol<F, SPDPF>
where
    F: PrimeField + LegendreSymbol + Serializable,
    SPDPF: SinglePointDpf<Value = F>,
    SPDPF::Key: Serializable + Sync,
{
    /// Create new instance of the stash protocol for a party `{0, 1, 2}` and given size.
    pub fn new(party_id: usize, stash_size: usize) -> Self {
        assert!(party_id < 3);
        assert!(stash_size > 0);
        assert!(compute_stash_prf_output_bitsize(stash_size) <= 128);

        Self {
            party_id,
            stash_size,
            access_counter: 0,
            state: State::New,
            stash_addresses_share: Vec::with_capacity(stash_size),
            stash_values_share: Vec::with_capacity(stash_size),
            stash_old_values_share: Vec::with_capacity(stash_size),
            address_tag_list: if party_id == PARTY_1 {
                Default::default()
            } else {
                Vec::with_capacity(stash_size)
            },
            select_party: None,
            doprf_party_1: None,
            doprf_party_2: None,
            doprf_party_3: None,
            masked_doprf_party_1: None,
            masked_doprf_party_2: None,
            masked_doprf_party_3: None,
            _phantom: PhantomData,
        }
    }

    fn init_with_runtimes<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        runtimes: Option<Runtimes>,
    ) -> Result<Option<Runtimes>, Error> {
        assert_eq!(self.state, State::New);

        let t_start = Instant::now();

        let prf_output_bitsize = compute_stash_prf_output_bitsize(self.stash_size);
        let legendre_prf_key = LegendrePrf::<F>::key_gen(prf_output_bitsize);

        // run DOPRF initilization
        match self.party_id {
            PARTY_1 => {
                let mut doprf_p1 = DOPrfParty1::from_legendre_prf_key(legendre_prf_key.clone());
                let mut mdoprf_p1 = MaskedDOPrfParty1::from_legendre_prf_key(legendre_prf_key);
                doprf_p1.init(comm)?;
                mdoprf_p1.init(comm)?;
                doprf_p1.preprocess(comm, self.stash_size)?;
                mdoprf_p1.preprocess(comm, self.stash_size)?;
                self.doprf_party_1 = Some(doprf_p1);
                self.masked_doprf_party_1 = Some(mdoprf_p1);
            }
            PARTY_2 => {
                let mut doprf_p2 = DOPrfParty2::new(prf_output_bitsize);
                let mut mdoprf_p2 = MaskedDOPrfParty2::new(prf_output_bitsize);
                doprf_p2.init(comm)?;
                mdoprf_p2.init(comm)?;
                doprf_p2.preprocess(comm, self.stash_size)?;
                mdoprf_p2.preprocess(comm, self.stash_size)?;
                self.doprf_party_2 = Some(doprf_p2);
                self.masked_doprf_party_2 = Some(mdoprf_p2);
            }
            PARTY_3 => {
                let mut doprf_p3 = DOPrfParty3::new(prf_output_bitsize);
                let mut mdoprf_p3 = MaskedDOPrfParty3::new(prf_output_bitsize);
                doprf_p3.init(comm)?;
                mdoprf_p3.init(comm)?;
                doprf_p3.preprocess(comm, self.stash_size)?;
                mdoprf_p3.preprocess(comm, self.stash_size)?;
                self.doprf_party_3 = Some(doprf_p3);
                self.masked_doprf_party_3 = Some(mdoprf_p3);
            }
            _ => panic!("invalid party id"),
        }

        // run Select initialiation and preprocessing
        {
            let mut select_party = SelectProtocol::default();
            select_party.init(comm)?;
            select_party.preprocess(comm, 3 * self.stash_size)?;
            self.select_party = Some(select_party);
        }

        let t_end = Instant::now();
        let runtimes = runtimes.map(|mut r| {
            r.record(ProtocolStep::Init, t_end - t_start);
            r
        });

        self.state = State::AwaitingRead;
        Ok(runtimes)
    }

    /// Perform a stash read and collect runtime data.
    pub fn read_with_runtimes<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        instruction: InstructionShare<F>,
        runtimes: Option<Runtimes>,
    ) -> Result<(StashStateShare<F>, Option<Runtimes>), Error> {
        assert_eq!(self.state, State::AwaitingRead);
        assert!(self.access_counter < self.stash_size);

        // 0. If the stash is empty, we are done
        if self.access_counter == 0 {
            self.state = State::AwaitingWrite;
            return Ok((
                StashStateShare {
                    flag: F::ZERO,
                    location: F::ZERO,
                    value: F::ZERO,
                },
                runtimes,
            ));
        }

        let t_start = Instant::now();

        let (
            flag_share,
            location_share,
            t_after_masked_address_tag,
            t_after_dpf_keygen,
            t_after_compute_flag_loc,
        ) = match self.party_id {
            PARTY_1 => {
                // 1. Compute tag y := PRF(k, <I.adr>) such that P1 obtains y + r and P2, P3 obtain the mask r.
                let masked_address_tag: u128 = {
                    let mdoprf_p1 = self.masked_doprf_party_1.as_mut().unwrap();
                    mdoprf_p1.eval_to_uint(comm, 1, &[instruction.address])?[0]
                };

                let t_after_masked_address_tag = Instant::now();

                // 2. Create and send DPF keys for the function f(x) = if x = y { 1 } else { 0 }
                {
                    let domain_size = 1 << compute_stash_prf_output_bitsize(self.stash_size);
                    let (dpf_key_2, dpf_key_3) =
                        SPDPF::generate_keys(domain_size, masked_address_tag, F::ONE);
                    comm.send(PARTY_2, dpf_key_2)?;
                    comm.send(PARTY_3, dpf_key_3)?;
                }

                let t_after_dpf_keygen = Instant::now();

                // 3. The other parties compute shares of <flag>, <loc>, i.e., if the address is present in
                //    the stash and if so, where it is. We just take 0s as our shares.
                (
                    F::ZERO,
                    F::ZERO,
                    t_after_masked_address_tag,
                    t_after_dpf_keygen,
                    t_after_dpf_keygen,
                )
            }
            PARTY_2 | PARTY_3 => {
                // 1. Compute tag y := PRF(k, <I.adr>) such that P1 obtains y + r and P2, P3 obtain the mask r.
                let address_tag_mask: u128 = match self.party_id {
                    PARTY_2 => {
                        let mdoprf_p2 = self.masked_doprf_party_2.as_mut().unwrap();
                        mdoprf_p2.eval_to_uint(comm, 1, &[instruction.address])?[0]
                    }
                    PARTY_3 => {
                        let mdoprf_p3 = self.masked_doprf_party_3.as_mut().unwrap();
                        mdoprf_p3.eval_to_uint(comm, 1, &[instruction.address])?[0]
                    }
                    _ => panic!("invalid party id"),
                };

                let t_after_masked_address_tag = Instant::now();

                // 2. Receive DPF key for the function f(x) = if x = y { 1 } else { 0 }
                let dpf_key_i: SPDPF::Key = {
                    let fut = comm.receive(PARTY_1)?;
                    fut.get()?
                };

                let t_after_dpf_keygen = Instant::now();

                // 3. Compute shares of <flag>, <loc>, i.e., if the address is present in the stash and if
                //    so, where it is
                {
                    let (flag_share, location_share) = self
                        .address_tag_list
                        .par_iter()
                        .enumerate()
                        .map(|(j, tag_j)| {
                            let dpf_value_j =
                                SPDPF::evaluate_at(&dpf_key_i, tag_j ^ address_tag_mask);
                            (dpf_value_j, F::from_u128(j as u128) * dpf_value_j)
                        })
                        .reduce(|| (F::ZERO, F::ZERO), |(a, b), (c, d)| (a + c, b + d));
                    let t_after_compute_flag_loc = Instant::now();
                    (
                        flag_share,
                        location_share,
                        t_after_masked_address_tag,
                        t_after_dpf_keygen,
                        t_after_compute_flag_loc,
                    )
                }
            }
            _ => panic!("invalid party id"),
        };

        // 4. Compute <loc> = if <flag> { <loc> } else { access_counter - 1 }
        let location_share = {
            let access_counter_share = if self.party_id == PARTY_1 {
                F::from_u128(self.access_counter as u128)
            } else {
                F::ZERO
            };
            self.select_party.as_mut().unwrap().select(
                comm,
                flag_share,
                location_share,
                access_counter_share,
            )?
        };

        let t_after_location_share = Instant::now();

        // 5. Reshare <flag> among all three parties
        let flag_share = match self.party_id {
            PARTY_1 => {
                let flag_share = F::random(thread_rng());
                comm.send(PARTY_2, flag_share)?;
                flag_share
            }
            PARTY_2 => {
                let fut_1_2 = comm.receive::<F>(PARTY_1)?;
                flag_share - fut_1_2.get()?
            }
            _ => flag_share,
        };

        let t_after_flag_share = Instant::now();

        // 6. Read the value <val> from the stash (if <flag>) or read a zero value
        let (
            value_share,
            t_after_convert_to_replicated,
            t_after_masked_index,
            t_after_dpf_key_distr,
        ) = {
            // a) convert the stash into replicated secret sharing
            let fut_prev = comm.receive_previous::<Vec<F>>()?;
            comm.send_slice_next(self.stash_values_share.as_ref())?;
            let stash_values_share_prev = fut_prev.get()?;

            let t_after_convert_to_replicated = Instant::now();

            // b) mask and reconstruct the stash index <loc>
            let index_bits = (self.access_counter as f64).log2().ceil() as u32;
            assert!(index_bits <= 16);
            let bit_mask = ((1 << index_bits) - 1) as u16;
            let (masked_loc, r_prev, r_next) =
                MaskIndexProtocol::mask_index(comm, index_bits, location_share)?;

            let t_after_masked_index = Instant::now();

            // c) use DPFs to read the stash value
            let fut_prev = comm.receive_previous::<SPDPF::Key>()?;
            let fut_next = comm.receive_next::<SPDPF::Key>()?;
            {
                let (dpf_key_prev, dpf_key_next) =
                    SPDPF::generate_keys(1 << index_bits, masked_loc as u128, F::ONE);
                comm.send_previous(dpf_key_prev)?;
                comm.send_next(dpf_key_next)?;
            }
            let dpf_key_prev = fut_prev.get()?;
            let dpf_key_next = fut_next.get()?;
            let t_after_dpf_key_distr = Instant::now();
            let value_share: F = (0..self.access_counter)
                .into_par_iter()
                .map(|j| {
                    let index_prev = ((j as u16 + r_prev) & bit_mask) as u128;
                    let index_next = ((j as u16 + r_next) & bit_mask) as u128;
                    SPDPF::evaluate_at(&dpf_key_prev, index_prev) * self.stash_values_share[j]
                        + SPDPF::evaluate_at(&dpf_key_next, index_next) * stash_values_share_prev[j]
                })
                .sum();
            (
                value_share,
                t_after_convert_to_replicated,
                t_after_masked_index,
                t_after_dpf_key_distr,
            )
        };

        let t_after_dpf_eval = Instant::now();

        let runtimes = runtimes.map(|mut r| {
            r.record(
                ProtocolStep::ReadMaskedAddressTag,
                t_after_masked_address_tag - t_start,
            );
            r.record(
                ProtocolStep::ReadDpfKeyGen,
                t_after_dpf_keygen - t_after_masked_address_tag,
            );
            r.record(
                ProtocolStep::ReadLookupFlagLocation,
                t_after_compute_flag_loc - t_after_dpf_keygen,
            );
            r.record(
                ProtocolStep::ReadComputeLocation,
                t_after_location_share - t_after_compute_flag_loc,
            );
            r.record(
                ProtocolStep::ReadReshareFlag,
                t_after_flag_share - t_after_location_share,
            );
            r.record(
                ProtocolStep::ReadConvertToReplicated,
                t_after_convert_to_replicated - t_after_flag_share,
            );
            r.record(
                ProtocolStep::ReadComputeMaskedIndex,
                t_after_masked_index - t_after_convert_to_replicated,
            );
            r.record(
                ProtocolStep::ReadDpfKeyDistribution,
                t_after_dpf_key_distr - t_after_masked_index,
            );
            r.record(
                ProtocolStep::ReadDpfEvaluations,
                t_after_dpf_eval - t_after_dpf_key_distr,
            );
            r
        });

        self.state = State::AwaitingWrite;
        Ok((
            StashStateShare {
                flag: flag_share,
                location: location_share,
                value: value_share,
            },
            runtimes,
        ))
    }

    /// Perform a stash write and collect runtime data.
    pub fn write_with_runtimes<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        instruction: InstructionShare<F>,
        stash_state: StashStateShare<F>,
        db_address_share: F,
        db_value_share: F,
        runtimes: Option<Runtimes>,
    ) -> Result<Option<Runtimes>, Error> {
        assert_eq!(self.state, State::AwaitingWrite);
        assert!(self.access_counter < self.stash_size);

        let t_start = Instant::now();

        // 1. Compute tag y := PRF(k, <db_adr>) such that P2, P3 obtain y.
        match self.party_id {
            PARTY_1 => {
                let doprf_p1 = self.doprf_party_1.as_mut().unwrap();
                doprf_p1.eval(comm, 1, &[db_address_share])?;
            }
            PARTY_2 => {
                let address_tag: u128 = {
                    let doprf_p2 = self.doprf_party_2.as_mut().unwrap();
                    let fut_3_2 = comm.receive(PARTY_3)?;
                    doprf_p2.eval(comm, 1, &[db_address_share])?;
                    fut_3_2.get()?
                };
                self.address_tag_list.push(address_tag);
            }
            PARTY_3 => {
                let address_tag: u128 = {
                    let doprf_p3 = self.doprf_party_3.as_mut().unwrap();
                    let tag = doprf_p3.eval_to_uint(comm, 1, &[db_address_share])?[0];
                    comm.send(PARTY_2, tag)?;
                    tag
                };
                self.address_tag_list.push(address_tag);
            }
            _ => panic!("invalid party id"),
        }

        let t_after_address_tag = Instant::now();

        // 2. Insert new triple (<db_adr>, <db_val>, <db_val> into stash.
        self.stash_addresses_share.push(db_address_share);
        self.stash_values_share.push(db_value_share);
        self.stash_old_values_share.push(db_value_share);

        let t_after_store_triple = Instant::now();

        // 3. Update stash
        let previous_value_share = self.select_party.as_mut().unwrap().select(
            comm,
            stash_state.flag,
            stash_state.value,
            db_value_share,
        )?;
        let t_after_select_previous_value = Instant::now();
        let value_share = self.select_party.as_mut().unwrap().select(
            comm,
            instruction.operation,
            instruction.value - previous_value_share,
            F::ZERO,
        )?;
        let t_after_select_value = Instant::now();
        let (t_after_masked_index, t_after_dpf_key_distr) = {
            // a) mask and reconstruct the stash index <loc>
            let index_bits = {
                let bits = usize::BITS - self.access_counter.leading_zeros();
                if bits > 0 {
                    bits
                } else {
                    1
                }
            };
            assert!(index_bits <= 16);
            let bit_mask = ((1 << index_bits) - 1) as u16;
            let (masked_loc, r_prev, r_next) =
                MaskIndexProtocol::mask_index(comm, index_bits, stash_state.location)?;

            let t_after_masked_index = Instant::now();

            // b) use DPFs to read the stash value
            let fut_prev = comm.receive_previous::<SPDPF::Key>()?;
            let fut_next = comm.receive_next::<SPDPF::Key>()?;
            {
                let (dpf_key_prev, dpf_key_next) =
                    SPDPF::generate_keys(1 << index_bits, masked_loc as u128, value_share);
                comm.send_previous(dpf_key_prev)?;
                comm.send_next(dpf_key_next)?;
            }
            let dpf_key_prev = fut_prev.get()?;
            let dpf_key_next = fut_next.get()?;
            let t_after_dpf_key_distr = Instant::now();
            self.stash_values_share
                .par_iter_mut()
                .enumerate()
                .for_each(|(j, svs_j)| {
                    let index_prev = ((j as u16).wrapping_add(r_prev) & bit_mask) as u128;
                    let index_next = ((j as u16).wrapping_add(r_next) & bit_mask) as u128;
                    *svs_j += SPDPF::evaluate_at(&dpf_key_prev, index_prev)
                        + SPDPF::evaluate_at(&dpf_key_next, index_next);
                });
            (t_after_masked_index, t_after_dpf_key_distr)
        };
        let t_after_dpf_eval = Instant::now();

        self.access_counter += 1;
        self.state = if self.access_counter == self.stash_size {
            State::AccessesExhausted
        } else {
            State::AwaitingRead
        };

        let runtimes = runtimes.map(|mut r| {
            r.record(ProtocolStep::WriteAddressTag, t_after_address_tag - t_start);
            r.record(
                ProtocolStep::WriteStoreTriple,
                t_after_store_triple - t_after_address_tag,
            );
            r.record(
                ProtocolStep::WriteSelectPreviousValue,
                t_after_select_previous_value - t_after_store_triple,
            );
            r.record(
                ProtocolStep::WriteSelectValue,
                t_after_select_value - t_after_select_previous_value,
            );
            r.record(
                ProtocolStep::WriteComputeMaskedIndex,
                t_after_masked_index - t_after_select_value,
            );
            r.record(
                ProtocolStep::WriteDpfKeyDistribution,
                t_after_dpf_key_distr - t_after_masked_index,
            );
            r.record(
                ProtocolStep::WriteDpfEvaluations,
                t_after_dpf_eval - t_after_dpf_key_distr,
            );
            r
        });

        Ok(runtimes)
    }
}

impl<F, SPDPF> Stash<F> for StashProtocol<F, SPDPF>
where
    F: PrimeField + LegendreSymbol + Serializable,
    SPDPF: SinglePointDpf<Value = F>,
    SPDPF::Key: Serializable + Sync,
{
    fn get_party_id(&self) -> usize {
        self.party_id
    }

    fn get_stash_size(&self) -> usize {
        self.stash_size
    }

    fn get_access_counter(&self) -> usize {
        self.access_counter
    }

    fn reset(&mut self) {
        *self = Self::new(self.party_id, self.stash_size);
    }

    fn init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error> {
        self.init_with_runtimes(comm, None).map(|_| ())
    }

    fn read<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        instruction: InstructionShare<F>,
    ) -> Result<StashStateShare<F>, Error> {
        self.read_with_runtimes(comm, instruction, None)
            .map(|x| x.0)
    }

    fn write<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        instruction: InstructionShare<F>,
        stash_state: StashStateShare<F>,
        db_address_share: F,
        db_value_share: F,
    ) -> Result<(), Error> {
        self.write_with_runtimes(
            comm,
            instruction,
            stash_state,
            db_address_share,
            db_value_share,
            None,
        )
        .map(|_| ())
    }

    fn get_stash_share(&self) -> (&[F], &[F], &[F]) {
        (
            &self.stash_addresses_share,
            &self.stash_values_share,
            &self.stash_old_values_share,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Operation;
    use communicator::unix::make_unix_communicators;
    use dpf::spdpf::DummySpDpf;
    use ff::Field;
    use std::thread;
    use utils::field::Fp;

    fn run_init<F>(
        mut stash_party: impl Stash<F> + Send + 'static,
        mut comm: impl AbstractCommunicator + Send + 'static,
    ) -> thread::JoinHandle<(impl Stash<F>, impl AbstractCommunicator)>
    where
        F: PrimeField + LegendreSymbol,
    {
        thread::spawn(move || {
            stash_party.init(&mut comm).unwrap();
            (stash_party, comm)
        })
    }

    fn run_read<F>(
        mut stash_party: impl Stash<F> + Send + 'static,
        mut comm: impl AbstractCommunicator + Send + 'static,
        instruction: InstructionShare<F>,
    ) -> thread::JoinHandle<(impl Stash<F>, impl AbstractCommunicator, StashStateShare<F>)>
    where
        F: PrimeField + LegendreSymbol,
    {
        thread::spawn(move || {
            let result = stash_party.read(&mut comm, instruction);
            (stash_party, comm, result.unwrap())
        })
    }

    fn run_write<F>(
        mut stash_party: impl Stash<F> + Send + 'static,
        mut comm: impl AbstractCommunicator + Send + 'static,
        instruction: InstructionShare<F>,
        stash_state: StashStateShare<F>,
        db_address_share: F,
        db_value_share: F,
    ) -> thread::JoinHandle<(impl Stash<F>, impl AbstractCommunicator)>
    where
        F: PrimeField + LegendreSymbol,
    {
        thread::spawn(move || {
            stash_party
                .write(
                    &mut comm,
                    instruction,
                    stash_state,
                    db_address_share,
                    db_value_share,
                )
                .unwrap();
            (stash_party, comm)
        })
    }

    #[test]
    fn test_stash() {
        type SPDPF = DummySpDpf<Fp>;

        let stash_size = 128;
        let mut num_accesses = 0;

        let party_1 = StashProtocol::<Fp, SPDPF>::new(PARTY_1, stash_size);
        let party_2 = StashProtocol::<Fp, SPDPF>::new(PARTY_2, stash_size);
        let party_3 = StashProtocol::<Fp, SPDPF>::new(PARTY_3, stash_size);
        assert_eq!(party_1.get_party_id(), PARTY_1);
        assert_eq!(party_2.get_party_id(), PARTY_2);
        assert_eq!(party_3.get_party_id(), PARTY_3);
        assert_eq!(party_1.get_stash_size(), stash_size);
        assert_eq!(party_2.get_stash_size(), stash_size);
        assert_eq!(party_3.get_stash_size(), stash_size);

        let (comm_3, comm_2, comm_1) = {
            let mut comms = make_unix_communicators(3);
            (
                comms.pop().unwrap(),
                comms.pop().unwrap(),
                comms.pop().unwrap(),
            )
        };

        let h1 = run_init(party_1, comm_1);
        let h2 = run_init(party_2, comm_2);
        let h3 = run_init(party_3, comm_3);
        let (party_1, comm_1) = h1.join().unwrap();
        let (party_2, comm_2) = h2.join().unwrap();
        let (party_3, comm_3) = h3.join().unwrap();

        assert_eq!(party_1.get_access_counter(), 0);
        assert_eq!(party_2.get_access_counter(), 0);
        assert_eq!(party_3.get_access_counter(), 0);

        // write a value 42 to address adr = 3
        let value = 42;
        let address = 3;
        let inst_w3_1 = InstructionShare {
            operation: Operation::Write.encode(),
            address: Fp::from_u128(address),
            value: Fp::from_u128(value),
        };
        let inst_w3_2 = InstructionShare {
            operation: Fp::ZERO,
            address: Fp::ZERO,
            value: Fp::ZERO,
        };
        let inst_w3_3 = inst_w3_2.clone();

        let h1 = run_read(party_1, comm_1, inst_w3_1);
        let h2 = run_read(party_2, comm_2, inst_w3_2);
        let h3 = run_read(party_3, comm_3, inst_w3_3);
        let (party_1, comm_1, state_1) = h1.join().unwrap();
        let (party_2, comm_2, state_2) = h2.join().unwrap();
        let (party_3, comm_3, state_3) = h3.join().unwrap();

        // since the stash is empty, st.flag must be zero
        assert_eq!(state_1.flag + state_2.flag + state_3.flag, Fp::ZERO);
        assert_eq!(
            state_1.location + state_2.location + state_3.location,
            Fp::ZERO
        );

        let h1 = run_write(
            party_1,
            comm_1,
            inst_w3_1,
            state_1,
            inst_w3_1.address,
            Fp::from_u128(0x71),
        );
        let h2 = run_write(
            party_2,
            comm_2,
            inst_w3_2,
            state_1,
            inst_w3_2.address,
            Fp::from_u128(0x72),
        );
        let h3 = run_write(
            party_3,
            comm_3,
            inst_w3_3,
            state_1,
            inst_w3_3.address,
            Fp::from_u128(0x73),
        );
        let (party_1, comm_1) = h1.join().unwrap();
        let (party_2, comm_2) = h2.join().unwrap();
        let (party_3, comm_3) = h3.join().unwrap();

        num_accesses += 1;

        assert_eq!(party_1.get_access_counter(), num_accesses);
        assert_eq!(party_2.get_access_counter(), num_accesses);
        assert_eq!(party_3.get_access_counter(), num_accesses);

        {
            let (st_adrs_1, st_vals_1, st_old_vals_1) = party_1.get_stash_share();
            let (st_adrs_2, st_vals_2, st_old_vals_2) = party_2.get_stash_share();
            let (st_adrs_3, st_vals_3, st_old_vals_3) = party_3.get_stash_share();
            assert_eq!(st_adrs_1.len(), num_accesses);
            assert_eq!(st_vals_1.len(), num_accesses);
            assert_eq!(st_old_vals_1.len(), num_accesses);
            assert_eq!(st_adrs_2.len(), num_accesses);
            assert_eq!(st_vals_2.len(), num_accesses);
            assert_eq!(st_old_vals_2.len(), num_accesses);
            assert_eq!(st_adrs_3.len(), num_accesses);
            assert_eq!(st_vals_3.len(), num_accesses);
            assert_eq!(st_old_vals_3.len(), num_accesses);
            assert_eq!(
                st_adrs_1[0] + st_adrs_2[0] + st_adrs_3[0],
                Fp::from_u128(address)
            );
            assert_eq!(
                st_vals_1[0] + st_vals_2[0] + st_vals_3[0],
                Fp::from_u128(value)
            );
        }

        // read again from address adr = 3, we should get the value 42 back
        let inst_r3_1 = InstructionShare {
            operation: Operation::Read.encode(),
            address: Fp::from_u128(3),
            value: Fp::ZERO,
        };
        let inst_r3_2 = InstructionShare {
            operation: Fp::ZERO,
            address: Fp::ZERO,
            value: Fp::ZERO,
        };
        let inst_r3_3 = inst_r3_2.clone();

        let h1 = run_read(party_1, comm_1, inst_r3_1);
        let h2 = run_read(party_2, comm_2, inst_r3_2);
        let h3 = run_read(party_3, comm_3, inst_r3_3);
        let (party_1, comm_1, state_1) = h1.join().unwrap();
        let (party_2, comm_2, state_2) = h2.join().unwrap();
        let (party_3, comm_3, state_3) = h3.join().unwrap();

        let st_flag = state_1.flag + state_2.flag + state_3.flag;
        let st_location = state_1.location + state_2.location + state_3.location;
        let st_value = state_1.value + state_2.value + state_3.value;
        assert_eq!(st_flag, Fp::ONE);
        assert_eq!(st_location, Fp::from_u128(0));
        assert_eq!(st_value, Fp::from_u128(value));

        let h1 = run_write(
            party_1,
            comm_1,
            inst_r3_1,
            state_1,
            Fp::from_u128(0x83),
            Fp::from_u128(0x93),
        );
        let h2 = run_write(
            party_2,
            comm_2,
            inst_r3_2,
            state_1,
            Fp::from_u128(0x83),
            Fp::from_u128(0x93),
        );
        let h3 = run_write(
            party_3,
            comm_3,
            inst_r3_3,
            state_1,
            Fp::from_u128(0x83),
            Fp::from_u128(0x93),
        );
        let (party_1, comm_1) = h1.join().unwrap();
        let (party_2, comm_2) = h2.join().unwrap();
        let (party_3, comm_3) = h3.join().unwrap();

        num_accesses += 1;

        assert_eq!(party_1.get_access_counter(), num_accesses);
        assert_eq!(party_2.get_access_counter(), num_accesses);
        assert_eq!(party_3.get_access_counter(), num_accesses);

        {
            let (st_adrs_1, st_vals_1, st_old_vals_1) = party_1.get_stash_share();
            let (st_adrs_2, st_vals_2, st_old_vals_2) = party_2.get_stash_share();
            let (st_adrs_3, st_vals_3, st_old_vals_3) = party_3.get_stash_share();
            assert_eq!(st_adrs_1.len(), num_accesses);
            assert_eq!(st_vals_1.len(), num_accesses);
            assert_eq!(st_old_vals_1.len(), num_accesses);
            assert_eq!(st_adrs_2.len(), num_accesses);
            assert_eq!(st_vals_2.len(), num_accesses);
            assert_eq!(st_old_vals_2.len(), num_accesses);
            assert_eq!(st_adrs_3.len(), num_accesses);
            assert_eq!(st_vals_3.len(), num_accesses);
            assert_eq!(st_old_vals_3.len(), num_accesses);
            assert_eq!(
                st_adrs_1[0] + st_adrs_2[0] + st_adrs_3[0],
                Fp::from_u128(address)
            );
            assert_eq!(
                st_vals_1[0] + st_vals_2[0] + st_vals_3[0],
                Fp::from_u128(value)
            );
        }

        // now write a value 0x1337 to address adr = 3
        let old_value = value;
        let value = 0x1337;
        let address = 3;
        let inst_w3_1 = InstructionShare {
            operation: Operation::Write.encode(),
            address: Fp::from_u128(address),
            value: Fp::from_u128(value),
        };
        let inst_w3_2 = InstructionShare {
            operation: Fp::ZERO,
            address: Fp::ZERO,
            value: Fp::ZERO,
        };
        let inst_w3_3 = inst_w3_2.clone();

        let h1 = run_read(party_1, comm_1, inst_w3_1);
        let h2 = run_read(party_2, comm_2, inst_w3_2);
        let h3 = run_read(party_3, comm_3, inst_w3_3);
        let (party_1, comm_1, state_1) = h1.join().unwrap();
        let (party_2, comm_2, state_2) = h2.join().unwrap();
        let (party_3, comm_3, state_3) = h3.join().unwrap();

        // since we already wrote to the address, it should be present in the stash
        assert_eq!(state_1.flag + state_2.flag + state_3.flag, Fp::ONE);
        assert_eq!(
            state_1.location + state_2.location + state_3.location,
            Fp::ZERO
        );
        assert_eq!(
            state_1.value + state_2.value + state_3.value,
            Fp::from_u128(old_value)
        );

        let h1 = run_write(
            party_1,
            comm_1,
            inst_w3_1,
            state_1,
            // inst_w3_1.address,
            Fp::from_u128(0x61),
            Fp::from_u128(0x71),
        );
        let h2 = run_write(
            party_2,
            comm_2,
            inst_w3_2,
            state_2,
            // inst_w3_2.address,
            Fp::from_u128(0x62),
            Fp::from_u128(0x72),
        );
        let h3 = run_write(
            party_3,
            comm_3,
            inst_w3_3,
            state_3,
            // inst_w3_3.address,
            Fp::from_u128(0x63),
            Fp::from_u128(0x73),
        );
        let (party_1, comm_1) = h1.join().unwrap();
        let (party_2, comm_2) = h2.join().unwrap();
        let (party_3, comm_3) = h3.join().unwrap();

        num_accesses += 1;

        assert_eq!(party_1.get_access_counter(), num_accesses);
        assert_eq!(party_2.get_access_counter(), num_accesses);
        assert_eq!(party_3.get_access_counter(), num_accesses);

        {
            let (st_adrs_1, st_vals_1, st_old_vals_1) = party_1.get_stash_share();
            let (st_adrs_2, st_vals_2, st_old_vals_2) = party_2.get_stash_share();
            let (st_adrs_3, st_vals_3, st_old_vals_3) = party_3.get_stash_share();
            assert_eq!(st_adrs_1.len(), num_accesses);
            assert_eq!(st_vals_1.len(), num_accesses);
            assert_eq!(st_old_vals_1.len(), num_accesses);
            assert_eq!(st_adrs_2.len(), num_accesses);
            assert_eq!(st_vals_2.len(), num_accesses);
            assert_eq!(st_old_vals_2.len(), num_accesses);
            assert_eq!(st_adrs_3.len(), num_accesses);
            assert_eq!(st_vals_3.len(), num_accesses);
            assert_eq!(st_old_vals_3.len(), num_accesses);
            assert_eq!(
                st_adrs_1[0] + st_adrs_2[0] + st_adrs_3[0],
                Fp::from_u128(address)
            );
            assert_eq!(
                st_vals_1[0] + st_vals_2[0] + st_vals_3[0],
                Fp::from_u128(value)
            );
        }

        // read again from address adr = 3, we should get the value 0x1337 back
        let inst_r3_1 = InstructionShare {
            operation: Operation::Read.encode(),
            address: Fp::from_u128(3),
            value: Fp::ZERO,
        };
        let inst_r3_2 = InstructionShare {
            operation: Fp::ZERO,
            address: Fp::ZERO,
            value: Fp::ZERO,
        };
        let inst_r3_3 = inst_r3_2.clone();

        let h1 = run_read(party_1, comm_1, inst_r3_1);
        let h2 = run_read(party_2, comm_2, inst_r3_2);
        let h3 = run_read(party_3, comm_3, inst_r3_3);
        let (party_1, comm_1, state_1) = h1.join().unwrap();
        let (party_2, comm_2, state_2) = h2.join().unwrap();
        let (party_3, comm_3, state_3) = h3.join().unwrap();

        let st_flag = state_1.flag + state_2.flag + state_3.flag;
        let st_location = state_1.location + state_2.location + state_3.location;
        let st_value = state_1.value + state_2.value + state_3.value;
        assert_eq!(st_flag, Fp::ONE);
        assert_eq!(st_location, Fp::from_u128(0));
        assert_eq!(st_value, Fp::from_u128(value));

        let h1 = run_write(
            party_1,
            comm_1,
            inst_r3_1,
            state_1,
            Fp::from_u128(0x83),
            Fp::from_u128(0x93),
        );
        let h2 = run_write(
            party_2,
            comm_2,
            inst_r3_2,
            state_2,
            Fp::from_u128(0x83),
            Fp::from_u128(0x93),
        );
        let h3 = run_write(
            party_3,
            comm_3,
            inst_r3_3,
            state_3,
            Fp::from_u128(0x83),
            Fp::from_u128(0x93),
        );
        let (party_1, _comm_1) = h1.join().unwrap();
        let (party_2, _comm_2) = h2.join().unwrap();
        let (party_3, _comm_3) = h3.join().unwrap();

        num_accesses += 1;

        assert_eq!(party_1.get_access_counter(), num_accesses);
        assert_eq!(party_2.get_access_counter(), num_accesses);
        assert_eq!(party_3.get_access_counter(), num_accesses);

        {
            let (st_adrs_1, st_vals_1, st_old_vals_1) = party_1.get_stash_share();
            let (st_adrs_2, st_vals_2, st_old_vals_2) = party_2.get_stash_share();
            let (st_adrs_3, st_vals_3, st_old_vals_3) = party_3.get_stash_share();
            assert_eq!(st_adrs_1.len(), num_accesses);
            assert_eq!(st_vals_1.len(), num_accesses);
            assert_eq!(st_old_vals_1.len(), num_accesses);
            assert_eq!(st_adrs_2.len(), num_accesses);
            assert_eq!(st_vals_2.len(), num_accesses);
            assert_eq!(st_old_vals_2.len(), num_accesses);
            assert_eq!(st_adrs_3.len(), num_accesses);
            assert_eq!(st_vals_3.len(), num_accesses);
            assert_eq!(st_old_vals_3.len(), num_accesses);
            assert_eq!(
                st_adrs_1[0] + st_adrs_2[0] + st_adrs_3[0],
                Fp::from_u128(address)
            );
            assert_eq!(
                st_vals_1[0] + st_vals_2[0] + st_vals_3[0],
                Fp::from_u128(value)
            );
        }
    }
}
