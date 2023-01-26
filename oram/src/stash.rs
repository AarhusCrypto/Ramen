use crate::common::{Error, InstructionShare};
use crate::doprf::{
    DOPrfParty1, DOPrfParty2, DOPrfParty3, LegendrePrf, MaskedDOPrfParty1, MaskedDOPrfParty2,
    MaskedDOPrfParty3,
};
use crate::mask_index::{MaskIndex, MaskIndexProtocol};
use crate::select::{Select, SelectProtocol};
use bitvec;
use communicator::{AbstractCommunicator, Fut, Serializable};
use dpf::spdpf::SinglePointDpf;
use ff::PrimeField;
use rand::thread_rng;
use std::io::Read;
use std::marker::PhantomData;
use utils::field::LegendreSymbol;

type BitVec = bitvec::vec::BitVec<u8>;
// type BitSlice = bitvec::slice::BitSlice<u8>;

#[derive(Clone, Copy, Debug, Default)]
pub struct StashEntryShare<F: PrimeField> {
    pub address: F,
    pub value: F,
    pub old_value: F,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StashStateShare<F: PrimeField> {
    pub flag: F,
    pub location: F,
    pub value: F,
}

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

pub trait Stash<F: PrimeField> {
    fn get_party_id(&self) -> usize;

    fn get_stash_size(&self) -> usize;

    fn get_access_counter(&self) -> usize;

    fn init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error>;

    fn read<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        instruction: InstructionShare<F>,
    ) -> Result<StashStateShare<F>, Error>;

    fn write<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        instruction: InstructionShare<F>,
        stash_state: StashStateShare<F>,
        db_address_share: F,
        db_value_share: F,
    ) -> Result<(), Error>;

    fn get_stash_share(&self) -> (&[F], &[F], &[F]);
}

fn compute_stash_prf_output_bitsize(stash_size: usize) -> usize {
    (stash_size as f64).log2().ceil() as usize + 40
}

fn bits_to_u64(mut bits: BitVec) -> u64 {
    assert!(bits.len() <= 64);
    let mut bytes = [0u8; 8];
    bits.force_align(); // important! otherwise, the first bit in the bitvec might not be the lsb
                        // of the first byte in the underlying byte vector
    bits.read(&mut bytes).unwrap();
    u64::from_le_bytes(bytes)
}

fn stash_read_value<C, F, SPDPF>(
    comm: &mut C,
    access_counter: usize,
    location_share: F,
    stash_values_share_mine: &[F],
) -> Result<F, Error>
where
    C: AbstractCommunicator,
    F: PrimeField + Serializable,
    SPDPF: SinglePointDpf<Value = F>,
    SPDPF::Key: Serializable,
{
    // a) convert the stash into replicated secret sharing
    let fut_prev = comm.receive_previous::<Vec<F>>()?;
    comm.send_next(stash_values_share_mine.to_vec())?;
    let stash_values_share_prev = fut_prev.get()?;

    // b) mask and reconstruct the stash index <loc>
    let index_bits = (access_counter as f64).log2().ceil() as u32;
    assert!(index_bits <= 16);
    let bit_mask = ((1 << index_bits) - 1) as u16;
    let (masked_loc, r_prev, r_next) =
        MaskIndexProtocol::mask_index(comm, index_bits, location_share)?;

    // c) use DPFs to read the stash value
    let fut_prev = comm.receive_previous::<SPDPF::Key>()?;
    let fut_next = comm.receive_next::<SPDPF::Key>()?;
    {
        let (dpf_key_prev, dpf_key_next) =
            SPDPF::generate_keys(1 << index_bits, masked_loc as u64, F::ONE);
        comm.send_previous(dpf_key_prev)?;
        comm.send_next(dpf_key_next)?;
    }
    let dpf_key_prev = fut_prev.get()?;
    let dpf_key_next = fut_next.get()?;
    let mut value_share = F::ZERO;
    for j in 0..access_counter {
        let index_prev = ((j as u16 + r_prev) & bit_mask) as u64;
        let index_next = ((j as u16 + r_next) & bit_mask) as u64;
        value_share += SPDPF::evaluate_at(&dpf_key_prev, index_prev) * stash_values_share_mine[j];
        value_share += SPDPF::evaluate_at(&dpf_key_next, index_next) * stash_values_share_prev[j];
    }
    Ok(value_share)
}

fn stash_write_value<C, F, SPDPF>(
    comm: &mut C,
    access_counter: usize,
    location_share: F,
    // old_value_share: F,
    value_share: F,
    stash_values_share_mine: &mut [F],
) -> Result<(), Error>
where
    C: AbstractCommunicator,
    F: PrimeField + Serializable,
    SPDPF: SinglePointDpf<Value = F>,
    SPDPF::Key: Serializable,
{
    // a) mask and reconstruct the stash index <loc>
    let index_bits = {
        let bits = (access_counter as f64).log2().ceil() as u32;
        if bits > 0 {
            bits
        } else {
            1
        }
    };
    assert!(index_bits <= 16);
    let bit_mask = ((1 << index_bits) - 1) as u16;
    let (masked_loc, r_prev, r_next) =
        MaskIndexProtocol::mask_index(comm, index_bits, location_share)?;

    eprintln!(
        "Party {}: masked index = {}, r_prev = {}, r_next = {}    ({} bits)",
        comm.get_my_id(),
        masked_loc,
        r_prev,
        r_next,
        index_bits
    );
    // b) use DPFs to read the stash value
    let fut_prev = comm.receive_previous::<SPDPF::Key>()?;
    let fut_next = comm.receive_next::<SPDPF::Key>()?;
    {
        let (dpf_key_prev, dpf_key_next) =
            SPDPF::generate_keys(1 << index_bits, masked_loc as u64, value_share);
        comm.send_previous(dpf_key_prev)?;
        comm.send_next(dpf_key_next)?;
    }
    let dpf_key_prev = fut_prev.get()?;
    let dpf_key_next = fut_next.get()?;
    for j in 0..=access_counter {
        let index_prev = ((j as u16 + r_prev) & bit_mask) as u64;
        let index_next = ((j as u16 + r_next) & bit_mask) as u64;
        stash_values_share_mine[j] += SPDPF::evaluate_at(&dpf_key_prev, index_prev);
        stash_values_share_mine[j] += SPDPF::evaluate_at(&dpf_key_next, index_next);
    }
    Ok(())
}

pub struct StashParty1<F, SPDPF>
where
    F: PrimeField + LegendreSymbol + Serializable,
    SPDPF: SinglePointDpf<Value = F>,
{
    stash_size: usize,
    access_counter: usize,
    state: State,
    stash_addresses_share: Vec<F>,
    stash_values_share: Vec<F>,
    stash_old_values_share: Vec<F>,
    doprf_party_1: Option<DOPrfParty1<F>>,
    masked_doprf_party_1: Option<MaskedDOPrfParty1<F>>,
    _phantom: PhantomData<SPDPF>,
}

impl<F, SPDPF> StashParty1<F, SPDPF>
where
    F: PrimeField + LegendreSymbol + Serializable,
    SPDPF: SinglePointDpf<Value = F>,
{
    pub fn new(stash_size: usize) -> Self {
        assert!(stash_size > 0);
        assert!(compute_stash_prf_output_bitsize(stash_size) <= 64);

        Self {
            stash_size,
            access_counter: 0,
            state: State::New,
            stash_addresses_share: Vec::with_capacity(stash_size),
            stash_values_share: Vec::with_capacity(stash_size),
            stash_old_values_share: Vec::with_capacity(stash_size),
            doprf_party_1: None,
            masked_doprf_party_1: None,
            _phantom: PhantomData,
        }
    }
}

impl<F, SPDPF> Stash<F> for StashParty1<F, SPDPF>
where
    F: PrimeField + LegendreSymbol + Serializable,
    SPDPF: SinglePointDpf<Value = F>,
    SPDPF::Key: Serializable,
{
    fn get_party_id(&self) -> usize {
        1
    }

    fn get_stash_size(&self) -> usize {
        self.stash_size
    }

    fn get_access_counter(&self) -> usize {
        self.access_counter
    }

    fn init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error> {
        assert_eq!(self.state, State::New);

        let prf_output_bitsize = compute_stash_prf_output_bitsize(self.stash_size);
        let legendre_prf_key = LegendrePrf::<F>::key_gen(prf_output_bitsize);
        self.doprf_party_1 = Some(DOPrfParty1::from_legendre_prf_key(legendre_prf_key.clone()));
        self.masked_doprf_party_1 =
            Some(MaskedDOPrfParty1::from_legendre_prf_key(legendre_prf_key));

        // run DOPRF initilization
        {
            let doprf_p1 = self.doprf_party_1.as_mut().unwrap();
            doprf_p1.init(comm)?;
            let mdoprf_p1 = self.masked_doprf_party_1.as_mut().unwrap();
            mdoprf_p1.init(comm)?;
        }

        // panic!("not implemented");
        self.state = State::AwaitingRead;
        Ok(())
    }

    fn read<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        instruction: InstructionShare<F>,
    ) -> Result<StashStateShare<F>, Error> {
        assert_eq!(self.state, State::AwaitingRead);
        assert!(self.access_counter < self.stash_size);

        // 0. If the stash is empty, we are done
        if self.access_counter == 0 {
            self.state = State::AwaitingWrite;
            return Ok(StashStateShare {
                flag: F::ZERO,
                location: F::ZERO,
                value: F::ZERO,
            });
        }

        // 1. Compute tag y := PRF(k, <I.adr>) such that P1 obtains y + r and P2, P3 obtain the mask r.
        let masked_address_tag = {
            let mdoprf_p1 = self.masked_doprf_party_1.as_mut().unwrap();
            // for now do preprocessing on the fly
            mdoprf_p1.preprocess(comm, 1)?;
            let mut masked_tag = mdoprf_p1.eval(comm, 1, &[instruction.address])?;
            bits_to_u64(masked_tag.pop().unwrap())
        };

        // 2. Create and send DPF keys for the function f(x) = if x = y { 1 } else { 0 }
        {
            let domain_size = 1 << compute_stash_prf_output_bitsize(self.stash_size);
            let (dpf_key_2, dpf_key_3) =
                SPDPF::generate_keys(domain_size, masked_address_tag, F::ONE);
            comm.send(PARTY_2, dpf_key_2)?;
            comm.send(PARTY_3, dpf_key_3)?;
        }

        // 3. The other parties compute shares of <flag>, <loc>, i.e., if the address is present in
        //    the stash and if so, where it is

        // 4. Compute <loc> = if <flag> { <loc> } else { access_counter - 1 }
        let location_share = SelectProtocol::select(
            comm,
            F::ZERO,
            F::ZERO,
            F::from_u128(self.access_counter as u128),
        )?;

        // 5. Reshare <flag> among all three parties
        let flag_share = {
            let flag_share = F::random(thread_rng());
            comm.send(PARTY_2, flag_share)?;
            flag_share
        };

        // 6. Read the value <val> from the stash (if <flag>) or read a zero value
        let value_share = stash_read_value::<C, F, SPDPF>(
            comm,
            self.access_counter,
            location_share,
            &self.stash_values_share,
        )?;

        // TODO: handle an empty stash differently
        self.state = State::AwaitingWrite;
        Ok(StashStateShare {
            flag: flag_share,
            location: location_share,
            value: value_share,
        })
    }

    fn write<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        instruction: InstructionShare<F>,
        stash_state: StashStateShare<F>,
        db_address_share: F,
        db_value_share: F,
    ) -> Result<(), Error> {
        assert_eq!(self.state, State::AwaitingWrite);
        assert!(self.access_counter < self.stash_size);

        // 1. Compute tag y := PRF(k, <db_adr>) such that P2, P3 obtain y.
        {
            let doprf_p1 = self.doprf_party_1.as_mut().unwrap();
            // for now do preprocessing on the fly
            doprf_p1.preprocess(comm, 1)?;
            doprf_p1.eval(comm, 1, &[db_address_share])?;
        };

        // 2. Insert new triple (<db_adr>, <db_val>, <db_val> into stash.
        self.stash_addresses_share.push(db_address_share);
        self.stash_values_share.push(db_value_share);
        self.stash_old_values_share.push(db_value_share);

        // 3. Update stash
        // - if I.op = write, we want to write I.val to index loc
        // - if I.op = read, we need to write db_val to index c
        //   (since that has been done already in step 2, it is essentially a no-op)
        {
            let previous_value_share =
                SelectProtocol::select(comm, stash_state.flag, stash_state.value, db_value_share)?;
            let location_share =
                SelectProtocol::select(comm, instruction.operation, stash_state.location, -F::ONE)?;
            let value_share = SelectProtocol::select(
                comm,
                instruction.operation,
                instruction.value - previous_value_share,
                F::ZERO,
            )?;
            // eprintln!(
            //     "P{}: prev_val = {:?}, loc = {:?}, upd = {:?}, vals = {:?}",
            //     comm.get_my_id() + 1,
            //     previous_value_share,
            //     location_share,
            //     value_share,
            //     self.stash_values_share
            // );
            stash_write_value::<C, F, SPDPF>(
                comm,
                self.access_counter,
                location_share,
                value_share,
                &mut self.stash_values_share,
            )?;
            // eprintln!(
            //     "P{}: prev_val = {:?}, loc = {:?}, upd = {:?}, vals = {:?}",
            //     comm.get_my_id() + 1,
            //     previous_value_share,
            //     location_share,
            //     value_share,
            //     self.stash_values_share
            // );
        }

        // todo!("not implemented");
        self.access_counter += 1;
        self.state = if self.access_counter == self.stash_size {
            State::AccessesExhausted
        } else {
            State::AwaitingRead
        };
        Ok(())
    }

    fn get_stash_share(&self) -> (&[F], &[F], &[F]) {
        (
            &self.stash_addresses_share,
            &self.stash_values_share,
            &self.stash_old_values_share,
        )
    }
}

pub struct StashParty2<F, SPDPF>
where
    F: PrimeField + LegendreSymbol + Serializable,
    SPDPF: SinglePointDpf<Value = F>,
{
    stash_size: usize,
    access_counter: usize,
    state: State,
    stash_addresses_share: Vec<F>,
    stash_values_share: Vec<F>,
    stash_old_values_share: Vec<F>,
    address_tag_list: Vec<u64>,
    doprf_party_2: Option<DOPrfParty2<F>>,
    masked_doprf_party_2: Option<MaskedDOPrfParty2<F>>,
    _phantom: PhantomData<SPDPF>,
}

impl<F, SPDPF> StashParty2<F, SPDPF>
where
    F: PrimeField + LegendreSymbol + Serializable,
    SPDPF: SinglePointDpf<Value = F>,
{
    pub fn new(stash_size: usize) -> Self {
        assert!(stash_size > 0);
        assert!(compute_stash_prf_output_bitsize(stash_size) <= 64);

        Self {
            stash_size,
            access_counter: 0,
            state: State::New,
            stash_addresses_share: Vec::with_capacity(stash_size),
            stash_values_share: Vec::with_capacity(stash_size),
            stash_old_values_share: Vec::with_capacity(stash_size),
            address_tag_list: Vec::with_capacity(stash_size),
            doprf_party_2: None,
            masked_doprf_party_2: None,
            _phantom: PhantomData,
        }
    }
}

impl<F, SPDPF> Stash<F> for StashParty2<F, SPDPF>
where
    F: PrimeField + LegendreSymbol + Serializable,
    SPDPF: SinglePointDpf<Value = F>,
    SPDPF::Key: Serializable,
{
    fn get_party_id(&self) -> usize {
        2
    }

    fn get_stash_size(&self) -> usize {
        self.stash_size
    }

    fn get_access_counter(&self) -> usize {
        self.access_counter
    }

    fn init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error> {
        assert_eq!(self.state, State::New);

        let prf_output_bitsize = compute_stash_prf_output_bitsize(self.stash_size);
        self.doprf_party_2 = Some(DOPrfParty2::new(prf_output_bitsize));
        self.masked_doprf_party_2 = Some(MaskedDOPrfParty2::new(prf_output_bitsize));

        // run DOPRF initilization
        {
            let doprf_p2 = self.doprf_party_2.as_mut().unwrap();
            doprf_p2.init(comm)?;
            let mdoprf_p2 = self.masked_doprf_party_2.as_mut().unwrap();
            mdoprf_p2.init(comm)?;
        }

        // panic!("not implemented");
        self.state = State::AwaitingRead;
        Ok(())
    }

    fn read<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        instruction: InstructionShare<F>,
    ) -> Result<StashStateShare<F>, Error> {
        assert_eq!(self.state, State::AwaitingRead);
        assert!(self.access_counter < self.stash_size);

        // 0. If the stash is empty, we are done
        if self.access_counter == 0 {
            self.state = State::AwaitingWrite;
            return Ok(StashStateShare {
                flag: F::ZERO,
                location: F::ZERO,
                value: F::ZERO,
            });
        }

        // 1. Compute tag y := PRF(k, <I.adr>) such that P1 obtains y + r and P2, P3 obtain the mask r.
        let address_tag_mask = {
            let mdoprf_p2 = self.masked_doprf_party_2.as_mut().unwrap();
            // for now do preprocessing on the fly
            mdoprf_p2.preprocess(comm, 1)?;
            let mut mask = mdoprf_p2.eval(comm, 1, &[instruction.address])?;
            bits_to_u64(mask.pop().unwrap())
        };

        // 2. Receive DPF key for the function f(x) = if x = y { 1 } else { 0 }
        let dpf_key_2: SPDPF::Key = {
            let fut = comm.receive(PARTY_1)?;
            fut.get()?
        };

        // 3. Compute shares of <flag>, <loc>, i.e., if the address is present in the stash and if
        //    so, where it is
        let (flag_share, location_share) = {
            let mut flag_share = F::ZERO;
            let mut location_share = F::ZERO;
            let mut j_as_field_element = F::ZERO;
            for j in 0..self.address_tag_list.len() {
                let dpf_value_j =
                    SPDPF::evaluate_at(&dpf_key_2, self.address_tag_list[j] ^ address_tag_mask);
                flag_share += dpf_value_j;
                location_share += j_as_field_element * dpf_value_j;
                j_as_field_element += F::ONE;
            }
            (flag_share, location_share)
        };

        // 4. Compute <loc> = if <flag> { <loc> } else { access_counter - 1 }
        let location_share = SelectProtocol::select(comm, flag_share, location_share, F::ZERO)?;

        // 5. Reshare <flag> among all three parties
        let flag_share = {
            let fut_1_2 = comm.receive::<F>(PARTY_1)?;
            flag_share - fut_1_2.get()?
        };

        // 6. Read the value <val> from the stash (if <flag>) or read a zero value
        let value_share = stash_read_value::<C, F, SPDPF>(
            comm,
            self.access_counter,
            location_share,
            &self.stash_values_share,
        )?;

        self.state = State::AwaitingWrite;
        Ok(StashStateShare {
            flag: flag_share,
            location: location_share,
            value: value_share,
        })
    }

    fn write<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        instruction: InstructionShare<F>,
        stash_state: StashStateShare<F>,
        db_address_share: F,
        db_value_share: F,
    ) -> Result<(), Error> {
        assert_eq!(self.state, State::AwaitingWrite);
        assert!(self.access_counter < self.stash_size);

        // 1. Compute tag y := PRF(k, <db_adr>) such that P2, P3 obtain y and append y to the tag
        //    list.
        let address_tag: u64 = {
            let doprf_p2 = self.doprf_party_2.as_mut().unwrap();
            // for now do preprocessing on the fly
            doprf_p2.preprocess(comm, 1)?;
            let fut_3_2 = comm.receive(PARTY_3)?;
            doprf_p2.eval(comm, 1, &[db_address_share])?;
            fut_3_2.get()?
        };
        self.address_tag_list.push(address_tag);

        // 2. Insert new triple (<db_adr>, <db_val>, <db_val> into stash.
        self.stash_addresses_share.push(db_address_share);
        self.stash_values_share.push(db_value_share);
        self.stash_old_values_share.push(db_value_share);

        // 3. Update stash
        // - if I.op = write, we want to write I.val to index loc
        // - if I.op = read, we need to write db_val to index c
        //   (since that has been done already in step 2, it is essentially a no-op)
        {
            let previous_value_share =
                SelectProtocol::select(comm, stash_state.flag, stash_state.value, db_value_share)?;
            let location_share =
                SelectProtocol::select(comm, instruction.operation, stash_state.location, -F::ONE)?;
            let value_share = SelectProtocol::select(
                comm,
                instruction.operation,
                instruction.value - previous_value_share,
                F::ZERO,
            )?;
            // eprintln!(
            //     "P{}: prev_val = {:?}, loc = {:?}, upd = {:?}, vals = {:?}",
            //     comm.get_my_id() + 1,
            //     previous_value_share,
            //     location_share,
            //     value_share,
            //     self.stash_values_share
            // );
            stash_write_value::<C, F, SPDPF>(
                comm,
                self.access_counter,
                location_share,
                value_share,
                &mut self.stash_values_share,
            )?;
            // eprintln!(
            //     "P{}: prev_val = {:?}, loc = {:?}, upd = {:?}, vals = {:?}",
            //     comm.get_my_id() + 1,
            //     previous_value_share,
            //     location_share,
            //     value_share,
            //     self.stash_values_share
            // );
        }

        self.access_counter += 1;
        self.state = if self.access_counter == self.stash_size {
            State::AccessesExhausted
        } else {
            State::AwaitingRead
        };
        Ok(())
    }

    fn get_stash_share(&self) -> (&[F], &[F], &[F]) {
        (
            &self.stash_addresses_share,
            &self.stash_values_share,
            &self.stash_old_values_share,
        )
    }
}

pub struct StashParty3<F, SPDPF>
where
    F: PrimeField + LegendreSymbol + Serializable,
    SPDPF: SinglePointDpf<Value = F>,
{
    stash_size: usize,
    access_counter: usize,
    state: State,
    stash_addresses_share: Vec<F>,
    stash_values_share: Vec<F>,
    stash_old_values_share: Vec<F>,
    address_tag_list: Vec<u64>,
    doprf_party_3: Option<DOPrfParty3<F>>,
    masked_doprf_party_3: Option<MaskedDOPrfParty3<F>>,
    _phantom: PhantomData<SPDPF>,
}

impl<F, SPDPF> StashParty3<F, SPDPF>
where
    F: PrimeField + LegendreSymbol + Serializable,
    SPDPF: SinglePointDpf<Value = F>,
{
    pub fn new(stash_size: usize) -> Self {
        assert!(stash_size > 0);
        assert!(compute_stash_prf_output_bitsize(stash_size) <= 64);

        Self {
            stash_size,
            access_counter: 0,
            state: State::New,
            stash_addresses_share: Vec::with_capacity(stash_size),
            stash_values_share: Vec::with_capacity(stash_size),
            stash_old_values_share: Vec::with_capacity(stash_size),
            address_tag_list: Vec::with_capacity(stash_size),
            doprf_party_3: None,
            masked_doprf_party_3: None,
            _phantom: PhantomData,
        }
    }
}

impl<F, SPDPF> Stash<F> for StashParty3<F, SPDPF>
where
    F: PrimeField + LegendreSymbol + Serializable,
    SPDPF: SinglePointDpf<Value = F>,
    SPDPF::Key: Serializable,
{
    fn get_party_id(&self) -> usize {
        3
    }

    fn get_stash_size(&self) -> usize {
        self.stash_size
    }

    fn get_access_counter(&self) -> usize {
        self.access_counter
    }

    fn init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error> {
        assert_eq!(self.state, State::New);

        let prf_output_bitsize = compute_stash_prf_output_bitsize(self.stash_size);
        self.doprf_party_3 = Some(DOPrfParty3::new(prf_output_bitsize));
        self.masked_doprf_party_3 = Some(MaskedDOPrfParty3::new(prf_output_bitsize));

        // run DOPRF initilization
        {
            let doprf_p3 = self.doprf_party_3.as_mut().unwrap();
            doprf_p3.init(comm)?;
            let mdoprf_p3 = self.masked_doprf_party_3.as_mut().unwrap();
            mdoprf_p3.init(comm)?;
        }

        // panic!("not implemented");
        self.state = State::AwaitingRead;
        Ok(())
    }

    fn read<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        instruction: InstructionShare<F>,
    ) -> Result<StashStateShare<F>, Error> {
        assert_eq!(self.state, State::AwaitingRead);
        assert!(self.access_counter < self.stash_size);

        // 0. If the stash is empty, we are done
        if self.access_counter == 0 {
            self.state = State::AwaitingWrite;
            return Ok(StashStateShare {
                flag: F::ZERO,
                location: F::ZERO,
                value: F::ZERO,
            });
        }

        // 1. Compute tag y := PRF(k, <I.adr>) such that P1 obtains y + r and P2, P3 obtain the mask r.
        let address_tag_mask = {
            let mdoprf_p3 = self.masked_doprf_party_3.as_mut().unwrap();

            // for now do preprocessing on the fly
            mdoprf_p3.preprocess(comm, 1)?;
            let mut mask = mdoprf_p3.eval(comm, 1, &[instruction.address])?;
            bits_to_u64(mask.pop().unwrap())
        };

        // 2. Receive DPF key for the function f(x) = if x = y { 1 } else { 0 }
        let dpf_key_3: SPDPF::Key = {
            let fut = comm.receive(PARTY_1)?;
            fut.get()?
        };

        // 3. Compute shares of <flag>, <loc>, i.e., if the address is present in the stash and if
        //    so, where it is
        let (flag_share, location_share) = {
            let mut flag_share = F::ZERO;
            let mut location_share = F::ZERO;
            let mut j_as_field_element = F::ZERO;
            for j in 0..self.address_tag_list.len() {
                let dpf_value_j =
                    SPDPF::evaluate_at(&dpf_key_3, self.address_tag_list[j] ^ address_tag_mask);
                flag_share += dpf_value_j;
                location_share += j_as_field_element * dpf_value_j;
                j_as_field_element += F::ONE;
            }
            (flag_share, location_share)
        };

        // 4. Compute <loc> = if <flag> { <loc> } else { access_counter - 1 }
        let location_share = SelectProtocol::select(comm, flag_share, location_share, F::ZERO)?;

        // 5. Reshare <flag> among all three parties (nothing to do for P3)

        // 6. Read the value <val> from the stash (if <flag>) or read a zero value
        let value_share = stash_read_value::<C, F, SPDPF>(
            comm,
            self.access_counter,
            location_share,
            &self.stash_values_share,
        )?;

        self.state = State::AwaitingWrite;
        Ok(StashStateShare {
            flag: flag_share,
            location: location_share,
            value: value_share,
        })
    }

    fn write<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        instruction: InstructionShare<F>,
        stash_state: StashStateShare<F>,
        db_address_share: F,
        db_value_share: F,
    ) -> Result<(), Error> {
        assert_eq!(self.state, State::AwaitingWrite);
        assert!(self.access_counter < self.stash_size);

        // 1. Compute tag y := PRF(k, <db_adr>) such that P2, P3 obtain y and append y to the tag
        //    list.
        let address_tag: u64 = {
            let doprf_p3 = self.doprf_party_3.as_mut().unwrap();
            // for now do preprocessing on the fly
            doprf_p3.preprocess(comm, 1)?;
            let mut tag = doprf_p3.eval(comm, 1, &[db_address_share])?;
            let tag = bits_to_u64(tag.pop().unwrap());
            comm.send(PARTY_2, tag)?;
            tag
        };
        self.address_tag_list.push(address_tag);

        // 2. Insert new triple (<db_adr>, <db_val>, <db_val> into stash.
        self.stash_addresses_share.push(db_address_share);
        self.stash_values_share.push(db_value_share);
        self.stash_old_values_share.push(db_value_share);

        // 3. Update stash
        // - if I.op = write, we want to write I.val to index loc
        //      if flag = true then loc < c
        //          -> previous value in st.val
        //      if flag = false then loc = c
        //          -> previous value is db_val
        // - if I.op = read, we need don't need to write anything
        //   (but still touch every entry)
        {
            let previous_value_share =
                SelectProtocol::select(comm, stash_state.flag, stash_state.value, db_value_share)?;
            let location_share =
                SelectProtocol::select(comm, instruction.operation, stash_state.location, -F::ONE)?;
            let value_share = SelectProtocol::select(
                comm,
                instruction.operation,
                instruction.value - previous_value_share,
                F::ZERO,
            )?;
            // eprintln!(
            //     "P{}: prev_val = {:?}, loc = {:?}, upd = {:?}, vals = {:?}",
            //     comm.get_my_id() + 1,
            //     previous_value_share,
            //     location_share,
            //     value_share,
            //     self.stash_values_share
            // );
            stash_write_value::<C, F, SPDPF>(
                comm,
                self.access_counter,
                location_share,
                value_share,
                &mut self.stash_values_share,
            )?;
            // eprintln!(
            //     "P{}: prev_val = {:?}, loc = {:?}, upd = {:?}, vals = {:?}",
            //     comm.get_my_id() + 1,
            //     previous_value_share,
            //     location_share,
            //     value_share,
            //     self.stash_values_share
            // );
        }

        self.access_counter += 1;
        self.state = if self.access_counter == self.stash_size {
            State::AccessesExhausted
        } else {
            State::AwaitingRead
        };

        Ok(())
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

        let party_1 = StashParty1::<Fp, SPDPF>::new(stash_size);
        let party_2 = StashParty2::<Fp, SPDPF>::new(stash_size);
        let party_3 = StashParty3::<Fp, SPDPF>::new(stash_size);
        assert_eq!(party_1.get_party_id(), 1);
        assert_eq!(party_2.get_party_id(), 2);
        assert_eq!(party_3.get_party_id(), 3);
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

        eprintln!(
            "stash_state = {{ flag = {:?}, location = {:?}, value = {:?} }}",
            state_1.flag + state_2.flag + state_3.flag,
            state_1.location + state_2.location + state_3.location,
            state_1.value + state_2.value + state_3.value
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
            eprintln!("{:?}    {:?}    {:?}", st_adrs_1, st_vals_1, st_old_vals_1);
            eprintln!("{:?}    {:?}    {:?}", st_adrs_2, st_vals_2, st_old_vals_2);
            eprintln!("{:?}    {:?}    {:?}", st_adrs_3, st_vals_3, st_old_vals_3);
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
            eprintln!("{:?}    {:?}    {:?}", st_adrs_1, st_vals_1, st_old_vals_1);
            eprintln!("{:?}    {:?}    {:?}", st_adrs_2, st_vals_2, st_old_vals_2);
            eprintln!("{:?}    {:?}    {:?}", st_adrs_3, st_vals_3, st_old_vals_3);
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

        eprintln!(
            "stash_state = {{ flag = {:?}, location = {:?}, value = {:?} }}",
            state_1.flag + state_2.flag + state_3.flag,
            state_1.location + state_2.location + state_3.location,
            state_1.value + state_2.value + state_3.value
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
            eprintln!("{:?}    {:?}    {:?}", st_adrs_1, st_vals_1, st_old_vals_1);
            eprintln!("{:?}    {:?}    {:?}", st_adrs_2, st_vals_2, st_old_vals_2);
            eprintln!("{:?}    {:?}    {:?}", st_adrs_3, st_vals_3, st_old_vals_3);
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
            eprintln!("{:?}    {:?}    {:?}", st_adrs_1, st_vals_1, st_old_vals_1);
            eprintln!("{:?}    {:?}    {:?}", st_adrs_2, st_vals_2, st_old_vals_2);
            eprintln!("{:?}    {:?}    {:?}", st_adrs_3, st_vals_3, st_old_vals_3);
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
