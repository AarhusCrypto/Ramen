use crate::common::Error;
use communicator::{AbstractCommunicator, Fut, Serializable};
use ff::PrimeField;
use rand::{thread_rng, Rng};

pub trait MaskIndex<F> {
    fn mask_index<C: AbstractCommunicator>(
        comm: &mut C,
        index_bits: u32,
        index_share: F,
    ) -> Result<(u16, u16, u16), Error>;
}

pub struct MaskIndexProtocol {}

impl<F> MaskIndex<F> for MaskIndexProtocol
where
    F: PrimeField + Serializable,
{
    fn mask_index<C: AbstractCommunicator>(
        comm: &mut C,
        index_bits: u32,
        index_share: F,
    ) -> Result<(u16, u16, u16), Error> {
        let random_bits = index_bits + 40;
        assert!(random_bits + 1 < F::NUM_BITS);
        assert!(index_bits <= 16);

        let bit_mask = (1 << index_bits) - 1;

        let fut_prev = comm.receive_previous::<F>()?;
        let fut_next = comm.receive_next::<(u16, F)>()?;

        // sample mask r_{i+1} and send it to P_{i-1}
        let r_next: u128 = thread_rng().gen_range(0..(1 << random_bits));
        // send masked share to P_{i+1}
        comm.send_next(index_share + F::from_u128(r_next))?;
        let r_next = (r_next & bit_mask) as u16;
        // send mask and our share to P_{i-1}
        comm.send_previous((r_next, index_share))?;

        let index_masked_prev_share = fut_prev.get()?;
        let (r_prev, index_next_share) = fut_next.get()?;

        let masked_index = index_share + index_next_share + index_masked_prev_share;
        let masked_index =
            u64::from_le_bytes(masked_index.to_repr().as_ref()[..8].try_into().unwrap());
        let masked_index = masked_index as u16 & bit_mask as u16;

        Ok((masked_index, r_prev, r_next))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use communicator::unix::make_unix_communicators;
    use ff::Field;
    use std::thread;
    use utils::field::Fp;

    fn run_mask_index<Proto: MaskIndex<F>, F>(
        mut comm: impl AbstractCommunicator + Send + 'static,
        index_bits: u32,
        index_share: F,
    ) -> thread::JoinHandle<(impl AbstractCommunicator, (u16, u16, u16))>
    where
        F: PrimeField + Serializable,
    {
        thread::spawn(move || {
            let result = Proto::mask_index(&mut comm, index_bits, index_share);
            (comm, result.unwrap())
        })
    }

    #[test]
    fn test_mask_index() {
        let (comm_3, comm_2, comm_1) = {
            let mut comms = make_unix_communicators(3);
            (
                comms.pop().unwrap(),
                comms.pop().unwrap(),
                comms.pop().unwrap(),
            )
        };
        let mut rng = thread_rng();
        let index_bits = 16;
        let bit_mask = ((1 << index_bits) - 1) as u16;
        let index = rng.gen_range(0..(1 << index_bits));
        let (index_2, index_3) = (Fp::random(&mut rng), Fp::random(&mut rng));
        let index_1 = Fp::from_u128(index as u128) - index_2 - index_3;

        // check for <c> = <0>
        let h1 = run_mask_index::<MaskIndexProtocol, _>(comm_1, index_bits, index_1);
        let h2 = run_mask_index::<MaskIndexProtocol, _>(comm_2, index_bits, index_2);
        let h3 = run_mask_index::<MaskIndexProtocol, _>(comm_3, index_bits, index_3);
        let (_, (mi_1, m3_1, m2_1)) = h1.join().unwrap();
        let (_, (mi_2, m1_2, m3_2)) = h2.join().unwrap();
        let (_, (mi_3, m2_3, m1_3)) = h3.join().unwrap();

        assert_eq!(m1_2, m1_3);
        assert_eq!(m2_1, m2_3);
        assert_eq!(m3_1, m3_2);
        assert_eq!(m1_2, m1_3);
        assert_eq!(mi_1, (index as u16).wrapping_add(m1_2) & bit_mask);
        assert_eq!(mi_2, (index as u16).wrapping_add(m2_1) & bit_mask);
        assert_eq!(mi_3, (index as u16).wrapping_add(m3_1) & bit_mask);
    }
}
