use crate::common::Error;
use communicator::{AbstractCommunicator, Fut, Serializable};
use ff::Field;
use rand::thread_rng;

/// Select between two shared value <a>, <b> based on a shared condition bit <c>:
/// Output <w> <- if <c> then <a> else <b>.
pub trait Select<F> {
    fn select<C: AbstractCommunicator>(
        comm: &mut C,
        b_share: F,
        x_share: F,
        y_share: F,
    ) -> Result<F, Error>;
}

const PARTY_1: usize = 0;
const PARTY_2: usize = 1;
const PARTY_3: usize = 2;

fn other_compute_party(my_id: usize) -> usize {
    match my_id {
        PARTY_2 => PARTY_3,
        PARTY_3 => PARTY_2,
        _ => panic!("invalid party id"),
    }
}

pub struct SelectProtocol {}

impl<F> Select<F> for SelectProtocol
where
    F: Field + Serializable,
{
    fn select<C: AbstractCommunicator>(
        comm: &mut C,
        c_share: F,
        a_share: F,
        b_share: F,
    ) -> Result<F, Error> {
        let my_id = comm.get_my_id();

        let output = b_share
            + if my_id == PARTY_1 {
                let mut rng = thread_rng();
                // create multiplication triple
                let x_2 = F::random(&mut rng);
                let x_3 = F::random(&mut rng);
                let y_2 = F::random(&mut rng);
                let y_3 = F::random(&mut rng);
                let z_2 = F::random(&mut rng);
                let z_3 = F::random(&mut rng);
                let z_1 = (x_2 + x_3) * (y_2 + y_3) - z_2 - z_3;
                debug_assert_eq!((x_2 + x_3) * (y_2 + y_3), z_1 + z_2 + z_3);
                let c_1_2 = F::random(&mut rng);
                let amb_1_2 = F::random(&mut rng);
                let c_1_3 = c_share - c_1_2;
                let amb_1_3 = (a_share - b_share) - amb_1_2;

                comm.send(PARTY_2, (x_2, y_2, z_2, c_1_2, amb_1_2))?;
                comm.send(PARTY_3, (x_3, y_3, z_3, c_1_3, amb_1_3))?;

                z_1
            } else {
                let fut_xzy = comm.receive::<(F, F, F, F, F)>(PARTY_1)?;
                let fut_de = comm.receive::<(F, F)>(other_compute_party(my_id))?;
                let (x_i, y_i, mut z_i, c_1_i, amb_1_i) = fut_xzy.get()?;
                let d_i = (c_share + c_1_i) - x_i;
                let e_i = (a_share - b_share + amb_1_i) - y_i;
                comm.send(other_compute_party(my_id), (d_i, e_i))?;
                let (d_j, e_j) = fut_de.get()?;
                let (d, e) = (d_i + d_j, e_i + e_j);

                z_i += e * (c_share + c_1_i) + d * (a_share - b_share + amb_1_i);
                if my_id == PARTY_2 {
                    z_i -= d * e;
                }

                z_i
            };

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use communicator::unix::make_unix_communicators;
    use std::thread;
    use utils::field::Fp;

    fn run_select<Proto: Select<F>, F>(
        mut comm: impl AbstractCommunicator + Send + 'static,
        c_share: F,
        a_share: F,
        b_share: F,
    ) -> thread::JoinHandle<(impl AbstractCommunicator, F)>
    where
        F: Field + Serializable,
    {
        thread::spawn(move || {
            let result = Proto::select(&mut comm, c_share, a_share, b_share);
            (comm, result.unwrap())
        })
    }

    #[test]
    fn test_select() {
        let (comm_3, comm_2, comm_1) = {
            let mut comms = make_unix_communicators(3);
            (
                comms.pop().unwrap(),
                comms.pop().unwrap(),
                comms.pop().unwrap(),
            )
        };
        let mut rng = thread_rng();

        let (a_1, a_2, a_3) = (
            Fp::random(&mut rng),
            Fp::random(&mut rng),
            Fp::random(&mut rng),
        );
        let a = a_1 + a_2 + a_3;
        let (b_1, b_2, b_3) = (
            Fp::random(&mut rng),
            Fp::random(&mut rng),
            Fp::random(&mut rng),
        );
        let b = b_1 + b_2 + b_3;
        let (c_2, c_3) = (Fp::random(&mut rng), Fp::random(&mut rng));

        let c0_1 = -c_2 - c_3;
        let c1_1 = Fp::ONE - c_2 - c_3;

        // check for <c> = <0>
        let h1 = run_select::<SelectProtocol, _>(comm_1, c0_1, a_1, b_1);
        let h2 = run_select::<SelectProtocol, _>(comm_2, c_2, a_2, b_2);
        let h3 = run_select::<SelectProtocol, _>(comm_3, c_3, a_3, b_3);
        let (comm_1, x_1) = h1.join().unwrap();
        let (comm_2, x_2) = h2.join().unwrap();
        let (comm_3, x_3) = h3.join().unwrap();

        assert_eq!(c0_1 + c_2 + c_3, Fp::ZERO);
        assert_eq!(x_1 + x_2 + x_3, b);

        // check for <c> = <1>
        let h1 = run_select::<SelectProtocol, _>(comm_1, c1_1, a_1, b_1);
        let h2 = run_select::<SelectProtocol, _>(comm_2, c_2, a_2, b_2);
        let h3 = run_select::<SelectProtocol, _>(comm_3, c_3, a_3, b_3);
        let (_, y_1) = h1.join().unwrap();
        let (_, y_2) = h2.join().unwrap();
        let (_, y_3) = h3.join().unwrap();

        assert_eq!(c1_1 + c_2 + c_3, Fp::ONE);
        assert_eq!(y_1 + y_2 + y_3, a);
    }
}
