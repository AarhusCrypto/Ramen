use crate::common::Error;
use communicator::{AbstractCommunicator, Fut, Serializable};
use ff::Field;
use itertools::izip;
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use std::collections::VecDeque;

/// Select between two shared value <a>, <b> based on a shared condition bit <c>:
/// Output <w> <- if <c> then <a> else <b>.
pub trait Select<F> {
    fn init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error>;

    fn preprocess<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        num: usize,
    ) -> Result<(), Error>;

    fn select<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        c_share: F,
        a_share: F,
        b_share: F,
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

#[derive(Default)]
pub struct SelectProtocol<F> {
    shared_prg_1: Option<ChaChaRng>,
    shared_prg_2: Option<ChaChaRng>,
    shared_prg_3: Option<ChaChaRng>,
    is_initialized: bool,
    num_preprocessed_invocations: usize,
    preprocessed_mt_x: VecDeque<F>,
    preprocessed_mt_y: VecDeque<F>,
    preprocessed_mt_z: VecDeque<F>,
    preprocessed_c_1_2: VecDeque<F>,
    preprocessed_amb_1_2: VecDeque<F>,
}

impl<F> Select<F> for SelectProtocol<F>
where
    F: Field + Serializable,
{
    fn init<C: AbstractCommunicator>(&mut self, comm: &mut C) -> Result<(), Error> {
        if comm.get_my_id() == PARTY_1 {
            self.shared_prg_2 = Some(ChaChaRng::from_seed(thread_rng().gen()));
            comm.send(PARTY_2, self.shared_prg_2.as_ref().unwrap().get_seed())?;
            self.shared_prg_3 = Some(ChaChaRng::from_seed(thread_rng().gen()));
            comm.send(PARTY_3, self.shared_prg_3.as_ref().unwrap().get_seed())?;
        } else {
            let fut_seed = comm.receive(PARTY_1)?;
            self.shared_prg_1 = Some(ChaChaRng::from_seed(fut_seed.get()?));
        }
        self.is_initialized = true;
        Ok(())
    }

    fn preprocess<C: AbstractCommunicator>(&mut self, comm: &mut C, n: usize) -> Result<(), Error> {
        assert!(self.is_initialized);

        let my_id = comm.get_my_id();

        if my_id == PARTY_1 {
            let x2s: Vec<F> = (0..n)
                .map(|_| F::random(self.shared_prg_2.as_mut().unwrap()))
                .collect();
            let y2s: Vec<F> = (0..n)
                .map(|_| F::random(self.shared_prg_2.as_mut().unwrap()))
                .collect();
            let z2s: Vec<F> = (0..n)
                .map(|_| F::random(self.shared_prg_2.as_mut().unwrap()))
                .collect();
            let x3s: Vec<F> = (0..n)
                .map(|_| F::random(self.shared_prg_3.as_mut().unwrap()))
                .collect();
            let y3s: Vec<F> = (0..n)
                .map(|_| F::random(self.shared_prg_3.as_mut().unwrap()))
                .collect();
            let z3s: Vec<F> = (0..n)
                .map(|_| F::random(self.shared_prg_3.as_mut().unwrap()))
                .collect();

            let z1s = izip!(x2s, y2s, z2s, x3s, y3s, z3s)
                .map(|(x_2, y_2, z_2, x_3, y_3, z_3)| (x_2 + x_3) * (y_2 + y_3) - z_2 - z_3);
            self.preprocessed_mt_z.extend(z1s);

            self.preprocessed_c_1_2
                .extend((0..n).map(|_| F::random(self.shared_prg_2.as_mut().unwrap())));
            self.preprocessed_amb_1_2
                .extend((0..n).map(|_| F::random(self.shared_prg_2.as_mut().unwrap())));
        } else {
            self.preprocessed_mt_x
                .extend((0..n).map(|_| F::random(self.shared_prg_1.as_mut().unwrap())));
            self.preprocessed_mt_y
                .extend((0..n).map(|_| F::random(self.shared_prg_1.as_mut().unwrap())));
            self.preprocessed_mt_z
                .extend((0..n).map(|_| F::random(self.shared_prg_1.as_mut().unwrap())));
            if my_id == PARTY_2 {
                self.preprocessed_c_1_2
                    .extend((0..n).map(|_| F::random(self.shared_prg_1.as_mut().unwrap())));
                self.preprocessed_amb_1_2
                    .extend((0..n).map(|_| F::random(self.shared_prg_1.as_mut().unwrap())));
            }
        }

        self.num_preprocessed_invocations += n;
        Ok(())
    }

    fn select<C: AbstractCommunicator>(
        &mut self,
        comm: &mut C,
        c_share: F,
        a_share: F,
        b_share: F,
    ) -> Result<F, Error> {
        let my_id = comm.get_my_id();

        // if further preprocessing is needed, do it now
        if self.num_preprocessed_invocations == 0 {
            self.preprocess(comm, 1)?;
        }
        self.num_preprocessed_invocations -= 1;

        if my_id == PARTY_1 {
            let c_1_2 = self.preprocessed_c_1_2.pop_front().unwrap();
            let amb_1_2 = self.preprocessed_amb_1_2.pop_front().unwrap();
            comm.send(PARTY_3, (c_share - c_1_2, (a_share - b_share) - amb_1_2))?;
            let z = self.preprocessed_mt_z.pop_front().unwrap();
            Ok(b_share + z)
        } else {
            let (c_1_i, amb_1_i) = if my_id == PARTY_2 {
                (
                    self.preprocessed_c_1_2.pop_front().unwrap(),
                    self.preprocessed_amb_1_2.pop_front().unwrap(),
                )
            } else {
                let fut_1 = comm.receive::<(F, F)>(PARTY_1)?;
                fut_1.get()?
            };
            let fut_de = comm.receive::<(F, F)>(other_compute_party(my_id))?;
            let x_i = self.preprocessed_mt_x.pop_front().unwrap();
            let y_i = self.preprocessed_mt_y.pop_front().unwrap();
            let mut z_i = self.preprocessed_mt_z.pop_front().unwrap();
            let d_i = (c_share + c_1_i) - x_i;
            let e_i = (a_share - b_share + amb_1_i) - y_i;
            comm.send(other_compute_party(my_id), (d_i, e_i))?;
            let (d_j, e_j) = fut_de.get()?;
            let (d, e) = (d_i + d_j, e_i + e_j);

            z_i += e * (c_share + c_1_i) + d * (a_share - b_share + amb_1_i);
            if my_id == PARTY_2 {
                z_i -= d * e;
            }

            Ok(b_share + z_i)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use communicator::unix::make_unix_communicators;
    use std::thread;
    use utils::field::Fp;

    fn run_init<Proto: Select<F> + Send + 'static, F>(
        mut comm: impl AbstractCommunicator + Send + 'static,
        mut proto: Proto,
    ) -> thread::JoinHandle<(impl AbstractCommunicator, Proto)>
    where
        F: Field + Serializable,
    {
        thread::spawn(move || {
            proto.init(&mut comm).unwrap();
            (comm, proto)
        })
    }

    fn run_select<Proto: Select<F> + Send + 'static, F>(
        mut comm: impl AbstractCommunicator + Send + 'static,
        mut proto: Proto,
        c_share: F,
        a_share: F,
        b_share: F,
    ) -> thread::JoinHandle<(impl AbstractCommunicator, Proto, F)>
    where
        F: Field + Serializable,
    {
        thread::spawn(move || {
            let result = proto.select(&mut comm, c_share, a_share, b_share);
            (comm, proto, result.unwrap())
        })
    }

    #[test]
    fn test_select() {
        let proto_1 = SelectProtocol::<Fp>::default();
        let proto_2 = SelectProtocol::<Fp>::default();
        let proto_3 = SelectProtocol::<Fp>::default();

        let (comm_3, comm_2, comm_1) = {
            let mut comms = make_unix_communicators(3);
            (
                comms.pop().unwrap(),
                comms.pop().unwrap(),
                comms.pop().unwrap(),
            )
        };

        let h1 = run_init(comm_1, proto_1);
        let h2 = run_init(comm_2, proto_2);
        let h3 = run_init(comm_3, proto_3);
        let (comm_1, proto_1) = h1.join().unwrap();
        let (comm_2, proto_2) = h2.join().unwrap();
        let (comm_3, proto_3) = h3.join().unwrap();

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
        let h1 = run_select(comm_1, proto_1, c0_1, a_1, b_1);
        let h2 = run_select(comm_2, proto_2, c_2, a_2, b_2);
        let h3 = run_select(comm_3, proto_3, c_3, a_3, b_3);
        let (comm_1, proto_1, x_1) = h1.join().unwrap();
        let (comm_2, proto_2, x_2) = h2.join().unwrap();
        let (comm_3, proto_3, x_3) = h3.join().unwrap();

        assert_eq!(c0_1 + c_2 + c_3, Fp::ZERO);
        assert_eq!(x_1 + x_2 + x_3, b);

        // check for <c> = <1>
        let h1 = run_select(comm_1, proto_1, c1_1, a_1, b_1);
        let h2 = run_select(comm_2, proto_2, c_2, a_2, b_2);
        let h3 = run_select(comm_3, proto_3, c_3, a_3, b_3);
        let (_, _, y_1) = h1.join().unwrap();
        let (_, _, y_2) = h2.join().unwrap();
        let (_, _, y_3) = h3.join().unwrap();

        assert_eq!(c1_1 + c_2 + c_3, Fp::ONE);
        assert_eq!(y_1 + y_2 + y_3, a);
    }
}
