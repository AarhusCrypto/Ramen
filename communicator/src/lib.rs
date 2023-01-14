pub mod communicator;
mod fut;
pub mod tcp;
pub mod traits;
pub mod unix;

use crate::traits::Serializable;
use std::io::Error as IoError;

/// Represent data of type T that we expect to receive
pub trait Fut<T> {
    /// Wait until the data has arrived and obtain it.
    fn get(self) -> Result<T, Error>;
}

/// Represent data consisting of multiple Ts that we expect to receive
pub trait MultiFut<T> {
    /// How many items of type T we expect.
    fn len(&self) -> usize;
    /// Wait until the data has arrived and obtain it.
    fn get(self) -> Result<Vec<T>, Error>;
    /// Wait until the data has arrived and write it into the provided buffer.
    fn get_into(self, buf: &mut [T]) -> Result<(), Error>;
}

/// Abstract communication interface between multiple parties
pub trait AbstractCommunicator: Clone {
    type Fut<T: Serializable>: Fut<T>;
    type MultiFut<T: Serializable>: MultiFut<T>;

    /// How many parties N there are in total
    fn get_num_parties(&self) -> usize;
    /// My party id in [0, N)
    fn get_my_id(&self) -> usize;

    /// Send a message of type T to given party
    fn send<T: Serializable>(&mut self, party_id: usize, val: T);
    /// Send a message of multiple Ts to given party
    fn send_slice<T: Serializable>(&mut self, party_id: usize, val: &[T]);

    /// Send a message of type T to next party
    fn send_next<T: Serializable>(&mut self, val: T) {
        self.send((self.get_my_id() + 1) % self.get_num_parties(), val);
    }
    /// Send a message of multiple Ts to next party
    fn send_next_slice<T: Serializable>(&mut self, val: &[T]) {
        self.send_slice((self.get_my_id() + 1) % self.get_num_parties(), val);
    }

    /// Send a message of type T to previous party
    fn send_previous<T: Serializable>(&mut self, val: T) {
        self.send(
            (self.get_num_parties() + self.get_my_id() - 1) % self.get_num_parties(),
            val,
        );
    }
    /// Send a message of multiple Ts to previous party
    fn send_previous_slice<T: Serializable>(&mut self, val: &[T]) {
        self.send_slice(
            (self.get_num_parties() + self.get_my_id() - 1) % self.get_num_parties(),
            val,
        );
    }

    /// Send a message of type T all parties
    fn broadcast<T: Serializable>(&mut self, val: T) {
        let my_id = self.get_my_id();
        for party_id in 0..self.get_num_parties() {
            if party_id == my_id {
                continue;
            }
            self.send(party_id, val.clone());
        }
    }
    /// Send a message of multiple Ts to all parties
    fn broadcast_slice<T: Serializable>(&mut self, val: &[T]) {
        let my_id = self.get_my_id();
        for party_id in 0..self.get_num_parties() {
            if party_id == my_id {
                continue;
            }
            self.send_slice(party_id, val);
        }
    }

    /// Expect to receive message of type T from given party.  Use the returned future to obtain
    /// the message once it has arrived.
    fn receive<T: Serializable>(&mut self, party_id: usize) -> Self::Fut<T>;
    /// Expect to receive message of multiple Ts from given party.  Use the returned future to obtain
    /// the message once it has arrived.
    fn receive_n<T: Serializable>(&mut self, party_id: usize, n: usize) -> Self::MultiFut<T>;

    /// Expect to receive message of type T from the next party.  Use the returned future to obtain
    /// the message once it has arrived.
    fn receive_next<T: Serializable>(&mut self) -> Self::Fut<T> {
        self.receive((self.get_my_id() + 1) % self.get_num_parties())
    }
    /// Expect to receive message of multiple Ts from the next party.  Use the returned future to obtain
    /// the message once it has arrived.
    fn receive_next_n<T: Serializable>(&mut self, n: usize) -> Self::MultiFut<T> {
        self.receive_n((self.get_my_id() + 1) % self.get_num_parties(), n)
    }

    /// Expect to receive message of type T from the previous party.  Use the returned future to obtain
    /// the message once it has arrived.
    fn receive_previous<T: Serializable>(&mut self) -> Self::Fut<T> {
        self.receive((self.get_num_parties() + self.get_my_id() - 1) % self.get_num_parties())
    }
    /// Expect to receive message of multiple Ts from the previous party.  Use the returned future to obtain
    /// the message once it has arrived.
    fn receive_previous_n<T: Serializable>(&mut self, n: usize) -> Self::MultiFut<T> {
        self.receive_n(
            (self.get_num_parties() + self.get_my_id() - 1) % self.get_num_parties(),
            n,
        )
    }

    /// Shutdown the communication system
    fn shutdown(&mut self);
}

/// Custom error type
#[derive(Debug)]
pub enum Error {
    /// The connection has not been established
    ConnectionSetupError,
    /// Some std::io::Error appeared
    IoError(IoError),
    /// Serialization of data failed
    SerializationError(String),
    /// Deserialization of data failed
    DeserializationError(String),
}

/// Enable automatic conversions from std::io::Error
impl From<IoError> for Error {
    fn from(e: IoError) -> Error {
        Error::IoError(e)
    }
}
