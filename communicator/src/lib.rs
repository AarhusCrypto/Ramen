pub mod communicator;
pub mod tcp;
pub mod unix;

use bincode::error::{EncodeError, DecodeError};
use std::io::Error as IoError;
use std::sync::mpsc::{RecvError, SendError};

pub trait Serializable: Clone + Send + 'static + bincode::Encode + bincode::Decode {}

impl<T> Serializable for T where T: Clone + Send + 'static + bincode::Encode + bincode::Decode {}

/// Represent data of type T that we expect to receive
pub trait Fut<T> {
    /// Wait until the data has arrived and obtain it.
    fn get(self) -> Result<T, Error>;
}

/// Abstract communication interface between multiple parties
pub trait AbstractCommunicator {
    type Fut<T: Serializable>: Fut<T>;

    /// How many parties N there are in total
    fn get_num_parties(&self) -> usize;

    /// My party id in [0, N)
    fn get_my_id(&self) -> usize;

    /// Send a message of type T to given party
    fn send<T: Serializable>(&mut self, party_id: usize, val: T) -> Result<(), Error>;

    /// Send a message of type T to next party
    fn send_next<T: Serializable>(&mut self, val: T) -> Result<(), Error> {
        self.send((self.get_my_id() + 1) % self.get_num_parties(), val)
    }

    /// Send a message of type T to previous party
    fn send_previous<T: Serializable>(&mut self, val: T) -> Result<(), Error> {
        self.send(
            (self.get_num_parties() + self.get_my_id() - 1) % self.get_num_parties(),
            val,
        )
    }

    /// Send a message of type T all parties
    fn broadcast<T: Serializable>(&mut self, val: T) -> Result<(), Error> {
        let my_id = self.get_my_id();
        for party_id in 0..self.get_num_parties() {
            if party_id == my_id {
                continue;
            }
            self.send(party_id, val.clone())?;
        }
        Ok(())
    }

    /// Expect to receive message of type T from given party.  Use the returned future to obtain
    /// the message once it has arrived.
    fn receive<T: Serializable>(&mut self, party_id: usize) -> Result<Self::Fut<T>, Error>;

    /// Expect to receive message of type T from the next party.  Use the returned future to obtain
    /// the message once it has arrived.
    fn receive_next<T: Serializable>(&mut self) -> Result<Self::Fut<T>, Error> {
        self.receive((self.get_my_id() + 1) % self.get_num_parties())
    }

    /// Expect to receive message of type T from the previous party.  Use the returned future to obtain
    /// the message once it has arrived.
    fn receive_previous<T: Serializable>(&mut self) -> Result<Self::Fut<T>, Error> {
        self.receive((self.get_num_parties() + self.get_my_id() - 1) % self.get_num_parties())
    }

    /// Shutdown the communication system
    fn shutdown(&mut self);
}

/// Custom error type
#[derive(Debug)]
pub enum Error {
    /// The connection has not been established
    ConnectionSetupError,
    /// The API was not used correctly
    LogicError(String),
    /// Some std::io::Error appeared
    IoError(IoError),
    /// Some std::sync::mpsc::RecvError appeared
    RecvError(RecvError),
    /// Some std::sync::mpsc::SendError appeared
    SendError(String),
    /// Some bincode::error::DecodeError appeared
    EncodeError(EncodeError),
    /// Some bincode::error::DecodeError appeared
    DecodeError(DecodeError),
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

/// Enable automatic conversions from std::sync::mpsc::RecvError
impl From<RecvError> for Error {
    fn from(e: RecvError) -> Error {
        Error::RecvError(e)
    }
}

/// Enable automatic conversions from std::sync::mpsc::SendError
impl<T> From<SendError<T>> for Error {
    fn from(e: SendError<T>) -> Error {
        Error::SendError(e.to_string())
    }
}

/// Enable automatic conversions from bincode::error::EncodeError
impl From<EncodeError> for Error {
    fn from(e: EncodeError) -> Error {
        Error::EncodeError(e)
    }
}

/// Enable automatic conversions from bincode::error::DecodeError
impl From<DecodeError> for Error {
    fn from(e: DecodeError) -> Error {
        Error::DecodeError(e)
    }
}
