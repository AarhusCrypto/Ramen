//! Basic types for the DORAM implementation.
use communicator::Error as CommunicationError;
use ff::PrimeField;

/// Type of an access operation.
pub enum Operation {
    /// Read from the memory.
    Read,
    /// Write to the memory.
    Write,
}

impl Operation {
    /// Encode an access operation as field element.
    ///
    /// Read is encoded as 0, and Write is encoded as 1.
    pub fn encode<F: PrimeField>(&self) -> F {
        match self {
            Self::Read => F::ZERO,
            Self::Write => F::ONE,
        }
    }

    /// Decode an encoded operation again.
    pub fn decode<F: PrimeField>(encoded_op: F) -> Self {
        if encoded_op == F::ZERO {
            Self::Read
        } else if encoded_op == F::ONE {
            Self::Write
        } else {
            panic!("invalid value")
        }
    }
}

/// Define an additive share of an access instruction.
#[derive(Clone, Copy, Debug, Default)]
pub struct InstructionShare<F: PrimeField> {
    /// Whether it is a read (0) or a write (1).
    pub operation: F,
    /// The address to access.
    pub address: F,
    /// The value that should (possibly) be written into memory.
    pub value: F,
}

/// Custom error type used in this library.
#[derive(Debug)]
pub enum Error {
    /// Wrap a [`communicator::Error`].
    CommunicationError(CommunicationError),
}

impl From<CommunicationError> for Error {
    fn from(e: CommunicationError) -> Self {
        Error::CommunicationError(e)
    }
}
