use communicator::Error as CommunicationError;
use ff::PrimeField;

pub enum Operation {
    Read,
    Write,
}

impl Operation {
    pub fn encode<F: PrimeField>(&self) -> F {
        match self {
            Self::Read => F::ZERO,
            Self::Write => F::ONE,
        }
    }
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

#[derive(Clone, Copy, Debug, Default)]
pub struct InstructionShare<F: PrimeField> {
    pub operation: F,
    pub address: F,
    pub value: F,
}

#[derive(Debug)]
pub enum Error {
    CommunicationError(CommunicationError),
}

impl From<CommunicationError> for Error {
    fn from(e: CommunicationError) -> Self {
        Error::CommunicationError(e)
    }
}
