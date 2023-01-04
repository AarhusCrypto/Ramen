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

pub struct InstructionShare<F: PrimeField> {
    operation_share: F,
    address_share: F,
    value: F,
}
