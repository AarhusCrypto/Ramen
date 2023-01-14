use crate::Error;
use core::array;

/// Allow a type to get serialized into bytes
pub trait Serializable: Clone + Sized {
    /// How many bytes are needed?
    fn bytes_required() -> usize;
    /// Convert to bytes and store them in a new vector
    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = vec![0u8; Self::bytes_required()];
        self.into_bytes(&mut buf)
            .expect("does not fail, buffer has the right size");
        buf
    }
    /// Convert to bytes and store them in the given buffer. Fails if the buffer has not the right size.
    fn into_bytes(&self, buf: &mut [u8]) -> Result<(), Error>;
    /// Convert the bytes in the given buffer into an object. Fails if the buffer has not the right size.
    fn from_bytes(buf: &[u8]) -> Result<Self, Error>;
}

/// Convert a slice of a serializable type into a new byte vector.
pub fn slice_to_bytes<T: Serializable>(slice: &[T]) -> Vec<u8> {
    slice.iter().flat_map(|x| x.to_bytes()).collect()
}

/// Convert a slice of a serializable type into bytes and write them into a given buffer.
pub fn slice_into_bytes<T: Serializable>(slice: &[T], buf: &mut [u8]) -> Result<(), Error> {
    let bytes_required = slice.len() * T::bytes_required();
    if !buf.len() == bytes_required {
        return Err(Error::SerializationError(
            "supplied buffer has unexpected size".to_owned(),
        ));
    }
    slice
        .iter()
        .zip(buf.chunks_exact_mut(T::bytes_required()))
        .for_each(|(x, c)| {
            x.into_bytes(c)
                .expect("does not fail, since chunks have the right size");
        });
    Ok(())
}

/// Convert given buffer of bytes into objects and store them in a given a slice.
pub fn slice_from_bytes<T: Serializable>(slice: &mut [T], buf: &[u8]) -> Result<(), Error> {
    let bytes_required = slice.len() * T::bytes_required();
    if !buf.len() == bytes_required {
        return Err(Error::DeserializationError(
            "supplied buffer has unexpected size".to_owned(),
        ));
    }
    for (i, c) in buf.chunks_exact(T::bytes_required()).enumerate() {
        match T::from_bytes(c) {
            Ok(v) => slice[i] = v,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

impl<T: Serializable + Default, const N: usize> Serializable for [T; N] {
    fn bytes_required() -> usize {
        T::bytes_required() * N
    }

    fn to_bytes(&self) -> Vec<u8> {
        slice_to_bytes(self.as_slice())
    }

    fn into_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
        slice_into_bytes(self.as_slice(), buf)
    }

    fn from_bytes(buf: &[u8]) -> Result<Self, Error> {
        if buf.len() != Self::bytes_required() {
            return Err(Error::DeserializationError(
                "supplied buffer has unexpected size".to_owned(),
            ));
        }

        let mut output = array::from_fn(|_| Default::default());
        slice_from_bytes(&mut output, buf)?;
        Ok(output)
    }
}

impl<T: Serializable, U: Serializable> Serializable for (T, U) {
    fn bytes_required() -> usize {
        T::bytes_required() + U::bytes_required()
    }

    fn to_bytes(&self) -> Vec<u8> {
        let num_t_bytes = T::bytes_required();
        let num_u_bytes = U::bytes_required();
        let mut buf = vec![0u8; num_t_bytes + num_u_bytes];
        self.0.into_bytes(&mut buf[..num_t_bytes]).unwrap();
        self.1.into_bytes(&mut buf[num_t_bytes..]).unwrap();
        buf
    }

    fn into_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
        let num_t_bytes = T::bytes_required();
        let num_u_bytes = U::bytes_required();
        if buf.len() != num_t_bytes + num_u_bytes {
            return Err(Error::SerializationError(
                "supplied buffer has unexpected size".to_owned(),
            ));
        }
        self.0.into_bytes(&mut buf[..num_t_bytes]).unwrap();
        self.1.into_bytes(&mut buf[num_t_bytes..]).unwrap();
        Ok(())
    }

    fn from_bytes(buf: &[u8]) -> Result<Self, Error> {
        let num_t_bytes = T::bytes_required();
        let num_u_bytes = U::bytes_required();
        if buf.len() != num_t_bytes + num_u_bytes {
            return Err(Error::DeserializationError(
                "supplied buffer has unexpected size".to_owned(),
            ));
        }
        let t = T::from_bytes(&buf[..num_t_bytes])?;
        let u = U::from_bytes(&buf[num_t_bytes..])?;
        Ok((t, u))
    }
}

macro_rules! impl_serializable_for_uints {
    ($type:ty) => {
        impl Serializable for $type {
            fn bytes_required() -> usize {
                <$type>::BITS as usize / 8
            }

            fn to_bytes(&self) -> Vec<u8> {
                let bytes = self.to_be_bytes();
                bytes.into()
            }

            fn into_bytes(&self, buf: &mut [u8]) -> Result<(), Error> {
                if !buf.len() == Self::bytes_required() {
                    return Err(Error::SerializationError("buffer to small".to_owned()));
                }
                buf.copy_from_slice(&self.to_be_bytes());
                Ok(())
            }

            fn from_bytes(buf: &[u8]) -> Result<Self, Error> {
                // assert_eq!(buf.len(), Self::bytes_required());
                match buf.try_into().map(Self::from_be_bytes) {
                    Ok(v) => Ok(v),
                    Err(_) => Err(Error::DeserializationError(
                        "supplied buffer has unexpected size".to_owned(),
                    )),
                }
            }
        }
    };
}

impl_serializable_for_uints!(u8);
impl_serializable_for_uints!(u16);
impl_serializable_for_uints!(u32);
impl_serializable_for_uints!(u64);
impl_serializable_for_uints!(u128);

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};

    macro_rules! make_test_serialiable_for_uints {
        ($test_name:ident, $type:ty) => {
            #[test]
            fn $test_name() {
                type T = $type;
                assert_eq!(T::bytes_required(), T::BITS as usize / 8);
                for _ in 0..100 {
                    let val: T = thread_rng().gen();
                    let expected_bytes = val.to_be_bytes();
                    assert_eq!(val.to_bytes(), expected_bytes);
                    let mut buf = vec![0u8; T::bytes_required()];
                    val.into_bytes(&mut buf).unwrap();
                    assert_eq!(buf, expected_bytes);
                    let new_val = T::from_bytes(&val.to_bytes());
                    assert!(new_val.is_ok());
                    assert_eq!(new_val.unwrap(), val);
                }
            }
        };
    }

    make_test_serialiable_for_uints!(test_serialize_u8, u8);
    make_test_serialiable_for_uints!(test_serialize_u16, u16);
    make_test_serialiable_for_uints!(test_serialize_u32, u32);
    make_test_serialiable_for_uints!(test_serialize_u64, u64);
    make_test_serialiable_for_uints!(test_serialize_u128, u128);

    macro_rules! make_test_serialiable_for_uint_arrays {
        ($test_name:ident, $type:ty, $len:expr) => {
            #[test]
            fn $test_name() {
                type T = $type;
                type A = [T; $len];
                assert_eq!(A::bytes_required(), T::BITS as usize / 8 * $len);
                for _ in 0..100 {
                    let val: A = array::from_fn(|_| thread_rng().gen());
                    let serialized = val.to_bytes();
                    let mut serialized2 = vec![0u8; A::bytes_required()];
                    val.into_bytes(&mut serialized2).unwrap();
                    assert_eq!(serialized.len(), A::bytes_required());
                    for i in 0..$len {
                        let expected_bytes = val[i].to_be_bytes();
                        assert_eq!(
                            serialized[i * T::bytes_required()..(i + 1) * T::bytes_required()],
                            expected_bytes
                        );
                    }
                    let new_val = <A>::from_bytes(&val.to_bytes());
                    assert!(new_val.is_ok());
                    assert_eq!(new_val.unwrap(), val);
                }
            }
        };
    }

    make_test_serialiable_for_uint_arrays!(test_serialize_u8_array, u8, 42);
    make_test_serialiable_for_uint_arrays!(test_serialize_u16_array, u16, 42);
    make_test_serialiable_for_uint_arrays!(test_serialize_u32_array, u32, 42);
    make_test_serialiable_for_uint_arrays!(test_serialize_u64_array, u64, 42);
    make_test_serialiable_for_uint_arrays!(test_serialize_u128_array, u128, 42);

    macro_rules! make_test_serialiable_for_pairs {
        ($test_name:ident, $type_t:ty, $type_u:ty) => {
            #[test]
            fn $test_name() {
                type T = $type_t;
                type U = $type_u;
                type P = (T, U);
                assert_eq!(
                    P::bytes_required(),
                    T::bytes_required() + U::bytes_required()
                );
                for _ in 0..100 {
                    let val: P = thread_rng().gen();
                    let serialized = val.to_bytes();
                    let mut serialized2 = vec![0u8; P::bytes_required()];
                    val.into_bytes(&mut serialized2).unwrap();
                    assert_eq!(serialized.len(), P::bytes_required());
                    let new_val = <P>::from_bytes(&val.to_bytes());
                    assert!(new_val.is_ok());
                    assert_eq!(new_val.unwrap(), val);
                }
            }
        };
    }

    make_test_serialiable_for_pairs!(test_serialize_pair_u8_u32, u8, u32);
    make_test_serialiable_for_pairs!(test_serialize_pair_u8array_u32, [u8; 13], u32);
    make_test_serialiable_for_pairs!(test_serialize_pair_u128array_u16array, u128, [u16; 7]);
    make_test_serialiable_for_pairs!(
        test_serialize_pair_nested_uints,
        u8,
        (u16, (u32, (u64, u128)))
    );
}
