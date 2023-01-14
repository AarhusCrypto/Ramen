use crate::{Error, Fut, MultiFut, Serializable};
use std::marker::PhantomData;
use std::sync::mpsc::Receiver;

pub struct BytesFut {
    pub size: usize,
    pub data_rx: Receiver<Vec<u8>>,
}

impl BytesFut {
    pub fn get(self) -> Vec<u8> {
        let buf = self.data_rx.recv().expect("receive failed");
        assert_eq!(buf.len(), self.size);
        buf
    }
}

pub struct MyFut<T: Serializable> {
    bytes_fut: BytesFut,
    _phantom: PhantomData<T>,
}

impl<T: Serializable> MyFut<T> {
    pub fn new(bytes_fut: BytesFut) -> Self {
        Self {
            bytes_fut,
            _phantom: PhantomData,
        }
    }
}

impl<T: Serializable> Fut<T> for MyFut<T> {
    fn get(self) -> Result<T, Error> {
        T::from_bytes(&self.bytes_fut.get())
    }
}

pub struct MyMultiFut<T: Serializable> {
    size: usize,
    bytes_fut: BytesFut,
    _phantom: PhantomData<T>,
}

impl<T: Serializable> MyMultiFut<T> {
    pub fn new(size: usize, bytes_fut: BytesFut) -> Self {
        Self {
            size,
            bytes_fut,
            _phantom: PhantomData,
        }
    }
}

impl<T: Serializable> MultiFut<T> for MyMultiFut<T> {
    fn len(&self) -> usize {
        self.size
    }

    fn get(self) -> Result<Vec<T>, Error> {
        let data_buf = self.bytes_fut.get();
        if data_buf.len() != self.size * T::bytes_required() {
            return Err(Error::DeserializationError(
                "received buffer of unexpected size".to_owned(),
            ));
        }
        let mut output = Vec::with_capacity(self.size);
        for c in data_buf.chunks_exact(T::bytes_required()) {
            match T::from_bytes(c) {
                Ok(v) => output.push(v),
                Err(e) => return Err(e),
            }
        }
        Ok(output)
    }

    fn get_into(self, buf: &mut [T]) -> Result<(), Error> {
        if buf.len() != self.size {
            return Err(Error::DeserializationError(
                "supplied buffer has unexpected size".to_owned(),
            ));
        }
        let data_buf = self.bytes_fut.get();
        if data_buf.len() != self.size * T::bytes_required() {
            return Err(Error::DeserializationError(
                "received buffer of unexpected size".to_owned(),
            ));
        }

        Ok(())
    }
}
