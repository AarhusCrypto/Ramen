use crate::{AbstractCommunicator, Error, Fut, Serializable};
use bincode;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::{Read, Write};
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender};
use std::thread;

pub struct MyFut<T: Serializable> {
    data_rx: Receiver<Result<T, Error>>,
}

impl<T: Serializable> MyFut<T> {
    pub fn new(data_rx: Receiver<Result<T, Error>>) -> Self {
        Self { data_rx }
    }
}

impl<T: Serializable> Fut<T> for MyFut<T> {
    fn get(self) -> Result<T, Error> {
        match self.data_rx.recv() {
            Ok(x) => x,
            Err(e) => Err(e.into()),
        }
    }
}

/// Thread to receive messages in the background.
#[derive(Debug)]
struct ReceiverThread {
    data_request_tx: Sender<Box<dyn FnOnce(&mut dyn Read) + Send>>,
    join_handle: thread::JoinHandle<()>,
}

impl ReceiverThread {
    pub fn from_reader<R: Debug + Read + Send + 'static>(mut reader: R) -> Self {
        let (data_request_tx, data_request_rx) = channel::<Box<dyn FnOnce(&mut dyn Read) + Send>>();
        let join_handle = thread::spawn(move || {
            for func in data_request_rx.iter() {
                func(&mut reader);
            }
        });
        Self {
            data_request_tx,
            join_handle,
        }
    }

    pub fn receive<T: Serializable>(&mut self) -> Result<MyFut<T>, Error> {
        let (data_tx, data_rx) = sync_channel(1);
        self.data_request_tx
            .send(Box::new(move |mut reader: &mut dyn Read| {
                let new: Result<T, Error> =
                    bincode::decode_from_std_read(&mut reader, bincode::config::standard())
                        .map_err(|e| e.into());
                data_tx.send(new).expect("send failed");
            }))?;
        Ok(MyFut::new(data_rx.into()))
    }

    pub fn join(self) {
        drop(self.data_request_tx);
        self.join_handle.join().expect("join failed")
    }
}

/// Thread to send messages in the background.
#[derive(Debug)]
struct SenderThread {
    data_submission_tx: Sender<Box<dyn FnOnce(&mut dyn Write) + Send>>,
    join_handle: thread::JoinHandle<()>,
}

impl SenderThread {
    pub fn from_writer<W: Debug + Write + Send + 'static>(mut writer: W) -> Self {
        let (data_submission_tx, data_submission_rx) =
            channel::<Box<dyn FnOnce(&mut dyn Write) + Send>>();
        let join_handle = thread::spawn(move || {
            for func in data_submission_rx.iter() {
                func(&mut writer);
            }
            writer.flush().expect("flush failed");
        });
        Self {
            data_submission_tx,
            join_handle,
        }
    }

    pub fn send<T: Serializable>(&mut self, data: T) -> Result<(), Error> {
        self.data_submission_tx
            .send(Box::new(move |mut writer: &mut dyn Write| {
                bincode::encode_into_std_write(data, &mut writer, bincode::config::standard())
                    .expect("encode failed");
            }))?;
        Ok(())
    }

    pub fn join(self) {
        drop(self.data_submission_tx);
        self.join_handle.join().expect("join failed")
    }
}

/// Communicator that uses background threads to send and receive messages.
#[derive(Debug)]
pub struct Communicator {
    num_parties: usize,
    my_id: usize,
    receiver_threads: HashMap<usize, ReceiverThread>,
    sender_threads: HashMap<usize, SenderThread>,
}

impl Communicator {
    /// Create a new Communicator from a collection of readers and writers that are connected with
    /// the other parties.
    pub fn from_reader_writer<
        R: Read + Send + Debug + 'static,
        W: Send + Write + Debug + 'static,
    >(
        num_parties: usize,
        my_id: usize,
        mut rw_map: HashMap<usize, (R, W)>,
    ) -> Self {
        assert_eq!(rw_map.len(), num_parties - 1);
        assert!((0..num_parties)
            .filter(|&pid| pid != my_id)
            .all(|pid| rw_map.contains_key(&pid)));

        let mut receiver_threads = HashMap::with_capacity(num_parties - 1);
        let mut sender_threads = HashMap::with_capacity(num_parties - 1);

        for pid in 0..num_parties {
            if pid == my_id {
                continue;
            }
            let (reader, writer) = rw_map.remove(&pid).unwrap();
            receiver_threads.insert(pid, ReceiverThread::from_reader(reader));
            sender_threads.insert(pid, SenderThread::from_writer(writer));
        }

        Self {
            num_parties,
            my_id,
            receiver_threads,
            sender_threads,
        }
    }
}

impl AbstractCommunicator for Communicator {
    type Fut<T: Serializable> = MyFut<T>;

    fn get_num_parties(&self) -> usize {
        self.num_parties
    }

    fn get_my_id(&self) -> usize {
        self.my_id
    }

    fn send<T: Serializable>(&mut self, party_id: usize, val: T) -> Result<(), Error> {
        match self.sender_threads.get_mut(&party_id) {
            Some(t) => {
                t.send(val)?;
                Ok(())
            }
            None => Err(Error::LogicError(format!(
                "SenderThread for party {} not found",
                party_id
            ))),
        }
    }

    fn receive<T: Serializable>(&mut self, party_id: usize) -> Result<Self::Fut<T>, Error> {
        match self.receiver_threads.get_mut(&party_id) {
            Some(t) => t.receive::<T>(),
            None => Err(Error::LogicError(format!(
                "ReceiverThread for party {} not found",
                party_id
            ))),
        }
    }

    fn shutdown(&mut self) {
        self.sender_threads.drain().for_each(|(_, t)| t.join());
        self.receiver_threads.drain().for_each(|(_, t)| t.join());
    }
}
