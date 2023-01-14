use crate::fut::{BytesFut, MyFut, MyMultiFut};
use crate::AbstractCommunicator;
use crate::Serializable;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::{Read, Write};
use std::sync::mpsc::{channel, sync_channel, Sender, SyncSender};
use std::thread;

/// Thread to receive messages in the background.
#[derive(Clone, Debug)]
struct ReceiverThread {
    data_request_tx: Sender<(usize, SyncSender<Vec<u8>>)>,
}

impl ReceiverThread {
    pub fn from_reader<R: Debug + Read + Send + 'static>(mut reader: R) -> Self {
        let (data_request_tx, data_request_rx) = channel::<(usize, SyncSender<Vec<u8>>)>();
        let _join_handle = thread::spawn(move || {
            for (size, sender) in data_request_rx.iter() {
                let mut buf = vec![0u8; size];
                reader.read_exact(&mut buf).expect("read failed");
                sender.send(buf).expect("send failed");
            }
        });

        Self { data_request_tx }
    }

    pub fn receive_bytes(&mut self, size: usize) -> BytesFut {
        let (data_tx, data_rx) = sync_channel(1);
        self.data_request_tx
            .send((size, data_tx))
            .expect("send failed");
        BytesFut { size, data_rx }
    }
}

/// Thread to send messages in the background.
#[derive(Clone, Debug)]
struct SenderThread {
    data_tx: Sender<Vec<u8>>,
}

impl SenderThread {
    pub fn from_writer<W: Debug + Write + Send + 'static>(mut writer: W) -> Self {
        let (data_tx, data_rx) = channel::<Vec<u8>>();
        let _join_handle = thread::spawn(move || {
            for buf in data_rx.iter() {
                writer.write_all(&buf).expect("write failed");
                writer.flush().expect("flush failed");
            }
            writer.flush().expect("flush failed");
        });

        Self { data_tx }
    }

    pub fn send_bytes(&mut self, buf: Vec<u8>) {
        self.data_tx.send(buf).expect("send failed");
    }
}

/// Communicator that uses background threads to send and receive messages.
#[derive(Clone, Debug)]
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
    type MultiFut<T: Serializable> = MyMultiFut<T>;

    fn get_num_parties(&self) -> usize {
        self.num_parties
    }

    fn get_my_id(&self) -> usize {
        self.my_id
    }

    fn send<T: Serializable>(&mut self, party_id: usize, val: T) {
        self.sender_threads
            .get_mut(&party_id)
            .expect(&format!("SenderThread for party {} not found", party_id))
            .send_bytes(val.to_bytes())
    }

    fn send_slice<T: Serializable>(&mut self, party_id: usize, val: &[T]) {
        let mut bytes = vec![0u8; val.len() * T::bytes_required()];
        for (i, v) in val.iter().enumerate() {
            bytes[i * T::bytes_required()..(i + 1) * T::bytes_required()]
                .copy_from_slice(&v.to_bytes());
        }
        self.sender_threads
            .get_mut(&party_id)
            .expect(&format!("SenderThread for party {} not found", party_id))
            .send_bytes(bytes);
    }

    fn receive<T: Serializable>(&mut self, party_id: usize) -> Self::Fut<T> {
        let bytes_fut = self
            .receiver_threads
            .get_mut(&party_id)
            .expect(&format!("ReceiverThread for party {} not found", party_id))
            .receive_bytes(T::bytes_required());
        MyFut::new(bytes_fut)
    }

    fn receive_n<T: Serializable>(&mut self, party_id: usize, n: usize) -> Self::MultiFut<T> {
        let bytes_fut = self
            .receiver_threads
            .get_mut(&party_id)
            .expect(&format!("ReceiverThread for party {} not found", party_id))
            .receive_bytes(n * T::bytes_required());
        MyMultiFut::new(n, bytes_fut)
    }

    fn shutdown(&mut self) {
        self.sender_threads.drain();
        self.receiver_threads.drain();
    }
}
