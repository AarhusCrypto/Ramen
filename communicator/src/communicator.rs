use crate::{AbstractCommunicator, CommunicationStats, Error, Fut, Serializable};
use bincode;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::{BufReader, BufWriter, Read, Write};
use std::marker::PhantomData;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

pub struct MyFut<T: Serializable> {
    buf_rx: Arc<Mutex<Receiver<Vec<u8>>>>,
    _phantom: PhantomData<T>,
}

impl<T: Serializable> MyFut<T> {
    fn new(buf_rx: Arc<Mutex<Receiver<Vec<u8>>>>) -> Self {
        Self {
            buf_rx,
            _phantom: PhantomData,
        }
    }
}

impl<T: Serializable> Fut<T> for MyFut<T> {
    fn get(self) -> Result<T, Error> {
        let buf = self.buf_rx.lock().unwrap().recv()?;
        let (data, size) = bincode::decode_from_slice(
            &buf,
            bincode::config::standard().skip_fixed_array_length(),
        )?;
        assert_eq!(size, buf.len());
        Ok(data)
    }
}

/// Thread to receive messages in the background.
#[derive(Debug)]
struct ReceiverThread {
    buf_rx: Arc<Mutex<Receiver<Vec<u8>>>>,
    join_handle: thread::JoinHandle<Result<(), Error>>,
    stats: Arc<Mutex<[usize; 2]>>,
}

impl ReceiverThread {
    pub fn from_reader<R: Debug + Read + Send + 'static>(reader: R) -> Self {
        let mut reader = BufReader::with_capacity(1 << 16, reader);
        let (buf_tx, buf_rx) = channel::<Vec<u8>>();
        let buf_rx = Arc::new(Mutex::new(buf_rx));
        let stats = Arc::new(Mutex::new([0usize; 2]));
        let stats_clone = stats.clone();
        let join_handle = thread::Builder::new()
            .name("Receiver".to_owned())
            .spawn(move || {
                loop {
                    let mut msg_size = [0u8; 4];
                    reader.read_exact(&mut msg_size)?;
                    let msg_size = u32::from_be_bytes(msg_size) as usize;
                    if msg_size == 0xffffffff {
                        return Ok(());
                    }
                    let mut buf = vec![0u8; msg_size];
                    reader.read_exact(&mut buf)?;
                    match buf_tx.send(buf) {
                        Ok(_) => (),
                        Err(_) => return Ok(()), // we need to shutdown
                    }
                    {
                        let mut guard = stats.lock().unwrap();
                        guard[0] += 1;
                        guard[1] += 4 + msg_size;
                    }
                }
            })
            .unwrap();
        Self {
            join_handle,
            buf_rx,
            stats: stats_clone,
        }
    }

    pub fn receive<T: Serializable>(&mut self) -> Result<MyFut<T>, Error> {
        Ok(MyFut::new(self.buf_rx.clone()))
    }

    pub fn join(self) -> Result<(), Error> {
        drop(self.buf_rx);
        self.join_handle.join().expect("join failed")
    }

    pub fn get_stats(&self) -> [usize; 2] {
        *self.stats.lock().unwrap()
    }

    pub fn reset_stats(&mut self) {
        *self.stats.lock().unwrap() = [0usize; 2];
    }
}

/// Thread to send messages in the background.
#[derive(Debug)]
struct SenderThread {
    buf_tx: Sender<Vec<u8>>,
    join_handle: thread::JoinHandle<Result<(), Error>>,
}

impl SenderThread {
    pub fn from_writer<W: Debug + Write + Send + 'static>(writer: W) -> Self {
        let mut writer = BufWriter::with_capacity(1 << 16, writer);
        let (buf_tx, buf_rx) = channel::<Vec<u8>>();
        let join_handle = thread::Builder::new()
            .name("Sender-1".to_owned())
            .spawn(move || {
                for buf in buf_rx.iter() {
                    writer.write_all(&((buf.len() as u32).to_be_bytes()))?;
                    debug_assert!(buf.len() <= u32::MAX as usize);
                    writer.write_all(&buf)?;
                    writer.flush()?;
                }
                writer.write_all(&[0xff, 0xff, 0xff, 0xff])?;
                writer.flush()?;
                Ok(())
            })
            .unwrap();
        Self {
            buf_tx,
            join_handle,
        }
    }

    pub fn send<T: Serializable>(&mut self, data: T) -> Result<usize, Error> {
        let buf =
            bincode::encode_to_vec(data, bincode::config::standard().skip_fixed_array_length())?;
        let num_bytes = 4 + buf.len();
        self.buf_tx.send(buf)?;
        Ok(num_bytes)
    }

    pub fn send_slice<T: Serializable>(&mut self, data: &[T]) -> Result<usize, Error> {
        let buf =
            bincode::encode_to_vec(data, bincode::config::standard().skip_fixed_array_length())?;
        let num_bytes = 4 + buf.len();
        self.buf_tx.send(buf)?;
        Ok(num_bytes)
    }

    pub fn join(self) -> Result<(), Error> {
        drop(self.buf_tx);
        self.join_handle.join().expect("join failed")
    }
}

/// Communicator that uses background threads to send and receive messages.
#[derive(Debug)]
pub struct Communicator {
    num_parties: usize,
    my_id: usize,
    comm_stats: HashMap<usize, CommunicationStats>,
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

        let comm_stats = (0..num_parties)
            .filter_map(|party_id| {
                if party_id == my_id {
                    None
                } else {
                    Some((party_id, Default::default()))
                }
            })
            .collect();

        Self {
            num_parties,
            my_id,
            comm_stats,
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
                let num_bytes = t.send(val)?;
                let cs = self.comm_stats.get_mut(&party_id).unwrap();
                cs.num_bytes_sent += num_bytes;
                cs.num_msgs_sent += 1;
                Ok(())
            }
            None => Err(Error::LogicError(format!(
                "SenderThread for party {party_id} not found"
            ))),
        }
    }

    fn send_slice<T: Serializable>(&mut self, party_id: usize, val: &[T]) -> Result<(), Error> {
        match self.sender_threads.get_mut(&party_id) {
            Some(t) => {
                let num_bytes = t.send_slice(val)?;
                let cs = self.comm_stats.get_mut(&party_id).unwrap();
                cs.num_bytes_sent += num_bytes;
                cs.num_msgs_sent += 1;
                Ok(())
            }
            None => Err(Error::LogicError(format!(
                "SenderThread for party {party_id} not found"
            ))),
        }
    }

    fn receive<T: Serializable>(&mut self, party_id: usize) -> Result<Self::Fut<T>, Error> {
        match self.receiver_threads.get_mut(&party_id) {
            Some(t) => t.receive::<T>(),
            None => Err(Error::LogicError(format!(
                "ReceiverThread for party {party_id} not found"
            ))),
        }
    }

    fn shutdown(&mut self) {
        self.sender_threads.drain().for_each(|(party_id, t)| {
            t.join()
                .unwrap_or_else(|_| panic!("join of sender thread {party_id} failed"))
        });
        self.receiver_threads.drain().for_each(|(party_id, t)| {
            t.join()
                .unwrap_or_else(|_| panic!("join of receiver thread {party_id} failed"))
        });
    }

    fn get_stats(&self) -> HashMap<usize, CommunicationStats> {
        let mut cs = self.comm_stats.clone();
        self.receiver_threads.iter().for_each(|(party_id, t)| {
            let [num_msgs_received, num_bytes_received] = t.get_stats();
            let cs_i = cs.get_mut(party_id).unwrap();
            cs_i.num_msgs_received = num_msgs_received;
            cs_i.num_bytes_received = num_bytes_received;
        });
        cs
    }

    fn reset_stats(&mut self) {
        self.comm_stats
            .iter_mut()
            .for_each(|(_, cs)| *cs = Default::default());
        self.receiver_threads
            .iter_mut()
            .for_each(|(_, t)| t.reset_stats());
    }
}
