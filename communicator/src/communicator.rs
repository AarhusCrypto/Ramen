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
    join_handle: thread::JoinHandle<Result<(usize, usize), Error>>,
}

impl ReceiverThread {
    pub fn from_reader<R: Debug + Read + Send + 'static>(reader: R) -> Self {
        let mut reader = BufReader::with_capacity(1 << 16, reader);
        let (buf_tx, buf_rx) = channel::<Vec<u8>>();
        let buf_rx = Arc::new(Mutex::new(buf_rx));
        let join_handle = thread::Builder::new()
            .name("Receiver".to_owned())
            .spawn(move || {
                let mut num_msgs_received = 0;
                let mut num_bytes_received = 0;
                loop {
                    let mut msg_size = [0u8; 4];
                    reader.read_exact(&mut msg_size)?;
                    let msg_size = u32::from_be_bytes(msg_size) as usize;
                    if msg_size == 0xffffffff {
                        return Ok((num_msgs_received, num_bytes_received));
                    }
                    let mut buf = vec![0u8; msg_size];
                    reader.read_exact(&mut buf)?;
                    match buf_tx.send(buf) {
                        Ok(_) => (),
                        Err(_) => return Ok((num_msgs_received, num_bytes_received)), // we need to shutdown
                    }
                    num_msgs_received += 1;
                    num_bytes_received += 4 + msg_size;
                }
            })
            .unwrap();
        Self {
            join_handle,
            buf_rx,
        }
    }

    pub fn receive<T: Serializable>(&mut self) -> Result<MyFut<T>, Error> {
        Ok(MyFut::new(self.buf_rx.clone()))
    }

    pub fn join(self) -> Result<(usize, usize), Error> {
        drop(self.buf_rx);
        self.join_handle.join().expect("join failed")
    }
}

/// Thread to send messages in the background.
#[derive(Debug)]
struct SenderThread {
    buf_tx: Sender<Vec<u8>>,
    join_handle: thread::JoinHandle<Result<(usize, usize), Error>>,
}

impl SenderThread {
    pub fn from_writer<W: Debug + Write + Send + 'static>(writer: W) -> Self {
        let mut writer = BufWriter::with_capacity(1 << 16, writer);
        let (buf_tx, buf_rx) = channel::<Vec<u8>>();
        let join_handle = thread::Builder::new()
            .name("Sender-1".to_owned())
            .spawn(move || {
                let mut num_msgs_sent = 0;
                let mut num_bytes_sent = 0;
                for buf in buf_rx.iter() {
                    writer.write_all(&((buf.len() as u32).to_be_bytes()))?;
                    writer.write_all(&buf)?;
                    writer.flush()?;
                    num_msgs_sent += 1;
                    num_bytes_sent += 4 + buf.len();
                }
                writer.write_all(&[0xff, 0xff, 0xff, 0xff])?;
                writer.flush()?;
                Ok((num_msgs_sent, num_bytes_sent))
            })
            .unwrap();
        Self {
            buf_tx,
            join_handle,
        }
    }

    pub fn send<T: Serializable>(&mut self, data: T) -> Result<(), Error> {
        let buf =
            bincode::encode_to_vec(data, bincode::config::standard().skip_fixed_array_length())?;
        self.buf_tx.send(buf)?;
        Ok(())
    }

    pub fn join(self) -> Result<(usize, usize), Error> {
        drop(self.buf_tx);
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

    fn shutdown(&mut self) -> HashMap<usize, CommunicationStats> {
        let mut comm_stats: HashMap<usize, CommunicationStats> = self
            .sender_threads
            .drain()
            .map(|(party_id, t)| {
                (party_id, {
                    let (num_msgs_sent, num_bytes_sent) = t
                        .join()
                        .expect(&format!("join of sender thread {party_id} failed"));
                    CommunicationStats {
                        num_msgs_sent,
                        num_bytes_sent,
                        num_msgs_received: 0,
                        num_bytes_received: 0,
                    }
                })
            })
            .collect();
        self.receiver_threads.drain().for_each(|(party_id, t)| {
            let (num_msgs_received, num_bytes_received) = t
                .join()
                .expect(&format!("join of receiver thread {party_id} failed"));
            let cs = comm_stats
                .get_mut(&party_id)
                .expect(&format!("no comm stats for party {party_id} found"));
            cs.num_msgs_received = num_msgs_received;
            cs.num_bytes_received = num_bytes_received;
        });

        comm_stats
    }
}
