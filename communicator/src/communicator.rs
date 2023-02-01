use crate::{AbstractCommunicator, Error, Fut, Serializable};
use bincode;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::{BufReader, BufWriter, Read, Write};
use std::marker::PhantomData;
use std::mem;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

struct SharedState {
    pub mutex: Mutex<Option<Vec<u8>>>,
    pub cvar: Condvar,
}

impl SharedState {
    pub fn new() -> Self {
        Self {
            mutex: Mutex::new(None),
            cvar: Condvar::new(),
        }
    }

    pub fn put(&self, data: Vec<u8>) {
        let mut lock = self.mutex.lock().unwrap();
        *lock = Some(data);
        self.cvar.notify_one();
    }

    pub fn get(&self) -> Vec<u8> {
        let mut val = None;
        let mut lock = self
            .cvar
            .wait_while(self.mutex.lock().unwrap(), |x| x.is_none())
            .unwrap();
        mem::swap(&mut val, &mut lock);
        val.unwrap()
    }
}

pub struct MyFut<T: Serializable> {
    shared_state: Arc<SharedState>,
    _phantom: PhantomData<T>,
}

impl<T: Serializable> MyFut<T> {
    fn new(shared_state: Arc<SharedState>) -> Self {
        Self {
            shared_state,
            _phantom: PhantomData,
        }
    }
}

impl<T: Serializable> Fut<T> for MyFut<T> {
    fn get(self) -> Result<T, Error> {
        let buf = self.shared_state.get();
        let (data, size) = bincode::decode_from_slice(&buf, bincode::config::standard())?;
        assert_eq!(size, buf.len());
        Ok(data)
    }
}

/// Thread to receive messages in the background.
#[derive(Debug)]
struct ReceiverThread {
    promise_tx: Sender<Arc<SharedState>>,
    join_handle_1: thread::JoinHandle<Result<(), Error>>,
    join_handle_2: thread::JoinHandle<Result<(), Error>>,
}

impl ReceiverThread {
    pub fn from_reader<R: Debug + Read + Send + 'static>(reader: R) -> Self {
        let mut reader = BufReader::new(reader);
        let (promise_tx, promise_rx) = channel::<Arc<SharedState>>();
        let (buf_tx, buf_rx) = channel::<Vec<u8>>();
        let join_handle_1 = thread::Builder::new()
            .name("Receiver-1".to_owned())
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
                }
            })
            .unwrap();
        let join_handle_2 = thread::Builder::new()
            .name("Receiver-2".to_owned())
            .spawn(move || {
                for shared_state in promise_rx {
                    let buf = buf_rx.recv()?;
                    shared_state.put(buf);
                }
                Ok(())
            })
            .unwrap();
        Self {
            promise_tx,
            join_handle_1,
            join_handle_2,
        }
    }

    pub fn receive<T: Serializable>(&mut self) -> Result<MyFut<T>, Error> {
        let shared_state_promise = Arc::new(SharedState::new());
        let shared_state_future = shared_state_promise.clone();
        self.promise_tx.send(shared_state_promise)?;
        Ok(MyFut::new(shared_state_future))
    }

    pub fn join(self) -> Result<(), Error> {
        drop(self.promise_tx);
        self.join_handle_1.join().expect("join failed")?;
        self.join_handle_2.join().expect("join failed")?;
        Ok(())
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
        let mut writer = BufWriter::with_capacity(1 << 15, writer);
        let (buf_tx, buf_rx) = channel::<Vec<u8>>();
        let join_handle = thread::Builder::new()
            .name("Sender-1".to_owned())
            .spawn(move || {
                for buf in buf_rx.iter() {
                    writer.write_all(&((buf.len() as u32).to_be_bytes()))?;
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

    pub fn send<T: Serializable>(&mut self, data: T) -> Result<(), Error> {
        let buf = bincode::encode_to_vec(data, bincode::config::standard())?;
        self.buf_tx.send(buf)?;
        Ok(())
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
        self.sender_threads
            .drain()
            .for_each(|(_, t)| t.join().unwrap());
        self.receiver_threads
            .drain()
            .for_each(|(_, t)| t.join().unwrap());
    }
}
