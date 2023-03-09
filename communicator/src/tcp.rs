//! Functionality for communicators using TCP sockets.

use crate::Communicator;
use crate::{AbstractCommunicator, Error};
use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::Duration;

/// Network connection options for a single party: Either we listen for an incoming connection, or
/// we connect to a given host and port.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkPartyInfo {
    /// Listen for the other party to connect.
    Listen,
    /// Connect to the other party at the given host and port.
    Connect(String, u16),
}

/// Network connection options
#[derive(Debug, Clone)]
pub struct NetworkOptions {
    /// Which address to listen on for incoming connections
    pub listen_host: String,
    /// Which port to listen on for incoming connections
    pub listen_port: u16,
    /// Connection info for each party
    pub connect_info: Vec<NetworkPartyInfo>,
    /// How long to try connecting before aborting
    pub connect_timeout_seconds: usize,
}

fn tcp_connect(
    my_id: usize,
    other_id: usize,
    host: &str,
    port: u16,
    timeout_seconds: usize,
) -> Result<TcpStream, Error> {
    // repeatedly try to connect
    fn connect_socket(host: &str, port: u16, timeout_seconds: usize) -> Result<TcpStream, Error> {
        // try every 100ms
        for _ in 0..(10 * timeout_seconds) {
            if let Ok(socket) = TcpStream::connect((host, port)) {
                return Ok(socket);
            }
            thread::sleep(Duration::from_millis(100));
        }
        match TcpStream::connect((host, port)) {
            Ok(socket) => Ok(socket),
            Err(e) => Err(Error::IoError(e)),
        }
    }
    // connect to the other party
    let mut stream = connect_socket(host, port, timeout_seconds)?;
    {
        // send our party id
        let bytes_written = stream.write(&(my_id as u32).to_be_bytes())?;
        if bytes_written != 4 {
            return Err(Error::ConnectionSetupError);
        }
        // check that we talk to the right party
        let mut other_id_bytes = [0u8; 4];
        stream.read_exact(&mut other_id_bytes)?;
        if u32::from_be_bytes(other_id_bytes) != other_id as u32 {
            return Err(Error::ConnectionSetupError);
        }
    }
    Ok(stream)
}

fn tcp_accept_connections(
    my_id: usize,
    options: &NetworkOptions,
) -> Result<HashMap<usize, TcpStream>, Error> {
    // prepare function output
    let mut output = HashMap::<usize, TcpStream>::new();
    // compute set of parties that should connect to us
    let mut expected_parties: HashSet<usize> = options
        .connect_info
        .iter()
        .enumerate()
        .filter_map(|(party_id, npi)| {
            if party_id != my_id && *npi == NetworkPartyInfo::Listen {
                Some(party_id)
            } else {
                None
            }
        })
        .collect();
    // if nobody should connect to us, we are done
    if expected_parties.is_empty() {
        return Ok(output);
    }
    // create a listender and iterate over incoming connections
    let listener = TcpListener::bind((options.listen_host.clone(), options.listen_port))?;
    for mut stream in listener.incoming().filter_map(Result::ok) {
        // see which party has connected
        let mut other_id_bytes = [0u8; 4];
        if stream.read_exact(&mut other_id_bytes).is_err() {
            continue;
        }
        let other_id = u32::from_be_bytes(other_id_bytes) as usize;
        // check if we expect this party
        if !expected_parties.contains(&other_id) {
            continue;
        }
        // respond with our party id
        if stream.write_all(&(my_id as u32).to_be_bytes()).is_err() {
            continue;
        }
        // connection has been established
        expected_parties.remove(&other_id);
        output.insert(other_id, stream);
        // check if we have received connections from every party
        if expected_parties.is_empty() {
            break;
        }
    }
    if !expected_parties.is_empty() {
        Err(Error::ConnectionSetupError)
    } else {
        Ok(output)
    }
}

/// Setup TCP connections
pub fn setup_connection(
    num_parties: usize,
    my_id: usize,
    options: &NetworkOptions,
) -> Result<HashMap<usize, TcpStream>, Error> {
    // make a copy of the options to pass it into the new thread
    let options_cpy: NetworkOptions = (*options).clone();

    // spawn thread to listen for incoming connections
    let listen_thread_handle = thread::spawn(move || tcp_accept_connections(my_id, &options_cpy));

    // prepare the map of connection we will return
    let mut output = HashMap::with_capacity(num_parties - 1);

    // connect to all parties that we are supposed to connect to
    for (party_id, info) in options.connect_info.iter().enumerate() {
        if party_id == my_id {
            continue;
        }
        match info {
            NetworkPartyInfo::Listen => {}
            NetworkPartyInfo::Connect(host, port) => {
                output.insert(
                    party_id,
                    tcp_connect(
                        my_id,
                        party_id,
                        host,
                        *port,
                        options.connect_timeout_seconds,
                    )?,
                );
            }
        }
    }

    // join the listen thread and obtain the connections that reached us
    let accepted_connections = match listen_thread_handle.join() {
        Ok(accepted_connections) => accepted_connections,
        Err(_) => return Err(Error::ConnectionSetupError),
    }?;

    // return the union of both maps
    output.extend(accepted_connections);
    Ok(output)
}

/// Create communicator using TCP connections
pub fn make_tcp_communicator(
    num_parties: usize,
    my_id: usize,
    options: &NetworkOptions,
) -> Result<impl AbstractCommunicator, Error> {
    // create connections with other parties
    let stream_map = setup_connection(num_parties, my_id, options)?;
    stream_map
        .iter()
        .for_each(|(_, s)| s.set_nodelay(true).expect("set_nodelay failed"));
    // use streams as reader/writer pairs
    let rw_map = stream_map
        .into_iter()
        .map(|(party_id, stream)| (party_id, (stream.try_clone().unwrap(), stream)))
        .collect();
    // create new communicator
    Ok(Communicator::from_reader_writer(num_parties, my_id, rw_map))
}

/// Create communicator using TCP connections via localhost
pub fn make_local_tcp_communicators(num_parties: usize) -> Vec<impl AbstractCommunicator> {
    let ports: [u16; 3] = [20_000, 20_001, 20_002];
    let opts: Vec<_> = (0..num_parties)
        .map(|party_id| NetworkOptions {
            listen_host: "localhost".to_owned(),
            listen_port: ports[party_id],
            connect_info: (0..num_parties)
                .map(|other_id| {
                    if other_id < party_id {
                        NetworkPartyInfo::Connect("localhost".to_owned(), ports[other_id])
                    } else {
                        NetworkPartyInfo::Listen
                    }
                })
                .collect(),
            connect_timeout_seconds: 3,
        })
        .collect();

    let communicators: Vec<_> = opts
        .iter()
        .enumerate()
        .map(|(party_id, opts)| {
            let opts_cpy = (*opts).clone();
            thread::spawn(move || make_tcp_communicator(num_parties, party_id, &opts_cpy))
        })
        .collect();
    communicators
        .into_iter()
        .map(|h| h.join().unwrap().unwrap())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Fut;
    use std::thread;

    #[test]
    fn test_tcp_communicators() {
        let num_parties = 3;
        let msg_0: u8 = 42;
        let msg_1: u32 = 0x_dead_beef;
        let msg_2: [u32; 2] = [0x_1333_3337, 0x_c0ff_ffee];

        let ports: [u16; 3] = [20_000, 20_001, 20_002];

        let opts: Vec<_> = (0..num_parties)
            .map(|party_id| NetworkOptions {
                listen_host: "localhost".to_owned(),
                listen_port: ports[party_id],
                connect_info: (0..num_parties)
                    .map(|other_id| {
                        if other_id < party_id {
                            NetworkPartyInfo::Connect("localhost".to_owned(), ports[other_id])
                        } else {
                            NetworkPartyInfo::Listen
                        }
                    })
                    .collect(),
                connect_timeout_seconds: 3,
            })
            .collect();

        let communicators: Vec<_> = opts
            .iter()
            .enumerate()
            .map(|(party_id, opts)| {
                let opts_cpy = (*opts).clone();
                thread::spawn(move || make_tcp_communicator(num_parties, party_id, &opts_cpy))
            })
            .collect();
        let communicators: Vec<_> = communicators
            .into_iter()
            .map(|h| h.join().unwrap().unwrap())
            .collect();

        let thread_handles: Vec<_> = communicators
            .into_iter()
            .enumerate()
            .map(|(party_id, mut communicator)| {
                thread::spawn(move || {
                    if party_id == 0 {
                        let fut_1 = communicator.receive::<u32>(1).unwrap();
                        let fut_2 = communicator.receive::<[u32; 2]>(2).unwrap();
                        communicator.send(1, msg_0).unwrap();
                        communicator.send(2, msg_0).unwrap();
                        let val_1 = fut_1.get();
                        let val_2 = fut_2.get();
                        assert!(val_1.is_ok());
                        assert!(val_2.is_ok());
                        assert_eq!(val_1.unwrap(), msg_1);
                        assert_eq!(val_2.unwrap(), msg_2);
                    } else if party_id == 1 {
                        let fut_0 = communicator.receive::<u8>(0).unwrap();
                        let fut_2 = communicator.receive::<[u32; 2]>(2).unwrap();
                        communicator.send(0, msg_1).unwrap();
                        communicator.send(2, msg_1).unwrap();
                        let val_0 = fut_0.get();
                        let val_2 = fut_2.get();
                        assert!(val_0.is_ok());
                        assert!(val_2.is_ok());
                        assert_eq!(val_0.unwrap(), msg_0);
                        assert_eq!(val_2.unwrap(), msg_2);
                    } else if party_id == 2 {
                        let fut_0 = communicator.receive::<u8>(0).unwrap();
                        let fut_1 = communicator.receive::<u32>(1).unwrap();
                        communicator.send(0, msg_2).unwrap();
                        communicator.send(1, msg_2).unwrap();
                        let val_0 = fut_0.get();
                        let val_1 = fut_1.get();
                        assert!(val_0.is_ok());
                        assert!(val_1.is_ok());
                        assert_eq!(val_0.unwrap(), msg_0);
                        assert_eq!(val_1.unwrap(), msg_1);
                    }
                    communicator.shutdown();
                })
            })
            .collect();

        thread_handles.into_iter().for_each(|h| h.join().unwrap());
    }
}
