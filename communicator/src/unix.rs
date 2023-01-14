use crate::communicator::Communicator;
use crate::AbstractCommunicator;
use std::collections::HashMap;
use std::os::unix::net::UnixStream;

/// Create a set of connected Communicators that are based on local Unix sockets
pub fn make_unix_communicators(num_parties: usize) -> Vec<impl AbstractCommunicator> {
    // prepare maps for each parties to store readers and writers to every other party
    let mut rw_maps: Vec<_> = (0..num_parties)
        .map(|_| HashMap::with_capacity(num_parties - 1))
        .collect();
    // create pairs of unix sockets connecting each pair of parties
    for party_i in 0..num_parties {
        for party_j in 0..party_i {
            let (stream_i_to_j, stream_j_to_i) = UnixStream::pair().unwrap();
            rw_maps[party_i].insert(party_j, (stream_i_to_j.try_clone().unwrap(), stream_i_to_j));
            rw_maps[party_j].insert(party_i, (stream_j_to_i.try_clone().unwrap(), stream_j_to_i));
        }
    }
    // create communicators from the reader/writer maps
    rw_maps
        .into_iter()
        .enumerate()
        .map(|(party_id, rw_map)| Communicator::from_reader_writer(num_parties, party_id, rw_map))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Fut;
    use std::thread;

    #[test]
    fn test_unix_communicators() {
        let num_parties = 3;
        let msg_0: u8 = 42;
        let msg_1: u32 = 0x_dead_beef;
        let msg_2: [u32; 2] = [0x_1333_3337, 0x_c0ff_ffee];

        let communicators = make_unix_communicators(num_parties);

        let thread_handles: Vec<_> = communicators
            .into_iter()
            .enumerate()
            .map(|(party_id, mut communicator)| {
                thread::spawn(move || {
                    if party_id == 0 {
                        let fut_1 = communicator.receive::<u32>(1);
                        let fut_2 = communicator.receive::<[u32; 2]>(2);
                        communicator.send(1, msg_0);
                        communicator.send(2, msg_0);
                        let val_1 = fut_1.get();
                        let val_2 = fut_2.get();
                        assert!(val_1.is_ok());
                        assert!(val_2.is_ok());
                        assert_eq!(val_1.unwrap(), msg_1);
                        assert_eq!(val_2.unwrap(), msg_2);
                    } else if party_id == 1 {
                        let fut_0 = communicator.receive::<u8>(0);
                        let fut_2 = communicator.receive::<[u32; 2]>(2);
                        communicator.send(0, msg_1);
                        communicator.send(2, msg_1);
                        let val_0 = fut_0.get();
                        let val_2 = fut_2.get();
                        assert!(val_0.is_ok());
                        assert!(val_2.is_ok());
                        assert_eq!(val_0.unwrap(), msg_0);
                        assert_eq!(val_2.unwrap(), msg_2);
                    } else if party_id == 2 {
                        let fut_0 = communicator.receive::<u8>(0);
                        let fut_1 = communicator.receive::<u32>(1);
                        communicator.send(0, msg_2);
                        communicator.send(1, msg_2);
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
