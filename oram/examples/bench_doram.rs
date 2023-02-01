use clap::{CommandFactory, Parser};
use communicator::tcp::{make_tcp_communicator, NetworkOptions, NetworkPartyInfo};
use communicator::AbstractCommunicator;
use cuckoo::hash::AesHashFunction;
use dpf::mpdpf::SmartMpDpf;
use dpf::spdpf::HalfTreeSpDpf;
use ff::{Field, PrimeField};
use oram::common::{InstructionShare, Operation};
use oram::oram::{DistributedOram, DistributedOramProtocol, Runtimes};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use std::process;
use std::time::Instant;
use utils::field::Fp;

type MPDPF = SmartMpDpf<Fp, HalfTreeSpDpf<Fp>, AesHashFunction<u16>>;
type DOram = DistributedOramProtocol<Fp, MPDPF, HalfTreeSpDpf<Fp>>;

#[derive(Debug, clap::Parser)]
struct Cli {
    /// ID of this party
    #[arg(long, short = 'i', value_parser = clap::value_parser!(u32).range(0..3))]
    pub party_id: u32,
    /// Log2 of the database size, must be even
    #[arg(long, short = 's', value_parser = parse_log_db_size)]
    pub log_db_size: u32,
    /// Which address to listen on for incoming connections
    #[arg(long, short = 'l')]
    pub listen_host: String,
    /// Which port to listen on for incoming connections
    #[arg(long, short = 'p', value_parser = clap::value_parser!(u16).range(1..))]
    pub listen_port: u16,
    /// Connection info for each party
    #[arg(long, short = 'c', value_name = "PARTY_ID>:<HOST>:<PORT", value_parser = parse_connect)]
    pub connect: Vec<(usize, String, u16)>,
    /// How long to try connecting before aborting
    #[arg(long, short = 't', default_value_t = 10)]
    pub connect_timeout_seconds: usize,
}

fn parse_log_db_size(s: &str) -> Result<u32, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let log_db_size: u32 = s.parse()?;
    if log_db_size & 1 == 1 {
        return Err(clap::Error::raw(
            clap::error::ErrorKind::InvalidValue,
            format!("log_db_size must be even"),
        )
        .into());
    }
    Ok(log_db_size)
}

fn parse_connect(
    s: &str,
) -> Result<(usize, String, u16), Box<dyn std::error::Error + Send + Sync + 'static>> {
    let parts: Vec<_> = s.split(":").collect();
    if parts.len() != 3 {
        return Err(clap::Error::raw(
            clap::error::ErrorKind::ValueValidation,
            format!("'{}' has not the format '<party-id>:<host>:<post>'", s),
        )
        .into());
    }
    let party_id: usize = parts[0].parse()?;
    let host = parts[1];
    let port: u16 = parts[2].parse()?;
    if port == 0 {
        return Err(clap::Error::raw(
            clap::error::ErrorKind::ValueValidation,
            "the port needs to be positive",
        )
        .into());
    }
    Ok((party_id, host.to_owned(), port))
}

fn main() {
    let cli = Cli::parse();

    let mut netopts = NetworkOptions {
        listen_host: cli.listen_host,
        listen_port: cli.listen_port,
        connect_info: vec![NetworkPartyInfo::Listen; 3],
        connect_timeout_seconds: cli.connect_timeout_seconds,
    };

    for c in cli.connect {
        if netopts.connect_info[c.0] != NetworkPartyInfo::Listen {
            println!(
                "{}",
                clap::Error::raw(
                    clap::error::ErrorKind::ValueValidation,
                    format!("multiple connect arguments for party {}", c.0),
                )
                .format(&mut Cli::command())
            );
            process::exit(1);
        }
        netopts.connect_info[c.0] = NetworkPartyInfo::Connect(c.1, c.2);
    }

    let mut comm = match make_tcp_communicator(3, cli.party_id as usize, &netopts) {
        Ok(comm) => comm,
        Err(e) => {
            eprintln!("network setup failed: {:?}", e);
            process::exit(1);
        }
    };

    let mut doram = DOram::new(cli.party_id as usize, cli.log_db_size);

    let db_size = 1 << cli.log_db_size;
    let db_share: Vec<_> = vec![Fp::ZERO; db_size];
    let stash_size = 1 << (cli.log_db_size >> 1);

    let instructions = if cli.party_id == 0 {
        let mut rng = ChaChaRng::from_seed([0u8; 32]);
        (0..stash_size)
            .map(|_| InstructionShare {
                operation: Operation::Write.encode(),
                address: Fp::from_u128(rng.gen_range(0..db_size) as u128),
                value: Fp::random(&mut rng),
            })
            .collect()
    } else {
        vec![
            InstructionShare {
                operation: Fp::ZERO,
                address: Fp::ZERO,
                value: Fp::ZERO
            };
            stash_size
        ]
    };

    let t_start = Instant::now();

    doram.init(&mut comm, &db_share).expect("init failed");

    let d_init = Instant::now() - t_start;

    let mut runtimes = Runtimes::default();

    let t_start = Instant::now();
    for (_i, inst) in instructions.iter().enumerate() {
        // println!("executing instruction #{i}: {inst:?}");
        runtimes = doram
            .access_with_runtimes(&mut comm, *inst, Some(runtimes))
            .expect("access failed")
            .1
            .unwrap();
    }
    let d_accesses = Instant::now() - t_start;

    println!("time init: {:.3} s", d_init.as_secs_f64());
    println!("time accesses: {:.3} s", d_accesses.as_secs_f64());
    println!(
        "time per accesses: {:.3} s",
        d_accesses.as_secs_f64() / stash_size as f64
    );

    comm.shutdown();

    runtimes.print(cli.party_id as usize + 1, stash_size);
}
