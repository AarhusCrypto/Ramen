use clap::{CommandFactory, Parser};
use communicator::tcp::{make_tcp_communicator, NetworkOptions, NetworkPartyInfo};
use communicator::{AbstractCommunicator, CommunicationStats};
use cuckoo::hash::AesHashFunction;
use dpf::mpdpf::SmartMpDpf;
use dpf::spdpf::HalfTreeSpDpf;
use ff::{Field, PrimeField};
use oram::common::{InstructionShare, Operation};
use oram::oram::{
    DistributedOram, DistributedOramProtocol, ProtocolStep as OramProtocolStep,
    Runtimes as OramRuntimes,
};
use oram::tools::BenchmarkMetaData;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use rayon;
use serde;
use serde_json;
use std::collections::HashMap;
use std::process;
use std::time::{Duration, Instant};
use strum::IntoEnumIterator;
use utils::field::Fp;

type MPDPF = SmartMpDpf<Fp, HalfTreeSpDpf<Fp>, AesHashFunction<u16>>;
type DOram = DistributedOramProtocol<Fp, MPDPF, HalfTreeSpDpf<Fp>>;

#[derive(Debug, clap::Parser)]
struct Cli {
    /// ID of this party
    #[arg(long, short = 'i', value_parser = clap::value_parser!(u32).range(0..3))]
    pub party_id: u32,
    /// Log2 of the database size, must be even
    #[arg(long, short = 's', value_parser = clap::value_parser!(u32).range(4..))]
    pub log_db_size: u32,
    /// Use preprocessing
    #[arg(long)]
    pub preprocess: bool,
    /// How many threads to use for the computation
    #[arg(long, short = 't', default_value_t = 1)]
    pub threads: usize,
    /// How many threads to use for the preprocessing phase (default: same as -t)
    #[arg(long, default_value_t = -1)]
    pub threads_prep: isize,
    /// How many threads to use for the online phase (default: same as -t)
    #[arg(long, default_value_t = -1)]
    pub threads_online: isize,
    /// Output statistics in JSON
    #[arg(long, short = 'j')]
    pub json: bool,
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
    #[arg(long, default_value_t = 10)]
    pub connect_timeout_seconds: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
struct BenchmarkResults {
    party_id: usize,
    log_db_size: u32,
    preprocess: bool,
    threads_prep: usize,
    threads_online: usize,
    comm_stats_preprocess: HashMap<usize, CommunicationStats>,
    comm_stats_access: HashMap<usize, CommunicationStats>,
    runtimes: HashMap<String, Duration>,
    meta: BenchmarkMetaData,
}

impl BenchmarkResults {
    pub fn new(
        cli: &Cli,
        comm_stats_preprocess: &HashMap<usize, CommunicationStats>,
        comm_stats_access: &HashMap<usize, CommunicationStats>,
        runtimes: &OramRuntimes,
    ) -> Self {
        let mut runtime_map = HashMap::new();
        for step in OramProtocolStep::iter() {
            runtime_map.insert(step.to_string(), runtimes.get(step));
        }

        let threads_prep = if cli.threads_prep < 0 {
            cli.threads
        } else {
            cli.threads_prep as usize
        };
        let threads_online = if cli.threads_online < 0 {
            cli.threads
        } else {
            cli.threads_online as usize
        };

        Self {
            party_id: cli.party_id as usize,
            log_db_size: cli.log_db_size,
            preprocess: cli.preprocess,
            threads_prep,
            threads_online,
            comm_stats_preprocess: comm_stats_preprocess.clone(),
            comm_stats_access: comm_stats_access.clone(),
            runtimes: runtime_map,
            meta: BenchmarkMetaData::collect(),
        }
    }
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
        listen_host: cli.listen_host.clone(),
        listen_port: cli.listen_port,
        connect_info: vec![NetworkPartyInfo::Listen; 3],
        connect_timeout_seconds: cli.connect_timeout_seconds,
    };

    let threads_prep = if cli.threads_prep < 0 {
        cli.threads
    } else {
        cli.threads_prep as usize
    };
    let threads_online = if cli.threads_online < 0 {
        cli.threads
    } else {
        cli.threads_online as usize
    };

    for c in cli.connect.iter() {
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
        netopts.connect_info[c.0] = NetworkPartyInfo::Connect(c.1.clone(), c.2);
    }

    let mut comm = match make_tcp_communicator(3, cli.party_id as usize, &netopts) {
        Ok(comm) => comm,
        Err(e) => {
            eprintln!("network setup failed: {:?}", e);
            process::exit(1);
        }
    };

    let mut doram = DOram::new(cli.party_id as usize, 1 << cli.log_db_size);

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

    doram.init(&mut comm, &db_share).expect("init failed");

    let thread_pool_prep = rayon::ThreadPoolBuilder::new()
        .thread_name(|i| format!("thread-prep-{i}"))
        .num_threads(threads_prep)
        .build()
        .unwrap();

    comm.reset_stats();
    let mut runtimes = OramRuntimes::default();

    let d_preprocess = if cli.preprocess {
        let t_start = Instant::now();

        runtimes = thread_pool_prep.install(|| {
            doram
                .preprocess_with_runtimes(&mut comm, 1, Some(runtimes))
                .expect("preprocess failed")
                .unwrap()
        });

        t_start.elapsed()
    } else {
        Default::default()
    };

    drop(thread_pool_prep);

    let comm_stats_preprocess = comm.get_stats();
    comm.reset_stats();

    let thread_pool_online = rayon::ThreadPoolBuilder::new()
        .thread_name(|i| format!("thread-online-{i}"))
        .num_threads(threads_online)
        .build()
        .unwrap();

    let t_start = Instant::now();
    for (_i, inst) in instructions.iter().enumerate() {
        // println!("executing instruction #{i}: {inst:?}");
        runtimes = thread_pool_online.install(|| {
            doram
                .access_with_runtimes(&mut comm, *inst, Some(runtimes))
                .expect("access failed")
                .1
                .unwrap()
        });
    }
    let d_accesses = Instant::now() - t_start;

    let comm_stats_access = comm.get_stats();

    drop(thread_pool_online);

    comm.shutdown();

    let results =
        BenchmarkResults::new(&cli, &comm_stats_preprocess, &comm_stats_access, &runtimes);

    if cli.json {
        println!("{}", serde_json::to_string(&results).unwrap());
    } else {
        println!(
            "time preprocess:  {:10.3} ms",
            d_preprocess.as_secs_f64() * 1000.0
        );
        println!(
            "   per accesses:  {:10.3} ms",
            d_preprocess.as_secs_f64() * 1000.0 / stash_size as f64
        );
        println!(
            "time accesses:    {:10.3} ms{}",
            d_accesses.as_secs_f64() * 1000.0,
            if cli.preprocess {
                "  (online only)"
            } else {
                ""
            }
        );
        println!(
            "   per accesses:  {:10.3} ms",
            d_accesses.as_secs_f64() * 1000.0 / stash_size as f64
        );
        runtimes.print(cli.party_id as usize + 1, stash_size);
        println!("communication preprocessing: {comm_stats_preprocess:#?}");
        println!("communication accesses: {comm_stats_access:#?}");
    }
}
