use clap::{CommandFactory, Parser};
use communicator::tcp::{make_tcp_communicator, NetworkOptions, NetworkPartyInfo};
use communicator::AbstractCommunicator;
use ff::Field;
use oram::doprf::{
    DOPrfParty1, DOPrfParty2, DOPrfParty3, JointDOPrf, MaskedDOPrfParty1, MaskedDOPrfParty2,
    MaskedDOPrfParty3,
};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use std::process;
use std::time::{Duration, Instant};
use utils::field::Fp;

const PARTY_1: usize = 0;
const PARTY_2: usize = 1;
const PARTY_3: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, strum_macros::Display)]
enum Mode {
    Alternating,
    Joint,
    Masked,
    Plain,
}

#[derive(Debug, clap::Parser)]
struct Cli {
    /// ID of this party
    #[arg(long, short = 'i', value_parser = clap::value_parser!(u32).range(0..3))]
    pub party_id: u32,
    /// Output bitsize of the DOPrf
    #[arg(long, short = 's', value_parser = clap::value_parser!(u32).range(1..))]
    pub bitsize: u32,
    /// Number of evaluations to compute
    #[arg(long, short = 'n', value_parser = clap::value_parser!(u32).range(1..))]
    pub num_evaluations: u32,
    /// Which protocol variant to benchmark
    #[arg(long, short = 'm', value_enum)]
    pub mode: Mode,
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

fn make_random_shares(n: usize) -> Vec<Fp> {
    let mut rng = ChaChaRng::from_seed([0u8; 32]);
    (0..n).map(|_| Fp::random(&mut rng)).collect()
}

fn bench_plain<C: AbstractCommunicator>(
    comm: &mut C,
    bitsize: usize,
    num_evaluations: usize,
) -> (Duration, Duration, Duration) {
    let shares = make_random_shares(num_evaluations);
    match comm.get_my_id() {
        PARTY_1 => {
            let mut p1 = DOPrfParty1::<Fp>::new(bitsize);
            let t_start = Instant::now();
            p1.init(comm).expect("init failed");
            let t_after_init = Instant::now();
            p1.preprocess(comm, num_evaluations)
                .expect("preprocess failed");
            let t_after_preprocess = Instant::now();
            for i in 0..num_evaluations {
                p1.eval(comm, 1, &[shares[i]]).expect("eval failed");
            }
            let t_after_eval = Instant::now();
            (
                t_after_init - t_start,
                t_after_preprocess - t_after_init,
                t_after_eval - t_after_preprocess,
            )
        }
        PARTY_2 => {
            let mut p2 = DOPrfParty2::<Fp>::new(bitsize);
            let t_start = Instant::now();
            p2.init(comm).expect("init failed");
            let t_after_init = Instant::now();
            p2.preprocess(comm, num_evaluations)
                .expect("preprocess failed");
            let t_after_preprocess = Instant::now();
            for i in 0..num_evaluations {
                p2.eval(comm, 1, &[shares[i]]).expect("eval failed");
            }
            let t_after_eval = Instant::now();
            (
                t_after_init - t_start,
                t_after_preprocess - t_after_init,
                t_after_eval - t_after_preprocess,
            )
        }
        PARTY_3 => {
            let mut p3 = DOPrfParty3::<Fp>::new(bitsize);
            let t_start = Instant::now();
            p3.init(comm).expect("init failed");
            let t_after_init = Instant::now();
            p3.preprocess(comm, num_evaluations)
                .expect("preprocess failed");
            let t_after_preprocess = Instant::now();
            for i in 0..num_evaluations {
                p3.eval(comm, 1, &[shares[i]]).expect("eval failed");
            }
            let t_after_eval = Instant::now();
            (
                t_after_init - t_start,
                t_after_preprocess - t_after_init,
                t_after_eval - t_after_preprocess,
            )
        }
        _ => panic!("invalid party id"),
    }
}

fn bench_masked<C: AbstractCommunicator>(
    comm: &mut C,
    bitsize: usize,
    num_evaluations: usize,
) -> (Duration, Duration, Duration) {
    let shares = make_random_shares(num_evaluations);
    match comm.get_my_id() {
        PARTY_1 => {
            let mut p1 = MaskedDOPrfParty1::<Fp>::new(bitsize);
            let t_start = Instant::now();
            p1.init(comm).expect("init failed");
            let t_after_init = Instant::now();
            p1.preprocess(comm, num_evaluations)
                .expect("preprocess failed");
            let t_after_preprocess = Instant::now();
            for i in 0..num_evaluations {
                p1.eval(comm, 1, &[shares[i]]).expect("eval failed");
            }
            let t_after_eval = Instant::now();
            (
                t_after_init - t_start,
                t_after_preprocess - t_after_init,
                t_after_eval - t_after_preprocess,
            )
        }
        PARTY_2 => {
            let mut p2 = MaskedDOPrfParty2::<Fp>::new(bitsize);
            let t_start = Instant::now();
            p2.init(comm).expect("init failed");
            let t_after_init = Instant::now();
            p2.preprocess(comm, num_evaluations)
                .expect("preprocess failed");
            let t_after_preprocess = Instant::now();
            for i in 0..num_evaluations {
                p2.eval(comm, 1, &[shares[i]]).expect("eval failed");
            }
            let t_after_eval = Instant::now();
            (
                t_after_init - t_start,
                t_after_preprocess - t_after_init,
                t_after_eval - t_after_preprocess,
            )
        }
        PARTY_3 => {
            let mut p3 = MaskedDOPrfParty3::<Fp>::new(bitsize);
            let t_start = Instant::now();
            p3.init(comm).expect("init failed");
            let t_after_init = Instant::now();
            p3.preprocess(comm, num_evaluations)
                .expect("preprocess failed");
            let t_after_preprocess = Instant::now();
            for i in 0..num_evaluations {
                p3.eval(comm, 1, &[shares[i]]).expect("eval failed");
            }
            let t_after_eval = Instant::now();
            (
                t_after_init - t_start,
                t_after_preprocess - t_after_init,
                t_after_eval - t_after_preprocess,
            )
        }
        _ => panic!("invalid party id"),
    }
}

fn bench_joint<C: AbstractCommunicator>(
    comm: &mut C,
    bitsize: usize,
    num_evaluations: usize,
) -> (Duration, Duration, Duration) {
    let shares = make_random_shares(num_evaluations);
    let mut p = JointDOPrf::<Fp>::new(bitsize);
    let t_start = Instant::now();
    p.init(comm).expect("init failed");
    let t_after_init = Instant::now();
    p.preprocess(comm, num_evaluations)
        .expect("preprocess failed");
    let t_after_preprocess = Instant::now();
    for i in 0..num_evaluations {
        p.eval_to_uint::<_, u128>(comm, &[shares[i]])
            .expect("eval failed");
    }
    let t_after_eval = Instant::now();
    (
        t_after_init - t_start,
        t_after_preprocess - t_after_init,
        t_after_eval - t_after_preprocess,
    )
}

fn bench_alternating<C: AbstractCommunicator>(
    comm: &mut C,
    bitsize: usize,
    num_evaluations: usize,
) -> (Duration, Duration, Duration) {
    let shares = make_random_shares(num_evaluations);
    let mut p1 = DOPrfParty1::<Fp>::new(bitsize);
    let mut p2 = DOPrfParty2::<Fp>::new(bitsize);
    let mut p3 = DOPrfParty3::<Fp>::new(bitsize);
    match comm.get_my_id() {
        PARTY_1 => {
            let t_start = Instant::now();
            p1.init(comm).expect("init failed");
            p2.init(comm).expect("init failed");
            p3.init(comm).expect("init failed");
            let t_after_init = Instant::now();
            p1.preprocess(comm, num_evaluations)
                .expect("preprocess failed");
            p2.preprocess(comm, num_evaluations)
                .expect("preprocess failed");
            p3.preprocess(comm, num_evaluations)
                .expect("preprocess failed");
            let t_after_preprocess = Instant::now();
            for i in 0..num_evaluations {
                p1.eval(comm, 1, &[shares[i]]).expect("eval failed");
                p2.eval(comm, 1, &[shares[i]]).expect("eval failed");
                p3.eval(comm, 1, &[shares[i]]).expect("eval failed");
            }
            let t_after_eval = Instant::now();
            (
                t_after_init - t_start,
                t_after_preprocess - t_after_init,
                t_after_eval - t_after_preprocess,
            )
        }
        PARTY_2 => {
            let t_start = Instant::now();
            p2.init(comm).expect("init failed");
            p3.init(comm).expect("init failed");
            p1.init(comm).expect("init failed");
            let t_after_init = Instant::now();
            p2.preprocess(comm, num_evaluations)
                .expect("preprocess failed");
            p3.preprocess(comm, num_evaluations)
                .expect("preprocess failed");
            p1.preprocess(comm, num_evaluations)
                .expect("preprocess failed");
            let t_after_preprocess = Instant::now();
            for i in 0..num_evaluations {
                p2.eval(comm, 1, &[shares[i]]).expect("eval failed");
                p3.eval(comm, 1, &[shares[i]]).expect("eval failed");
                p1.eval(comm, 1, &[shares[i]]).expect("eval failed");
            }
            let t_after_eval = Instant::now();
            (
                t_after_init - t_start,
                t_after_preprocess - t_after_init,
                t_after_eval - t_after_preprocess,
            )
        }
        PARTY_3 => {
            let t_start = Instant::now();
            p3.init(comm).expect("init failed");
            p1.init(comm).expect("init failed");
            p2.init(comm).expect("init failed");
            let t_after_init = Instant::now();
            p3.preprocess(comm, num_evaluations)
                .expect("preprocess failed");
            p1.preprocess(comm, num_evaluations)
                .expect("preprocess failed");
            p2.preprocess(comm, num_evaluations)
                .expect("preprocess failed");
            let t_after_preprocess = Instant::now();
            for i in 0..num_evaluations {
                p3.eval(comm, 1, &[shares[i]]).expect("eval failed");
                p1.eval(comm, 1, &[shares[i]]).expect("eval failed");
                p2.eval(comm, 1, &[shares[i]]).expect("eval failed");
            }
            let t_after_eval = Instant::now();
            (
                t_after_init - t_start,
                t_after_preprocess - t_after_init,
                t_after_eval - t_after_preprocess,
            )
        }
        _ => panic!("invalid party id"),
    }
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

    let (d_init, d_preprocess, d_eval) = match cli.mode {
        Mode::Plain => bench_plain(
            &mut comm,
            cli.bitsize as usize,
            cli.num_evaluations as usize,
        ),
        Mode::Masked => bench_masked(
            &mut comm,
            cli.bitsize as usize,
            cli.num_evaluations as usize,
        ),
        Mode::Joint => bench_joint(
            &mut comm,
            cli.bitsize as usize,
            cli.num_evaluations as usize,
        ),
        Mode::Alternating => bench_alternating(
            &mut comm,
            cli.bitsize as usize,
            cli.num_evaluations as usize,
        ),
    };

    comm.shutdown();

    println!("=========== DOPrf ============");
    println!("mode: {}", cli.mode);
    println!("- {} bit output", cli.bitsize);
    println!("- {} evaluations", cli.num_evaluations);
    println!("time init:        {:3.3} s", d_init.as_secs_f64());
    println!("time preprocess:  {:3.3} s", d_preprocess.as_secs_f64());
    println!(
        "  per evaluation: {:3.3} s",
        d_preprocess.as_secs_f64() / cli.num_evaluations as f64
    );
    println!("time eval:        {:3.3} s", d_eval.as_secs_f64());
    println!(
        "  per evaluation: {:3.3} s",
        d_eval.as_secs_f64() / cli.num_evaluations as f64
    );
    println!("==============================");
}
