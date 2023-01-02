use cuckoo::cuckoo::{Hasher, Parameters};
use cuckoo::hash::AesHashFunction;

fn main() {
    let log_domain_sizes = [4, 8, 12, 16, 20, 24, 26];

    println!("domain  inputs  #bucks  maxbucks  avgbucks  #emptybucks");

    for log_domain_size in log_domain_sizes {
        let log_number_inputs = log_domain_size / 2;
        let params = Parameters::<AesHashFunction<u32>, _>::sample(1 << log_number_inputs);
        let number_buckets = params.get_number_buckets();
        let hasher = Hasher::new(params);
        let buckets = hasher.hash_domain_into_buckets(1 << log_domain_size);
        let max_bucket_size = buckets.iter().map(|b| b.len()).max().unwrap();
        let avg_bucket_size = buckets.iter().map(|b| b.len()).sum::<usize>() / number_buckets;
        let number_empty_buckets = buckets.iter().map(|b| b.len()).filter(|&l| l == 0).count();
        println!(
            "{:6}  {:6}  {:6}  {:8}  {:8}  {:11}",
            log_domain_size,
            log_number_inputs,
            number_buckets,
            max_bucket_size,
            avg_bucket_size,
            number_empty_buckets
        );
    }
}
