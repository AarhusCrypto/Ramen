use crate::hash::HashFunction;
use core::array;
use libm::erf;
use std::f64::consts::SQRT_2;

const NUMBER_HASH_FUNCTIONS: usize = 3;

#[derive(Clone, Copy, Debug)]
pub struct Parameters<H: HashFunction> {
    number_inputs: usize,
    number_buckets: usize,
    hash_function_descriptions: [H::Description; 3],
}

impl<H: HashFunction> Parameters<H> {
    /// Samples three hash functions
    pub fn sample(number_inputs: usize) -> Self {
        let number_buckets = Self::compute_number_buckets(number_inputs);
        let hash_function_descriptions =
            array::from_fn(|_| H::sample(number_buckets.try_into().unwrap()).to_description());

        Parameters::<H> {
            number_inputs,
            number_buckets,
            hash_function_descriptions,
        }
    }

    /// Compute how many buckets we need for the cuckoo table
    fn compute_number_buckets(number_inputs: usize) -> usize {
        assert_ne!(number_inputs, 0);

        let statistical_security_parameter = 40;

        // computation taken from here:
        // https://github.com/ladnir/cryptoTools/blob/85da63e335c3ad3019af3958b48d3ff6750c3d92/cryptoTools/Common/CuckooIndex.cpp#L129-L150

        let log_number_inputs = (number_inputs as f64).log2().ceil();
        let a_max = 123.5;
        let b_max = -130.0;
        let a_sd = 2.3;
        let b_sd = 2.18;
        let a_mean = 6.3;
        let b_mean = 6.45;
        let a = a_max / 2.0 * (1.0 + erf((log_number_inputs - a_mean) / (a_sd * SQRT_2)));
        let b = b_max / 2.0 * (1.0 + erf((log_number_inputs - b_mean) / (b_sd * SQRT_2)))
            - log_number_inputs;
        let e = (statistical_security_parameter as f64 - b) / a + 0.3;

        (e * number_inputs as f64).ceil() as usize
    }

    /// Return the number of buckets
    pub fn get_number_buckets(&self) -> usize {
        self.number_buckets
    }

    /// Return the number of inputs these parameters are specified for
    pub fn get_number_inputs(&self) -> usize {
        self.number_inputs
    }
}

pub struct Hasher<H: HashFunction> {
    parameters: Parameters<H>,
    hash_functions: [H; 3],
}

impl<H: HashFunction> Hasher<H> {
    const UNOCCUPIED: u64 = u64::MAX;

    /// Create `Hasher` object with given parameters
    pub fn new(parameters: Parameters<H>) -> Self {
        let hash_functions =
            array::from_fn(|i| H::from_description(parameters.hash_function_descriptions[i]));
        Hasher::<H> {
            parameters,
            hash_functions,
        }
    }

    /// Return the parameters
    pub fn get_parameters(&self) -> &Parameters<H> {
        &self.parameters
    }

    /// Hash a single item with the given hash function
    pub fn hash_single(&self, hash_function_index: usize, item: u64) -> u64 {
        assert!(hash_function_index < NUMBER_HASH_FUNCTIONS);
        self.hash_functions[hash_function_index].hash_single(item)
    }

    /// Hash the whole domain [0, domain_size) with all three hash functions
    pub fn hash_domain(&self, domain_size: u64) -> [Vec<u64>; NUMBER_HASH_FUNCTIONS] {
        array::from_fn(|i| self.hash_functions[i].hash_range(0..domain_size))
    }

    /// Hash the given items with all three hash functions
    pub fn hash_items(&self, items: &[u64]) -> [Vec<u64>; NUMBER_HASH_FUNCTIONS] {
        array::from_fn(|i| self.hash_functions[i].hash_slice(items))
    }

    /// Hash the whole domain [0, domain_size) into buckets with all three hash functions
    pub fn hash_domain_into_buckets(&self, domain_size: u64) -> Vec<Vec<u64>> {
        let mut hash_table = vec![Vec::new(); self.parameters.number_buckets as usize];
        let hashes = self.hash_domain(domain_size);
        for x in 0..domain_size {
            for hash_function_index in 0..NUMBER_HASH_FUNCTIONS {
                let h = hashes[hash_function_index][x as usize];
                hash_table[h as usize].push(x);
            }
        }
        hash_table
    }

    /// Hash the given items into buckets all three hash functions
    pub fn hash_items_into_buckets(&self, items: &[u64]) -> Vec<Vec<u64>> {
        let mut hash_table = vec![Vec::new(); self.parameters.number_buckets as usize];
        let hashes = self.hash_items(items);
        for (i, &x) in items.iter().enumerate() {
            for hash_function_index in 0..NUMBER_HASH_FUNCTIONS {
                let h = hashes[hash_function_index][i as usize];
                hash_table[h as usize].push(x);
            }
        }
        hash_table
    }

    /// Perform cuckoo hashing to place the given items into a vector
    /// NB: number of items must match the number of items used to generate the parameters
    pub fn cuckoo_hash_items(&self, items: &[u64]) -> (Vec<u64>, Vec<usize>) {
        let number_inputs = self.parameters.number_inputs;
        let number_buckets = self.parameters.number_buckets;

        assert_eq!(
            items.len(),
            number_inputs,
            "#items must match number inputs specified in the parameters"
        );

        // create cuckoo hash table to store all inputs
        // we use u64::MAX to denote an empty entry
        let mut cuckoo_table = vec![Self::UNOCCUPIED; self.parameters.number_buckets];
        // store the indices of the items mapped into each bucket
        let mut cuckoo_table_indices = vec![0usize; number_buckets];

        let hashes = self.hash_items(items);

        // keep track of which hash function we need to use next for an item
        let mut next_hash_function = vec![0usize; number_buckets];

        // if we need more than this number of steps to insert an item, we have found
        // a cycle (this should only happen with negligible probability if the
        // parameters are chosen correctly)
        // const auto max_number_tries = NUMBER_HASH_FUNCTIONS * number_inputs_;
        let max_number_tries = number_inputs + 1;

        for input_j in 0..number_inputs {
            let mut index = input_j;
            let mut item = items[index];
            let mut try_k = 0;
            while try_k < max_number_tries {
                // try to (re)insert item with current index
                let hash: usize = hashes[next_hash_function[index]][index].try_into().unwrap();
                // increment hash function counter for this item s.t. we use the next hash
                // function next time
                next_hash_function[index] = (next_hash_function[index] + 1) % NUMBER_HASH_FUNCTIONS;
                if cuckoo_table[hash] == Self::UNOCCUPIED {
                    // the bucket was free, so we can insert the item
                    cuckoo_table[hash] = item;
                    cuckoo_table_indices[hash] = index;
                    break;
                }
                // the bucket was occupied, so we evict the item in the table and insert
                // it with the next hash function
                (cuckoo_table[hash], item) = (item, cuckoo_table[hash]);
                (cuckoo_table_indices[hash], index) = (index, cuckoo_table_indices[hash]);
                try_k += 1;
            }
            if try_k >= max_number_tries {
                assert!(false, "cycle detected"); // TODO: error handling
                                                  // return absl::InvalidArgumentError("Cuckoo::HashCuckoo -- Cycle detected");
            }
        }

        (cuckoo_table, cuckoo_table_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::AesHashFunction;
    use rand::{seq::SliceRandom, thread_rng, Rng};

    fn gen_random_numbers(n: usize) -> Vec<u64> {
        (0..n).map(|_| thread_rng().gen()).collect()
    }

    fn create_hasher<H: HashFunction>(number_inputs: usize) -> Hasher<H> {
        let params = Parameters::<H>::sample(number_inputs);
        Hasher::<H>::new(params)
    }

    fn test_hash_cuckoo_with_param<H: HashFunction>(log_number_inputs: usize) {
        let number_inputs = 1 << log_number_inputs;
        let inputs = gen_random_numbers(number_inputs);
        let cuckoo = create_hasher::<H>(number_inputs);
        let (cuckoo_table_items, cuckoo_table_indices) = cuckoo.cuckoo_hash_items(&inputs);

        let number_buckets = cuckoo.get_parameters().get_number_buckets();
        // check dimensions
        assert_eq!(cuckoo_table_items.len(), number_buckets);
        assert_eq!(cuckoo_table_indices.len(), number_buckets);
        // check that we have the right number of things in the table
        let num_unoccupied_entries = cuckoo_table_items
            .iter()
            .copied()
            .filter(|&x| x == Hasher::<H>::UNOCCUPIED)
            .count();
        assert_eq!(number_buckets - num_unoccupied_entries, number_inputs);
        // keep track of which items we have seen in the cuckoo table
        let mut found_inputs_in_table = vec![false; number_inputs];
        for bucket_i in 0..number_buckets {
            if cuckoo_table_items[bucket_i] != Hasher::<H>::UNOCCUPIED {
                let index = cuckoo_table_indices[bucket_i];
                // check that the right item is here
                assert_eq!(cuckoo_table_items[bucket_i], inputs[index]);
                // check that we have not yet seen this item
                assert!(!found_inputs_in_table[index]);
                // remember that we have seen this item
                found_inputs_in_table[index] = true;
            }
        }
        // check that we have found all inputs in the cuckoo table
        assert!(found_inputs_in_table.iter().all(|&x| x));
    }

    fn test_hash_domain_into_buckets_with_param<H: HashFunction>(log_number_inputs: usize) {
        let number_inputs = 1 << log_number_inputs;
        let cuckoo = create_hasher::<H>(number_inputs);
        let domain_size = 1 << 10;
        let number_buckets = cuckoo.get_parameters().get_number_buckets();

        let hash_table = cuckoo.hash_domain_into_buckets(domain_size);
        assert_eq!(hash_table.len(), number_buckets);
        for bucket in &hash_table {
            // Check that items inside each bucket are sorted
            // assert!(bucket.iter().is_sorted());  // `is_sorted` is currently nightly only
            assert!(bucket.windows(2).all(|w| w[0] <= w[1]))
        }

        // Check that hashing is deterministic
        let hash_table2 = cuckoo.hash_domain_into_buckets(domain_size);
        assert_eq!(hash_table, hash_table2);

        let hashes = cuckoo.hash_domain(domain_size);

        for element in 0..domain_size {
            if hashes[0][element as usize] == hashes[1][element as usize]
                && hashes[0][element as usize] == hashes[2][element as usize]
            {
                let hash = hashes[0][element as usize] as usize;
                let idx_start = hash_table[hash]
                    .as_slice()
                    .partition_point(|x| x < &element);
                let idx_end = hash_table[hash]
                    .as_slice()
                    .partition_point(|x| x <= &element);
                // check that the element is in the bucket
                assert_ne!(idx_start, hash_table[hash].len());
                assert_eq!(hash_table[hash][idx_start], element);
                // check that the element occurs three times
                assert_eq!(idx_end - idx_start, 3);
            } else if hashes[0][element as usize] == hashes[1][element as usize] {
                let hash = hashes[0][element as usize] as usize;
                let idx_start = hash_table[hash]
                    .as_slice()
                    .partition_point(|x| x < &element);
                let idx_end = hash_table[hash]
                    .as_slice()
                    .partition_point(|x| x <= &element);
                // check that the element is in the bucket
                assert_ne!(idx_start, hash_table[hash].len());
                assert_eq!(hash_table[hash][idx_start], element);
                // check that the element occurs two times
                assert_eq!(idx_end - idx_start, 2);

                let hash_other = hashes[2][element as usize] as usize;
                assert!(hash_table[hash_other]
                    .as_slice()
                    .binary_search(&element)
                    .is_ok());
            } else if hashes[0][element as usize] == hashes[2][element as usize] {
                let hash = hashes[0][element as usize] as usize;
                let idx_start = hash_table[hash]
                    .as_slice()
                    .partition_point(|x| x < &element);
                let idx_end = hash_table[hash]
                    .as_slice()
                    .partition_point(|x| x <= &element);
                // check that the element is in the bucket
                assert_ne!(idx_start, hash_table[hash].len());
                assert_eq!(hash_table[hash][idx_start], element);
                // check that the element occurs two times
                assert_eq!(idx_end - idx_start, 2);

                let hash_other = hashes[1][element as usize] as usize;
                assert!(hash_table[hash_other]
                    .as_slice()
                    .binary_search(&element)
                    .is_ok());
            } else if hashes[1][element as usize] == hashes[2][element as usize] {
                let hash = hashes[1][element as usize] as usize;
                let idx_start = hash_table[hash]
                    .as_slice()
                    .partition_point(|x| x < &element);
                let idx_end = hash_table[hash]
                    .as_slice()
                    .partition_point(|x| x <= &element);
                // check that the element is in the bucket
                assert_ne!(idx_start, hash_table[hash].len());
                assert_eq!(hash_table[hash][idx_start], element);
                // check that the element occurs two times
                assert_eq!(idx_end - idx_start, 2);

                let hash_other = hashes[0][element as usize] as usize;
                assert!(hash_table[hash_other]
                    .as_slice()
                    .binary_search(&element)
                    .is_ok());
            } else {
                for hash_j in 0..NUMBER_HASH_FUNCTIONS {
                    let hash = hashes[hash_j][element as usize] as usize;
                    assert!(hash_table[hash].as_slice().binary_search(&element).is_ok());
                }
            }
        }

        let num_items_in_hash_table: usize = hash_table.iter().map(|v| v.len()).sum();
        assert_eq!(num_items_in_hash_table as u64, 3 * domain_size);
    }

    fn test_buckets_cuckoo_consistency_with_param<H: HashFunction>(number_inputs: usize) {
        let domain_size = 1 << 10;
        let cuckoo = create_hasher::<H>(number_inputs);

        // To generate random numbers in the domain, we generate the entire domain and do a random shuffle

        let shuffled_domain = {
            let mut vec: Vec<u64> = (0..domain_size).collect();
            vec.shuffle(&mut thread_rng());
            vec
        };

        // Checking that every element in a cuckoo bucket exists in the corresponding bucket of HashSimpleDomain
        let hash_table = cuckoo.hash_domain_into_buckets(domain_size);
        let (cuckoo_table_items, _) = cuckoo.cuckoo_hash_items(&shuffled_domain[..number_inputs]);
        let number_buckets = cuckoo.get_parameters().get_number_buckets();

        for bucket_i in 0..number_buckets {
            if cuckoo_table_items[bucket_i] != Hasher::<H>::UNOCCUPIED {
                assert!(hash_table[bucket_i]
                    .as_slice()
                    .binary_search(&cuckoo_table_items[bucket_i])
                    .is_ok());
            }
        }
    }

    #[test]
    fn test_hash_cuckoo() {
        for n in 5..10 {
            test_hash_cuckoo_with_param::<AesHashFunction>(n);
        }
    }

    #[test]
    fn test_hash_domain_into_buckets() {
        for n in 5..10 {
            test_hash_domain_into_buckets_with_param::<AesHashFunction>(n);
        }
    }

    #[test]
    fn test_buckets_cuckoo_consistency() {
        for n in 5..10 {
            test_buckets_cuckoo_consistency_with_param::<AesHashFunction>(n);
        }
    }
}
