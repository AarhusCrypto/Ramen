use core::fmt::Debug;
use core::marker::PhantomData;
use core::ops::Add;
use num::traits::Zero;

pub trait MultiPointDpfKey {
    fn get_party_id(&self) -> usize;
    fn get_log_domain_size(&self) -> u64;
    fn get_number_points(&self) -> usize;
}

pub trait MultiPointDpf {
    type Key: Clone + MultiPointDpfKey;
    type Value: Add<Output = Self::Value> + Copy + Debug + Eq + Zero;

    fn generate_keys(
        log_domain_size: u64,
        alphas: &[u64],
        betas: &[Self::Value],
    ) -> (Self::Key, Self::Key);
    fn evaluate_at(key: &Self::Key, index: u64) -> Self::Value;
    fn evaluate_domain(key: &Self::Key) -> Vec<Self::Value> {
        (0..(1 << key.get_log_domain_size()))
            .map(|x| Self::evaluate_at(&key, x))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct DummyMpDpfKey<V> {
    party_id: usize,
    log_domain_size: u64,
    number_points: usize,
    alphas: Vec<u64>,
    betas: Vec<V>,
}

impl<V> MultiPointDpfKey for DummyMpDpfKey<V> {
    fn get_party_id(&self) -> usize {
        self.party_id
    }
    fn get_log_domain_size(&self) -> u64 {
        self.log_domain_size
    }
    fn get_number_points(&self) -> usize {
        self.number_points
    }
}

pub struct DummyMpDpf<V>
where
    V: Add<Output = V> + Copy + Debug + Eq + Zero,
{
    phantom: PhantomData<V>,
}

impl<V> MultiPointDpf for DummyMpDpf<V>
where
    V: Add<Output = V> + Copy + Debug + Eq + Zero,
{
    type Key = DummyMpDpfKey<V>;
    type Value = V;

    fn generate_keys(log_domain_size: u64, alphas: &[u64], betas: &[V]) -> (Self::Key, Self::Key) {
        assert_eq!(
            alphas.len(),
            betas.len(),
            "alphas and betas must be the same size"
        );
        assert!(
            alphas.iter().all(|alpha| alpha < &(1 << log_domain_size)),
            "all alphas must be in the domain"
        );
        assert!(alphas.windows(2).all(|w| w[0] <= w[1]));
        let number_points = alphas.len();
        (
            DummyMpDpfKey {
                party_id: 0,
                log_domain_size,
                number_points,
                alphas: alphas.iter().copied().collect(),
                betas: betas.iter().copied().collect(),
            },
            DummyMpDpfKey {
                party_id: 1,
                log_domain_size,
                number_points,
                alphas: alphas.iter().copied().collect(),
                betas: betas.iter().copied().collect(),
            },
        )
    }

    fn evaluate_at(key: &Self::Key, index: u64) -> V {
        if key.get_party_id() == 0 {
            match key.alphas.binary_search(&index) {
                Ok(i) => key.betas[i],
                Err(_) => V::zero(),
            }
        } else {
            V::zero()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::{Distribution, Standard};
    use rand::{thread_rng, Rng};

    fn test_mpdpf_with_param<MPDPF: MultiPointDpf>(log_domain_size: u64, number_points: usize)
    where
        Standard: Distribution<MPDPF::Value>,
    {
        assert!(number_points <= (1 << log_domain_size));
        let domain_size = 1 << log_domain_size;
        let alphas = {
            let mut alphas: Vec<u64> = (0..number_points)
                .map(|_| thread_rng().gen_range(0..domain_size))
                .collect();
            alphas.sort();
            alphas
        };
        let betas: Vec<MPDPF::Value> = (0..number_points).map(|_| thread_rng().gen()).collect();
        let (key_0, key_1) = MPDPF::generate_keys(log_domain_size, &alphas, &betas);

        let out_0 = MPDPF::evaluate_domain(&key_0);
        let out_1 = MPDPF::evaluate_domain(&key_1);
        for i in 0..domain_size {
            let value = MPDPF::evaluate_at(&key_0, i) + MPDPF::evaluate_at(&key_1, i);
            assert_eq!(value, out_0[i as usize] + out_1[i as usize]);
            let expected_result = match alphas.binary_search(&i) {
                Ok(i) => betas[i],
                Err(_) => MPDPF::Value::zero(),
            };
            assert_eq!(value, expected_result);
        }
    }

    #[test]
    fn test_mpdpf() {
        for log_domain_size in 5..10 {
            for log_number_points in 0..5 {
                test_mpdpf_with_param::<DummyMpDpf<u64>>(log_domain_size, 1 << log_number_points);
            }
        }
    }
}
