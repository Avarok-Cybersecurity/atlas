// SPDX-License-Identifier: AGPL-3.0-only
//! Tests of the Mersenne Twister generators against values every NumPy and CPython user knows.

use super::mt::{Mt19937, NumpyRandom, PyRandom};

/// CPython's `random()`: two draws, 27 and 26 bits.
fn res53(g: &mut Mt19937) -> f64 {
    let a = f64::from(g.next_u32() >> 5);
    let b = f64::from(g.next_u32() >> 6);
    (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
}

#[test]
fn mt19937_reference_output_for_seed_5489() {
    let mut g = Mt19937::from_seed(5489);
    assert_eq!(g.next_u32(), 3_499_211_612);
    assert_eq!(g.next_u32(), 581_869_302);
}

#[test]
fn init_by_array_matches_cpython_random_seed_0_and_42() {
    // random.seed(0); random.random() == 0.8444218515250481; random.seed(42) gives 0.6394267984578837
    assert_eq!(res53(&mut Mt19937::from_key(&[0])), 0.844_421_851_525_048_1);
    assert_eq!(
        res53(&mut Mt19937::from_key(&[42])),
        0.639_426_798_457_883_7
    );
}

#[test]
fn numpy_legacy_rand_and_randn() {
    // np.random.RandomState(0).rand() == 0.5488135039273248, RandomState(42).rand() == 0.3745401188473625
    assert_eq!(NumpyRandom::new(0).rand(), 0.548_813_503_927_324_8);
    assert_eq!(NumpyRandom::new(42).rand(), 0.374_540_118_847_362_5);
    // np.random.RandomState(0).randn(2) == [1.764052345967664, 0.4001572083672233]
    let mut r = NumpyRandom::new(0);
    assert!((r.randn() - 1.764_052_345_967_664).abs() < 1e-12);
    assert!((r.randn() - 0.400_157_208_367_223_3).abs() < 1e-12);
}

#[test]
fn numpy_randint_shuffle_and_choice_stay_in_range_and_permute() {
    let mut r = NumpyRandom::new(7);
    for _ in 0..200 {
        let v = r.randint(3, 20);
        assert!((3..20).contains(&v));
    }
    assert_eq!(r.randint(5, 6), 5);
    let mut v: Vec<usize> = (0..50).collect();
    r.shuffle(&mut v);
    let mut s = v.clone();
    s.sort_unstable();
    assert_eq!(s, (0..50).collect::<Vec<_>>());
    assert_ne!(v, (0..50).collect::<Vec<_>>());
    assert!(r.choice_indices(3, 100).iter().all(|&i| i < 3));
    // same seed, same stream
    assert_eq!(
        NumpyRandom::new(9).randn_vec(5),
        NumpyRandom::new(9).randn_vec(5)
    );
}

#[test]
fn python_random_is_deterministic_and_samples_without_repeats() {
    let pool: Vec<usize> = (0..100).collect();
    let a = PyRandom::from_str_seed("20261112:gsm8k").sample(&pool, 30);
    let b = PyRandom::from_str_seed("20261112:gsm8k").sample(&pool, 30);
    let c = PyRandom::from_str_seed("20261112:mmlu").sample(&pool, 30);
    assert_eq!(a, b);
    assert_ne!(a, c);
    let mut u = a.clone();
    u.sort_unstable();
    u.dedup();
    assert_eq!(u.len(), 30);
    // the small-population branch (n <= setsize) and the set branch both return k distinct items
    let small = PyRandom::from_str_seed("x").sample(&(0..10).collect::<Vec<_>>(), 4);
    assert_eq!(small.len(), 4);
    let mut s = PyRandom::from_str_seed("y");
    let mut v: Vec<usize> = (0..20).collect();
    s.shuffle(&mut v);
    v.sort_unstable();
    assert_eq!(v, (0..20).collect::<Vec<_>>());
    assert!((0..1000).all(|_| s.randbelow(7) < 7));
}
