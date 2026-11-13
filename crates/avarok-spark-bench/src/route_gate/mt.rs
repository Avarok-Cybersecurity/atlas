// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the Mersenne Twister and the two seeded generators of the old harness, reproduced bit for bit so a
//! fixed seed gives the same folds, resamples and prompt selection as the Python study did: NumPy's legacy
//! `RandomState` (`rand`, `randn`, `randint`, `shuffle`, `choice`) and CPython's `random.Random` (string seeds,
//! `getrandbits`, `shuffle`, `sample`). Standard MT19937 only; no code copied.

use sha2::{Digest, Sha512};

const N: usize = 624;
const M: usize = 397;

/// MT19937 core (32-bit output).
#[derive(Clone)]
pub struct Mt19937 {
    mt: [u32; N],
    idx: usize,
}

impl Mt19937 {
    /// `init_genrand(seed)`: NumPy's `RandomState(int)` seeding.
    pub fn from_seed(seed: u32) -> Self {
        let mut mt = [0u32; N];
        mt[0] = seed;
        for i in 1..N {
            mt[i] = 1_812_433_253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { mt, idx: N }
    }

    /// `init_by_array(key)`: CPython's seeding (after it turns the seed into 32-bit words).
    pub fn from_key(key: &[u32]) -> Self {
        let mut g = Self::from_seed(19_650_218);
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            let prev = g.mt[i - 1] ^ (g.mt[i - 1] >> 30);
            g.mt[i] = (g.mt[i] ^ prev.wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                g.mt[0] = g.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let prev = g.mt[i - 1] ^ (g.mt[i - 1] >> 30);
            g.mt[i] = (g.mt[i] ^ prev.wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                g.mt[0] = g.mt[N - 1];
                i = 1;
            }
        }
        g.mt[0] = 0x8000_0000;
        g.idx = N;
        g
    }

    fn refill(&mut self) {
        for k in 0..N {
            let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % N] & 0x7fff_ffff);
            let mut v = self.mt[(k + M) % N] ^ (y >> 1);
            if y & 1 != 0 {
                v ^= 0x9908_b0df;
            }
            self.mt[k] = v;
        }
        self.idx = 0;
    }

    /// Next 32 random bits.
    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= N {
            self.refill();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
}

/// Smallest `2^k - 1` that is `>= v`.
fn gen_mask(v: u64) -> u64 {
    let mut m = v;
    m |= m >> 1;
    m |= m >> 2;
    m |= m >> 4;
    m |= m >> 8;
    m |= m >> 16;
    m |= m >> 32;
    m
}

/// NumPy's legacy `RandomState`.
pub struct NumpyRandom {
    mt: Mt19937,
    cached_gauss: Option<f64>,
}

impl NumpyRandom {
    /// `RandomState(seed)`; the seed is reduced to 32 bits like NumPy does for an int.
    pub fn new(seed: u64) -> Self {
        Self {
            mt: Mt19937::from_seed((seed & 0xffff_ffff) as u32),
            cached_gauss: None,
        }
    }

    /// `rand()`: a double in [0, 1) from 53 bits.
    pub fn rand(&mut self) -> f64 {
        let a = (self.mt.next_u32() >> 5) as f64;
        let b = (self.mt.next_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }

    /// `randn()`: the legacy polar Box-Muller with the second value cached.
    pub fn randn(&mut self) -> f64 {
        if let Some(g) = self.cached_gauss.take() {
            return g;
        }
        loop {
            let x1 = 2.0 * self.rand() - 1.0;
            let x2 = 2.0 * self.rand() - 1.0;
            let r2 = x1 * x1 + x2 * x2;
            if r2 < 1.0 && r2 != 0.0 {
                let f = (-2.0 * r2.ln() / r2).sqrt();
                self.cached_gauss = Some(f * x1);
                return f * x2;
            }
        }
    }

    /// `randn(n)` as a vector.
    pub fn randn_vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.randn()).collect()
    }

    /// `rand(n)` as a vector.
    pub fn rand_vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.rand()).collect()
    }

    /// `randint(low, high)`: an integer in `[low, high)` by masked rejection on 32-bit draws (no draw when the range
    /// holds one value).
    pub fn randint(&mut self, low: i64, high: i64) -> i64 {
        let rng = (high - 1 - low) as u64;
        if rng == 0 {
            return low;
        }
        let mask = gen_mask(rng);
        loop {
            let v = u64::from(self.mt.next_u32()) & mask;
            if v <= rng {
                return low + v as i64;
            }
        }
    }

    /// `choice(a, size, replace=True)`: `size` indices into a population of `pop` items.
    pub fn choice_indices(&mut self, pop: usize, size: usize) -> Vec<usize> {
        (0..size)
            .map(|_| self.randint(0, pop as i64) as usize)
            .collect()
    }

    /// In-place `shuffle` of a 1-d array (Fisher-Yates from the end, `random_interval` draws).
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let max = i as u64;
            let mask = gen_mask(max);
            let j = loop {
                let v = u64::from(self.mt.next_u32()) & mask;
                if v <= max {
                    break v as usize;
                }
            };
            x.swap(i, j);
        }
    }
}

/// CPython's `random.Random`.
pub struct PyRandom {
    mt: Mt19937,
}

impl PyRandom {
    /// `Random(text)`: version-2 string seeding (`int.from_bytes(text + sha512(text))`, then `init_by_array`).
    pub fn from_str_seed(text: &str) -> Self {
        let mut bytes = text.as_bytes().to_vec();
        bytes.extend_from_slice(&Sha512::digest(text.as_bytes()));
        Self {
            mt: Mt19937::from_key(&key_words(&bytes)),
        }
    }

    /// `getrandbits(k)` for `1 <= k <= 32`.
    pub fn getrandbits(&mut self, k: u32) -> u32 {
        self.mt.next_u32() >> (32 - k)
    }

    /// `_randbelow(n)`: rejection on `bit_length(n)` bits.
    pub fn randbelow(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        let k = usize::BITS - n.leading_zeros();
        loop {
            let r = self.getrandbits(k) as usize;
            if r < n {
                return r;
            }
        }
    }

    /// `shuffle(x)`.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i + 1);
            x.swap(i, j);
        }
    }

    /// `sample(population, k)` (the list-copy branch for small populations, the selected-set branch otherwise).
    pub fn sample<T: Clone>(&mut self, population: &[T], k: usize) -> Vec<T> {
        let n = population.len();
        assert!(k <= n, "sample larger than population");
        let mut setsize = 21usize;
        if k > 5 {
            // 4 ** ceil(log(3k, 4)): the smallest power of four that is >= 3k.
            let mut table = 1usize;
            while table < k * 3 {
                table *= 4;
            }
            setsize += table;
        }
        let mut result = Vec::with_capacity(k);
        if n <= setsize {
            let mut pool: Vec<T> = population.to_vec();
            for i in 0..k {
                let j = self.randbelow(n - i);
                result.push(pool[j].clone());
                pool[j] = pool[n - i - 1].clone();
            }
        } else {
            let mut selected = std::collections::HashSet::new();
            for _ in 0..k {
                let mut j = self.randbelow(n);
                while selected.contains(&j) {
                    j = self.randbelow(n);
                }
                selected.insert(j);
                result.push(population[j].clone());
            }
        }
        result
    }
}

/// The little-endian 32-bit words of a big-endian byte string read as an integer (CPython's seed key).
fn key_words(be: &[u8]) -> Vec<u32> {
    let first = be.iter().position(|&b| b != 0);
    let Some(first) = first else {
        return vec![0];
    };
    let trimmed = &be[first..];
    let words = trimmed.len().div_ceil(4);
    let mut out = vec![0u32; words];
    for (i, &b) in trimmed.iter().rev().enumerate() {
        out[i / 4] |= u32::from(b) << (8 * (i % 4));
    }
    out
}
