// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: tests of the RetroInfer CPU oracle: build invariants and
//! determinism, the three synthetic attention patterns of L33
//! (Gaussian clusters recalled near 1, uniform keys failing on purpose, a
//! sink-dominated pattern), and the degenerate cases where the merged formula
//! must equal dense softmax exactly.

use super::config::RetroParams;
use super::oracle::{HeadKv, KmeansSpec, OracleLayer, build_head, init_positions};
use super::oracle_eval::{
    Zones, dense_attention, measure_head, retro_attention, score_clusters, select,
};
use super::state::IndexCounts;

/// Deterministic test numbers (xorshift64*).
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }
    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    pub(crate) fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    pub(crate) fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    pub(crate) fn unit(&mut self, d: usize) -> Vec<f32> {
        let v: Vec<f32> = (0..d).map(|_| self.normal()).collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / n).collect()
    }
}

/// Keys drawn around `clusters` random directions (`scale` along the
/// direction, `noise` per dim), in blocks of `run` consecutive tokens per
/// cluster; values random. Returns (k, v, the direction of each cluster).
pub(crate) fn clustered(
    n: usize,
    d: usize,
    clusters: usize,
    run: usize,
    scale: f32,
    noise: f32,
    seed: u64,
) -> (Vec<f32>, Vec<f32>, Vec<Vec<f32>>) {
    let mut r = Rng::new(seed);
    let dirs: Vec<Vec<f32>> = (0..clusters).map(|_| r.unit(d)).collect();
    let mut k = Vec::with_capacity(n * d);
    let mut v = Vec::with_capacity(n * d);
    for t in 0..n {
        let c = (t / run) % clusters;
        for &x in &dirs[c] {
            k.push(scale * x + noise * r.normal());
        }
        for _ in 0..d {
            v.push(r.normal());
        }
    }
    (k, v, dirs)
}

fn params() -> RetroParams {
    RetroParams {
        segment_tokens: 1024,
        min_context: 1024,
        update_interval: 256,
        ..RetroParams::default()
    }
}

fn layer_counts(
    p: &RetroParams,
    kv: HeadKv<'_>,
    end: u32,
) -> (super::oracle::HeadIndex, IndexCounts) {
    let l = OracleLayer::build(&[kv], p, end);
    (l.heads.into_iter().next().unwrap(), l.counts)
}

#[test]
fn uniform_stride_init_matches_upstream() {
    // torch.arange(4) * 2.5 + 1.25 -> [1.25, 3.75, 6.25, 8.75] -> int64.
    assert_eq!(init_positions(10, 4), vec![1, 3, 6, 8]);
    assert_eq!(init_positions(8, 8), (0..8).collect::<Vec<_>>());
}

#[test]
fn build_partitions_the_range_and_sums_values() {
    let d = 16;
    let (k, v, _) = clustered(700, d, 12, 9, 3.0, 0.3, 1);
    let kv = HeadKv {
        k: &k,
        v: &v,
        head_dim: d,
    };
    let spec = KmeansSpec {
        centroids: 40,
        segments: 2,
        iters: 10,
    };
    let idx = build_head(kv, 30, 690, spec);
    assert_eq!(idx.n_centroids(), 40);
    assert_eq!(
        idx.sizes.iter().sum::<u32>(),
        660,
        "every token assigned once"
    );
    let mut seen: Vec<u32> = idx.lists.iter().flatten().copied().collect();
    seen.sort_unstable();
    assert_eq!(seen, (30..690).collect::<Vec<u32>>());
    for (c, list) in idx.lists.iter().enumerate() {
        assert_eq!(list.len() as u32, idx.sizes[c]);
        assert!(list.windows(2).all(|w| w[0] < w[1]), "members ascending");
        if list.is_empty() {
            continue;
        }
        for dd in 0..d {
            let vs: f64 = list
                .iter()
                .map(|&p| f64::from(v[p as usize * d + dd]))
                .sum();
            assert!(
                (f64::from(idx.value_sum[c * d + dd]) - vs).abs() < 1e-3,
                "value sum"
            );
            let km: f64 = list
                .iter()
                .map(|&p| f64::from(k[p as usize * d + dd]))
                .sum::<f64>()
                / list.len() as f64;
            assert!(
                (f64::from(idx.centroids[c * d + dd]) - km).abs() < 1e-4,
                "centroid is the member mean"
            );
        }
    }
    assert_eq!(build_head(kv, 30, 690, spec), idx, "deterministic");
}

#[test]
fn gaussian_clusters_are_recalled() {
    let d = 32;
    let n = 2048u32;
    let (k, v, dirs) = clustered(n as usize, d, 128, 16, 4.0, 0.1, 7);
    let kv = HeadKv {
        k: &k,
        v: &v,
        head_dim: d,
    };
    let p = params();
    let (idx, c) = layer_counts(&p, kv, n - p.window);
    let scale = 1.0 / (d as f32).sqrt();
    let q: Vec<f32> = dirs[7].iter().map(|x| 16.0 * x).collect();
    let z = select(
        &score_clusters(&idx, c.n_centroids as usize, &q, 1, scale),
        c.nprobe as usize,
        c.es as usize,
    );
    let m = measure_head(kv, &idx, &c, &z, &q, n, scale, None);
    assert!(m.mass_total >= 0.97, "{m:?}");
    assert!(m.mass_exact >= 0.9, "{m:?}");
    assert!(m.out_rel_err <= 0.05, "{m:?}");
}

#[test]
fn uniform_keys_fail_on_purpose() {
    let d = 32;
    let n = 2048u32;
    let mut r = Rng::new(3);
    let k: Vec<f32> = (0..n as usize * d).map(|_| r.normal()).collect();
    let v: Vec<f32> = (0..n as usize * d).map(|_| r.normal()).collect();
    let kv = HeadKv {
        k: &k,
        v: &v,
        head_dim: d,
    };
    let p = params();
    let (idx, c) = layer_counts(&p, kv, n - p.window);
    let scale = 1.0 / (d as f32).sqrt();
    let q: Vec<f32> = (0..d).map(|_| r.normal()).collect();
    let z = select(
        &score_clusters(&idx, c.n_centroids as usize, &q, 1, scale),
        c.nprobe as usize,
        c.es as usize,
    );
    let m = measure_head(kv, &idx, &c, &z, &q, n, scale, None);
    // No structure to retrieve: the exact zone holds about its token share.
    assert!(m.mass_exact < 0.3, "{m:?}");
    assert!(m.jaccard < 0.5, "{m:?}");
}

#[test]
fn sink_dominated_attention_is_kept_exact() {
    let d = 32;
    let n = 2048u32;
    let (mut k, v, _) = clustered(n as usize, d, 64, 16, 1.0, 0.5, 11);
    let mut r = Rng::new(5);
    let sink_dir = r.unit(d);
    for t in 0..4 {
        for (j, x) in sink_dir.iter().enumerate() {
            k[t * d + j] = 6.0 * x;
        }
    }
    let kv = HeadKv {
        k: &k,
        v: &v,
        head_dim: d,
    };
    let p = params();
    let (idx, c) = layer_counts(&p, kv, n - p.window);
    let scale = 1.0 / (d as f32).sqrt();
    let q: Vec<f32> = sink_dir.iter().map(|x| 12.0 * x).collect();
    let z = select(
        &score_clusters(&idx, c.n_centroids as usize, &q, 1, scale),
        c.nprobe as usize,
        c.es as usize,
    );
    let m = measure_head(kv, &idx, &c, &z, &q, n, scale, None);
    assert!(m.mass_exact >= 0.95, "{m:?}");
    assert!(m.out_rel_err <= 0.05, "{m:?}");
}

#[test]
fn one_token_clusters_estimate_dense_exactly() {
    let d = 16;
    let n = 96usize;
    let mut r = Rng::new(9);
    let k: Vec<f32> = (0..n * d).map(|_| r.normal()).collect();
    let v: Vec<f32> = (0..n * d).map(|_| r.normal()).collect();
    let kv = HeadKv {
        k: &k,
        v: &v,
        head_dim: d,
    };
    let idx = build_head(
        kv,
        0,
        n,
        KmeansSpec {
            centroids: n,
            segments: 1,
            iters: 10,
        },
    );
    assert!(
        idx.sizes.iter().all(|&s| s == 1),
        "each key is its own cluster"
    );
    let q: Vec<f32> = (0..d).map(|_| r.normal()).collect();
    let scale = 0.25;
    let all: Vec<u32> = (0..n as u32).collect();
    let (retro, _) = retro_attention(kv, &idx, &[], &all, &q, scale);
    let (dense, _) = dense_attention(kv, n, &q, scale);
    for (a, b) in retro.iter().zip(&dense) {
        assert!((a - b).abs() <= 1e-5 * b.abs().max(1.0), "{a} vs {b}");
    }
}

#[test]
fn retrieving_every_cluster_is_dense() {
    let d = 16;
    let n = 1024u32;
    let (k, v, dirs) = clustered(n as usize, d, 32, 8, 2.0, 0.5, 13);
    let kv = HeadKv {
        k: &k,
        v: &v,
        head_dim: d,
    };
    let p = params();
    let (idx, c) = layer_counts(&p, kv, n - p.window);
    let z = Zones {
        retrieval: (0..c.n_centroids)
            .filter(|&i| idx.sizes[i as usize] > 0)
            .collect(),
        estimation: vec![],
    };
    let q: Vec<f32> = dirs[3].iter().map(|x| 3.0 * x).collect();
    let m = measure_head(kv, &idx, &c, &z, &q, n, 0.25, None);
    assert!(m.out_rel_err < 1e-9, "{m:?}");
    assert!((m.mass_exact - 1.0).abs() < 1e-9, "{m:?}");
    assert!((m.jaccard - 1.0).abs() < 1e-12);
}

#[test]
fn group_scores_sum_softmax_over_query_heads() {
    let d = 8;
    let (k, v, dirs) = clustered(512, d, 16, 32, 3.0, 0.1, 17);
    let kv = HeadKv {
        k: &k,
        v: &v,
        head_dim: d,
    };
    let idx = build_head(
        kv,
        0,
        512,
        KmeansSpec {
            centroids: 16,
            segments: 1,
            iters: 10,
        },
    );
    let mut q = Vec::new();
    q.extend(dirs[2].iter().map(|x| 4.0 * x));
    q.extend(dirs[9].iter().map(|x| 4.0 * x));
    let s = score_clusters(&idx, 16, &q, 2, 0.5);
    let live: f64 = s.iter().filter(|x| x.is_finite()).sum();
    assert!(
        (live - 2.0).abs() < 1e-9,
        "two heads, each a softmax summing to 1: {live}"
    );
    let z = select(&s, 2, 3);
    assert_eq!(z.retrieval.len(), 2);
    assert!(z.estimation.iter().all(|c| !z.retrieval.contains(c)));
    for w in z.retrieval.windows(2) {
        assert!(s[w[0] as usize] >= s[w[1] as usize], "descending");
    }
}
