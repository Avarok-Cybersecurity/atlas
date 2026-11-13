// SPDX-License-Identifier: AGPL-3.0-only
// Translated for Atlas from microsoft/RetrievalAttention (RetroInfer), MIT License,
// Copyright (c) Microsoft Corporation. Upstream: cache_hub/kmeans.py:1-260
// (_triton_assign_kernel, _triton_update_kernel, _triton_k_means_train, triton_reverse_index,
// triton_index_add, segment_k_means) and cache_hub/retroinfer_cache.py:540-556 (mean
// centering and the mean added back). Changes: a plain Rust CPU reference instead of Triton
// kernels; f64 accumulation and a fixed reduction order (upstream uses relaxed atomics);
// ties go to the lower centroid index; cluster member lists sorted ascending (upstream order
// follows atomics); scoped threads over KV heads; an append form for decode-time updates.

//! Responsibility: the CPU reference of the RetroInfer index build, the oracle
//! every GPU build is tested against and the index the shadow mode measures:
//! centering, segmented spherical k-means, value sums and member lists, per KV
//! head, plus the append that a decode-time update performs. Scoring, zone
//! selection and attention are in `oracle_eval.rs`.

use super::config::{self, RetroParams};
use super::state::IndexCounts;

/// One KV head's keys and values, `[n][head_dim]` f32, positions `0..n`.
#[derive(Clone, Copy)]
pub struct HeadKv<'a> {
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub head_dim: usize,
}

impl HeadKv<'_> {
    pub fn len(&self) -> usize {
        self.k.len() / self.head_dim.max(1)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn key(&self, pos: usize) -> &[f32] {
        &self.k[pos * self.head_dim..(pos + 1) * self.head_dim]
    }
    pub fn value(&self, pos: usize) -> &[f32] {
        &self.v[pos * self.head_dim..(pos + 1) * self.head_dim]
    }
}

/// One KV head's index.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HeadIndex {
    pub head_dim: usize,
    /// `[nc][head_dim]` plain means of the members (mean added back).
    pub centroids: Vec<f32>,
    /// `[nc][head_dim]` sums of the members' values.
    pub value_sum: Vec<f32>,
    /// `[nc]` member counts; 0 = empty.
    pub sizes: Vec<u32>,
    /// `[nc]` member positions, ascending.
    pub lists: Vec<Vec<u32>>,
}

impl HeadIndex {
    pub fn n_centroids(&self) -> usize {
        self.sizes.len()
    }
    pub fn centroid(&self, c: usize) -> &[f32] {
        &self.centroids[c * self.head_dim..(c + 1) * self.head_dim]
    }
    pub fn value_sum_of(&self, c: usize) -> &[f32] {
        &self.value_sum[c * self.head_dim..(c + 1) * self.head_dim]
    }
    /// Append `other`'s clusters after this index's (a decode-time update).
    pub fn append(&mut self, other: HeadIndex) {
        if self.head_dim == 0 {
            self.head_dim = other.head_dim;
        }
        self.centroids.extend(other.centroids);
        self.value_sum.extend(other.value_sum);
        self.sizes.extend(other.sizes);
        self.lists.extend(other.lists);
    }
}

/// How one build clusters its range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KmeansSpec {
    pub centroids: usize,
    pub segments: usize,
    pub iters: usize,
}

impl KmeansSpec {
    /// The prefill-time build of `tokens` keys (upstream `generate_config`).
    pub fn initial(p: &RetroParams, tokens: u32) -> Self {
        let segments = config::n_segments(tokens, p.segment_tokens);
        Self {
            centroids: config::n_centroids(tokens, p.avg_cluster, segments) as usize,
            segments: segments as usize,
            iters: p.kmeans_iters as usize,
        }
    }

    /// One decode-time update segment (upstream `_update_kv_cache`).
    pub fn update(p: &RetroParams) -> Self {
        Self {
            centroids: p.update_centroids() as usize,
            segments: 1,
            iters: p.kmeans_iters as usize,
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

/// Index of the centroid in `cents[lo..hi]` with the largest dot product with
/// `x`; ties go to the lower index (upstream keeps the first maximum).
fn argmax(x: &[f32], cents: &[f32], hd: usize, lo: usize, hi: usize) -> usize {
    let mut best = lo;
    let mut best_v = f64::NEG_INFINITY;
    for c in lo..hi {
        let v = dot(x, &cents[c * hd..(c + 1) * hd]);
        if v > best_v {
            best_v = v;
            best = c;
        }
    }
    best
}

/// One assign + update pass over `x[t_lo..t_hi)` and centroids `[c_lo, c_hi)`.
/// A centroid with no member keeps its previous value (upstream stores only
/// where the count is positive). Returns the assignment of the pass's tokens.
#[allow(clippy::too_many_arguments)]
fn pass(
    x: &[f32],
    cents: &mut [f32],
    hd: usize,
    t_lo: usize,
    t_hi: usize,
    c_lo: usize,
    c_hi: usize,
    normalize: bool,
) -> (Vec<usize>, Vec<u32>) {
    let nc = c_hi - c_lo;
    let mut sums = vec![0f64; nc * hd];
    let mut counts = vec![0u32; nc];
    let mut assign = Vec::with_capacity(t_hi - t_lo);
    for t in t_lo..t_hi {
        let xt = &x[t * hd..(t + 1) * hd];
        let c = argmax(xt, cents, hd, c_lo, c_hi);
        assign.push(c);
        let s = &mut sums[(c - c_lo) * hd..(c - c_lo + 1) * hd];
        for (a, &b) in s.iter_mut().zip(xt) {
            *a += f64::from(b);
        }
        counts[c - c_lo] += 1;
    }
    for (i, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let s = &sums[i * hd..(i + 1) * hd];
        let n = f64::from(count);
        let mut m: Vec<f64> = s.iter().map(|&v| v / n).collect();
        if normalize {
            let norm = m.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for v in &mut m {
                    *v /= norm;
                }
            }
        }
        for (d, &v) in cents[(c_lo + i) * hd..(c_lo + i + 1) * hd]
            .iter_mut()
            .zip(&m)
        {
            *d = v as f32;
        }
    }
    (assign, counts)
}

/// Uniform-stride initial centroid positions: `int(i * n/nc + n/nc/2)` computed
/// in f32 as upstream's float32 `torch.arange` does.
pub fn init_positions(n: usize, nc: usize) -> Vec<usize> {
    let step = (n as f64 / nc as f64) as f32;
    let half = (n as f64 / nc as f64 / 2.0) as f32;
    (0..nc)
        .map(|i| ((i as f32) * step + half) as usize)
        .map(|p| p.min(n.saturating_sub(1)))
        .collect()
}

/// Build one KV head's index over positions `[first, end)` of `kv`.
///
/// Upstream `segment_k_means` on mean-centered keys: centroids start at
/// uniform-stride keys; `iters - 1` passes train each segment's centroids on
/// that segment's tokens with L2-normalized updates (the trailing
/// `n - segments * (n / segments)` tokens sit out the training); one final
/// pass assigns every token among all centroids and stores plain means. The
/// centering mean is added back; value sums are of the raw values.
pub fn build_head(kv: HeadKv<'_>, first: usize, end: usize, spec: KmeansSpec) -> HeadIndex {
    let hd = kv.head_dim;
    let n = end.saturating_sub(first);
    let nc = spec.centroids.min(n).max(usize::from(n > 0));
    let mut out = HeadIndex {
        head_dim: hd,
        ..Default::default()
    };
    if n == 0 {
        return out;
    }
    let segs = spec.segments.max(1).min(nc);
    let mut mean = vec![0f64; hd];
    for t in first..end {
        for (m, &v) in mean.iter_mut().zip(kv.key(t)) {
            *m += f64::from(v);
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    let x: Vec<f32> = (first..end)
        .flat_map(|t| {
            kv.key(t)
                .iter()
                .zip(&mean)
                .map(|(&v, &m)| (f64::from(v) - m) as f32)
        })
        .collect();
    let mut cents: Vec<f32> = init_positions(n, nc)
        .iter()
        .flat_map(|&p| x[p * hd..(p + 1) * hd].iter().copied())
        .collect();
    // Upstream asserts nc % segments == 0; the sizes in config.rs guarantee it.
    let seg_tokens = n / segs;
    let seg_c = nc / segs;
    for _ in 0..spec.iters.saturating_sub(1) {
        for s in 0..segs {
            pass(
                &x,
                &mut cents,
                hd,
                s * seg_tokens,
                (s + 1) * seg_tokens,
                s * seg_c,
                (s + 1) * seg_c,
                true,
            );
        }
    }
    let (assign, counts) = pass(&x, &mut cents, hd, 0, n, 0, nc, false);
    for c in cents.chunks_mut(hd) {
        for (d, &mv) in c.iter_mut().zip(&mean) {
            *d = (f64::from(*d) + mv) as f32;
        }
    }
    let mut vsum = vec![0f64; nc * hd];
    let mut lists = vec![Vec::new(); nc];
    for (i, &c) in assign.iter().enumerate() {
        let pos = first + i;
        for (a, &b) in vsum[c * hd..(c + 1) * hd].iter_mut().zip(kv.value(pos)) {
            *a += f64::from(b);
        }
        lists[c].push(u32::try_from(pos).unwrap_or(u32::MAX));
    }
    out.centroids = cents;
    out.value_sum = vsum.iter().map(|&v| v as f32).collect();
    out.sizes = counts;
    out.lists = lists;
    out
}

/// Build every KV head over the same range, one scoped thread per head.
pub fn build_heads(
    heads: &[HeadKv<'_>],
    first: usize,
    end: usize,
    spec: KmeansSpec,
) -> Vec<HeadIndex> {
    std::thread::scope(|scope| {
        let jobs: Vec<_> = heads
            .iter()
            .map(|&h| scope.spawn(move || build_head(h, first, end, spec)))
            .collect();
        jobs.into_iter()
            .map(|j| j.join().expect("retro oracle build thread panicked"))
            .collect()
    })
}

/// A layer's oracle index across its KV heads, with the counts a device index
/// would carry, built and grown by the same policy as the device path
/// (`update.rs`).
#[derive(Clone, Debug, Default)]
pub struct OracleLayer {
    pub heads: Vec<HeadIndex>,
    pub counts: IndexCounts,
    /// Seconds spent building and updating it.
    pub build_secs: f64,
}

impl OracleLayer {
    /// The initial build over `[sink, indexed_end)`.
    pub fn build(heads: &[HeadKv<'_>], p: &RetroParams, indexed_end: u32) -> Self {
        let t0 = std::time::Instant::now();
        let sink = p.sink.min(indexed_end);
        let tokens = indexed_end - sink;
        let spec = KmeansSpec::initial(p, tokens);
        let idx = build_heads(heads, sink as usize, indexed_end as usize, spec);
        let nc = u32::try_from(spec.centroids.min(tokens as usize)).unwrap_or(u32::MAX);
        let np = config::nprobe(nc, p.retrieval_budget);
        Self {
            heads: idx,
            counts: IndexCounts {
                n_centroids: nc,
                nprobe: np,
                es: config::estimation_clusters(nc, p.estimation_budget, np),
                indexed_end,
                sink,
                listed: tokens,
            },
            build_secs: t0.elapsed().as_secs_f64(),
        }
    }

    /// Append one update segment `[indexed_end, indexed_end + tokens)`.
    pub fn update(&mut self, heads: &[HeadKv<'_>], p: &RetroParams, tokens: u32) {
        let t0 = std::time::Instant::now();
        let first = self.counts.indexed_end;
        let spec = KmeansSpec::update(p);
        let add = build_heads(heads, first as usize, (first + tokens) as usize, spec);
        for (h, a) in self.heads.iter_mut().zip(add) {
            h.append(a);
        }
        let uc = u32::try_from(spec.centroids.min(tokens as usize)).unwrap_or(0);
        self.counts = super::update::counts_after_update(&self.counts, p, uc, tokens);
        self.build_secs += t0.elapsed().as_secs_f64();
    }

    /// Grow the index the way the device path does until `len` tokens exist:
    /// every full update interval beyond the window is clustered.
    pub fn catch_up(&mut self, heads: &[HeadKv<'_>], p: &RetroParams, len: u32, block: u32) {
        while let Some(tokens) = super::update::due(&self.counts, len, p, block) {
            self.update(heads, p, tokens);
        }
    }
}
