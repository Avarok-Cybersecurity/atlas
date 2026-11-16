// SPDX-License-Identifier: AGPL-3.0-only
// Translated for Atlas from microsoft/RetrievalAttention (RetroInfer), MIT License,
// Copyright (c) Microsoft Corporation. Upstream: config/config.py (generate_config: segment
// count, centroid count rounded to a multiple of lcm(8, segments)) and
// cache_hub/retroinfer_cache.py:57-121 (update segment and update centroids, nprobe and
// estimation-zone sizes). Changes: Rust; Atlas flags with AVAROK_RETRO_* twins; Python's
// round-half-to-even kept exactly; capacities for fixed device buffers (upstream grows
// tensors with torch.cat); a parameter hash for the index-cache key.

//! Responsibility: the RetroInfer parameters (what the flags say) and every
//! size derived from them: segments, centroids, zone sizes, update sizes, and
//! the capacities of the fixed device buffers ([`RetroCfg`]), which the L34
//! kernels read.

use anyhow::{Result, ensure};

use crate::layers::qwen3_attention::sparse_seam::KvDtypeTag;

/// Fraction of clusters in the retrieval zone (exact attention over their tokens).
pub const DEFAULT_RETRIEVAL_BUDGET: f64 = 0.018;
/// Fraction of clusters in the estimation zone (attention over centroids).
pub const DEFAULT_ESTIMATION_BUDGET: f64 = 0.232;
/// Tokens per k-means segment at index build.
pub const DEFAULT_SEGMENT_TOKENS: u32 = 8192;
/// Average tokens per cluster (one centroid per this many keys).
pub const DEFAULT_AVG_CLUSTER: u32 = 16;
/// Attention-sink tokens kept exact at the start of the context.
pub const DEFAULT_SINK: u32 = 4;
/// Most recent tokens kept exact.
pub const DEFAULT_WINDOW: u32 = 64;
/// Below this context a sequence runs dense attention.
pub const DEFAULT_MIN_CONTEXT: u32 = 16384;
/// Generated tokens clustered into the index at a time.
pub const DEFAULT_UPDATE_INTERVAL: u32 = 1024;
/// Device memory for indexes, the index cache and build scratch, in MiB.
pub const DEFAULT_INDEX_CACHE_MB: u32 = 2048;
/// Shadow mode samples every N-th decode step of a layer.
pub const DEFAULT_SHADOW_EVERY: u32 = 64;
/// k-means iterations (upstream `segment_k_means(num_iters=10)`: nine training
/// passes per segment, then one global assignment pass).
pub const KMEANS_ITERS: u32 = 10;
/// Most rows of one sequence a verify step may carry (K+1 drafts) under retro.
pub const MAX_VERIFY_ROWS: u32 = 17;
/// Head dim the v1 kernels serve (Qwen3.5 full attention).
pub const HEAD_DIM: u32 = 256;
/// Largest retrieval list the kernels hold (the selection's in-CTA sort, L34a
/// `select::MAX_RETRIEVAL`; the gather's shared cluster table allows more).
/// A verify union beyond it is cut and counted; at default budgets it binds
/// only above about 400K tokens of context.
pub const MAX_RETRIEVAL_CAP: u32 = 1024;

/// The RetroInfer parameters after CLI, environment and defaults are merged.
#[derive(Clone, Debug, PartialEq)]
pub struct RetroParams {
    /// f64, as upstream's Python floats (the zone sizes round the same way).
    pub retrieval_budget: f64,
    pub estimation_budget: f64,
    pub segment_tokens: u32,
    pub avg_cluster: u32,
    pub sink: u32,
    pub window: u32,
    pub min_context: u32,
    pub update_interval: u32,
    pub kmeans_iters: u32,
    /// `--retro-shadow`: measure the zones against dense attention, output unchanged.
    pub shadow: bool,
    pub index_cache_mb: u32,
    /// `AVAROK_RETRO_SHADOW_EVERY`.
    pub shadow_every: u32,
    /// `AVAROK_RETRO_SHADOW_LAYERS` (attention-layer indexes); `None` = all.
    pub shadow_layers: Option<Vec<u32>>,
}

impl Default for RetroParams {
    fn default() -> Self {
        Self {
            retrieval_budget: DEFAULT_RETRIEVAL_BUDGET,
            estimation_budget: DEFAULT_ESTIMATION_BUDGET,
            segment_tokens: DEFAULT_SEGMENT_TOKENS,
            avg_cluster: DEFAULT_AVG_CLUSTER,
            sink: DEFAULT_SINK,
            window: DEFAULT_WINDOW,
            min_context: DEFAULT_MIN_CONTEXT,
            update_interval: DEFAULT_UPDATE_INTERVAL,
            kmeans_iters: KMEANS_ITERS,
            shadow: false,
            index_cache_mb: DEFAULT_INDEX_CACHE_MB,
            shadow_every: DEFAULT_SHADOW_EVERY,
            shadow_layers: None,
        }
    }
}

impl RetroParams {
    /// Refuse values the index cannot be built with. The message names the flag.
    pub fn validate(&self) -> Result<()> {
        let rb = self.retrieval_budget;
        let eb = self.estimation_budget;
        ensure!(
            rb.is_finite() && rb > 0.0 && rb <= 1.0,
            "--retro-retrieval-budget must be in (0, 1], got {rb}"
        );
        ensure!(
            eb.is_finite() && (0.0..1.0).contains(&eb),
            "--retro-estimation-budget must be in [0, 1), got {eb}"
        );
        ensure!(
            rb + eb <= 1.0,
            "--retro-retrieval-budget + --retro-estimation-budget must not exceed 1, got {}",
            rb + eb
        );
        ensure!(
            (256..=1 << 20).contains(&self.segment_tokens),
            "--retro-segment-tokens must be in 256..=1048576, got {}",
            self.segment_tokens
        );
        ensure!(
            (2..=1024).contains(&self.avg_cluster),
            "--retro-avg-cluster must be in 2..=1024, got {}",
            self.avg_cluster
        );
        ensure!(
            self.sink <= 4096,
            "--retro-sink must be at most 4096, got {}",
            self.sink
        );
        ensure!(
            (16..=65536).contains(&self.window),
            "--retro-window must be in 16..=65536, got {}",
            self.window
        );
        ensure!(
            self.update_interval >= 64 && self.update_interval >= 8 * self.avg_cluster,
            "--retro-update-interval must be at least 64 and at least 8 x --retro-avg-cluster \
             ({}), got {}",
            8 * self.avg_cluster,
            self.update_interval
        );
        let floor = self.sink + self.window + self.update_interval;
        ensure!(
            self.min_context >= floor,
            "--retro-min-context must be at least --retro-sink + --retro-window + \
             --retro-update-interval ({floor}), got {}",
            self.min_context
        );
        ensure!(
            self.index_cache_mb >= 64,
            "--retro-index-cache-mb must be at least 64, got {}",
            self.index_cache_mb
        );
        ensure!(
            self.shadow_every >= 1,
            "AVAROK_RETRO_SHADOW_EVERY must be at least 1"
        );
        ensure!(
            (2..=64).contains(&self.kmeans_iters),
            "k-means iterations must be in 2..=64, got {}",
            self.kmeans_iters
        );
        Ok(())
    }

    /// FNV-1a over every parameter that changes the index a build produces
    /// (the zone budgets do not: they only change selection). Part of the
    /// index-cache key, so a cached index is reused only under the same build.
    pub fn build_hash(&self) -> u64 {
        let mut h = Fnv::new();
        for v in [
            self.segment_tokens,
            self.avg_cluster,
            self.sink,
            self.window,
            self.update_interval,
            self.kmeans_iters,
        ] {
            h.write(&v.to_le_bytes());
        }
        h.finish()
    }

    /// Centroids an update segment adds.
    pub fn update_centroids(&self) -> u32 {
        update_centroids(self.update_interval, self.avg_cluster)
    }

    /// Whether shadow mode samples attention layer `layer`.
    pub fn shadow_layer(&self, layer: u32) -> bool {
        self.shadow_layers
            .as_ref()
            .is_none_or(|l| l.contains(&layer))
    }
}

/// FNV-1a 64, the hash the index-cache keys use.
pub(crate) struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Self {
        Self::new()
    }
}

impl Fnv {
    pub(crate) fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    pub(crate) fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }
    pub(crate) fn finish(&self) -> u64 {
        self.0
    }
}

/// Python's `round()` on a float: half to even.
pub fn round_half_even(x: f64) -> i64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && (r as i64) % 2 != 0 {
        (r - x.signum()) as i64
    } else {
        r as i64
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Least common multiple.
pub fn lcm(a: u32, b: u32) -> u32 {
    if a == 0 || b == 0 {
        0
    } else {
        a / gcd(a, b) * b
    }
}

/// `max(round(tokens / segment_tokens), 1)` (upstream `n_segments`).
pub fn n_segments(tokens: u32, segment_tokens: u32) -> u32 {
    let r = round_half_even(f64::from(tokens) / f64::from(segment_tokens.max(1)));
    u32::try_from(r.max(1)).unwrap_or(u32::MAX)
}

/// Centroids for an index over `tokens` keys: `round(tokens / avg)`, at least
/// `lcm(8, segments)`, moved to the nearer multiple of that lcm (the lower one
/// on a tie), exactly as upstream's `generate_config`.
pub fn n_centroids(tokens: u32, avg_cluster: u32, segments: u32) -> u32 {
    let factor = i64::from(lcm(8, segments.max(1)));
    let n = round_half_even(f64::from(tokens) / f64::from(avg_cluster.max(1))).max(factor);
    let lower = (n / factor) * factor;
    let upper = lower + factor;
    let pick = if (n - lower).abs() <= (n - upper).abs() {
        lower
    } else {
        upper
    };
    u32::try_from(pick).unwrap_or(u32::MAX)
}

/// Retrieval-zone clusters: `min(max(round(nc * budget), 1), nc)`.
pub fn nprobe(nc: u32, retrieval_budget: f64) -> u32 {
    let r = round_half_even(f64::from(nc) * retrieval_budget).max(1);
    u32::try_from(r).unwrap_or(u32::MAX).min(nc)
}

/// Estimation-zone clusters: `min(round(nc * budget), nc - nprobe)`.
pub fn estimation_clusters(nc: u32, estimation_budget: f64, nprobe: u32) -> u32 {
    let r = round_half_even(f64::from(nc) * estimation_budget).max(0);
    u32::try_from(r)
        .unwrap_or(u32::MAX)
        .min(nc.saturating_sub(nprobe))
}

/// Centroids one update segment adds: `max(round(interval / avg) // 8 * 8, 8)`.
pub fn update_centroids(interval: u32, avg_cluster: u32) -> u32 {
    let r = round_half_even(f64::from(interval) / f64::from(avg_cluster.max(1))).max(0);
    let r = u32::try_from(r).unwrap_or(u32::MAX);
    (r / 8 * 8).max(8)
}

/// Retrieval clusters an update adds: `max(round(update_centroids * budget), 1)`.
pub fn update_nprobe(update_centroids: u32, retrieval_budget: f64) -> u32 {
    let r = round_half_even(f64::from(update_centroids) * retrieval_budget).max(1);
    u32::try_from(r).unwrap_or(u32::MAX)
}

/// Estimation clusters an update adds: `max(round(update_centroids * budget), 1)`.
pub fn update_es(update_centroids: u32, estimation_budget: f64) -> u32 {
    let r = round_half_even(f64::from(update_centroids) * estimation_budget).max(1);
    u32::try_from(r).unwrap_or(u32::MAX)
}

/// Upper bound on the centroids an index over at most `context` tokens can
/// hold: the initial build (at most `ceil(context / avg)` plus one lcm step)
/// plus every update's rounding slack.
pub fn centroid_capacity(p: &RetroParams, context: u32) -> u32 {
    let segs = n_segments(context, p.segment_tokens);
    let uc = p.update_centroids();
    let per_interval = p.update_interval / p.avg_cluster.max(1);
    let updates = context.div_ceil(p.update_interval.max(1));
    context.div_ceil(p.avg_cluster.max(1))
        + lcm(8, segs)
        + uc
        + updates.saturating_mul(uc.saturating_sub(per_interval))
}

/// The geometry and capacities one attention layer's retro buffers are sized
/// for. The kernels (L34a, L34b) read it; the host (L33) builds it from the
/// first hook's `SeamCtx` and the parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct RetroCfg {
    pub params: RetroParams,
    pub kv_heads: u32,
    pub num_q_heads: u32,
    /// Query heads per KV head (GQA group).
    pub group: u32,
    pub head_dim: u32,
    /// The attention scale of the dense kernels (`SeamCtx::inv_sqrt_d`).
    pub inv_sqrt_d: f32,
    pub kv_dtype: KvDtypeTag,
    /// Longest context a sequence can reach (`max_blocks_per_seq * block_size`).
    pub max_context: u32,
    pub block_size: u32,
    /// Centroid capacity of an index over `max_context` tokens.
    pub cap_centroids: u32,
    /// Retrieval clusters a zone list can hold (the verify union included).
    pub retrieval_cap: u32,
    /// Estimation clusters a zone list can hold.
    pub es_cap: u32,
    /// Tokens the exact buffer holds per KV head.
    pub exact_cap: u32,
    /// Query rows `select_zones` scores at once (1; verify selects per row).
    pub query_rows: u32,
}

impl RetroCfg {
    /// Size every buffer for `max_context` tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: RetroParams,
        kv_heads: u32,
        num_q_heads: u32,
        head_dim: u32,
        inv_sqrt_d: f32,
        kv_dtype: KvDtypeTag,
        max_context: u32,
        block_size: u32,
    ) -> Result<Self> {
        ensure!(
            kv_heads > 0 && num_q_heads.is_multiple_of(kv_heads),
            "retro: {num_q_heads} query heads do not group over {kv_heads} KV heads"
        );
        let cap_centroids = centroid_capacity(&params, max_context);
        let (retrieval_cap, es_cap) = zone_caps(&params, cap_centroids);
        let retrieval_cap = retrieval_cap.min(MAX_RETRIEVAL_CAP);
        let exact_cap = exact_capacity(&params, retrieval_cap);
        Ok(Self {
            group: num_q_heads / kv_heads,
            params,
            kv_heads,
            num_q_heads,
            head_dim,
            inv_sqrt_d,
            kv_dtype,
            max_context,
            block_size: block_size.max(1),
            cap_centroids,
            retrieval_cap,
            es_cap,
            exact_cap,
            query_rows: 1,
        })
    }

    /// Bytes `build::build_layer_index` may use as scratch for a build of
    /// `tokens` keys into `centroids` centroids, per the contract in
    /// `state::PendingBuild`: per KV head an i32 assignment and a BF16 centered
    /// key copy per token, an f32 sum and an i32 count per centroid, and an f32
    /// mean; plus 1 MiB for tile partials.
    pub fn build_scratch_bytes(&self, tokens: u32, centroids: u32) -> usize {
        let hd = self.head_dim as usize;
        let per_head = tokens as usize * (4 + 2 * hd) + centroids as usize * (4 * hd + 8) + 4 * hd;
        self.kv_heads as usize * per_head + (1 << 20)
    }

    /// Bytes `select::select_zones` may use as scratch: an f32 score per
    /// centroid per KV head and per query row, plus 64 KiB for radix-select
    /// histograms.
    pub fn select_scratch_bytes(&self) -> usize {
        self.kv_heads as usize * self.cap_centroids as usize * 4 * self.query_rows.max(1) as usize
            + (64 << 10)
    }
}

/// Capacities of the retrieval and estimation lists for an index of at most
/// `cap_c` centroids. The zone sizes are linear in the centroid count between
/// the initial build and the updates, so the larger end of either bounds them;
/// the retrieval list holds twice that for the verify union.
pub fn zone_caps(p: &RetroParams, cap_c: u32) -> (u32, u32) {
    let uc = p.update_centroids().max(1);
    let updates = cap_c / uc + 1;
    let un = update_nprobe(uc, p.retrieval_budget);
    let ue = update_es(uc, p.estimation_budget);
    let np_build = (f64::from(cap_c) * p.retrieval_budget).ceil() as u32 + 1;
    let es_build = (f64::from(cap_c) * p.estimation_budget).ceil() as u32 + 1;
    let nprobe_cap = np_build.max(updates * un).min(cap_c.max(1));
    let es_cap = es_build.max(updates * ue).min(cap_c.max(1));
    (2 * nprobe_cap, es_cap)
}

/// Exact-buffer tokens per KV head: the sink, the window, one update interval
/// of not-yet-clustered tail, the verify rows, and the retrieval clusters at
/// twice the average cluster size (the gather stops at capacity and counts it).
pub fn exact_capacity(p: &RetroParams, retrieval_cap: u32) -> u32 {
    p.sink + p.window + p.update_interval + MAX_VERIFY_ROWS + retrieval_cap * p.avg_cluster * 2
}
