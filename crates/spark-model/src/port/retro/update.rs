// SPDX-License-Identifier: AGPL-3.0-only
// Translated for Atlas from microsoft/RetrievalAttention (RetroInfer), MIT License,
// Copyright (c) Microsoft Corporation. Upstream: cache_hub/retroinfer_cache.py:595-728
// (_update_kv_cache, decode_update_kv_cache: when 1,024 generated tokens accumulate past the
// window they are clustered with one segment and appended; nprobe and the estimation zone
// grow by UPDATE_NPROBE and UPDATE_ES). Changes: Rust; positions instead of a moving steady
// zone (the tail stays in the paged KV cache and is read in place); update ranges aligned
// to the KV block size so the index-cache keys stay block-exact; fixed-capacity buffers
// instead of torch.cat, refused (dense fallback) when full instead of reallocated mid-graph.

//! Responsibility: the index growth policy, shared by the device path and the
//! oracle: when the initial build runs and over which range, when a
//! decode-time update is due, the counts after a build, an update or an append
//! after a cached prefix, and the [`PendingBuild`] each of those hands the
//! build kernel.

use spark_runtime::gpu::DevicePtr;

use super::config::{self, RetroCfg, RetroParams};
use super::oracle::KmeansSpec;
use super::state::{IndexCounts, PendingBuild};

fn align_down(v: u32, block: u32) -> u32 {
    let b = block.max(1);
    v / b * b
}

/// Tokens of an update segment: the interval, aligned down to the block size.
pub fn update_tokens(p: &RetroParams, block: u32) -> u32 {
    align_down(p.update_interval, block).max(block.max(1))
}

/// The end of the initial index for a sequence of `len` tokens, or `None`
/// below `--retro-min-context` (dense). The index covers `[sink, end)`; `end`
/// is `len - window` aligned down to the block size, so at least `window`
/// tokens stay exact.
pub fn initial_end(len: u32, p: &RetroParams, block: u32) -> Option<u32> {
    if len < p.min_context || len <= p.window {
        return None;
    }
    let end = align_down(len - p.window, block);
    (end >= p.sink + 8 * p.avg_cluster).then_some(end)
}

/// Tokens the next update clusters, when one is due at `len`: the exact tail
/// `[indexed_end, len)` has reached `window + interval`.
pub fn due(c: &IndexCounts, len: u32, p: &RetroParams, block: u32) -> Option<u32> {
    if c.n_centroids == 0 {
        return None;
    }
    let tail = len.saturating_sub(c.indexed_end);
    let tokens = update_tokens(p, block);
    (tail >= p.window + tokens).then_some(tokens)
}

/// Counts after the initial build of `centroids` centroids over `[sink, end)`.
pub fn counts_after_build(p: &RetroParams, sink: u32, end: u32, centroids: u32) -> IndexCounts {
    let np = config::nprobe(centroids, p.retrieval_budget);
    IndexCounts {
        n_centroids: centroids,
        nprobe: np,
        es: config::estimation_clusters(centroids, p.estimation_budget, np),
        indexed_end: end,
        sink,
        listed: end - sink,
    }
}

/// Counts after one update segment of `tokens` keys into `uc` centroids
/// (upstream: nprobe += UPDATE_NPROBE, es += UPDATE_ES).
pub fn counts_after_update(c: &IndexCounts, p: &RetroParams, uc: u32, tokens: u32) -> IndexCounts {
    let n_centroids = c.n_centroids + uc;
    let nprobe = (c.nprobe + config::update_nprobe(p.update_centroids(), p.retrieval_budget))
        .min(n_centroids);
    let es = (c.es + config::update_es(p.update_centroids(), p.estimation_budget))
        .min(n_centroids.saturating_sub(nprobe));
    IndexCounts {
        n_centroids,
        nprobe,
        es,
        indexed_end: c.indexed_end + tokens,
        sink: c.sink,
        listed: c.listed + tokens,
    }
}

/// Counts after a segmented append of `added` centroids over `tokens` keys (a
/// sequence that reused a cached prefix index clusters its own remainder):
/// the zone sizes are recomputed from the new total, as a fresh build would.
pub fn counts_after_append(
    c: &IndexCounts,
    p: &RetroParams,
    added: u32,
    tokens: u32,
) -> IndexCounts {
    let n = c.n_centroids + added;
    let np = config::nprobe(n, p.retrieval_budget);
    IndexCounts {
        n_centroids: n,
        nprobe: np,
        es: config::estimation_clusters(n, p.estimation_budget, np),
        indexed_end: c.indexed_end + tokens,
        sink: c.sink,
        listed: c.listed + tokens,
    }
}

/// One build the device path runs: the pending kernel input and the counts
/// the index carries once it finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuildStep {
    pub first_pos: u32,
    pub tokens: u32,
    pub spec: KmeansSpec,
    pub after: IndexCounts,
}

impl BuildStep {
    /// The kernel input, with the caller's scratch.
    pub fn pending(
        &self,
        before: &IndexCounts,
        scratch: DevicePtr,
        scratch_bytes: usize,
    ) -> PendingBuild {
        PendingBuild {
            first_pos: self.first_pos,
            tokens: self.tokens,
            segments: u32::try_from(self.spec.segments).unwrap_or(1),
            centroids: u32::try_from(self.spec.centroids).unwrap_or(0),
            centroid_offset: before.n_centroids,
            token_offset: before.listed,
            iters: u32::try_from(self.spec.iters).unwrap_or(config::KMEANS_ITERS),
            scratch,
            scratch_bytes,
        }
    }

    /// Scratch bytes the build needs.
    pub fn scratch_bytes(&self, cfg: &RetroCfg) -> usize {
        cfg.build_scratch_bytes(self.tokens, u32::try_from(self.spec.centroids).unwrap_or(0))
    }
}

/// The initial build for a sequence of `len` tokens, or `None` (dense).
pub fn initial_step(len: u32, p: &RetroParams, block: u32) -> Option<BuildStep> {
    let end = initial_end(len, p, block)?;
    let sink = p.sink.min(end);
    let tokens = end - sink;
    let spec = KmeansSpec::initial(p, tokens);
    let nc = u32::try_from(spec.centroids.min(tokens as usize)).ok()?;
    Some(BuildStep {
        first_pos: sink,
        tokens,
        spec: KmeansSpec {
            centroids: nc as usize,
            ..spec
        },
        after: counts_after_build(p, sink, end, nc),
    })
}

/// The update due at `len`, if any.
pub fn update_step(c: &IndexCounts, len: u32, p: &RetroParams, block: u32) -> Option<BuildStep> {
    let tokens = due(c, len, p, block)?;
    let spec = KmeansSpec::update(p);
    let uc = u32::try_from(spec.centroids.min(tokens as usize)).ok()?;
    Some(BuildStep {
        first_pos: c.indexed_end,
        tokens,
        spec: KmeansSpec {
            centroids: uc as usize,
            ..spec
        },
        after: counts_after_update(c, p, uc, tokens),
    })
}

/// After a cached prefix index covering `c.indexed_end` was installed for a
/// sequence of `len` tokens: the segmented append of the sequence's own
/// remainder, when it is longer than one update interval (shorter remainders
/// stay in the exact tail and the regular updates take them).
pub fn append_step(c: &IndexCounts, len: u32, p: &RetroParams, block: u32) -> Option<BuildStep> {
    let end = initial_end(len, p, block)?;
    if end <= c.indexed_end || end - c.indexed_end < update_tokens(p, block) {
        return None;
    }
    let tokens = end - c.indexed_end;
    let spec = KmeansSpec::initial(p, tokens);
    let added = u32::try_from(spec.centroids.min(tokens as usize)).ok()?;
    Some(BuildStep {
        first_pos: c.indexed_end,
        tokens,
        spec: KmeansSpec {
            centroids: added as usize,
            ..spec
        },
        after: counts_after_append(c, p, added, tokens),
    })
}

/// Whether an index of `after` counts fits buffers sized for `cap_centroids`
/// centroids and `cap_tokens` listed tokens.
pub fn fits(after: &IndexCounts, cap_centroids: u32, cap_tokens: u32) -> bool {
    after.n_centroids <= cap_centroids && after.listed <= cap_tokens
}
