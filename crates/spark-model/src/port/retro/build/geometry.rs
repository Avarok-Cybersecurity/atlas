// SPDX-License-Identifier: AGPL-3.0-only
// Translated for Atlas from microsoft/RetrievalAttention (RetroInfer), MIT License,
// Copyright (c) Microsoft Corporation. Upstream: cache_hub/kmeans.py:225-260 (segment_k_means:
// the uniform-stride initialization, the segment split, the training tail), commit 03f912c.
// Changes: the sizes of one device build job (offsets for appends, CSR tiling, scratch layout) are
// Atlas's; the cluster and segment counts themselves come from the host (`config.rs`).

//! Responsibility: the host-side shape of one k-means build job: what a `PendingBuild` asks for,
//! normalized the way the CPU oracle normalizes it (at most one centroid per token, at most one
//! segment per centroid), the CSR tiling of its passes, the layout of its scratch inside the
//! caller's buffer, and the uniform-stride initialization arithmetic the kernel uses.

use anyhow::{Result, ensure};

use super::super::state::PendingBuild;

/// Head dimension the v1 kernels are written for (Qwen3.5 full attention).
pub const HEAD_DIM: u32 = 256;
/// CSR tiles per head and pass (`RK_TILES`).
pub const TILES: u32 = 64;
/// Token chunks of the two-stage key mean (`RK_MEAN_CHUNKS`).
pub const MEAN_CHUNKS: u32 = 64;
/// Smallest CSR tile (tokens), a multiple of the warp width.
pub const MIN_TILE_TOKENS: u32 = 256;
/// Scratch section alignment.
pub const ALIGN: usize = 256;

/// One build: cluster the keys at absolute positions `[first_pos, first_pos + n_tokens)` into
/// `n_centroids` clusters written at cluster index `centroid_offset`, their members at list slot
/// `token_offset` onward.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Job {
    pub first_pos: u32,
    pub n_tokens: u32,
    pub n_centroids: u32,
    pub n_segments: u32,
    /// Tokens per training segment (`n_tokens / n_segments`); the remainder joins only the final pass.
    pub seg_tokens: u32,
    /// Centroids per training segment (`n_centroids / n_segments`).
    pub seg_centroids: u32,
    pub centroid_offset: u32,
    pub token_offset: u32,
    /// k-means passes: `iters - 1` normalized segmented passes, then one plain global pass.
    pub iters: u32,
    /// CSR tile length.
    pub tile_tokens: u32,
    /// CSR tiles per head (`<= TILES`).
    pub n_tiles: u32,
}

impl Job {
    /// A job; the counts are normalized as `oracle::build_head` does: `nc = min(centroids,
    /// tokens)`, `segments = min(max(segments, 1), nc)`, at least one pass.
    pub fn new(
        first_pos: u32,
        tokens: u32,
        centroids: u32,
        segments: u32,
        iters: u32,
        centroid_offset: u32,
        token_offset: u32,
    ) -> Result<Self> {
        ensure!(tokens > 0, "retro build: the pending build has no tokens");
        ensure!(
            centroids > 0,
            "retro build: the pending build asks for no centroids"
        );
        let nc = centroids.min(tokens);
        let s = segments.max(1).min(nc);
        let tile_tokens = tokens
            .div_ceil(TILES)
            .next_multiple_of(32)
            .max(MIN_TILE_TOKENS);
        Ok(Self {
            first_pos,
            n_tokens: tokens,
            n_centroids: nc,
            n_segments: s,
            seg_tokens: tokens / s,
            seg_centroids: nc / s,
            centroid_offset,
            token_offset,
            iters: iters.max(1),
            tile_tokens,
            n_tiles: tokens.div_ceil(tile_tokens),
        })
    }

    /// The job a `PendingBuild` describes.
    pub fn from_pending(p: &PendingBuild) -> Result<Self> {
        Self::new(
            p.first_pos,
            p.tokens,
            p.centroids,
            p.segments,
            p.iters,
            p.centroid_offset,
            p.token_offset,
        )
    }

    /// One past the last absolute position the job reads.
    pub fn end_pos(&self) -> u32 {
        self.first_pos + self.n_tokens
    }
}

/// Byte offsets of a build's scratch inside the caller's scratch buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchLayout {
    /// `[kv_heads][n_tokens]` i32 assignment of the current pass.
    pub assign: usize,
    /// `[kv_heads][TILES][n_centroids]` i32 per-tile counts, then per-tile write cursors.
    pub hist: usize,
    /// `[kv_heads][n_centroids + 1]` i32 training-pass CSR starts.
    pub cstart: usize,
    /// `[kv_heads][n_centroids]` i32 training-pass cluster sizes.
    pub csize: usize,
    /// `[kv_heads][n_tokens]` i32 training-pass members (absolute positions).
    pub members: usize,
    /// `[kv_heads][MEAN_CHUNKS][256]` f32 chunk sums.
    pub mean_part: usize,
    /// `[kv_heads][256]` f32 key mean (BF16 values).
    pub mean: usize,
    /// `[bt_blocks]` i32 block table copy (used only when the hook gives no device table).
    pub block_table: usize,
    pub total: usize,
}

/// The scratch layout of a job over `kv_heads` heads with room for a `bt_blocks` block table.
pub fn scratch_layout(kv_heads: u32, job: &Job, bt_blocks: u32) -> ScratchLayout {
    let al = |x: usize| x.next_multiple_of(ALIGN);
    let h = kv_heads as usize;
    let n = job.n_tokens as usize;
    let nc = job.n_centroids as usize;
    let d = HEAD_DIM as usize;
    let assign = 0;
    let hist = al(assign + h * n * 4);
    let cstart = al(hist + h * TILES as usize * nc * 4);
    let csize = al(cstart + h * (nc + 1) * 4);
    let members = al(csize + h * nc * 4);
    let mean_part = al(members + h * n * 4);
    let mean = al(mean_part + h * MEAN_CHUNKS as usize * d * 4);
    let block_table = al(mean + h * d * 4);
    let total = al(block_table + bt_blocks as usize * 4);
    ScratchLayout {
        assign,
        hist,
        cstart,
        csize,
        members,
        mean_part,
        mean,
        block_table,
        total,
    }
}

/// Uniform-stride initialization (`kmeans.py:233-238`): centroid `i` starts at the key of local
/// token `trunc(f32(i) * f32(n / nc) + f32(n / nc / 2))`, every product and sum rounded to f32 as
/// the float32 `torch.arange` arithmetic does. Returns `(step, half_step)` for the kernel.
pub fn init_steps(n_tokens: u32, nc: u32) -> (f32, f32) {
    let step = f64::from(n_tokens) / f64::from(nc.max(1));
    (step as f32, (step / 2.0) as f32)
}

/// The local token centroid `i` is initialized from (the kernel's exact arithmetic).
pub fn init_index(i: u32, n_tokens: u32, nc: u32) -> u32 {
    let (step, half) = init_steps(n_tokens, nc);
    let f = (i as f32) * step + half;
    (f as u32).min(n_tokens.saturating_sub(1))
}
