// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the scorer's tiling plan: how one [`super::score_blocks`] call cuts its
//! work so the device scratch never exceeds the caller's budget, and where each piece of
//! scratch lives. Pure host arithmetic, shared by the launcher, the f32 emulator and the
//! tests, so all three walk the same tiles in the same order.
//!
//! Per attention layer the scorer merges the view blocks' page means into an f32
//! `[chunk, kv_heads, head_dim]` buffer (`merged`), computes the logits of one row tile
//! against every block (`logits`, `[rows_tile, q_heads, n_blocks]` f32, all blocks at once
//! because the softmax normalizes over them), reduces the per-(row, head) max and sum
//! (`stats`), the per-(row, group) renormalizer in max mode (`gnorm`), and adds the tile's
//! contribution to the output scores. When every block fits one chunk (up to 8,192 blocks,
//! 1M tokens, on the 4B shape with the default 64 MiB) the merge runs once per layer;
//! beyond that it is redone per row tile, which is correct at any size and only slower.

use anyhow::{Result, bail};

/// Every scratch region starts on this boundary.
pub const ALIGN: usize = 256;

/// Most f32 accumulators one logits warp keeps (rows x heads of one group per CTA).
pub const MAX_ACC: usize = 32;

/// Shared-memory budget of one logits CTA (its query rows in f32).
pub const LOGITS_SMEM_BYTES: usize = 48 << 10;

/// View blocks one logits CTA covers (one per warp, 8 warps).
pub const LOGITS_BLOCKS_PER_CTA: usize = 8;

fn align(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

/// The shapes of one scoring call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScoreDims {
    pub n_pages: usize,
    pub pages_per_block: usize,
    pub rows: usize,
    pub layers: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl ScoreDims {
    pub fn n_blocks(&self) -> usize {
        super::n_blocks(self.n_pages, self.pages_per_block)
    }

    /// Query heads per KV head.
    pub fn group(&self) -> usize {
        self.q_heads / self.kv_heads.max(1)
    }

    /// Reject shapes the kernels do not serve, with the reason.
    pub fn validate(&self) -> Result<()> {
        let d = self;
        if d.n_pages == 0 || d.pages_per_block == 0 || d.rows == 0 || d.layers == 0 {
            bail!("kv_select: empty scoring call ({d:?})");
        }
        if d.kv_heads == 0 || d.q_heads == 0 || !d.q_heads.is_multiple_of(d.kv_heads) {
            bail!(
                "kv_select: {} query heads are not a multiple of {} KV heads",
                d.q_heads,
                d.kv_heads
            );
        }
        if d.head_dim == 0 || d.head_dim > 1024 {
            bail!("kv_select: head_dim {} outside 1..=1024", d.head_dim);
        }
        if d.group() > acc_slots(d.head_dim) {
            bail!(
                "kv_select: GQA group of {} heads exceeds the logits kernel's {} accumulators at head_dim {}",
                d.group(),
                acc_slots(d.head_dim),
                d.head_dim
            );
        }
        if u32::try_from(d.n_pages).is_err() || u32::try_from(d.rows).is_err() {
            bail!(
                "kv_select: {} pages / {} rows exceed u32",
                d.n_pages,
                d.rows
            );
        }
        Ok(())
    }
}

/// Accumulators one logits CTA may use at this head dimension (bounded by shared memory).
pub fn acc_slots(head_dim: usize) -> usize {
    (LOGITS_SMEM_BYTES / (head_dim.max(1) * 4)).clamp(1, MAX_ACC)
}

/// Byte offsets of every scratch region (from the scratch base).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScratchLayout {
    /// `[n_pages]` i32 physical page ids.
    pub pages: usize,
    /// `[chunk_blocks, kv_heads, head_dim]` f32 merged block means.
    pub merged: usize,
    /// `[chunk_blocks]` f32 total tokens per merged block.
    pub wsum: usize,
    /// `[rows_tile, q_heads, n_blocks]` f32 logits.
    pub logits: usize,
    /// `[rows_tile * q_heads, 2]` f32 (max, sum of exp).
    pub stats: usize,
    /// `[rows_tile * kv_heads]` f32 max-mode renormalizers.
    pub gnorm: usize,
    /// Bytes used, every region aligned.
    pub total: usize,
}

impl ScratchLayout {
    fn new(d: &ScoreDims, chunk_blocks: usize, rows_tile: usize) -> Self {
        let nb = d.n_blocks();
        let mut at = 0usize;
        let mut take = |bytes: usize| {
            let off = at;
            at += align(bytes);
            off
        };
        let pages = take(d.n_pages * 4);
        let merged = take(chunk_blocks * d.kv_heads * d.head_dim * 4);
        let wsum = take(chunk_blocks * 4);
        let logits = take(rows_tile * d.q_heads * nb * 4);
        let stats = take(rows_tile * d.q_heads * 8);
        let gnorm = take(rows_tile * d.kv_heads * 4);
        Self {
            pages,
            merged,
            wsum,
            logits,
            stats,
            gnorm,
            total: at,
        }
    }
}

/// How one call is tiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScorePlan {
    pub dims: ScoreDims,
    /// View blocks merged at once.
    pub chunk_blocks: usize,
    /// Query rows whose logits are held at once.
    pub rows_tile: usize,
    /// Query rows one logits CTA loads (rows x group <= the accumulator slots).
    pub cta_rows: usize,
    pub layout: ScratchLayout,
}

impl ScorePlan {
    /// The largest tiles that fit `scratch_bytes`, preferring one merge per layer.
    pub fn new(dims: ScoreDims, scratch_bytes: usize) -> Result<Self> {
        dims.validate()?;
        let nb = dims.n_blocks();
        let per_merged = dims.kv_heads * dims.head_dim * 4 + 4;
        let mut chunk = nb.min((scratch_bytes / 2 / per_merged).max(1));
        loop {
            if let Some(rows_tile) = fit_rows(&dims, chunk, scratch_bytes) {
                let cta_rows = (acc_slots(dims.head_dim) / dims.group()).max(1);
                let layout = ScratchLayout::new(&dims, chunk, rows_tile);
                debug_assert!(layout.total <= scratch_bytes);
                return Ok(Self {
                    dims,
                    chunk_blocks: chunk,
                    rows_tile,
                    cta_rows,
                    layout,
                });
            }
            if chunk == 1 {
                let need = ScratchLayout::new(&dims, 1, 1).total;
                bail!(
                    "kv_select: scorer scratch of {scratch_bytes} bytes cannot hold one row of \
                     {nb} view block logits ({need} bytes needed); raise the scratch"
                );
            }
            chunk = chunk.div_ceil(2);
        }
    }

    pub fn n_blocks(&self) -> usize {
        self.dims.n_blocks()
    }

    pub fn n_chunks(&self) -> usize {
        self.n_blocks().div_ceil(self.chunk_blocks)
    }

    pub fn n_row_tiles(&self) -> usize {
        self.dims.rows.div_ceil(self.rows_tile)
    }

    /// `(first block, blocks)` of chunk `c`.
    pub fn chunk(&self, c: usize) -> (usize, usize) {
        let start = c * self.chunk_blocks;
        (start, self.chunk_blocks.min(self.n_blocks() - start))
    }

    /// `(first row, rows)` of row tile `t`.
    pub fn row_tile(&self, t: usize) -> (usize, usize) {
        let start = t * self.rows_tile;
        (start, self.rows_tile.min(self.dims.rows - start))
    }

    /// Kernel launches one call issues (the tests pin the launcher to it).
    pub fn launches(&self, max_mode: bool) -> usize {
        let chunks = self.n_chunks();
        let tiles = self.n_row_tiles();
        let merges = if chunks == 1 { 1 } else { tiles * chunks };
        let per_tile = chunks + 2 + usize::from(max_mode);
        self.dims.layers * (merges + tiles * per_tile)
    }
}

/// The most rows (at most `dims.rows`) whose tile fits beside a `chunk`-block merge.
fn fit_rows(dims: &ScoreDims, chunk: usize, scratch_bytes: usize) -> Option<usize> {
    if ScratchLayout::new(dims, chunk, 1).total > scratch_bytes {
        return None;
    }
    let (mut lo, mut hi) = (1usize, dims.rows);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if ScratchLayout::new(dims, chunk, mid).total <= scratch_bytes {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Some(lo)
}
