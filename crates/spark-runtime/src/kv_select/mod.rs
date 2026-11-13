// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the bounded KV view's block scorer and picker (port lane L31): the
//! consumer-side views of the page sketch and the captured query rows, the page-softmax
//! scorer that runs on the device ([`score_blocks`]), the picker that turns scores into a
//! view ([`pick`]) and the diagnostic recall mass ([`recall_mass`]).
//!
//! # Vocabulary
//! A **page** is one Atlas KV block (`block_size` tokens, 16 on every shipped target). A
//! **view block** is `pages_per_block` consecutive pages of one sequence (8 by default, so
//! 128 tokens): the unit the scorer ranks and the picker keeps or drops. The sketch holds
//! one pre-RoPE mean key per (page, attention layer, KV head) plus the page's token count
//! (`kv_cache::sketch`); a view block's mean is the count-weighted mean of its pages.
//!
//! # Scoring (page-softmax)
//! For every captured query row `r`, attention layer `l` and query head `h`:
//! `logit_b = dot(q[l, r, h], M_b[l, kv_head(h)]) / sqrt(head_dim)` over the eligible view
//! blocks `b` (a block whose pages hold no sketched token is not eligible), `p = softmax_b`.
//! [`GqaAgg::Mean`] averages `p` over the query heads; [`GqaAgg::Max`] takes the per-GQA-group
//! maximum and renormalizes it over the blocks, then averages over the groups. Both then
//! average over layers and SUM over rows, so the scores of one call total `rows`. FP32
//! accumulation on the device, an f64 oracle on the host ([`oracle`]) and an f32 host model
//! of the device's exact loop structure ([`emulate`]).
//!
//! # Picking
//! The partial tail block first, then the `recent` newest blocks, the `sink` oldest blocks
//! and the mandatory blocks (the query span), then descending score with ties toward the
//! newer block, until the budget; the output is chronological. Nothing here reads a row or
//! a flag: lane L32 (`spark-model/src/port/kv_view`) owns the wiring, the row
//! `kv_view_selection` and the flags, and passes plain values down.
//!
//! # Query capture
//! [`capture`] holds the device slots of the captured query rows (the seam adapter that
//! fills them is `spark-model/src/port/kv_view/query_capture.rs`).
//!
//! # Cost when off
//! Nothing in this module allocates or launches unless a caller constructs a
//! [`ScoreScratch`] or a [`capture::QueryCapture`]; with the row off no caller exists.

use anyhow::{Result, bail};

use crate::gpu::{DevicePtr, GpuBackend};

pub mod capture;
pub mod emulate;
pub mod oracle;
mod pick;
mod plan;
mod score;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_capture;
#[cfg(test)]
mod tests_pick;
#[cfg(test)]
mod tests_sketch;

pub use pick::{PickCfg, expand_to_pages, pick, recall_mass};
pub use plan::{ScoreDims, ScorePlan, ScratchLayout};
pub use score::{KERNEL_MODULE, kernels_available, score_blocks};

/// Pages per view block when the caller has no reason to change it (128 tokens at 16).
pub const DEFAULT_PAGES_PER_BLOCK: usize = 8;

/// Device scratch the scorer is sized for: 1M tokens (7,813 view blocks of 128 tokens)
/// and 512 query rows on the Qwen3.5-4B shape fit with one merge per layer.
pub const DEFAULT_SCRATCH_BYTES: usize = 64 << 20;

/// How the query heads of one GQA group combine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GqaAgg {
    /// Average the per-head block probabilities over every query head.
    #[default]
    Mean,
    /// Per GQA group, the maximum probability over the group's heads, renormalized over
    /// the blocks, then averaged over the groups.
    Max,
}

impl GqaAgg {
    /// `"mean"` or `"max"`; `None` for anything else.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "mean" => Some(Self::Mean),
            "max" => Some(Self::Max),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Max => "max",
        }
    }

    /// The device kernels' mode word.
    pub(crate) fn mode(self) -> u32 {
        match self {
            Self::Mean => 0,
            Self::Max => 1,
        }
    }
}

/// One sequence's page sketch as the scorer reads it. `kv_cache::sketch::DeviceSketch`
/// implements it over the shared per-page buffers.
pub trait SketchView {
    /// The sequence's physical page (KV block) ids, in chronological order.
    fn pages(&self) -> &[u32];
    /// Physical pages in the sketch buffers (every id in `pages` is below it).
    fn num_blocks(&self) -> usize;
    /// Attention layers in the sketch.
    fn layers(&self) -> usize;
    fn kv_heads(&self) -> usize;
    fn head_dim(&self) -> usize;
    /// Tokens per page (the KV block size).
    fn page_tokens(&self) -> usize;
    /// `[num_blocks, layers, kv_heads, head_dim]` BF16 pre-RoPE mean keys.
    fn means(&self) -> DevicePtr;
    /// `[num_blocks]` u8 tokens behind each mean (0 = not sketched).
    fn counts(&self) -> DevicePtr;
}

/// The captured query rows of one sequence as the scorer reads them.
/// `port::kv_view::query_capture::CapturedRows` implements it.
pub trait QueryView {
    /// Valid rows per layer.
    fn rows(&self) -> usize;
    fn layers(&self) -> usize;
    fn q_heads(&self) -> usize;
    fn head_dim(&self) -> usize;
    /// `[layers, layer_stride_rows, q_heads, head_dim]` BF16 pre-RoPE, post-norm Q.
    fn data(&self) -> DevicePtr;
    /// Row capacity of one layer's slice of `data` (at least `rows`).
    fn layer_stride_rows(&self) -> usize {
        self.rows()
    }
}

/// What [`score_blocks`] needs besides the two views.
#[derive(Clone, Copy, Debug)]
pub struct ScoreCfg {
    pub pages_per_block: usize,
    pub gqa: GqaAgg,
    /// Device scratch owned by the caller (a [`ScoreScratch`]); never allocated per call.
    pub scratch: DevicePtr,
    pub scratch_bytes: usize,
}

/// Device scratch for [`score_blocks`], allocated once when the view is enabled.
#[derive(Debug)]
pub struct ScoreScratch {
    ptr: DevicePtr,
    bytes: usize,
}

impl ScoreScratch {
    /// Allocate `bytes` of scratch (use [`DEFAULT_SCRATCH_BYTES`] unless measured).
    pub fn new(gpu: &dyn GpuBackend, bytes: usize) -> Result<Self> {
        if bytes < 4096 {
            bail!("kv_select: scorer scratch of {bytes} bytes is too small (at least 4096)");
        }
        let ptr = gpu.alloc(bytes)?;
        Ok(Self { ptr, bytes })
    }

    pub fn ptr(&self) -> DevicePtr {
        self.ptr
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// A [`ScoreCfg`] over this scratch.
    pub fn cfg(&self, pages_per_block: usize, gqa: GqaAgg) -> ScoreCfg {
        ScoreCfg {
            pages_per_block,
            gqa,
            scratch: self.ptr,
            scratch_bytes: self.bytes,
        }
    }

    /// Return the scratch to the backend.
    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.ptr)
    }
}

/// View blocks covering `n_pages` pages.
pub fn n_blocks(n_pages: usize, pages_per_block: usize) -> usize {
    if pages_per_block == 0 {
        return 0;
    }
    n_pages.div_ceil(pages_per_block)
}
