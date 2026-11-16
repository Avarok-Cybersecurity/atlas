// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm after Quest (page min/max upper-bound selection) and the page-sparse prototype in
// arczhi/llama-tierkv scripts-5060ti/proto/proto_page_sparse.cu (MIT, Copyright (c) 2023-2026 The
// ggml authors; file has no header of its own). Launch geometry written for Atlas's kernels in
// `kernels/ada/common/attn_page_sparse_ada.cu`; no prototype source was read.

//! Responsibility: the host ABI of the three Quest kernels (port lane L30):
//! lazy resolution of `attn_page_sparse_ada` (present only in the Ada image;
//! every other target issues no lookup), the per-layer buffer view, and one
//! launch function per entry point with its grid. Shared by the selector and by
//! `examples/page_sparse_microtest.rs`, so the oracle runs exactly these
//! launches.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::page_summary::SummaryLayout;

/// The module `kernels/ada/common/attn_page_sparse_ada.cu` compiles to.
pub const MODULE: &str = "attn_page_sparse_ada";
pub const ENTRY_UPDATE: &str = "attn_ps_summary_update";
pub const ENTRY_SCORE: &str = "attn_ps_page_score";
pub const ENTRY_COMPACT: &str = "attn_ps_compact";

/// Mirrors of the kernel's `#define`s.
pub const UPDATE_THREADS: u32 = 128;
pub const SCORE_THREADS: u32 = 256;
pub const PAGES_PER_CTA: u32 = 32;
pub const COMPACT_THREADS: u32 = 1024;
/// Candidates one compaction CTA ranks; longer histories stay dense (identity).
pub const MAX_CANDIDATES: u32 = 8192;

/// The three resolved entry points.
#[derive(Clone, Copy, Debug)]
pub struct Kernels {
    pub update: KernelHandle,
    pub score: KernelHandle,
    pub compact: KernelHandle,
}

impl Kernels {
    /// The kernels, or `None` when the module is not in this image (no lookup
    /// is issued then) or an entry did not resolve (one warning).
    pub fn resolve(gpu: &dyn GpuBackend) -> Option<Self> {
        if !gpu.has_module(MODULE) {
            return None;
        }
        let probe = |e: &str| crate::layers::try_target_kernel(gpu, MODULE, e);
        let k = Self {
            update: probe(ENTRY_UPDATE),
            score: probe(ENTRY_SCORE),
            compact: probe(ENTRY_COMPACT),
        };
        if k.update.0 == 0 || k.score.0 == 0 || k.compact.0 == 0 {
            tracing::warn!(
                "{MODULE} is in the image but an entry did not resolve (update={} score={} \
                 compact={}); Quest page-sparse attention stays dense",
                k.update.0 != 0,
                k.score.0 != 0,
                k.compact.0 != 0
            );
            return None;
        }
        Some(k)
    }
}

/// One attention layer's pools and summary regions, in kernel terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerBufs {
    pub k_pool: DevicePtr,
    pub vals: DevicePtr,
    pub meta: DevicePtr,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub page_blocks: u32,
    /// Bytes from one block of this layer's K pool to the next.
    pub block_stride: u64,
    /// Bytes per key element: 2 (BF16) or 1 (FP8 E4M3).
    pub elem_bytes: u32,
    pub cap_blocks: u32,
}

impl LayerBufs {
    /// The view of attention layer `layer` of an arena at `base`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layout: &SummaryLayout,
        base: DevicePtr,
        layer: usize,
        k_pool: DevicePtr,
        block_stride: u64,
        elem_bytes: u32,
    ) -> Self {
        let g = layout.layers[layer];
        Self {
            k_pool,
            vals: layout.layer_vals(base, layer),
            meta: layout.layer_meta(base, layer),
            kv_heads: g.kv_heads,
            head_dim: g.head_dim,
            block_size: layout.block_size,
            page_blocks: layout.page_blocks,
            block_stride,
            elem_bytes,
            cap_blocks: layout.cap_blocks,
        }
    }

    pub fn page_tokens(&self) -> u32 {
        self.page_blocks * self.block_size
    }
}

/// Where the update kernel finds a group's block table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableRef {
    /// `[groups * rows_per_seq, row_stride]` i32.
    Flat(DevicePtr),
    /// A device array of per-stream table pointers (batched prefill), and the
    /// stream this launch covers.
    Ptrs { ptrs: DevicePtr, index: u32 },
}

/// The tokens an update launch folds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateRange {
    /// Decode/verify: per group, `rows_per_seq` new tokens ending at its rows'
    /// device extents (graph-safe).
    Device {
        seq_lens: DevicePtr,
        rows_per_seq: u32,
        groups: u32,
    },
    /// Prefill: one group's written range, known on the host (eager).
    Host { first_pos: u32, count: u32 },
}

/// CTAs along x an update needs for `count` tokens at any alignment.
pub fn pages_touched(count: u32, page_tokens: u32) -> u32 {
    if count == 0 {
        0
    } else {
        (count - 1) / page_tokens.max(1) + 2
    }
}

/// Fold the range's keys into its pages' summaries.
pub fn launch_update(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    l: &LayerBufs,
    table: TableRef,
    row_stride: u32,
    range: UpdateRange,
    stream: u64,
) -> Result<()> {
    let (bt, ptrs, group_base) = match table {
        TableRef::Flat(p) => (p, DevicePtr::NULL, 0),
        TableRef::Ptrs { ptrs, index } => (DevicePtr::NULL, ptrs, index),
    };
    let (seq_lens, first, count, rows, groups, span) = match range {
        UpdateRange::Device {
            seq_lens,
            rows_per_seq,
            groups,
        } => (seq_lens, 0, 0, rows_per_seq, groups, rows_per_seq),
        UpdateRange::Host { first_pos, count } => (DevicePtr::NULL, first_pos, count, 1, 1, count),
    };
    let x = pages_touched(span, l.page_tokens());
    if x == 0 || groups == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, k.update)
        .grid([x, l.kv_heads, groups])
        .block([UPDATE_THREADS, 1, 1])
        .arg_ptr(l.k_pool)
        .arg_ptr(l.vals)
        .arg_ptr(l.meta)
        .arg_ptr(bt)
        .arg_ptr(ptrs)
        .arg_ptr(seq_lens)
        .arg_u32(first)
        .arg_u32(count)
        .arg_u32(row_stride)
        .arg_u32(rows)
        .arg_u32(group_base)
        .arg_u32(l.kv_heads)
        .arg_u32(l.head_dim)
        .arg_u32(l.block_size)
        .arg_u32(l.page_blocks)
        .arg_u64(l.block_stride)
        .arg_u32(l.elem_bytes)
        .arg_u32(l.cap_blocks)
        .launch(stream)
}

/// The selection budget the score and compaction kernels share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    pub sink_pages: u32,
    pub recent_tokens: u32,
    pub topk_pages: u32,
    pub min_ctx: u32,
}

/// One step's rows as the attention reads them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepRows {
    /// `[groups * rows_per_seq, row_stride]` i32 dense block table.
    pub block_table: DevicePtr,
    /// `[groups * rows_per_seq]` i32 extents.
    pub seq_lens: DevicePtr,
    pub rows_per_seq: u32,
    pub groups: u32,
    pub row_stride: u32,
}

/// Score slots per group: one per page of a full-length row.
pub fn score_stride(row_stride: u32, page_blocks: u32) -> u32 {
    row_stride.div_ceil(page_blocks.max(1))
}

/// The Quest bound of every candidate page into `scores` (`[groups, score_stride]` f32).
#[allow(clippy::too_many_arguments)]
pub fn launch_score(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    l: &LayerBufs,
    s: &StepRows,
    q: DevicePtr,
    q_row_stride: u32,
    num_q_heads: u32,
    scores: DevicePtr,
    b: &Budget,
    gqa_sum: u32,
    stream: u64,
) -> Result<()> {
    let stride = score_stride(s.row_stride, l.page_blocks);
    if s.groups == 0 || stride == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, k.score)
        .grid([stride.div_ceil(PAGES_PER_CTA), s.groups, 1])
        .block([SCORE_THREADS, 1, 1])
        .arg_ptr(l.k_pool)
        .arg_ptr(l.vals)
        .arg_ptr(l.meta)
        .arg_ptr(s.block_table)
        .arg_ptr(s.seq_lens)
        .arg_ptr(q)
        .arg_ptr(scores)
        .arg_u32(s.rows_per_seq)
        .arg_u32(s.row_stride)
        .arg_u32(q_row_stride)
        .arg_u32(num_q_heads)
        .arg_u32(l.kv_heads)
        .arg_u32(l.head_dim)
        .arg_u32(l.block_size)
        .arg_u32(l.page_blocks)
        .arg_u64(l.block_stride)
        .arg_u32(l.elem_bytes)
        .arg_u32(l.cap_blocks)
        .arg_u32(b.sink_pages)
        .arg_u32(b.recent_tokens)
        .arg_u32(stride)
        .arg_u32(gqa_sum)
        .arg_u32(b.topk_pages)
        .arg_u32(b.min_ctx)
        .launch(stream)
}

/// The selector-owned outputs of a compaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewOut {
    /// `[groups * rows_per_seq, row_stride]` i32.
    pub block_table: DevicePtr,
    /// `[groups * rows_per_seq]` i32.
    pub seq_lens: DevicePtr,
    /// Two u64 counters (sequence steps, pages kept), or NULL with `count_stats` 0.
    pub stats: DevicePtr,
}

/// Top-k selection and the compacted tables.
#[allow(clippy::too_many_arguments)]
pub fn launch_compact(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    block_size: u32,
    page_blocks: u32,
    s: &StepRows,
    scores: DevicePtr,
    out: &ViewOut,
    b: &Budget,
    count_stats: bool,
    stream: u64,
) -> Result<()> {
    if s.groups == 0 {
        return Ok(());
    }
    let count = u32::from(count_stats && !out.stats.is_null());
    KernelLaunch::new(gpu, k.compact)
        .grid([s.groups, 1, 1])
        .block([COMPACT_THREADS, 1, 1])
        .arg_ptr(s.block_table)
        .arg_ptr(s.seq_lens)
        .arg_ptr(scores)
        .arg_ptr(out.block_table)
        .arg_ptr(out.seq_lens)
        .arg_ptr(out.stats)
        .arg_u32(s.rows_per_seq)
        .arg_u32(s.row_stride)
        .arg_u32(score_stride(s.row_stride, page_blocks))
        .arg_u32(block_size)
        .arg_u32(page_blocks)
        .arg_u32(b.sink_pages)
        .arg_u32(b.recent_tokens)
        .arg_u32(b.topk_pages)
        .arg_u32(b.min_ctx)
        .arg_u32(count)
        .launch(stream)
}
