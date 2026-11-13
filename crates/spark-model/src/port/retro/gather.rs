// SPDX-License-Identifier: AGPL-3.0-only
// Translated for Atlas from microsoft/RetrievalAttention (RetroInfer), MIT License,
// Copyright (c) Microsoft Corporation. Upstream: cache_hub/retroinfer_cache.py:754-815
// (`sparse_attention`, the execution buffer of steady zone plus retrieval clusters) and
// library/retroinfer/retroinfer_kernels/src/gather_copy.cu:100-217 (access pattern only).
// Changes: rewritten for Atlas's paged BF16/FP8 KV at head_dim 256, the index's CSR member
// lists, a fixed-capacity per-head scratch with device counts, host launch plumbing in Rust.

//! Responsibility: the RetroInfer exact-zone gather (port lane L34b): the host ABI of
//! `kernels/ada/common/retro_gather_ada.cu` (module, entries, geometry, argument order) and
//! the contract entry point [`gather_exact`] that L33's selector calls. The buffer layout and
//! entry order are `state::ExactBuf`'s contract; the CPU reference that models the kernel on
//! the same layouts is the child module `gather_reference.rs`, re-exported here.
//!
//! The launch is capture-safe: no allocation, no synchronization and no host read of a
//! device value; every count the kernel needs (header, selection counts, row lengths) lives
//! on the device, so a captured graph replays it while the selection changes underneath.
//! The module is compiled only by `kernels/ada`; on any other target
//! [`GatherKernels::available`] answers false and [`gather_exact`] fails after one logged
//! line, which the selector turns into a dense fallback.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::state::{ExactBuf, IndexListRef, ZoneLists};
use crate::layers::qwen3_attention::sparse_seam::{KvDtypeTag, KvLayerView, SeqBatch};

#[path = "gather_reference.rs"]
mod reference;
pub use reference::{
    HostExact, HostLists, HostPaged, HostZones, exact_positions, reference_gather,
};

/// Module the kernel file compiles to (its file stem).
pub const MODULE: &str = "retro_gather_ada";
pub const ENTRY_BF16: &str = "retro_gather_exact_bf16";
pub const ENTRY_FP8: &str = "retro_gather_exact_fp8";
/// The only head dimension the RetroInfer kernels serve (Qwen3.5 full attention).
pub const HEAD_DIM: u32 = super::config::HEAD_DIM;
/// Threads per CTA (eight warps); mirrors `RG_THREADS`.
pub const THREADS: u32 = 256;
/// Warps per CTA; mirrors `RG_WARPS`.
pub const WARPS: u32 = 8;
/// Largest number of CTAs per kv head.
pub const MAX_GRID_Y: u32 = 64;
/// Largest retrieval capacity: the per-CTA cluster table, `(3 * cap + 1) * 4` bytes of
/// dynamic shared memory, stays under the 48 KiB default.
pub const MAX_RETRIEVAL_CAP: u32 = 4000;
/// Largest rows of one sequence per launch (`config::MAX_VERIFY_ROWS` fits).
pub const MAX_ROWS: u32 = 64;

/// The exact buffer's device arrays, as both kernels read or write them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactOut {
    /// BF16 `[kv_heads][cap][256]` keys and values.
    pub k: DevicePtr,
    pub v: DevicePtr,
    /// i32 `[kv_heads][cap]` absolute positions (-1 = skip).
    pub pos: DevicePtr,
    /// i32 `[kv_heads]` valid entries and dropped retrieval tokens.
    pub valid: DevicePtr,
    pub dropped: DevicePtr,
    pub kv_heads: u32,
    pub cap: u32,
}

impl ExactOut {
    /// The device arrays of `e`.
    pub fn of(e: &ExactBuf) -> Self {
        Self {
            k: e.k,
            v: e.v,
            pos: e.positions,
            valid: e.valid_len,
            dropped: e.dropped,
            kv_heads: e.kv_heads,
            cap: e.capacity,
        }
    }
}

/// The two resolved gather entries.
#[derive(Clone, Copy, Debug)]
pub struct GatherKernels {
    pub bf16: KernelHandle,
    pub fp8: KernelHandle,
}

impl GatherKernels {
    /// True when the module is compiled into this image. Issues no kernel lookup.
    pub fn available(gpu: &dyn GpuBackend) -> bool {
        gpu.has_module(MODULE)
    }

    /// Resolve both entries (memoized per backend).
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        ensure!(
            Self::available(gpu),
            "retro gather: module `{MODULE}` is not in this kernel image (RetroInfer runs only \
             on a target that compiles kernels/ada/common/retro_gather_ada.cu; build with \
             AVAROK_TARGET_HW=ada)"
        );
        Ok(Self {
            bf16: gpu.op_cache().kernel(gpu, MODULE, ENTRY_BF16)?,
            fp8: gpu.op_cache().kernel(gpu, MODULE, ENTRY_FP8)?,
        })
    }

    pub fn for_dtype(&self, dtype: KvDtypeTag) -> KernelHandle {
        match dtype {
            KvDtypeTag::Bf16 => self.bf16,
            KvDtypeTag::Fp8 => self.fp8,
        }
    }
}

/// Everything one gather launch reads and writes.
#[derive(Clone, Copy, Debug)]
pub struct GatherLaunch {
    pub kv: KvLayerView,
    /// i32 `[max_blocks_per_seq]`: the sequence's block table.
    pub block_table: DevicePtr,
    /// i32 `[rows]`: the gather covers the longest row.
    pub row_lens: DevicePtr,
    pub rows: u32,
    pub max_blocks_per_seq: u32,
    /// i32 `[kv_heads][retrieval_cap]` rank-order ids, i32 `[kv_heads][2]` counts.
    pub retrieval_ids: DevicePtr,
    pub counts: DevicePtr,
    pub retrieval_cap: u32,
    pub lists: IndexListRef,
    pub out: ExactOut,
}

/// Bytes from one block of the layer's pool to the next, as the kernel takes it.
pub fn block_stride(kv: &KvLayerView) -> u64 {
    if kv.block_stride_bytes > 0 {
        kv.block_stride_bytes
    } else {
        min_block_stride(kv)
    }
}

fn min_block_stride(kv: &KvLayerView) -> u64 {
    let elem = kv.dtype.bytes_per_elem() as u64;
    u64::from(kv.block_size) * u64::from(kv.kv_heads) * u64::from(kv.head_dim) * elem
}

/// Launch grid: one CTA row per kv head, enough CTAs for about four entries per warp.
pub fn gather_grid(kv_heads: u32, cap: u32) -> [u32; 3] {
    [
        kv_heads,
        div_ceil(cap.max(1), WARPS * 4).clamp(1, MAX_GRID_Y),
        1,
    ]
}

/// Dynamic shared memory: the cluster table (stored prefix, list start, member count).
pub fn gather_smem(retrieval_cap: u32) -> u32 {
    (3 * retrieval_cap + 1) * 4
}

/// Refuse a launch the kernel cannot serve. Host values only.
pub fn validate(g: &GatherLaunch) -> Result<()> {
    let (kv, l) = (&g.kv, &g.lists.layout);
    ensure!(
        kv.head_dim == HEAD_DIM,
        "retro gather: head_dim {} (served: {HEAD_DIM})",
        kv.head_dim
    );
    ensure!(
        kv.kv_heads > 0 && kv.kv_heads == g.out.kv_heads && kv.kv_heads == l.kv_heads,
        "retro gather: kv heads differ (pool {}, exact buffer {}, index {})",
        kv.kv_heads,
        g.out.kv_heads,
        l.kv_heads
    );
    ensure!(
        l.head_dim == HEAD_DIM,
        "retro gather: index head_dim {}",
        l.head_dim
    );
    ensure!(
        kv.block_size > 0 && g.max_blocks_per_seq > 0,
        "retro gather: empty block table"
    );
    ensure!(
        (1..=MAX_ROWS).contains(&g.rows),
        "retro gather: {} rows (1..={MAX_ROWS})",
        g.rows
    );
    ensure!(
        g.retrieval_cap <= MAX_RETRIEVAL_CAP,
        "retro gather: retrieval capacity {} over {MAX_RETRIEVAL_CAP}",
        g.retrieval_cap
    );
    ensure!(
        g.out.cap > 0 && l.cap_centroids > 0,
        "retro gather: zero capacity"
    );
    let stride = block_stride(kv);
    ensure!(
        stride >= min_block_stride(kv) && stride.is_multiple_of(16),
        "retro gather: block stride {stride} bytes (needs >= {} and a multiple of 16)",
        min_block_stride(kv)
    );
    let ptrs = [
        ("k_pool", kv.k_pool),
        ("v_pool", kv.v_pool),
        ("block_table", g.block_table),
        ("row_lens", g.row_lens),
        ("counts", g.counts),
        ("index header", g.lists.header),
        ("list_offsets", g.lists.list_offsets),
        ("list_tokens", g.lists.list_tokens),
        ("exact k", g.out.k),
        ("exact v", g.out.v),
        ("exact positions", g.out.pos),
        ("exact valid_len", g.out.valid),
        ("exact dropped", g.out.dropped),
    ];
    for (name, p) in ptrs {
        ensure!(!p.is_null(), "retro gather: {name} is NULL");
    }
    ensure!(
        g.retrieval_cap == 0 || !g.retrieval_ids.is_null(),
        "retro gather: ids are NULL"
    );
    for (name, p) in [
        ("k_pool", kv.k_pool),
        ("v_pool", kv.v_pool),
        ("exact k", g.out.k),
        ("exact v", g.out.v),
    ] {
        ensure!(
            p.0.is_multiple_of(16),
            "retro gather: {name} {p:?} is not 16-byte aligned"
        );
    }
    Ok(())
}

/// Enqueue one gather on `stream`. Capture-safe.
pub fn launch_gather(
    gpu: &dyn GpuBackend,
    k: &GatherKernels,
    g: &GatherLaunch,
    stream: u64,
) -> Result<()> {
    validate(g)?;
    let (kv, l) = (&g.kv, &g.lists);
    KernelLaunch::new(gpu, k.for_dtype(kv.dtype))
        .grid(gather_grid(kv.kv_heads, g.out.cap))
        .block([THREADS, 1, 1])
        .shared_mem(gather_smem(g.retrieval_cap))
        .arg_ptr(kv.k_pool)
        .arg_ptr(kv.v_pool)
        .arg_ptr(g.block_table)
        .arg_ptr(g.row_lens)
        .arg_u32(g.rows)
        .arg_u32(g.max_blocks_per_seq)
        .arg_u32(kv.block_size)
        .arg_u32(kv.kv_heads)
        .arg_u64(block_stride(kv))
        .arg_f32(kv.kv_scale)
        .arg_f32(kv.v_scale)
        .arg_ptr(g.retrieval_ids)
        .arg_ptr(g.counts)
        .arg_u32(g.retrieval_cap)
        .arg_ptr(l.header)
        .arg_ptr(l.list_offsets)
        .arg_ptr(l.list_tokens)
        .arg_u32(l.layout.cap_centroids)
        .arg_u32(l.layout.cap_tokens)
        .arg_ptr(g.out.k)
        .arg_ptr(g.out.v)
        .arg_ptr(g.out.pos)
        .arg_ptr(g.out.valid)
        .arg_ptr(g.out.dropped)
        .arg_u32(g.out.cap)
        .launch(stream)
}

/// Contract entry (master plan L34): the sink, the retrieval clusters' members and the tail
/// of ONE logical sequence into `exact`, BF16 or FP8-dequantized, in `state::ExactBuf`'s
/// order. `s` is the sequence's rows (one row, or the K+1 rows of a verify; the gather covers
/// the longest, and the attention trims per row with `ExactBuf::row_trim`). `zones` must be
/// bound to the index it was selected from (`ZoneLists::bind`). Graph-safe.
pub fn gather_exact(
    gpu: &dyn GpuBackend,
    kv: &KvLayerView,
    s: &SeqBatch,
    zones: &ZoneLists,
    exact: &mut ExactBuf,
    stream: u64,
) -> Result<()> {
    if !GatherKernels::available(gpu) {
        if gpu.op_cache().once("retro_gather_ada_absent") {
            tracing::warn!(
                "retro attention: module `{MODULE}` is not in this kernel image; RetroInfer \
                 steps fall back to dense attention"
            );
        }
        bail!("retro gather: module `{MODULE}` is not in this kernel image");
    }
    ensure!(
        s.n >= 1 && s.n <= s.rows_per_seq.max(1),
        "retro gather: {} rows with {} per sequence; the selector gathers one sequence per call",
        s.n,
        s.rows_per_seq
    );
    ensure!(
        !s.block_table.is_null() && !s.seq_lens.is_null(),
        "retro gather: decode and verify only (this site has no device block table)"
    );
    let Some(lists) = zones.lists else {
        bail!("retro gather: the zone lists are not bound to an index (ZoneLists::bind)");
    };
    let g = GatherLaunch {
        kv: *kv,
        block_table: s.block_table,
        row_lens: s.seq_lens,
        rows: u32::try_from(s.n)?,
        max_blocks_per_seq: s.max_blocks_per_seq,
        retrieval_ids: zones.retrieval_ids,
        counts: zones.counts,
        retrieval_cap: zones.retrieval_cap,
        lists,
        out: ExactOut::of(exact),
    };
    launch_gather(gpu, &GatherKernels::resolve(gpu)?, &g, stream)
}
