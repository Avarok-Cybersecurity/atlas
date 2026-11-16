// SPDX-License-Identifier: AGPL-3.0-only
// Translated for Atlas from microsoft/RetrievalAttention (RetroInfer), MIT License,
// Copyright (c) Microsoft Corporation. Upstream: cache_hub/retroinfer_cache.py:765-770 (the
// centroid softmax, the GQA group sum, the empty-cluster mask, torch.topk and the zone split),
// commit 03f912c; the CUTLASS batch_gemm_softmax is not copied. Changes: three graph-capturable
// CUDA kernels (`kernels/ada/common/retro_select_ada.cu`) with fixed capacities and device counts,
// an exact radix select and sort with a lowest-index tie-break, FP32 scores, a multi-row form.

//! Responsibility: decode-time zone selection of one layer (lane L34a): `select_zones`, called by
//! the host side's custom attention (L33, `attend.rs`) once per decode row, layer and sequence,
//! ranks every non-empty cluster of the index by its group-summed centroid softmax and writes,
//! per KV head, the first `nprobe` ids (the retrieval zone) and the next `es` (the estimation
//! zone), both in rank order, with their counts on the device (the `ZoneLists` contract in
//! `state.rs`). No host copy, no allocation and no synchronization: the three launches are safe
//! to capture into a CUDA graph, and the sizes come from the index header, so a replay serves
//! whatever index is current.
//!
//! The zone sizes are the header's, clamped to the non-empty clusters, the list capacities and
//! [`MAX_SELECTED`]. The scale is `RetroCfg::inv_sqrt_d`. With rows > 1 (`select_zones_rows`)
//! the softmax of every row's query heads is summed, so the rows share one selection.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::build::geometry::HEAD_DIM;
use super::config::RetroCfg;
use super::state::{LayerIndex, ZoneLists};

pub mod reference;
#[cfg(test)]
mod tests;

/// The kernel module (`kernels/ada/common/retro_select_ada.cu`).
pub const MODULE: &str = "retro_select_ada";
const ENTRIES: [&str; 3] = ["retro_sel_stats", "retro_sel_prob", "retro_sel_pick"];
/// Centroids per scoring CTA (`RS_TILE_C`).
pub const TILE_C: u32 = 128;
/// Query heads one selection sums over, rows times the GQA group (`RS_MAX_RG`).
pub const MAX_RG: u32 = 32;
/// Largest retrieval plus estimation zone the in-CTA sort holds (`RS_MAX_SELECTED`); at the
/// default budgets that is every context up to 262,144 tokens. A larger zone is cut to it (the
/// estimation zone loses its lowest-ranked clusters) and the counts say so.
pub const MAX_SELECTED: u32 = 4096;
/// Largest retrieval list the host allocates (`config::MAX_RETRIEVAL_CAP`, kept equal by a test);
/// any retrieval capacity up to [`MAX_SELECTED`] is served.
pub const MAX_RETRIEVAL: u32 = 1024;

/// Byte offsets of the selection scratch (the per-tile softmax stats).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchLayout {
    pub tiles_cap: u32,
    /// `[kv_heads][rg][tiles_cap]` f32 tile max, then the same for the tile sum of exponentials.
    pub tmax: usize,
    pub tsum: usize,
    pub total: usize,
}

/// The scratch layout for `rg` query heads per KV head over `cap_centroids` centroid slots.
pub fn scratch_layout(kv_heads: u32, rg: u32, cap_centroids: u32) -> ScratchLayout {
    let al = |x: usize| x.next_multiple_of(256);
    let tiles_cap = cap_centroids.div_ceil(TILE_C);
    let stats = kv_heads as usize * rg as usize * tiles_cap as usize * 4;
    let tmax = 0;
    let tsum = al(tmax + stats);
    ScratchLayout {
        tiles_cap,
        tmax,
        tsum,
        total: al(tsum + stats),
    }
}

/// Scratch bytes one selection of `rows` rows needs (`RetroCfg::select_scratch_bytes` is at least
/// this for one row; `select/tests.rs` checks it).
pub fn scratch_bytes(cfg: &RetroCfg, rows: u32) -> usize {
    scratch_layout(cfg.kv_heads, cfg.group * rows.max(1), cfg.cap_centroids).total
}

/// True when this target built `retro_select_ada` with every entry point (Ada only).
pub fn available(gpu: &dyn GpuBackend) -> bool {
    resolve(gpu).is_some()
}

fn resolve(gpu: &dyn GpuBackend) -> Option<[KernelHandle; 3]> {
    if !gpu.has_module(MODULE) {
        return None;
    }
    let mut h = [KernelHandle(0); 3];
    for (slot, entry) in h.iter_mut().zip(ENTRIES) {
        *slot = crate::layers::try_target_kernel(gpu, MODULE, entry);
        if slot.0 == 0 {
            return None;
        }
    }
    Some(h)
}

/// Select the zones of one decode row: `q` is the row's `[num_q_heads][256]` BF16 (post-RoPE).
/// Graph-safe (see the module docs).
pub fn select_zones(
    gpu: &dyn GpuBackend,
    idx: &LayerIndex,
    q: DevicePtr,
    cfg: &RetroCfg,
    zl: &mut ZoneLists,
    stream: u64,
) -> Result<()> {
    select_zones_rows(gpu, idx, q, 1, cfg.num_q_heads * HEAD_DIM, cfg, zl, stream)
}

/// [`select_zones`] over `rows` query rows `q_row_stride` BF16 elements apart, one selection
/// shared by all of them (the verify rows of one sequence).
pub fn select_zones_rows(
    gpu: &dyn GpuBackend,
    idx: &LayerIndex,
    q: DevicePtr,
    rows: u32,
    q_row_stride: u32,
    cfg: &RetroCfg,
    zl: &mut ZoneLists,
    stream: u64,
) -> Result<()> {
    let l = &idx.layout;
    ensure!(
        l.head_dim == HEAD_DIM,
        "retro select: head_dim {} unsupported",
        l.head_dim
    );
    ensure!(
        cfg.kv_heads == l.kv_heads
            && cfg.kv_heads > 0
            && cfg.group * cfg.kv_heads == cfg.num_q_heads,
        "retro select: the config ({} query heads over {} KV heads) does not match the index ({} KV heads)",
        cfg.num_q_heads,
        cfg.kv_heads,
        l.kv_heads
    );
    ensure!(
        idx.num_q_heads == 0 || idx.num_q_heads == cfg.num_q_heads,
        "retro select: the index was allocated for {} query heads, the config has {}",
        idx.num_q_heads,
        cfg.num_q_heads
    );
    ensure!(rows >= 1, "retro select: no query rows");
    let rg = rows * cfg.group;
    ensure!(
        rg <= MAX_RG,
        "retro select: {rows} rows x group {} exceed {MAX_RG} query heads",
        cfg.group
    );
    ensure!(
        rows == 1 || q_row_stride >= cfg.num_q_heads * HEAD_DIM,
        "retro select: row stride {q_row_stride} is shorter than a row"
    );
    let lay = scratch_layout(l.kv_heads, rg, l.cap_centroids);
    ensure!(
        !zl.scratch.is_null() && !zl.scores.is_null() && zl.scratch_bytes >= lay.total,
        "retro select: {} bytes of selection scratch, this call needs {} (and a scores buffer)",
        zl.scratch_bytes,
        lay.total
    );
    let Some([k_stats, k_prob, k_pick]) = resolve(gpu) else {
        bail!("retro select: kernel module {MODULE} is not built for this target");
    };
    let (tmax, tsum) = (zl.scratch.offset(lay.tmax), zl.scratch.offset(lay.tsum));
    let grid = [lay.tiles_cap, l.kv_heads, 1];
    KernelLaunch::new(gpu, k_stats)
        .grid(grid)
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_u32(q_row_stride)
        .arg_u32(rows)
        .arg_u32(cfg.group)
        .arg_ptr(idx.centroids)
        .arg_ptr(idx.cluster_size)
        .arg_u32(l.cap_centroids)
        .arg_ptr(idx.header)
        .arg_ptr(tmax)
        .arg_ptr(tsum)
        .arg_u32(lay.tiles_cap)
        .arg_f32(cfg.inv_sqrt_d)
        .launch(stream)?;
    KernelLaunch::new(gpu, k_prob)
        .grid(grid)
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_u32(q_row_stride)
        .arg_u32(rows)
        .arg_u32(cfg.group)
        .arg_ptr(idx.centroids)
        .arg_ptr(idx.cluster_size)
        .arg_u32(l.cap_centroids)
        .arg_ptr(idx.header)
        .arg_ptr(tmax)
        .arg_ptr(tsum)
        .arg_u32(lay.tiles_cap)
        .arg_f32(cfg.inv_sqrt_d)
        .arg_ptr(zl.scores)
        .launch(stream)?;
    KernelLaunch::new(gpu, k_pick)
        .grid([l.kv_heads, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(zl.scores)
        .arg_u32(l.cap_centroids)
        .arg_ptr(idx.header)
        .arg_ptr(zl.retrieval_ids)
        .arg_ptr(zl.estimation_ids)
        .arg_ptr(zl.counts)
        .arg_u32(zl.retrieval_cap)
        .arg_u32(zl.es_cap)
        .launch(stream)
}
