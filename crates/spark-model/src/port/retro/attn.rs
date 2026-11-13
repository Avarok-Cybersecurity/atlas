// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm: RetroInfer, arXiv 2505.02922 (VLDB 2026), Section 4.2, Eq 2-5.
// Translated for Atlas from microsoft/RetrievalAttention (RetroInfer), MIT License,
// Copyright (c) Microsoft Corporation. Upstream: cache_hub/retroinfer_cache.py:771-815
// (estimation-zone weighted attention merged with the exact zone). Changes: one single-pass
// union softmax instead of two attention calls and an LSE merge, the index read in place
// through the estimation id list, per-row trims of a shared exact buffer, host launch
// plumbing in Rust.

//! Responsibility: the RetroInfer estimation attention (port lane L34b): the host ABI of
//! `kernels/ada/common/retro_attn_ada.cu` (module, the five group entries, geometry, argument
//! order) and the contract entry point [`estimation_attention`] that L33's selector calls
//! after [`super::gather::gather_exact`], plus [`estimation_attention_rows`], which runs the
//! K+1 rows of a verify in one launch. The f64 CPU reference on the same layouts is the child
//! module `attn_reference.rs`, re-exported here:
//!
//! ```text
//! m   = max( max_i s_i , max_c s_c )          s = (q . key) * scale
//! num = sum_i exp(s_i - m) v_i  +  sum_c exp(s_c - m) value_sum_c
//! den = sum_i exp(s_i - m)      +  sum_c n_c exp(s_c - m)
//! out = num / den,  lse = m + ln(den)         (zeros and -inf when nothing is attended)
//! ```
//!
//! over the exact entries `[0, valid_len[h] - trim)` (positions >= 0) and the estimation
//! clusters (size > 0). The launch is capture-safe (no allocation, synchronization or host
//! read of a device value) and deterministic (fixed per-warp partition and merge tree, no
//! atomics). The module is compiled only by `kernels/ada`; elsewhere
//! [`AttnKernels::available`] answers false.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::gather::{ExactOut, HEAD_DIM, MAX_ROWS};
use super::state::{ExactBuf, LayerIndex, ZoneLists};

#[path = "attn_reference.rs"]
mod reference;
pub use reference::{HostAttnIn, bf16_f32, reference_attention};

/// Module the kernel file compiles to (its file stem).
pub const MODULE: &str = "retro_attn_ada";
/// Query heads per kv head the kernel file instantiates.
pub const GROUPS: [u32; 5] = [1, 2, 4, 6, 8];
/// Threads per CTA (eight warps); mirrors `RA_THREADS`.
pub const THREADS: u32 = 256;
/// Static shared memory of one CTA: `smem_o[8][256]` f32 plus the running max and sum.
pub const STATIC_SMEM_BYTES: u32 = 8 * 256 * 4 + 2 * 8 * 4;

/// The entry for `group` query heads per kv head.
pub fn entry_name(group: u32) -> Option<&'static str> {
    Some(match group {
        1 => "retro_est_attn_g1",
        2 => "retro_est_attn_g2",
        4 => "retro_est_attn_g4",
        6 => "retro_est_attn_g6",
        8 => "retro_est_attn_g8",
        _ => return None,
    })
}

/// The index sections the attention reads (BF16 centroids and value sums, i32 sizes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexView {
    pub centroids: DevicePtr,
    pub value_sum: DevicePtr,
    pub cluster_size: DevicePtr,
    pub cap_centroids: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
}

impl IndexView {
    pub fn of(index: &LayerIndex) -> Self {
        Self {
            centroids: index.centroids,
            value_sum: index.value_sum,
            cluster_size: index.cluster_size,
            cap_centroids: index.layout.cap_centroids,
            kv_heads: index.layout.kv_heads,
            head_dim: index.layout.head_dim,
        }
    }
}

/// Everything one attention launch reads and writes.
#[derive(Clone, Copy, Debug)]
pub struct AttnLaunch {
    /// BF16 `[rows][q_row_stride]` post-RoPE queries.
    pub q: DevicePtr,
    pub q_row_stride: u32,
    pub rows: u32,
    /// Entries the LAST row drops from the end of every head's buffer; row `r` drops
    /// `row_trim + rows - 1 - r`.
    pub row_trim: u32,
    pub q_heads: u32,
    pub scale: f32,
    pub exact: ExactOut,
    /// i32 `[kv_heads][es_cap]` estimation ids, i32 `[kv_heads][2]` counts.
    pub est_ids: DevicePtr,
    pub counts: DevicePtr,
    pub es_cap: u32,
    pub index: IndexView,
    /// BF16 `[rows][q_heads][256]`.
    pub out: DevicePtr,
    /// F32 `[rows][q_heads]`, NULL for none.
    pub lse: DevicePtr,
}

/// The resolved entry of one GQA group.
#[derive(Clone, Copy, Debug)]
pub struct AttnKernels {
    pub group: u32,
    pub entry: KernelHandle,
}

impl AttnKernels {
    /// True when the module is compiled into this image. Issues no kernel lookup.
    pub fn available(gpu: &dyn GpuBackend) -> bool {
        gpu.has_module(MODULE)
    }

    /// Resolve the entry for `group` query heads per kv head (memoized per backend).
    pub fn resolve(gpu: &dyn GpuBackend, group: u32) -> Result<Self> {
        ensure!(
            Self::available(gpu),
            "retro attention: module `{MODULE}` is not in this kernel image (RetroInfer runs \
             only on a target that compiles kernels/ada/common/retro_attn_ada.cu; build with \
             AVAROK_TARGET_HW=ada)"
        );
        let Some(name) = entry_name(group) else {
            bail!("retro attention: no kernel for GQA group {group} (served: {GROUPS:?})");
        };
        Ok(Self {
            group,
            entry: gpu.op_cache().kernel(gpu, MODULE, name)?,
        })
    }
}

/// Query heads per kv head of a launch.
pub fn group_of(a: &AttnLaunch) -> Result<u32> {
    let kvh = a.exact.kv_heads;
    ensure!(
        kvh > 0 && a.q_heads > 0 && a.q_heads.is_multiple_of(kvh),
        "retro attention: {} query heads over {kvh} kv heads (ExactBuf::num_q_heads unset?)",
        a.q_heads
    );
    let g = a.q_heads / kvh;
    ensure!(
        GROUPS.contains(&g),
        "retro attention: GQA group {g} (served: {GROUPS:?})"
    );
    Ok(g)
}

/// Refuse a launch the kernel cannot serve. Host values only.
pub fn validate(a: &AttnLaunch) -> Result<()> {
    let ix = &a.index;
    ensure!(
        ix.head_dim == HEAD_DIM,
        "retro attention: head_dim {} (served: {HEAD_DIM})",
        ix.head_dim
    );
    ensure!(
        ix.kv_heads == a.exact.kv_heads,
        "retro attention: index for {} kv heads, exact buffer for {}",
        ix.kv_heads,
        a.exact.kv_heads
    );
    group_of(a)?;
    ensure!(
        (1..=MAX_ROWS).contains(&a.rows),
        "retro attention: {} rows (1..={MAX_ROWS})",
        a.rows
    );
    ensure!(
        a.q_row_stride >= a.q_heads * HEAD_DIM && (a.q_row_stride * 2).is_multiple_of(16),
        "retro attention: Q row stride {} (needs >= {} and 16-byte rows)",
        a.q_row_stride,
        a.q_heads * HEAD_DIM
    );
    ensure!(
        a.scale.is_finite() && a.scale > 0.0,
        "retro attention: scale {}",
        a.scale
    );
    ensure!(
        a.exact.cap > 0 && ix.cap_centroids > 0,
        "retro attention: zero capacity"
    );
    let ptrs = [
        ("q", a.q),
        ("exact k", a.exact.k),
        ("exact v", a.exact.v),
        ("exact positions", a.exact.pos),
        ("exact valid_len", a.exact.valid),
        ("counts", a.counts),
        ("centroids", ix.centroids),
        ("value_sum", ix.value_sum),
        ("cluster_size", ix.cluster_size),
        ("out", a.out),
    ];
    for (name, p) in ptrs {
        ensure!(!p.is_null(), "retro attention: {name} is NULL");
    }
    ensure!(
        a.es_cap == 0 || !a.est_ids.is_null(),
        "retro attention: estimation ids are NULL"
    );
    for (name, p) in [
        ("q", a.q),
        ("exact k", a.exact.k),
        ("exact v", a.exact.v),
        ("centroids", ix.centroids),
        ("value_sum", ix.value_sum),
        ("out", a.out),
    ] {
        ensure!(
            p.0.is_multiple_of(16),
            "retro attention: {name} {p:?} is not 16-byte aligned"
        );
    }
    Ok(())
}

/// Enqueue one estimation attention on `stream`. Capture-safe.
pub fn launch_attention(
    gpu: &dyn GpuBackend,
    k: &AttnKernels,
    a: &AttnLaunch,
    stream: u64,
) -> Result<()> {
    validate(a)?;
    let g = group_of(a)?;
    ensure!(
        g == k.group,
        "retro attention: kernel for group {}, launch needs {g}",
        k.group
    );
    KernelLaunch::new(gpu, k.entry)
        .grid([a.exact.kv_heads, a.rows, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(a.q)
        .arg_u32(a.q_row_stride)
        .arg_u32(a.rows)
        .arg_u32(a.row_trim)
        .arg_ptr(a.exact.k)
        .arg_ptr(a.exact.v)
        .arg_ptr(a.exact.pos)
        .arg_ptr(a.exact.valid)
        .arg_u32(a.exact.cap)
        .arg_ptr(a.est_ids)
        .arg_ptr(a.counts)
        .arg_u32(a.es_cap)
        .arg_ptr(a.index.centroids)
        .arg_ptr(a.index.value_sum)
        .arg_ptr(a.index.cluster_size)
        .arg_u32(a.index.cap_centroids)
        .arg_ptr(a.out)
        .arg_ptr(a.lse)
        .arg_u32(a.q_heads)
        .arg_u32(a.exact.kv_heads)
        .arg_f32(a.scale)
        .launch(stream)
}

/// The scale an exact buffer carries (`ExactBuf::inv_sqrt_d`, 0 meaning `1 / sqrt(256)`).
pub fn scale_of(exact: &ExactBuf) -> f32 {
    if exact.inv_sqrt_d > 0.0 {
        exact.inv_sqrt_d
    } else {
        1.0 / (HEAD_DIM as f32).sqrt()
    }
}

/// `rows` consecutive query rows (`q_row_stride` BF16 apart) through one launch: row `r`
/// trims `exact.row_trim + rows - 1 - r` entries, so the K+1 rows of a verify gathered once
/// attend causally with `row_trim = 0`. Output `[rows][num_q_heads][256]`, LSE
/// `[rows][num_q_heads]`.
#[allow(clippy::too_many_arguments)]
pub fn estimation_attention_rows(
    gpu: &dyn GpuBackend,
    q: DevicePtr,
    q_row_stride: u32,
    rows: u32,
    exact: &ExactBuf,
    index: &LayerIndex,
    zones: &ZoneLists,
    out: DevicePtr,
    lse: Option<DevicePtr>,
    stream: u64,
) -> Result<()> {
    if !AttnKernels::available(gpu) {
        if gpu.op_cache().once("retro_attn_ada_absent") {
            tracing::warn!(
                "retro attention: module `{MODULE}` is not in this kernel image; RetroInfer \
                 steps fall back to dense attention"
            );
        }
        bail!("retro attention: module `{MODULE}` is not in this kernel image");
    }
    let a = AttnLaunch {
        q,
        q_row_stride,
        rows,
        row_trim: exact.row_trim,
        q_heads: exact.num_q_heads,
        scale: scale_of(exact),
        exact: ExactOut::of(exact),
        est_ids: zones.estimation_ids,
        counts: zones.counts,
        es_cap: zones.es_cap,
        index: IndexView::of(index),
        out,
        lse: lse.unwrap_or(DevicePtr::NULL),
    };
    let k = AttnKernels::resolve(gpu, group_of(&a)?)?;
    launch_attention(gpu, &k, &a, stream)
}

/// Contract entry (master plan L34): the single-pass union softmax of report 74 R5 for ONE
/// query row `q` (`[num_q_heads][256]` BF16) over the exact buffer minus its last
/// `exact.row_trim` entries and the estimation zone of `zones`, writing BF16
/// `[num_q_heads][256]` to `out` and, when given, F32 `[num_q_heads]` LSE.
#[allow(clippy::too_many_arguments)]
pub fn estimation_attention(
    gpu: &dyn GpuBackend,
    q: DevicePtr,
    exact: &ExactBuf,
    index: &LayerIndex,
    zones: &ZoneLists,
    out: DevicePtr,
    lse: Option<DevicePtr>,
    stream: u64,
) -> Result<()> {
    let stride = exact.num_q_heads * HEAD_DIM;
    estimation_attention_rows(gpu, q, stride, 1, exact, index, zones, out, lse, stream)
}
