// SPDX-License-Identifier: AGPL-3.0-only
// Translated for Atlas from microsoft/RetrievalAttention (RetroInfer), MIT License,
// Copyright (c) Microsoft Corporation. Upstream: cache_hub/kmeans.py (segment_k_means) and
// cache_hub/retroinfer_cache.py:505-561 (the prefill-time index build) and :595-711 (the decode-time
// update that clusters new tokens and appends them), commit 03f912c. Changes: the index is built on
// the device from Atlas's paged KV pool, into the host's fixed-capacity index at the host's
// offsets, deterministically; see `kernels/ada/common/retro_kmeans_ada.cu` for the kernel changes.

//! Responsibility: the RetroInfer index build of one attention layer (lane L34a): the public entry
//! point `build_layer_index` the host side (L33, `device.rs`) calls for the initial build at the
//! end of a prefill and for every decode-time update, and the glue that turns its `PendingBuild`,
//! `KvLayerView` and `SeqBatch` into a k-means build on the GPU.
//!
//! # Contract (`state.rs`, `PendingBuild` and `LayerIndex`)
//! Per KV head, the keys at positions `[first_pos, first_pos + tokens)` are clustered into
//! `min(centroids, tokens)` clusters with `min(segments, clusters)` segments and `iters` passes,
//! written at cluster index `centroid_offset`: BF16 centroids (the member mean with the centering
//! mean added back), BF16 value sums, i32 sizes, and the CSR (`list_offsets[centroid_offset] ==
//! token_offset`, the end offset after the last cluster, members ascending in `list_tokens`).
//! Nothing below the offsets is touched (an update appends), the header is the host's, the only
//! memory used is the pending build's scratch, and nothing synchronizes: the host does after it.
//!
//! # Determinism
//! Two builds of the same KV are bitwise equal: no float atomics, fixed reduction orders, exact
//! tie-breaks. The CPU mirror (`reference`) reproduces every stage bit for bit except the
//! tensor-core dot products of the assignment, where a near tie may resolve the other way. The
//! semantic reference is L33's `oracle.rs` (f64, no BF16 rounding); the GPU microtest
//! (`examples/retro_build_microtest.rs`) compares with both.
//!
//! # Off is free
//! Nothing here runs unless the `retro_attn` row is on and the host side calls it. The kernels are
//! looked up lazily through `try_target_kernel`; a target without `retro_kmeans_ada` (every target
//! but Ada) issues no lookup, `available` answers false, and a build returns an error naming the
//! missing module, which the host side turns into its dense fallback.

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::qwen3_attention::sparse_seam::{KvDtypeTag, KvLayerView, SeqBatch};

use super::config::RetroCfg;
use super::state::{IndexLayout, LayerIndex};

pub mod geometry;
#[cfg(test)]
mod kernel_source_tests;
pub mod launch;
pub mod reference;
#[cfg(test)]
mod tests;

pub use geometry::{Job, ScratchLayout};

/// True when this target built `retro_kmeans_ada` with every entry point (Ada only).
pub fn available(gpu: &dyn GpuBackend) -> bool {
    launch::Kernels::resolve(gpu).is_some()
}

/// Scratch bytes a build of `tokens` keys into `centroids` clusters over `kv_heads` heads uses,
/// plus room for a `bt_blocks` block table when the hook passes no device table.
/// `RetroCfg::build_scratch_bytes` is at least this (`build/tests.rs` checks it).
pub fn scratch_bytes(kv_heads: u32, tokens: u32, centroids: u32, bt_blocks: u32) -> Result<usize> {
    let job = Job::new(0, tokens, centroids, 1, 1, 0, 0)?;
    Ok(geometry::scratch_layout(kv_heads, &job, bt_blocks).total)
}

/// Run the index's pending build (`idx.pending`, set by the host) over the one sequence of `s`.
/// Eager only, never inside a graph capture; asynchronous on `stream`.
pub fn build_layer_index(
    gpu: &dyn GpuBackend,
    kv: &KvLayerView,
    s: &SeqBatch,
    cfg: &RetroCfg,
    idx: &mut LayerIndex,
    stream: u64,
) -> Result<()> {
    let pending = idx.pending.context(
        "retro build: the index has no pending build (the host sets state::PendingBuild)",
    )?;
    ensure!(
        cfg.kv_heads == kv.kv_heads && cfg.head_dim == kv.head_dim,
        "retro build: the config is for {} KV heads of {}, the layer has {} of {}",
        cfg.kv_heads,
        cfg.head_dim,
        kv.kv_heads,
        kv.head_dim
    );
    let job = Job::from_pending(&pending)?;
    build_job(
        gpu,
        kv,
        s,
        idx,
        &job,
        pending.scratch,
        pending.scratch_bytes,
        stream,
    )
}

/// [`build_layer_index`] with an explicit job and scratch (the tests and the microtest call it).
pub fn build_job(
    gpu: &dyn GpuBackend,
    kv: &KvLayerView,
    s: &SeqBatch,
    idx: &LayerIndex,
    job: &Job,
    scratch: DevicePtr,
    scratch_bytes: usize,
    stream: u64,
) -> Result<()> {
    check_layout(kv, &idx.layout)?;
    ensure!(
        u64::from(job.centroid_offset) + u64::from(job.n_centroids)
            <= u64::from(idx.layout.cap_centroids)
            && u64::from(job.token_offset) + u64::from(job.n_tokens)
                <= u64::from(idx.layout.cap_tokens),
        "retro build: clusters [{}, +{}) and list slots [{}, +{}) exceed the index capacities {} and {}",
        job.centroid_offset,
        job.n_centroids,
        job.token_offset,
        job.n_tokens,
        idx.layout.cap_centroids,
        idx.layout.cap_tokens
    );
    ensure!(
        !gpu.stream_is_capturing(stream),
        "retro build: called inside a CUDA graph capture (builds run on eager steps)"
    );
    let rows = s.rows_per_seq.max(1);
    ensure!(
        s.n >= 1 && s.n / rows == 1,
        "retro build: one sequence per call ({} rows of {rows} given)",
        s.n
    );
    let len = *s
        .host_seq_lens
        .first()
        .context("retro build: the hook gave no sequence length")?;
    ensure!(
        job.end_pos() <= len,
        "retro build: positions [{}, {}) reach past the sequence's {len} tokens",
        job.first_pos,
        job.end_pos()
    );
    let blocks = job.end_pos().div_ceil(kv.block_size);
    let device_table = !s.block_table.is_null();
    let lay = geometry::scratch_layout(kv.kv_heads, job, if device_table { 0 } else { blocks });
    ensure!(
        !scratch.is_null() && scratch_bytes >= lay.total,
        "retro build: {scratch_bytes} bytes of build scratch, the job needs {}",
        lay.total
    );
    let Some(k) = launch::Kernels::resolve(gpu) else {
        bail!(
            "retro build: kernel module {} is not built for this target",
            launch::MODULE
        );
    };
    let sc = launch::Scratch::at(scratch, &lay);
    let table = if device_table {
        s.block_table
    } else {
        let host = s
            .host_block_tables
            .first()
            .filter(|t| t.len() >= blocks as usize)
            .context("retro build: the hook gave neither a device block table nor a long enough host one")?;
        let bytes: Vec<u8> = host[..blocks as usize]
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect();
        gpu.copy_h2d_async(&bytes, sc.block_table, stream)?;
        sc.block_table
    };
    let pool = pool_args(kv, table);
    launch::run(
        gpu,
        &k,
        &pool,
        job,
        &launch::IndexOut::of(idx, job),
        &sc,
        stream,
    )
}

fn check_layout(kv: &KvLayerView, l: &IndexLayout) -> Result<()> {
    ensure!(
        kv.head_dim == geometry::HEAD_DIM && l.head_dim == geometry::HEAD_DIM,
        "retro build: head_dim {} (index {}); the kernels serve {} only",
        kv.head_dim,
        l.head_dim,
        geometry::HEAD_DIM
    );
    ensure!(
        kv.kv_heads == l.kv_heads && kv.kv_heads > 0,
        "retro build: the layer has {} KV heads, the index {}",
        kv.kv_heads,
        l.kv_heads
    );
    ensure!(kv.block_size > 0, "retro build: block_size 0");
    let dense = u64::from(kv.block_size)
        * u64::from(kv.kv_heads)
        * u64::from(kv.head_dim)
        * kv.dtype.bytes_per_elem() as u64;
    ensure!(
        kv.block_stride_bytes == 0 || kv.block_stride_bytes >= dense,
        "retro build: block stride {} bytes is shorter than one block ({dense})",
        kv.block_stride_bytes
    );
    Ok(())
}

/// The pool arguments of a layer view: the seam's block stride, or the dense one when 0.
pub fn pool_args(kv: &KvLayerView, block_table: DevicePtr) -> launch::PoolArgs {
    let fp8 = kv.dtype == KvDtypeTag::Fp8;
    let dense = u64::from(kv.block_size)
        * u64::from(kv.kv_heads)
        * u64::from(kv.head_dim)
        * kv.dtype.bytes_per_elem() as u64;
    launch::PoolArgs {
        k_pool: kv.k_pool,
        v_pool: kv.v_pool,
        block_table,
        block_size: kv.block_size,
        block_stride_bytes: if kv.block_stride_bytes == 0 {
            dense
        } else {
            kv.block_stride_bytes
        },
        kv_heads: kv.kv_heads,
        fp8,
        k_scale: if fp8 { kv.kv_scale } else { 1.0 },
        v_scale: if fp8 { kv.v_scale } else { 1.0 },
    }
}
