// SPDX-License-Identifier: AGPL-3.0-only
// Translated for Atlas from microsoft/RetrievalAttention (RetroInfer), MIT License,
// Copyright (c) Microsoft Corporation. Upstream: cache_hub/kmeans.py (segment_k_means: the pass
// structure, the normalized segmented passes and one plain global pass), commit 03f912c. Changes:
// the pass sequence drives Atlas's own CUDA kernels (`kernels/ada/common/retro_kmeans_ada.cu`)
// through `KernelLaunch`, with a stable CSR per pass instead of atomics, into caller offsets.

//! Responsibility: the device side of one index build: resolving the `retro_kmeans_ada` entry
//! points, the pointers and strides of the index sections a job writes and of its scratch, one
//! launcher per kernel (public so the GPU microtest can run each stage on its own against the CPU
//! mirror), and the full pass sequence.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::super::state::LayerIndex;
use super::geometry::{HEAD_DIM, Job, MEAN_CHUNKS, ScratchLayout, TILES, init_steps};

/// The kernel module (`kernels/ada/common/retro_kmeans_ada.cu`).
pub const MODULE: &str = "retro_kmeans_ada";

const ENTRIES: [&str; 8] = [
    "retro_km_mean_partial",
    "retro_km_mean_final",
    "retro_km_init",
    "retro_km_assign",
    "retro_km_count",
    "retro_km_scan",
    "retro_km_scatter",
    "retro_km_update",
];

/// Resolved entry points of [`MODULE`].
#[derive(Clone, Copy, Debug)]
pub struct Kernels {
    mean_partial: KernelHandle,
    mean_final: KernelHandle,
    init: KernelHandle,
    assign: KernelHandle,
    count: KernelHandle,
    scan: KernelHandle,
    scatter: KernelHandle,
    update: KernelHandle,
}

impl Kernels {
    /// Every entry point, or `None` when this target does not build the module (every target but
    /// Ada) or one entry point is missing. A target without the module issues no lookup.
    pub fn resolve(gpu: &dyn GpuBackend) -> Option<Self> {
        if !gpu.has_module(MODULE) {
            return None;
        }
        let mut h = [KernelHandle(0); 8];
        for (slot, entry) in h.iter_mut().zip(ENTRIES) {
            *slot = crate::layers::try_target_kernel(gpu, MODULE, entry);
            if slot.0 == 0 {
                return None;
            }
        }
        Some(Self {
            mean_partial: h[0],
            mean_final: h[1],
            init: h[2],
            assign: h[3],
            count: h[4],
            scan: h[5],
            scatter: h[6],
            update: h[7],
        })
    }
}

/// Launches one [`run`] of `iters` passes issues (the mock tests pin the sequence).
pub fn launches_per_build(iters: u32) -> usize {
    3 + 5 * iters as usize
}

/// Where a layer's keys and values live: the paged pools, a device block table of the sequence's
/// physical blocks in order, and the dequantization.
#[derive(Clone, Copy, Debug)]
pub struct PoolArgs {
    pub k_pool: DevicePtr,
    pub v_pool: DevicePtr,
    /// `[ceil(end / block_size)]` i32 physical block ids.
    pub block_table: DevicePtr,
    pub block_size: u32,
    pub block_stride_bytes: u64,
    pub kv_heads: u32,
    /// True for an FP8 E4M3 pool, false for BF16.
    pub fp8: bool,
    pub k_scale: f32,
    pub v_scale: f32,
}

/// The index sections one job writes, offset to its first cluster, with their per-head strides
/// (elements; a centroid row is 256 BF16).
#[derive(Clone, Copy, Debug)]
pub struct IndexOut {
    pub centroids: DevicePtr,
    pub value_sum: DevicePtr,
    pub cent_stride: u32,
    pub cluster_size: DevicePtr,
    pub size_stride: u32,
    pub list_offsets: DevicePtr,
    pub offsets_stride: u32,
    /// The head's list base (slots are absolute: `token_offset` onward).
    pub list_tokens: DevicePtr,
    pub tokens_stride: u32,
}

impl IndexOut {
    /// The sections of `idx` a job writes (`state.rs` layout).
    pub fn of(idx: &LayerIndex, job: &Job) -> Self {
        let (cs, os, ts) = idx.layout.head_strides();
        let co = job.centroid_offset as usize;
        let row = idx.layout.head_dim as usize * 2;
        Self {
            centroids: idx.centroids.offset(co * row),
            value_sum: idx.value_sum.offset(co * row),
            cent_stride: cs as u32,
            cluster_size: idx.cluster_size.offset(co * 4),
            size_stride: cs as u32,
            list_offsets: idx.list_offsets.offset(co * 4),
            offsets_stride: os as u32,
            list_tokens: idx.list_tokens,
            tokens_stride: ts as u32,
        }
    }
}

/// Build scratch pointers inside the caller's scratch buffer.
#[derive(Clone, Copy, Debug)]
pub struct Scratch {
    pub assign: DevicePtr,
    pub hist: DevicePtr,
    pub cstart: DevicePtr,
    pub csize: DevicePtr,
    pub members: DevicePtr,
    pub mean_part: DevicePtr,
    pub mean: DevicePtr,
    pub block_table: DevicePtr,
}

impl Scratch {
    pub fn at(base: DevicePtr, l: &ScratchLayout) -> Self {
        Self {
            assign: base.offset(l.assign),
            hist: base.offset(l.hist),
            cstart: base.offset(l.cstart),
            csize: base.offset(l.csize),
            members: base.offset(l.members),
            mean_part: base.offset(l.mean_part),
            mean: base.offset(l.mean),
            block_table: base.offset(l.block_table),
        }
    }
}

/// The CSR destinations of one pass: absolute starts (`base` onward), sizes, members.
#[derive(Clone, Copy, Debug)]
pub struct CsrOut {
    pub cstart: DevicePtr,
    pub start_stride: u32,
    pub csize: DevicePtr,
    pub size_stride: u32,
    pub members: DevicePtr,
    pub mem_stride: u32,
    pub base: i32,
}

impl CsrOut {
    /// A training pass's CSR, in scratch.
    pub fn training(s: &Scratch, job: &Job) -> Self {
        Self {
            cstart: s.cstart,
            start_stride: job.n_centroids + 1,
            csize: s.csize,
            size_stride: job.n_centroids,
            members: s.members,
            mem_stride: job.n_tokens,
            base: 0,
        }
    }

    /// The final pass's CSR, into the index.
    pub fn index(out: &IndexOut, job: &Job) -> Self {
        Self {
            cstart: out.list_offsets,
            start_stride: out.offsets_stride,
            csize: out.cluster_size,
            size_stride: out.size_stride,
            members: out.list_tokens,
            mem_stride: out.tokens_stride,
            base: i32::try_from(job.token_offset).unwrap_or(i32::MAX),
        }
    }
}

fn pool_args<'a>(l: KernelLaunch<'a>, pool: &PoolArgs) -> KernelLaunch<'a> {
    l.arg_ptr(pool.k_pool)
        .arg_ptr(pool.block_table)
        .arg_u32(pool.block_size)
        .arg_u64(pool.block_stride_bytes)
        .arg_u32(pool.kv_heads)
        .arg_u32(u32::from(pool.fp8))
        .arg_f32(pool.k_scale)
}

/// Key mean of the job's span (`s.mean`, BF16-rounded values as f32).
pub fn mean(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    pool: &PoolArgs,
    job: &Job,
    s: &Scratch,
    stream: u64,
) -> Result<()> {
    pool_args(KernelLaunch::new(gpu, k.mean_partial), pool)
        .grid([MEAN_CHUNKS, pool.kv_heads, 1])
        .block([HEAD_DIM, 1, 1])
        .arg_ptr(s.mean_part)
        .arg_u32(job.n_tokens)
        .arg_u32(job.first_pos)
        .arg_u32(job.n_tokens.div_ceil(MEAN_CHUNKS))
        .launch(stream)?;
    KernelLaunch::new(gpu, k.mean_final)
        .grid([pool.kv_heads, 1, 1])
        .block([HEAD_DIM, 1, 1])
        .arg_ptr(s.mean_part)
        .arg_ptr(s.mean)
        .arg_u32(job.n_tokens)
        .launch(stream)
}

/// Uniform-stride initialization of the job's centroids from the centered keys.
pub fn init(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    pool: &PoolArgs,
    job: &Job,
    out: &IndexOut,
    s: &Scratch,
    stream: u64,
) -> Result<()> {
    let (step, half) = init_steps(job.n_tokens, job.n_centroids);
    pool_args(KernelLaunch::new(gpu, k.init), pool)
        .grid([job.n_centroids.div_ceil(8), pool.kv_heads, 1])
        .block([256, 1, 1])
        .arg_ptr(s.mean)
        .arg_ptr(out.centroids)
        .arg_u32(out.cent_stride)
        .arg_u32(job.n_centroids)
        .arg_u32(job.n_tokens)
        .arg_u32(job.first_pos)
        .arg_f32(step)
        .arg_f32(half)
        .launch(stream)
}

/// One assignment pass. `segmented`: the training form (segments of `job.seg_tokens` tokens
/// against `job.seg_centroids` centroids); otherwise every token against every centroid.
pub fn assign(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    pool: &PoolArgs,
    job: &Job,
    out: &IndexOut,
    s: &Scratch,
    segmented: bool,
    stream: u64,
) -> Result<()> {
    let (seg_tokens, seg_cents, n_seg) = if segmented {
        (job.seg_tokens, job.seg_centroids, job.n_segments)
    } else {
        (job.n_tokens, job.n_centroids, 1)
    };
    pool_args(KernelLaunch::new(gpu, k.assign), pool)
        .grid([job.n_tokens.div_ceil(128), pool.kv_heads, 1])
        .block([256, 1, 1])
        .arg_ptr(s.mean)
        .arg_ptr(out.centroids)
        .arg_u32(out.cent_stride)
        .arg_ptr(s.assign)
        .arg_u32(job.n_tokens)
        .arg_u32(job.n_tokens)
        .arg_u32(job.first_pos)
        .arg_u32(seg_tokens)
        .arg_u32(seg_cents)
        .arg_u32(n_seg)
        .launch(stream)
}

/// Stable CSR of `s.assign` (count, scan, scatter): clusters in id order, members ascending.
pub fn csr(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    kv_heads: u32,
    job: &Job,
    s: &Scratch,
    out: CsrOut,
    stream: u64,
) -> Result<()> {
    let nc = job.n_centroids;
    let hist_bytes = kv_heads as usize * TILES as usize * nc as usize * 4;
    gpu.memset_async(s.hist, 0, hist_bytes, stream)?;
    KernelLaunch::new(gpu, k.count)
        .grid([job.n_tiles, kv_heads, 1])
        .block([256, 1, 1])
        .arg_ptr(s.assign)
        .arg_u32(job.n_tokens)
        .arg_ptr(s.hist)
        .arg_u32(nc)
        .arg_u32(job.n_tokens)
        .arg_u32(job.tile_tokens)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.scan)
        .grid([kv_heads, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(s.hist)
        .arg_u32(nc)
        .arg_ptr(out.cstart)
        .arg_u32(out.start_stride)
        .arg_ptr(out.csize)
        .arg_u32(out.size_stride)
        .arg_u32(nc)
        .arg_u32(job.n_tiles)
        .arg_i32(out.base)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.scatter)
        .grid([job.n_tiles, kv_heads, 1])
        .block([32, 1, 1])
        .arg_ptr(s.assign)
        .arg_u32(job.n_tokens)
        .arg_ptr(s.hist)
        .arg_u32(nc)
        .arg_ptr(out.members)
        .arg_u32(out.mem_stride)
        .arg_u32(job.n_tokens)
        .arg_u32(job.tile_tokens)
        .arg_u32(job.first_pos)
        .launch(stream)
}

/// Centroid update from a CSR. `final_pass`: plain means plus the key mean, and the value sums.
pub fn update(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    pool: &PoolArgs,
    job: &Job,
    out: &IndexOut,
    s: &Scratch,
    csr: CsrOut,
    final_pass: bool,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k.update)
        .grid([job.n_centroids.div_ceil(8), pool.kv_heads, 1])
        .block([256, 1, 1])
        .arg_ptr(pool.k_pool)
        .arg_ptr(pool.v_pool)
        .arg_ptr(pool.block_table)
        .arg_u32(pool.block_size)
        .arg_u64(pool.block_stride_bytes)
        .arg_u32(pool.kv_heads)
        .arg_u32(u32::from(pool.fp8))
        .arg_f32(pool.k_scale)
        .arg_f32(pool.v_scale)
        .arg_ptr(s.mean)
        .arg_ptr(out.centroids)
        .arg_ptr(out.value_sum)
        .arg_u32(out.cent_stride)
        .arg_ptr(csr.cstart)
        .arg_u32(csr.start_stride)
        .arg_ptr(csr.csize)
        .arg_u32(csr.size_stride)
        .arg_ptr(csr.members)
        .arg_u32(csr.mem_stride)
        .arg_u32(job.n_centroids)
        .arg_u32(u32::from(final_pass))
        .launch(stream)
}

/// The whole build once the block table is on the device: mean, init, `iters - 1` normalized
/// segmented passes, then the final global pass into the index.
pub fn run(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    pool: &PoolArgs,
    job: &Job,
    out: &IndexOut,
    s: &Scratch,
    stream: u64,
) -> Result<()> {
    if job.n_tiles > TILES {
        bail!(
            "retro build: {} CSR tiles exceed the kernel's {TILES}",
            job.n_tiles
        );
    }
    mean(gpu, k, pool, job, s, stream)?;
    init(gpu, k, pool, job, out, s, stream)?;
    let train = CsrOut::training(s, job);
    let fin = CsrOut::index(out, job);
    for it in 0..job.iters {
        let last = it + 1 == job.iters;
        let csr_out = if last { fin } else { train };
        assign(gpu, k, pool, job, out, s, !last, stream)?;
        csr(gpu, k, pool.kv_heads, job, s, csr_out, stream)?;
        update(gpu, k, pool, job, out, s, csr_out, last, stream)?;
    }
    Ok(())
}
