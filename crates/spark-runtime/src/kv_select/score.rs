// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: [`score_blocks`], the device launcher of the page-softmax scorer
//! (`kernels/ada/common/kv_page_score_ada.cu`): it validates the two views, plans the tiles
//! ([`super::ScorePlan`]), uploads the page list into the caller's scratch and enqueues the
//! merge, logits, stats, group-norm and accumulate kernels on `stream`. It never allocates,
//! never synchronizes and never reads a row; scores land in `out_scores` (`[n_blocks]` f32)
//! when the stream reaches them.
//!
//! The kernels exist only in the Ada image. On any other target [`kernels_available`] is
//! false and [`score_blocks`] refuses with one error (the bounded view is opt-in and its
//! wiring refuses to start without them), so no lookup is issued where nothing was built.

use anyhow::{Context, Result, bail};

use super::plan::{LOGITS_BLOCKS_PER_CTA, ScoreDims, ScorePlan};
use super::{GqaAgg, QueryView, ScoreCfg, SketchView};
use crate::gpu::{DevicePtr, GpuBackend, KernelHandle};
use crate::kernel_args::KernelLaunch;

/// The scorer's kernel module (the file stem of its `.cu`).
pub const KERNEL_MODULE: &str = "kv_page_score_ada";

const K_MERGE: &str = "kv_block_merge_f32";
const K_LOGITS: &str = "kv_block_logits_f32";
const K_STATS: &str = "kv_logit_stats_f32";
const K_GNORM: &str = "kv_group_norm_f32";
const K_ACCUM: &str = "kv_score_accumulate_f32";

/// Threads of the merge kernel's CTA (loops over `head_dim`).
const MERGE_THREADS: u32 = 128;
/// Threads of the logits CTA (8 warps, one view block each).
const LOGITS_THREADS: u32 = 256;
/// Threads of the reduction kernels and of the accumulate kernel.
const REDUCE_THREADS: u32 = 256;

/// True when this image carries the scorer kernels (the Ada image).
pub fn kernels_available(gpu: &dyn GpuBackend) -> bool {
    gpu.has_module(KERNEL_MODULE)
}

struct Kernels {
    merge: KernelHandle,
    logits: KernelHandle,
    stats: KernelHandle,
    gnorm: KernelHandle,
    accum: KernelHandle,
}

impl Kernels {
    fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        if !kernels_available(gpu) {
            bail!(
                "kv_select: module {KERNEL_MODULE} is not in this image (the bounded KV view \
                 needs the Ada kernels; build with AVAROK_TARGET_HW=ada)"
            );
        }
        let k = |name: &str| {
            gpu.kernel(KERNEL_MODULE, name)
                .with_context(|| format!("kv_select: {KERNEL_MODULE}::{name}"))
        };
        Ok(Self {
            merge: k(K_MERGE)?,
            logits: k(K_LOGITS)?,
            stats: k(K_STATS)?,
            gnorm: k(K_GNORM)?,
            accum: k(K_ACCUM)?,
        })
    }
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).with_context(|| format!("kv_select: {what} = {v} exceeds u32"))
}

/// The shapes of one call, checked against both views.
pub(crate) fn dims_of(
    sketch: &dyn SketchView,
    query: &dyn QueryView,
    pages_per_block: usize,
) -> Result<ScoreDims> {
    if sketch.layers() != query.layers() {
        bail!(
            "kv_select: sketch has {} attention layers, the query capture {}",
            sketch.layers(),
            query.layers()
        );
    }
    if sketch.head_dim() != query.head_dim() {
        bail!(
            "kv_select: sketch head_dim {} != query head_dim {}",
            sketch.head_dim(),
            query.head_dim()
        );
    }
    if query.rows() > query.layer_stride_rows() {
        bail!(
            "kv_select: {} query rows exceed the capture's {} rows per layer",
            query.rows(),
            query.layer_stride_rows()
        );
    }
    if let Some(bad) = sketch
        .pages()
        .iter()
        .find(|&&p| p as usize >= sketch.num_blocks())
    {
        bail!(
            "kv_select: page {bad} is outside the sketch's {} pages",
            sketch.num_blocks()
        );
    }
    let dims = ScoreDims {
        n_pages: sketch.pages().len(),
        pages_per_block,
        rows: query.rows(),
        layers: sketch.layers(),
        q_heads: query.q_heads(),
        kv_heads: sketch.kv_heads(),
        head_dim: sketch.head_dim(),
    };
    dims.validate()?;
    Ok(dims)
}

/// Score every view block of one sequence on `stream`.
///
/// `out_scores` is `[n_blocks]` f32 on the device, `n_blocks = ceil(pages / pages_per_block)`
/// in chronological order. The call is eager (it uploads the page list); it refuses a
/// capturing stream. Deterministic: equal inputs give equal bits.
pub fn score_blocks(
    gpu: &dyn GpuBackend,
    sketch: &dyn SketchView,
    query: &dyn QueryView,
    cfg: &ScoreCfg,
    out_scores: DevicePtr,
    stream: u64,
) -> Result<()> {
    if gpu.stream_is_capturing(stream) {
        bail!(
            "kv_select: score_blocks is eager-only (it uploads the page list); the stream is capturing"
        );
    }
    if cfg.scratch.is_null() || out_scores.is_null() {
        bail!("kv_select: score_blocks needs a scratch buffer and an output buffer");
    }
    let dims = dims_of(sketch, query, cfg.pages_per_block)?;
    let plan = ScorePlan::new(dims, cfg.scratch_bytes)?;
    let k = Kernels::resolve(gpu)?;
    let nb = plan.n_blocks();
    let base = cfg.scratch;
    let lay = plan.layout;

    let page_bytes: Vec<u8> = sketch
        .pages()
        .iter()
        .flat_map(|p| p.to_le_bytes())
        .collect();
    gpu.copy_h2d_async(&page_bytes, base.offset(lay.pages), stream)?;
    gpu.memset_async(out_scores, 0, nb * 4, stream)?;

    let heads_avg = match cfg.gqa {
        GqaAgg::Mean => dims.q_heads,
        GqaAgg::Max => dims.kv_heads,
    };
    let weight = 1.0f32 / (dims.layers * heads_avg) as f32;
    let scale = 1.0f32 / (dims.head_dim as f32).sqrt();
    let s = Shapes {
        n_pages: u32_of(dims.n_pages, "pages")?,
        ppb: u32_of(dims.pages_per_block, "pages_per_block")?,
        layers: u32_of(dims.layers, "layers")?,
        q_heads: u32_of(dims.q_heads, "q_heads")?,
        kv_heads: u32_of(dims.kv_heads, "kv_heads")?,
        head_dim: u32_of(dims.head_dim, "head_dim")?,
        n_blocks: u32_of(nb, "view blocks")?,
        rows_cap: u32_of(query.layer_stride_rows(), "query rows per layer")?,
    };
    let p = Ptrs {
        means: sketch.means(),
        counts: sketch.counts(),
        q: query.data(),
        pages: base.offset(lay.pages),
        merged: base.offset(lay.merged),
        wsum: base.offset(lay.wsum),
        logits: base.offset(lay.logits),
        stats: base.offset(lay.stats),
        gnorm: base.offset(lay.gnorm),
        scores: out_scores,
    };

    for layer in 0..dims.layers {
        let layer = layer as u32;
        if plan.n_chunks() == 1 {
            merge(gpu, &k, &s, &p, layer, plan.chunk(0), stream)?;
        }
        for t in 0..plan.n_row_tiles() {
            let (row0, rows_t) = plan.row_tile(t);
            for c in 0..plan.n_chunks() {
                if plan.n_chunks() > 1 {
                    merge(gpu, &k, &s, &p, layer, plan.chunk(c), stream)?;
                }
                let (c0, clen) = plan.chunk(c);
                let cta_rows = plan.cta_rows as u32;
                let group = dims.group() as u32;
                KernelLaunch::new(gpu, k.logits)
                    .grid([
                        clen.div_ceil(LOGITS_BLOCKS_PER_CTA) as u32,
                        s.kv_heads,
                        (rows_t as u32).div_ceil(cta_rows),
                    ])
                    .block([LOGITS_THREADS, 1, 1])
                    .shared_mem(cta_rows * group * s.head_dim * 4)
                    .arg_ptr(p.q)
                    .arg_u32(s.rows_cap)
                    .arg_u32(layer)
                    .arg_u32(row0 as u32)
                    .arg_u32(rows_t as u32)
                    .arg_u32(s.q_heads)
                    .arg_u32(s.kv_heads)
                    .arg_u32(s.head_dim)
                    .arg_u32(cta_rows)
                    .arg_ptr(p.merged)
                    .arg_ptr(p.wsum)
                    .arg_u32(c0 as u32)
                    .arg_u32(clen as u32)
                    .arg_u32(s.n_blocks)
                    .arg_ptr(p.logits)
                    .arg_f32(scale)
                    .launch(stream)?;
            }
            let rows_t = rows_t as u32;
            KernelLaunch::new(gpu, k.stats)
                .grid([rows_t * s.q_heads, 1, 1])
                .block([REDUCE_THREADS, 1, 1])
                .arg_ptr(p.logits)
                .arg_u32(s.n_blocks)
                .arg_ptr(p.stats)
                .launch(stream)?;
            let gnorm = if cfg.gqa == GqaAgg::Max {
                KernelLaunch::new(gpu, k.gnorm)
                    .grid([rows_t * s.kv_heads, 1, 1])
                    .block([REDUCE_THREADS, 1, 1])
                    .arg_ptr(p.logits)
                    .arg_ptr(p.stats)
                    .arg_u32(s.n_blocks)
                    .arg_u32(s.q_heads)
                    .arg_u32(s.kv_heads)
                    .arg_ptr(p.gnorm)
                    .launch(stream)?;
                p.gnorm
            } else {
                DevicePtr::NULL
            };
            KernelLaunch::new(gpu, k.accum)
                .grid([s.n_blocks.div_ceil(REDUCE_THREADS), 1, 1])
                .block([REDUCE_THREADS, 1, 1])
                .arg_ptr(p.logits)
                .arg_ptr(p.stats)
                .arg_ptr(gnorm)
                .arg_u32(rows_t)
                .arg_u32(s.q_heads)
                .arg_u32(s.kv_heads)
                .arg_u32(s.n_blocks)
                .arg_u32(cfg.gqa.mode())
                .arg_f32(weight)
                .arg_ptr(p.scores)
                .launch(stream)?;
        }
    }
    Ok(())
}

/// The scalar shapes every launch passes.
struct Shapes {
    n_pages: u32,
    ppb: u32,
    layers: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    n_blocks: u32,
    rows_cap: u32,
}

/// The device buffers every launch reads or writes.
struct Ptrs {
    means: DevicePtr,
    counts: DevicePtr,
    q: DevicePtr,
    pages: DevicePtr,
    merged: DevicePtr,
    wsum: DevicePtr,
    logits: DevicePtr,
    stats: DevicePtr,
    gnorm: DevicePtr,
    scores: DevicePtr,
}

/// Merge one chunk of view blocks of one layer into `merged` / `wsum`.
fn merge(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    s: &Shapes,
    p: &Ptrs,
    layer: u32,
    (c0, clen): (usize, usize),
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k.merge)
        .grid([clen as u32, s.kv_heads, 1])
        .block([MERGE_THREADS, 1, 1])
        .arg_ptr(p.means)
        .arg_ptr(p.counts)
        .arg_ptr(p.pages)
        .arg_u32(s.n_pages)
        .arg_u32(s.ppb)
        .arg_u32(layer)
        .arg_u32(s.layers)
        .arg_u32(s.kv_heads)
        .arg_u32(s.head_dim)
        .arg_u32(c0 as u32)
        .arg_ptr(p.merged)
        .arg_ptr(p.wsum)
        .launch(stream)
}
