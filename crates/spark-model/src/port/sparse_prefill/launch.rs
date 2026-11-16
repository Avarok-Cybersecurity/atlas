// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm after XAttention (antidiagonal block scoring for block-sparse prefill), described in its paper;
// independent implementation, no source copied (the reference repository has no detected license).

//! Responsibility: the host side of `kernels/ada/common/prefill_block_sparse_ada.cu`: the eligibility
//! verdict (which names the guard that declined), the launch geometry and scratch sizes, lazy kernel
//! resolution through the backend's `OpCache`, and the two launches (score + select, then block-sparse
//! attention), with the sampled read-back that feeds `/metrics`.
//!
//! Scratch is three grow-only `OpCache` buffers (per-CTA log-sum-exp and mass slots, the block masks,
//! three counters), a few MB at 256K context: it is not charged to the KV budget, which is why the
//! row's receipt includes a placement check at the longest context. One prefill chunk runs at a time per
//! model, the assumption the dense path's activation arena makes too.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::{BLOCK, Config, HEAD_DIMS, STRIDES, TILE_Q, metrics};

/// The module the kernel file compiles to (only `kernels/ada` builds it).
pub const MODULE: &str = "prefill_block_sparse_ada";

/// `ENTRY_SCORE[fp8][hd 128 | 256][stride 4 | 8 | 16]`.
pub const ENTRY_SCORE: [[[&str; 3]; 2]; 2] = [
    [
        [
            "prefill_sparse_score_ada_bf16_hd128_s4",
            "prefill_sparse_score_ada_bf16_hd128_s8",
            "prefill_sparse_score_ada_bf16_hd128_s16",
        ],
        [
            "prefill_sparse_score_ada_bf16_hd256_s4",
            "prefill_sparse_score_ada_bf16_hd256_s8",
            "prefill_sparse_score_ada_bf16_hd256_s16",
        ],
    ],
    [
        [
            "prefill_sparse_score_ada_fp8_hd128_s4",
            "prefill_sparse_score_ada_fp8_hd128_s8",
            "prefill_sparse_score_ada_fp8_hd128_s16",
        ],
        [
            "prefill_sparse_score_ada_fp8_hd256_s4",
            "prefill_sparse_score_ada_fp8_hd256_s8",
            "prefill_sparse_score_ada_fp8_hd256_s16",
        ],
    ],
];

/// `ENTRY_ATTN[fp8][hd 128 | 256]`.
pub const ENTRY_ATTN: [[&str; 2]; 2] = [
    [
        "prefill_sparse_attn_ada_bf16_hd128",
        "prefill_sparse_attn_ada_bf16_hd256",
    ],
    [
        "prefill_sparse_attn_ada_fp8_hd128",
        "prefill_sparse_attn_ada_fp8_hd256",
    ],
];

pub const SCORE_THREADS: u32 = 256;
pub const ATTN_THREADS: u32 = 128;
/// Shared-row padding, BF16 elements (the kernel's `psa::PAD`).
const PAD: u32 = 8;
/// Strided rows per block at stride 4 (the kernel's `psa::MAX_R`).
const MAX_R: u32 = 32;
/// The reference part's SM count, if the driver cannot be asked.
const REFERENCE_SM_COUNT: u32 = 34;
/// Every this many sparse layer calls, one waits for the stream and reads the kept-block counters.
pub const SAMPLE_EVERY: u64 = 8;

/// `sizeof(psa::ScoreTiles<hd>)`: the strided query block plus row normalizers and reduction slots.
pub fn score_smem(hd: u32) -> u32 {
    BLOCK * (hd + PAD) * 2 + MAX_R * 4 + (SCORE_THREADS / 32) * 4 + 2 * 4
}

/// Key rows per attention sub-tile (`psa::AttnBc`).
pub fn attn_bc(hd: u32) -> u32 {
    if hd >= 256 { 32 } else { 64 }
}

/// `sizeof(psa::AttnTiles<hd>)`: Q tile plus one K and one V sub-tile, BF16.
pub fn attn_smem(hd: u32) -> u32 {
    (TILE_Q + 2 * attn_bc(hd)) * (hd + PAD) * 2
}

/// Scoring CTAs resident per SM (the tile is 67.8 KB at hd 256, 34.8 KB at 128).
pub fn score_ctas_per_sm(hd: u32) -> u32 {
    if hd >= 256 { 1 } else { 2 }
}

/// The KV cache dtype of the layer, as far as this path is concerned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvKind {
    Bf16,
    /// FP8 E4M3 with per-tensor K and V scales.
    Fp8,
    /// Anything else (NVFP4, TurboQuant): the dense kernels serve it.
    Other,
}

/// Everything the two launches need, as plain values (the layer bridge fills it).
#[derive(Clone, Copy, Debug)]
pub struct Call {
    /// `[q_len][nq][hd]` BF16, post-RoPE.
    pub q: DevicePtr,
    /// `[q_len][nq][hd]` BF16, written by the attention launch.
    pub out: DevicePtr,
    pub k_pool: DevicePtr,
    pub v_pool: DevicePtr,
    /// This sequence's block table, one u32 per KV block.
    pub block_table: DevicePtr,
    pub q_len: u32,
    pub kv_len: u32,
    pub q_offset: u32,
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub block_size: u32,
    /// Elements per physical KV block (`PagedKvCache::cache_stride`).
    pub cache_stride: u64,
    pub kv: KvKind,
    pub k_scale: f32,
    pub v_scale: f32,
    pub inv_sqrt_d: f32,
    pub sliding_window: u32,
    pub graph_capture: bool,
    /// A sparse-attention seam selector is registered (L29): its prefill View may already have
    /// compacted this chunk's extent, and the two approximations are not stacked.
    pub seam_active: bool,
    pub layer: usize,
}

/// Why a chunk keeps the dense kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decline {
    /// The row is off (the byte-identical default).
    Off,
    /// The threshold keeps every block: the dense kernel is the identity.
    KeepAll,
    /// The chunk attends to fewer than `min_tokens` keys.
    Short,
    /// Inside CUDA graph capture.
    Capture,
    /// A sparse-attention seam selector is active (two approximations are never stacked).
    Seam,
    /// No queries, or a key extent shorter than the chunk.
    Empty,
    KvDtype,
    HeadDim,
    SlidingWindow,
    Gqa,
    /// The module is not in this image, or an entry did not resolve.
    Absent,
}

impl Decline {
    /// One-shot log key (per backend) for the declines worth one line.
    fn log_key(self) -> Option<&'static str> {
        match self {
            Decline::Off | Decline::Short | Decline::KeepAll | Decline::Empty => None,
            Decline::Capture => Some("sparse_prefill:decline:capture"),
            Decline::Seam => Some("sparse_prefill:decline:seam"),
            Decline::KvDtype => Some("sparse_prefill:decline:kv_dtype"),
            Decline::HeadDim => Some("sparse_prefill:decline:head_dim"),
            Decline::SlidingWindow => Some("sparse_prefill:decline:window"),
            Decline::Gqa => Some("sparse_prefill:decline:gqa"),
            Decline::Absent => Some("sparse_prefill:decline:absent"),
        }
    }
}

/// The verdict for one chunk, before any GPU call. `Off` is checked first, so with the row off this is
/// one branch.
pub fn eligibility(cfg: &Config, c: &Call) -> Result<(), Decline> {
    if !cfg.enabled {
        return Err(Decline::Off);
    }
    if cfg.keeps_all() {
        return Err(Decline::KeepAll);
    }
    if c.q_len == 0 || (c.kv_len as u64) < c.q_offset as u64 + c.q_len as u64 {
        return Err(Decline::Empty);
    }
    if c.kv_len < cfg.min_tokens {
        return Err(Decline::Short);
    }
    if c.graph_capture {
        return Err(Decline::Capture);
    }
    if c.seam_active {
        return Err(Decline::Seam);
    }
    if c.kv == KvKind::Other {
        return Err(Decline::KvDtype);
    }
    if !HEAD_DIMS.contains(&c.hd) {
        return Err(Decline::HeadDim);
    }
    if c.sliding_window != 0 {
        return Err(Decline::SlidingWindow);
    }
    if c.nkv == 0 || c.nq == 0 || !c.nq.is_multiple_of(c.nkv) || c.block_size == 0 {
        return Err(Decline::Gqa);
    }
    Ok(())
}

/// Launch geometry of one chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    /// First query block of the chunk (absolute).
    pub q_block_lo: u32,
    /// Query blocks the chunk touches.
    pub n_qb: u32,
    /// Key blocks of the attended extent.
    pub nkb: u32,
    /// u32 words per (head, query block) mask.
    pub mask_words: u32,
    /// (head, query block) items of the scoring stage.
    pub items: u32,
    /// Attention CTAs along x (64-row tiles from `floor(q_offset / 64) * 64`).
    pub n_tiles: u32,
}

impl Geometry {
    pub fn new(c: &Call) -> Self {
        let q_end = c.q_offset + c.q_len;
        let q_block_lo = c.q_offset / BLOCK;
        let n_qb = (q_end - 1) / BLOCK - q_block_lo + 1;
        let nkb = c.kv_len.div_ceil(BLOCK);
        let tile0 = c.q_offset / TILE_Q * TILE_Q;
        Self {
            q_block_lo,
            n_qb,
            nkb,
            mask_words: nkb.div_ceil(32),
            items: n_qb * c.nq,
            n_tiles: (q_end - tile0).div_ceil(TILE_Q),
        }
    }

    /// Scoring grid: persistent CTAs, at most the resident count.
    pub fn score_grid(&self, sm_count: u32, hd: u32) -> u32 {
        self.items
            .min(sm_count.max(1) * score_ctas_per_sm(hd))
            .max(1)
    }

    /// Bytes of the per-CTA scratch slots: `(R + 1) * nkb` FP32 each (log-sum-exp rows, then masses).
    pub fn score_scratch_bytes(&self, grid: u32, stride: u32) -> usize {
        grid as usize * (BLOCK / stride + 1) as usize * self.nkb as usize * 4
    }

    /// Bytes of the block masks.
    pub fn mask_bytes(&self) -> usize {
        self.items as usize * self.mask_words as usize * 4
    }
}

/// Bytes of the counters: kept blocks, candidate blocks, items (u64 each).
pub const STATS_BYTES: usize = 24;

/// The two resolved entries for one (dtype, head_dim, stride).
#[derive(Clone, Copy, Debug)]
pub struct Kernels {
    pub score: KernelHandle,
    pub attn: KernelHandle,
}

/// The (score, attention) entry names for a dtype, head width and stride; `None` outside the compiled
/// set.
pub fn entry_names(kv: KvKind, hd: u32, stride: u32) -> Option<(&'static str, &'static str)> {
    let f = match kv {
        KvKind::Bf16 => 0,
        KvKind::Fp8 => 1,
        KvKind::Other => return None,
    };
    let d = HEAD_DIMS.iter().position(|&x| x == hd)?;
    let s = STRIDES.iter().position(|&x| x == stride)?;
    Some((ENTRY_SCORE[f][d][s], ENTRY_ATTN[f][d]))
}

/// Resolve the entries through the backend's `OpCache`; no lookup at all where the module is absent.
pub fn resolve(gpu: &dyn GpuBackend, kv: KvKind, hd: u32, stride: u32) -> Option<Kernels> {
    if !gpu.has_module(MODULE) {
        return None;
    }
    let (score, attn) = entry_names(kv, hd, stride)?;
    let cache = gpu.op_cache();
    match (
        cache.kernel(gpu, MODULE, score),
        cache.kernel(gpu, MODULE, attn),
    ) {
        (Ok(score), Ok(attn)) if score.0 != 0 && attn.0 != 0 => Some(Kernels { score, attn }),
        _ => {
            if gpu.op_cache().once("sparse_prefill:unresolved") {
                tracing::warn!(
                    "{MODULE} is in the image but {score} or {attn} did not resolve; dense prefill"
                );
            }
            None
        }
    }
}

/// The device buffers of one call (the microtest passes its own).
#[derive(Clone, Copy, Debug)]
pub struct Buffers {
    pub scratch: DevicePtr,
    pub masks: DevicePtr,
    pub stats: DevicePtr,
}

/// Stage 1: score every (head, query block) and write the masks and counters (counters accumulate;
/// zero them first).
#[allow(clippy::too_many_arguments)]
pub fn launch_score(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    c: &Call,
    g: &Geometry,
    cfg: &Config,
    b: &Buffers,
    grid: u32,
    stream: u64,
) -> Result<()> {
    let k_scale = if c.kv == KvKind::Fp8 { c.k_scale } else { 1.0 };
    let logit_scale = c.inv_sqrt_d / cfg.stride as f32 * k_scale;
    KernelLaunch::new(gpu, k.score)
        .grid([grid, 1, 1])
        .block([SCORE_THREADS, 1, 1])
        .shared_mem(score_smem(c.hd))
        .arg_ptr(c.q)
        .arg_ptr(c.k_pool)
        .arg_ptr(c.block_table)
        .arg_ptr(b.scratch)
        .arg_ptr(b.masks)
        .arg_ptr(b.stats)
        .arg_u32(c.q_len)
        .arg_u32(c.kv_len)
        .arg_u32(c.q_offset)
        .arg_u32(c.nq)
        .arg_u32(c.nkv)
        .arg_u32(c.block_size)
        .arg_u64(c.cache_stride)
        .arg_u32(g.q_block_lo)
        .arg_u32(g.n_qb)
        .arg_u32(g.nkb)
        .arg_u32(g.mask_words)
        .arg_f32(logit_scale)
        .arg_f32(cfg.threshold)
        .launch(stream)
}

/// Stage 2: flash attention over the kept blocks of `masks`.
pub fn launch_attn(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    c: &Call,
    g: &Geometry,
    masks: DevicePtr,
    stream: u64,
) -> Result<()> {
    let (k_scale, v_scale) = if c.kv == KvKind::Fp8 {
        (c.k_scale, c.v_scale)
    } else {
        (1.0, 1.0)
    };
    KernelLaunch::new(gpu, k.attn)
        .grid([g.n_tiles, c.nq, 1])
        .block([ATTN_THREADS, 1, 1])
        .shared_mem(attn_smem(c.hd))
        .arg_ptr(c.q)
        .arg_ptr(c.k_pool)
        .arg_ptr(c.v_pool)
        .arg_ptr(c.block_table)
        .arg_ptr(masks)
        .arg_ptr(c.out)
        .arg_u32(c.q_len)
        .arg_u32(c.kv_len)
        .arg_u32(c.q_offset)
        .arg_u32(c.nq)
        .arg_u32(c.nkv)
        .arg_u32(c.block_size)
        .arg_u64(c.cache_stride)
        .arg_u32(g.q_block_lo)
        .arg_u32(g.n_qb)
        .arg_u32(g.mask_words)
        .arg_f32(c.inv_sqrt_d)
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .launch(stream)
}

/// The device's SM count, asked once.
fn sm_count(gpu: &dyn GpuBackend) -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SMS: AtomicU32 = AtomicU32::new(0);
    match SMS.load(Ordering::Relaxed) {
        0 => {
            let n = gpu.sm_count().unwrap_or(REFERENCE_SM_COUNT).max(1);
            SMS.store(n, Ordering::Relaxed);
            n
        }
        n => n,
    }
}

/// Run one chunk's attention block-sparse. `Ok(false)`: declined, nothing was launched, the caller
/// runs the dense kernel. `Ok(true)`: `c.out` holds the attention output.
pub fn run(gpu: &dyn GpuBackend, cfg: &Config, c: &Call, stream: u64) -> Result<bool> {
    if let Err(d) = eligibility(cfg, c) {
        note_decline(gpu, d, c);
        return Ok(false);
    }
    let Some(k) = resolve(gpu, c.kv, c.hd, cfg.stride) else {
        note_decline(gpu, Decline::Absent, c);
        return Ok(false);
    };
    let g = Geometry::new(c);
    let grid = g.score_grid(sm_count(gpu), c.hd);
    let cache = gpu.op_cache();
    let b = Buffers {
        scratch: cache.scratch(
            gpu,
            "sparse_prefill:score",
            g.score_scratch_bytes(grid, cfg.stride),
        )?,
        masks: cache.scratch(gpu, "sparse_prefill:masks", g.mask_bytes())?,
        stats: cache.scratch(gpu, "sparse_prefill:stats", STATS_BYTES)?,
    };
    gpu.memset_async(b.stats, 0, STATS_BYTES, stream)?;
    launch_score(gpu, &k, c, &g, cfg, &b, grid, stream)?;
    launch_attn(gpu, &k, c, &g, b.masks, stream)?;
    let n = metrics::note_layer();
    if n == 1 || n.is_multiple_of(SAMPLE_EVERY) {
        gpu.synchronize(stream)?;
        let mut raw = [0u8; STATS_BYTES];
        gpu.copy_d2h(b.stats, &mut raw)?;
        let word =
            |i: usize| u64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().expect("8 bytes"));
        let (kept, candidates) = (word(0), word(1));
        metrics::note_sample(kept, candidates);
        if gpu.op_cache().once("sparse_prefill:first") {
            tracing::info!(
                "sparse prefill active: layer {} kv_len {} chunk {}+{}: kept {kept} of {candidates} key \
                 blocks ({:.1}%), threshold {} stride {}",
                c.layer,
                c.kv_len,
                c.q_offset,
                c.q_len,
                100.0 * kept as f64 / candidates.max(1) as f64,
                cfg.threshold,
                cfg.stride
            );
        }
    }
    Ok(true)
}

fn note_decline(gpu: &dyn GpuBackend, d: Decline, c: &Call) {
    if let Some(key) = d.log_key()
        && gpu.op_cache().once(key)
    {
        tracing::info!(
            "sparse prefill not taken ({d:?}): layer {} kv {:?} head_dim {} window {} nq {} nkv {}; \
             the dense prefill kernel serves",
            c.layer,
            c.kv,
            c.hd,
            c.sliding_window,
            c.nq,
            c.nkv
        );
    }
}
