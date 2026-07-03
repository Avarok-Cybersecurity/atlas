// SPDX-License-Identifier: AGPL-3.0-only

//! Tensor-parallel weight sharding helpers.
//!
//! Megatron-style TP slices each weight tensor along one of two axes:
//!
//! - **Column-parallel** (Q/K/V proj, gate_proj, up_proj, lm_head): weight
//!   shape `[out, in]` becomes `[out / tp, in]`. Rank `r` keeps rows
//!   `[r * out / tp, (r + 1) * out / tp)`. This is a single contiguous
//!   slice in row-major layout — one `copy_d2d`.
//!
//! - **Row-parallel** (O proj, down_proj): weight `[out, in]` becomes
//!   `[out, in / tp]`. Rank `r` keeps cols
//!   `[r * in / tp, (r + 1) * in / tp)`. Per-row strided copy because the
//!   surviving slice is non-contiguous in row-major layout.
//!
//! 1D per-output vectors (q_norm_full, k_norm_full, gate_proj bias, etc.)
//! shard with the same axis as their associated GEMM's column-parallel output.
//!
//! All shard helpers operate on BF16 weights *before* NVFP4 quantization;
//! sharding the packed FP4 storage + FP8 scales is mechanical but adds two
//! more axes to bookkeep, and pre-quant slicing keeps the existing quantize
//! path untouched.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::weight_map::DenseWeight;

/// Bytes per BF16 element.
const BF16_BYTES: usize = 2;

/// TP shard kind for a 2D BF16 weight `[out_dim, in_dim]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpShardKind {
    /// Replicated: every TP rank holds the full tensor (norm scalars,
    /// embedding-tied weights, MTP heads in v1).
    Replicated,
    /// Column-parallel: split `out_dim` evenly across ranks. Rank `r` keeps
    /// rows `[r * out_dim / tp, (r + 1) * out_dim / tp)`.
    ColumnParallel,
    /// Row-parallel: split `in_dim` evenly across ranks. Rank `r` keeps
    /// cols `[r * in_dim / tp, (r + 1) * in_dim / tp)`.
    RowParallel,
}

/// Shard a BF16 dense weight `[out_dim, in_dim]` according to `kind`.
///
/// Returns `(sharded_ptr, sharded_out, sharded_in)`. When `tp_size == 1`
/// or `kind == Replicated`, returns the source pointer untouched and the
/// caller must NOT free the source separately (no shard happened).
///
/// Otherwise allocates a new device buffer holding the local rank's slice,
/// copies into it, and returns the new pointer. The caller owns the source
/// and must `gpu.free` it after the shard is built.
pub fn shard_dense_bf16(
    src: DevicePtr,
    out_dim: usize,
    in_dim: usize,
    kind: TpShardKind,
    tp_rank: usize,
    tp_size: usize,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, usize, usize)> {
    if tp_size <= 1 || kind == TpShardKind::Replicated {
        return Ok((src, out_dim, in_dim));
    }
    ensure!(tp_rank < tp_size, "tp_rank {tp_rank} >= tp_size {tp_size}");
    match kind {
        TpShardKind::Replicated => unreachable!("handled above"),
        TpShardKind::ColumnParallel => {
            ensure!(
                out_dim.is_multiple_of(tp_size),
                "ColumnParallel: out_dim {out_dim} not divisible by tp_size {tp_size}",
            );
            let local_out = out_dim / tp_size;
            let row_bytes = in_dim * BF16_BYTES;
            let local_bytes = local_out * row_bytes;
            let dst = gpu.alloc(local_bytes)?;
            let src_offset = tp_rank * local_out * row_bytes;
            let src_slice = DevicePtr(src.0 + src_offset as u64);
            gpu.copy_d2d(src_slice, dst, local_bytes)?;
            Ok((dst, local_out, in_dim))
        }
        TpShardKind::RowParallel => {
            ensure!(
                in_dim.is_multiple_of(tp_size),
                "RowParallel: in_dim {in_dim} not divisible by tp_size {tp_size}",
            );
            let local_in = in_dim / tp_size;
            let local_row_bytes = local_in * BF16_BYTES;
            let src_row_bytes = in_dim * BF16_BYTES;
            let local_bytes = out_dim * local_row_bytes;
            let dst = gpu.alloc(local_bytes)?;
            // Per-row strided copy: row r of dst comes from row r of src,
            // starting at column `tp_rank * local_in`.
            let col_offset_bytes = tp_rank * local_row_bytes;
            for r in 0..out_dim {
                let src_off = r * src_row_bytes + col_offset_bytes;
                let dst_off = r * local_row_bytes;
                gpu.copy_d2d(
                    DevicePtr(src.0 + src_off as u64),
                    DevicePtr(dst.0 + dst_off as u64),
                    local_row_bytes,
                )?;
            }
            Ok((dst, out_dim, local_in))
        }
    }
}

/// Shard a 1D BF16 vector `[dim]` (e.g. q_norm_full, gate_proj bias) on
/// dim 0. Used for per-output vectors that pair with column-parallel GEMMs.
pub fn shard_dense_1d_bf16(
    src: DevicePtr,
    dim: usize,
    tp_rank: usize,
    tp_size: usize,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, usize)> {
    if tp_size <= 1 {
        return Ok((src, dim));
    }
    ensure!(tp_rank < tp_size, "tp_rank {tp_rank} >= tp_size {tp_size}");
    ensure!(
        dim.is_multiple_of(tp_size),
        "shard_dense_1d_bf16: dim {dim} not divisible by tp_size {tp_size}",
    );
    let local_dim = dim / tp_size;
    let local_bytes = local_dim * BF16_BYTES;
    let dst = gpu.alloc(local_bytes)?;
    let src_offset = tp_rank * local_bytes;
    gpu.copy_d2d(DevicePtr(src.0 + src_offset as u64), dst, local_bytes)?;
    Ok((dst, local_dim))
}

/// Convenience wrapper: shard a `DenseWeight` BF16 tensor. The source weight
/// is freed by the caller — this fn allocates a new device buffer.
pub fn shard_dense_weight(
    src: &DenseWeight,
    out_dim: usize,
    in_dim: usize,
    kind: TpShardKind,
    tp_rank: usize,
    tp_size: usize,
    gpu: &dyn GpuBackend,
) -> Result<(DenseWeight, usize, usize)> {
    let (ptr, n, k) = shard_dense_bf16(src.weight, out_dim, in_dim, kind, tp_rank, tp_size, gpu)?;
    Ok((DenseWeight { weight: ptr }, n, k))
}

// ════════════════════════════════════════════════════════════════════
// Higher-level helpers — DRY across per-architecture weight loaders.
//
// Each loader was repeating the same dimension math + the same Q/K/V/O
// (col, col, col, row) Megatron pattern. The four helpers below capture
// the isomorphism so a new loader only needs the format-specific load
// closure, not the dimension bookkeeping.
//
// Cross-loader patterns extracted:
//   1. Attention QKVO: 3× ColumnParallel + 1× RowParallel. `TpAttentionDims`
//      reconstructs full pre-shard sizes from `config` (which `main.rs`
//      already TP-divided for head counts), then `load_qkvo_tp` sequences
//      the four loads via a caller closure.
//   2. Q/K norm pair: 1D shards aligned with the QKV column-parallel axis.
//      `load_qk_norms_tp` calls a 1D-shard closure for `q_norm`/`k_norm`.
//   3. MoE expert projections: 2× ColumnParallel (gate, up) + 1× RowParallel
//      (down) on the routed-expert intermediate dim. `TpMoeDims` + the
//      caller-side closure mirror the QKVO pattern but on the MoE axes.
//
// Per-quantization-format byte-slicing primitives (`shard_dense_bf16`
// above, `shard_quantized_nvfp4` and `shard_fp8_block_scaled` below)
// stay in this module so each loader can pick the matching primitive
// from inside its closure.
// ════════════════════════════════════════════════════════════════════

/// Pre-TP-shard attention dimensions reconstructed from `config`.
///
/// `main.rs` divides `num_attention_heads` and `num_key_value_heads` by
/// `tp_world_size` at startup, so by the time a loader runs, `config`
/// holds **per-rank-local** head counts. The `full_*` fields multiply
/// back up to the pre-shard sizes that `slice_for_rank` and friends
/// expect.
///
/// When `config.attn_gated` is true (Qwen3-Next), the Q projection
/// output dim is doubled — the second half is the per-token gate
/// applied after attention. `full_q_n` includes the gate; `full_o_in`
/// does NOT (O proj's input dim matches the un-gated attention
/// output, since the gate is applied before O proj).
#[derive(Debug, Clone, Copy)]
pub struct TpAttentionDims {
    pub tp_rank: usize,
    /// `tp_world_size` clamped to `>= 1`. Loaders should treat
    /// `tp_size == 1` as the no-shard fast path.
    pub tp_size: usize,
    /// Hidden size (model embed dim) — never sharded.
    pub h: usize,
    pub head_dim: usize,
    /// Q-projection output dim. For gated attention this is doubled
    /// (the second half is the gate).
    pub full_q_n: usize,
    /// O-projection input dim. Equals the un-gated attention output —
    /// `num_attention_heads * tp_size * head_dim`, NOT doubled.
    pub full_o_in: usize,
    /// `num_key_value_heads_local * tp_size * head_dim` — full K/V pre-shard.
    pub full_kv_n: usize,
    /// Whether the loader is operating on a gated-attention config.
    pub gated: bool,
}

impl TpAttentionDims {
    pub fn from_config(config: &ModelConfig) -> Self {
        let tp_size = config.tp_world_size.max(1);
        let head_dim = config.head_dim;
        let num_heads_local = config.num_attention_heads;
        let num_kv_heads_local = config.num_key_value_heads;
        let gated = config.attn_gated;
        let attn_out = num_heads_local * tp_size * head_dim;
        let q_factor = if gated { 2 } else { 1 };
        Self {
            tp_rank: config.tp_rank,
            tp_size,
            h: config.hidden_size,
            head_dim,
            full_q_n: attn_out * q_factor,
            full_o_in: attn_out,
            full_kv_n: num_kv_heads_local * tp_size * head_dim,
            gated,
        }
    }

    /// `(out_dim, in_dim, kind)` for a given QKVO projection.
    pub fn proj_shape(&self, name: &str) -> Option<(usize, usize, TpShardKind)> {
        match name {
            "q_proj" => Some((self.full_q_n, self.h, TpShardKind::ColumnParallel)),
            "k_proj" | "v_proj" => Some((self.full_kv_n, self.h, TpShardKind::ColumnParallel)),
            "o_proj" => Some((self.h, self.full_o_in, TpShardKind::RowParallel)),
            _ => None,
        }
    }
}

/// Sequence the four Q/K/V/O loads via a loader-supplied closure. The
/// closure receives `(name, full_out, full_in, kind)` and returns the
/// loader's representation of that projection (BF16 dense, NVFP4
/// quantized, FP8 block-scaled — varies by format).
///
/// Returns `[Q, K, V, O]`; callers destructure with
/// `let [q, k, v, o] = load_qkvo_tp(config, |name, n, k, kind| { ... })?;`.
pub fn load_qkvo_tp<F, T>(config: &ModelConfig, mut proj_loader: F) -> Result<[T; 4]>
where
    F: FnMut(&str, usize, usize, TpShardKind) -> Result<T>,
{
    let dims = TpAttentionDims::from_config(config);
    let q = proj_loader("q_proj", dims.full_q_n, dims.h, TpShardKind::ColumnParallel)?;
    let k = proj_loader(
        "k_proj",
        dims.full_kv_n,
        dims.h,
        TpShardKind::ColumnParallel,
    )?;
    let v = proj_loader(
        "v_proj",
        dims.full_kv_n,
        dims.h,
        TpShardKind::ColumnParallel,
    )?;
    // O proj input dim is the un-gated attention output. For gated
    // models (Qwen3-Next), this differs from `full_q_n` which includes
    // the doubled gate.
    let o = proj_loader("o_proj", dims.h, dims.full_o_in, TpShardKind::RowParallel)?;
    Ok([q, k, v, o])
}

/// Q/K-norm 1D shard pair. The closure receives `(name, full_dim)` and
/// returns the loader's sharded norm — typically a `DenseWeight`.
/// `q_norm` is sharded against `full_q_n`; `k_norm` against `full_kv_n`.
/// Returns `(q_norm, k_norm)`.
pub fn load_qk_norms_tp<F, T>(config: &ModelConfig, mut norm_loader: F) -> Result<(T, T)>
where
    F: FnMut(&str, usize) -> Result<T>,
{
    let dims = TpAttentionDims::from_config(config);
    let q_norm = norm_loader("q_norm", dims.full_q_n)?;
    let k_norm = norm_loader("k_norm", dims.full_kv_n)?;
    Ok((q_norm, k_norm))
}

/// Pre-TP-shard dimensions for MoE expert projections. Unlike attention,
/// `main.rs` does NOT divide `moe_intermediate_size` by `tp_size`, so
/// `full_inter == config.moe_intermediate_size`. Local size is computed
/// here for downstream callers.
#[derive(Debug, Clone, Copy)]
pub struct TpMoeDims {
    pub tp_rank: usize,
    pub tp_size: usize,
    pub h: usize,
    /// Full MoE intermediate dim (NOT yet TP-divided).
    pub full_inter: usize,
    /// Local (post-shard) MoE intermediate dim.
    pub local_inter: usize,
}

impl TpMoeDims {
    pub fn from_config(config: &ModelConfig) -> Self {
        let tp_size = config.tp_world_size.max(1);
        let full_inter = config.moe_intermediate_size;
        Self {
            tp_rank: config.tp_rank,
            tp_size,
            h: config.hidden_size,
            full_inter,
            local_inter: full_inter / tp_size,
        }
    }

    /// `(out_dim, in_dim, kind)` for one of `gate_proj` / `up_proj` /
    /// `down_proj`. Gate/up are column-parallel on inter; down is
    /// row-parallel on inter (so `[h, inter]` rows truncate to `[h, inter/tp]`).
    pub fn proj_shape(&self, name: &str) -> Option<(usize, usize, TpShardKind)> {
        match name {
            "gate_proj" | "up_proj" => Some((self.full_inter, self.h, TpShardKind::ColumnParallel)),
            "down_proj" => Some((self.h, self.full_inter, TpShardKind::RowParallel)),
            _ => None,
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// GDN HeadParallel — tensor-parallel sharding of the Gated-DeltaNet
// (linear_attention / SSM) layers for Qwen3.5 / 3.6.
//
// The GDN recurrence is embarrassingly parallel across *value-head groups*:
// each TP rank owns a contiguous range of key/value heads, runs the whole
// scan locally with LOCAL nk/nv/conv_dim, and the ranks reconcile with a
// single all-reduce after `out_proj` (row-parallel, exactly like attention
// after `o_proj`). No cross-rank comm inside the scan.
//
// CRITICAL LAYOUT FACT — the in-projection is stored as *segmented*
// contiguous blocks, NOT one flat matrix:
//
//   in_proj_qkv : [Q | K | V]        rows = nk·kd + nk·kd + nv·vd  (= conv_dim)
//   in_proj_z   : [Z]                rows = nv·vd
//   → gpu_concat_rows → QKVZ         [Q | K | V | Z]
//
// A naive "first out_dim/tp rows" slice is WRONG: it would give rank 0 the
// whole Q block plus part of K. Each segment (Q, K, V, Z) must be sliced by
// the LOCAL head range *independently*, then the local slices re-concatenated
// in the same [Q|K|V|Z] order. The `segment_copy_plan` below encodes exactly
// that: one contiguous copy per segment, packed back-to-back into the local
// buffer.
//
// The depthwise `conv1d` weight `[conv_dim, d_conv]` is sharded with the SAME
// segment pattern as QKV (its channels ARE the QKV channels — one filter per
// channel), NOT replicated. `a_log`/`dt_bias` (`[nv]` FP32), `norm`
// (`[nv·vd]` BF16) and `out_proj` (`[h, nv·vd]`, row-parallel) shard on the
// value-head axis. The BA gate buffer is per-group interleaved but the rank
// boundary always lands on a group boundary, so it slices contiguously.
// ════════════════════════════════════════════════════════════════════

/// Pre-TP-shard GDN (linear-attention / SSM) dimensions reconstructed from
/// `config`.
///
/// Mirrors [`TpAttentionDims`]: `topology.rs` divides
/// `linear_num_key_heads` / `linear_num_value_heads` by `tp_world_size` at
/// startup, so by the time a loader runs `config` holds **per-rank-local**
/// head counts. The `full_*` fields multiply back up to the pre-shard sizes
/// that the segment slicers expect. Head *dims* (`kd`, `vd`) and the hidden
/// size `h` are never sharded.
#[derive(Debug, Clone, Copy)]
pub struct TpGdnDims {
    pub tp_rank: usize,
    /// `tp_world_size` clamped to `>= 1`. Loaders treat `tp_size == 1` as the
    /// no-shard fast path.
    pub tp_size: usize,
    /// Hidden size (model embed dim) — never sharded.
    pub h: usize,
    /// Key head dim (`linear_key_head_dim`) — never sharded.
    pub kd: usize,
    /// Value head dim (`linear_value_head_dim`) — never sharded.
    pub vd: usize,
    /// Per-rank key heads (Q and K share this count).
    pub local_nk: usize,
    /// Full pre-shard key heads = `local_nk * tp_size`.
    pub full_nk: usize,
    /// Per-rank value heads.
    pub local_nv: usize,
    /// Full pre-shard value heads = `local_nv * tp_size`.
    pub full_nv: usize,
}

impl TpGdnDims {
    pub fn from_config(config: &ModelConfig) -> Self {
        let tp_size = config.tp_world_size.max(1);
        let local_nk = config.linear_num_key_heads;
        let local_nv = config.linear_num_value_heads;
        Self {
            tp_rank: config.tp_rank,
            tp_size,
            h: config.hidden_size,
            kd: config.linear_key_head_dim,
            vd: config.linear_value_head_dim,
            local_nk,
            full_nk: local_nk * tp_size,
            local_nv,
            full_nv: local_nv * tp_size,
        }
    }

    /// Full (pre-shard) key projection width: `full_nk * kd`.
    pub fn full_key_dim(&self) -> usize {
        self.full_nk * self.kd
    }
    /// Local key projection width: `local_nk * kd`.
    pub fn local_key_dim(&self) -> usize {
        self.local_nk * self.kd
    }
    /// Full (pre-shard) value projection width: `full_nv * vd`.
    pub fn full_value_dim(&self) -> usize {
        self.full_nv * self.vd
    }
    /// Local value projection width: `local_nv * vd`.
    pub fn local_value_dim(&self) -> usize {
        self.local_nv * self.vd
    }
    /// Full conv / QKV width: `2*full_nk*kd + full_nv*vd`.
    pub fn full_conv_dim(&self) -> usize {
        2 * self.full_key_dim() + self.full_value_dim()
    }
    /// Local conv / QKV width: `2*local_nk*kd + local_nv*vd`.
    pub fn local_conv_dim(&self) -> usize {
        2 * self.local_key_dim() + self.local_value_dim()
    }
    /// Full QKVZ out dim: `2*full_nk*kd + 2*full_nv*vd`.
    pub fn full_qkvz_out(&self) -> usize {
        self.full_conv_dim() + self.full_value_dim()
    }
    /// Local QKVZ out dim: `2*local_nk*kd + 2*local_nv*vd`.
    pub fn local_qkvz_out(&self) -> usize {
        self.local_conv_dim() + self.local_value_dim()
    }

    /// Full-row segment list for the `[Q|K|V]` in-projection.
    fn qkv_segments(&self) -> [usize; 3] {
        [self.full_key_dim(), self.full_key_dim(), self.full_value_dim()]
    }
    /// Full-row segment list for the concatenated `[Q|K|V|Z]` in-projection.
    fn qkvz_segments(&self) -> [usize; 4] {
        [
            self.full_key_dim(),
            self.full_key_dim(),
            self.full_value_dim(),
            self.full_value_dim(),
        ]
    }
}

/// A single device-to-device copy in a segmented-slice plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CopyOp {
    src_off: usize,
    dst_off: usize,
    len: usize,
}

/// Build the copy plan for a SEGMENTED row-slice.
///
/// `segments` lists the full (pre-shard) row count of each contiguous block
/// (Q, K, V[, Z] for QKVZ). Each block is sliced *independently* to the local
/// rank's head range `[tp_rank * seg/tp, (tp_rank+1) * seg/tp)` and the local
/// slices are packed back-to-back into the output buffer, preserving segment
/// order. `row_bytes` is the byte width of one row (`in_dim * elem_bytes`).
///
/// Returns `(ops, local_total_rows)`. Every segment must be divisible by
/// `tp_size` — the caller has already reconstructed `full_*` as
/// `local_* * tp_size`, so this holds by construction, but it is checked to
/// fail loudly on a mis-wired config rather than silently corrupt heads.
fn segment_copy_plan(
    segments: &[usize],
    row_bytes: usize,
    tp_rank: usize,
    tp_size: usize,
) -> Result<(Vec<CopyOp>, usize)> {
    ensure!(tp_rank < tp_size, "tp_rank {tp_rank} >= tp_size {tp_size}");
    let mut ops = Vec::with_capacity(segments.len());
    let mut src_rows = 0usize; // running offset into the source (in rows)
    let mut dst_rows = 0usize; // running offset into the packed dst (in rows)
    for (i, &seg) in segments.iter().enumerate() {
        ensure!(
            seg.is_multiple_of(tp_size),
            "segment {i} ({seg} rows) not divisible by tp_size {tp_size}",
        );
        let local = seg / tp_size;
        ops.push(CopyOp {
            src_off: (src_rows + tp_rank * local) * row_bytes,
            dst_off: dst_rows * row_bytes,
            len: local * row_bytes,
        });
        src_rows += seg;
        dst_rows += local;
    }
    Ok((ops, dst_rows))
}

/// Execute a segmented row-slice on the GPU. `row_elems` is the number of
/// elements per row (`in_dim`); `elem_bytes` its size (2 = BF16, 4 = FP32).
/// Returns `(local_ptr, local_total_rows)`. For `tp_size <= 1` returns the
/// source untouched (no allocation, caller must not double-free).
fn slice_segments(
    src: DevicePtr,
    segments: &[usize],
    row_elems: usize,
    elem_bytes: usize,
    tp_rank: usize,
    tp_size: usize,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, usize)> {
    let full_rows: usize = segments.iter().sum();
    if tp_size <= 1 {
        return Ok((src, full_rows));
    }
    let row_bytes = row_elems * elem_bytes;
    let (ops, local_rows) = segment_copy_plan(segments, row_bytes, tp_rank, tp_size)?;
    let dst = gpu.alloc(local_rows * row_bytes)?;
    for op in &ops {
        gpu.copy_d2d(src.offset(op.src_off), dst.offset(op.dst_off), op.len)?;
    }
    Ok((dst, local_rows))
}

/// Shard the `[Q|K|V]` (`in_proj_qkv`) BF16 weight `[full_conv_dim, h]` to the
/// local rank's `[local_conv_dim, h]`, slicing Q, K and V independently by the
/// local head range. Returns `(ptr, local_rows, h)`.
pub fn shard_gdn_qkv_rows(
    src: DevicePtr,
    dims: &TpGdnDims,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, usize, usize)> {
    let (ptr, rows) = slice_segments(
        src,
        &dims.qkv_segments(),
        dims.h,
        BF16_BYTES,
        dims.tp_rank,
        dims.tp_size,
        gpu,
    )?;
    Ok((ptr, rows, dims.h))
}

/// Shard the concatenated `[Q|K|V|Z]` (`in_proj_qkvz`) BF16 weight
/// `[full_qkvz_out, h]` to the local rank's `[local_qkvz_out, h]`, slicing all
/// four segments independently. Returns `(ptr, local_rows, h)`.
pub fn shard_gdn_qkvz_rows(
    src: DevicePtr,
    dims: &TpGdnDims,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, usize, usize)> {
    let (ptr, rows) = slice_segments(
        src,
        &dims.qkvz_segments(),
        dims.h,
        BF16_BYTES,
        dims.tp_rank,
        dims.tp_size,
        gpu,
    )?;
    Ok((ptr, rows, dims.h))
}

/// Shard the BA gate BF16 weight `[2*full_nv, h]` to `[2*local_nv, h]`.
///
/// The interleave is per key-head group (`[β₀..β_{vpg-1}, α₀..α_{vpg-1}]` per
/// group, `vpg = nv/nk`), but rank `r` owns key-head groups
/// `[r*local_nk, (r+1)*local_nk)` which map to the contiguous row range
/// `[r*2*local_nv, (r+1)*2*local_nv)` — the rank boundary always lands on a
/// group boundary, so a single contiguous slice preserves the interleave.
pub fn shard_gdn_ba_rows(
    src: DevicePtr,
    dims: &TpGdnDims,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, usize, usize)> {
    // Group-boundary alignment guarantee: full_nk divisible by tp_size ⇒
    // each rank gets whole groups.
    ensure!(
        dims.full_nk.is_multiple_of(dims.tp_size),
        "BA: full_nk {} not divisible by tp_size {}",
        dims.full_nk,
        dims.tp_size,
    );
    let (ptr, rows) = slice_segments(
        src,
        &[2 * dims.full_nv],
        dims.h,
        BF16_BYTES,
        dims.tp_rank,
        dims.tp_size,
        gpu,
    )?;
    Ok((ptr, rows, dims.h))
}

/// Shard the depthwise `conv1d` BF16 weight `[full_conv_dim, d_conv]` to
/// `[local_conv_dim, d_conv]`. Channels ARE the QKV channels (one filter per
/// channel), so this uses the SAME `[Q|K|V]` segment pattern as the QKV
/// in-projection — the conv is NOT replicated across ranks.
pub fn shard_gdn_conv_rows(
    src: DevicePtr,
    dims: &TpGdnDims,
    d_conv: usize,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, usize, usize)> {
    let (ptr, rows) = slice_segments(
        src,
        &dims.qkv_segments(),
        d_conv,
        BF16_BYTES,
        dims.tp_rank,
        dims.tp_size,
        gpu,
    )?;
    Ok((ptr, rows, d_conv))
}

/// Shard a per-value-head 1D vector on the value-head axis. Handles BF16
/// (`norm`, `[full_nv*vd]` → `[local_nv*vd]` with `elem_bytes = 2`,
/// `unit = vd`) and FP32 (`a_log` / `dt_bias`, `[full_nv]` → `[local_nv]` with
/// `elem_bytes = 4`, `unit = 1`). `unit` is the number of elements per value
/// head. Returns `(ptr, local_len_elems)`.
pub fn shard_gdn_value_vector(
    src: DevicePtr,
    dims: &TpGdnDims,
    unit: usize,
    elem_bytes: usize,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, usize)> {
    let full_len = dims.full_nv * unit;
    if dims.tp_size <= 1 {
        return Ok((src, full_len));
    }
    ensure!(
        dims.tp_rank < dims.tp_size,
        "tp_rank {} >= tp_size {}",
        dims.tp_rank,
        dims.tp_size,
    );
    let local_len = dims.local_nv * unit;
    let local_bytes = local_len * elem_bytes;
    let dst = gpu.alloc(local_bytes)?;
    let src_off = dims.tp_rank * local_bytes;
    gpu.copy_d2d(src.offset(src_off), dst, local_bytes)?;
    Ok((dst, local_len))
}

/// Shard the `out_proj` BF16 weight `[h, full_value_dim]` row-parallel on its
/// input dim (value_dim). Rank `r` keeps columns
/// `[r*local_value_dim, (r+1)*local_value_dim)` of every output row; the
/// partial products are summed with an all-reduce after the GEMM (mirrors
/// attention `o_proj`). Returns `(ptr, h, local_value_dim)`.
pub fn shard_gdn_out_proj_row_parallel(
    src: DevicePtr,
    dims: &TpGdnDims,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, usize, usize)> {
    shard_dense_bf16(
        src,
        dims.h,
        dims.full_value_dim(),
        TpShardKind::RowParallel,
        dims.tp_rank,
        dims.tp_size,
        gpu,
    )
}

mod quant_shard;
pub use quant_shard::{shard_fp8_block_scaled, shard_quantized_nvfp4};

#[cfg(test)]
mod tests;
