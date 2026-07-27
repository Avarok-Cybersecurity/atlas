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
            tracing::debug!(
                target: "spark_model::tp_shard",
                out_dim, in_dim, local_in, src_row_bytes, local_row_bytes,
                tp_rank, tp_size, src = src.0,
                "dense row-parallel shard (per-row strided)"
            );
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

mod gdn;
pub use gdn::*;

mod quant_shard;
pub use quant_shard::{shard_fp8_block_scaled, shard_quantized_nvfp4};

#[cfg(test)]
mod tests;
