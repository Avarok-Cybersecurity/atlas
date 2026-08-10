// SPDX-License-Identifier: AGPL-3.0-only

//! GDN HeadParallel — tensor-parallel sharding of the Gated-DeltaNet (SSM /
//! linear-attention) layers. Split out of `tp_shard.rs` (file-size cap).

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{BF16_BYTES, TpShardKind, shard_dense_bf16};

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
/// Mirrors [`super::TpAttentionDims`]: `topology.rs` divides
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
    pub(crate) fn qkv_segments(&self) -> [usize; 3] {
        [
            self.full_key_dim(),
            self.full_key_dim(),
            self.full_value_dim(),
        ]
    }
    /// Full-row segment list for the concatenated `[Q|K|V|Z]` in-projection.
    pub(crate) fn qkvz_segments(&self) -> [usize; 4] {
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
pub(crate) struct CopyOp {
    pub(crate) src_off: usize,
    pub(crate) dst_off: usize,
    pub(crate) len: usize,
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
pub(crate) fn segment_copy_plan(
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
    tracing::debug!(
        target: "spark_model::tp_shard",
        ?segments, full_rows, row_elems, elem_bytes, local_rows,
        tp_rank, tp_size, src = src.0,
        "gdn segmented row-slice"
    );
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
    tracing::debug!(
        target: "spark_model::tp_shard",
        full_nv = dims.full_nv, local_nv = dims.local_nv, unit, elem_bytes,
        full_len, local_len, local_bytes, src_off, tp_rank = dims.tp_rank,
        src = src.0,
        "gdn value-vector shard (per-value-head axis)"
    );
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
