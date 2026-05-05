// SPDX-License-Identifier: AGPL-3.0-only

//! Format-specific shard primitives for pre-quantized weights.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::TpShardKind;
use crate::weight_map::{Fp8Weight, QuantizedWeight};

// ── Format-specific shard primitives ───────────────────────────────
//
// `shard_dense_bf16` (above) handles the BF16 case used by every
// loader's "load BF16 → shard → quantize" path. The two below handle
// the formats that ship pre-quantized on disk:
//
//   - NVFP4 packed: `weight` is `[N, K/2]` u8, `weight_scale` is
//     `[N, K/group_size]` u8 (FP8 E4M3), `weight_scale_2` is a single
//     f32. Both byte tensors slice on the same N (column-parallel) or
//     K axis (row-parallel); the per-tensor scale is replicated.
//
//   - FP8 block-scaled: `weight` is `[N, K]` FP8 bytes, `row_scale` is
//     `[N/block_size, K/block_size]` BF16. Both slice on N
//     (column-parallel) or K (row-parallel) at block granularity.
//
// All helpers preserve the "tp_size == 1 returns source untouched" fast
// path matching `shard_dense_bf16`.

/// Shard an NVFP4-quantized weight. The packed weight is `[N, K/2]` u8;
/// the per-group scale is `[N, K/group_size]` u8 (FP8); `weight_scale_2`
/// is a per-tensor f32 (replicated across all ranks).
///
/// `out_dim` (= N) and `in_dim` (= K) are pre-shard, full-tensor dims.
/// Returns a freshly-allocated, sharded `QuantizedWeight`. Caller frees
/// the source if `tp_size > 1`.
pub fn shard_quantized_nvfp4(
    src: &QuantizedWeight,
    out_dim: usize,
    in_dim: usize,
    kind: TpShardKind,
    tp_rank: usize,
    tp_size: usize,
    group_size: usize,
    gpu: &dyn GpuBackend,
) -> Result<QuantizedWeight> {
    if tp_size <= 1 || kind == TpShardKind::Replicated {
        return Ok(*src);
    }
    ensure!(
        in_dim.is_multiple_of(group_size),
        "NVFP4 in_dim {in_dim} not divisible by group_size {group_size}",
    );
    let half_in = in_dim / 2; // packed weight stride
    let scale_in = in_dim / group_size;
    match kind {
        TpShardKind::Replicated => unreachable!("handled above"),
        TpShardKind::ColumnParallel => {
            ensure!(
                out_dim.is_multiple_of(tp_size),
                "NVFP4 ColumnParallel: out_dim {out_dim} not divisible by tp_size {tp_size}",
            );
            let local_out = out_dim / tp_size;
            // Packed weight: [N, K/2] u8 → slice top local_out rows.
            let w_row_bytes = half_in;
            let w_local_bytes = local_out * w_row_bytes;
            let w_dst = gpu.alloc(w_local_bytes)?;
            let w_offset = tp_rank * local_out * w_row_bytes;
            gpu.copy_d2d(
                DevicePtr(src.weight.0 + w_offset as u64),
                w_dst,
                w_local_bytes,
            )?;
            // Scale: [N, K/group_size] u8 → slice same N axis.
            let s_row_bytes = scale_in;
            let s_local_bytes = local_out * s_row_bytes;
            let s_dst = gpu.alloc(s_local_bytes)?;
            let s_offset = tp_rank * local_out * s_row_bytes;
            gpu.copy_d2d(
                DevicePtr(src.weight_scale.0 + s_offset as u64),
                s_dst,
                s_local_bytes,
            )?;
            Ok(QuantizedWeight {
                weight: w_dst,
                weight_scale: s_dst,
                weight_scale_2: src.weight_scale_2,
                input_scale: src.input_scale,
            })
        }
        TpShardKind::RowParallel => {
            ensure!(
                in_dim.is_multiple_of(tp_size),
                "NVFP4 RowParallel: in_dim {in_dim} not divisible by tp_size {tp_size}",
            );
            ensure!(
                half_in.is_multiple_of(tp_size),
                "NVFP4 RowParallel: half_in {half_in} (=K/2) not divisible by tp_size {tp_size}",
            );
            ensure!(
                scale_in.is_multiple_of(tp_size),
                "NVFP4 RowParallel: scale_in {scale_in} (=K/group_size) not divisible by tp_size {tp_size}",
            );
            let local_w_in = half_in / tp_size;
            let local_s_in = scale_in / tp_size;
            // Per-row strided copy on the K axis for both weight and scale.
            let w_local_row_bytes = local_w_in;
            let w_src_row_bytes = half_in;
            let w_local_bytes = out_dim * w_local_row_bytes;
            let w_dst = gpu.alloc(w_local_bytes)?;
            let w_col_offset = tp_rank * w_local_row_bytes;
            for r in 0..out_dim {
                gpu.copy_d2d(
                    DevicePtr(src.weight.0 + (r * w_src_row_bytes + w_col_offset) as u64),
                    DevicePtr(w_dst.0 + (r * w_local_row_bytes) as u64),
                    w_local_row_bytes,
                )?;
            }
            let s_local_row_bytes = local_s_in;
            let s_src_row_bytes = scale_in;
            let s_local_bytes = out_dim * s_local_row_bytes;
            let s_dst = gpu.alloc(s_local_bytes)?;
            let s_col_offset = tp_rank * s_local_row_bytes;
            for r in 0..out_dim {
                gpu.copy_d2d(
                    DevicePtr(src.weight_scale.0 + (r * s_src_row_bytes + s_col_offset) as u64),
                    DevicePtr(s_dst.0 + (r * s_local_row_bytes) as u64),
                    s_local_row_bytes,
                )?;
            }
            Ok(QuantizedWeight {
                weight: w_dst,
                weight_scale: s_dst,
                weight_scale_2: src.weight_scale_2,
                input_scale: src.input_scale,
            })
        }
    }
}

/// Shard an FP8 block-scaled weight. `weight` is `[N, K]` FP8 bytes;
/// `row_scale` is `[N/block_size, K/block_size]` BF16. Both slice on
/// the same axis at block granularity.
pub fn shard_fp8_block_scaled(
    src: &Fp8Weight,
    kind: TpShardKind,
    tp_rank: usize,
    tp_size: usize,
    block_size: usize,
    gpu: &dyn GpuBackend,
) -> Result<Fp8Weight> {
    if tp_size <= 1 || kind == TpShardKind::Replicated {
        return Ok(*src);
    }
    let n = src.n as usize;
    let k = src.k as usize;
    ensure!(
        n.is_multiple_of(block_size),
        "FP8 N {n} not divisible by block_size {block_size}",
    );
    ensure!(
        k.is_multiple_of(block_size),
        "FP8 K {k} not divisible by block_size {block_size}",
    );
    let scale_n = n / block_size;
    let scale_k = k / block_size;
    let bf16 = 2usize;
    match kind {
        TpShardKind::Replicated => unreachable!("handled above"),
        TpShardKind::ColumnParallel => {
            ensure!(
                n.is_multiple_of(tp_size),
                "FP8 ColumnParallel: N {n} not divisible by tp_size {tp_size}",
            );
            ensure!(
                scale_n.is_multiple_of(tp_size),
                "FP8 ColumnParallel: scale_n {scale_n} not divisible by tp_size {tp_size}",
            );
            let local_n = n / tp_size;
            let local_scale_n = scale_n / tp_size;
            // Weight: [N, K] u8 → slice top local_n rows.
            let w_row_bytes = k;
            let w_local_bytes = local_n * w_row_bytes;
            let w_dst = gpu.alloc(w_local_bytes)?;
            let w_offset = tp_rank * local_n * w_row_bytes;
            gpu.copy_d2d(
                DevicePtr(src.weight.0 + w_offset as u64),
                w_dst,
                w_local_bytes,
            )?;
            // Scale: [scale_n, scale_k] BF16 → slice top local_scale_n rows.
            let s_row_bytes = scale_k * bf16;
            let s_local_bytes = local_scale_n * s_row_bytes;
            let s_dst = gpu.alloc(s_local_bytes)?;
            let s_offset = tp_rank * local_scale_n * s_row_bytes;
            gpu.copy_d2d(
                DevicePtr(src.row_scale.0 + s_offset as u64),
                s_dst,
                s_local_bytes,
            )?;
            Ok(Fp8Weight {
                weight: w_dst,
                row_scale: s_dst,
                n: local_n as u32,
                k: src.k,
            })
        }
        TpShardKind::RowParallel => {
            ensure!(
                k.is_multiple_of(tp_size),
                "FP8 RowParallel: K {k} not divisible by tp_size {tp_size}",
            );
            ensure!(
                scale_k.is_multiple_of(tp_size),
                "FP8 RowParallel: scale_k {scale_k} not divisible by tp_size {tp_size}",
            );
            let local_k = k / tp_size;
            let local_scale_k = scale_k / tp_size;
            // Weight: per-row strided on K.
            let w_local_row_bytes = local_k;
            let w_src_row_bytes = k;
            let w_local_bytes = n * w_local_row_bytes;
            let w_dst = gpu.alloc(w_local_bytes)?;
            let w_col_offset = tp_rank * w_local_row_bytes;
            for r in 0..n {
                gpu.copy_d2d(
                    DevicePtr(src.weight.0 + (r * w_src_row_bytes + w_col_offset) as u64),
                    DevicePtr(w_dst.0 + (r * w_local_row_bytes) as u64),
                    w_local_row_bytes,
                )?;
            }
            // Scale: per-row strided on scale_k.
            let s_local_row_bytes = local_scale_k * bf16;
            let s_src_row_bytes = scale_k * bf16;
            let s_local_bytes = scale_n * s_local_row_bytes;
            let s_dst = gpu.alloc(s_local_bytes)?;
            let s_col_offset = tp_rank * s_local_row_bytes;
            for r in 0..scale_n {
                gpu.copy_d2d(
                    DevicePtr(src.row_scale.0 + (r * s_src_row_bytes + s_col_offset) as u64),
                    DevicePtr(s_dst.0 + (r * s_local_row_bytes) as u64),
                    s_local_row_bytes,
                )?;
            }
            Ok(Fp8Weight {
                weight: w_dst,
                row_scale: s_dst,
                n: src.n,
                k: local_k as u32,
            })
        }
    }
}
