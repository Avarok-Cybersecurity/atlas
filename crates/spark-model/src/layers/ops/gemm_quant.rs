// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// FP8×FP8 GEMM: A [M, K] FP8 × B [N, K] FP8 → C [M, N] BF16.
///
/// Both A (activations) and B (weights) are pre-converted FP8 E4M3.
/// No BF16→FP8 conversion in inner loop — pure MMA throughput.
/// Grid: (ceil(N/128), ceil(M/64))  Block: (128, 1, 1)
pub fn fp8_fp8_gemm_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_fp8: DevicePtr,
    b_fp8: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(a_fp8)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// M128 variant of fp8_gemm_n128: halves B re-reads for large M (ISL > 128).
///
/// Each CTA covers 128 rows of A, loading B once for both 64-row halves.
/// ~2× speedup on out_proj (K=value_dim, N=h) at ISL≥128.
///
/// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_n128_m128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    b_fp8: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// M128 variant of fp8_fp8_gemm_n128: halves B re-reads for large M (ISL > 128).
///
/// Each CTA covers 128 rows of A, loading B once for both 64-row halves.
/// ~2× speedup on Q/K/V projections (FP8 activations × FP8 weights) at ISL≥128.
/// Compact FP8 A smem → 6 blocks/SM vs 3 for fp8_gemm_t_m128.
///
/// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn fp8_fp8_gemm_n128_m128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_fp8: DevicePtr,
    b_fp8: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([128, 1, 1])
        .arg_ptr(a_fp8)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Dense BF16 GEMV (M=1): C = A @ B^T for single-row activations.
///
/// A: [1, K] BF16, B: [N, K] BF16, C: [1, N] BF16.
/// 8 outputs/block, 32 threads (1 warp) per output. Single-warp shuffle reduction.
///
/// Kernel: `dense_gemv_bf16(A, B, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn dense_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Dense FP8-weight GEMV (M=1): C = A @ (dequant(B_fp8) * row_scale).
///
/// A: `[1, K]` BF16, B: `[N, K]` FP8 E4M3, row_scale: `[N]` f32, C: `[1, N]` BF16.
/// Halves weight bandwidth vs dense_gemv (1 byte/weight instead of 2).
/// 4 outputs/block, 64 threads (2 warps) per output.
///
/// Kernel: `dense_gemv_fp8w(A, B, row_scale, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn dense_gemv_fp8w(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W8A16 GEMV (M=1): C = A @ dequant_lut(B_fp8) * row_scale for FP8 E4M3 weights.
///
/// A: `[1, K]` BF16, B: `[N, K]` FP8 E4M3 bytes, row_scale: `[N]` f32, C: `[1, N]` BF16.
/// Uses a 256-entry E4M3 LUT in shared memory for branchless dequant (no hardware
/// FP4/FP8 conversion PTX needed — works on SM121 without `cvt.rn.satfinite`).
/// 4 outputs/block, 64 threads (2 warps) per output. Cross-warp smem reduction.
///
/// Kernel: `w8a16_gemv(A, B, row_scale, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    row_scale: DevicePtr,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(row_scale)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W8A16 GEMM (M>1): `C[M,N] = A[M,K] @ dequant(B[N,K])` for prefill.
///
/// Uses 256-entry E4M3 LUT + BF16 2D block scales.
/// Grid: (ceil(N/64), ceil(M/64), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    // Launch geometry is target-specific because the `w8a16_gemm` kernel SOURCE
    // differs per target. The native-HIP (gfx1151) kernel is a 256×128 M×N
    // tile / 512-thread (16-warp) block (kernels/strix-hip/common/w8a16_gemm.cu)
    // — it raises warp occupancy and per-CTA M-reuse for prefill GEMM. Every
    // other target keeps the original 64×64 / 128-thread kernel
    // (kernels/gb10/common/w8a16_gemm.cu). Keep these two in lockstep with their
    // `.cu` `M_TILE`/`N_TILE`/`THREADS`.
    #[cfg(atlas_hip)]
    let (grid, block) = ([div_ceil(n, 128), div_ceil(m, 256), 1], [512, 1, 1]);
    #[cfg(not(atlas_hip))]
    let (grid, block) = ([div_ceil(n, 64), div_ceil(m, 64), 1], [128, 1, 1]);
    KernelLaunch::new(gpu, kernel)
        .grid(grid)
        .block(block)
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W8A16 GEMM pipelined (M>1): bit-identical (cosine=1.0) faster rewrite of
/// `w8a16_gemm` — same args, same numerics, ~4.6× faster on GB10/sm_121.
///
/// Fix-A occupancy + cp.async pipelined kernel: 128×32 tile (M×N), 256-thread
/// block (8 warps). Geometry mirrors the validated `w8a16_microtest`
/// `"w8a16_gemm_pipelined"` arm (PM_M_TILE=128, PM_N_TILE=32).
///
/// Grid: (ceil(N/32), ceil(M/128), 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_pipelined(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 32), div_ceil(m, 128), 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Per-token-per-128-K-group FP8 activation quantization. Output: A_fp8
/// [M, K] FP8 E4M3 + a_scale [M, K/128] FP32. Matches vLLM's
/// `per_token_group_quant_fp8`.
///
/// Grid: (K/128, M, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn per_token_group_quant_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input_bf16: DevicePtr,
    output_fp8: DevicePtr,
    a_scale: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    // Grid: (M, K/128, 1). Putting M on grid X (max 2^31-1) avoids the
    // 65535 limit on grid Y for large MoE total_expanded counts.
    KernelLaunch::new(gpu, kernel)
        .grid([m, k / 128, 1])
        .block([128, 1, 1])
        .arg_ptr(input_bf16)
        .arg_ptr(output_fp8)
        .arg_ptr(a_scale)
        .arg_u32(m)
        .arg_u32(k)
        .launch(stream)
}

/// W8A8 + FP32 epilogue GEMM with per-token activation scales and
/// per-block weight scales — vLLM-equivalent FP8 numerics.
///
///   C[M, N] = bf16( Σ_g (FP8 MMA over K-group g) × a_scale[M, g] × b_scale[N/128, g] )
///
/// Inputs:
///   - `a_fp8`     [M, K] FP8 E4M3
///   - `a_scale`   [M, K/128] FP32 (from per_token_group_quant_fp8)
///   - `b_fp8`     [N, K] FP8 E4M3
///   - `b_scale`   [N/128, K/128] BF16 (existing checkpoint layout)
///   - `output`    [M, N] BF16
///
/// Grid: (ceil(N/128), ceil(M/64), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_t_blockscaled(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_fp8: DevicePtr,
    a_scale: DevicePtr,
    b_fp8: DevicePtr,
    b_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(a_fp8)
        .arg_ptr(a_scale)
        .arg_ptr(b_fp8)
        .arg_ptr(b_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Fused gate GEMV + topK softmax for M=1 decode.
///
/// Single kernel that computes `gate[num_experts] = A[K] @ B_gate[num_experts, K]`
/// then extracts top-K indices + softmax weights. Saves 1 launch vs separate
/// gate GEMV + topK kernels.
///
/// Grid: (1, 1, 1)  Block: (256, 1, 1) — single CTA, uses shared memory reduction
#[allow(clippy::too_many_arguments)]
pub fn moe_gate_topk_fused(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_weight: &QuantizedWeight,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    k: u32,
    top_k: u32,
    normalize: u32,
    stream: u64,
) -> Result<()> {
    // Dynamic shared memory: K BF16 values for input broadcast
    let smem_bytes = k as usize * 2;
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .shared_mem(smem_bytes as u32)
        .arg_ptr(input)
        .arg_ptr(gate_weight.weight)
        .arg_ptr(gate_weight.weight_scale)
        .arg_f32(gate_weight.weight_scale_2)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(k)
        .arg_u32(top_k)
        .arg_u32(normalize)
        .launch(stream)
}

/// Build the compacted (expert, m_tile, n_tile) work-list for the
/// persistent grouped-GEMM grid. Single-block, thread-0 serial — mirrors the
/// `moe_sort_by_expert` launch style (grid `[1,1,1]`, block `[256,1,1]`).
///
/// `n_tiles = div_ceil(N, 64)` (PM4_N_TILE) and `m_tile = 128` (PM4_M_TILE).
/// Writes `worklist[*total_tiles * 2]` (word0=expert, word1=(m_tile<<6)|n_tile)
/// and `total_tiles[0]`.
///
/// SAME-STREAM INVARIANT: the caller MUST launch `moe_fp8_grouped_gemm` on
/// the SAME `stream` so the kernel's read of `total_tiles`/`worklist`
/// happens-after this write (no cross-stream event is inserted).
#[allow(clippy::too_many_arguments)]
pub fn moe_build_tile_worklist(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    expert_offsets: DevicePtr, // [num_experts + 1]
    weight_ptrs: DevicePtr,    // [num_experts] → [N, K] FP8 (0 = remote)
    worklist: DevicePtr,       // [worst_case_tiles * 2] u32 (out)
    total_tiles: DevicePtr,    // [1] i32 (out)
    num_experts: u32,
    n_tiles: u32, // div_ceil(N, 64) — PM4_N_TILE
    m_tile: u32,  // PM4_M_TILE = 128
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(expert_offsets)
        .arg_ptr(weight_ptrs)
        .arg_ptr(worklist)
        .arg_ptr(total_tiles)
        .arg_u32(num_experts)
        .arg_u32(n_tiles)
        .arg_u32(m_tile)
        .launch(stream)
}

/// FP8 grouped GEMM for sorted MoE prefill — grid-compaction over the COMPACTED
/// work-list built by `moe_build_tile_worklist`. THE routed-expert FP8 prefill
/// kernel.
///
/// The kernel grid-strides by `gridDim.x`, so the launch is sized to
/// `max_tiles` — the caller's exact upper bound on the work-item (tile) count
/// (`wl_cap_items`). This covers the whole work-list in ~one pass instead of
/// serializing dozens of tiles per CTA behind sync barriers (the old fixed
/// 96-CTA persistent grid left the GPU >90% idle: ~0.2% occupancy / ~16%
/// MemUnitBusy, measured on gfx1151). Oversubscription is safe (extra CTAs
/// exit the loop immediately); undersizing is merely slower, never wrong.
///
/// `max_tiles` is clamped to `MAX_GRID_CTAS` so a pathological worklist bound
/// cannot request an unbounded grid.
///
/// SAME-STREAM INVARIANT: MUST be launched on the SAME `stream` as the
/// preceding `moe_build_tile_worklist` (read-after-write of `total_tiles`).
///
/// Grid: (max_tiles.clamp(1, MAX_GRID_CTAS), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_fp8_grouped_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,            // [total_tokens, K] BF16
    weight_ptrs: DevicePtr,      // [num_experts] → [N, K] FP8
    scale_ptrs: DevicePtr,       // [num_experts] → [N/128, K/128] FP32
    output: DevicePtr,           // [total_expanded, N] BF16
    expert_offsets: DevicePtr,   // [num_experts + 1]
    sorted_token_ids: DevicePtr, // [total_expanded] or NULL
    num_experts: u32,
    n: u32,
    k: u32,
    worklist: DevicePtr,    // [*total_tiles * 2] u32 (built on the same stream)
    total_tiles: DevicePtr, // [1] i32 (built on the same stream)
    max_tiles: u32,         // caller's upper bound on tile count (wl_cap_items)
    stream: u64,
) -> Result<()> {
    // The kernel strides by gridDim.x, so the grid is sized to the work-list's
    // tile-count upper bound. Clamp to MAX_GRID_CTAS to bound the launch.
    const MAX_GRID_CTAS: u32 = 16384;
    let grid_ctas = max_tiles.clamp(1, MAX_GRID_CTAS);
    // Block size is target-specific because the kernel SOURCE differs. The
    // native-HIP (gfx1151) kernel is a 16-warp / 512-thread block with a 2-D
    // (8 warp-rows x 2 warp-cols) warp grid: it keeps the 128x64 tile geometry
    // (so the work-list packing is unchanged) but splits the 4 WMMA n-sub-tiles
    // across 2 warp-columns, doubling warp occupancy for latency hiding on the
    // long-K gate/up GEMM (kernels/strix-hip/common/moe_fp8_grouped_gemm.cu).
    // Every other target keeps the 8-warp / 256-thread M-only kernel
    // (kernels/gb10/common/moe_fp8_grouped_gemm.cu). Keep this in lockstep with
    // that .cu PM4_THREADS.
    #[cfg(atlas_hip)]
    let block = [512u32, 1, 1];
    #[cfg(not(atlas_hip))]
    let block = [256u32, 1, 1];
    KernelLaunch::new(gpu, kernel)
        .grid([grid_ctas, 1, 1])
        .block(block)
        .arg_ptr(input)
        .arg_ptr(weight_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n)
        .arg_u32(k)
        .arg_ptr(worklist)
        .arg_ptr(total_tiles)
        .launch(stream)
}

/// W8A8 + FP32 epilogue grouped MoE GEMM (vLLM-equivalent).
///
/// A_fp8 must be pre-quantized via `per_token_group_quant_fp8`. Both
/// `a_scale` (per-token, FP32) and `b_scale` (per-block, BF16) are applied
/// in the FP32 epilogue per K=128 block.
#[allow(clippy::too_many_arguments)]
pub fn moe_w8a8_grouped_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_fp8: DevicePtr,            // [total_tokens, K] FP8 E4M3
    a_scale: DevicePtr,          // [total_tokens, K/128] FP32
    weight_ptrs: DevicePtr,      // [num_experts] → [N, K] FP8
    scale_ptrs: DevicePtr,       // [num_experts] → [N/128, K/128] BF16
    output: DevicePtr,           // [total_expanded, N] BF16
    expert_offsets: DevicePtr,   // [num_experts + 1]
    sorted_token_ids: DevicePtr, // [total_expanded] or NULL
    num_experts: u32,
    n: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a_fp8)
        .arg_ptr(a_scale)
        .arg_ptr(weight_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// BF16 grouped GEMM for sorted MoE prefill (FP8-dequant-on-load path).
///
/// BF16 activations × BF16 expert weights via pointer table. No scale.
/// Used when expert weights have been dequanted from FP8 to BF16 at load
/// time (ATLAS_FP8_DEQUANT_MOE_TO_BF16=1). Eliminates the per-layer 0.989
/// cosine ceiling that comes from FP8 quantization itself.
///
/// Grid: (ceil(N/64), max_m_tiles, num_experts)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_bf16_grouped_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,            // [total_tokens, K] BF16
    weight_ptrs: DevicePtr,      // [num_experts] → [N, K] BF16
    output: DevicePtr,           // [total_expanded, N] BF16
    expert_offsets: DevicePtr,   // [num_experts + 1]
    sorted_token_ids: DevicePtr, // [total_expanded] or NULL
    num_experts: u32,
    n: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight_ptrs)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W8A16 Transposed GEMM: `C[M,N] = A[M,K] @ dequant(B_t[K,N])` with coalesced reads.
///
/// Uses transposed FP8 weights `B_t[K,N]` and `block_scale_t[K/128, N/128]` for
/// coalesced N-dimension reads. ~14x faster than non-transposed w8a16_gemm at long M.
/// Grid: (ceil(N/64), ceil(M/64), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight_t: DevicePtr,      // [K, N] FP8 transposed
    block_scale_t: DevicePtr, // [K/128, N/128] BF16 transposed
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight_t)
        .arg_ptr(block_scale_t)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Pipelined transposed W8A16 GEMM (kernel `w8a16_gemm_t_pipelined`): same
/// transposed args as `w8a16_gemm_t`, ~4.2x via smem-LUT + K_STEP32 +
/// K-contiguous smem_B + 128x32 occupancy tile.
/// Grid: (ceil(N/32), ceil(M/128), 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_t_pipelined(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight_t: DevicePtr,
    block_scale_t: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 32), div_ceil(m, 128), 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight_t)
        .arg_ptr(block_scale_t)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Transpose FP8 weight matrix on GPU: `B[N,K]` → `B_t[K,N]`.
/// Grid: (ceil(N*K/256), 1, 1)  Block: (256, 1, 1)
pub fn transpose_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr, // [N, K]
    dst: DevicePtr, // [K, N]
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let total = n as u64 * k as u64;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Widen an FP8 block-scale tensor to FP32 on the GPU.
///
/// `src` is `[total]` BF16 (`in_is_fp32 == false`) or FP32 (`in_is_fp32 ==
/// true`); `dst` is `[total]` FP32. Lossless BF16→FP32 widen / straight copy.
/// Run once at load so downstream FP8 block-scale kernels read `const float*`.
/// Grid: (ceil(total/256), 1, 1)  Block: (256, 1, 1)
pub fn widen_block_scale_f32(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    total: u32,
    in_is_fp32: bool,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(total)
        .arg_u32(in_is_fp32 as u32)
        .launch(stream)
}

/// Transpose block scales: [N/128, K/128] → [K/128, N/128].
pub fn transpose_block_scale(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    n_blocks: u32,
    k_blocks: u32,
    stream: u64,
) -> Result<()> {
    let total = n_blocks * k_blocks;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(n_blocks)
        .arg_u32(k_blocks)
        .launch(stream)
}

// ── Unified quantization dispatch ────────────────────────────────────
//
// These wrappers select the correct kernel based on the QuantWeight
// variant. Adding a new quant format requires only a new match arm here.
