// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Result, ensure};
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

/// Dense BF16 GEMV, batched over 2 rows (M=2): one pass over the weight
/// produces both output rows, halving weight bandwidth vs two `dense_gemv`
/// launches. Bit-identical to two M=1 `dense_gemv` calls — each row's
/// accumulator follows the same K-iteration/reduction order.
///
/// `input`: `[2, K]` BF16 (contiguous); `output`: two rows at
/// `output + t * out_stride` (BF16 elements). Used by the K=2 MTP verify
/// path for the GDN `in_proj_qkvz` (dequant-to-BF16 on FP8 checkpoints),
/// which otherwise re-read the full projection weight once per verify token.
///
/// Kernel: `dense_gemv_bf16_batch2(A, B, C, N, K, out_stride)`
#[allow(clippy::too_many_arguments)]
pub fn dense_gemv_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    out_stride: u32,
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
        .arg_u32(out_stride)
        .launch(stream)
}

/// Dense BF16 batched GEMV (M rows): `C[t] = A[t] @ B^T` for `t` in `[0, M)`.
///
/// The M-row generalisation of [`dense_gemv_batch2`]. Reads the BF16 weight
/// matrix ONCE for all M rows instead of M times, which is the whole point:
/// at decode the BF16 projections (q/k/v/o + shared expert) are pure weight
/// streaming, so M separate M=1 GEMVs make the step scale linearly with the
/// number of concurrent sequences.
///
/// Bit-identical to M separate `dense_gemv` calls (same K-iteration order and
/// reduction tree per row; the kernel dir builds with --fmad=false).
///
/// `input`: `[M, K]` BF16 contiguous. `output`: M rows at
/// `output + t * out_stride` (BF16 elements). Caller must pass `m <= 8`
/// (MAX_M in the kernel); larger batches should use a tiled GEMM.
///
/// Kernel: `dense_gemv_bf16_batchm(A, B, C, M, N, K, out_stride)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
/// Mirror of `MAX_M` in `kernels/gb10/common/dense_gemv_bf16_batchm.cu`.
/// The kernel clamps silently above this, so the Rust side must refuse.
///
/// 🔴 16 since 2026-09-02. The old 8 was the kernel's compiled row array, never an
/// arithmetic boundary: each row is an independent FP32 accumulator over the same `kv`
/// order, `m` appears in no row's operand sequence, and the fold is per-row. So every
/// width up to `MAX_M` is bit-identical both to the narrower tier and to M serial
/// `dense_gemv_bf16` calls. Verified on the 12 real GLM-5.3 prefill shapes with cold
/// weights, including the regression direction that matters — m <= 8 byte-unchanged,
/// because decode, the MTP verify arm and the BF16 lm_head arm all run m <= 8 on this
/// same kernel (`scripts/glm53-dense-bf16/bench_m16.cu`, spark-bench).
///
/// 🪤 This constant is load-bearing OUTSIDE the GEMV: it gates the lm_head batched arm
/// (`model/impl_a3.rs`), the MTP row dispatch (`layers/mtp_head/row_dispatch.rs`) and it
/// sizes `verify_k` for the KDA/DSA/MLP workspaces (`weight_loader/glm5_next_load.rs`).
/// Raising it widens those arms and grows per-layer scratch — a memory-budget change, not
/// only a kernel one.
pub const DENSE_GEMV_BATCHM_MAX_M: u32 = 16;

/// The band the batched GEMV is allowed to CLAIM on the decode paths: the MTP row dispatch
/// and the BF16 lm_head arm.
///
/// 🔴 Deliberately still 8, and NOT the same thing as the kernel's `MAX_M`. Those two sites
/// pick between `dense_gemv_bf16_batchm` and a **reassociating** kernel (the pipelined /
/// tile GEMM), so the band's upper edge decides which bits a decode of that width produces.
/// Widening the GEMV tier to 16 for prefill would silently move widths 9..=16 off the tile
/// GEMM they have always used — a numerics change on the MTP / DFlash γ>8 window, on a path
/// the prefill measurement says nothing about. Moving this edge needs its own A/B and its
/// own byte gate against the sealed decode reference; until then the decode band is frozen
/// where it was measured (+6 % at C=2, +24 % at C=4; NEGATIVE above 8 against the tile GEMM,
/// -14.4 % at C=16 — commits 84d5b763c / 78d276832).
pub const DENSE_GEMV_BATCHM_DECODE_MAX_M: u32 = 8;

pub fn dense_gemv_batchm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    // The kernel caps rows at a compile-time MAX_M 8 and CLAMPS rather than
    // erroring, so an over-large m used to mean "rows 8..m are silently never
    // written". Refuse instead: a caller that wants more rows must use a
    // kernel that can do them (dense_gemm_tc), not get 8 rows of truth and
    // stale memory for the rest.
    ensure!(
        (1..=DENSE_GEMV_BATCHM_MAX_M).contains(&m),
        "dense_gemv_batchm: m={m} outside 1..={DENSE_GEMV_BATCHM_MAX_M} \
         (kernel MAX_M clamps silently; use dense_gemm_tc for wider batches)"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
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
/// Launch geometry is target-specific because the KERNEL is, exactly as it is
/// for `w8a16_gemm` above: [`Fp8ActQuant`] carries both handles and hands back
/// the entry point and the grid TOGETHER, so a Hopper handle can never be
/// launched on the shared kernel's grid. Block is 128 threads in both arms.
///
///   shared (`per_token_group_quant_fp8`)        Grid: (M, K/128, 1)
///   hopper (`per_token_group_quant_fp8_hopper`) Grid: (M, ceil(K/128 / 8), 1)
///
/// WHICH of the two runs is `Fp8ActQuant::pick`, and it is width-dependent:
/// the twin is 3.30-3.59x at prefill M and 0.76x-0.95x at M <= 25 for
/// K in {5120, 6144} (round-16 receipt SS 2.1), so it takes the launch only
/// when its own grid clears `2 x sm_count` CTAs. Rule and thresholds:
/// `layers/ops/fp8_act_quant_floor.rs`. The route line is said ONCE PER
/// BRANCH from here — this is the single launch site, so a serve log carries
/// the positive at the first prefill width and the negative at the first
/// decode width.
///
/// M on grid X (max 2^31-1) in both: grid Y stops at 65535 and MoE
/// `total_expanded` exceeds it. Keep the Hopper arm in lockstep with
/// `kernels/hopper/common/fp8_act_quant_hopper.cu` — it re-derives its own
/// group span from `gridDim.y`, so any Y in `1..=K/128` is CORRECT and this
/// one is merely the fast one. Both kernels emit bit-identical FP8 bytes and
/// scales (#928; `native_fp8_act_quant_hopper_microtest`).
#[allow(clippy::too_many_arguments)]
pub fn per_token_group_quant_fp8(
    gpu: &dyn GpuBackend,
    quant: Fp8ActQuant,
    input_bf16: DevicePtr,
    output_fp8: DevicePtr,
    a_scale: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let pick = quant.pick(m, k);
    super::fp8_quant_log(&pick, m, k);
    KernelLaunch::new(gpu, pick.kernel)
        .grid(pick.grid)
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
    super::log_gemm_shape(gpu, "fp8_gemm_t_blockscaled", m, n, k);
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

/// W8A8 + FP32 epilogue grouped MoE GEMM — PM4 geometry over the COMPACTED
/// work-list built by `moe_build_tile_worklist` (kernel
/// `moe_w8a8_grouped_gemm_pm4`, same module/numerics as
/// `moe_w8a8_grouped_gemm`: bit-identical output, measured).
///
/// Same grid-compaction contract as `moe_fp8_grouped_gemm`: the kernel
/// grid-strides by `gridDim.x` over the work-list, so the launch is sized to
/// `max_tiles` (`wl_cap_items`), clamped to `MAX_GRID_CTAS`. Oversubscription
/// is safe; undersizing is merely slower, never wrong.
///
/// SAME-STREAM INVARIANT: MUST be launched on the SAME `stream` as the
/// preceding `moe_build_tile_worklist` (read-after-write of `total_tiles`).
///
/// Grid: (max_tiles.clamp(1, MAX_GRID_CTAS), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_w8a8_grouped_gemm_pm4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_fp8: DevicePtr,            // [total_tokens, K] FP8 E4M3
    a_scale: DevicePtr,          // [total_tokens, K/128] FP32
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
    const MAX_GRID_CTAS: u32 = 16384;
    let grid_ctas = max_tiles.clamp(1, MAX_GRID_CTAS);
    // gb10-only kernel (256 threads, __launch_bounds__(256,2)); other targets
    // fall back to the dense-grid `moe_w8a8_grouped_gemm` (handle gating at
    // the dispatch site).
    KernelLaunch::new(gpu, kernel)
        .grid([grid_ctas, 1, 1])
        .block([256, 1, 1])
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
        .arg_ptr(worklist)
        .arg_ptr(total_tiles)
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

/// W8A16 transposed M128 GEMM (kernel `w8a16_gemm_t_m128`): FP8 E4M3 analog of
/// `w4a16_gemm_n128_m128_v2`. 128×128 (M×N) tile, two 64-row chunks, 8 warps,
/// parallel-chunk `m16n8k16.bf16.bf16` MMA + two-level FP32 block-scale fold.
/// Same transposed contract as `w8a16_gemm_t` (`B_t[K,N]` + block_scale_t[K/128,
/// N/128]); reuses the transpose_fp8 / transpose_block_scale output as-is.
/// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_n128_m128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight_t: DevicePtr,      // [K, N] FP8 transposed
    block_scale_t: DevicePtr, // [K/128, N/128] FP32 transposed
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    super::log_gemm_shape(gpu, "w8a16_gemm_t_m128", m, n, k);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
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
    super::log_gemm_shape(gpu, "w8a16_gemm_t_pipelined", m, n, k);
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
/// `src` is `[total]` BF16 (0), FP32 (1), or F8_E8M0 (2); `dst` is `[total]`
/// FP32. E8M0 uses the exact `exp << 23` power-of-two representation.
/// Run once at load so downstream FP8 block-scale kernels read `const float*`.
/// Grid: (ceil(total/256), 1, 1)  Block: (256, 1, 1)
pub fn widen_block_scale_f32(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    total: u32,
    input_dtype: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(total)
        .arg_u32(input_dtype)
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

/// The three BF16 dense kernels one projection site can land on, resolved once.
///
/// GLM-5.3 binds one of these per mixer/MLP site; `batchm` is `0` on a backend that
/// does not carry `dense_gemv_bf16_batchm`, and [`dense_mm_bf16`] then falls back to
/// the tile GEMM exactly as before.
#[derive(Clone, Copy)]
pub struct DenseMmKernels {
    /// `dense_gemm_bf16` — 16×16 tile GEMM. The only arm that handles `M > 8`.
    pub gemm: KernelHandle,
    /// `dense_gemv_bf16` — `M == 1`.
    pub gemv: KernelHandle,
    /// `dense_gemv_bf16_batchm` — `2 ..= 8`, ONE weight sweep. `0` = unavailable.
    pub batchm: KernelHandle,
}

/// `C[M, N] = A[M, K] @ B[N, K]^T`, BF16 in and out, output row stride `N`.
///
/// 🔴 **The M dispatch is the whole point.** At `M == 1` the tile GEMM's grid collapses
/// (73 GB/s against a 254 GB/s part). At `2 ..= 8` it is ~94 % padding and measured 3.6×
/// SLOWER than the batched GEMV on this exact workload (`multi_seq/qkv.rs::wide_verify_gemm`).
/// `batchm` reads the weight matrix ONCE for all M rows — which is what makes a K-token
/// speculative verify cost one weight sweep instead of K.
///
/// 🪤 `batchm` is **bit-identical to M separate `dense_gemv` calls** (same K-iteration order
/// and reduction tree per row, `--fmad=false`), so batching K rows that were previously K
/// serial single-row decodes does not move a single bit. The tile-GEMM arm is NOT
/// bit-identical to either — it reassociates. Widening a site past 8 rows changes numerics.
#[allow(clippy::too_many_arguments)]
pub fn dense_mm_bf16(
    gpu: &dyn GpuBackend,
    k: &DenseMmKernels,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    // 🪤 Grid is COUPLED to each kernel's `N_PER_BLOCK` (4 outputs / 256-thread block for
    // both GEMV arms, `GEMM_TILE` for the tile arm). Never hand-roll these div_ceils.
    if m == 1 && k.gemv.0 != 0 {
        return KernelLaunch::new(gpu, k.gemv)
            .grid([div_ceil(n as u32, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(b)
            .arg_ptr(c)
            .arg_u32(n as u32)
            .arg_u32(kk as u32)
            .launch(stream);
    }
    // 🪤 A missing batchm handle falls back SILENTLY to the tile GEMM, which is 3.6x slower
    // at these widths — exactly the failure `announce_dispatch` exists to prevent elsewhere.
    if m > 1 && k.batchm.0 == 0 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                "dense_mm_bf16: no dense_gemv_bf16_batchm on this target -- M>1 sites fall \
                 back to the tile GEMM (measured 3.6x slower at M<=8)"
            );
        });
    }
    if (2..=DENSE_GEMV_BATCHM_MAX_M as usize).contains(&m) && k.batchm.0 != 0 {
        return KernelLaunch::new(gpu, k.batchm)
            .grid([div_ceil(n as u32, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(b)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(kk as u32)
            // Contiguous `[M, N]` output — the layout `dense_gemm_bf16` writes.
            .arg_u32(n as u32)
            .launch(stream);
    }
    const GEMM_TILE: u32 = 16;
    KernelLaunch::new(gpu, k.gemm)
        .grid([
            (n as u32).div_ceil(GEMM_TILE),
            (m as u32).div_ceil(GEMM_TILE),
            1,
        ])
        .block([GEMM_TILE, GEMM_TILE, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(stream)
}
