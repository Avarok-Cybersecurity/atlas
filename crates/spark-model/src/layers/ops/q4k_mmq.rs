// SPDX-License-Identifier: AGPL-3.0-only
//
// Launchers for the vendored llama Q4_K MMQ FFN prefill GEMM (ATLAS_FFN_MMQ).
// Kernels in kernels/gb10/qwen3.6-27b/nvfp4/q4k_mmq.cu + q4k_quantize.cu (verified
// 54.9/53.7 TFLOP/s gate/up·down, +25%/+10% vs faith2, rel_err 6-7e-3).
// Pipeline: weights NVFP4 -> dequant_nvfp4_to_bf16 -> q4k_quantize (at load); per-prefill
// activation bf16 -> q8_1_mmq, then MMQ -> bf16 (fused store, no cast).
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Q4_K block size (256 weights -> 144-byte block_q4_K).
pub const QK_K: u32 = 256;
/// sizeof(block_q4_K) bytes.
pub const Q4K_BLOCK_BYTES: usize = 144;
/// Dynamic shared memory for the Q4_K MMQ kernel (mmq_x=mmq_y=128, GB10). >48KB -> registry sets attr.
pub const Q4K_MMQ_SMEM: u32 = 57856;
const CUDA_QUANTIZE_BLOCK_SIZE_MMQ: u32 = 128;

/// Bytes for the Q4_K-quantized form of an [nrows, n_per_row] weight (n_per_row % 256 == 0).
pub fn q4k_weight_bytes(nrows: u32, n_per_row: u32) -> usize {
    (nrows as usize) * (n_per_row as usize / QK_K as usize) * Q4K_BLOCK_BYTES
}

/// q8_1_mmq activation scratch bytes for [m, k]; generous (kpad rounded to 256).
pub fn q8_1_scratch_bytes(m: u32, k: u32) -> usize {
    let kpad = div_ceil(k, QK_K) * QK_K;
    (m as usize) * (kpad as usize) * 4 + (1 << 20)
}

/// Dequantize NVFP4 weight [n, k] (packed E2M1 + E4M3 group scales + per-tensor scale2) -> bf16 [n, k].
pub fn dequant_nvfp4_to_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    packed: DevicePtr,
    scales: DevicePtr,
    out_bf16: DevicePtr,
    scale2: f32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(packed)
        .arg_ptr(scales)
        .arg_ptr(out_bf16)
        .arg_f32(scale2)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Quantize bf16 weights [nrows, n_per_row] -> GGML block_q4_K (at model load).
pub fn quantize_weight_q4k(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input_bf16: DevicePtr,
    out_q4k: DevicePtr,
    nrows: u32,
    n_per_row: u32,
    stream: u64,
) -> Result<()> {
    let total_sb = (nrows as u64) * (n_per_row as u64 / QK_K as u64);
    let grid_x = div_ceil(total_sb as u32, 128);
    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(input_bf16)
        .arg_ptr(out_q4k)
        .arg_u32(nrows)
        .arg_u32(n_per_row)
        .launch(stream)
}

/// Quantize bf16 activations [m, k] -> q8_1_mmq (DS4 layout) into `out_q8`.
pub fn quantize_act_q8_1(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle, // atlas_q8_1_quantize_ds4_bf16
    input_bf16: DevicePtr,
    out_q8: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let kpad = div_ceil(k, QK_K) * QK_K;
    let grid_y = div_ceil(kpad, 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    KernelLaunch::new(gpu, kernel)
        .grid([m, grid_y, 1])
        .block([CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1])
        .arg_ptr(input_bf16)
        .arg_ptr(out_q8)
        .arg_u64(k as u64) // ne00
        .arg_u64(k as u64) // s01 (contiguous rows)
        .arg_u64(kpad as u64) // ne0
        .arg_u32(m) // ne1
        .launch(stream)
}

/// Device-side GROUPED MoE MMQ: one launch over (N-tiles, worst-case M-tiles,
/// num_experts). Each CTA reads its expert's row range from `expert_offsets`
/// (device i32\[ne+1\]) — NO host readback, so it is CUDA-graph-capture-legal.
/// `weight_base` is expert 0's packed blocks (experts are one contiguous stack,
/// stride `n_out*(k/256)` blocks apart). Activations `a_q8` are the whole sorted
/// \[total_expanded, k\] q8_1 buffer (quantized once); output `dst_bf16` is written
/// in SORTED order \[total_expanded, n_out\] for the caller's unpermute-reduce.
/// `is_q6k` picks the Q6_K weight/vec_dot path (D4-quantized activations) vs Q4_K.
#[allow(clippy::too_many_arguments)]
pub fn q4k_grouped_gemm(
    gpu: &dyn GpuBackend,
    kernel_nc: KernelHandle,
    kernel_wc: KernelHandle,
    weight_base: DevicePtr,    // expert 0 packed blocks (contiguous stack)
    a_q8: DevicePtr,           // sorted q8_1 activations [total_expanded, k]
    expert_offsets: DevicePtr, // [ne+1] i32 device (sorted cumulative counts)
    dst_bf16: DevicePtr,       // sorted output [total_expanded, n_out]
    n_out: u32,                // output features per expert (nrows_x)
    k: u32,                    // input features (ncols_x, %256==0)
    num_experts: u32,
    total_expanded: u32, // sorted-buffer row count (worst-case M bound)
    stream: u64,
) -> Result<()> {
    // nc assumes n_out % 128 == 0 (true for Laguna inter=1024 / hidden=3072);
    // wc guards a ragged N. M raggedness is always handled in-kernel via
    // per-expert tile_y_max_j, independent of this choice.
    let kernel = if n_out.is_multiple_of(128) {
        kernel_nc
    } else {
        kernel_wc
    };
    let stride_row_x = k / QK_K; // K-blocks per weight row
    let stride_channel_x = n_out * stride_row_x; // per-expert block stride
    let max_m_tiles = div_ceil(total_expanded, 128);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_out, 128), max_m_tiles, num_experts])
        .block([32, 8, 1])
        .shared_mem(Q4K_MMQ_SMEM)
        .arg_ptr(weight_base)
        .arg_ptr(a_q8)
        .arg_ptr(expert_offsets)
        .arg_ptr(dst_bf16)
        .arg_u32(n_out) // nrows_x
        .arg_u32(k) // ncols_x
        .arg_u32(stride_row_x) // stride_row_x
        .arg_u32(stride_channel_x) // stride_channel_x
        .arg_u32(total_expanded) // ncols_y (y-buffer row stride)
        .arg_u32(n_out) // stride_col_dst (sorted dst row stride)
        .launch(stream)
}

/// Fused n=1 DECODE MoE GEMV (two occupancy-friendly launches): gate+up+silu
/// (staging s_act into `gate_silu` = the arena expert_gate_out), then Q4_K/Q6_K
/// down. Output tiled across grid.y for high block count (the single-block-per-
/// slot fusion starved GB10's SMs). Replaces permute+quant+grouped-GEMM at decode.
/// CUDA-graph-legal: no alloc/sync, device-side sorted ids only.
#[allow(clippy::too_many_arguments)]
pub fn moe_q4k_decode_fused(
    gpu: &dyn GpuBackend,
    gate_up_k: KernelHandle, // atlas_moe_q4k_decode_gate_up
    down_k: KernelHandle,    // atlas_moe_q4k_decode_down
    expert_input: DevicePtr,
    gate_base: DevicePtr,
    up_base: DevicePtr,
    down_base: DevicePtr,
    sorted_token_ids: DevicePtr,
    sorted_expert_ids: DevicePtr,
    gate_silu: DevicePtr, // [total_expanded, inter] BF16 scratch (expert_gate_out)
    expert_down_out: DevicePtr,
    h: u32,
    inter: u32,
    total_expanded: u32,
    down_is_q6k: bool,
    stream: u64,
) -> Result<()> {
    const DEC_BLOCK: u32 = 256;
    const DEC_WARPS: u32 = DEC_BLOCK / 32; // 8 output channels per block
    // Kernel 1: gate+up+silu → gate_silu. grid.y tiles inter across blocks.
    KernelLaunch::new(gpu, gate_up_k)
        .grid([total_expanded, div_ceil(inter, DEC_WARPS), 1])
        .block([DEC_BLOCK, 1, 1])
        .shared_mem(h * 4)
        .arg_ptr(expert_input)
        .arg_ptr(gate_base)
        .arg_ptr(up_base)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(sorted_expert_ids)
        .arg_ptr(gate_silu)
        .arg_u32(h)
        .arg_u32(inter)
        .arg_u32(total_expanded)
        .launch(stream)?;
    // Kernel 2: down. grid.y tiles h across blocks.
    KernelLaunch::new(gpu, down_k)
        .grid([total_expanded, div_ceil(h, DEC_WARPS), 1])
        .block([DEC_BLOCK, 1, 1])
        .shared_mem(inter * 4)
        .arg_ptr(gate_silu)
        .arg_ptr(down_base)
        .arg_ptr(sorted_expert_ids)
        .arg_ptr(expert_down_out)
        .arg_u32(h)
        .arg_u32(inter)
        .arg_u32(total_expanded)
        .arg_u32(if down_is_q6k { 1 } else { 0 })
        .launch(stream)
}

/// FUSED device-side GROUPED gate+up MoE MMQ: ONE launch computes BOTH the gate
/// and up projections per CTA (shared empty-expert early-return + ids setup, then
/// the verified `mul_mat_q_process_tile` twice). Halves the scheduled-CTA count
/// vs two `q4k_grouped_gemm` calls and collapses two launches into one. gate and
/// up share shape [inter, h], so every stride is identical; only the two weight
/// bases and two sorted outputs differ. smem unchanged (Q4K_MMQ_SMEM); the
/// accumulator resets between the two passes so no register spill.
#[allow(clippy::too_many_arguments)]
pub fn q4k_grouped_gemm_gate_up(
    gpu: &dyn GpuBackend,
    kernel_nc: KernelHandle,
    kernel_wc: KernelHandle,
    gate_base: DevicePtr,
    up_base: DevicePtr,
    a_q8: DevicePtr,
    expert_offsets: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    n_out: u32,
    k: u32,
    num_experts: u32,
    total_expanded: u32,
    stream: u64,
) -> Result<()> {
    let kernel = if n_out.is_multiple_of(128) {
        kernel_nc
    } else {
        kernel_wc
    };
    let stride_row_x = k / QK_K;
    let stride_channel_x = n_out * stride_row_x; // per-expert block stride (gate == up)
    let max_m_tiles = div_ceil(total_expanded, 128);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_out, 128), max_m_tiles, num_experts]) // grid.x NOT doubled
        .block([32, 8, 1])
        .shared_mem(Q4K_MMQ_SMEM)
        .arg_ptr(gate_base)
        .arg_ptr(up_base)
        .arg_ptr(a_q8)
        .arg_ptr(expert_offsets)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_u32(n_out) // nrows_x
        .arg_u32(k) // ncols_x
        .arg_u32(stride_row_x) // stride_row_x
        .arg_u32(stride_channel_x) // stride_channel_x (both projections)
        .arg_u32(total_expanded) // ncols_y
        .arg_u32(n_out) // stride_col_dst
        .launch(stream)
}

/// Q4_K MMQ GEMM: C\[m,n\] (bf16) = A_q8\[m,k\] x W_q4k\[n,k\]. Fused bf16 store.
pub fn q4k_mmq_gemm(
    gpu: &dyn GpuBackend,
    kernel_nc: KernelHandle, // atlas_q4k_mmq128_nc
    kernel_wc: KernelHandle, // atlas_q4k_mmq128_wc
    a_q8: DevicePtr,         // q8_1_mmq activations
    w_q4k: DevicePtr,        // block_q4_K weights [n, k]
    out_bf16: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    // `nc` (no-check) assumes BOTH the N tiles and the M (ncols_dst) tiles are
    // full 128-wide; it is only safe when n % 128 == 0 AND m % 128 == 0. The
    // dense-FFN caller always pads m, but the keep-packed MoE arm drives this
    // per-expert with a RAGGED m (each expert's routed-row count), so force the
    // bounds-checked `wc` variant whenever either dim is not tile-aligned —
    // otherwise a ragged-m tile writes past the output slice into adjacent
    // arena buffers (observed as a downstream CUDA_ERROR_ILLEGAL_ADDRESS).
    let kernel = if !n.is_multiple_of(128) || !m.is_multiple_of(128) {
        kernel_wc
    } else {
        kernel_nc
    };
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([32, 8, 1])
        .shared_mem(Q4K_MMQ_SMEM)
        .arg_ptr(w_q4k) // x = weights
        .arg_ptr(a_q8) // y = q8_1 activations
        .arg_ptr(out_bf16) // dst
        .arg_u32(n) // nrows_x
        .arg_u32(m) // ncols_dst
        .arg_u32(k) // ncols_x
        .arg_u32(k / QK_K) // stride_row_x = K/256
        .arg_u32(m) // ncols_y
        .arg_u32(n) // stride_col_dst
        .launch(stream)
}
