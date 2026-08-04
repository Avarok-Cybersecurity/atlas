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
    let kernel = if !n.is_multiple_of(128) {
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
