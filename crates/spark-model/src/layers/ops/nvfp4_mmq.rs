// SPDX-License-Identifier: AGPL-3.0-only
//
// Launchers for the vendored llama NVFP4 W4A4 MMQ FFN prefill GEMM (ATLAS_FFN_NVFP4_MMQ).
// Kernels in kernels/gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu (Blackwell block-scale MMA
// kind::mxf4nvf4.m16n8k64, e2m1×e2m1, ue4m3 group-16 scales).
// Microbench (GB10, M=4096): gate/up 80.2 TFLOP/s, down 79.7 — vs w4a16 t_m128 ~51 (1.57x).
// Correctness: rel_err 1.6e-3 vs same-quant CPU ref (= bf16-output rounding); the hardware
// decodes ue4m3 scales as STANDARD e4m3 on both operands, so the checkpoint's per-16 scale
// bytes are byte-copy correct and the only missing factor is the per-tensor FP32 scale2 —
// folded by the caller in atlas_nvfp4_silu_mul_scaled (empirical ratio 0.99 ≈ 1.0, see
// scratchpad nvfp4_mmq_bench.cu).
// Pipeline: weights repacked ONCE at load (raw bit shuffle, checkpoint layout →
// block_nvfp4); per prefill: activations bf16 → block_fp4_mmq (shared ffn_act_q8 scratch),
// then MMQ → bf16 out; scale2 folded in the SiLU-mul.
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// NVFP4 block: 64 weights -> 36-byte block_nvfp4 {4×ue4m3 scales, 32B e2m1 nibbles}.
pub const QK_NVFP4: u32 = 64;
pub const NVFP4_BLOCK_BYTES: usize = 36;
/// block_fp4_mmq (activation y): 256 values -> 144 bytes (== block_q8_1_mmq).
const FP4_MMQ_Y_BLOCK_VALS: u32 = 256;
const FP4_MMQ_Y_BLOCK_BYTES: usize = 144;
/// Dynamic shared memory: ids(512) + x-tile(128*MMQ_MMA_TILE_X_K_FP4=76*4) + y-tile(128*144).
pub const NVFP4_MMQ_SMEM: u32 = 57856;
const QUANT_BLOCK_THREADS: u32 = 128;

/// Bytes for the block_nvfp4 form of an [n, k] weight (k % 64 == 0).
pub fn nvfp4_mmq_weight_bytes(n: u32, k: u32) -> usize {
    (n as usize) * (k as usize / QK_NVFP4 as usize) * NVFP4_BLOCK_BYTES
}

/// block_fp4_mmq activation scratch bytes for [m, k]. +1MB slack: the kernel's smem copy
/// loop rounds the last y-slice read up to warp granularity (same convention as
/// q8_1_scratch_bytes). Always ≤ q8_1_scratch_bytes(m, k) → fits the shared ffn_act_q8.
pub fn fp4_act_scratch_bytes(m: u32, k: u32) -> usize {
    let bpc = div_ceil(k, FP4_MMQ_Y_BLOCK_VALS) as usize;
    (m as usize) * bpc * FP4_MMQ_Y_BLOCK_BYTES + (1 << 20)
}

/// Repack a checkpoint NVFP4 weight (packed E2M1 [n, k/2] low=even/high=odd + E4M3
/// [n, k/16] scales) into llama block_nvfp4 rows \[n\]\[k/64\]. Raw bit shuffle — the e2m1
/// codes and e4m3 scale bytes are reused verbatim (scale2 folded downstream).
pub fn nvfp4_mmq_repack(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle, // atlas_nvfp4_repack
    packed: DevicePtr,
    scales: DevicePtr,
    out_blocks: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let nblocks = (n as u64) * (k as u64 / QK_NVFP4 as u64);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(nblocks as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(packed)
        .arg_ptr(scales)
        .arg_ptr(out_blocks)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Quantize bf16 activations [m, k] -> block_fp4_mmq (e2m1 + ue4m3 group-16, ±2 scale
/// search) into `out_y`. One thread per 16-value group; ne0 padded to 256.
pub fn nvfp4_mmq_quantize_act(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle, // atlas_nvfp4_quantize_bf16
    input_bf16: DevicePtr,
    out_y: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let kpad = div_ceil(k, FP4_MMQ_Y_BLOCK_VALS) * FP4_MMQ_Y_BLOCK_VALS;
    let grid_y = div_ceil(kpad, 16 * QUANT_BLOCK_THREADS);
    KernelLaunch::new(gpu, kernel)
        .grid([m, grid_y, 1])
        .block([QUANT_BLOCK_THREADS, 1, 1])
        .arg_ptr(input_bf16)
        .arg_ptr(out_y)
        .arg_u64(k as u64) // ne00
        .arg_u64(k as u64) // s01 (contiguous rows)
        .arg_u64(kpad as u64) // ne0
        .arg_u32(m) // ne1
        .launch(stream)
}

/// NVFP4 W4A4 MMQ GEMM: C\[m,n\] (bf16, missing ×scale2) = A_fp4\[m,k\] x W_nvfp4\[n,k\].
pub fn nvfp4_mmq_gemm(
    gpu: &dyn GpuBackend,
    kernel_nc: KernelHandle, // atlas_nvfp4_mmq128_nc
    kernel_wc: KernelHandle, // atlas_nvfp4_mmq128_wc
    a_fp4: DevicePtr,        // block_fp4_mmq activations
    w_nvfp4: DevicePtr,      // block_nvfp4 weights [n, k]
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
        .shared_mem(NVFP4_MMQ_SMEM)
        .arg_ptr(w_nvfp4) // x = weights
        .arg_ptr(a_fp4) // y = fp4 activations
        .arg_ptr(out_bf16) // dst
        .arg_u32(n) // nrows_x
        .arg_u32(m) // ncols_dst
        .arg_u32(k) // ncols_x
        .arg_u32(k / QK_NVFP4) // stride_row_x = K/64
        .arg_u32(m) // ncols_y
        .arg_u32(n) // stride_col_dst
        .launch(stream)
}

/// Fused SiLU-mul + block_fp4_mmq quantize for the down-MMQ path: reads RAW gate/up MMQ
/// outputs, applies the scale2 folds + swiglu clamp + SiLU-mul, and quantizes straight
/// into the down GEMM's y-format — the intermediate bf16 activation tensor is never
/// written (this round-trip is why the unfused down arm measured neutral).
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_silu_mul_quant(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle, // atlas_nvfp4_silu_mul_quant
    gate: DevicePtr,
    up: DevicePtr,
    out_y: DevicePtr,
    gate_scale: f32,
    up_scale: f32,
    m: u32,
    k: u32, // inter
    stream: u64,
) -> Result<()> {
    let kpad = div_ceil(k, FP4_MMQ_Y_BLOCK_VALS) * FP4_MMQ_Y_BLOCK_VALS;
    let grid_y = div_ceil(kpad, 16 * QUANT_BLOCK_THREADS);
    KernelLaunch::new(gpu, kernel)
        .grid([m, grid_y, 1])
        .block([QUANT_BLOCK_THREADS, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(out_y)
        .arg_f32(gate_scale)
        .arg_f32(up_scale)
        .arg_u64(k as u64) // ne00
        .arg_u64(kpad as u64) // ne0
        .arg_u32(m) // ne1
        .launch(stream)
}

/// In-place ×scale2 for the down-projection MMQ output ([m, h] bf16).
pub fn nvfp4_scale_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle, // atlas_nvfp4_scale_bf16
    data: DevicePtr,
    scale: f32,
    total: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(data)
        .arg_f32(scale)
        .arg_u32(total)
        .launch(stream)
}

/// SiLU(gate×gs)×(up×us) with the per-projection scale2 fold (swiglu ±10 clamp,
/// mirrors moe_silu_mul). In-place safe (out may alias gate).
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_silu_mul_scaled(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle, // atlas_nvfp4_silu_mul_scaled
    gate: DevicePtr,
    up: DevicePtr,
    out: DevicePtr,
    gate_scale: f32,
    up_scale: f32,
    total: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(out)
        .arg_f32(gate_scale)
        .arg_f32(up_scale)
        .arg_u32(total)
        .launch(stream)
}
