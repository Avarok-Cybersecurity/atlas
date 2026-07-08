// SPDX-License-Identifier: AGPL-3.0-only

//! int8 W4A8 prefill GEMM wrappers, extracted piecewise from `gemm_dense.rs`
//! (500-LoC cap): one-time NVFP4→int8 weight requant + the faith2 two-launch
//! prefill (activation requant → int8×int8 block-scaled GEMM).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Requant an NVFP4 weight (packed E2M1 + per-16 E4M3 block scales + per-tensor
/// `scale2`) into an int8 weight + per-32 F32 block scale, for the int8 W4A8
/// prefill GEMM (`int8_gemm_faith2`). One-time conversion per weight at load (or
/// lazily on first int8 prefill).
///
/// Reads `W_packed[N, K/2]`, `W_e4m3[N, K/16]`, `scale2` → `W_i8[N, K]` (signed
/// int8) + `W_scale[N, K/32]` (F32). The per-16 NVFP4 scales are re-blocked to
/// per-32 int8 scales by the kernel.
///
/// Grid: (ceil(N*(K/32) / 128), 1, 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn requant_w_nvfp4_int8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    w_packed: DevicePtr,
    w_e4m3: DevicePtr,
    scale2: f32,
    w_i8: DevicePtr,
    w_scale: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let blocks = n * (k / 32);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(blocks, 128), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(w_packed)
        .arg_ptr(w_e4m3)
        .arg_f32(scale2)
        .arg_ptr(w_i8)
        .arg_ptr(w_scale)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// int8 W4A8 prefill GEMM: requant BF16 activations to int8 (per-32 F32 scale)
/// then `C = (A_i8 * W_i8)` folded with per-32 A/W block scales via
/// `int8_gemm_faith2`. The weight is already int8 (see `requant_w_nvfp4_int8`).
///
/// A_bf16: [M, K] BF16 activations. W_i8: [N, K] int8 weights. W_scale: [N, K/32]
/// F32. `a_i8_scratch` / `a_scale_scratch` are caller-owned scratch buffers of at
/// least `M*K` bytes and `M*(K/32)*4` bytes respectively. Out: [M, N] BF16.
///
/// Two launches on `stream` (stream-ordered): requant_a → faith2.
///   requant_a grid: (ceil(M*(K/32) / 128), 1, 1)  block: (128, 1, 1)
///   faith2    grid: (ceil(N/128), ceil(M/128), 1)  block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn int8_gemm_faith2_prefill(
    gpu: &dyn GpuBackend,
    faith2_kernel: KernelHandle,
    requant_a_kernel: KernelHandle,
    a_bf16: DevicePtr,
    w_i8: DevicePtr,
    w_scale: DevicePtr,
    a_i8_scratch: DevicePtr,
    a_scale_scratch: DevicePtr,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    // (a) BF16 acts → int8 + per-32 F32 scale.
    let a_blocks = m * (k / 32);
    KernelLaunch::new(gpu, requant_a_kernel)
        .grid([div_ceil(a_blocks, 128), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(a_bf16)
        .arg_ptr(a_i8_scratch)
        .arg_ptr(a_scale_scratch)
        .arg_u32(m)
        .arg_u32(k)
        .launch(stream)?;
    // (b) int8 × int8 GEMM with per-32 block scales → BF16 out.
    KernelLaunch::new(gpu, faith2_kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([256, 1, 1])
        .arg_ptr(a_i8_scratch)
        .arg_ptr(w_i8)
        .arg_ptr(a_scale_scratch)
        .arg_ptr(w_scale)
        .arg_ptr(out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
