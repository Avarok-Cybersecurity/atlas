// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `ops.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Unified GEMV dispatch: select kernel based on weight quantization format.
///
/// Eliminates cascading if/else chains in layer forward methods. The enum
/// branch (~1 cycle) is negligible vs GPU kernel launch overhead (~5μs).
#[allow(clippy::too_many_arguments)]
pub fn quant_gemv(
    gpu: &dyn GpuBackend,
    gemv_nvfp4: KernelHandle,
    gemv_fp8: KernelHandle,
    gemv_dense: KernelHandle,
    input: DevicePtr,
    weight: &crate::weight_map::QuantWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    use crate::weight_map::QuantWeight;
    match weight {
        QuantWeight::Nvfp4(w) => w4a16_gemv(gpu, gemv_nvfp4, input, w, output, n, k, stream),
        QuantWeight::Fp8(w) => w8a16_gemv(
            gpu,
            gemv_fp8,
            input,
            w.weight,
            w.row_scale,
            output,
            n,
            k,
            stream,
        ),
        QuantWeight::Dense(w) => dense_gemv(gpu, gemv_dense, input, w, output, n, k, stream),
    }
}

/// Unified GEMM dispatch: select kernel based on weight quantization format.
///
/// For M>1 prefill projections (Q/K/V/O). Falls back to dense GEMM for BF16.
#[allow(clippy::too_many_arguments)]
pub fn quant_gemm(
    gpu: &dyn GpuBackend,
    gemm_nvfp4: KernelHandle,
    gemm_fp8: KernelHandle,
    gemm_dense: KernelHandle,
    input: DevicePtr,
    weight: &crate::weight_map::QuantWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    use crate::weight_map::QuantWeight;
    match weight {
        QuantWeight::Nvfp4(w) => w4a16_gemm(gpu, gemm_nvfp4, input, w, output, m, n, k, stream),
        QuantWeight::Fp8(w) => w8a16_gemm(
            gpu,
            gemm_fp8,
            input,
            w.weight,
            w.row_scale,
            output,
            m,
            n,
            k,
            stream,
        ),
        QuantWeight::Dense(w) => dense_gemm(gpu, gemm_dense, input, w, output, m, n, k, stream),
    }
}

/// W4A16 GEMV (M=1): C = A @ dequant(B) for single-row activations.
///
/// A: [1, K] BF16, B: NVFP4 packed, C: [1, N] BF16.
/// 4 outputs/block, 64 threads (2 warps) per output. Cross-warp smem reduction.
///
/// Kernel: `w4a16_gemv(A, B_packed, B_scale, scale2, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn w4a16_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
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
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 double-GEMV (M=2): reads weights once, computes 2 outputs.
///
/// A: [2, K] BF16 contiguous, B: NVFP4 packed, C: [2, N] BF16 contiguous.
/// Same weight bandwidth as single GEMV — eliminates GEMM M=2 tile waste.
///
/// Kernel: `w4a16_gemv_batch2(A, B_packed, B_scale, scale2, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn w4a16_gemv_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
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
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 triple-GEMV (M=3): reads weights once, computes 3 outputs.
///
/// A: [3, K] BF16 contiguous, B: NVFP4 packed, C: [3, N] BF16 contiguous.
/// For K=3 speculative verification.
///
/// Kernel: `w4a16_gemv_batch3(A, B_packed, B_scale, scale2, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn w4a16_gemv_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
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
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W4A16 GEMV with inline Q/Gate deinterleave on output write.
///
/// Same as `w4a16_gemv` but writes Q and Gate to deinterleaved positions,
/// eliminating the separate `deinterleave_qg` kernel (12 graph nodes saved).
///
/// Kernel: `w4a16_gemv_qg(A, B, S, s2, C, N, K, num_heads, head_dim)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .launch(stream)
}

/// W4A16 GEMV with inline QKVZ deinterleave on output write.
///
/// Same as `w4a16_gemv` but writes to deinterleaved output locations,
/// eliminating the separate `deinterleave_qkvz` kernel.
///
/// Kernel: `w4a16_gemv_qkvz(A, B, S, s2, C, N, K, ng, kd, vpg, vd)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qkvz(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_groups: u32,
    head_k_dim: u32,
    vheads_per_group: u32,
    head_v_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_groups)
        .arg_u32(head_k_dim)
        .arg_u32(vheads_per_group)
        .arg_u32(head_v_dim)
        .launch(stream)
}

/// Q+Gate GEMV for 2 tokens with inline deinterleave.
///
/// Reads the Q+Gate weight matrix once, produces 2 deinterleaved output
/// vectors (Q|Gate for each token). Replaces 2× `w4a16_gemv_qg` calls.
///
/// Kernel: `w4a16_gemv_qg_batch2(A, B, S, s2, C, N, K, num_heads, head_dim)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
/// Input A: [2, K], Output C: [2, N] deinterleaved [Q|G] per token.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .launch(stream)
}

/// W4A16 GEMV batch3 with inline Q/Gate deinterleave.
///
/// Reads the Q+Gate weight matrix once, produces 3 deinterleaved output
/// vectors (Q|Gate for each token). For K=3 speculative verification.
///
/// Kernel: `w4a16_gemv_qg_batch3(A, B, S, s2, C, N, K, num_heads, head_dim)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
/// Input A: [3, K], Output C: [3, N] deinterleaved [Q|G] per token.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .launch(stream)
}

/// Dual-projection GEMV for 3 tokens (K+V or any 2 weight matrices).
///
/// Reads each weight matrix once, produces 3 output vectors per projection.
/// `blockIdx.z` selects projection 0 or 1.
///
/// Kernel: `w4a16_gemv_dual_batch3(A, B0, S0, s2_0, C0, B1, S1, s2_1, C1, N, K)`
/// Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
/// Input A: [3, K], Output C0: [3, N], C1: [3, N].
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight0: &QuantizedWeight,
    output0: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight0.weight)
        .arg_ptr(weight0.weight_scale)
        .arg_f32(weight0.weight_scale_2)
        .arg_ptr(output0)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Dual-projection GEMV for 2 tokens (K+V or any 2 weight matrices).
///
/// Reads each weight matrix once, produces 2 output vectors per projection.
/// `blockIdx.z` selects projection 0 or 1.
///
/// Kernel: `w4a16_gemv_dual_batch2(A, B0, S0, s2_0, C0, B1, S1, s2_1, C1, N, K)`
/// Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
/// Input A: [2, K], Output C0: [2, N], C1: [2, N].
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight0: &QuantizedWeight,
    output0: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight0.weight)
        .arg_ptr(weight0.weight_scale)
        .arg_f32(weight0.weight_scale_2)
        .arg_ptr(output0)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

// ── Position embeddings ────────────────────────────────────────────
