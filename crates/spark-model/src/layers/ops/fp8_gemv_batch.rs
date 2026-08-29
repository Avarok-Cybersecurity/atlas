// SPDX-License-Identifier: AGPL-3.0-only

//! FP8-weight dual-GEMV (batch=2) dispatch.
//!
//! `dense_gemv_fp8w_batch2` computes two output rows from one pass over the
//! FP8 weight matrix — the batch=2 sibling of `dense_gemv_fp8w`. It halves
//! FP8 weight bandwidth vs two M=1 GEMV launches and is bit-identical to
//! running `dense_gemv_fp8w` twice (per-token reduction order unchanged).
//! Used by the K=2 MTP verify path where the two verify positions share
//! weights but have distinct activations (lm_head, attention Q/K/V/O, SSM
//! out_proj).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::Fp8DenseWeight;

/// Register-tiled batched row-scaled FP8 GEMV (M<=8, T=2 outputs/thread) —
/// the FP8 twin of `w4a16_gemv_batch8_rt2`, for the DFlash drafter PROPOSE
/// path. `input` `[M, K]` BF16, `output` `[M, N]` BF16; per-row f32 scale
/// applied at write-out inside the kernel. Replaces the prefill-class tile
/// GEMMs (`fp8_gemm_t_row_scaled` M64-tile / `_m16`) that pad 87%/50% of
/// their M-tile at M=8 (~100 GB/s measured vs 180+ for the rt family).
/// Drafter-side numerics: correctness-free under strict-argmax accept.
/// Kernel: `fp8_gemv_rowscale_batch8_rt2` (module `fp8_gemv_rt`).
/// Grid: (ceil(N/8), 1, 1)  Block: (256, 1, 1). Requires K % 16 == 0.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemv_rowscale_batch8_rt2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=8).contains(&m),
        "fp8_gemv_rowscale_batch8_rt2: m={m} outside 1..=8 (kernel MAX_M)"
    );
    ensure!(
        k.is_multiple_of(16),
        "fp8_gemv_rowscale_batch8_rt2: K={k} not a multiple of 16"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// FP8-weight dual-GEMV. `input` is `[2, K]` BF16, `output` is `[2, N]` BF16.
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn dense_gemv_fp8w_batch2(
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

/// Block-scaled FP8 batched GEMV (M<=4). `input` is `[M, K]` BF16, `output` is
/// `[M, N]` BF16; `weight`/`block_scale` are the raw `w8a16_gemv` pointers (2D
/// block-scaled FP8). One pass over the FP8 weight serves all M rows — the M=4
/// sibling of `w8a16_gemv`, replacing `w8a16_gemm_pipelined` for n<=4 batched
/// decode (which pads M to a 128-row MMA tile). Bit-identical per-row to
/// `w8a16_gemv`. Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch4(
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
        .grid([div_ceil(n, 4), 1, 1])
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

/// Block-scaled FP8 dual-GEMV (batch=2). `input` is `[2, K]` BF16, `output` is
/// `[2, N]` BF16; `weight`/`block_scale` are the raw `w8a16_gemv` pointers.
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
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
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
