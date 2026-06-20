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

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::Fp8DenseWeight;

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
