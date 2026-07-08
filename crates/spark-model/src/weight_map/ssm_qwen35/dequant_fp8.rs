// SPDX-License-Identifier: AGPL-3.0-only

//! FP8 block-slice dequant helper split out of `ssm_qwen35.rs` for the
//! ≤500 LoC file-size cap.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Dequant a block-scaled FP8 weight to a fresh BF16 device buffer, given
/// device pointers (not store keys). Mirrors `dequant_fp8_blockscaled_to_bf16`
/// but operates on an aliased slice of a *fused* expert tensor
/// (`experts.gate_up_proj` / `experts.down_proj`), where per-expert weights
/// cannot be addressed by name. Caller owns and frees the returned buffer.
#[allow(clippy::too_many_arguments)]
pub(super) fn dequant_fp8_block_slice_bf16(
    gpu: &dyn GpuBackend,
    weight_ptr: DevicePtr,
    scale_ptr: DevicePtr,
    n: usize,
    k: usize,
    sn: usize,
    sk: usize,
    scale_is_f32: bool,
) -> Result<DevicePtr> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
    let out = gpu.alloc(n * k * 2)?; // BF16 = 2 bytes/element
    let block_n = (n / sn) as u32;
    let block_k = (k / sk) as u32;
    let stream = gpu.default_stream();
    let kernel = gpu.kernel(
        "dequant_fp8_blockscaled_bf16",
        "dequant_fp8_blockscaled_bf16",
    )?;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(k as u32, 64), div_ceil(n as u32, 4), 1])
        .block([64, 4, 1])
        .arg_ptr(weight_ptr)
        .arg_ptr(scale_ptr)
        .arg_ptr(out)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(block_n)
        .arg_u32(block_k)
        .arg_u32(sk as u32)
        .arg_u32(scale_is_f32 as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    Ok(out)
}
