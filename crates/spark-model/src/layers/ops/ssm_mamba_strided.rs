// SPDX-License-Identifier: AGPL-3.0-only

//! Strided multi-sequence Mamba-2 decode launchers.
//!
//! Sibling of `ssm_mamba.rs` per the house `#[path]` idiom — that file is
//! already at the repo's 500-LoC cap, so the milestone-B strided twins land
//! here instead of growing it further.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::DenseWeight;

/// Causal conv1d decode update with a REAL bias and INDEPENDENT input/output
/// row strides, so N concurrent Nemotron-H sequences go in ONE launch.
///
/// Kernel: `causal_conv1d_update_strided(conv_state, new_input, weight, bias,
///          output, batch, dim, d_conv, input_stride, output_stride)`
/// Grid: (ceil(dim/256), batch, 1)  Block: (256, 1, 1)
///
/// Identical math to `causal_conv1d_update`; the only difference is that the
/// input and output row strides are passed explicitly instead of both being
/// assumed equal to `dim`. The Nemotron-H concurrent-decode path feeds this
/// straight from the batched `in_proj` output, whose rows are `in_proj_size`
/// apart (10304 on Lightning-30B), while the conv output is `d_xbc`-strided
/// (6144) — so the non-strided kernel would read sequence b>=1 from
/// `b*d_xbc`, landing in the previous sequence's dt/z region (correct at n=1,
/// silently corrupt at n>=2).
///
/// Distinct from [`super::conv1d_update_l2norm_strided`], which hardcodes a
/// NULL bias, applies a per-head L2 norm, and writes FP32. Nemotron needs the
/// other combination: real bias, no L2, BF16 out.
///
/// `conv_state` keeps the `(b * dim + ch) * d_conv` layout, so the caller must
/// have verified the per-sequence pool slots are contiguous (dense prefix).
///
/// BIT-PARITY: one launch at `batch = n` is byte-identical to `n` launches at
/// `batch = 1` with pre-offset pointers — only base addresses change. Proven
/// by `examples/conv1d_biased_strided_microtest.rs`.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_update_biased_strided(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: &DenseWeight,
    bias: DevicePtr,
    output: DevicePtr,
    dim: u32,
    d_conv: u32,
    batch_size: u32,
    input_stride: u32,
    output_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(dim, 256), batch_size, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(bias)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(dim)
        .arg_u32(d_conv)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .launch(stream)
}

#[cfg(test)]
#[path = "ssm_mamba_strided_tests.rs"]
mod ssm_mamba_strided_tests;
