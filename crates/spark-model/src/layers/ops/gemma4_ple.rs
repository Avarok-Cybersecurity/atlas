// SPDX-License-Identifier: AGPL-3.0-only

//! Gemma-4 E2B per-layer-embedding (PLE) ops.
//!
//! The single kernel here (`gemma4_ple_mul`) multiplies a decoder layer's
//! 256-dim PLE gate vector (`h`, contiguous `[num_tokens, 256]`) by the
//! layer's strided slice of the model-level combined PLE buffer
//! (`[num_tokens, num_layers*256]` row-major). The strided read avoids a
//! transposed staging copy of the combined buffer on every pass.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// `h[t*d + d'] *= ple[t*row_stride + layer_col + d']` (BF16, FP32 compute).
///
/// - `h`: contiguous `[num_tokens, ple_dim]` BF16 — the PLE gate vector
///   (input_gate output), multiplied in place.
/// - `ple`: base of the combined `[num_tokens, row_stride]` BF16 buffer
///   built by the model-level precompute.
/// - `layer_col`: byte-free element column offset of this layer's slice
///   (= `layer_idx * ple_dim`).
/// - `row_stride`: elements per token row in `ple` (= `num_layers * ple_dim`).
pub fn gemma4_ple_mul(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h: DevicePtr,
    ple: DevicePtr,
    layer_col: u32,
    row_stride: u32,
    num_tokens: u32,
    ple_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([ple_dim, 1, 1])
        .arg_ptr(h)
        .arg_ptr(ple)
        .arg_u32(layer_col)
        .arg_u32(row_stride)
        .arg_u32(num_tokens)
        .arg_u32(ple_dim)
        .launch(stream)
}
