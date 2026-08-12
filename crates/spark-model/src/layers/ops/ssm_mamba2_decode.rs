// SPDX-License-Identifier: AGPL-3.0-only

//! Strided (multi-sequence) Mamba-2 SSM decode launch.
//!
//! Kept in its own module rather than appended to `ssm_mamba.rs` so the
//! strided conv1d work and the strided scan work never touch the same file,
//! and so `ssm_mamba.rs` stays under the file-size cap.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Mamba-2 SSM decode for N concurrent sequences in ONE launch.
///
/// Kernel: `mamba2_ssm_decode_strided` (module `mamba2_ssm`).
///
/// Identical recurrence to `mamba2_ssm_decode`; the four activation tensors
/// carry explicit per-SEQUENCE row strides (BF16 elements) instead of the
/// hardcoded dense strides. That lets the concurrent-decode path feed x/B/C
/// straight from the `d_xbc`-strided conv output and `dt` from the
/// `in_proj_size`-strided projection row, replacing the per-row `batch=1` loop
/// (one launch per sequence per layer) with a single `batch=n` launch.
///
/// **Bit-parity**: one launch at `batch_size = n` is byte-identical to `n`
/// launches at `batch_size = 1` with pre-offset pointers — each `(b, head)`
/// block is independent and the strides move base addresses only, never the
/// order of an accumulation. Proven by
/// `examples/mamba2_strided_microtest.rs`.
///
/// **Caller contract**: `h_state` is NOT strided. Pool slots must be dense in
/// slice order (`slot_i == base + i * num_heads*head_dim*state_size * 4`),
/// proven before dispatch; pad rows that alias a shared dummy slot MUST NOT be
/// covered by the launch.
///
/// Grid: (num_heads, batch_size, 1)  Block: (state_size, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssm_decode_strided(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    x: DevicePtr,
    b_proj: DevicePtr,
    c_proj: DevicePtr,
    dt_raw: DevicePtr,
    a_log: DevicePtr,
    d_param: DevicePtr,
    dt_bias: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    num_heads: u32,
    head_dim: u32,
    state_size: u32,
    n_groups: u32,
    dt_min: f32,
    dt_max: f32,
    x_stride: u32,
    bc_stride: u32,
    dt_stride: u32,
    y_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_heads, batch_size, 1])
        .block([state_size, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(x)
        .arg_ptr(b_proj)
        .arg_ptr(c_proj)
        .arg_ptr(dt_raw)
        .arg_ptr(a_log)
        .arg_ptr(d_param)
        .arg_ptr(dt_bias)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(state_size)
        .arg_u32(n_groups)
        .arg_f32(dt_min)
        .arg_f32(dt_max)
        // Strides last — they are appended to the non-strided arg layout.
        .arg_u32(x_stride)
        .arg_u32(bc_stride)
        .arg_u32(dt_stride)
        .arg_u32(y_stride)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    /// ONE launch covers all N sequences, at the same (num_heads, N, 1) /
    /// (state_size, 1, 1) geometry the per-row loop used at batch=1 — the whole
    /// point of the strided form. Lightning shape.
    #[test]
    fn strided_decode_is_one_launch_over_n_seqs() {
        let gpu = MockGpuBackend::new();
        let k = gpu
            .kernel("mamba2_ssm", "mamba2_ssm_decode_strided")
            .unwrap();
        let p = gpu.alloc(1024).unwrap();
        mamba2_ssm_decode_strided(
            &gpu, k, p, p, p, p, p, p, p, p, p, 8, 64, 64, 128, 8, 1e-9, 1e9, 6144, 6144, 10304,
            4096, 0,
        )
        .unwrap();
        let l = gpu.launches_snapshot();
        assert_eq!(l.len(), 1, "strided decode must be a single launch");
        assert_eq!(l[0].grid, [64, 8, 1]);
        assert_eq!(l[0].block, [128, 1, 1]);
    }

    /// Puzzle geometry (state_size=96) keeps the block at state_size, not a
    /// padded 128 — the kernel's `n_warps` epilogue depends on it.
    #[test]
    fn strided_decode_block_tracks_state_size() {
        let gpu = MockGpuBackend::new();
        let k = gpu
            .kernel("mamba2_ssm", "mamba2_ssm_decode_strided")
            .unwrap();
        let p = gpu.alloc(1024).unwrap();
        mamba2_ssm_decode_strided(
            &gpu, k, p, p, p, p, p, p, p, p, p, 4, 64, 64, 96, 8, 1e-9, 1e9, 5632, 5632, 9792,
            4096, 0,
        )
        .unwrap();
        let l = gpu.launches_snapshot();
        assert_eq!(l[0].grid, [64, 4, 1]);
        assert_eq!(l[0].block, [96, 1, 1]);
    }
}
