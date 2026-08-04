// SPDX-License-Identifier: AGPL-3.0-only

//! Mamba-2 SSD chunked-scan launchers (cumsum / CB bmm / fused scan) and their
//! tiling constants. Split from `ssm_mamba.rs` (500-LoC cap).

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::*;

/// SSD chunk length (must match `SSD_L` in the kernel).
pub const SSD_L: u32 = 64;
/// head_dim rows per SSD scan block (must match `SSD_PT` in the kernel).
pub const SSD_PT: u32 = 64;

/// K1: per-chunk dt (softplus+clamp) and inclusive cumsum of the log-decay.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssd_cumsum(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    dt_raw: DevicePtr,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    dt_out: DevicePtr,
    da_cs: DevicePtr,
    seq_len: u32,
    num_heads: u32,
    nchunks: u32,
    batch_size: u32,
    dt_stride: u32,
    dt_min: f32,
    dt_max: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([nchunks, num_heads, batch_size])
        .block([SSD_L, 1, 1])
        .arg_ptr(dt_raw)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(dt_out)
        .arg_ptr(da_cs)
        .arg_u32(seq_len)
        .arg_u32(num_heads)
        .arg_u32(nchunks)
        .arg_u32(dt_stride)
        .arg_f32(dt_min)
        .arg_f32(dt_max)
        .launch(stream)
}

/// K2: `CB[c][g][t][s] = C_t . B_s` (raw, fp32).
#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssd_bmm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    b_proj: DevicePtr,
    c_proj: DevicePtr,
    cb: DevicePtr,
    seq_len: u32,
    nchunks: u32,
    n_groups: u32,
    state_size: u32,
    batch_size: u32,
    bc_stride: u32,
    stream: u64,
) -> Result<()> {
    // smem: sC[L][N] + sB[L][N] bf16
    let smem = 2 * SSD_L * state_size * 2;
    KernelLaunch::new(gpu, kernel)
        .grid([nchunks, n_groups, batch_size])
        .block([128, 1, 1])
        .shared_mem(smem)
        .arg_ptr(b_proj)
        .arg_ptr(c_proj)
        .arg_ptr(cb)
        .arg_u32(seq_len)
        .arg_u32(nchunks)
        .arg_u32(n_groups)
        .arg_u32(state_size)
        .arg_u32(bc_stride)
        .launch(stream)
}

/// K3: fused chunk_state + state_passing + chunk_scan (h0 stays in shared memory,
/// so the per-chunk `states` tensor vLLM round-trips through DRAM never exists).
#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssd_scan(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    x: DevicePtr,
    b_proj: DevicePtr,
    c_proj: DevicePtr,
    d_param: DevicePtr,
    dt_f32: DevicePtr,
    da_cs: DevicePtr,
    cb: DevicePtr,
    output: DevicePtr,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    state_size: u32,
    n_groups: u32,
    nchunks: u32,
    batch_size: u32,
    x_stride: u32,
    bc_stride: u32,
    y_stride: u32,
    stream: u64,
) -> Result<()> {
    // sH[PT][N+1] f32 | double-buffered streaming tiles: sB[2][L][N] |
    // sCM[2][L][N] | sX[2][L][PT] bf16 | sdA[2][L] + sdt[2][L] f32.
    // (sHb and sXt were dropped -- derived on the fly; see the kernel.)
    let smem = SSD_PT * (state_size + 1) * 4
        + 2 * SSD_L * state_size * 2
        + 2 * SSD_L * state_size * 2
        + 2 * SSD_L * SSD_PT * 2
        + 2 * SSD_L * 4
        + 2 * SSD_L * 4;
    KernelLaunch::new(gpu, kernel)
        .grid([num_heads, head_dim / SSD_PT, batch_size])
        .block([512, 1, 1]) // 16 warps, 2 warp-tasks each (see kernel)
        .shared_mem(smem)
        .arg_ptr(h_state)
        .arg_ptr(x)
        .arg_ptr(b_proj)
        .arg_ptr(c_proj)
        .arg_ptr(d_param)
        .arg_ptr(dt_f32)
        .arg_ptr(da_cs)
        .arg_ptr(cb)
        .arg_ptr(output)
        .arg_u32(seq_len)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(state_size)
        .arg_u32(n_groups)
        .arg_u32(nchunks)
        .arg_u32(x_stride)
        .arg_u32(bc_stride)
        .arg_u32(y_stride)
        .launch(stream)
}
