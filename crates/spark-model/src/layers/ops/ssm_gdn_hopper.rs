// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for the Hopper GDN decode twins.
//!
//! The kernels are `kernels/hopper/common/gdn_decode_hopper.cu`, which exists
//! only under `kernels/hopper`, so [`KernelHandle`]s for them resolve on that
//! target and are `KernelHandle(0)` everywhere else. That — not an env var —
//! is what keeps every other hardware set on its gb10 parents; the kill
//! switch below exists to run the A/B, not to enable the tier.
//!
//! SSOT for the geometry and the numbers: `GDN-DECODE-ATTRIBUTION.md`
//! (#927/#928) and the kernel's own header.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// H100/H200 SXM5 streaming multiprocessor count.
///
/// Only a fallback: [`gdn_hopper_cols_per_cta`] takes the real count from
/// `GpuBackend::sm_count()` and uses this when the query fails, so a wrong
/// constant costs a suboptimal tile width, never a wrong answer.
pub const HOPPER_SM_COUNT: u32 = 132;

/// Widest tile the C=1 twin will use — the parent's own block shape.
pub const GDN_HOPPER_MAX_COLS: u32 = 128;

/// Columns per CTA for `gated_delta_rule_decode_f32_hopper`.
///
/// The parent launches `num_v_heads * rows` CTAs of `v_dim` threads. On this
/// model that is 48 CTAs at C=1 against an H100's 132 SMs, so 84 SMs get no
/// work — the whole point of the twin. Narrowing the tile to 32 columns turns
/// one CTA into `v_dim / 32` of them and reaches every SM.
///
/// It narrows ONLY when the natural grid cannot fill the device. Where it can
/// — the n >= 2 shapes, and every GB10 shape, since GB10 has exactly 48 SMs —
/// a narrower tile just costs warps per CTA: measured on dgx2 at C=1, 32
/// columns reads 1.42x the parent where 128 columns reads 1.97x.
///
/// `v_dim` is returned unchanged when it is not a multiple of 32; the kernel
/// guards the tail, but there is no reason to create a ragged grid.
pub fn gdn_hopper_cols_per_cta(v_dim: u32, natural_ctas: u32, sm_count: u32) -> u32 {
    let widest = v_dim.min(GDN_HOPPER_MAX_COLS);
    if natural_ctas >= sm_count || !v_dim.is_multiple_of(32) {
        return widest;
    }
    // Smallest tile that still fills the device, floored at one warp.
    for cols in [64u32, 32u32] {
        if cols >= widest {
            continue;
        }
        let ctas = natural_ctas * v_dim.div_ceil(cols);
        if ctas >= sm_count {
            return cols;
        }
    }
    if v_dim / 32 > 1 { 32 } else { widest }
}

/// `sm_count()` with the Hopper fallback, so a backend that cannot answer
/// still gets a defensible tile width instead of an error.
pub fn gdn_hopper_sm_count(gpu: &dyn GpuBackend) -> u32 {
    gpu.sm_count().unwrap_or(HOPPER_SM_COUNT).max(1)
}

/// Preconditions the twins state in their header: `smem_k`/`smem_q` are sized
/// for 128, and the loop steps by 4.
pub fn gdn_hopper_dims_ok(k_dim: u32, v_dim: u32) -> bool {
    k_dim <= GDN_HOPPER_MAX_COLS && k_dim.is_multiple_of(4) && v_dim > 0
}

/// The strided twin additionally needs one CTA per head: its state-norm clamp
/// reduces across all four warps of a 128-thread block, exactly as the parent
/// does, so the head cannot be split and `v_dim` cannot differ from 128.
pub fn gdn_hopper_strided_dims_ok(k_dim: u32, v_dim: u32) -> bool {
    gdn_hopper_dims_ok(k_dim, v_dim) && v_dim == GDN_HOPPER_MAX_COLS
}

/// Hopper twin of [`super::gdn_decode_f32_strided`] — same arguments, same
/// bits, grid `(1, num_v_heads, batch_size)` and block `(128, 1, 1)`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_f32_strided_hopper(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        gdn_hopper_strided_dims_ok(k_dim, v_dim),
        "gated_delta_rule_decode_f32_strided_hopper needs k_dim <= 128, k_dim % 4 == 0 and \
         v_dim == 128 (the head-wide state-norm reduction); got k_dim={k_dim} v_dim={v_dim}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([1, num_v_heads, batch_size])
        .block([v_dim, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .arg_u32(out_stride)
        .launch(stream)
}

/// Hopper twin of [`super::gdn_decode`]'s FP32 entry — same arguments, same
/// bits, grid `(v_dim / cols, num_v_heads, batch_size)`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_f32_hopper(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        gdn_hopper_dims_ok(k_dim, v_dim),
        "gated_delta_rule_decode_f32_hopper needs k_dim <= 128 and k_dim % 4 == 0; \
         got k_dim={k_dim} v_dim={v_dim}"
    );
    let natural = num_v_heads.saturating_mul(batch_size).max(1);
    let cols = gdn_hopper_cols_per_cta(v_dim, natural, gdn_hopper_sm_count(gpu));
    KernelLaunch::new(gpu, kernel)
        .grid([v_dim.div_ceil(cols), num_v_heads, batch_size])
        .block([cols, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .launch(stream)
}

#[cfg(test)]
#[path = "ssm_gdn_hopper_tests.rs"]
mod tests;
