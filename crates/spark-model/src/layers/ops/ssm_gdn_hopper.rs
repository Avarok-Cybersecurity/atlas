// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for the Hopper GDN decode twins.
//!
//! The kernels are `kernels/hopper/common/gdn_decode_hopper.cu`, which exists
//! only under `kernels/hopper` (and, by symlink, `kernels/b200`), so
//! [`KernelHandle`]s for them are `KernelHandle(0)` on every other target and
//! the launchers below fall through to the gb10 parents there without reading
//! anything.
//!
//! On a target that DOES have them, the choice is a declared lever —
//! `[defaults] gdn_decode_hopper`, resolved in
//! [`super::target_defaults`] — and not kernel presence. It was presence
//! (#927) until H100 round 12 measured the twins, which is the whole content
//! of [`gdn_decode_hopper_enabled`] below.
//!
//! SSOT for the geometry and the numbers: `GDN-DECODE-ATTRIBUTION.md`
//! (#927/#928/round 12) and the kernel's own header.

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

/// Are the Hopper GDN decode twins selected? — `[defaults] gdn_decode_hopper`,
/// with `ATLAS_GDN_DECODE_HOPPER` overriding and `ATLAS_NO_GDN_HOPPER=1`
/// outranking both ([`super::target_defaults`]).
///
/// ⚠️ **FALSE on every target today, hopper included.** #927 selected the twins
/// by KERNEL PRESENCE, so a hopper build ran them unconditionally. H100 round
/// 12 (2026-09-11, 1xH100 80GB HBM3, Qwen/Qwen3.8-27B-FP8 @ `cc5a21e46`,
/// `h100-round12-report.md`) measured that choice three independent ways and
/// all three came out negative:
///
/// | measurement | twin vs parent |
/// |---|---|
/// | `native_gdn_decode_hopper_microtest`, contiguous n=1 | 13.62 vs 11.30 us — **0.83x** |
/// | the same microtest, the other 11 legs | 0.98-1.01x — null |
/// | nsys, C=1 step, 48 launches | 912.7 vs 854.2 us — **+6.8%** |
/// | nsys, n=16 step, 48 launches | 2749.9 vs 2744.8 us — +0.19%, null |
/// | serve A/B, cell F vs cell E, C=1 | **+0.41%** aggregate, -0.43% TPOT |
///
/// The serve delta is larger than either cell's rep spread (0.02% / 0.04%) and
/// points the same way on both metrics, so it is a sign rather than noise —
/// and a small one. Numerics are not the reason: the twins are BIT-IDENTICAL
/// to their parents on all 12 legs (`state_diff`/`out_diff` 0), so this row is
/// a pure speed claim and the claim failed. A 132-SM H100 leaves the
/// column-tiled grid nothing to fill — the parent already saturates the device
/// at n>=4, and at n=1 the extra CTAs cost more in launch and reduction than
/// they recover.
///
/// It is an H100 finding, NOT a verdict on the kernel, which is why
/// `gdn_decode_hopper.cu` stays in `kernels/hopper/HARDWARE.toml`'s
/// `[kernels] overrides` and keeps being compiled: on a smaller-SM Hopper part
/// the trade may go the other way, and `ATLAS_GDN_DECODE_HOPPER=1` is how the
/// next part measures it.
pub fn gdn_decode_hopper_enabled() -> bool {
    super::target_defaults::resolved().gdn_decode_hopper.value
}

/// Is the UNSTRIDED twin usable for this shape? — the lever, a resolved
/// handle, and the kernel's own dimension contract, in one place.
pub fn gdn_decode_hopper_selected(twin: KernelHandle, k_dim: u32, v_dim: u32) -> bool {
    twin.0 != 0 && gdn_decode_hopper_enabled() && gdn_hopper_dims_ok(k_dim, v_dim)
}

/// The strided twin's version of [`gdn_decode_hopper_selected`].
pub fn gdn_decode_strided_hopper_selected(twin: KernelHandle, k_dim: u32, v_dim: u32) -> bool {
    twin.0 != 0 && gdn_decode_hopper_enabled() && gdn_hopper_strided_dims_ok(k_dim, v_dim)
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

/// The C=1 GDN decode launch — twin or gb10 parent, chosen ONCE, here.
///
/// One entry point rather than an `if` at each dispatch site: the decision is
/// three conditions (the lever, a resolved handle, the kernel's dimension
/// contract) and the two arms take the SAME arguments, so a call site that
/// spelled it itself would be a second copy of a rule that has already moved
/// once. `twin` is `KernelHandle(0)` off hopper and whenever the caller is not
/// on the FP32 state, which makes this the parent's plain launcher there.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_f32_auto(
    gpu: &dyn GpuBackend,
    parent: KernelHandle,
    twin: KernelHandle,
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
    if gdn_decode_hopper_selected(twin, k_dim, v_dim) {
        return gdn_decode_f32_hopper(
            gpu,
            twin,
            h_state,
            query,
            key,
            value,
            gate,
            beta,
            output,
            batch_size,
            num_k_heads,
            num_v_heads,
            k_dim,
            v_dim,
            stream,
        );
    }
    super::gdn_decode(
        gpu,
        parent,
        h_state,
        query,
        key,
        value,
        gate,
        beta,
        output,
        batch_size,
        num_k_heads,
        num_v_heads,
        k_dim,
        v_dim,
        stream,
    )
}

/// [`gdn_decode_f32_auto`] for the batched decode path's strided launch.
///
/// THREE arms, not two, and they are ordered by receipt. `twin_smem` is the
/// ONE-READ twin (#927, `ops::ssm_gdn_strided_hopper`): it keeps the parent's
/// (thread -> column) partition exactly and reads the f32 state once for 96 of
/// its 128 rows, which is a claim about TRAFFIC and is on by default for
/// `n >= 4`. `twin` is the COLUMN-TILED twin (round 12): it re-partitions
/// columns to fill a 132-SM device at n=1, which is a claim about OCCUPANCY,
/// and is off on a measured loss. They are tried in that order because at the
/// shape where both could run — a wide batch — the traffic claim is the one
/// with a cost receipt behind it, and because the column-tiled twin's own
/// nsys number at n=16 is a null (+0.19%).
///
/// The route line is emitted ONCE PER PROCESS, naming the entry that was
/// launched and the grid it was launched with, because a lever nobody can see
/// engage is how a campaign spends a round measuring the arm it thought it had
/// turned off (`ssm_gdn_tc_route`'s lesson, twice).
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_f32_strided_auto(
    gpu: &dyn GpuBackend,
    parent: KernelHandle,
    twin: KernelHandle,
    twin_smem: KernelHandle,
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
    let sm_count = gdn_hopper_sm_count(gpu);
    let smem_arm = super::gdn_decode_strided_smem_selected(
        twin_smem,
        batch_size,
        num_v_heads,
        k_dim,
        v_dim,
        sm_count,
    );
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::info!(
                "{}",
                super::gdn_decode_strided_smem_route_line(
                    smem_arm,
                    num_v_heads,
                    batch_size,
                    sm_count
                )
            );
        });
    }
    if smem_arm {
        return super::gdn_decode_f32_strided_hopper_smem(
            gpu,
            twin_smem,
            h_state,
            query,
            key,
            value,
            gate,
            beta,
            output,
            batch_size,
            num_k_heads,
            num_v_heads,
            k_dim,
            v_dim,
            qk_stride,
            v_stride,
            gb_stride,
            out_stride,
            stream,
        );
    }
    if gdn_decode_strided_hopper_selected(twin, k_dim, v_dim) {
        return gdn_decode_f32_strided_hopper(
            gpu,
            twin,
            h_state,
            query,
            key,
            value,
            gate,
            beta,
            output,
            batch_size,
            num_k_heads,
            num_v_heads,
            k_dim,
            v_dim,
            qk_stride,
            v_stride,
            gb_stride,
            out_stride,
            stream,
        );
    }
    super::gdn_decode_f32_strided(
        gpu,
        parent,
        h_state,
        query,
        key,
        value,
        gate,
        beta,
        output,
        batch_size,
        num_k_heads,
        num_v_heads,
        k_dim,
        v_dim,
        qk_stride,
        v_stride,
        gb_stride,
        out_stride,
        stream,
    )
}

#[cfg(test)]
#[path = "ssm_gdn_hopper_lever_tests.rs"]
mod lever_tests;

#[cfg(test)]
#[path = "ssm_gdn_hopper_tests.rs"]
mod tests;
