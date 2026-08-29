// SPDX-License-Identifier: AGPL-3.0-only

//! W4A8 integer-DP4A decode GEMV dispatch (strix-hip / gfx1151 only).
//!
//! These wrap the additive DP4A kernels in
//! `kernels/strix-hip/common/w4a16_gemv_dp4a.cu`. They are selected ONLY on
//! gfx1151 behind the `ATLAS_W4A16_DP4A` flag (see `layers::dense_ffn`); the
//! float E2M1-LUT path (`w4a16_gemv*`) is untouched and remains the default on
//! every target. The win is on the bandwidth-bound LPDDR5X part: int8 v_dot4
//! (`__builtin_amdgcn_sudot4`) + branchless v_perm codebook replace per-weight
//! FP32 FMA, validated cosine 0.999991 vs the float oracle on real gfx1151.
//!
//! The activation int8 quant is HOISTED: it runs once per distinct activation
//! (gate/up share the post-norm input; down uses silu(gate)*up), not once per
//! GEMV — per-call quant is break-even, hoisted quant is the +12% lever.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

/// Group size: one weight scale + one activation scale per 16 elements.
/// SSOT with `DP4A_GROUP_SIZE` in `w4a16_gemv_dp4a.cu`.
pub const DP4A_GROUP_SIZE: u32 = 16;

/// Runtime gate for the W4A8 integer-DP4A decode path. OFF by default (PCND: no
/// implicit production default — the float E2M1-LUT path stays the default on
/// every target). Set `ATLAS_W4A16_DP4A=1` to enable on gfx1151 builds that
/// carry the DP4A kernels; on any other target the kernel handles miss
/// (`KernelHandle(0)`) and callers fall back to the float path regardless.
pub fn dp4a_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("ATLAS_W4A16_DP4A").as_deref() == Ok("1"))
}

/// Quantize one BF16 activation row `[1, K]` to int8 `[1, K]` + per-16-group
/// f32 scales `[K/16]` (symmetric block-q8_1, d = amax/127). Hoisted: call once
/// per distinct activation, then feed `aq`/`a_scale` to one or more
/// [`w4a16_gemv_dp4a`] GEMVs.
///
/// Kernel: `quantize_act_int8_g16(A, a_q, a_scale, K)`
/// Grid: (K/16, 1, 1)  Block: (16, 1, 1)
pub fn quantize_act_int8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    a_q: DevicePtr,
    a_scale: DevicePtr,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(k, DP4A_GROUP_SIZE), 1, 1])
        .block([DP4A_GROUP_SIZE, 1, 1])
        .arg_ptr(input)
        .arg_ptr(a_q)
        .arg_ptr(a_scale)
        .arg_u32(k)
        .launch(stream)
}

/// Fused `silu(gate)*up` activation prep for the down-proj: materializes the
/// hidden then int8-quantizes it (identical math to the float
/// `w4a16_gemv_silu_input` inline activation + [`quantize_act_int8`]). Hoists the
/// down-proj input quant out of the GEMV.
///
/// Kernel: `silu_mul_quant_int8_g16(gate, up, a_q, a_scale, K)`
/// Grid: (K/16, 1, 1)  Block: (16, 1, 1)
pub fn silu_mul_quant_int8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    a_q: DevicePtr,
    a_scale: DevicePtr,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(k, DP4A_GROUP_SIZE), 1, 1])
        .block([DP4A_GROUP_SIZE, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(a_q)
        .arg_ptr(a_scale)
        .arg_u32(k)
        .launch(stream)
}

/// W4A8 integer-DP4A GEMV (M=1) from a PRE-QUANTIZED int8 activation.
/// `C[1,N] = A_int8 @ dequant(B)`. Same weight bandwidth/layout as the float
/// `w4a16_gemv`; the activation is int8 with per-16-group scales.
///
/// Kernel: `w4a16_gemv_dp4a(a_q, a_scale, B_packed, B_scale, scale2, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn w4a16_gemv_dp4a(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_q: DevicePtr,
    a_scale: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a_q)
        .arg_ptr(a_scale)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
