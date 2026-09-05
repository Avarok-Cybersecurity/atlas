// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3.8-Flash-Next low-rank mHC dispatch.
//!
//! Companion to `hyper_connection.rs`, which drives DeepSeek-V4's Sinkhorn
//! mixer. Both families share the `[T, hc_mult, H]` FP32 highway and the same
//! four kernel NAMES — a model shadow overrides the whole
//! `hyper_connection.cu` file, so `qwen3.8-flash-next` resolves
//! `hyper_connection::hc_pre` to the low-rank kernel while
//! `deepseek-v4-flash` resolves it to the Sinkhorn one. The two take
//! DIFFERENT argument lists, which is why the launches live apart.
//!
//! `hc_expand` is byte-identical across both and is not duplicated here.
//!
//! Selection is by WEIGHTS, not by model name: `HcSiteWeights::lowrank`
//! being `Some` is what routes here. A model that somehow carried both would
//! be a load-time bug, not a silent dispatch coin-flip.

use anyhow::Result;
#[path = "hyper_connection_lowrank_gemm.rs"]
mod gemm;
pub(crate) use gemm::hc_pre_gemm;

use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layers::qwen3_attention::HcLowRank;

/// `ATLAS_QWEN4EXP_NO_HC_GEMM=1`: revert the large-T collapse to the fused
/// FP32 kernel (deploy-time kill switch; the GEMM path rounds `normed` to
/// BF16 before the projections).
fn hc_gemm_disabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_NO_HC_GEMM").as_deref() == Ok("1"))
}

/// `ATLAS_HC_DECODE_SPLIT=1`: keep the pre-cuBLASLt split path for
/// decode-shaped T (A/B escape hatch, same convention as the GEMM kill
/// switch above).
pub(crate) fn hc_decode_split_forced() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ATLAS_HC_DECODE_SPLIT").as_deref() == Ok("1"))
}

/// Collapse the `hc_mult` streams to one, and emit the per-stream injection
/// weights the matching [`hc_post_lowrank`] needs.
///
/// `streams [T, hc, H] -> y_out [T, H]`, `inj_out [T, hc]`.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre_lowrank(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    w: &HcLowRank,
    y_out: DevicePtr,
    inj_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        !w.inject_w.is_null(),
        "hc_pre_lowrank needs block_inject_weight; a site loaded without one \
         is the model-level mixer and must use hc_head_lowrank"
    );
    // SMALL T (decode): three multi-block launches instead of the fused
    // kernel, whose grid=[T] means grid=[1] at decode — one block, one SM,
    // ~13 MB of weights per call (measured 2.0 ms; the whole token was
    // 96 x that). The fused kernel stays for prefill, where grid=[T]
    // already fills the machine and skips the global round trip.
    if num_tokens <= 64 && !scratch.is_null() {
        // Decode-shaped T: the GEMM decomposition with cuBLASLt for the
        // three projections. The split path's hand-rolled k_down/k_fin each
        // stream ~6.5 MB of low-rank weights well off the bandwidth floor —
        // the same GEMM-shaped-work-on-hand-rolled-kernels defect class as
        // the prefill collapse and the batched-decode QKVZ arms, and the
        // same cure. ATLAS_HC_DECODE_SPLIT=1 keeps the split path (A/B).
        if !hc_decode_split_forced() {
            return hc_pre_gemm(
                gpu,
                streams,
                w,
                y_out,
                inj_out,
                scratch,
                num_tokens,
                hidden_size,
                hc_mult,
                norm_eps,
                /* inject */ true,
                /* use_cublas */ true,
                /* row_exact */ false,
                stream,
            );
        }
        return hc_pre_split(
            gpu,
            streams,
            w,
            y_out,
            inj_out,
            scratch,
            num_tokens,
            hidden_size,
            hc_mult,
            norm_eps,
            /* inject */ true,
            stream,
        );
    }
    // LARGE T (prefill): tensor-core GEMM formulation — 47% of prefill was
    // this collapse running as FP32 warp loops. Kill switch reverts to the
    // fused kernel below.
    if !scratch.is_null() && !hc_gemm_disabled() {
        return hc_pre_gemm(
            gpu,
            streams,
            w,
            y_out,
            inj_out,
            scratch,
            num_tokens,
            hidden_size,
            hc_mult,
            norm_eps,
            /* inject */ true,
            /* use_cublas */ false,
            /* row_exact */ false,
            stream,
        );
    }
    // Block 1024 + dynamic shared for the staged normed vector [hc*H] and
    // the rank vector — the warp-cooperative core. This launch WAS the whole
    // decode budget at block 256 with per-thread serial rows (4.5 ms/call,
    // x96 calls/token); see the kernel's PERFORMANCE SHAPE note.
    let smem = (hc_mult * hidden_size + w.rank as u32) * 4;
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([1024, 1, 1])
        .shared_mem(smem)
        .arg_ptr(streams)
        .arg_ptr(w.norm_w)
        .arg_ptr(w.down_w)
        .arg_ptr(w.up_w)
        .arg_ptr(w.inject_w)
        .arg_ptr(y_out)
        .arg_ptr(inj_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(w.rank as u32)
        .arg_f32(norm_eps)
        .launch(stream)
}

/// The model-level mixer (`use_combine=False`): the same collapse with no
/// injection vector.
///
/// This is also the model's FINAL NORMALIZATION — the checkpoint ships no
/// `model.norm.weight` because `hc_norm` here plays that role.
#[allow(clippy::too_many_arguments)]
pub fn hc_head_lowrank(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    w: &HcLowRank,
    y_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    stream: u64,
) -> Result<()> {
    if num_tokens <= 64 && !scratch.is_null() {
        if !hc_decode_split_forced() {
            return hc_pre_gemm(
                gpu,
                streams,
                w,
                y_out,
                DevicePtr::NULL,
                scratch,
                num_tokens,
                hidden_size,
                hc_mult,
                norm_eps,
                /* inject */ false,
                /* use_cublas */ true,
                /* row_exact */ false,
                stream,
            );
        }
        return hc_pre_split(
            gpu,
            streams,
            w,
            y_out,
            DevicePtr::NULL,
            scratch,
            num_tokens,
            hidden_size,
            hc_mult,
            norm_eps,
            /* inject */ false,
            stream,
        );
    }
    // Same GEMM formulation as hc_pre — the head is the identical collapse
    // minus the injection GEMM (hc_pre_mix skips inj on a null inj_pre).
    if !scratch.is_null() && !hc_gemm_disabled() {
        return hc_pre_gemm(
            gpu,
            streams,
            w,
            y_out,
            DevicePtr::NULL,
            scratch,
            num_tokens,
            hidden_size,
            hc_mult,
            norm_eps,
            /* inject */ false,
            /* use_cublas */ false,
            /* row_exact */ false,
            stream,
        );
    }
    let smem = (hc_mult * hidden_size + w.rank as u32) * 4;
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([1024, 1, 1])
        .shared_mem(smem)
        .arg_ptr(streams)
        .arg_ptr(w.norm_w)
        .arg_ptr(w.down_w)
        .arg_ptr(w.up_w)
        .arg_ptr(y_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(w.rank as u32)
        .arg_f32(norm_eps)
        .launch(stream)
}

/// Inject the block output back into every stream:
/// `out[t, s*H + d] = residual[t, s*H + d] + block_out[t, d] * inj[t, s]`.
///
/// Note there is no `comb` argument: DeepSeek mixes streams with a full
/// `[hc, hc]` combine matrix on the way back, Qwen scales by one scalar per
/// stream. Passing a combine matrix here would not type-check, which is the
/// point of keeping the two launches separate.
#[allow(clippy::too_many_arguments)]
pub fn hc_post_lowrank(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    block_out: DevicePtr,
    residual: DevicePtr,
    inj: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(residual)
        .arg_ptr(inj)
        .arg_ptr(out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// The three-launch collapse for small T. Same math as the fused kernel;
/// the parity probe's T=8 fixture runs THIS path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_pre_split(
    gpu: &dyn GpuBackend,
    streams: DevicePtr,
    w: &HcLowRank,
    y_out: DevicePtr,
    inj_out: DevicePtr,
    scratch: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    inject: bool,
    stream: u64,
) -> Result<()> {
    let hc_dim = hc_mult * hidden_size;
    // Scratch layout: normed [T<=64, hc_dim] then low [T<=64, rank], F32.
    let normed = scratch;
    let low = scratch.offset(64 * hc_dim as usize * 4);

    let k_stage = gpu.kernel("hyper_connection", "hc_pre_stage")?;
    let k_down = gpu.kernel("hyper_connection", "hc_pre_down")?;
    let k_fin = gpu.kernel("hyper_connection", "hc_pre_finish")?;

    KernelLaunch::new(gpu, k_stage)
        .grid([num_tokens, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(w.norm_w)
        .arg_ptr(normed)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_f32(norm_eps)
        .launch(stream)?;

    // Spread rank rows over enough blocks to occupy the part even at T=1.
    let dsplit = (48 / num_tokens.max(1)).clamp(1, 10);
    KernelLaunch::new(gpu, k_down)
        .grid([num_tokens, dsplit, 1])
        .block([1024, 1, 1])
        .arg_ptr(normed)
        .arg_ptr(w.down_w)
        .arg_ptr(low)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(w.rank as u32)
        .launch(stream)?;

    let fsplit = (48 / num_tokens.max(1)).clamp(1, 10);
    KernelLaunch::new(gpu, k_fin)
        .grid([num_tokens, fsplit, 1])
        .block([256, 1, 1])
        .shared_mem(w.rank as u32 * 4)
        .arg_ptr(normed)
        .arg_ptr(low)
        .arg_ptr(w.up_w)
        .arg_ptr(if inject { w.inject_w } else { DevicePtr::NULL })
        .arg_ptr(y_out)
        .arg_ptr(inj_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(w.rank as u32)
        .launch(stream)
}
