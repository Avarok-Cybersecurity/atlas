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
pub(crate) mod gemm;
pub(crate) use gemm::{hc_pre_gemm, hc_pre_gemm_folding};

use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::hyper_connection_lowrank_head::{hc_post_lowrank, hc_pre_split};
use super::hyper_connection_lowrank_rows::{HC_DEC_MAX_T, hc_pre_rows};
use super::hyper_connection_post_fold::{HcDeferredPost, HcPreArm, hc_pre_arm};
use crate::layers::qwen3_attention::HcLowRank;

/// `ATLAS_QWEN4EXP_NO_HC_GEMM=1`: revert the large-T collapse to the fused
/// FP32 kernel (deploy-time kill switch; the GEMM path rounds `normed` to
/// BF16 before the projections).
pub(crate) fn hc_gemm_disabled() -> bool {
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

/// `ATLAS_HC_PREFILL_CUBLAS=1`: route the large-T collapse's three low-rank
/// projections through cuBLASLt instead of `dense_gemm_bf16_pipelined` — the
/// move that took DECODE's collapse from 254/265 to 122/131 us a layer.
///
/// ★ MEASURED A LOSS, kept as the escape hatch that records it: 1488 / 1431
/// vs 1507 / 1489 tok/s at 8K / 32K (qwen4_exp NVFP4, TP=2 x EP=2, engagement
/// confirmed by `cublas=true` on the arm line). At large M the tile GEMM is in
/// its regime; the decode precedent does NOT transfer. Default stays off.
pub(crate) fn hc_prefill_cublas() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ATLAS_HC_PREFILL_CUBLAS").as_deref() == Ok("1"))
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
    hc_pre_lowrank_folding(
        gpu,
        kernel,
        streams,
        w,
        y_out,
        inj_out,
        scratch,
        num_tokens,
        hidden_size,
        hc_mult,
        norm_eps,
        None,
        stream,
    )
}

/// [`hc_pre_lowrank`] that may also settle the PREVIOUS site's `hc_post`,
/// folded into this collapse's stage kernel. See
/// [`super::hyper_connection_post_fold`] — `hc_post_folds_into_next_pre` is the
/// one function that decides, and the caller must have consulted it.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre_lowrank_folding(
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
    deferred: Option<HcDeferredPost>,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        !w.inject_w.is_null(),
        "hc_pre_lowrank needs block_inject_weight; a site loaded without one \
         is the model-level mixer and must use hc_head_lowrank"
    );
    let arm = hc_pre_arm(scratch, num_tokens, hidden_size, hc_mult, w.rank as u32);
    // Only `hc_pre_gemm` reaches a `hc_pre_stage_bf16` launch. On every other
    // arm, pay the deferred residual off with the real `hc_post` first —
    // identical arithmetic and identical bytes to never having folded — and SAY
    // SO, loudly: a fast path that quietly degrades into a fallback reads as a
    // clean pass, and this one would be invisible in the output.
    let deferred = match deferred {
        Some(d) if !arm.stages_bf16() => {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                tracing::warn!(
                    ?arm,
                    num_tokens,
                    "hc_post fold DECLINED at the launch — applying the deferred residual \
                     as a separate hc_post. The site's predicate and this dispatch \
                     disagree; the answer is right and the saving is gone."
                )
            });
            let (block_out, inj) = d.apply();
            hc_post_lowrank(
                gpu,
                gpu.kernel("hyper_connection", "hc_post")?,
                block_out,
                streams,
                inj,
                streams,
                num_tokens,
                hidden_size,
                hc_mult,
                stream,
            )?;
            None
        }
        other => other,
    };
    // SMALL T (decode): three multi-block launches instead of the fused
    // kernel, whose grid=[T] means grid=[1] at decode — one block, one SM,
    // ~13 MB of weights per call (measured 2.0 ms; the whole token was
    // 96 x that). The fused kernel stays for prefill, where grid=[T]
    // already fills the machine and skips the global round trip.
    if arm == HcPreArm::DecodeRows {
        {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                tracing::info!(
                    num_tokens,
                    hidden_size,
                    hc_mult,
                    rank = w.rank,
                    "hc_pre_lowrank arm: DECODE-ROWS"
                )
            });
        }
        return hc_pre_rows(
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
    // T > 8 but still decode-shaped: CHUNK onto the decode-rows arm rather
    // than fall to the GEMM decomposition. See the module note — same cliff
    // as the two out_proj arms, same fix, and rows are independent.
    if arm == HcPreArm::DecodeRowsChunked {
        {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                tracing::info!(
                    num_tokens,
                    chunk = HC_DEC_MAX_T,
                    "hc_pre_lowrank arm: DECODE-ROWS CHUNKED (T > HC_DEC_MAX_T; \
                     ATLAS_NO_HC_PRE_CHUNK restores the cuBLASLt GEMM)"
                );
            });
        }
        let h = hidden_size as usize;
        let hc = hc_mult as usize;
        let mut off = 0u32;
        while off < num_tokens {
            let take = (num_tokens - off).min(HC_DEC_MAX_T);
            // A ragged tail below the contract's floor would refuse; the
            // contract admits 1..=8, so every chunk is in range.
            hc_pre_rows(
                gpu,
                streams.offset(off as usize * hc * h * 4),
                w,
                y_out.offset(off as usize * h * 2),
                inj_out.offset(off as usize * hc * 4),
                scratch,
                take,
                hidden_size,
                hc_mult,
                norm_eps,
                /* inject */ true,
                stream,
            )?;
            off += take;
        }
        return Ok(());
    }
    // Decode-shaped T: the GEMM decomposition with cuBLASLt for the
    // three projections. The split path's hand-rolled k_down/k_fin each
    // stream ~6.5 MB of low-rank weights well off the bandwidth floor —
    // the same GEMM-shaped-work-on-hand-rolled-kernels defect class as
    // the prefill collapse and the batched-decode QKVZ arms, and the
    // same cure. ATLAS_HC_DECODE_SPLIT=1 keeps the split path (A/B).
    if arm == HcPreArm::DecodeGemm {
        {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                tracing::info!(
                    num_tokens,
                    hidden_size,
                    hc_mult,
                    rank = w.rank,
                    "hc_pre_lowrank arm: DECODE-GEMM(cuBLASLt)"
                )
            });
        }
        return hc_pre_gemm_folding(
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
            deferred,
            stream,
        );
    }
    if arm == HcPreArm::DecodeSplit {
        {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                tracing::info!(
                    num_tokens,
                    hidden_size,
                    hc_mult,
                    rank = w.rank,
                    "hc_pre_lowrank arm: DECODE-SPLIT"
                )
            });
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
    if arm == HcPreArm::PrefillGemm {
        {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                tracing::info!(
                    num_tokens,
                    hidden_size,
                    hc_mult,
                    rank = w.rank,
                    cublas = hc_prefill_cublas(),
                    "hc_pre_lowrank arm: PREFILL-GEMM"
                )
            });
        }
        return hc_pre_gemm_folding(
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
            /* use_cublas */ hc_prefill_cublas(),
            /* row_exact */ false,
            deferred,
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
