// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

//! Sub-init helpers for `TransformerModel::new`, hoisted to keep
//! `impl_a1.rs` under the 500 LoC cap.
//!
//! Each helper mirrors the equivalent inline block in `new()` 1:1.

use std::sync::Arc;

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::ssm_pool::SsmStatePool;
use crate::speculative::DraftProposer;
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// Miscellaneous scratch kernels/buffers built by `TransformerModel::new`:
/// SSM state normalization, logit softcapping, FP32 logits, embedding
/// scale, GDN prefill buffers, and the SSM verify scratch tensor.
pub(super) struct MiscScratch {
    pub ssm_state_norm_kernel: KernelHandle,
    pub logit_softcap_kernel: KernelHandle,
    pub logit_softcap_fp32_kernel: KernelHandle,
    pub use_fp32_logits: bool,
    pub logits_fp32_buf: DevicePtr,
    pub embed_scale_kernel: KernelHandle,
    pub ssm_norm_ptrs_buf: DevicePtr,
    pub gdn_buf_qkv: DevicePtr,
    pub gdn_buf_gate_beta: DevicePtr,
    pub gdn_buf_out: DevicePtr,
    pub gdn_buf_z: DevicePtr,
    pub gdn_buf_max_len: usize,
    pub ssm_verify_h_tmp: DevicePtr,
}

/// Mirrors the equivalent inline block in `new()` 1:1 (SSM norm kernel
/// through GDN prefill buffers and the verify scratch tensor).
pub(super) fn build_misc_scratch(
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    max_batch_tokens: usize,
    max_seq_len: usize,
    ssm_pool: &SsmStatePool,
) -> Result<MiscScratch> {
    // SSM state normalization kernel + pointer buffer (for chunked prefill).
    let ssm_state_norm_kernel = gpu
        .kernel("ssm_state_norm", "ssm_state_clamp_norm_fused")
        .unwrap_or(KernelHandle(0));

    // Logit softcapping (Gemma-4: cap=30.0). Only load if model uses it.
    let logit_softcap_kernel = if config.final_logit_softcapping > 0.0 {
        gpu.kernel("logit_softcap", "logit_softcap_bf16")
            .unwrap_or_else(|e| {
                tracing::warn!("logit_softcap kernel not found: {e}");
                KernelHandle(0)
            })
    } else {
        KernelHandle(0)
    };
    // FP32 softcap variant — only loaded when both softcap and FP32
    // residual are active (i.e. Gemma-4 dense). Other models keep the
    // BF16 softcap (or no softcap at all).
    // The FP32 logit softcap variant required an FP32 residual stream,
    // which no longer exists, so the BF16 softcap path is always taken.
    let logit_softcap_fp32_kernel = KernelHandle(0);
    // FP32 logits gate. The LM head produces FP32 (rather than BF16)
    // logits when the residual stream is FP32 AND the LM head is a
    // dense BF16 weight (no NVFP4 quant). NVFP4 LM heads keep their
    // existing path because that quantization is a much larger
    // precision floor than the BF16 store; FP32 wouldn't help there.
    // Today this only affects Gemma-4 dense (model_type=="gemma4",
    // num_experts==0, tied BF16 embed→lm_head).
    // Gemma-4-31B FP32 lm_head experiment. Disabled by default —
    // session 2026-05-01 verified the BF16 lm_head store is NOT the
    // source of Gemma-4's haiku argmax flip: FP32 view of step-1
    // logits keeps top1=` a` (21.85), top2=` waves` (21.706) — same
    // 0.14-margin tiebreak as BF16. The drift is upstream in attention
    // or MLP, not in the lm_head precision boundary. Code paths kept
    // wired so a future bisection (Phase 2 of the plan) can re-enable
    // via `ATLAS_GEMMA4_FP32_LMHEAD=1`. Keep `use_fp32_logits=false`
    // by default so the rest of the model behaves identically to the
    // pre-fix BF16 path on every model family.
    // FP32 lm_head + softcap. Default OFF — empirically the gain on
    // Gemma-4-31B is marginal (Creative occasionally cleaner; fib still
    // fails the same broken-indentation pattern) but the cost is huge:
    // FP32 forces host-side sampling (vocab=262144 × 4 bytes per
    // decode step → ~1 MB D2H per token) which crushes decode TPS
    // from ~35 tok/s to ~6 tok/s on Gemma-4-31B. Not worth it without
    // a GPU-side FP32 argmax kernel. `ATLAS_GEMMA4_FP32_LMHEAD=1`
    // re-enables for bisection / future work.
    //
    // The earlier "FP32 doesn't fix haiku" comment in this file was
    // arrived at via incomplete bisection (the scheduler readback
    // always assumed BF16 — see commit 16b2f3a's commit body). The
    // 2026-05-01 evening run with the dispatch wired confirmed the
    // bisection's *qualitative* conclusion: FP32 lm_head + softcap
    // doesn't materially fix Gemma-4's structural NVFP4 attention
    // drift on greedy code generation. Fix is upstream of lm_head.
    // FP32 logits (ATLAS_GEMMA4_FP32_LMHEAD) required an FP32 residual
    // stream as a precondition. With the residual stream now always BF16,
    // the FP32 logits path can never activate, so it is permanently off.
    let use_fp32_logits = false;
    // Dedicated FP32 logits scratch — only the single-token decode path
    // uses it. Prefill and batched-decode lm_head still write BF16 to the
    // shared `buffers.logits()`. Sized for one row of `vocab_size` FP32.
    let logits_fp32_buf = if use_fp32_logits {
        let bytes = config.vocab_size * 4;
        let p = gpu.alloc(bytes)?;
        tracing::info!(
            "FP32 LM head + softcap active (model_type={}, vocab={}). \
             Decode logits scratch: {} bytes.",
            config.model_type,
            config.vocab_size,
            bytes,
        );
        p
    } else {
        DevicePtr::NULL
    };

    // Embedding scale (Gemma-4: sqrt(hidden_size)). Only load if model uses it.
    let embed_scale_kernel = if config.embed_scale > 0.0 {
        gpu.kernel("embed_scale", "bf16_scale_inplace")
            .unwrap_or_else(|e| {
                tracing::warn!("embed_scale kernel not found: {e}");
                KernelHandle(0)
            })
    } else {
        KernelHandle(0)
    };
    if config.embed_scale > 0.0 {
        tracing::info!(
            "Embedding scale: {:.4} (sqrt({}))",
            config.embed_scale,
            config.hidden_size
        );
    }
    let ssm_norm_ptrs_buf = if ssm_pool.num_ssm_layers > 0 {
        gpu.alloc(ssm_pool.num_ssm_layers * 8)
            .unwrap_or(DevicePtr::NULL)
    } else {
        DevicePtr::NULL
    };

    // GDN prefill buffers: sized for max_batch_tokens (the prefill chunk size),
    // NOT max_seq_len. For prompts longer than this, prefill_twophase falls back
    // to standard chunked prefill which carries h_state/conv_state between chunks.
    // The GDN recurrence is sequential anyway, so chunking is mathematically identical.
    let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let nv = config.linear_num_value_heads;
    let conv_dim = key_dim * 2 + value_dim;
    // GDN buffers only needed when GDN linear attention layers exist
    // (conv_dim > 0). Mamba-2 models (Nemotron) have conv_dim=0 — skip alloc
    // to avoid cuMemAlloc(0) error.
    let gdn_buf_max_len = max_batch_tokens.min(max_seq_len);
    let (gdn_buf_qkv, gdn_buf_gate_beta, gdn_buf_out, gdn_buf_z) = if conv_dim > 0 {
        let qkv = gpu.alloc(gdn_buf_max_len * conv_dim * 2)?;
        let gb = gpu.alloc(gdn_buf_max_len * nv * 2 * 4)?;
        let o = gpu.alloc(gdn_buf_max_len * value_dim * 2)?;
        let z = gpu.alloc(gdn_buf_max_len * value_dim * 2)?;
        let total_mb =
            (gdn_buf_max_len * (conv_dim * 2 + nv * 2 * 4 + value_dim * 2 * 2)) / (1024 * 1024);
        tracing::info!(
            "GDN prefill buffers: {total_mb} MB for {gdn_buf_max_len} tokens (chunked SSM prefill)"
        );
        (qkv, gb, o, z)
    } else {
        (
            DevicePtr::NULL,
            DevicePtr::NULL,
            DevicePtr::NULL,
            DevicePtr::NULL,
        )
    };

    let ssm_verify_h_tmp = if ssm_pool.h_bytes > 0 {
        gpu.alloc(ssm_pool.h_bytes)?
    } else {
        DevicePtr::NULL
    };

    Ok(MiscScratch {
        ssm_state_norm_kernel,
        logit_softcap_kernel,
        logit_softcap_fp32_kernel,
        use_fp32_logits,
        logits_fp32_buf,
        embed_scale_kernel,
        ssm_norm_ptrs_buf,
        gdn_buf_qkv,
        gdn_buf_gate_beta,
        gdn_buf_out,
        gdn_buf_z,
        gdn_buf_max_len,
        ssm_verify_h_tmp,
    })
}

/// Build the MTP draft proposer when speculative decoding is requested.
///
/// `mtp_weights` is a `Vec<MtpWeights>`:
///   - empty  → no MTP weights in checkpoint; proposer disabled
///   - len 1  → single-module MTP (Qwen3.5 family): build `MtpHead`
///   - len N>1 → multi-module MTP (MiniMax M2, DeepSeek-V3 style):
///     build `MultiModuleMtpHead` with N heads
///
/// Returns `None` when speculative decoding is off, when no MTP weights
/// are available, or when no NVFP4 draft head is available.
///
/// `lm_head_nvfp4` here is the resolved *draft* head: the main NVFP4 head
/// (NVFP4-main default) or a separate draft-only NVFP4 head built when the
/// main head is kept BF16 (`skip_lm_head_quantization()`). The MTP head's
/// final hidden→vocab projection (`forward_one`) is hard-wired to
/// `w4a16_gemv` over a `QuantizedWeight`, so an NVFP4 head is required for
/// drafting. Correctness is unaffected: every draft is re-verified by the
/// main `lm_head_batched` (BF16 when the main head is BF16), so an
/// approximate draft head only changes acceptance rate, never an accepted
/// token.
pub(super) fn build_mtp_proposer(
    use_speculative: bool,
    mtp_weights: Vec<MtpWeights>,
    embed_tokens: DenseWeight,
    lm_head_nvfp4: Option<QuantizedWeight>,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    mtp_quant: crate::layers::MtpQuantization,
    mtp_vocab_size: u32,
    max_seq_len: usize,
) -> Option<Arc<dyn DraftProposer>> {
    if !use_speculative {
        if !mtp_weights.is_empty() {
            tracing::info!(
                "MTP weights available ({} module(s)) but --speculative not set, skipping MTP head construction",
                mtp_weights.len()
            );
        }
        return None;
    }
    if mtp_weights.is_empty() {
        return None;
    }
    let lm_nvfp4 = match lm_head_nvfp4 {
        Some(w) => w,
        None => {
            tracing::warn!(
                "MTP weights found but no NVFP4 LM head — speculative decoding disabled."
            );
            return None;
        }
    };
    let build_head = |mtp_wts: MtpWeights| {
        crate::layers::MtpHead::new(
            mtp_wts,
            embed_tokens,
            lm_nvfp4,
            config,
            gpu,
            mtp_quant,
            mtp_vocab_size,
            max_seq_len,
        )
    };
    if mtp_weights.len() == 1 {
        match build_head(mtp_weights.into_iter().next().unwrap()) {
            Ok(head) => {
                tracing::info!("MTP speculative decoding: ENABLED (single-module)");
                Some(Arc::new(head) as Arc<dyn DraftProposer>)
            }
            Err(e) => {
                tracing::warn!("Failed to build MTP head: {e}. Speculative decoding disabled.");
                None
            }
        }
    } else {
        let count = mtp_weights.len();
        let heads: Result<Vec<_>> = mtp_weights.into_iter().map(build_head).collect();
        match heads.and_then(crate::layers::mtp_multi::MultiModuleMtpHead::new) {
            Ok(multi) => {
                tracing::info!("MTP speculative decoding: ENABLED (multi-module, {count} heads)");
                Some(Arc::new(multi) as Arc<dyn DraftProposer>)
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to build multi-module MTP: {e}. Speculative decoding disabled."
                );
                None
            }
        }
    }
}
