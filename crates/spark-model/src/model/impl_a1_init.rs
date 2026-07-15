// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

//! Sub-init helpers for `TransformerModel::new`, hoisted to keep
//! `impl_a1.rs` under the 500 LoC cap.
//!
//! Each helper mirrors the equivalent inline block in `new()` 1:1.

use std::sync::Arc;

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::speculative::DraftProposer;
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// Allocate the GDN prefill scratch buffers, hoisted from
/// `TransformerModel::new` 1:1. Returns
/// `(qkv, gate_beta, out, z, gdn_buf_len)`. Buffers are only allocated when
/// GDN linear-attention layers exist (`conv_dim > 0`); Mamba-2 models
/// (Nemotron, conv_dim=0) get `DevicePtr::NULL`s to avoid `cuMemAlloc(0)`.
pub(super) fn build_gdn_prefill_buffers(
    config: &ModelConfig,
    max_batch_tokens: usize,
    max_seq_len: usize,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, DevicePtr, DevicePtr, DevicePtr, usize)> {
    let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let nv = config.linear_num_value_heads;
    let conv_dim = key_dim * 2 + value_dim;
    // GDN buffers only needed when GDN linear attention layers exist
    // (conv_dim > 0). Mamba-2 models (Nemotron) have conv_dim=0 — skip alloc
    // to avoid cuMemAlloc(0) error.
    let gdn_buf_len = max_batch_tokens.min(max_seq_len);
    let (gdn_qkv, gdn_gate_beta, gdn_out, gdn_z) = if conv_dim > 0 {
        let qkv = gpu.alloc(gdn_buf_len * conv_dim * 2)?;
        let gb = gpu.alloc(gdn_buf_len * nv * 2 * 4)?;
        let o = gpu.alloc(gdn_buf_len * value_dim * 2)?;
        let z = gpu.alloc(gdn_buf_len * value_dim * 2)?;
        let total_mb =
            (gdn_buf_len * (conv_dim * 2 + nv * 2 * 4 + value_dim * 2 * 2)) / (1024 * 1024);
        tracing::info!(
            "GDN prefill buffers: {total_mb} MB for {gdn_buf_len} tokens (chunked SSM prefill)"
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
    Ok((gdn_qkv, gdn_gate_beta, gdn_out, gdn_z, gdn_buf_len))
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

/// Build the optional SSM snapshot spill tier for `TransformerModel::new`,
/// hoisted 1:1 to keep `impl_a1.rs` under the 500-LoC cap. Returns `None`
/// (the byte-identical default) unless `ATLAS_SSM_TIER` is set on a recurrent
/// model; otherwise selects the env-driven backend keyed by the model
/// fingerprint (so different models sharing one peer can't collide).
///
/// `blob_bytes` MUST be `SsmSnapshotPool::spill_blob_bytes()` so the tier's
/// fixed blob sizing matches the spill/fault-in gathers.
pub(super) fn build_ssm_tier_store(
    config: &ModelConfig,
    blob_bytes: usize,
    num_ssm_layers: usize,
) -> Result<Option<Arc<dyn super::ssm_tier::SnapshotBlobStore>>> {
    if super::ssm_tier::ssm_tier_enabled() && num_ssm_layers > 0 {
        let fp = super::ssm_tier::ModelFingerprint::derive(config, blob_bytes)?;
        Ok(Some(super::ssm_tier::build_tier_store(fp, blob_bytes)?))
    } else {
        Ok(None)
    }
}
