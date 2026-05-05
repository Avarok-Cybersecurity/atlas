// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

//! Sub-init helpers for `TransformerModel::new`, hoisted to keep
//! `impl_a1.rs` under the 500 LoC cap.
//!
//! Each helper mirrors the equivalent inline block in `new()` 1:1.

use std::sync::Arc;

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;

use crate::speculative::DraftProposer;
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// Build the MTP draft proposer when speculative decoding is requested.
///
/// `mtp_weights` is a `Vec<MtpWeights>`:
///   - empty  → no MTP weights in checkpoint; proposer disabled
///   - len 1  → single-module MTP (Qwen3.5 family): build `MtpHead`
///   - len N>1 → multi-module MTP (MiniMax M2, DeepSeek-V3 style):
///     build `MultiModuleMtpHead` with N heads
///
/// Returns `None` when speculative decoding is off, when no MTP weights
/// are available, or when the LM head is not NVFP4 (MTP heads need the
/// NVFP4 LM head for shared output projection).
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
