// SPDX-License-Identifier: AGPL-3.0-only

//! Small `ModelWeightLoader` method bodies split out of `qwen35_dense.rs`
//! for the ≤500 LoC file-size cap. Called from the trait impl in the parent.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::weight_map::{DenseWeight, dense, dense_auto_fp8_or_bf16};

pub(super) fn load_embedding(store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
    let prefix = &config.weight_prefix;
    dense(store, &format!("{prefix}.embed_tokens.weight"))
}

pub(super) fn load_final_norm(store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
    let prefix = &config.weight_prefix;
    dense(store, &format!("{prefix}.norm.weight"))
}

/// Load the LM head, dequanting an **FP8** head to BF16 and otherwise handing
/// the tensor through untouched.
///
/// `dense()` performs no dtype check — it hands the raw device pointer to the
/// consumer. That is correct for the two layouts Atlas already supported:
/// a BF16 head, and a Standard-NVFP4 head (`weight` U8-packed +
/// `weight_scale`/`weight_scale_2`, e.g. nvidia/Qwen3.6-27B-NVFP4), which the
/// LM-head consumer unpacks itself. Both MUST keep the passthrough.
///
/// It is NOT correct for FP8. Mixed-precision NVFP4 checkpoints (unsloth
/// Qwen3.6-*-NVFP4, re-quantized 2026-07-10) keep `lm_head` as FP8 E4M3 + a
/// per-row `weight_scale` while the body of the net is NVFP4. `lm_head` is the
/// largest tensor in the model ([248320, 5120] = 1.27 GB of 1-byte elements on
/// the 27B); handed to a BF16 GEMM it is a 2.54 GB read off a 1.27 GB
/// allocation, surfacing as `CUDA_ERROR_ILLEGAL_ADDRESS` at the first sync
/// after model build.
///
/// So: intercept FP8 only. Dispatching every dtype through
/// `dense_auto_fp8_or_bf16` would hard-error on the U8 packed head
/// (`unsupported dtype UInt8`) and break every Standard-NVFP4 checkpoint.
pub(super) fn load_lm_head(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    for prefix in ["lm_head", "language_model.lm_head", "model.lm_head"] {
        let key = format!("{prefix}.weight");
        if !store.contains(&key) {
            continue;
        }
        let is_fp8 = store
            .get(&key)
            .map(|w| w.dtype == WeightDtype::FP8E4M3)
            .unwrap_or(false);
        return if is_fp8 {
            dense_auto_fp8_or_bf16(store, prefix, gpu)
        } else {
            dense(store, &key)
        };
    }
    // Tied embeddings: the head IS the embedding table (BF16 in every
    // checkpoint Atlas supports — no FP8 embed_tokens has been seen).
    let prefix = &config.weight_prefix;
    dense(store, &format!("{prefix}.embed_tokens.weight"))
}
