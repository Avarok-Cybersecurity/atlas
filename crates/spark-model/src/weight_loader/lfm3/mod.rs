// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from HuggingFace transformers (models/lfm3, Apache-2.0, Copyright 2026 The HuggingFace Team) and vLLM (vllm/model_executor/models/lfm3.py, Apache-2.0, Copyright contributors to the vLLM project). Changes: ported from Python/Triton to Rust/CUDA for Atlas.

//! Responsibility: the Liquid AI LFM3 weight loader (target model only; text
//! checkpoints `LiquidAI/LFM3-1.7B` and `-FP8`).
//!
//! LFM3 is 24 layers of `h + post_op_norm(op(op_norm(h)))` then `h +
//! post_ffn_norm(swiglu(ffn_norm(h)))`, where `op` is a gated short convolution
//! (13 layers, lane L38's `Lfm3ConvLayer`), GQA full attention (6) or GQA
//! sliding-window attention (5, window 256). Attention rides the existing
//! `Qwen3AttentionLayer` (ungated Q, per-head QK norm, a per-head sigmoid
//! output gate, the four-norm structure, a window, table RoPE); the FFN is the
//! existing `DenseFfnLayer`. What this loader adds:
//!
//! * the tensor map and shape validation (`names.rs`): a wrong checkpoint
//!   fails at load with every discrepancy listed;
//! * the two RoPE tables (`rope_tables.rs`), in f64, one with an exact-zero
//!   frequency tail for the half-rotated full-attention layers;
//! * the FP8 routes (`fp8.rs`): Liquid's own E4M3 checkpoint kept native, and
//!   the opt-in `lfm3_fp8_all` BF16 -> FP8 conversion;
//! * the bridge to L38's conv layer (`conv_bridge.rs`).
//!
//! The tied head is the BF16 embedding table; `ModelConfig::
//! skip_lm_head_quantization` is true for `lfm3`, so it is never requantized.
//! Tensor parallelism, LoRA and speculative decoding are not supported (no
//! MTP head exists; the conv layer declines rollback and multi-row verify).
//!
//! License: the WEIGHTS are under Liquid AI's LFM Open License v1.0 (free
//! below USD 10M annual revenue, commercial use above it is not licensed).
//! Atlas bundles no weights; `docs/porting/lfm3.md` quotes the terms and the
//! boot log names them whenever an LFM3 checkpoint loads.

mod attention;
mod conv_bridge;
mod ffn;
mod fp8;
mod load_layers;
mod names;
mod rope_tables;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_fp8;
#[cfg(test)]
mod tests_layers;

use anyhow::Result;
use avarok_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, MtpWeights, dense, dense_auto, dense_auto_fp8_or_bf16};

/// Theta of the sliding-attention layers. `ModelConfig` has one `rope_theta`
/// (the full-attention layers'); `avarok_core`'s `parse_lfm3` refuses any
/// other sliding theta so this constant and the checkpoint cannot disagree.
pub(crate) const SLIDING_ROPE_THETA: f64 = 10_000.0;

/// Logged once whenever an LFM3 checkpoint loads.
pub(crate) const LICENSE_BANNER: &str = "LFM3 weights are licensed by Liquid AI under the LFM Open \
     License v1.0: commercial use is licensed only below USD 10,000,000 annual revenue; at or \
     above that threshold commercial use is not licensed under that agreement. Atlas bundles no \
     LFM3 weights; see docs/porting/lfm3.md";

/// A norm weight, the conv taps or the head gate: BF16 as stored, FP32
/// converted to BF16 once (`dense_auto`).
fn dense_norm(store: &WeightStore, gpu: &dyn GpuBackend, name: &str) -> Result<DenseWeight> {
    dense_auto(store, name, gpu)
}

pub struct Lfm3WeightLoader;

impl ModelWeightLoader for Lfm3WeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        load_layers::load_layers(store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, names::EMBED)
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense_norm(store, gpu, names::FINAL_NORM)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        // Tied checkpoints carry no `lm_head.weight`: the head IS the
        // embedding table. An untied one (not seen for LFM3) may ship the
        // head E4M3; the head consumer reads BF16, so dequantize that case.
        match store.get(names::LM_HEAD) {
            Ok(t) if t.dtype == WeightDtype::FP8E4M3 => {
                dense_auto_fp8_or_bf16(store, "lm_head", gpu)
            }
            Ok(_) => dense(store, names::LM_HEAD),
            Err(_) => dense(store, names::EMBED),
        }
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        // No MTP head and no published drafter (its tokenizer matches nothing
        // else Atlas serves).
        Ok(None)
    }
}
