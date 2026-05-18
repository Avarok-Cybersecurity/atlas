// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-4.7-Flash weight loader (DeepSeek-V3-style MLA + 64-expert MoE + 1-module MTP).
//!
//! Scaffolding only — phase-0 of the GLM-4.7-Flash integration. All
//! `load_*` entry points return `unimplemented!()`. See
//! `docs/design/GLM-4.7-FLASH-IMPL-PLAN.md` for the phased plan.
//!
//! Architecture (`config.json` → `Glm4MoeLiteForCausalLM`):
//!   * 47 hidden layers; `first_k_dense_replace = 1` → layer 0 is a dense
//!     FFN (gate/up/down), layers 1–46 are MoE blocks.
//!   * MLA attention with q-LoRA rank 768, kv-LoRA rank 512,
//!     qk_nope_head_dim 192, qk_rope_head_dim 64, v_head_dim 256,
//!     20 Q heads = 20 KV heads. Attention weights stay BF16 (the
//!     checkpoint's `ignore: re:.*self_attn.*` rule keeps them out of
//!     NVFP4). Tensor names:
//!       - `q_a_proj`, `q_a_layernorm`, `q_b_proj`
//!       - `kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`
//!       - `o_proj`
//!   * MoE: 64 routed experts + 1 shared expert, top-k 4,
//!     `topk_method = "noaux_tc"` (sigmoid + `e_score_correction_bias`,
//!     `routed_scaling_factor = 1.8`). Gate weight `[64, 2048]` BF16,
//!     bias `[64]` BF16. Each expert {gate, up, down}_proj is NVFP4.
//!   * MTP: 1 nextn module at index 47. Tensors:
//!     `model.layers.47.{eh_proj, embed_tokens, enorm, hnorm, input_layernorm,
//!     post_attention_layernorm, self_attn.*, mlp.gate.*, mlp.experts.*}`
//!     `eh_proj` is NVFP4 `[hidden, 2*hidden]` (vs Qwen3.5's BF16
//!     `mtp.fc`) — needs an NVFP4 variant of `load_mtp`.
//!   * NVFP4 layout: compressed-tensors w/ `weight_packed`, `weight_scale`,
//!     `weight_global_scale`, `input_global_scale` — same format as
//!     Qwen3.5 NVFP4 checkpoints; reuse `qwen35` helpers.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, MtpWeights};

pub struct Glm4LiteWeightLoader;

impl ModelWeightLoader for Glm4LiteWeightLoader {
    fn supports_tp(&self) -> bool {
        // GB10 is single-GPU; TP=1 covers the deployment target. MLA
        // shards differently from standard QKV (the LoRA-A projections
        // are replicated, LoRA-B is column-parallel) — defer until a
        // multi-GPU GLM target appears.
        false
    }

    fn load_layers(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        // Phase 2 — Weight loader (see plan §Phase 2).
        //
        // For layer 0 (dense FFN):
        //   - input_layernorm, post_attention_layernorm: BF16 RMSNorm weights.
        //   - mlp.{gate,up,down}_proj: NVFP4 compressed-tensors. Use the
        //     same load path as `qwen35_dense::load_layers` dense FFN.
        //   - self_attn.*: MLA (see below).
        //
        // For layers 1..46 (MoE blocks):
        //   - MoE gate: BF16 `[n_routed_experts, hidden]` + bias `[n_routed]`.
        //   - 64 routed experts + 1 shared expert, all NVFP4.
        //   - self_attn.*: MLA (see below).
        //
        // MLA loading (every layer):
        //   - q_a_proj BF16 `[q_lora_rank=768, hidden=2048]`
        //   - q_a_layernorm BF16 `[768]`
        //   - q_b_proj BF16 `[num_heads(20) * (qk_nope(192)+qk_rope(64)) = 5120, 768]`
        //   - kv_a_proj_with_mqa BF16 `[kv_lora_rank(512)+qk_rope(64)=576, 2048]`
        //   - kv_a_layernorm BF16 `[512]`
        //   - kv_b_proj BF16 `[num_heads(20)*(qk_nope(192)+v_head_dim(256))=8960, 512]`
        //   - o_proj BF16 `[hidden(2048), num_heads(20)*v_head_dim(256)=5120]`
        //
        // First-pass correctness: use the MLA→GQA expansion path from
        // `mistral_loader::gpu_matmul` (Q = q_b @ q_a, K/V split from
        // kv_b @ kv_a[:kv_lora]) to validate output before enabling
        // `qwen3_attention::decode::attention_forward_mla`.
        unimplemented!(
            "Glm4LiteWeightLoader::load_layers: phase-2 unimplemented — \
             see docs/design/GLM-4.7-FLASH-IMPL-PLAN.md §Phase 2"
        )
    }

    fn load_embedding(&self, _store: &WeightStore, _config: &ModelConfig) -> Result<DenseWeight> {
        // `model.embed_tokens.weight` BF16, shape `[vocab=154880, hidden=2048]`.
        unimplemented!("Glm4LiteWeightLoader::load_embedding: phase-2 unimplemented")
    }

    fn load_final_norm(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        // `model.norm.weight` BF16 RMSNorm, `[hidden]`.
        unimplemented!("Glm4LiteWeightLoader::load_final_norm: phase-2 unimplemented")
    }

    fn load_lm_head(&self, _store: &WeightStore, _config: &ModelConfig) -> Result<DenseWeight> {
        // `lm_head.weight` BF16 `[vocab=154880, hidden=2048]`. The
        // checkpoint's `ignore: ["lm_head", "re:.*embed.*"]` rule keeps
        // both tied embeddings and the head in BF16. `tie_word_embeddings`
        // is false in the config, so do NOT fall back to embed_tokens.
        unimplemented!("Glm4LiteWeightLoader::load_lm_head: phase-2 unimplemented")
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        // Phase 4 — MTP head (DeepSeek-V3 single-module layout at index 47).
        //
        // Probe `model.layers.{num_hidden_layers}.eh_proj.weight_packed`
        // before loading. Return `Ok(None)` cleanly when absent so MTP is
        // optional. When present, load:
        //   - eh_proj: NVFP4 `[hidden, 2*hidden]` — needs an NVFP4
        //     variant of `crate::weight_map::load_mtp` (Qwen3.5's helper
        //     assumes BF16 `mtp.fc`).
        //   - embed_tokens: BF16 `[vocab, hidden]` (separate from the
        //     main embedding; DeepSeek-V3 trains them independently).
        //   - enorm, hnorm, input_layernorm, post_attention_layernorm:
        //     BF16 RMSNorm `[hidden]`.
        //   - self_attn.* MLA block + mlp.gate + 64 routed + 1 shared
        //     experts — same shape as a layers 1–46 MoE block.
        //
        // Return shape: `Some(MtpWeights { ... })` consumed by
        // `crate::layers::mtp_head` for K=2 speculative decoding.
        unimplemented!(
            "Glm4LiteWeightLoader::load_mtp_weights: phase-4 unimplemented — \
             see docs/design/GLM-4.7-FLASH-IMPL-PLAN.md §Phase 4"
        )
    }
}
