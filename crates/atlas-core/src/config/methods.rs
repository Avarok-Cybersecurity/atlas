// SPDX-License-Identifier: AGPL-3.0-only

//! [`ModelConfig`] inherent helper methods. Split out of `config.rs` for
//! file-size budget. Pure derived getters + small predicates over the
//! struct fields.

#![allow(unused_imports)]

use super::{LayerType, ModelConfig};

impl ModelConfig {
    /// GQA ratio: number of Q heads per KV head.
    pub fn gqa_ratio(&self) -> usize {
        if self.num_key_value_heads == 0 {
            1
        } else {
            self.num_attention_heads / self.num_key_value_heads
        }
    }

    /// Layer type for a given layer index.
    /// Falls back to full_attention_interval if layer_types is empty.
    pub fn layer_type(&self, layer_idx: usize) -> LayerType {
        if !self.layer_types.is_empty() {
            self.layer_types
                .get(layer_idx)
                .cloned()
                .unwrap_or(LayerType::FullAttention)
        } else if self.full_attention_interval > 0
            && (layer_idx + 1).is_multiple_of(self.full_attention_interval)
        {
            LayerType::FullAttention
        } else {
            LayerType::LinearAttention
        }
    }

    /// Number of full attention layers.
    pub fn num_attention_layers(&self) -> usize {
        if !self.layer_types.is_empty() {
            self.layer_types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count()
        } else if self.full_attention_interval == 0 {
            self.num_hidden_layers
        } else {
            self.num_hidden_layers / self.full_attention_interval
        }
    }

    /// Number of SSM (linear attention) layers.
    pub fn num_ssm_layers(&self) -> usize {
        if !self.layer_types.is_empty() {
            self.layer_types
                .iter()
                .filter(|t| **t == LayerType::LinearAttention)
                .count()
        } else {
            self.num_hidden_layers - self.num_attention_layers()
        }
    }

    /// Rotary embedding dimension.
    ///
    /// Priority:
    /// 1. Explicit `rotary_dim` field (MiniMax M2 — integer in config.json).
    /// 2. `partial_rotary_factor * head_dim` (Qwen3/Gemma-4 convention — float).
    pub fn rotary_dim(&self) -> usize {
        if self.rotary_dim > 0 {
            self.rotary_dim
        } else {
            (self.partial_rotary_factor * self.head_dim as f64) as usize
        }
    }

    /// SSM projection output size: Q + K + V + Z concatenated.
    pub fn ssm_qkvz_size(&self) -> usize {
        let q = self.linear_num_key_heads * self.linear_key_head_dim;
        let k = self.linear_num_key_heads * self.linear_key_head_dim;
        let v = self.linear_num_value_heads * self.linear_value_head_dim;
        let z = self.linear_num_value_heads * self.linear_value_head_dim;
        q + k + v + z
    }

    /// SSM QKV projection output size (without Z): Q + K + V.
    pub fn ssm_qkv_size(&self) -> usize {
        let q = self.linear_num_key_heads * self.linear_key_head_dim;
        let k = self.linear_num_key_heads * self.linear_key_head_dim;
        let v = self.linear_num_value_heads * self.linear_value_head_dim;
        q + k + v
    }

    /// SSM Z gate projection output size.
    pub fn ssm_z_size(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// SSM beta+alpha projection output size.
    pub fn ssm_ba_size(&self) -> usize {
        // beta: num_value_heads, alpha: num_value_heads
        self.linear_num_value_heads * 2
    }

    /// Range of expert indices local to this EP rank.
    /// Returns (start, end) where start is inclusive and end is exclusive.
    pub fn local_expert_range(&self) -> (usize, usize) {
        if self.ep_world_size <= 1 {
            return (0, self.num_experts);
        }
        let per_rank = self.num_experts / self.ep_world_size;
        let start = self.ep_rank * per_rank;
        let end = if self.ep_rank == self.ep_world_size - 1 {
            self.num_experts // last rank gets remainder
        } else {
            start + per_rank
        };
        (start, end)
    }

    /// Whether the given expert ID is local to this EP rank.
    pub fn is_local_expert(&self, expert_id: usize) -> bool {
        let (start, end) = self.local_expert_range();
        expert_id >= start && expert_id < end
    }

    /// Range `[start, end)` of a `total`-sized dimension owned by this TP rank.
    /// `total` must be divisible by `tp_world_size`. Returns `(0, total)` when
    /// TP is disabled.
    pub fn tp_shard_range(&self, total: usize) -> (usize, usize) {
        if self.tp_world_size <= 1 {
            return (0, total);
        }
        debug_assert!(
            total.is_multiple_of(self.tp_world_size),
            "tp_shard_range: total={} not divisible by tp_world_size={}",
            total,
            self.tp_world_size,
        );
        let per_rank = total / self.tp_world_size;
        let start = self.tp_rank * per_rank;
        (start, start + per_rank)
    }

    /// Per-rank shard size for a `total`-sized dimension under TP.
    pub fn tp_shard_dim(&self, total: usize) -> usize {
        if self.tp_world_size <= 1 {
            return total;
        }
        total / self.tp_world_size
    }

    /// Weight key prefix for layer-level weights.
    /// Returns `"model.layers"` for flat models (qwen3_next),
    /// or `"model.language_model.layers"` for conditional generation models (qwen3_5_moe).
    pub fn layer_prefix(&self, layer_idx: usize) -> String {
        if self.weight_prefix.is_empty() {
            format!("model.layers.{layer_idx}")
        } else {
            format!("{}.layers.{layer_idx}", self.weight_prefix)
        }
    }

    /// Derive model-agnostic capabilities from this config.
    pub fn capabilities(&self) -> crate::capabilities::ModelCapabilities {
        crate::capabilities::ModelCapabilities::from_config(self)
    }

    // ── Factory sub-dispatch predicates ──
    // Used only by loader_for_config() to select the right weight loader
    // within the qwen3_5_moe model_type family. Not for general use —
    // prefer config fields (attn_gated, nested_config) or capabilities.

    /// Factory use only. Prefer `config.attn_gated` or `config.capabilities()`.
    pub fn is_qwen35(&self) -> bool {
        self.model_type == "qwen3_5_moe"
    }

    /// Factory use only.
    pub fn is_qwen35_dense(&self) -> bool {
        self.model_type == "qwen3_5" && self.num_experts == 0
    }

    /// Factory use only.
    pub fn is_qwen3_vl(&self) -> bool {
        self.model_type == "qwen3_vl_moe"
    }

    /// Whether to skip NVFP4 quantization of the LM head.
    /// MLA models (kv_lora_rank > 0) lose logit precision under NVFP4.
    /// Gemma-4 dense (31B): the LM head ties to BF16 embed_tokens whose
    /// rows have heavy outliers (final_norm.weight max=510, several
    /// embedding rows in similar range). The runtime BF16→NVFP4 path
    /// uses a single per-tensor absmax for `scale2`, which forces a
    /// coarse scale that loses ~7 bits in normal-magnitude rows. For a
    /// 262 144-row vocab matrix that compounds into the 0.14-margin
    /// argmax flip on creative prompts (verified 2026-05-01 via FP32
    /// lm_head bisection: NVFP4 output had top1=` a` 21.85 vs FP32 BF16
    /// view top1=` a` 21.85 — quantization noise was visible in the
    /// SAME logit channel that flipped the tiebreak). Skipping the
    /// runtime quantization keeps the LM head as plain BF16 dense; the
    /// FP32 lm_head path (gated by `ATLAS_GEMMA4_FP32_LMHEAD=1`) can
    /// then act on full-precision weights without the NVFP4 floor.
    pub fn skip_lm_head_quantization(&self) -> bool {
        if self.kv_lora_rank > 0 {
            return true;
        }
        if self.model_type == "gemma4" && self.num_experts == 0 {
            // Allow rollback via env for A/B testing.
            return std::env::var("ATLAS_GEMMA4_LMHEAD_NVFP4").ok().as_deref() != Some("1");
        }
        false
    }

    /// Whether to use FP32 residual stream (hidden_states/residual buffers).
    /// Enabled for GDN models (Qwen3.5) where BF16 truncation across 48 layers
    /// FP32 residual prevents BF16 truncation across 48 layers but costs 2x
    /// memory bandwidth (135 → 63 tok/s on GB10's LPDDR5X).
    /// Controlled by ATLAS_FP32_RESIDUAL env var (default: from HARDWARE.toml).
    /// On bandwidth-limited systems (GB10), BF16 gives 2x throughput.
    /// On HBM systems (B200, H100), FP32 is free and improves long-context quality.
    pub fn use_fp32_residual(&self) -> bool {
        // Gemma-4 dense variants (31B) opt into FP32 residual to survive
        // cumulative layer_scalar underflow. Gemma-4 MoE variants (26B)
        // have a dual-FFN decode branch (`qwen3_attention/trait_impl.rs`
        // line 144-180) where `moe_ffn.forward(hidden, …)` reads the raw
        // hidden pointer — that path is not FP32-aware. Enabling FP32
        // residual on 26B makes the MoE path read FP32 bytes as BF16,
        // corrupting the whole decode ("<unused6226>" collapse).
        //
        // Gate on "has MoE experts": dense Gemma-4 (31B) gets FP32; MoE
        // Gemma-4 (26B) keeps BF16. Env var `ATLAS_GEMMA4_FP32=0`
        // disables even for 31B for A/B testing.
        if self.model_type == "gemma4" {
            if self.num_experts > 0 {
                return false; // Gemma-4 MoE path not FP32-safe
            }
            return std::env::var("ATLAS_GEMMA4_FP32").ok().as_deref() != Some("0");
        }
        if self.linear_num_key_heads == 0 {
            return false; // Other non-GDN models have BF16 residual only
        }
        // Check env override first, then fall back to build-time default
        match std::env::var("ATLAS_FP32_RESIDUAL") {
            Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
            Err(_) => {
                // Build-time default from HARDWARE.toml (baked via env at compile)
                option_env!("ATLAS_HW_FP32_RESIDUAL").map_or(false, |v| v == "1" || v == "true")
            }
        }
    }

    /// Mamba-2 d_inner = mamba_num_heads * mamba_head_dim.
    pub fn mamba2_d_inner(&self) -> usize {
        self.mamba_num_heads * self.mamba_head_dim
    }

    /// Mamba-2 d_xBC = d_inner + 2 * n_groups * ssm_state_size.
    /// This is the dimension that goes through conv1d (x + B + C concatenated).
    pub fn mamba2_d_xbc(&self) -> usize {
        self.mamba2_d_inner() + 2 * self.n_groups * self.ssm_state_size
    }

    /// Mamba-2 in_proj output size = z + xBC + dt.
    pub fn mamba2_in_proj_size(&self) -> usize {
        self.mamba2_d_inner() + self.mamba2_d_xbc() + self.mamba_num_heads
    }

    /// Per-layer SSM hidden state size in bytes (FP32).
    /// Dispatches on SSM architecture: Mamba-2 vs GDN, using config fields.
    pub fn ssm_h_state_bytes(&self) -> usize {
        if self.mamba_num_heads > 0 && self.mamba_head_dim > 0 {
            // Mamba-2: h[num_heads, head_dim, state_size] FP32
            self.mamba_num_heads * self.mamba_head_dim * self.ssm_state_size * 4
        } else {
            // GDN: h[nv, vd, kd] FP32
            self.linear_num_value_heads * self.linear_value_head_dim * self.linear_key_head_dim * 4
        }
    }

    /// Per-layer SSM conv state size in bytes (FP32).
    pub fn ssm_conv_state_bytes(&self) -> usize {
        let d_conv = self.linear_conv_kernel_dim;
        if self.mamba_num_heads > 0 && self.mamba_head_dim > 0 {
            // Mamba-2: conv: [d_xBC, d_conv] FP32
            self.mamba2_d_xbc() * d_conv * 4
        } else {
            // GDN: conv: [conv_dim, d_conv] FP32
            let conv_dim = self.linear_num_key_heads * self.linear_key_head_dim * 2
                + self.linear_num_value_heads * self.linear_value_head_dim;
            conv_dim * d_conv * 4
        }
    }

    /// SSM state normalization dimensions: (num_heads, k_dim, v_dim).
    /// Used by the state normalization kernel to prevent drift.
    pub fn ssm_state_norm_dims(&self) -> (usize, usize, usize) {
        if self.mamba_num_heads > 0 && self.mamba_head_dim > 0 {
            (
                self.mamba_num_heads,
                self.mamba_head_dim,
                self.ssm_state_size,
            )
        } else {
            (
                self.linear_num_value_heads,
                self.linear_key_head_dim,
                self.linear_value_head_dim,
            )
        }
    }

    /// MoE expert input dimension: latent size if LatentMoE, else hidden_size.
    pub fn moe_input_size(&self) -> usize {
        if self.moe_latent_size > 0 {
            self.moe_latent_size
        } else {
            self.hidden_size
        }
    }

    /// Number of MoE-only layers (Nemotron-H).
    pub fn num_moe_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| **t == LayerType::Moe)
            .count()
    }
}
