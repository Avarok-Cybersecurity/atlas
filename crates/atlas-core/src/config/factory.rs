// SPDX-License-Identifier: AGPL-3.0-only

//! Hard-coded model factories. Split out of `config.rs` for file-size budget.

#![allow(unused_imports)]

use super::{LayerType, ModelConfig, QuantizationConfig};

impl ModelConfig {
    pub fn qwen3_next_80b_nvfp4() -> Self {
        Self {
            hidden_size: 2048,
            num_hidden_layers: 48,
            intermediate_size: 5120,
            vocab_size: 151936,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            head_dim: 256,
            partial_rotary_factor: 0.25,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_num_value_heads: 32,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            num_experts: 512,
            num_experts_per_tok: 10,
            moe_intermediate_size: 512,
            shared_expert_intermediate_size: 512,
            norm_topk_prob: true,
            decoder_sparse_step: 1,
            layer_types: {
                let mut types = Vec::with_capacity(48);
                for i in 0..48 {
                    if (i + 1) % 4 == 0 {
                        types.push(LayerType::FullAttention);
                    } else {
                        types.push(LayerType::LinearAttention);
                    }
                }
                types
            },
            full_attention_interval: 4,
            sliding_window: 0,
            max_position_embeddings: 262144,
            rope_theta: 10_000_000.0,
            rms_norm_eps: 1e-6,
            bos_token_id: 151643,
            eos_token_id: 151645,
            tie_word_embeddings: false,
            model_type: "qwen3_next".to_string(),
            mtp_num_hidden_layers: 1,
            weight_prefix: String::new(),
            ep_rank: 0,
            ep_world_size: 1,
            tp_rank: 0,
            tp_world_size: 1,
            hybrid_override_pattern: String::new(),
            mamba_num_heads: 0,
            mamba_head_dim: 0,
            ssm_state_size: 0,
            n_groups: 0,
            expand: 0,
            n_routed_experts: 0,
            norm_eps: 0.0,
            conv_kernel: 0,
            moe_shared_expert_intermediate_size: 0,
            routed_scaling_factor: 1.0,
            moe_latent_size: 0,
            vision: None,
            quantization_config: None,
            attn_gated: true,
            nested_config: false,
            mrope_section: [0, 0, 0],
            mrope_interleaved: false,
            kv_lora_rank: 0,
            kv_layer_dims: Vec::new(),
            q_lora_rank: 0,
            qk_nope_head_dim: 0,
            qk_rope_head_dim: 0,
            v_head_dim: 0,
            yarn_factor: 0.0,
            yarn_beta_slow: 0.0,
            yarn_beta_fast: 0.0,
            yarn_original_max_position_embeddings: 0,
            llama_4_scaling_beta: 0.0,
            llama_4_scaling_original_max_position_embeddings: 0,
            fp8_kv_calibration_tokens: 0,
            final_logit_softcapping: 0.0,
            embed_scale: 0.0,
            // MiniMax M2 fields — all default to "unused" (empty/0) so the
            // base Qwen3-Next-80B template behaves identically to before.
            scoring_func: String::new(),
            use_routing_bias: false,
            qk_norm_type: String::new(),
            num_mtp_modules: 0,
            mtp_transformer_layers: 0,
            rotary_dim: 0,
            dflash_capture_layers: Vec::new(),
        }
    }
}
