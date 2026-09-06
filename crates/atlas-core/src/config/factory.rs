// SPDX-License-Identifier: AGPL-3.0-only

//! Hard-coded model factories. Split out of `config.rs` for file-size budget.

#![allow(unused_imports)]

use super::{LayerType, ModelConfig, QuantizationConfig};

impl ModelConfig {
    pub fn qwen3_next_80b_nvfp4() -> Self {
        Self {
            profile: false,
            hidden_size: 2048,
            num_hidden_layers: 48,
            intermediate_size: 5120,
            vocab_size: 151936,
            num_attention_heads: 16,
            num_attention_heads_per_layer: Vec::new(),
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
            lm_head_bf16_override: None,
            lm_head_fp8: false,
            model_type: "qwen3_next".to_string(),
            mtp_num_hidden_layers: 1,
            dspark_block_size: 0,
            dspark_noise_token_id: 0,
            dspark_target_layer_ids: Vec::new(),
            dspark_markov_rank: 0,
            weight_prefix: String::new(),
            // No plan: every expert loads. `--expert-category` installs one
            // after the config is parsed.
            bel: None,
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
            mlp_only_layers: Vec::new(),
            moe_latent_size: 0,
            moe_intermediate_sizes: Vec::new(),
            num_experts_per_toks: Vec::new(),
            vision: None,
            quantization_config: None,
            attn_gated: true,
            gdn_norm_sigmoid: false,
            nested_config: false,
            mrope_section: [0, 0, 0],
            mrope_interleaved: false,
            kv_lora_rank: 0,
            zero_expert_num: 0,
            ngram_vocab_size_ratio: 0,
            emb_neighbor_num: 0,
            emb_split_num: 0,
            ngram_vocab_size_base: 0,
            ngram_split_parts: 0,
            ple_layer_ids: Vec::new(),
            ple_conv_kernel_size: 0,
            kv_layer_dims: Vec::new(),
            q_lora_rank: 0,
            qk_nope_head_dim: 0,
            qk_rope_head_dim: 0,
            v_head_dim: 0,
            o_lora_rank: 0,
            o_groups: 0,
            yarn_mscale: 1.0,
            yarn_mscale_all_dim: 0.0,
            hc_mult: 0,
            hc_sinkhorn_iters: 0,
            hc_lowrank: 0,
            final_norm_identity: false,
            hc_eps: 1e-6,
            compress_ratios: Vec::new(),
            index_compress_ratio: 0,
            index_n_heads: 0,
            index_head_dim: 0,
            index_topk: 0,
            num_hash_layers: 0,
            yarn_factor: 0.0,
            yarn_beta_slow: 0.0,
            yarn_beta_fast: 0.0,
            yarn_original_max_position_embeddings: 0,
            yarn_attention_factor: 1.0,
            llama_4_scaling_beta: 0.0,
            llama_4_scaling_original_max_position_embeddings: 0,
            fp8_kv_calibration_tokens: 0,
            fp8_kv_headroom: 2.0,
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
            dflash_gamma: None,
            adapter_max_rank: 0,
        }
    }
}
