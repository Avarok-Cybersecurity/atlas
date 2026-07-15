// SPDX-License-Identifier: AGPL-3.0-only

//! Tests split out of `config.rs` for file-size budget.

#![allow(unused_imports)]

use super::*;

#[test]
fn test_qwen3_default_config() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    assert_eq!(cfg.num_hidden_layers, 48);
    assert_eq!(cfg.num_experts, 512);
    assert_eq!(cfg.num_attention_layers(), 12);
    assert_eq!(cfg.num_ssm_layers(), 36);
    assert_eq!(cfg.gqa_ratio(), 8);
    assert_eq!(cfg.rotary_dim(), 64);
    assert_eq!(cfg.vocab_size, 151936);
    assert_eq!(cfg.layer_type(2), LayerType::LinearAttention);
    assert_eq!(cfg.layer_type(3), LayerType::FullAttention);
    assert_eq!(cfg.layer_type(47), LayerType::FullAttention);
    assert_eq!(cfg.ssm_qkvz_size(), 2048 + 2048 + 4096 + 4096);
    assert_eq!(cfg.ssm_ba_size(), 64);
}

#[test]
fn test_parse_actual_config() {
    let json = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test_data/qwen3_config.json"
    ));
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_experts, 512);
    assert_eq!(cfg.num_hidden_layers, 48);
    assert_eq!(cfg.layer_types.len(), 48);
    assert_eq!(cfg.layer_types[0], LayerType::LinearAttention);
    assert_eq!(cfg.layer_types[3], LayerType::FullAttention);
    assert_eq!(cfg.vocab_size, 151936);
    assert_eq!(cfg.rope_theta, 10_000_000.0);
    assert!(cfg.norm_topk_prob);
    assert!(!cfg.tie_word_embeddings);
    assert_eq!(cfg.rms_norm_eps, 1e-6);
    assert_eq!(cfg.partial_rotary_factor, 0.25);
    assert_eq!(cfg.model_type, "qwen3_next");
    assert!(cfg.weight_prefix.is_empty());
}

#[test]
fn test_parse_qwen35_nested_config() {
    let json = r#"{
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "vocab_size": 248320,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ],
            "rope_parameters": {
                "rope_theta": 10000000,
                "rope_type": "default"
            },
            "mtp_num_hidden_layers": 1
        }
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "qwen3_5_moe");
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_hidden_layers, 40);
    assert_eq!(cfg.num_experts, 256);
    assert_eq!(cfg.num_experts_per_tok, 8);
    assert_eq!(cfg.vocab_size, 248320);
    assert_eq!(cfg.num_attention_layers(), 10);
    assert_eq!(cfg.num_ssm_layers(), 30);
    assert_eq!(cfg.layer_types.len(), 40);
    assert_eq!(cfg.eos_token_id, 248044);
    assert_eq!(cfg.rope_theta, 10_000_000.0);
    assert!(cfg.is_qwen35());
    assert!(cfg.norm_topk_prob); // Qwen3.5 unconditionally normalizes
    assert_eq!(cfg.ssm_qkv_size(), 2048 + 2048 + 4096); // 8192
    assert_eq!(cfg.ssm_z_size(), 4096);
    assert_eq!(cfg.mtp_num_hidden_layers, 1);
}

#[test]
fn test_parse_qwen3_vl_config() {
    let json = r#"{
        "model_type": "qwen3_vl_moe",
        "text_config": {
            "model_type": "qwen3_vl_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "vocab_size": 151936,
            "rope_theta": 5000000,
            "norm_topk_prob": true
        }
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "qwen3_vl_moe");
    assert!(cfg.is_qwen3_vl());
    assert!(!cfg.is_qwen35());
    assert!(cfg.capabilities().has_nested_config);
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert_eq!(cfg.num_experts, 128);
    assert_eq!(cfg.num_hidden_layers, 48);
    // Pure attention: all layers are FullAttention (full_attention_interval defaults to 1)
    assert_eq!(cfg.num_attention_layers(), 48);
    assert_eq!(cfg.num_ssm_layers(), 0);
    assert_eq!(cfg.gqa_ratio(), 8);
    // Full rotary: partial_rotary_factor defaults to 1.0
    assert_eq!(cfg.rotary_dim(), 128);
    assert_eq!(cfg.rope_theta, 5_000_000.0);
    assert!(cfg.norm_topk_prob);
}

/// Qwen3.5-VL detection: the trunk `model_type` stays `qwen3_5`
/// (same as the text-only variant) but the upstream config ships a
/// `vision_config` block plus `architectures =
/// ["Qwen3_5ForConditionalGeneration"]`. `is_qwen3_vl()` must
/// distinguish via the parsed `config.vision` so the factory routes
/// the checkpoint to the VL weight loader instead of the dense LLM
/// loader.
#[test]
fn test_parse_qwen3_5_vl_config() {
    let json = r#"{
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5",
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 9216,
            "vocab_size": 248320,
            "rope_theta": 10000000.0
        },
        "vision_config": {
            "hidden_size": 1024,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "patch_size": 16,
            "spatial_merge_size": 2
        }
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "qwen3_5");
    assert!(
        cfg.is_qwen3_vl(),
        "Qwen3.5-VL detected via model_type=qwen3_5 + vision_config presence"
    );
    assert!(cfg.vision.is_some());
}

/// Counter-test: a text-only `qwen3_5` config WITHOUT `vision_config`
/// must NOT be misclassified as VL. Pins the gate condition is
/// actually using `vision.is_some()`, not just model_type.
#[test]
fn test_qwen3_5_text_only_not_vl() {
    let json = r#"{
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5",
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 151936,
            "rope_theta": 10000000.0
        }
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "qwen3_5");
    assert!(
        !cfg.is_qwen3_vl(),
        "qwen3_5 without vision_config must not be classified as VL"
    );
}

/// Regression for the alpha-2.99 dispatch bug:
/// Kbenkhaled/Qwen3.5-27B-NVFP4 is a *dense* hybrid (top model_type
/// "qwen3_5", num_experts=0) that nonetheless enables MRoPE in
/// text_config.rope_parameters. Pre-c0cde18, the MRoPE detector
/// rewrote model_type → "qwen3_6_moe" unconditionally, then the
/// kernel dispatcher couldn't find a target for
/// (qwen3_6_moe, hidden_size=5120) — only (qwen3_6_moe, 2048) for
/// qwen3.6-35b-a3b exists. The fix gates the rewrite on
/// is_moe(top_model_type). This test pins that contract.
#[test]
fn test_kbenkhaled_qwen35_27b_dense_mrope_no_rewrite() {
    let json = r#"{
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 17408,
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 248320,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "rope_parameters": {
                "rope_theta": 10000000,
                "rope_type": "default",
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10]
            }
        }
    }"#;
    let cfg = parse_config(json).unwrap();
    // Critical: dense + MRoPE must NOT be rewritten to qwen3_6_moe,
    // or the dispatcher won't find the qwen3.5-27b kernel target.
    assert_eq!(cfg.model_type, "qwen3_5");
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.num_experts, 0);
    // MRoPE flags still parsed so the kernel uses the right rope path.
    assert!(cfg.mrope_interleaved);
    assert_eq!(cfg.mrope_section, [11, 11, 10]);
}

#[test]
fn test_layer_prefix() {
    let cfg80b = ModelConfig::qwen3_next_80b_nvfp4();
    assert_eq!(cfg80b.layer_prefix(3), "model.layers.3");

    let mut cfg35 = ModelConfig::qwen3_next_80b_nvfp4();
    cfg35.weight_prefix = "model.language_model".to_string();
    assert_eq!(cfg35.layer_prefix(3), "model.language_model.layers.3");
}

#[test]
fn test_parse_nemotron_h_config() {
    let json = r#"{
        "model_type": "nemotron_h",
        "hidden_size": 2688,
        "num_hidden_layers": 52,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "intermediate_size": 1856,
        "n_routed_experts": 128,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 1856,
        "moe_shared_expert_intermediate_size": 3712,
        "vocab_size": 131072,
        "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "ssm_state_size": 128,
        "n_groups": 8,
        "expand": 2,
        "conv_kernel": 4,
        "norm_eps": 1e-5,
        "rope_theta": 10000,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": true
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "nemotron_h");
    assert_eq!(cfg.hidden_size, 2688);
    assert_eq!(cfg.num_hidden_layers, 52);
    assert_eq!(cfg.num_experts, 128);
    assert_eq!(cfg.num_experts_per_tok, 6);
    assert_eq!(cfg.shared_expert_intermediate_size, 3712);
    assert_eq!(cfg.rms_norm_eps, 1e-5);
    assert_eq!(cfg.linear_conv_kernel_dim, 4);
    assert_eq!(cfg.mamba2_d_inner(), 4096); // 64*64, NOT expand*hidden
    // Pattern: 23 M + 23 E + 6 * = 52
    assert_eq!(cfg.layer_types.len(), 52);
    assert_eq!(cfg.num_ssm_layers(), 23);
    assert_eq!(cfg.num_moe_layers(), 23);
    assert_eq!(cfg.num_attention_layers(), 6);
    assert_eq!(cfg.layer_type(0), LayerType::LinearAttention); // M
    assert_eq!(cfg.layer_type(1), LayerType::Moe); // E
    assert_eq!(cfg.layer_type(5), LayerType::FullAttention); // *
    assert_eq!(cfg.gqa_ratio(), 16); // 32/2
    assert_eq!(cfg.rotary_dim(), 128); // partial_rotary_factor=1.0
    assert_eq!(cfg.routed_scaling_factor, 2.5);
}

#[test]
fn test_parse_nemotron_h_puzzle_config() {
    // Minimal Puzzle-shaped schedule: 4 layers with heterogeneous MoE dims.
    // Full checkpoint has 88 layers; this covers dispatch + per-layer lookup.
    let json = r#"{
        "model_type": "nemotron_h_puzzle",
        "architectures": ["NemotronHPuzzleForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": null,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "intermediate_size": 21504,
        "n_routed_experts": 512,
        "n_shared_experts": 1,
        "moe_latent_size": 1024,
        "moe_shared_expert_intermediate_size": 5376,
        "vocab_size": 131072,
        "layers_block_type": ["mamba", "moe", "attention", "moe"],
        "block_configs": [
            {"block_type": "mamba"},
            {"block_type": "moe", "moe_intermediate_size": 1280, "num_experts_per_tok": 4},
            {"block_type": "attention"},
            {"block_type": "moe", "moe_intermediate_size": 2688, "num_experts_per_tok": 22}
        ],
        "mamba_num_heads": 128,
        "mamba_head_dim": 64,
        "ssm_state_size": 96,
        "n_groups": 8,
        "expand": 2,
        "conv_kernel": 4,
        "norm_eps": 1e-5,
        "routed_scaling_factor": 5.0,
        "norm_topk_prob": true
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "nemotron_h_puzzle");
    assert_eq!(cfg.num_hidden_layers, 4);
    assert_eq!(cfg.num_experts, 512);
    assert_eq!(cfg.moe_latent_size, 1024);
    assert_eq!(cfg.shared_expert_intermediate_size, 5376);
    assert_eq!(cfg.layer_types.len(), 4);
    assert_eq!(cfg.layer_type(0), LayerType::LinearAttention);
    assert_eq!(cfg.layer_type(1), LayerType::Moe);
    assert_eq!(cfg.layer_type(2), LayerType::FullAttention);
    assert_eq!(cfg.layer_type(3), LayerType::Moe);
    assert_eq!(cfg.num_moe_layers(), 2);
    // Per-layer schedule
    assert_eq!(cfg.moe_intermediate_size_for(1), 1280);
    assert_eq!(cfg.num_experts_per_tok_for(1), 4);
    assert_eq!(cfg.moe_intermediate_size_for(3), 2688);
    assert_eq!(cfg.num_experts_per_tok_for(3), 22);
    // Non-MoE layers fall back to scalar max
    assert_eq!(cfg.moe_intermediate_size, 2688);
    assert_eq!(cfg.num_experts_per_tok, 22);
    assert_eq!(cfg.max_moe_intermediate_size(), 2688);
    assert_eq!(cfg.weight_prefix, "backbone");
}

#[test]
fn test_expert_parallelism_range() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    // Single GPU: all experts local
    assert_eq!(cfg.local_expert_range(), (0, 512));
    assert!(cfg.is_local_expert(0));
    assert!(cfg.is_local_expert(511));

    // EP=2, rank 0: experts 0..256
    cfg.ep_rank = 0;
    cfg.ep_world_size = 2;
    assert_eq!(cfg.local_expert_range(), (0, 256));
    assert!(cfg.is_local_expert(0));
    assert!(cfg.is_local_expert(255));
    assert!(!cfg.is_local_expert(256));
    assert!(!cfg.is_local_expert(511));

    // EP=2, rank 1: experts 256..512
    cfg.ep_rank = 1;
    assert_eq!(cfg.local_expert_range(), (256, 512));
    assert!(!cfg.is_local_expert(0));
    assert!(!cfg.is_local_expert(255));
    assert!(cfg.is_local_expert(256));
    assert!(cfg.is_local_expert(511));
}

#[test]
fn test_tensor_parallelism_range() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();

    // Single rank: full range, full dim.
    assert_eq!(cfg.tp_shard_range(2048), (0, 2048));
    assert_eq!(cfg.tp_shard_dim(2048), 2048);

    // TP=2, rank 0: lower half.
    cfg.tp_world_size = 2;
    cfg.tp_rank = 0;
    assert_eq!(cfg.tp_shard_range(2048), (0, 1024));
    assert_eq!(cfg.tp_shard_dim(2048), 1024);

    // TP=2, rank 1: upper half.
    cfg.tp_rank = 1;
    assert_eq!(cfg.tp_shard_range(2048), (1024, 2048));
    assert_eq!(cfg.tp_shard_dim(2048), 1024);

    // TP=4, rank 2: third quarter.
    cfg.tp_world_size = 4;
    cfg.tp_rank = 2;
    assert_eq!(cfg.tp_shard_range(2048), (1024, 1536));
}

#[test]
fn test_parse_gemma4_config() {
    let json = r#"{
        "model_type": "gemma4",
        "tie_word_embeddings": true,
        "final_logit_softcapping": 30.0,
        "text_config": {
            "hidden_size": 5376,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 16,
            "head_dim": 256,
            "intermediate_size": 21504,
            "vocab_size": 262144,
            "hidden_activation": "gelu_pytorch_tanh",
            "sliding_window": 1024,
            "attention_pattern": [
                "sliding_attention", "sliding_attention",
                "full_attention", "sliding_attention"
            ],
            "full_attention_config": {
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25
            },
            "sliding_attention_config": {
                "rope_theta": 10000.0
            },
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144
        }
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "gemma4");
    assert_eq!(cfg.hidden_size, 5376);
    assert_eq!(cfg.num_hidden_layers, 4);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 16);
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.intermediate_size, 21504);
    assert_eq!(cfg.vocab_size, 262144);
    assert_eq!(cfg.rms_norm_eps, 1e-6);
    assert_eq!(cfg.max_position_embeddings, 262144);
    assert_eq!(cfg.rope_theta, 10000.0); // sliding theta
    assert_eq!(cfg.partial_rotary_factor, 0.25);
    assert!(cfg.tie_word_embeddings);
    assert!(!cfg.attn_gated);
    assert!(cfg.nested_config);
    // All 4 layers are FullAttention (no SSM)
    assert_eq!(cfg.layer_types.len(), 4);
    assert_eq!(cfg.num_attention_layers(), 4);
    assert_eq!(cfg.num_ssm_layers(), 0);
    // No MoE
    assert_eq!(cfg.num_experts, 0);
    // No MTP
    assert_eq!(cfg.mtp_num_hidden_layers, 0);
    // No SSM fields
    assert_eq!(cfg.linear_num_key_heads, 0);
    // GQA ratio
    assert_eq!(cfg.gqa_ratio(), 2); // 32/16
    // Rotary dim
    assert_eq!(cfg.rotary_dim(), 64); // 0.25 * 256
}

#[test]
fn test_parse_deepseek_v4_config() {
    let json = r#"{
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "qk_rope_head_dim": 64,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 2048,
        "norm_topk_prob": true,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "routed_scaling_factor": 1.5,
        "sliding_window": 128,
        "max_position_embeddings": 1048576,
        "rope_theta": 10000,
        "rms_norm_eps": 1e-06,
        "vocab_size": 129280,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "tie_word_embeddings": false,
        "num_nextn_predict_layers": 1,
        "rope_scaling": {
            "type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32,
            "beta_slow": 1
        },
        "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0],
        "num_hash_layers": 3
    }"#;
    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "deepseek_v4");
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.num_hidden_layers, 43);
    assert_eq!(cfg.num_attention_heads, 64);
    assert_eq!(cfg.num_key_value_heads, 1);
    assert_eq!(cfg.head_dim, 512);
    assert_eq!(cfg.q_lora_rank, 1024);
    assert_eq!(cfg.o_lora_rank, 1024);
    assert_eq!(cfg.qk_rope_head_dim, 64);
    assert_eq!(cfg.kv_lora_rank, 512); // fallback default
    assert_eq!(cfg.qk_nope_head_dim, 448); // head_dim - qk_rope_head_dim
    assert_eq!(cfg.v_head_dim, 512); // fallback to head_dim
    assert_eq!(cfg.num_experts, 256);
    assert_eq!(cfg.num_experts_per_tok, 6);
    assert_eq!(cfg.moe_intermediate_size, 2048);
    assert_eq!(cfg.shared_expert_intermediate_size, 2048);
    assert!(cfg.norm_topk_prob);
    assert_eq!(cfg.scoring_func, "sqrtsoftplus"); // preserved, no fallback
    assert!(cfg.use_routing_bias);
    assert_eq!(cfg.num_mtp_modules, 1);
    assert_eq!(cfg.mtp_transformer_layers, 1);
    assert_eq!(cfg.sliding_window, 128);
    assert_eq!(cfg.max_position_embeddings, 1048576);
    assert_eq!(cfg.rope_theta, 10000.0);
    assert_eq!(cfg.rms_norm_eps, 1e-6);
    assert_eq!(cfg.vocab_size, 129280);
    assert_eq!(cfg.bos_token_id, 0);
    assert_eq!(cfg.eos_token_id, 1);
    assert!(!cfg.tie_word_embeddings);
    assert!(!cfg.attn_gated);
    assert_eq!(cfg.weight_prefix, "model");
    // DeepSeek-V4 ships 44 compress_ratios for 43 layers — the last
    // trailing value is an artifact, not an error.
    assert_eq!(cfg.compress_ratios.len(), 44);
    assert_eq!(cfg.num_hash_layers, 3);
    // Fallback: all layers treated as FullAttention
    assert_eq!(cfg.num_attention_layers(), 43);
    assert_eq!(cfg.num_ssm_layers(), 0);
    // Capabilities
    let caps = cfg.capabilities();
    assert!(caps.has_attention_layers);
    assert!(caps.has_moe_layers);
    assert!(!caps.has_ssm_layers);
    assert!(caps.has_mtp);
    assert_eq!(caps.attention_type, crate::capabilities::AttentionType::Mla);
}

// Shared fixture: the Holo-3.1-35B config, which is byte-for-byte the
// Qwen3.6-35B-A3B config MINUS the MTP head (Hcompany strips it). The
// genuine Qwen3.6 release is this fixture PLUS mtp_num_hidden_layers=1.
const HOLO31_VLM_CONFIG: &str = r#"{
        "model_type": "qwen3_5_moe",
        "image_token_id": 248056,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "vocab_size": 248320,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ],
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "rope_theta": 10000000,
                "rope_type": "default"
            }
        },
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "out_hidden_size": 2048,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
        }
    }"#;

#[test]
fn test_parse_holo31_vlm_config() {
    let cfg = parse_config(HOLO31_VLM_CONFIG).unwrap();
    assert_eq!(cfg.model_type, "holo3_1_moe");
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_experts, 256);
    assert_eq!(cfg.num_attention_layers(), 10);
    assert_eq!(cfg.num_ssm_layers(), 30);
    assert_eq!(cfg.mrope_section, [11, 11, 10]);
    assert!(cfg.mrope_interleaved);

    let vision = cfg.vision.expect("Holo3.1 must parse vision_config");
    assert_eq!(vision.depth, 27);
    assert_eq!(vision.hidden_size, 1152);
    assert_eq!(vision.out_hidden_size, 2048);
    assert!(vision.deepstack_visual_indexes.is_empty());
    assert_eq!(vision.image_pad_token_id, 248056);
}

// Regression: the FLAGSHIP Qwen/Qwen3.6-35B-A3B-FP8 checkpoint carries the
// SAME vision tower + image_token_id 248056 as Holo-3.1 but ships an MTP
// head (mtp_num_hidden_layers=1). It must stay qwen3_6_moe — the Holo
// discriminator misclassifying it broke kernel-target resolution
// (2026-07-02, webserver_ok flagship gate).
#[test]
fn test_qwen36_35b_with_mtp_is_not_holo() {
    let json = HOLO31_VLM_CONFIG.replace(
        "\"model_type\": \"qwen3_5_moe_text\",",
        "\"model_type\": \"qwen3_5_moe_text\",\n            \"mtp_num_hidden_layers\": 1,",
    );
    let cfg = parse_config(&json).unwrap();
    assert_eq!(cfg.model_type, "qwen3_6_moe");
    assert_eq!(cfg.mtp_num_hidden_layers, 1);
    // Vision tower still parses — the flagship ships a ViT even though
    // text-only serving keeps H/W position IDs at zero.
    assert!(cfg.vision.is_some());
}

#[test]
fn test_parse_nllb_m2m100_config() {
    let json = r#"{
        "activation_function": "relu",
        "architectures": ["M2M100ForConditionalGeneration"],
        "bos_token_id": 0,
        "d_model": 2048,
        "decoder_attention_heads": 16,
        "decoder_ffn_dim": 8192,
        "decoder_layers": 24,
        "encoder_attention_heads": 16,
        "encoder_ffn_dim": 8192,
        "encoder_layers": 24,
        "eos_token_id": 2,
        "is_encoder_decoder": true,
        "max_position_embeddings": 1024,
        "model_type": "m2m_100",
        "num_hidden_layers": 24,
        "pad_token_id": 1,
        "scale_embedding": true,
        "use_cache": true,
        "vocab_size": 256206
    }"#;

    let cfg = parse_config(json).unwrap();
    assert_eq!(cfg.model_type, "m2m_100");
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_hidden_layers, 24);
    assert_eq!(cfg.intermediate_size, 8192);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_key_value_heads, 16);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.max_position_embeddings, 1024);
    assert_eq!(cfg.vocab_size, 256206);
    assert_eq!(cfg.weight_prefix, "model.decoder");
    assert!(!cfg.attn_gated);
}

#[test]
fn test_parse_nllb_rejects_missing_required_dimension() {
    let json = r#"{
        "bos_token_id": 0,
        "d_model": 2048,
        "decoder_ffn_dim": 8192,
        "decoder_layers": 24,
        "eos_token_id": 2,
        "max_position_embeddings": 1024,
        "model_type": "nllb",
        "vocab_size": 256206
    }"#;

    let err = parse_config(json).unwrap_err().to_string();
    assert!(
        err.contains("nllb config missing required field `decoder_attention_heads`"),
        "{err}"
    );
}
