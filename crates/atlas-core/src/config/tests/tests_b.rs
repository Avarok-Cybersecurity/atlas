// SPDX-License-Identifier: AGPL-3.0-only

//! Additional config tests split out of `config/tests.rs` for the
//! ≤500 LoC file-size cap.

#![allow(unused_imports)]

use super::super::*;

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
