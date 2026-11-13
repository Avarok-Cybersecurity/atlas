// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: tests of the Qwen3.5 / 3.6 / 3.8 GDN-hybrid GGUF config (`qwen35`, `qwen35moe`).

use super::*;
use crate::config::LayerType;

// ── Qwen3.5 / 3.6 / 3.8 GDN hybrids (`qwen35`, `qwen35moe`) ──

/// The metadata of the real `Qwen3.8-27B-UD-IQ4_XS.gguf` (read from its header,
/// 2026-11-13): 65 blocks = 64 trunk layers + 1 MTP block, no `vocab_size` key.
fn qwen35_27b_meta() -> Meta {
    Meta::default()
        .s("general.architecture", "qwen35")
        .u("qwen35.block_count", 65)
        .u("qwen35.context_length", 262144)
        .u("qwen35.embedding_length", 5120)
        .u("qwen35.feed_forward_length", 17408)
        .u("qwen35.attention.head_count", 24)
        .u("qwen35.attention.head_count_kv", 4)
        .u("qwen35.attention.key_length", 256)
        .u("qwen35.attention.value_length", 256)
        .f(
            "qwen35.attention.layer_norm_rms_epsilon",
            9.999999974752427e-07,
        )
        .f("qwen35.rope.freq_base", 10_000_000.0)
        .u("qwen35.rope.dimension_count", 64)
        .ua("qwen35.rope.dimension_sections", &[11, 11, 10, 0])
        .u("qwen35.nextn_predict_layers", 1)
        .u("qwen35.full_attention_interval", 4)
        .u("qwen35.ssm.conv_kernel", 4)
        .u("qwen35.ssm.state_size", 128)
        .u("qwen35.ssm.group_count", 16)
        .u("qwen35.ssm.time_step_rank", 48)
        .u("qwen35.ssm.inner_size", 6144)
        .u("tokenizer.ggml.eos_token_id", 248046)
        .u("tokenizer.ggml.bos_token_id", 248044)
}

fn qwen35_inputs(m: &Meta) -> GgufConfigInputs<'_> {
    GgufConfigInputs {
        meta: m,
        token_embd_vocab: Some(248320),
        has_output_weight: true,
    }
}

#[test]
fn qwen35_27b_maps_to_the_dense_gdn_hybrid() {
    let m = qwen35_27b_meta();
    let c = config_from_gguf(&qwen35_inputs(&m)).unwrap();
    assert_eq!(c.model_type, "qwen3_5");
    assert!(c.is_qwen35_dense());
    assert_eq!(c.num_experts, 0);
    assert!(c.attn_gated);
    // 65 blocks, one of them the MTP head.
    assert_eq!(c.num_hidden_layers, 64);
    assert_eq!(c.mtp_num_hidden_layers, 1);
    assert_eq!(c.hidden_size, 5120);
    assert_eq!(c.intermediate_size, 17408);
    assert_eq!(c.num_attention_heads, 24);
    assert_eq!(c.num_key_value_heads, 4);
    assert_eq!(c.head_dim, 256);
    assert_eq!(c.vocab_size, 248320);
    assert_eq!(c.max_position_embeddings, 262144);
    assert!(!c.tie_word_embeddings);
    assert_eq!(c.weight_prefix, "model");
    assert_eq!(c.eos_token_id, 248046);
    assert_eq!(c.bos_token_id, 248044);
    assert!((c.rope_theta - 10_000_000.0).abs() < 1e-3);
    // GDN geometry: 16 key heads x 128, 48 value heads x 128 (6144 / 48), conv 4.
    assert_eq!(c.linear_num_key_heads, 16);
    assert_eq!(c.linear_key_head_dim, 128);
    assert_eq!(c.linear_num_value_heads, 48);
    assert_eq!(c.linear_value_head_dim, 128);
    assert_eq!(c.linear_conv_kernel_dim, 4);
    // Partial RoPE: 64 of 256 dims rotate, MRoPE [11, 11, 10] interleaved.
    assert!((c.partial_rotary_factor - 0.25).abs() < 1e-12);
    assert_eq!(c.mrope_section, [11, 11, 10]);
    assert!(c.mrope_interleaved);
}

#[test]
fn qwen35_layer_types_put_full_attention_every_fourth_layer() {
    let m = qwen35_27b_meta();
    let c = config_from_gguf(&qwen35_inputs(&m)).unwrap();
    assert_eq!(c.full_attention_interval, 4);
    assert_eq!(c.layer_types.len(), 64);
    let full: Vec<usize> = c
        .layer_types
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == LayerType::FullAttention)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(full, (0..16).map(|n| 4 * n + 3).collect::<Vec<_>>());
    assert_eq!(
        c.layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count(),
        48
    );
    // The kernel target `qwen3.8-27b` declares exactly this split (16 full, 48 GDN).
    let caps = c.capabilities();
    assert!(caps.has_ssm_layers && caps.has_attention_layers && caps.has_mtp);
    assert!(!caps.has_moe_layers);
}

/// The GGUF-derived config equals the one the safetensors `config.json` path
/// builds for the same model (Qwen3.5-4B), field for field on everything the
/// hybrid loader reads.
#[test]
fn qwen35_gguf_config_matches_the_safetensors_config_json() {
    let layer_types: Vec<&str> = (0..32)
        .map(|i| {
            if (i + 1usize).is_multiple_of(4) {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    let json = serde_json::json!({
        "model_type": "qwen3_5",
        "tie_word_embeddings": true,
        "text_config": {
            "attn_output_gate": true,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "layer_types": layer_types,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "num_attention_heads": 16,
            "num_hidden_layers": 32,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
            "vocab_size": 248320,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25
            }
        }
    })
    .to_string();
    let st = crate::config::parse_config(&json).unwrap();

    let m = Meta::default()
        .s("general.architecture", "qwen35")
        .u("qwen35.block_count", 33)
        .u("qwen35.context_length", 262144)
        .u("qwen35.embedding_length", 2560)
        .u("qwen35.feed_forward_length", 9216)
        .u("qwen35.attention.head_count", 16)
        .u("qwen35.attention.head_count_kv", 4)
        .u("qwen35.attention.key_length", 256)
        .f("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
        .f("qwen35.rope.freq_base", 10_000_000.0)
        .u("qwen35.rope.dimension_count", 64)
        .ua("qwen35.rope.dimension_sections", &[11, 11, 10, 0])
        .u("qwen35.nextn_predict_layers", 1)
        .u("qwen35.full_attention_interval", 4)
        .u("qwen35.ssm.conv_kernel", 4)
        .u("qwen35.ssm.state_size", 128)
        .u("qwen35.ssm.group_count", 16)
        .u("qwen35.ssm.time_step_rank", 32)
        .u("qwen35.ssm.inner_size", 4096)
        .u("tokenizer.ggml.eos_token_id", 248044);
    let gg = config_from_gguf(&GgufConfigInputs {
        meta: &m,
        token_embd_vocab: Some(248320),
        has_output_weight: false,
    })
    .unwrap();

    assert_eq!(gg.model_type, st.model_type);
    assert_eq!(gg.attn_gated, st.attn_gated);
    assert_eq!(gg.num_experts, st.num_experts);
    assert_eq!(gg.hidden_size, st.hidden_size);
    assert_eq!(gg.num_hidden_layers, st.num_hidden_layers);
    assert_eq!(gg.intermediate_size, st.intermediate_size);
    assert_eq!(gg.vocab_size, st.vocab_size);
    assert_eq!(gg.num_attention_heads, st.num_attention_heads);
    assert_eq!(gg.num_key_value_heads, st.num_key_value_heads);
    assert_eq!(gg.head_dim, st.head_dim);
    assert_eq!(gg.max_position_embeddings, st.max_position_embeddings);
    assert!((gg.partial_rotary_factor - st.partial_rotary_factor).abs() < 1e-12);
    assert!((gg.rope_theta - st.rope_theta).abs() < 1e-3);
    assert!((gg.rms_norm_eps - st.rms_norm_eps).abs() < 1e-12);
    assert_eq!(gg.mrope_section, st.mrope_section);
    assert_eq!(gg.mrope_interleaved, st.mrope_interleaved);
    assert_eq!(gg.layer_types, st.layer_types);
    assert_eq!(gg.full_attention_interval, st.full_attention_interval);
    assert_eq!(gg.linear_num_key_heads, st.linear_num_key_heads);
    assert_eq!(gg.linear_key_head_dim, st.linear_key_head_dim);
    assert_eq!(gg.linear_num_value_heads, st.linear_num_value_heads);
    assert_eq!(gg.linear_value_head_dim, st.linear_value_head_dim);
    assert_eq!(gg.linear_conv_kernel_dim, st.linear_conv_kernel_dim);
    assert_eq!(gg.mtp_num_hidden_layers, st.mtp_num_hidden_layers);
    assert_eq!(gg.tie_word_embeddings, st.tie_word_embeddings);
    assert_eq!(gg.eos_token_id, st.eos_token_id);
}

#[test]
fn qwen35_without_an_mtp_head_keeps_every_block_as_trunk() {
    let mut m = qwen35_27b_meta().u("qwen35.block_count", 64);
    m.u.remove("qwen35.nextn_predict_layers");
    let c = config_from_gguf(&qwen35_inputs(&m)).unwrap();
    assert_eq!(c.num_hidden_layers, 64);
    assert_eq!(c.mtp_num_hidden_layers, 0);
    assert!(!c.capabilities().has_mtp);
}

#[test]
fn qwen35_requires_each_hybrid_key() {
    for suffix in [
        "full_attention_interval",
        "ssm.conv_kernel",
        "ssm.state_size",
        "ssm.group_count",
        "ssm.time_step_rank",
        "ssm.inner_size",
        "rope.dimension_count",
        "context_length",
    ] {
        let key = format!("qwen35.{suffix}");
        let mut m = qwen35_27b_meta();
        m.u.remove(&key);
        let err = config_from_gguf(&qwen35_inputs(&m))
            .unwrap_err()
            .to_string();
        assert!(err.contains(&key), "missing {key}, unexpected: {err}");
    }
}

#[test]
fn qwen35_rejects_inconsistent_hybrid_geometry() {
    for (suffix, value) in [
        ("full_attention_interval", 0),
        ("full_attention_interval", 65),
        ("ssm.group_count", 0),
        ("ssm.group_count", 5),
        ("ssm.time_step_rank", 0),
        ("ssm.inner_size", 6145),
        ("ssm.state_size", 0),
        ("ssm.conv_kernel", 0),
        ("rope.dimension_count", 0),
        ("rope.dimension_count", 512),
        // the MRoPE sections sum to 32, so a 128-wide rotary dimension disagrees
        ("rope.dimension_count", 128),
        ("attention.value_length", 128),
        ("nextn_predict_layers", 65),
    ] {
        let key = format!("qwen35.{suffix}");
        let m = qwen35_27b_meta().u(&key, value);
        let err = config_from_gguf(&qwen35_inputs(&m))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(suffix),
            "invalid {key}={value} failed for: {err}"
        );
    }
}

#[test]
fn qwen35_rejects_unsupported_mrope_sections() {
    for sections in [&[11u64, 11][..], &[11, 11, 10, 3], &[11, 11, 10, 0, 0]] {
        let m = qwen35_27b_meta().ua("qwen35.rope.dimension_sections", sections);
        let err = config_from_gguf(&qwen35_inputs(&m))
            .unwrap_err()
            .to_string();
        assert!(err.contains("dimension_sections"), "{sections:?}: {err}");
    }
}

#[test]
fn qwen35_without_mrope_sections_is_scalar_rope() {
    let mut m = qwen35_27b_meta();
    m.ua.remove("qwen35.rope.dimension_sections");
    let c = config_from_gguf(&qwen35_inputs(&m)).unwrap();
    assert_eq!(c.mrope_section, [0, 0, 0]);
    assert!(!c.mrope_interleaved);
    assert!((c.partial_rotary_factor - 0.25).abs() < 1e-12);
}

fn qwen35_moe_meta() -> Meta {
    // Qwen3.5/3.6-35B-A3B shaped: 40 blocks, 256 experts of width 512, one shared expert.
    Meta::default()
        .s("general.architecture", "qwen35moe")
        .u("qwen35moe.block_count", 40)
        .u("qwen35moe.context_length", 262144)
        .u("qwen35moe.embedding_length", 2048)
        .u("qwen35moe.attention.head_count", 16)
        .u("qwen35moe.attention.head_count_kv", 2)
        .u("qwen35moe.attention.key_length", 256)
        .f("qwen35moe.rope.freq_base", 10_000_000.0)
        .u("qwen35moe.rope.dimension_count", 64)
        .ua("qwen35moe.rope.dimension_sections", &[11, 11, 10, 0])
        .u("qwen35moe.full_attention_interval", 4)
        .u("qwen35moe.ssm.conv_kernel", 4)
        .u("qwen35moe.ssm.state_size", 128)
        .u("qwen35moe.ssm.group_count", 16)
        .u("qwen35moe.ssm.time_step_rank", 32)
        .u("qwen35moe.ssm.inner_size", 4096)
        .u("qwen35moe.expert_count", 256)
        .u("qwen35moe.expert_used_count", 8)
        .u("qwen35moe.expert_feed_forward_length", 512)
        .u("qwen35moe.expert_shared_feed_forward_length", 512)
}

fn qwen35_moe_inputs(m: &Meta) -> GgufConfigInputs<'_> {
    GgufConfigInputs {
        meta: m,
        token_embd_vocab: Some(248320),
        has_output_weight: true,
    }
}

#[test]
fn qwen35_moe_maps_to_the_qwen36_moe_target() {
    let m = qwen35_moe_meta();
    let c = config_from_gguf(&qwen35_moe_inputs(&m)).unwrap();
    // MRoPE-interleaved MoE takes the Qwen3.6 name, as the safetensors path does.
    assert_eq!(c.model_type, "qwen3_6_moe");
    assert!(c.attn_gated);
    assert_eq!(c.num_hidden_layers, 40);
    assert_eq!(c.mtp_num_hidden_layers, 0);
    assert_eq!(c.num_experts, 256);
    assert_eq!(c.num_experts_per_tok, 8);
    assert_eq!(c.moe_intermediate_size, 512);
    assert_eq!(c.shared_expert_intermediate_size, 512);
    assert!(c.norm_topk_prob);
    assert_eq!(c.intermediate_size, 0);
    assert_eq!(c.layer_types.len(), 40);
    assert_eq!(c.linear_num_value_heads, 32);
    assert!(c.capabilities().has_moe_layers);

    // Without MRoPE sections the MoE keeps the Qwen3.5 name.
    let mut m = qwen35_moe_meta();
    m.ua.remove("qwen35moe.rope.dimension_sections");
    let c = config_from_gguf(&qwen35_moe_inputs(&m)).unwrap();
    assert_eq!(c.model_type, "qwen3_5_moe");
    assert!(c.is_qwen35());
}

#[test]
fn qwen35_moe_requires_each_expert_key() {
    for suffix in [
        "expert_count",
        "expert_used_count",
        "expert_feed_forward_length",
        "expert_shared_feed_forward_length",
    ] {
        let key = format!("qwen35moe.{suffix}");
        let mut m = qwen35_moe_meta();
        m.u.remove(&key);
        let err = config_from_gguf(&qwen35_moe_inputs(&m))
            .unwrap_err()
            .to_string();
        assert!(err.contains(&key), "missing {key} failed for: {err}");
    }
}
