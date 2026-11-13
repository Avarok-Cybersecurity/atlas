// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use std::collections::HashMap;

/// Minimal in-memory GgufMeta for tests.
#[derive(Default)]
struct Meta {
    u: HashMap<String, u64>,
    f: HashMap<String, f64>,
    s: HashMap<String, String>,
    arr: HashMap<String, usize>,
    ua: HashMap<String, Vec<u64>>,
}
impl Meta {
    fn ua(mut self, k: &str, v: &[u64]) -> Self {
        self.ua.insert(k.into(), v.to_vec());
        self
    }
    fn u(mut self, k: &str, v: u64) -> Self {
        self.u.insert(k.into(), v);
        self
    }
    fn f(mut self, k: &str, v: f64) -> Self {
        self.f.insert(k.into(), v);
        self
    }
    fn s(mut self, k: &str, v: &str) -> Self {
        self.s.insert(k.into(), v.into());
        self
    }
}
impl GgufMeta for Meta {
    fn get_u64(&self, k: &str) -> Option<u64> {
        self.u.get(k).copied()
    }
    fn get_f64(&self, k: &str) -> Option<f64> {
        self.f.get(k).copied()
    }
    fn get_str(&self, k: &str) -> Option<&str> {
        self.s.get(k).map(String::as_str)
    }
    fn get_arr_len(&self, k: &str) -> Option<usize> {
        self.arr.get(k).copied()
    }
    fn get_u64_arr(&self, k: &str) -> Option<Vec<u64>> {
        self.ua.get(k).cloned()
    }
}

fn llama_meta() -> Meta {
    Meta::default()
        .s("general.architecture", "llama")
        .u("llama.embedding_length", 4096)
        .u("llama.block_count", 32)
        .u("llama.feed_forward_length", 11008)
        .u("llama.attention.head_count", 32)
        .u("llama.attention.head_count_kv", 8)
        .u("llama.context_length", 4096)
        .u("llama.vocab_size", 32000)
        .f("llama.attention.layer_norm_rms_epsilon", 1e-6)
        .f("llama.rope.freq_base", 500_000.0)
}

fn gemma2_meta() -> Meta {
    Meta::default()
        .s("general.architecture", "gemma2")
        .u("gemma2.embedding_length", 2304)
        .u("gemma2.block_count", 26)
        .u("gemma2.feed_forward_length", 9216)
        .u("gemma2.attention.head_count", 8)
        .u("gemma2.attention.head_count_kv", 4)
        .u("gemma2.attention.key_length", 256)
        .u("gemma2.context_length", 8192)
        .u("gemma2.vocab_size", 256000)
        .u("gemma2.attention.sliding_window", 4096)
        .f("gemma2.attention.layer_norm_rms_epsilon", 1e-6)
        .f("gemma2.final_logit_softcapping", 30.0)
}

#[test]
fn llama_dense_maps_to_mistral() {
    let m = llama_meta();
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    };
    let c = config_from_gguf(&inp).unwrap();
    assert_eq!(c.model_type, "mistral");
    assert_eq!(c.hidden_size, 4096);
    assert_eq!(c.num_hidden_layers, 32);
    assert_eq!(c.intermediate_size, 11008);
    assert_eq!(c.num_attention_heads, 32);
    assert_eq!(c.num_key_value_heads, 8);
    assert_eq!(c.head_dim, 128); // 4096/32
    assert_eq!(c.vocab_size, 32000);
    assert_eq!(c.max_position_embeddings, 4096);
    assert!((c.rms_norm_eps - 1e-6).abs() < 1e-12);
    assert!((c.rope_theta - 500_000.0).abs() < 1e-3);
    assert!(!c.attn_gated);
    assert!(!c.tie_word_embeddings); // has_output_weight = true
    assert_eq!(c.weight_prefix, "model"); // HF-name prefix the GGUF loader emits
    assert_eq!(c.num_experts, 0);
}

#[test]
fn explicit_key_length_wins() {
    let m = llama_meta().u("llama.attention.key_length", 96);
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    };
    let c = config_from_gguf(&inp).unwrap();
    assert_eq!(c.head_dim, 96);
}

#[test]
fn explicit_key_length_must_be_nonzero() {
    let m = llama_meta().u("llama.attention.key_length", 0);
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    };
    let err = config_from_gguf(&inp).unwrap_err();
    assert!(err.to_string().contains("llama.attention.key_length"));
}

#[test]
fn kv_heads_default_to_mha() {
    let mut m = llama_meta();
    m.u.remove("llama.attention.head_count_kv");
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    };
    let c = config_from_gguf(&inp).unwrap();
    assert_eq!(c.num_key_value_heads, c.num_attention_heads);
}

#[test]
fn explicit_kv_heads_must_form_valid_gqa_groups() {
    let mut accepted = Vec::new();
    for kv_heads in [0, 7, 64] {
        let mut m = llama_meta();
        m.u.insert("llama.attention.head_count_kv".into(), kv_heads);
        let inp = GgufConfigInputs {
            meta: &m,
            token_embd_vocab: None,
            has_output_weight: true,
        };
        match config_from_gguf(&inp) {
            Ok(_) => accepted.push(kv_heads),
            Err(err) => assert!(err.to_string().contains("llama.attention.head_count_kv")),
        }
    }
    assert!(
        accepted.is_empty(),
        "accepted invalid KV head counts: {accepted:?}"
    );
}

#[test]
fn vocab_from_token_embd_rows() {
    let mut m = llama_meta();
    m.u.remove("llama.vocab_size");
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: Some(128256),
        has_output_weight: true,
    };
    let c = config_from_gguf(&inp).unwrap();
    assert_eq!(c.vocab_size, 128256);
}

#[test]
fn vocab_sources_must_be_nonzero_and_consistent() {
    let mut zero_tensor_rows = llama_meta();
    zero_tensor_rows.u.remove("llama.vocab_size");

    let mut zero_token_list = llama_meta();
    zero_token_list.u.remove("llama.vocab_size");
    zero_token_list
        .arr
        .insert("tokenizer.ggml.tokens".into(), 0);

    let cases = [
        (
            "zero metadata vocab",
            llama_meta().u("llama.vocab_size", 0),
            None,
        ),
        ("zero token embedding rows", zero_tensor_rows, Some(0)),
        ("zero tokenizer list", zero_token_list, None),
        ("metadata/tensor mismatch", llama_meta(), Some(128256)),
    ];
    let mut accepted = Vec::new();
    for (name, meta, token_embd_vocab) in cases {
        let inputs = GgufConfigInputs {
            meta: &meta,
            token_embd_vocab,
            has_output_weight: true,
        };
        if config_from_gguf(&inputs).is_ok() {
            accepted.push(name);
        }
    }
    assert!(
        accepted.is_empty(),
        "invalid vocab sources accepted: {accepted:?}"
    );
}

#[test]
fn tied_embeddings_when_no_output_tensor() {
    let m = llama_meta();
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: false,
    };
    let c = config_from_gguf(&inp).unwrap();
    assert!(c.tie_word_embeddings);
}

#[test]
fn qwen3_dense_maps_attention_controls() {
    let m = Meta::default()
        .s("general.architecture", "qwen3")
        .u("qwen3.embedding_length", 2048)
        .u("qwen3.block_count", 28)
        .u("qwen3.feed_forward_length", 6144)
        .u("qwen3.attention.head_count", 16)
        .u("qwen3.attention.head_count_kv", 8)
        .u("qwen3.attention.key_length", 128)
        .u("qwen3.context_length", 40960)
        .u("qwen3.vocab_size", 151936)
        .f("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
        .f("qwen3.rope.freq_base", 1000000.0);
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    };
    let c = config_from_gguf(&inp).unwrap();
    assert_eq!(c.model_type, "qwen3_5");
    assert_eq!(c.num_experts, 0);
    assert!(c.is_qwen35_dense());
    assert!(!c.attn_gated);
    assert_eq!(c.head_dim, 128);
    assert!((c.rms_norm_eps - 1e-6).abs() < 1e-12);
    assert!((c.rope_theta - 1_000_000.0).abs() < 1e-3);
}

#[test]
fn qwen3_moe_populates_expert_fields() {
    let m = Meta::default()
        .s("general.architecture", "qwen3moe")
        .u("qwen3moe.embedding_length", 2048)
        .u("qwen3moe.block_count", 48)
        .u("qwen3moe.feed_forward_length", 6144)
        .u("qwen3moe.attention.head_count", 32)
        .u("qwen3moe.attention.head_count_kv", 4)
        .u("qwen3moe.attention.key_length", 128)
        .u("qwen3moe.context_length", 32768)
        .u("qwen3moe.vocab_size", 151936)
        .u("qwen3moe.expert_count", 128)
        .u("qwen3moe.expert_used_count", 8)
        .u("qwen3moe.expert_feed_forward_length", 768)
        .f("qwen3moe.attention.layer_norm_rms_epsilon", 1e-6)
        .f("qwen3moe.rope.freq_base", 1000000.0);
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    };
    let c = config_from_gguf(&inp).unwrap();
    assert_eq!(c.model_type, "qwen3_5_moe");
    assert_eq!(c.intermediate_size, 6144);
    assert_eq!(c.num_experts, 128);
    assert_eq!(c.num_experts_per_tok, 8);
    assert_eq!(c.moe_intermediate_size, 768);
}

#[test]
fn gemma_sets_embed_scale_and_softcap() {
    let m = gemma2_meta();
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: false,
    };
    let c = config_from_gguf(&inp).unwrap();
    assert_eq!(c.model_type, "gemma4");
    assert!(!c.attn_gated);
    assert_eq!(c.sliding_window, 4096);
    assert!((c.embed_scale - (2304f32).sqrt()).abs() < 1e-3);
    assert!((c.final_logit_softcapping - 30.0).abs() < 1e-3);
}

#[test]
fn gemma_validates_logit_softcap_domain() {
    let mut accepted = Vec::new();
    for value in [-30.0, f64::NAN, f64::INFINITY, f64::MAX] {
        let mut m = gemma2_meta();
        m.f.insert("gemma2.final_logit_softcapping".into(), value);
        let inp = GgufConfigInputs {
            meta: &m,
            token_embd_vocab: None,
            has_output_weight: false,
        };
        match config_from_gguf(&inp) {
            Ok(_) => accepted.push(value),
            Err(err) => assert!(err.to_string().contains("gemma2.final_logit_softcapping")),
        }
    }
    assert!(
        accepted.is_empty(),
        "accepted invalid softcaps: {accepted:?}"
    );

    let mut disabled = gemma2_meta();
    disabled
        .f
        .insert("gemma2.final_logit_softcapping".into(), 0.0);
    let inp = GgufConfigInputs {
        meta: &disabled,
        token_embd_vocab: None,
        has_output_weight: false,
    };
    assert_eq!(config_from_gguf(&inp).unwrap().final_logit_softcapping, 0.0);
}

#[test]
fn missing_architecture_errors() {
    let mut m = llama_meta();
    m.s.remove("general.architecture");
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    };
    let err = config_from_gguf(&inp).unwrap_err().to_string();
    assert!(err.contains("general.architecture"), "unexpected: {err}");
}

#[test]
fn missing_required_dimensions_name_each_key() {
    for key in [
        "embedding_length",
        "block_count",
        "feed_forward_length",
        "attention.head_count",
        "context_length",
    ] {
        let mut m = llama_meta();
        m.u.remove(&format!("llama.{key}"));
        let inp = GgufConfigInputs {
            meta: &m,
            token_embd_vocab: None,
            has_output_weight: true,
        };
        let err = config_from_gguf(&inp).unwrap_err().to_string();
        assert!(err.contains(key), "missing {key}, unexpected: {err}");
    }
}

fn qwen3_moe_meta() -> Meta {
    Meta::default()
        .s("general.architecture", "qwen3moe")
        .u("qwen3moe.embedding_length", 2048)
        .u("qwen3moe.block_count", 48)
        .u("qwen3moe.feed_forward_length", 768)
        .u("qwen3moe.attention.head_count", 32)
        .u("qwen3moe.context_length", 32768)
        .u("qwen3moe.vocab_size", 151936)
        .u("qwen3moe.expert_count", 128)
        .u("qwen3moe.expert_used_count", 8)
        .u("qwen3moe.expert_feed_forward_length", 768)
}

#[test]
fn qwen3_moe_requires_each_expert_dimension() {
    for suffix in [
        "expert_count",
        "expert_used_count",
        "expert_feed_forward_length",
    ] {
        let mut m = qwen3_moe_meta();
        let key = format!("qwen3moe.{suffix}");
        m.u.remove(&key);
        let inp = GgufConfigInputs {
            meta: &m,
            token_embd_vocab: None,
            has_output_weight: true,
        };
        let err = config_from_gguf(&inp).unwrap_err().to_string();
        assert!(err.contains(&key), "missing {key} failed for: {err}");
    }
}

#[test]
fn qwen3_moe_rejects_invalid_expert_geometry() {
    for (suffix, value) in [
        ("expert_count", 0),
        ("expert_used_count", 0),
        ("expert_used_count", 129),
        ("expert_feed_forward_length", 0),
    ] {
        let key = format!("qwen3moe.{suffix}");
        let m = qwen3_moe_meta().u(&key, value);
        let inp = GgufConfigInputs {
            meta: &m,
            token_embd_vocab: None,
            has_output_weight: true,
        };
        let err = config_from_gguf(&inp).unwrap_err().to_string();
        assert!(
            err.contains(&key),
            "invalid {key}={value} failed for: {err}"
        );
    }
}

#[test]
fn mamba_architecture_is_rejected_before_dimension_parsing() {
    let m = Meta::default()
        .s("general.architecture", "mamba")
        .u("mamba.embedding_length", 4096)
        .u("mamba.block_count", 32)
        .u("mamba.feed_forward_length", 11008)
        .u("mamba.attention.head_count", 32)
        .u("mamba.context_length", 4096)
        .u("mamba.vocab_size", 32000);
    let inp = GgufConfigInputs {
        meta: &m,
        token_embd_vocab: None,
        has_output_weight: true,
    };
    let err = config_from_gguf(&inp).unwrap_err().to_string();
    assert!(
        err.contains("GGUF general.architecture 'mamba' has no Atlas model_type mapping"),
        "unexpected: {err}"
    );
    // The message lists the supported arches, the Qwen3.5 hybrids included.
    assert!(err.contains("qwen35, qwen35moe"), "unexpected: {err}");
}

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
