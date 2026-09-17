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
}
impl Meta {
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
}
