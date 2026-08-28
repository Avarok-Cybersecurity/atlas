// SPDX-License-Identifier: AGPL-3.0-only

//! LongCat-Flash(-Lite) n-gram configuration parser.
//!
//! `meituan-longcat/LongCat-Flash-Lite` (`LongcatFlashNgramForCausalLM`) is
//! the shipped reference for the n-gram-embedding model family that
//! Qwen3.8-Flash-Next announces ("+51B N-gram embeddings", arxiv
//! 2601.21204). Its config uses several non-HF-standard key names, mapped
//! here onto `ModelConfig`'s existing fields:
//!
//!   num_layers            → num_hidden_layers = 2 * num_layers (the HF
//!                            modeling file makes the same 2x expansion: each
//!                            checkpoint layer is a dual-sublayer "shortcut"
//!                            block = 2 MLA attention + 2 dense MLP + one
//!                            shortcut MoE. Atlas serves each SUBLAYER as one
//!                            engine layer, so the engine layer count, the KV
//!                            sizing and `layer_types` all use 2x; the loader
//!                            iterates checkpoint layers `num_hidden_layers/2`)
//!   n_routed_experts      → num_experts
//!   moe_topk              → num_experts_per_tok
//!   expert_ffn_hidden_size→ moe_intermediate_size
//!   ffn_hidden_size       → intermediate_size
//!
//! MLA fields (kv_lora_rank / q_lora_rank / qk_{nope,rope}_head_dim /
//! v_head_dim) parse through the existing DeepSeek-lineage fields verbatim.
//! The n-gram trio (ngram_vocab_size_ratio / emb_neighbor_num /
//! emb_split_num) parses directly. `zero_expert_num` (zero-computation
//! identity experts) is recorded via the raw config for the backbone port —
//! there is no ModelConfig field for it yet.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::super::{ModelConfig, finalize_config};

pub(crate) fn parse_longcat_ngram(raw: &Value) -> Result<ModelConfig> {
    let mut normalized = raw.clone();
    let object = normalized
        .as_object_mut()
        .context("longcat config.json must be an object")?;

    // Key renames onto HF-standard names ModelConfig already derives.
    for (from, to) in [
        ("num_layers", "num_hidden_layers"),
        ("n_routed_experts", "num_experts"),
        ("moe_topk", "num_experts_per_tok"),
        ("expert_ffn_hidden_size", "moe_intermediate_size"),
        ("ffn_hidden_size", "intermediate_size"),
    ] {
        if let Some(v) = object.get(from).cloned()
            && !object.contains_key(to)
        {
            object.insert(to.into(), v);
        }
    }

    // The HF modeling file's own quirk, mirrored: each checkpoint layer is
    // TWO engine sublayers (2x MLA + 2x dense MLP + shortcut MoE), so every
    // engine-facing layer count is 2x the checkpoint `num_layers`.
    if let Some(n) = object.get("num_hidden_layers").and_then(Value::as_u64) {
        object.insert("num_hidden_layers".into(), Value::from(n * 2));
    }

    // eos_token_id may be an array (like laguna); take the primary.
    if let Some(eos) = object.get("eos_token_id").cloned() {
        let primary = match &eos {
            Value::Number(n) => n.as_u64(),
            Value::Array(ids) => ids.first().and_then(Value::as_u64),
            _ => None,
        }
        .context("longcat eos_token_id must be an integer or non-empty integer array")?;
        object.insert("eos_token_id".into(), Value::from(primary));
    }

    let mut config: ModelConfig =
        serde_json::from_value(normalized).context("Failed to parse longcat config.json")?;

    ensure!(
        config.hidden_size > 0,
        "longcat hidden_size must be non-zero"
    );
    ensure!(
        config.num_hidden_layers > 0,
        "longcat num_layers must be non-zero"
    );
    ensure!(config.vocab_size > 0, "longcat vocab_size must be non-zero");

    // The n-gram trio: all three present or all absent (a partial set means
    // a key rename we have not learned yet — refuse loudly, this parser's
    // whole job is to be updated the day Qwen3.8-Flash-Next lands).
    let ngram_fields = [
        config.ngram_vocab_size_ratio,
        config.emb_neighbor_num,
        config.emb_split_num,
    ];
    let present = ngram_fields.iter().filter(|&&v| v > 0).count();
    ensure!(
        present == 0 || present == 3,
        "longcat n-gram config is partial (ratio={}, neighbor={}, split={}) — \
         all three of ngram_vocab_size_ratio / emb_neighbor_num / emb_split_num \
         must be present together",
        config.ngram_vocab_size_ratio,
        config.emb_neighbor_num,
        config.emb_split_num,
    );
    if present == 3 {
        ensure!(
            config.emb_neighbor_num >= 2,
            "emb_neighbor_num must be >= 2 (largest n-gram size)"
        );
        let num_tables = config.emb_split_num * (config.emb_neighbor_num - 1);
        ensure!(
            config.hidden_size.is_multiple_of(num_tables),
            "hidden_size {} must divide evenly by the {} n-gram tables \
             (emb_split_num {} x (emb_neighbor_num {} - 1))",
            config.hidden_size,
            num_tables,
            config.emb_split_num,
            config.emb_neighbor_num,
        );
    }

    // MLA attention geometry the shared MLA loader pipeline expects:
    // head_dim = the FULL qk head width (nope + rope), one KV "head" per
    // attention head (MLA has a single shared latent; the per-head fields
    // exist for the wq_b/wkv_b splits).
    if config.head_dim == 0 {
        config.head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim;
    }
    if config.num_key_value_heads == 0 {
        config.num_key_value_heads = config.num_attention_heads;
    }
    // Every sublayer is MLA full attention (the model has no GDN/SSM mix).
    if config.layer_types.is_empty() {
        config.layer_types = vec![super::super::LayerType::FullAttention; config.num_hidden_layers];
    }
    // PLAIN rope, explicitly. LongCat declares no rope scaling (the HF
    // reference takes its `rope_type == "default"` branch: no YaRN, no
    // mscale), but the shared MLA loader ALWAYS builds a YaRN inv_freq table
    // and its `compute_yarn_inv_freq` defaults to factor=128 when a config is
    // silent — which would silently YaRN-scale every rope frequency. At
    // factor 1.0 the interpolation and extrapolation terms are identical, so
    // the table reduces to the plain 1/theta^(2j/dim) rope, and
    // `yarn_rope_mscale` short-circuits to 1.0 (its `factor <= 1.0` gate).
    if config.yarn_factor == 0.0 {
        config.yarn_factor = 1.0;
    }

    // Router: fp32 softmax over num_experts + zero_expert_num logits, top-k
    // SELECTED with e_score_correction_bias but WEIGHTED by the unbiased
    // softmax scores * routed_scaling_factor, never renormalized among the
    // selected k.
    config.scoring_func = "softmax".to_string();
    config.norm_topk_prob = false;

    config.model_type = "longcat_flash_ngram".to_string();
    finalize_config(&mut config, raw)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real LongCat-Flash-Lite config.json subset (values verified
    /// against the HF repo @ main, 2026-08-25).
    fn lite_config() -> Value {
        serde_json::json!({
            "architectures": ["LongcatFlashNgramForCausalLM"],
            "model_type": "longcat_flash_ngram",
            "vocab_size": 131072,
            "hidden_size": 3072,
            "ffn_hidden_size": 6144,
            "expert_ffn_hidden_size": 1024,
            "num_layers": 14,
            "num_attention_heads": 32,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128,
            "n_routed_experts": 256,
            "moe_topk": 12,
            "routed_scaling_factor": 6.0,
            "zero_expert_num": 128,
            "ngram_vocab_size_ratio": 78,
            "emb_neighbor_num": 4,
            "emb_split_num": 4,
            "rope_theta": 5000000.0,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 327680,
            "eos_token_id": 2,
            "torch_dtype": "bfloat16"
        })
    }

    #[test]
    fn parses_lite_config() {
        let c = parse_longcat_ngram(&lite_config()).unwrap();
        // 14 checkpoint layers → 28 engine sublayers (dual-sublayer blocks).
        assert_eq!(c.num_hidden_layers, 28);
        assert_eq!(c.layer_types.len(), 28);
        assert_eq!(c.head_dim, 192);
        assert_eq!(c.num_key_value_heads, 32);
        assert_eq!(c.zero_expert_num, 128);
        assert_eq!(c.scoring_func, "softmax");
        assert!(!c.norm_topk_prob);
        assert_eq!(c.routed_scaling_factor, 6.0);
        // Plain rope: factor 1.0 makes the shared YaRN table reduce to it.
        assert_eq!(c.yarn_factor, 1.0);
        assert_eq!(c.num_experts, 256);
        assert_eq!(c.num_experts_per_tok, 12);
        assert_eq!(c.moe_intermediate_size, 1024);
        assert_eq!(c.intermediate_size, 6144);
        assert_eq!(c.kv_lora_rank, 512);
        assert_eq!(c.q_lora_rank, 1536);
        assert_eq!(c.ngram_vocab_size_ratio, 78);
        assert_eq!(c.emb_neighbor_num, 4);
        assert_eq!(c.emb_split_num, 4);
        // 12 tables at hidden/12 = 256 dims each
        let tables = c.emb_split_num * (c.emb_neighbor_num - 1);
        assert_eq!(tables, 12);
        assert_eq!(c.hidden_size / tables, 256);
    }

    /// The REAL checkpoint declares no `model_type` — only `architectures`
    /// — so dispatch must route it by architecture name or the family is
    /// silently lost to the generic parse.
    #[test]
    fn architectures_only_config_routes_to_longcat() {
        let mut v = lite_config();
        v.as_object_mut().unwrap().remove("model_type");
        let json = serde_json::to_string(&v).unwrap();
        let c = crate::config::parse_config(&json).unwrap();
        assert_eq!(c.model_type, "longcat_flash_ngram");
        assert_eq!(c.num_hidden_layers, 28);
        assert_eq!(c.zero_expert_num, 128);
    }

    #[test]
    fn rejects_partial_ngram_trio() {
        let mut v = lite_config();
        v.as_object_mut().unwrap().remove("emb_split_num");
        assert!(parse_longcat_ngram(&v).is_err());
    }

    #[test]
    fn non_ngram_longcat_parses() {
        let mut v = lite_config();
        for k in [
            "ngram_vocab_size_ratio",
            "emb_neighbor_num",
            "emb_split_num",
        ] {
            v.as_object_mut().unwrap().remove(k);
        }
        let c = parse_longcat_ngram(&v).unwrap();
        assert_eq!(c.ngram_vocab_size_ratio, 0);
    }
}
