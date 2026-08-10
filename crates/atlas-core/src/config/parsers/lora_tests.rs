// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the PEFT `adapter_config.json` parser: happy-path scaling and the
//! named hard-rejections for every unsupported PEFT feature.

use super::*;

fn base_json() -> serde_json::Value {
    serde_json::json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "Hcompany/Holo-3.1-0.8B",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "use_rslora": false,
        "use_dora": false,
        "rank_pattern": {},
        "alpha_pattern": {},
        "modules_to_save": null,
        "layers_to_transform": null,
        "target_modules": ["k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    })
}

#[test]
fn happy_path_scaling_alpha_over_r() {
    let cfg = parse_peft_adapter_config(&base_json().to_string()).unwrap();
    assert_eq!(cfg.r, 16);
    assert_eq!(cfg.lora_alpha, 32.0);
    assert!(!cfg.use_rslora);
    assert_eq!(cfg.scaling(), 2.0);
    assert_eq!(cfg.target_modules.len(), 6);
}

#[test]
fn rslora_scaling_alpha_over_sqrt_r() {
    let mut j = base_json();
    j["use_rslora"] = serde_json::json!(true);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert_eq!(cfg.scaling(), 8.0); // 32 / sqrt(16)
}

#[test]
fn float_alpha_accepted() {
    let mut j = base_json();
    j["lora_alpha"] = serde_json::json!(16.5);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert_eq!(cfg.lora_alpha, 16.5);
}

#[test]
fn layers_to_transform_array_accepted() {
    // The generated Holo fixture carries layers_to_transform=[3,7,...];
    // it must be ACCEPTED (kept for logging), not rejected.
    let mut j = base_json();
    j["layers_to_transform"] = serde_json::json!([3, 7, 11, 15, 19, 23]);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert_eq!(cfg.layers_to_transform, Some(vec![3, 7, 11, 15, 19, 23]));
}

#[test]
fn missing_use_rslora_rejected_named() {
    let mut j = base_json();
    j.as_object_mut().unwrap().remove("use_rslora");
    let err = parse_peft_adapter_config(&j.to_string())
        .unwrap_err()
        .to_string();
    assert!(err.contains("REJECT(use_rslora)"), "{err}");
}

#[test]
fn q_proj_accepted() {
    // q_proj is now a supported target: on attn_output_gate models the raw
    // interleaved [Q|gate] output is the exact width the PEFT lora_B trained
    // against, so the delta folds like k/v/o. Must parse, not reject.
    let mut j = base_json();
    j["target_modules"] = serde_json::json!(["q_proj", "v_proj"]);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert!(cfg.target_modules.iter().any(|m| m == "q_proj"));
}

#[test]
fn gdn_module_rejected_named() {
    for m in [
        "in_proj_qkvz",
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
        "conv1d",
    ] {
        let mut j = base_json();
        j["target_modules"] = serde_json::json!([m]);
        let err = parse_peft_adapter_config(&j.to_string())
            .unwrap_err()
            .to_string();
        assert!(err.contains("REJECT(gdn)"), "{m}: {err}");
    }
}

#[test]
fn router_gate_target_accepted() {
    // Feature-1: the MoE router leaf `gate` (DISTINCT from `gate_proj`) is
    // accepted by the config gate; the runtime key gate + ATLAS_LORA_EXPERTS
    // decide whether it actually loads.
    let mut j = base_json();
    j["target_modules"] = serde_json::json!(["gate"]);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert_eq!(cfg.target_modules, vec!["gate"]);
    // Full-path router entry validates on its leaf too.
    let mut j = base_json();
    j["target_modules"] = serde_json::json!(["model.layers.3.mlp.gate"]);
    assert!(parse_peft_adapter_config(&j.to_string()).is_ok());
}

#[test]
fn all_linear_rejected_named() {
    let mut j = base_json();
    j["target_modules"] = serde_json::json!("all-linear");
    let err = parse_peft_adapter_config(&j.to_string())
        .unwrap_err()
        .to_string();
    assert!(err.contains("REJECT(target_modules)"), "{err}");
}

#[test]
fn dora_bias_rank_pattern_rejected_named() {
    for (key, val, tag) in [
        ("use_dora", serde_json::json!(true), "REJECT(use_dora)"),
        ("bias", serde_json::json!("lora_only"), "REJECT(bias)"),
        (
            "rank_pattern",
            serde_json::json!({"k_proj": 8}),
            "REJECT(rank_pattern)",
        ),
        (
            // `lm_head`/`embed_tokens` now ACCEPT (token overlay); an
            // arbitrary module still rejects.
            "modules_to_save",
            serde_json::json!(["classifier"]),
            "REJECT(modules_to_save)",
        ),
        (
            "target_parameters",
            serde_json::json!(["mlp.experts.gate_up_proj"]),
            "REJECT(target_parameters)",
        ),
        (
            "peft_type",
            serde_json::json!("ADALORA"),
            "REJECT(peft_type)",
        ),
    ] {
        let mut j = base_json();
        j[key] = val;
        let err = parse_peft_adapter_config(&j.to_string())
            .unwrap_err()
            .to_string();
        assert!(err.contains(tag), "{key}: {err}");
    }
}

#[test]
fn full_path_target_validates_on_leaf() {
    let mut j = base_json();
    j["target_modules"] = serde_json::json!(["model.layers.3.self_attn.k_proj"]);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert_eq!(cfg.target_modules, vec!["model.layers.3.self_attn.k_proj"]);
}

#[test]
fn zero_rank_rejected() {
    let mut j = base_json();
    j["r"] = serde_json::json!(0);
    assert!(parse_peft_adapter_config(&j.to_string()).is_err());
}

// ---- Feature 2: token overlay (embed / lm_head / vocab-extension) ----

#[test]
fn modules_to_save_embed_lmhead_accepted() {
    let mut j = base_json();
    j["modules_to_save"] = serde_json::json!(["embed_tokens", "lm_head"]);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert_eq!(cfg.modules_to_save, vec!["embed_tokens", "lm_head"]);
}

#[test]
fn modules_to_save_full_path_validates_on_leaf() {
    let mut j = base_json();
    j["modules_to_save"] = serde_json::json!(["base_model.model.model.embed_tokens"]);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert_eq!(cfg.modules_to_save, vec!["embed_tokens"]);
}

#[test]
fn modules_to_save_unknown_leaf_rejected() {
    let mut j = base_json();
    j["modules_to_save"] = serde_json::json!(["embed_tokens", "score"]);
    let err = parse_peft_adapter_config(&j.to_string())
        .unwrap_err()
        .to_string();
    assert!(err.contains("REJECT(modules_to_save)"), "{err}");
}

#[test]
fn trainable_token_indices_list_form_parsed() {
    let mut j = base_json();
    j["trainable_token_indices"] = serde_json::json!([256205, 42, 42, 7]);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    // deduped, first-occurrence order preserved (delta rows align to it)
    assert_eq!(cfg.trainable_token_indices, vec![256205, 42, 7]);
}

#[test]
fn trainable_token_indices_dict_form_unioned() {
    let mut j = base_json();
    j["trainable_token_indices"] = serde_json::json!({
        "embed_tokens": [256205, 10],
        "lm_head": [10, 99]
    });
    let mut cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    // Order across dict values is object-iteration dependent; compare as a set.
    cfg.trainable_token_indices.sort_unstable();
    assert_eq!(cfg.trainable_token_indices, vec![10, 99, 256205]);
}

#[test]
fn trainable_token_indices_negative_rejected() {
    let mut j = base_json();
    j["trainable_token_indices"] = serde_json::json!([-1]);
    let err = parse_peft_adapter_config(&j.to_string())
        .unwrap_err()
        .to_string();
    assert!(err.contains("REJECT(trainable_token_indices)"), "{err}");
}

#[test]
fn empty_target_modules_with_overlay_accepted() {
    // A pure `trainable_tokens` adapter targets no LoRA module.
    let mut j = base_json();
    j["target_modules"] = serde_json::json!([]);
    j["trainable_token_indices"] = serde_json::json!([256205]);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert!(cfg.target_modules.is_empty());
    assert_eq!(cfg.trainable_token_indices, vec![256205]);
}

#[test]
fn empty_target_modules_without_overlay_rejected() {
    let mut j = base_json();
    j["target_modules"] = serde_json::json!([]);
    let err = parse_peft_adapter_config(&j.to_string())
        .unwrap_err()
        .to_string();
    assert!(err.contains("REJECT(target_modules)"), "{err}");
}

#[test]
fn absent_target_modules_with_modules_to_save_accepted() {
    let mut j = base_json();
    j.as_object_mut().unwrap().remove("target_modules");
    j["modules_to_save"] = serde_json::json!(["lm_head"]);
    let cfg = parse_peft_adapter_config(&j.to_string()).unwrap();
    assert!(cfg.target_modules.is_empty());
    assert_eq!(cfg.modules_to_save, vec!["lm_head"]);
}
