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
            "modules_to_save",
            serde_json::json!(["lm_head"]),
            "REJECT(modules_to_save)",
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
