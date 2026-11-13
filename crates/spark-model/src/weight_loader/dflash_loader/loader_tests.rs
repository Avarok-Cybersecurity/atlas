// SPDX-License-Identifier: AGPL-3.0-only

//! Parser tests for the DFlash drafter's `config.json`.
//!
//! Split from `dflash_loader.rs` for the 500-LoC cap, which the file crossed
//! when the DFlash2 sub-config gained its own block-size resolution. Exact
//! piecewise copy — no test changed in the move.

use super::*;

const SHIPPED_CONFIG: &str = r#"{
    "hidden_size": 2048,
    "num_hidden_layers": 8,
    "intermediate_size": 6144,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "vocab_size": 248320,
    "draft_vocab_size": 248320,
    "tie_word_embeddings": false,
    "block_size": 16,
    "rope_theta": 10000000.0,
    "rope_scaling": null,
    "dflash_config": {
        "mask_token_id": 248070,
        "target_layer_ids": [1, 10, 19, 28, 37]
    }
}"#;

#[test]
fn shipped_qwen3_6_fields_reach_the_runtime_config() {
    let config = parse_dflash_config(SHIPPED_CONFIG).expect("parse drafter config");
    assert_eq!(config.num_hidden_layers, 8);
    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.intermediate_size, 6144);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 4);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.vocab_size, 248320);
    assert_eq!(config.draft_vocab_size, Some(248320));
    assert!(!config.tie_word_embeddings);
    assert_eq!(config.block_size, 16);
    assert_eq!(config.rope_theta, 10_000_000.0);
    assert!(config.rope_scaling.is_none());
    let sub = config.dflash_config.expect("dflash_config present");
    assert_eq!(sub.mask_token_id, 248070);
    assert_eq!(sub.target_layer_ids, vec![1, 10, 19, 28, 37]);
}

#[test]
fn omitted_optional_fields_use_runtime_defaults() {
    let config = parse_dflash_config(
        r#"{
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 256
        }"#,
    )
    .unwrap();

    assert_eq!(config.block_size, 16);
    assert_eq!(config.rope_theta, 10_000_000.0);
    assert!(!config.tie_word_embeddings);
    assert!(config.draft_vocab_size.is_none());
    assert!(config.dflash_config.is_none());
    assert!(config.rope_scaling.is_none());
}

#[test]
fn malformed_runtime_field_is_rejected_with_parser_context() {
    let malformed =
        SHIPPED_CONFIG.replace("\"mask_token_id\": 248070", "\"mask_token_id\": \"bad\"");
    let error = parse_dflash_config(&malformed).unwrap_err();
    assert_eq!(error.to_string(), "Parsing DFlash drafter config.json");
    assert!(format!("{error:#}").contains("invalid type: string \"bad\""));
}

/// A DFlash2 checkpoint states its trained block size INSIDE
/// `dflash_config`; the top-level field is absent and serde fills its
/// default of 16. Resolving from the top level alone therefore runs an
/// 8-block drafter at gamma=16 -- 0% accept on every verify step, and
/// drafter pools sized for twice the block. Hermetic: parses a literal,
/// no checkpoint needed.
#[test]
fn effective_block_size_prefers_the_drafters_own_value() {
    let json = r#"{
        "hidden_size": 5120, "num_hidden_layers": 5,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "intermediate_size": 17408, "vocab_size": 248320, "head_dim": 128,
        "dflash_config": {
            "block_size": 8, "mask_token_id": 248070,
            "target_layer_ids": [1, 10, 19, 28, 37]
        }
    }"#;
    let cfg = parse_dflash_config(json).expect("parses");
    assert_eq!(cfg.block_size, 16, "top-level default is still 16");
    assert_eq!(
        cfg.effective_block_size(),
        8,
        "the drafter's own block_size must win over the top-level default"
    );
}

/// A DFlash1 checkpoint states it top-level only: nothing to prefer, so
/// the resolved value is unchanged from before this resolver existed.
#[test]
fn effective_block_size_falls_back_to_the_top_level() {
    let json = r#"{
        "hidden_size": 2048, "num_hidden_layers": 8,
        "num_attention_heads": 32, "num_key_value_heads": 4,
        "intermediate_size": 6144, "vocab_size": 248320, "head_dim": 128,
        "block_size": 16
    }"#;
    let cfg = parse_dflash_config(json).expect("parses");
    assert_eq!(cfg.effective_block_size(), 16);
}

// ── z-lab/Qwen3.5-4B-DFlash on the Ada 4B (lane L14) ─────────────────────
//
// The fixture is the drafter's own `config.json`, byte for byte (Apache-2.0,
// see fixtures/README.md). It differs from the Qwen3.6 drafters in the ways
// below, each pinned by a test so a parser change cannot silently reintroduce
// a mismatch.

/// Verbatim `config.json` of `z-lab/Qwen3.5-4B-DFlash`.
const CFG_4B: &str = include_str!("fixtures/qwen3_5_4b_dflash_config.json");

/// `kernels/ada/holo-3.1-4b/MODEL.toml`, the file that pairs the drafter with
/// the 4B on Ada.
const ADA_4B_MODEL_TOML: &str = include_str!("../../../../../kernels/ada/holo-3.1-4b/MODEL.toml");

/// `key = value` lines of one `[section]` of a MODEL.toml (no toml crate in
/// spark-model; the section holds only scalars and one flat integer array).
fn toml_section(text: &str, section: &str) -> Vec<(String, String)> {
    let mut inside = false;
    let mut out = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            inside = t == format!("[{section}]");
            continue;
        }
        if !inside || t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = t.split_once('=') {
            let v = v.split('#').next().unwrap_or("").trim();
            out.push((k.trim().to_string(), v.trim_matches('"').to_string()));
        }
    }
    out
}

fn toml_get(rows: &[(String, String)], key: &str) -> String {
    rows.iter()
        .find(|(k, _)| k == key)
        .unwrap_or_else(|| panic!("MODEL.toml has no `{key}`"))
        .1
        .clone()
}

#[test]
fn qwen3_5_4b_drafter_fields_reach_the_runtime_config() {
    let cfg = parse_dflash_config(CFG_4B).expect("parse 4B drafter config");
    assert_eq!(cfg.hidden_size, 2560);
    assert_eq!(cfg.num_hidden_layers, 6);
    assert_eq!(cfg.intermediate_size, 9216);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.vocab_size, 248320);
    assert!(
        cfg.tie_word_embeddings,
        "shares the target's embedding/LM head"
    );
    let sub = cfg.dflash_config.as_ref().expect("dflash_config");
    // NOT the Qwen3.6 mask id (248070) and eight taps, not five.
    assert_eq!(sub.mask_token_id, 248077);
    assert_eq!(sub.target_layer_ids, vec![1, 5, 9, 13, 17, 21, 25, 29]);
    assert_eq!(cfg.tap_count(), 8);
    assert_eq!(cfg.fc_in_features(2560), 20480);
    assert_eq!(sub.block_size, Some(16));
    assert_eq!(cfg.effective_block_size(), 16);
    // Block 16 serves at gamma 15 (verify width K = 16, the widest wyN width
    // with kernel coverage under both h-state dtypes).
    assert_eq!(crate::layers::qwen3_ssm::default_dflash_gamma(16), 15);
    assert_eq!(cfg.num_target_layers, Some(32));
}

#[test]
fn qwen3_5_4b_drafter_attention_kinds_and_rope_are_normalized() {
    let cfg = parse_dflash_config(CFG_4B).expect("parse");
    // Five sliding layers (window 4096) and a full-attention last layer.
    assert_eq!(cfg.sliding_window, Some(4096));
    assert!(cfg.use_sliding_window);
    assert_eq!(cfg.layer_types.len(), cfg.num_hidden_layers);
    assert_eq!(cfg.sliding_layer_count(), 5);
    assert!((0..5).all(|l| cfg.layer_is_sliding(l)));
    assert!(!cfg.layer_is_sliding(5));
    assert!(!cfg.layer_is_sliding(6), "out of range is not sliding");
    // `rope_parameters {rope_type: "default"}` is plain RoPE: the block is
    // dropped so the head takes its plain arm without an unknown-type warning,
    // and the stated theta survives.
    assert!(cfg.rope_scaling.is_none());
    assert_eq!(cfg.rope_theta, 10_000_000.0);
}

#[test]
fn an_older_drafter_has_no_sliding_layers() {
    let cfg = parse_dflash_config(SHIPPED_CONFIG).unwrap();
    assert_eq!(cfg.sliding_layer_count(), 0);
    assert!(!cfg.layer_is_sliding(0));
    assert_eq!(cfg.tap_count(), 5);
    assert_eq!(cfg.num_target_layers, None);
}

#[test]
fn rope_parameters_theta_is_used_when_the_top_level_states_none() {
    let json = r#"{
        "hidden_size": 64, "num_hidden_layers": 1, "intermediate_size": 128,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32, "vocab_size": 256,
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000}
    }"#;
    let cfg = parse_dflash_config(json).unwrap();
    assert_eq!(cfg.rope_theta, 1_000_000.0);
    assert!(cfg.rope_scaling.is_none());
    // A top-level theta wins over the block's.
    let both = json.replace(
        "\"vocab_size\": 256,",
        "\"vocab_size\": 256, \"rope_theta\": 500000,",
    );
    assert_eq!(parse_dflash_config(&both).unwrap().rope_theta, 500_000.0);
}

#[test]
fn yarn_rope_parameters_are_left_for_the_head() {
    let json = r#"{
        "hidden_size": 64, "num_hidden_layers": 1, "intermediate_size": 128,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32, "vocab_size": 256,
        "rope_parameters": {"rope_type": "yarn", "factor": 8.0}
    }"#;
    let cfg = parse_dflash_config(json).unwrap();
    let s = cfg.rope_scaling.expect("yarn block kept");
    assert_eq!(s.rope_type.as_deref(), Some("yarn"));
    assert_eq!(s.factor, Some(8.0));
}

#[test]
fn the_4b_drafter_matches_the_ada_holo_4b_target() {
    let cfg = parse_dflash_config(CFG_4B).unwrap();
    // holo-3.1-4b: hidden 2560, 32 layers, vocab 248320 (MODEL.toml [model]).
    let model = toml_section(ADA_4B_MODEL_TOML, "model");
    let hidden: usize = toml_get(&model, "hidden_dim").parse().unwrap();
    let layers: usize = toml_get(&model, "layers_total").parse().unwrap();
    let vocab: usize = toml_get(&model, "vocab_size").parse().unwrap();
    assert_eq!((hidden, layers, vocab), (2560, 32, 248320));
    cfg.validate_for_target(hidden, layers, vocab)
        .expect("drafter pairs with holo-3.1-4b");
}

#[test]
fn the_ada_model_toml_dflash_block_mirrors_the_drafter_config() {
    let cfg = parse_dflash_config(CFG_4B).unwrap();
    let sub = cfg.dflash_config.as_ref().unwrap();
    let rows = toml_section(ADA_4B_MODEL_TOML, "dflash");
    assert_eq!(toml_get(&rows, "draft_model"), "z-lab/Qwen3.5-4B-DFlash");
    assert_eq!(toml_get(&rows, "gamma"), cfg.block_size.to_string());
    assert_eq!(
        toml_get(&rows, "window_size"),
        cfg.sliding_window.unwrap().to_string()
    );
    assert_eq!(
        toml_get(&rows, "mask_token_id"),
        sub.mask_token_id.to_string()
    );
    let ids: Vec<usize> = toml_get(&rows, "target_layer_ids")
        .trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    assert_eq!(ids, sub.target_layer_ids);
}

#[test]
fn a_mismatched_pairing_is_refused_with_the_reason() {
    let cfg = parse_dflash_config(CFG_4B).unwrap();
    // Target with fewer layers than the deepest tap (29) needs.
    let e = cfg.validate_for_target(2560, 28, 248320).unwrap_err();
    assert!(e.to_string().contains("taps target layer 29"), "{e}");
    // Right depth for the taps, wrong depth for the training target.
    let e = cfg.validate_for_target(2560, 40, 248320).unwrap_err();
    assert!(e.to_string().contains("32-layer target"), "{e}");
    // Different vocabulary (shared embedding / LM head would misalign).
    let e = cfg.validate_for_target(2560, 32, 151936).unwrap_err();
    assert!(e.to_string().contains("vocabulary"), "{e}");
    assert!(cfg.validate_for_target(0, 32, 248320).is_err());
}

#[test]
fn unordered_or_missing_taps_are_refused() {
    let unordered = CFG_4B.replace("\"target_layer_ids\": [", "\"target_layer_ids\": [30,");
    let cfg = parse_dflash_config(&unordered).unwrap();
    let e = cfg.validate_for_target(2560, 32, 248320).unwrap_err();
    assert!(e.to_string().contains("strictly increasing"), "{e}");
    let none = parse_dflash_config(
        r#"{"hidden_size":64,"num_hidden_layers":1,"intermediate_size":128,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"vocab_size":256}"#,
    )
    .unwrap();
    assert!(none.validate_for_target(64, 4, 256).is_err());
}

#[test]
fn an_older_drafter_without_num_target_layers_still_validates() {
    // Qwen3.6-35B-A3B drafter against its 40-layer target: no regression for
    // the targets that already wire a drafter.
    let cfg = parse_dflash_config(SHIPPED_CONFIG).unwrap();
    cfg.validate_for_target(2048, 40, 248320)
        .expect("35B pairing");
}
