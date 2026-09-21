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
