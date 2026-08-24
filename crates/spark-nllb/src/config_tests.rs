// SPDX-License-Identifier: AGPL-3.0-only

//! Config parsing tests, incl. real-world HF quirks.

use super::*;

fn config_json(d_model: usize, encoder_heads: usize, decoder_heads: usize) -> String {
    format!(
        r#"{{
            "d_model": {d_model}, "encoder_layers": 24, "decoder_layers": 24,
            "encoder_attention_heads": {encoder_heads}, "decoder_attention_heads": {decoder_heads},
            "encoder_ffn_dim": 8192, "decoder_ffn_dim": 8192,
            "vocab_size": 256205, "max_position_embeddings": 1024
        }}"#
    )
}

#[test]
fn rejects_attention_geometry_the_runtime_cannot_execute() {
    for (name, json, expected) in [
        (
            "zero model width",
            config_json(0, 16, 16),
            "d_model must be greater than zero",
        ),
        (
            "zero heads",
            config_json(1024, 0, 16),
            "encoder_attention_heads must be greater than zero",
        ),
        (
            "zero decoder heads",
            config_json(1024, 16, 0),
            "decoder_attention_heads must be greater than zero",
        ),
        (
            "fractional head width",
            config_json(1025, 16, 16),
            "d_model (1025) must be divisible by encoder_attention_heads (16)",
        ),
        (
            "different decoder layout",
            config_json(1024, 16, 8),
            "decoder_attention_heads (8) must equal encoder_attention_heads (16)",
        ),
    ] {
        let error = NllbConfig::from_json(&json).expect_err(name);
        let message = format!("{error:#}");
        assert!(
            message.contains(expected),
            "{name}: expected {expected:?}, got {message}"
        );
    }
}

#[test]
fn rejects_activation_the_runtime_does_not_implement() {
    let json = config_json(1024, 16, 16).replace(
        "\n        }",
        r#",
            "activation_function": "gelu"
        }"#,
    );
    let error = NllbConfig::from_json(&json).expect_err("the runtime only implements ReLU");
    assert_eq!(
        format!("{error:#}"),
        "invalid NLLB config.json: unsupported activation_function \"gelu\"; spark-nllb implements only \"relu\""
    );
}

#[test]
fn parses_runtime_dimensions_and_defaults() {
    let cfg = NllbConfig::from_json(&config_json(1024, 16, 16)).unwrap();

    assert_eq!(cfg.d_model, 1024);
    assert_eq!(cfg.head_dim(), 64);
    assert_eq!(cfg.encoder_layers, 24);
    assert_eq!(cfg.decoder_layers, 24);
    assert_eq!(cfg.encoder_ffn_dim, 8192);
    assert_eq!(cfg.decoder_ffn_dim, 8192);
    assert_eq!(cfg.vocab_size, 256205);
    assert_eq!(cfg.pad_token_id, 1);
    assert_eq!(cfg.eos_token_id, 2);
    assert_eq!(cfg.decoder_start_token_id, 2);
    assert_eq!(cfg.embed_scale(), 32.0);
    assert_eq!(cfg.max_length, 200);
}

#[test]
fn honors_explicit_runtime_controls() {
    let json = config_json(1024, 16, 16).replace(
        "\n        }",
        r#",
            "pad_token_id": 7,
            "eos_token_id": 9,
            "decoder_start_token_id": 11,
            "scale_embedding": false
        }"#,
    );
    let cfg = NllbConfig::from_json(&json).unwrap();

    assert_eq!(cfg.pad_token_id, 7);
    assert_eq!(cfg.eos_token_id, 9);
    assert_eq!(cfg.decoder_start_token_id, 11);
    assert_eq!(cfg.embed_scale(), 1.0);
}

#[test]
fn tolerates_null_max_length() {
    // The Kuku-Yalanji / distilled-NLLB checkpoints ship `"max_length": null`,
    // which serde(default) alone rejects — this must fall back to the default.
    let json = r#"{
        "d_model": 1024, "encoder_layers": 24, "decoder_layers": 24,
        "encoder_attention_heads": 16, "decoder_attention_heads": 16,
        "encoder_ffn_dim": 8192, "decoder_ffn_dim": 8192,
        "vocab_size": 256205, "max_position_embeddings": 1024,
        "max_length": null
    }"#;
    let cfg = NllbConfig::from_json(json).expect("null max_length must parse");
    assert_eq!(cfg.max_length, 200);
}
