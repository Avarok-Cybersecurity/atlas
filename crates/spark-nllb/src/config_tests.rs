// SPDX-License-Identifier: AGPL-3.0-only

//! Config parsing tests, incl. real-world HF quirks.

use super::*;

#[test]
fn parses_minimal_config() {
    let json = r#"{
        "d_model": 1024, "encoder_layers": 24, "decoder_layers": 24,
        "encoder_attention_heads": 16, "decoder_attention_heads": 16,
        "encoder_ffn_dim": 8192, "decoder_ffn_dim": 8192,
        "vocab_size": 256205, "max_position_embeddings": 1024
    }"#;
    let cfg = NllbConfig::from_json(json).unwrap();
    assert_eq!(cfg.d_model, 1024);
    assert_eq!(cfg.head_dim(), 64);
    assert_eq!(cfg.max_length, 200); // default when absent
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
