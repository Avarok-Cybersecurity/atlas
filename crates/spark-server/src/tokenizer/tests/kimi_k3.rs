// SPDX-License-Identifier: AGPL-3.0-only

use crate::tokenizer::ChatTokenizer;
use serde_json::json;

#[test]
fn official_k3_chat_refuses_unimplemented_xtml_before_template_fallback() {
    let dir = tempfile::tempdir().unwrap();
    tokenizers::Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default())
        .save(dir.path().join("tokenizer.json"), false)
        .unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        json!({"tokenizer_class":"TikTokenTokenizer"}).to_string(),
    )
    .unwrap();
    let tokenizer =
        ChatTokenizer::from_model_dir(dir.path(), 163585, true, "kimi_k3", None, false).unwrap();
    let messages = [json!({"role":"user","content":"Hello"})];
    for thinking in [false, true] {
        for result in [
            tokenizer.apply_chat_template_jinja(&messages, None, thinking, false),
            tokenizer.apply_chat_template_openai(&messages, None, thinking, false),
        ] {
            let error = result.unwrap_err().to_string();
            assert!(error.contains("Kimi K3 XTML"), "unexpected error: {error}");
        }
    }
}

#[test]
fn official_k3_xtml_detection_does_not_confuse_the_twin() {
    let dir = tempfile::tempdir().unwrap();
    assert!(!super::super::kimi_k3::uses_xtml(dir.path(), "kimi_k3").unwrap());
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        json!({"tokenizer_class":"PreTrainedTokenizerFast", "chat_template":"{{ messages }}"})
            .to_string(),
    )
    .unwrap();
    assert!(!super::super::kimi_k3::uses_xtml(dir.path(), "kimi_k3").unwrap());
    std::fs::write(dir.path().join("tiktoken.model"), "fixture").unwrap();
    assert!(super::super::kimi_k3::uses_xtml(dir.path(), "kimi_k3").unwrap());
    assert!(!super::super::kimi_k3::uses_xtml(dir.path(), "qwen3").unwrap());
}

#[test]
fn official_k3_guard_also_covers_tool_requests_and_template_overrides() {
    let dir = tempfile::tempdir().unwrap();
    let templates = dir.path().join("jinja-templates/openai");
    std::fs::create_dir_all(&templates).unwrap();
    std::fs::write(templates.join("kimi_k3.jinja"), "wrong template").unwrap();
    std::fs::write(dir.path().join("tiktoken.model"), "fixture").unwrap();
    tokenizers::Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default())
        .save(dir.path().join("tokenizer.json"), false)
        .unwrap();
    let tokenizer =
        ChatTokenizer::from_model_dir(dir.path(), 163585, true, "kimi_k3", Some(dir.path()), false)
            .unwrap();
    let error = tokenizer
        .apply_chat_template_openai(
            &[json!({"role":"user","content":"Check weather"})],
            Some(&[json!({"type":"function","function":{"name":"weather"}})]),
            false,
            false,
        )
        .unwrap_err();
    assert!(error.to_string().contains("Kimi K3 XTML"));
}
