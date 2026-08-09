// SPDX-License-Identifier: AGPL-3.0-only

//! Legacy /v1/completions echo + logprobs wire types.

use crate::openai::*;

#[test]
fn completion_request_echo_logprobs_n_deser() {
    let req: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "prompt": "hello",
        "echo": true,
        "logprobs": 3,
        "n": 2,
        "max_tokens": 0,
        "stream_options": {"include_usage": true},
        "user": "eval-harness",
        "suffix": "tail",
        "best_of": 2,
    }))
    .expect("valid completion request");
    assert!(req.echo);
    assert_eq!(req.logprobs, Some(3));
    assert_eq!(req.n, 2);
    assert_eq!(req.max_tokens, 0);
    assert!(req.stream_options.expect("stream_options").include_usage);
}

#[test]
fn completion_request_defaults_match_openai_spec() {
    // echo=false, n=1, logprobs absent — the spec defaults; a request
    // that names none of the new fields must behave exactly as before.
    let req: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "prompt": "hello",
    }))
    .expect("valid completion request");
    assert!(!req.echo);
    assert_eq!(req.n, 1);
    assert!(req.logprobs.is_none());
    assert!(req.stream_options.is_none());
}

#[test]
fn completion_logprobs_serializes_four_parallel_arrays_with_nulls() {
    let lp = CompletionLogprobs {
        tokens: vec!["He".into(), "llo".into()],
        token_logprobs: vec![None, Some(-0.5)],
        top_logprobs: vec![
            None,
            Some(std::collections::HashMap::from([(
                "llo".to_string(),
                -0.5f32,
            )])),
        ],
        text_offset: vec![0, 2],
    };
    let v = serde_json::to_value(&lp).expect("serialize");
    // Legacy shape: 4 parallel arrays, JSON null for the first echoed token.
    assert!(v["token_logprobs"][0].is_null());
    assert!(v["top_logprobs"][0].is_null());
    assert_eq!(v["tokens"].as_array().map(Vec::len), Some(2));
    assert_eq!(v["text_offset"][1], 2);
}

#[test]
fn completion_response_carries_system_fingerprint_and_optional_logprobs() {
    let usage = Usage {
        prompt_tokens: 1,
        completion_tokens: 1,
        total_tokens: 2,
        prompt_tokens_details: None,
        completion_tokens_details: None,
        time_to_first_token_ms: 0.0,
        response_tokens_per_second: 0.0,
    };
    let resp = CompletionResponse::new("m", "hi".into(), usage, "stop");
    let v = serde_json::to_value(&resp).expect("serialize");
    assert_eq!(v["system_fingerprint"], "fp_atlas");
    // logprobs must be ABSENT (not null) when not requested — some
    // clients treat an explicit null as a malformed logprobs block.
    assert!(v["choices"][0].get("logprobs").is_none());
}

#[test]
fn completion_request_n_bounds_are_handler_enforced() {
    // serde accepts any usize; the HANDLER rejects n==0 and n>128 with a
    // 400 (OpenAI spec bound). This test locks the parse side: values
    // arrive intact for the handler check (no silent serde clamping).
    let req: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "m", "prompt": "p", "n": 4096,
    }))
    .expect("parse");
    assert_eq!(req.n, 4096);
}
