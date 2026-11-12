// SPDX-License-Identifier: AGPL-3.0-only

//! Request parsing for `/completion`: honored fields, accepted no-ops, and the
//! 400 paths for fields Atlas cannot honor.

use axum::http::StatusCode;
use serde_json::{Value, json};

use super::super::toy_codec::{BOS, ToyCodec};
use super::*;

fn defaults() -> SamplingDefaults {
    SamplingDefaults {
        temperature: 0.7,
        top_k: 20,
        top_p: 0.8,
        min_p: 0.0,
        top_n_sigma: 0.0,
        repetition_penalty: 1.05,
    }
}

fn ctx(grammar_available: bool) -> ParseContext<'static> {
    ParseContext {
        codec: &ToyCodec,
        defaults: defaults(),
        max_seq_len: 64,
        grammar_available,
    }
}

fn parse(v: Value) -> Result<CompletionSpec, CompatError> {
    parse_completion(v.as_object().expect("object"), &ctx(true))
}

fn ok(v: Value) -> CompletionSpec {
    parse(v.clone()).unwrap_or_else(|e| panic!("{v} should parse: {}", e.message))
}

fn rejected(v: Value) -> CompatError {
    let err = parse(v.clone()).expect_err(&format!("{v} must be rejected"));
    assert_eq!(err.status, StatusCode::BAD_REQUEST, "{v}: {}", err.message);
    err
}

#[test]
fn a_bare_prompt_gets_the_server_defaults() {
    let s = ok(json!({"prompt": "hi"}));
    assert_eq!(
        s.prompt_tokens,
        vec![BOS, 204, 205],
        "string prompts get add_special"
    );
    assert_eq!(s.temperature, 0.7);
    assert_eq!(s.top_k, 20);
    assert_eq!(s.top_p, 0.8);
    assert_eq!(s.repetition_penalty, 1.05);
    assert_eq!(s.n_predict, -1);
    assert_eq!(s.max_tokens, 64 - 3, "-1 = until the context is full");
    assert!(!s.stream && s.n_probs == 0 && !s.return_tokens);
    assert_eq!(s.seed, None);
    assert!(s.constraint.is_none() && s.stop.is_empty());
}

#[test]
fn the_sampling_fields_a_client_sets_are_honored() {
    let s = ok(json!({
        "prompt": "x", "n_predict": 10, "temperature": 0.2, "top_k": 5, "top_p": 0.5,
        "min_p": 0.1, "seed": 42, "stream": true, "stop": ["\n", "END"],
        "repeat_penalty": 1.2, "presence_penalty": 0.5, "frequency_penalty": -0.5,
        "top_n_sigma": 2.0, "return_tokens": true, "cache_prompt": true
    }));
    assert_eq!(s.max_tokens, 10);
    assert_eq!(
        (s.temperature, s.top_k, s.top_p, s.min_p),
        (0.2, 5, 0.5, 0.1)
    );
    assert_eq!(s.seed, Some(42));
    assert!(s.stream && s.return_tokens);
    assert_eq!(s.stop, vec!["\n".to_string(), "END".to_string()]);
    assert_eq!(s.repetition_penalty, 1.2);
    assert_eq!((s.presence_penalty, s.frequency_penalty), (0.5, -0.5));
    assert_eq!(s.top_n_sigma, 2.0);
}

#[test]
fn null_means_absent_and_aliases_are_read() {
    let s = ok(json!({"prompt": "x", "temperature": null, "n_predict": null, "max_tokens": 7}));
    assert_eq!(s.temperature, 0.7);
    assert_eq!(
        s.max_tokens, 7,
        "max_tokens is llama.cpp's alias of n_predict"
    );
    let s = ok(json!({"prompt": "x", "logprobs": 3}));
    assert_eq!(s.n_probs, 3, "logprobs is llama.cpp's alias of n_probs");
    let s = ok(json!({"prompt": "x", "n_predict": 5, "max_tokens": 9}));
    assert_eq!(s.max_tokens, 5, "n_predict outranks its aliases");
}

#[test]
fn llama_cpp_clamping_rules() {
    let s = ok(json!({"prompt": "x", "temperature": -1, "top_k": -4, "top_p": 7, "min_p": -2}));
    assert_eq!(
        (s.temperature, s.top_k, s.top_p, s.min_p),
        (0.0, 0, 1.0, 0.0)
    );
    let s = ok(json!({"prompt": "x", "seed": -1}));
    assert_eq!(s.seed, None, "-1 is a random seed");
    let s = ok(json!({"prompt": "x", "top_n_sigma": -1}));
    assert_eq!(s.top_n_sigma, 0.0, "negative disables, which is 0 in Atlas");
    let s = ok(json!({"prompt": "x", "dry_multiplier": 0.8, "dry_base": 0.5}));
    assert_eq!(
        (s.dry_multiplier, s.dry_base),
        (0.8, 1.75),
        "a base below 1 is the default"
    );
}

#[test]
fn n_predict_is_bounded_by_the_context() {
    let s = ok(json!({"prompt": "x", "n_predict": 100000}));
    assert_eq!(s.max_tokens, 64 - 2);
    assert!(
        rejected(json!({"prompt": "x", "n_predict": -2}))
            .message
            .contains("n_predict")
    );
    let zero = rejected(json!({"prompt": "x", "n_predict": 0}));
    assert!(zero.message.contains("n_predict=0"));
}

#[test]
fn prompt_shapes() {
    let ids = |s: CompletionSpec| s.prompt_tokens;
    assert_eq!(ids(ok(json!({"prompt": [200, 201]}))), vec![200, 201]);
    assert_eq!(
        ids(ok(json!({"prompt": [7, "a"]}))),
        vec![7, 197],
        "no BOS after a leading id"
    );
    assert_eq!(
        ids(ok(json!({"prompt": ["a"]}))),
        vec![BOS, 197],
        "one-element array is one prompt"
    );
    assert_eq!(
        ids(ok(json!({"prompt": {"prompt_string": "a"}}))),
        vec![BOS, 197]
    );
}

#[test]
fn bad_prompts_are_400s() {
    assert!(rejected(json!({})).message.contains("prompt"));
    assert!(rejected(json!({"prompt": null})).message.contains("prompt"));
    rejected(json!({"prompt": []}));
    rejected(json!({"prompt": 5}));
    rejected(json!({"prompt": [100000]}));
    rejected(json!({"prompt": [true]}));
}

#[test]
fn several_prompts_and_multimodal_prompts_are_refused_not_truncated() {
    let e = rejected(json!({"prompt": ["a", "b"]}));
    assert!(e.message.contains("several prompts"));
    rejected(json!({"prompt": [[1, 2], [3]]}));
    let e = rejected(json!({"prompt": {"prompt_string": "a", "multimodal_data": ["AA=="]}}));
    assert!(e.message.contains("multimodal"));
    rejected(json!({"prompt": "a", "image_data": [{"id": 1}]}));
    assert!(
        rejected(json!({"prompt": "a", "n_cmpl": 2}))
            .message
            .contains("n_cmpl")
    );
    assert!(
        rejected(json!({"prompt": "a", "n": 3}))
            .message
            .contains("n_cmpl")
    );
    ok(json!({"prompt": "a", "n_cmpl": 1}));
}

#[test]
fn an_oversized_prompt_is_a_context_error() {
    let long = "a".repeat(64);
    let e = rejected(json!({"prompt": long}));
    assert_eq!(e.kind, ErrorKind::ExceedContext);
    assert_eq!(e.body()["error"]["type"], "exceed_context_size_error");
}

use super::super::error::ErrorKind;

#[test]
fn unsupported_samplers_are_refused_by_name_unless_off() {
    for (field, bad) in [
        ("mirostat", json!(1)),
        ("typical_p", json!(0.9)),
        ("xtc_probability", json!(0.5)),
        ("dynatemp_range", json!(0.5)),
        ("min_keep", json!(2)),
        ("n_indent", json!(4)),
        ("repeat_last_n", json!(128)),
        ("adaptive_target", json!(0.5)),
        ("t_max_predict_ms", json!(100)),
        ("ignore_eos", json!(true)),
        ("post_sampling_probs", json!(true)),
        ("timings_per_token", json!(true)),
        ("return_progress", json!(true)),
        ("backend_sampling", json!(true)),
        ("reasoning_control", json!(true)),
        ("reasoning_budget_tokens", json!(100)),
        ("lora", json!([{"id": 0, "scale": 1.0}])),
        ("response_fields", json!(["content"])),
        ("samplers", json!(["top_k", "temperature"])),
        ("samplers", json!("kt")),
    ] {
        let e = rejected(json!({"prompt": "a", field: bad.clone()}));
        assert!(
            e.message.contains(&format!("'{field}'")),
            "{field}: {}",
            e.message
        );
    }
}

#[test]
fn the_same_fields_are_accepted_at_their_off_values() {
    ok(json!({
        "prompt": "a", "mirostat": 0, "typical_p": 1.0, "xtc_probability": 0.0,
        "dynatemp_range": 0.0, "min_keep": 0, "n_indent": 0, "repeat_last_n": 64,
        "adaptive_target": -1.0, "t_max_predict_ms": 0, "ignore_eos": false,
        "post_sampling_probs": false, "timings_per_token": false, "lora": [],
        "response_fields": [], "reasoning_budget_tokens": -1,
        "samplers": ["dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"],
        "id_slot": -1, "n_keep": 0, "n_discard": 0, "n_cache_reuse": 0, "echo": false,
        "cache_prompt": false, "verbose": false, "stream_options": {"include_usage": true},
        "sse_ping_interval": 5, "reasoning_format": "none"
    }));
}

#[test]
fn dry_knobs_without_an_equivalent_matter_only_when_dry_is_on() {
    ok(json!({"prompt": "a", "dry_penalty_last_n": 5, "dry_sequence_breakers": ["x"]}));
    let e = rejected(json!({"prompt": "a", "dry_multiplier": 0.8, "dry_penalty_last_n": 5}));
    assert!(e.message.contains("dry_penalty_last_n"));
    rejected(json!({"prompt": "a", "dry_multiplier": 0.8, "dry_sequence_breakers": ["x"]}));
    ok(json!({
        "prompt": "a", "dry_multiplier": 0.8, "dry_penalty_last_n": 64,
        "dry_sequence_breakers": ["\n", ":", "\"", "*"]
    }));
}

#[test]
fn n_probs_is_bounded_and_post_sampling_is_refused() {
    assert_eq!(ok(json!({"prompt": "a", "n_probs": 20})).n_probs, 20);
    let e = rejected(json!({"prompt": "a", "n_probs": 21}));
    assert!(e.message.contains("n_probs"));
    rejected(json!({"prompt": "a", "n_probs": -1}));
    rejected(json!({"prompt": "a", "logprobs": true}));
}

#[test]
fn out_of_range_values_are_400s() {
    rejected(json!({"prompt": "a", "temperature": 3.0}));
    rejected(json!({"prompt": "a", "presence_penalty": 5}));
    rejected(json!({"prompt": "a", "frequency_penalty": -3}));
    rejected(json!({"prompt": "a", "top_n_sigma": 0}));
    rejected(json!({"prompt": "a", "temperature": "hot"}));
    rejected(json!({"prompt": "a", "stream": "yes"}));
    rejected(json!({"prompt": "a", "top_k": 1.5}));
}

#[test]
fn stop_accepts_a_string_or_an_array_of_strings_only() {
    assert_eq!(
        ok(json!({"prompt": "a", "stop": "x"})).stop,
        vec!["x".to_string()]
    );
    rejected(json!({"prompt": "a", "stop": [1]}));
    rejected(json!({"prompt": "a", "stop": 1}));
}

#[test]
fn logit_bias_forms() {
    let a = 100 + u32::from(b'a');
    let s = ok(json!({"prompt": "x", "logit_bias": [[a, 1.5], [a + 1, false]]}));
    assert_eq!(s.logit_bias, vec![(a, 1.5), (a + 1, f32::NEG_INFINITY)]);
    let s = ok(json!({"prompt": "x", "logit_bias": {a.to_string(): -2.0}}));
    assert_eq!(s.logit_bias, vec![(a, -2.0)]);
    let s = ok(json!({"prompt": "x", "logit_bias": [["ab", 0.5]]}));
    assert_eq!(
        s.logit_bias,
        vec![(a, 0.5), (a + 1, 0.5)],
        "a string biases every token of it"
    );
    rejected(json!({"prompt": "x", "logit_bias": [[100000, 1.0]]}));
    rejected(json!({"prompt": "x", "logit_bias": [[1]]}));
    rejected(json!({"prompt": "x", "logit_bias": [[1, "high"]]}));
    rejected(json!({"prompt": "x", "logit_bias": 5}));
}

#[test]
fn constraints_reach_the_spec_and_never_fall_back_to_free_text() {
    let s = ok(json!({"prompt": "a", "json_schema": {"type": "object"}}));
    assert!(matches!(s.constraint, Some(Constraint::JsonSchema(_))));
    let s = ok(json!({"prompt": "a", "grammar": "root ::= \"a\""}));
    assert!(matches!(s.constraint, Some(Constraint::Ebnf(_))));
    let off = parse_completion(
        json!({"prompt": "a", "json_schema": {"type": "object"}})
            .as_object()
            .unwrap(),
        &ctx(false),
    )
    .expect_err("engine down");
    assert_eq!(off.status, StatusCode::BAD_REQUEST);
    rejected(json!({"prompt": "a", "grammar": "root ::= ("}));
}

#[test]
fn generation_settings_report_what_the_request_ran_with() {
    let s = ok(json!({"prompt": "a", "n_predict": 4, "seed": 9, "grammar": "root ::= \"a\""}));
    let g = s.generation_settings("m", 64);
    assert_eq!(g["n_ctx"], 64);
    assert_eq!(g["model"], "m");
    assert_eq!(g["seed"], 9);
    assert_eq!(g["n_predict"], 4);
    assert_eq!(g["grammar"], "root ::= \"a\"");
}
