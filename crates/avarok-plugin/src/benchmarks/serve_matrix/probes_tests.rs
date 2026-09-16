// SPDX-License-Identifier: AGPL-3.0-only

//! Probe request shape and the codegen structure check.

use super::*;

#[test]
fn the_base_probe_body_is_deterministic_and_penalty_free() {
    // `tests/single_gpu_suite.py` sends repetition_penalty=1.05 on the codegen
    // probe. It penalises `\n` — the most repeated token in source — and the
    // generated function arrives with 3 newlines instead of 8, i.e. as a
    // syntax error, while the control image is byte-identical in both
    // conditions. Carrying it across would grade the harness, not the model.
    let body = with_user(base_body("m", 512), "write fib".into());
    assert_eq!(
        body,
        serde_json::json!({
            "model": "m",
            "stream": true,
            "temperature": 0.0,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "write fib"}],
        })
    );
}

#[test]
fn the_tool_probe_declares_the_tool_it_grades() {
    let tools = weather_tool();
    assert_eq!(
        tools,
        serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            },
        }])
    );
}

#[test]
fn a_well_formed_function_passes_the_structure_check() {
    let good = "def fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)\n";
    assert_eq!(code_shape(good), Ok(()));
}

#[test]
fn a_reply_collapsed_onto_one_line_is_caught() {
    // Exactly what repetition_penalty on `\n` produces.
    let collapsed = "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)";
    let err = code_shape(collapsed).expect_err("no indented body");
    assert!(err.contains("indented body"), "{err}");
}

#[test]
fn prose_with_no_function_is_caught_separately_from_a_broken_one() {
    let prose = "Sure! Fibonacci is a sequence where each number is the sum of the previous two.";
    let err = code_shape(prose).expect_err("no function");
    assert!(err.contains("`def fib`"), "{err}");
}

#[test]
fn a_similar_name_or_prose_fragment_is_not_a_fib_function() {
    for text in [
        "def fibonacci(n):\n    return n",
        "I would write def fib(n): in Python.\n    This is still prose.",
    ] {
        let err = code_shape(text).expect_err("not an actual fib definition");
        assert!(err.contains("`def fib`"), "{err}");
    }
}

#[test]
fn the_check_accepts_the_common_aliases_a_model_reaches_for() {
    // A fenced block, and a docstring before the body.
    let fenced =
        "```python\ndef fib(n: int) -> int:\n    \"\"\"nth Fibonacci.\"\"\"\n    return n\n```";
    assert_eq!(code_shape(fenced), Ok(()));
}

#[test]
fn only_a_real_4xx_from_the_server_excuses_the_tool_probe() {
    // `NotApplicable` is not a bar, so anything this accepts is scored as a
    // pass. These four all carry "400" or "tool" and are NOT rejections —
    // matching on loose substrings excused a dead server as a parser gap.
    for not_a_rejection in [
        "request exceeded 400s",
        "connecting to http://127.0.0.1:8400: connection refused",
        "endpoint returned \"HTTP/1.1 502 Bad Gateway\": upstream sent no tool support",
        "endpoint returned \"HTTP/1.1 503 Service Unavailable\": model_not_loaded",
    ] {
        assert!(
            !is_client_rejection(not_a_rejection),
            "excused as a known gap: {not_a_rejection}"
        );
    }
    for rejection in [
        "endpoint returned \"HTTP/1.1 400 Bad Request\": tool calling is not supported",
        "endpoint returned \"HTTP/1.1 422 Unprocessable Entity\": no parser for this model",
    ] {
        assert!(
            is_client_rejection(rejection),
            "not recognised: {rejection}"
        );
    }
}

#[test]
fn the_tool_probe_requires_a_weather_call_for_paris() {
    let outcome = |name: &str, arguments: &str| http::ChatOutcome {
        tool_calls: vec![http::ToolCall {
            id: "call-1".into(),
            name: name.into(),
            arguments: arguments.into(),
        }],
        ..Default::default()
    };

    assert_eq!(
        score_tool_call(&outcome("get_weather", r#"{"location":"Paris"}"#)),
        Signal::Pass
    );
    for bad in [
        outcome("get_weather", r#"{"location":"London"}"#),
        outcome("get_weather", r#"{"location":""}"#),
        outcome("other_tool", r#"{"location":"Paris"}"#),
        outcome("get_weather", "not-json"),
        http::ChatOutcome::default(),
    ] {
        assert!(matches!(score_tool_call(&bad), Signal::Fail(_)));
    }
}

#[test]
fn the_needle_is_not_something_the_filler_could_produce() {
    let filler = stats::make_prompt(4096, stats::PromptMode::Natural, "t");
    assert!(!filler.contains(NEEDLE));
}
