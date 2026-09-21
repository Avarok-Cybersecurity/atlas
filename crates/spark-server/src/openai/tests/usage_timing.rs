// SPDX-License-Identifier: AGPL-3.0-only

//! The usage block's raw timing components (2026-09-20).
//!
//! The server always measured TTFT and the decode window but published
//! only a rate derived from them, so a client wanting Inter-Token
//! Latency had to time SSE arrivals. These pin that the raw components
//! now ride the wire, that the shipped keys are untouched, and that the
//! AIPerf identity holds on the numbers as published.

use crate::ir;
use crate::openai::Usage;

fn ir_usage(completion_tokens: usize, ttft_ms: f64, decode_ms: f64) -> ir::Usage {
    ir::Usage {
        prompt_tokens: 40,
        completion_tokens,
        cached_prompt_tokens: 8,
        reasoning_tokens: 3,
        accepted_prediction_tokens: 5,
        time_to_first_token_ms: ttft_ms,
        decode_time_ms: decode_ms,
        response_tokens_per_second: ir::Usage::decode_rate_tok_s(completion_tokens, decode_ms),
    }
}

#[test]
fn wire_usage_carries_the_raw_timing_components_and_keeps_the_shipped_keys() {
    let v = serde_json::to_value(Usage::from(&ir_usage(10, 100.0, 900.0))).unwrap();
    // NEW, additive.
    assert_eq!(v["decode_time_ms"], 900.0, "{v}");
    assert_eq!(v["total_time_ms"], 1000.0, "{v}");
    // SHIPPED — vLLM-comparison tooling and OpenAI-compat clients read
    // these by name; neither may move or change meaning.
    assert_eq!(v["time_to_first_token_ms"], 100.0, "{v}");
    assert_eq!(v["response_token/s"], 10.0, "{v}");
    assert_eq!(v["prompt_tokens"], 40);
    assert_eq!(v["completion_tokens"], 10);
    assert_eq!(v["total_tokens"], 50);
    assert_eq!(v["prompt_tokens_details"]["cached_tokens"], 8);
    assert_eq!(v["completion_tokens_details"]["reasoning_tokens"], 3);
    assert_eq!(
        v["completion_tokens_details"]["accepted_prediction_tokens"],
        5
    );
}

/// AIPerf: `ITL = (request_latency − TTFT) / (output_tokens − 1)`. On the
/// server's own clock that difference must be exactly the decode window,
/// and both must agree with the shipped rate — otherwise the three
/// published numbers describe three different requests.
#[test]
fn the_wire_components_satisfy_the_aiperf_identity() {
    let u = Usage::from(&ir_usage(10, 100.0, 900.0));
    let itl_from_total = (u.total_time_ms - u.time_to_first_token_ms) / 9.0;
    let itl_from_window = u.decode_time_ms / 9.0;
    let itl_from_rate = 1000.0 / u.response_tokens_per_second;
    assert!((itl_from_total - itl_from_window).abs() < 1e-9);
    assert!((itl_from_window - itl_from_rate).abs() < 1e-9);
    assert!((itl_from_window - 100.0).abs() < 1e-9);
}

/// The total is DERIVED from the two stored components — the encoder
/// never assembles it — so changing either primitive moves it.
#[test]
fn total_time_is_the_sum_of_its_two_primitives() {
    assert_eq!(ir_usage(2, 12.5, 0.0).total_time_ms(), 12.5);
    assert_eq!(ir_usage(2, 12.5, 87.5).total_time_ms(), 100.0);
    assert_eq!(ir_usage(2, 0.0, 87.5).total_time_ms(), 87.5);
}

/// The shipped rate's undefined case is 0.0 (contract); the raw window
/// is still published so a consumer can tell "one token" from "instant".
#[test]
fn decode_rate_is_zero_when_undefined_and_n_minus_one_per_second_otherwise() {
    assert_eq!(ir::Usage::decode_rate_tok_s(1, 500.0), 0.0);
    assert_eq!(ir::Usage::decode_rate_tok_s(0, 500.0), 0.0);
    assert_eq!(ir::Usage::decode_rate_tok_s(5, 0.0), 0.0);
    assert_eq!(ir::Usage::decode_rate_tok_s(5, 1000.0), 4.0);
    // A one-token response still publishes its (tiny) raw window.
    let v = serde_json::to_value(Usage::from(&ir_usage(1, 100.0, 0.4))).unwrap();
    assert_eq!(v["response_token/s"], 0.0);
    assert_eq!(v["decode_time_ms"], 0.4);
}
