// SPDX-License-Identifier: AGPL-3.0-only

//! End-to-end correctness tests against a running Atlas server.
//!
//! Every test in this file calls `require_server()` which panics if no
//! HTTP listener is reachable on `$ATLAS_SERVER_URL` (default
//! `http://localhost:8888`). They are therefore **all `#[ignore]`** so
//! that default `cargo test --workspace` is green on a stock CI host.
//!
//! To run them locally against a live server:
//!
//! ```bash
//! # In one terminal:
//! ./target/release/spark serve <model>
//!
//! # In another:
//! cargo test -p atlas-spark-bench --test correctness -- --ignored
//! ```

use parking_lot::Mutex;

use atlas_spark_bench::{require_server, send_blocking, send_streaming};

// Q/A tests must run sequentially — concurrent model requests degrade output quality.
static SERIAL: Mutex<()> = Mutex::new(());

#[test]
#[ignore = "requires a running Atlas server at $ATLAS_SERVER_URL"]
fn qa_capital_of_france_blocking() {
    let _lock = SERIAL.lock();
    let url = require_server();
    let r = send_blocking(&url, "What is the capital of France?", 50).unwrap();
    let lower = r.text.to_lowercase();
    assert!(lower.contains("paris"), "Expected 'paris' in: {}", r.text);
}

#[test]
#[ignore = "requires a running Atlas server at $ATLAS_SERVER_URL"]
fn qa_capital_of_france_streaming() {
    let _lock = SERIAL.lock();
    let url = require_server();
    let r = send_streaming(&url, "What is the capital of France?", 50).unwrap();
    let lower = r.text.to_lowercase();
    assert!(lower.contains("paris"), "Expected 'paris' in: {}", r.text);
}

#[test]
#[ignore = "requires a running Atlas server at $ATLAS_SERVER_URL"]
fn qa_counting_contains_digits() {
    let _lock = SERIAL.lock();
    let url = require_server();
    let r = send_blocking(&url, "Count from 1 to 5, just the numbers.", 50).unwrap();
    assert!(
        r.text.contains('1') && r.text.contains('5'),
        "Expected '1' and '5' in: {}",
        r.text
    );
}

#[test]
#[ignore = "requires a running Atlas server at $ATLAS_SERVER_URL"]
fn streaming_produces_tokens() {
    let _lock = SERIAL.lock();
    let url = require_server();
    let r = send_streaming(&url, "Say hello.", 20).unwrap();
    assert!(r.token_count > 0, "Expected >0 tokens, got 0");
    assert!(!r.text.is_empty(), "Expected non-empty text");
}

#[test]
#[ignore = "requires a running Atlas server at $ATLAS_SERVER_URL"]
fn finish_reason_is_valid() {
    let _lock = SERIAL.lock();
    let url = require_server();
    let r = send_streaming(&url, "Hi", 10).unwrap();
    assert!(
        r.finish_reason == "stop" || r.finish_reason == "length",
        "Unexpected finish_reason: {}",
        r.finish_reason
    );
}

#[test]
#[ignore = "requires a running Atlas server at $ATLAS_SERVER_URL"]
fn models_endpoint_returns_model() {
    let url = require_server();
    let resp = ureq::get(&format!("{url}/v1/models")).call().unwrap();
    let mut body = resp.into_body();
    let parsed: serde_json::Value = body.read_json().unwrap();
    let data = parsed["data"].as_array().expect("data should be an array");
    assert!(!data.is_empty(), "No models returned");
    assert!(data[0]["id"].as_str().is_some(), "Model id missing");
}

#[test]
#[ignore = "requires a running Atlas server at $ATLAS_SERVER_URL"]
fn health_endpoint_ok() {
    let url = require_server();
    let resp = ureq::get(&format!("{url}/health")).call().unwrap();
    assert_eq!(resp.status(), 200);
}

#[test]
#[ignore = "requires a running Atlas server at $ATLAS_SERVER_URL"]
fn blocking_streaming_consistency() {
    let _lock = SERIAL.lock();
    let url = require_server();
    let b = send_blocking(&url, "What is the capital of France?", 30).unwrap();
    let s = send_streaming(&url, "What is the capital of France?", 30).unwrap();
    assert!(
        b.text.to_lowercase().contains("paris"),
        "Blocking missing 'paris': {}",
        b.text
    );
    assert!(
        s.text.to_lowercase().contains("paris"),
        "Streaming missing 'paris': {}",
        s.text
    );
}
