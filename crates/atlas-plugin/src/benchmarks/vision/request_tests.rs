// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn a_data_uri_is_what_the_api_accepts() {
    let u = data_uri(&[0x89, b'P', b'N', b'G']);
    assert!(u.starts_with("data:image/png;base64,"));
    assert!(
        !u.contains("http"),
        "must not emit a remote URL — the API rejects those"
    );
}

#[test]
fn images_precede_the_prompt_and_order_is_preserved() {
    // Order is load-bearing: the multi-image probe asks which image came
    // FIRST, so a builder that reordered content would make that probe test
    // the builder rather than the engine.
    let a = b"\x89PNG-A".as_slice();
    let b = b"\x89PNG-B".as_slice();
    let v = body("m", &[a, b], "which is first?", 32);
    let content = v["messages"][0]["content"]
        .as_array()
        .expect("content array");
    assert_eq!(content.len(), 3, "two images then one text part");
    assert_eq!(content[0]["type"], "image_url");
    assert_eq!(content[1]["type"], "image_url");
    assert_eq!(content[2]["type"], "text");
    let first = content[0]["image_url"]["url"].as_str().unwrap();
    let second = content[1]["image_url"]["url"].as_str().unwrap();
    assert_eq!(
        first,
        data_uri(a),
        "first image is not first in the payload"
    );
    assert_ne!(first, second);
}

#[test]
fn a_probe_with_no_images_sends_only_text() {
    // The non-vacuity control. If this ever attached an image the control
    // would pass trivially and stop guarding anything.
    let v = body("m", &[], "no image here", 16);
    let content = v["messages"][0]["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "text");
}

#[test]
fn sampling_is_greedy_and_thinking_is_off() {
    // Both are correctness requirements, not preferences. Temperature > 0
    // makes probes flaky; thinking on a thinking-first checkpoint can eat the
    // whole max_tokens budget and return EMPTY content, which looks exactly
    // like a vision failure.
    let v = body("m", &[], "x", 16);
    assert_eq!(v["temperature"], 0.0);
    assert_eq!(v["chat_template_kwargs"]["enable_thinking"], false);
    assert_eq!(v["stream"], true, "chat_stream requires it");
}

#[test]
fn vision_tokens_subtracts_the_measured_overhead() {
    // 215 prompt_tokens at 19 of template = the 196 measured live for a
    // 448x448 image on 2026-08-14.
    assert_eq!(vision_tokens(215, 19).unwrap(), 196);
}

#[test]
fn an_impossible_subtraction_is_an_error_not_a_wrap() {
    // usize underflow would produce an enormous count and a nonsense verdict.
    // Failing loudly says the calibration no longer applies.
    let e = vision_tokens(5, 19).unwrap_err();
    let msg = format!("{e}");
    assert!(
        msg.contains("below the measured template overhead"),
        "{msg}"
    );
}
