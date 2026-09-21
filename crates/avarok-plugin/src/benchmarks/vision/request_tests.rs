// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn a_data_uri_is_what_the_api_accepts() {
    assert_eq!(
        data_uri(&[0x89, b'P', b'N', b'G']),
        "data:image/png;base64,iVBORw=="
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
    assert_eq!(
        v,
        serde_json::json!({
            "model": "m",
            "stream": true,
            "temperature": 0.0,
            "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": false},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,iVBORy1B"
                }},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,iVBORy1C"
                }},
                {"type": "text", "text": "which is first?"}
            ]}]
        })
    );
}

#[test]
fn a_probe_with_no_images_sends_only_text() {
    // The non-vacuity control. If this ever attached an image the control
    // would pass trivially and stop guarding anything.
    let v = body("m", &[], "no image here", 16);
    assert_eq!(
        v,
        serde_json::json!({
            "model": "m",
            "stream": true,
            "temperature": 0.0,
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": false},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "no image here"}
            ]}]
        })
    );
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
    assert_eq!(
        e.to_string(),
        "prompt_tokens 5 is below the measured template overhead 19 — the calibration request \
         and this one did not render the same template, so the subtraction is meaningless"
    );
}
