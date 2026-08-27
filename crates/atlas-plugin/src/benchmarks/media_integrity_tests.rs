// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the integrity cells and the request shapes. The checks
//! themselves need a served model; what is checkable here is the scoring and
//! that the bodies say what they are supposed to say.

use super::*;

#[test]
fn a_pass_is_measured_and_passed() {
    let c = Cell::Pass {
        id: "x",
        detail: "d".into(),
    };
    assert!(c.passed() && c.measured());
}

#[test]
fn a_failure_is_measured_but_not_passed() {
    let c = Cell::Fail {
        id: "x",
        detail: "d".into(),
    };
    assert!(!c.passed() && c.measured());
}

/// ★ A skip must NOT count as measured. A suite that skipped everything and
/// reported a pass is how a capability stops being tested without anyone
/// deciding that it should.
#[test]
fn a_skip_is_neither_passed_nor_measured() {
    let c = Cell::Skipped {
        id: "x",
        why: "no decoder".into(),
    };
    assert!(!c.passed());
    assert!(!c.measured(), "a skip must not inflate the denominator");
}

#[test]
fn an_error_is_neither_passed_nor_measured() {
    let c = Cell::Error {
        id: "x",
        msg: "boom".into(),
    };
    assert!(!c.passed() && !c.measured());
}

#[test]
fn cell_lines_preserve_variant_evidence() {
    let cases = [
        (
            Cell::Pass {
                id: "concurrency",
                detail: "four distinct replies".into(),
            },
            "concurrency",
            "concurrency: four distinct replies",
        ),
        (
            Cell::Fail {
                id: "prefix-cache-isolation",
                detail: "served the first image".into(),
            },
            "prefix-cache-isolation",
            "prefix-cache-isolation: FAILED — served the first image",
        ),
        (
            Cell::Skipped {
                id: "history",
                why: "no decoder".into(),
            },
            "history",
            "history: skipped — no decoder",
        ),
        (
            Cell::Error {
                id: "long-prompt",
                msg: "connection reset".into(),
            },
            "long-prompt",
            "long-prompt: connection reset",
        ),
    ];

    for (cell, expected_id, expected_line) in cases {
        assert_eq!(cell.id(), expected_id);
        assert_eq!(cell.line(), expected_line);
    }
}

#[test]
fn an_image_request_carries_the_image_then_the_prompt() {
    let b = image_request("m", "image/png", b"bytes", "what colour", 32);
    assert_eq!(
        b,
        serde_json::json!({
            "model": "m",
            "stream": true,
            "temperature": 0.0,
            "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": false},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,Ynl0ZXM="
                }},
                {"type": "text", "text": "what colour"},
            ]}],
        })
    );
}

/// The mime is carried through rather than assumed to be PNG — the decode
/// variants deliberately include a JPEG.
#[test]
fn the_mime_type_is_not_hardcoded() {
    let b = image_request("m", "image/jpeg", b"x", "p", 8);
    assert!(
        b["messages"][0]["content"][0]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/jpeg;base64,")
    );
}
