// SPDX-License-Identifier: AGPL-3.0-only

//! The mid-stream error frame on the legacy `/v1/completions` SSE stream.
//!
//! The frame is the ONLY way a streaming client learns why its request
//! died: the stream carries HTTP 200, so a malformed frame is not a
//! degraded error message, it is no error message at all — the client's
//! JSON parser raises on the frame and the real reason never surfaces.

use crate::api::compact::completion_error_frame;

/// Messages that actually reach this frame: `send_error_to_sink` forwards
/// anyhow `{e:#}` chains verbatim, and those routinely quote a path, a
/// token or an inner error.
const HOSTILE: &[&str] = &[
    r#"swap-in failed: open "/var/spill/swap_7.bin": No such file"#,
    "prefill failed: CUDA error\nlaunch failed",
    r"grammar compile failed near \x00",
    r#"tool "bash" rejected: unbalanced """#,
];

#[test]
fn an_error_message_survives_the_frame_as_parseable_json() {
    for msg in HOSTILE {
        let frame = completion_error_frame(msg);
        let v: serde_json::Value = serde_json::from_str(&frame).unwrap_or_else(|e| {
            panic!("frame for {msg:?} must be JSON a client can parse, got {frame:?}: {e}")
        });
        // Parseable is not enough — the reason has to arrive intact.
        assert_eq!(
            v.get("error").and_then(|e| e.as_str()),
            Some(*msg),
            "frame must round-trip the reason verbatim: {frame}"
        );
    }
}

#[test]
fn the_wire_shape_is_unchanged_for_an_ordinary_message() {
    // Pins the envelope: this fix changes the ENCODING, not the contract
    // every existing client already parses.
    assert_eq!(
        completion_error_frame("Scheduler queue closed"),
        r#"{"error":"Scheduler queue closed"}"#
    );
}
