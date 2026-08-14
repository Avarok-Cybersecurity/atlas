// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for request shaping and the skip predicate.

use super::*;

#[test]
fn a_video_request_carries_the_clip_then_the_prompt() {
    let b = video_body("m", "video/mp4", b"\x00\x00\x00\x20ftyp", "go", 32);
    let content = b["messages"][0]["content"].as_array().expect("array");
    assert_eq!(content[0]["type"], "video_url");
    assert!(
        content[0]["video_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:video/mp4;base64,")
    );
    assert_eq!(content[1]["type"], "text");
    assert_eq!(b["temperature"], 0.0);
    assert_eq!(b["chat_template_kwargs"]["enable_thinking"], false);
}

/// The image must come FIRST. That order is the contract the mixed leg exists
/// to test, so the request states it deliberately rather than incidentally.
#[test]
fn a_mixed_request_puts_the_image_before_the_video() {
    let b = mixed_body("m", b"png", "image/gif", b"GIF89a", "go", 32);
    let content = b["messages"][0]["content"].as_array().expect("array");
    assert_eq!(content[0]["type"], "image_url");
    assert_eq!(content[1]["type"], "video_url");
    assert_eq!(content[2]["type"], "text");
}

#[test]
fn the_control_carries_no_media_at_all() {
    let b = text_only_body("m", "go", 32);
    assert!(
        b["messages"][0]["content"].is_string(),
        "a content ARRAY could render a vision marker even with no parts"
    );
}

/// ★ These strings are the server's operator-facing errors, asserted in
/// `video_decode_ffmpeg`'s own tests. Matching them is what turns "this
/// deployment has no decoder" into a SKIP rather than a failure — and if the
/// wording ever changes, this test fails loudly instead of the skips silently
/// becoming failures.
#[test]
fn a_missing_decoder_is_recognized_as_a_skip() {
    for msg in [
        "this container needs ffmpeg to decode and subprocess decoding is disabled; \
         pass --video-allow-ffmpeg to enable it, or send an animated GIF",
        "could not run \"ffmpeg\" — is ffmpeg installed and on PATH? \
         (set --video-ffmpeg-path to point at it)",
        "\"/nonexistent/ffmpeg\" could not be run: No such file or directory",
    ] {
        assert!(is_decoder_unavailable(msg), "should skip: {msg}");
    }
}

/// A genuine decode failure must NOT be mistaken for a missing decoder — that
/// would turn a real defect into a green skip, which is the worse of the two
/// mistakes this predicate can make.
#[test]
fn a_real_decode_failure_is_not_a_skip() {
    for msg in [
        "decoder failed: Invalid data found when processing input",
        "the container decoded to zero frames (is there a video stream?)",
        "decoded output exceeded the 1024-byte cap",
        "video has 1 usable frame(s) but temporal_patch_size is 2",
    ] {
        assert!(!is_decoder_unavailable(msg), "should NOT skip: {msg}");
    }
}

#[test]
fn the_order_prompt_asks_for_a_scoreable_answer() {
    assert!(ORDER_PROMPT.contains("order"));
    assert!(
        ORDER_PROMPT.contains("only the color names"),
        "an open-ended prompt invites prose that cannot be scored"
    );
}
