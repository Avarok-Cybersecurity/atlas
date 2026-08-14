// SPDX-License-Identifier: AGPL-3.0-only

//! Wire-format tests for video content parts, on both APIs.
//!
//! A sibling file rather than an inline module: `chat_message.rs` is not on
//! the file-size-cap allow-list and these cases carried it past 500 lines.

use super::*;

fn parse_chat(content: serde_json::Value) -> ParsedContent {
    let msg = serde_json::json!({"role": "user", "content": content});
    let m: IncomingMessage = serde_json::from_value(msg).expect("deserialise");
    m.content
}

/// The OpenAI-shaped spelling, and the one vLLM documents.
#[test]
fn chat_completions_carries_a_video_url_object() {
    let c = parse_chat(serde_json::json!([
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAA"}},
        {"type": "text", "text": "what happens?"}
    ]));
    assert_eq!(c.videos, vec!["data:video/mp4;base64,AAA"]);
    assert_eq!(c.text, "what happens?");
    assert!(
        c.images.is_empty(),
        "a video must not be counted as an image"
    );
}

/// Qwen's own examples use a flat `video` key, so both are accepted.
#[test]
fn chat_completions_carries_the_flat_video_spelling() {
    let c = parse_chat(serde_json::json!([
        {"type": "video", "video": "data:image/gif;base64,BBB"}
    ]));
    assert_eq!(c.videos, vec!["data:image/gif;base64,BBB"]);
}

#[test]
fn chat_completions_carries_images_and_videos_together() {
    let c = parse_chat(serde_json::json!([
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,III"}},
        {"type": "video_url", "video_url": {"url": "data:image/gif;base64,VVV"}},
        {"type": "text", "text": "compare"}
    ]));
    assert_eq!(c.images, vec!["data:image/png;base64,III"]);
    assert_eq!(c.videos, vec!["data:image/gif;base64,VVV"]);
}

/// ★ The regression this slice exists to prevent. Before it, a video part
/// hit the `_ => {}` catch-all and vanished: the request succeeded, the
/// model answered from the surrounding text, and nothing anywhere said a
/// video had been discarded.
#[test]
fn a_video_part_is_no_longer_silently_dropped() {
    let c = parse_chat(serde_json::json!([
        {"type": "video_url", "video_url": {"url": "data:image/gif;base64,ZZZ"}}
    ]));
    assert!(
        !c.videos.is_empty(),
        "the video was dropped on the floor again"
    );
}

#[test]
fn responses_api_carries_input_video() {
    let item = serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_video", "video_url": {"url": "data:image/gif;base64,RRR"}},
            {"type": "input_text", "text": "describe"}
        ]
    });
    let m = IncomingMessage::from_responses_input_item(&item).expect("message item");
    assert_eq!(m.content.videos, vec!["data:image/gif;base64,RRR"]);
    assert_eq!(m.content.text, "describe");
}

#[test]
fn responses_api_accepts_the_flat_video_url_string() {
    let item = serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [{"type": "video_url", "video_url": "data:image/gif;base64,SSS"}]
    });
    let m = IncomingMessage::from_responses_input_item(&item).expect("message item");
    assert_eq!(m.content.videos, vec!["data:image/gif;base64,SSS"]);
}

/// Text-only content must be untouched by any of this — the overwhelming
/// majority of requests take this path.
#[test]
fn text_only_content_gains_no_videos() {
    let c = parse_chat(serde_json::json!("just a string"));
    assert_eq!(c.text, "just a string");
    assert!(c.videos.is_empty() && c.images.is_empty());
}

#[test]
fn an_unknown_part_type_is_still_ignored() {
    let c = parse_chat(serde_json::json!([
        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AAA"}},
        {"type": "text", "text": "hi"}
    ]));
    assert_eq!(c.text, "hi");
    assert!(c.videos.is_empty(), "audio must not be read as video");
}
