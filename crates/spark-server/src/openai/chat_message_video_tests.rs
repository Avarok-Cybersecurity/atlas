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

/// The parsed media as `(kind, uri)` pairs — both what each item is and
/// where it sits in the sequence, which is the property these tests pin.
fn media(c: &ParsedContent) -> Vec<(MediaKind, &str)> {
    c.media.iter().map(|m| (m.kind, m.uri.as_str())).collect()
}

/// The OpenAI-shaped spelling, and the one vLLM documents.
#[test]
fn chat_completions_carries_a_video_url_object() {
    let c = parse_chat(serde_json::json!([
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAA"}},
        {"type": "text", "text": "what happens?"}
    ]));
    assert_eq!(
        media(&c),
        vec![(MediaKind::Video, "data:video/mp4;base64,AAA")],
        "a video must not be counted as an image"
    );
    assert_eq!(c.text, "what happens?");
}

/// Qwen's own examples use a flat `video` key, so both are accepted.
#[test]
fn chat_completions_carries_the_flat_video_spelling() {
    let c = parse_chat(serde_json::json!([
        {"type": "video", "video": "data:image/gif;base64,BBB"}
    ]));
    assert_eq!(
        media(&c),
        vec![(MediaKind::Video, "data:image/gif;base64,BBB")]
    );
}

#[test]
fn chat_completions_carries_images_and_videos_together() {
    let c = parse_chat(serde_json::json!([
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,III"}},
        {"type": "video_url", "video_url": {"url": "data:image/gif;base64,VVV"}},
        {"type": "text", "text": "compare"}
    ]));
    assert_eq!(
        media(&c),
        vec![
            (MediaKind::Image, "data:image/png;base64,III"),
            (MediaKind::Video, "data:image/gif;base64,VVV"),
        ]
    );
}

/// ★ The reordering regression. Media used to be parsed into two lists,
/// one per modality, so a video sent FIRST arrived behind the image no
/// matter what the client wrote — and nothing errored, because the pad
/// runs and the encoder rows still agreed with each other. Only the model
/// saw it, as an answer about the wrong item.
#[test]
fn a_video_sent_before_an_image_stays_before_it() {
    let c = parse_chat(serde_json::json!([
        {"type": "video_url", "video_url": {"url": "data:image/gif;base64,VVV"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,III"}},
        {"type": "text", "text": "which came first?"}
    ]));
    assert_eq!(
        media(&c),
        vec![
            (MediaKind::Video, "data:image/gif;base64,VVV"),
            (MediaKind::Image, "data:image/png;base64,III"),
        ],
        "the client's order must survive the wire parse"
    );
}

/// Interleaving beyond a single pair — three items alternating, so a fix
/// that merely swapped the two groups would not pass.
#[test]
fn alternating_media_keeps_every_position() {
    let c = parse_chat(serde_json::json!([
        {"type": "video_url", "video_url": {"url": "v1"}},
        {"type": "image_url", "image_url": {"url": "i1"}},
        {"type": "video_url", "video_url": {"url": "v2"}},
    ]));
    assert_eq!(
        media(&c),
        vec![
            (MediaKind::Video, "v1"),
            (MediaKind::Image, "i1"),
            (MediaKind::Video, "v2"),
        ]
    );
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
        c.media.iter().any(|m| m.kind == MediaKind::Video),
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
    assert_eq!(
        media(&m.content),
        vec![(MediaKind::Video, "data:image/gif;base64,RRR")]
    );
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
    assert_eq!(
        media(&m.content),
        vec![(MediaKind::Video, "data:image/gif;base64,SSS")]
    );
}

/// The same ordering contract on `/v1/responses` — a separate parser, so a
/// separate case.
#[test]
fn responses_api_preserves_media_order() {
    let item = serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_video", "video_url": {"url": "vvv"}},
            {"type": "input_image", "image_url": "iii"},
        ]
    });
    let m = IncomingMessage::from_responses_input_item(&item).expect("message item");
    assert_eq!(
        media(&m.content),
        vec![(MediaKind::Video, "vvv"), (MediaKind::Image, "iii")]
    );
}

/// Text-only content must be untouched by any of this — the overwhelming
/// majority of requests take this path.
#[test]
fn text_only_content_gains_no_videos() {
    let c = parse_chat(serde_json::json!("just a string"));
    assert_eq!(c.text, "just a string");
    assert!(c.media.is_empty());
}

#[test]
fn an_unknown_part_type_is_still_ignored() {
    let c = parse_chat(serde_json::json!([
        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AAA"}},
        {"type": "text", "text": "hi"}
    ]));
    assert_eq!(c.text, "hi");
    assert!(c.media.is_empty(), "audio must not be read as video");
}
