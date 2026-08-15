// SPDX-License-Identifier: AGPL-3.0-only

//! ★ The reordering regression, at the site that builds the three vectors
//! which must agree: the encoder inputs, the pad counts, and (via
//! `MsgEntry::media`) the rendered markers.
//!
//! These call `collect_message_media` directly rather than going through
//! `build_msg_entries`, because collection is where order is decided and
//! nothing here has to decode: a URI is carried verbatim at this stage, so
//! the cases run without a vision config, real pixels, or ffmpeg.

use super::collect_message_media;
use crate::ir::MediaKind;
use crate::ir::message::{ContentPart, ImageData, ImageSource, Message, Role, VideoSource};

fn img(uri: &str) -> ContentPart {
    ContentPart::Image(ImageSource {
        data: ImageData::Base64(uri.into()),
    })
}

fn vid(uri: &str) -> ContentPart {
    ContentPart::Video(VideoSource {
        data: ImageData::Base64(uri.into()),
    })
}

fn message(parts: Vec<ContentPart>) -> Message {
    Message {
        role: Role::User,
        content: parts,
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    }
}

/// Collect and report `(kind, uri)` per item, plus the pad-count length.
fn collect(m: &Message) -> (Vec<(MediaKind, String)>, usize) {
    let mut media = Vec::new();
    let mut pads = Vec::new();
    collect_message_media(
        m,
        &mut media,
        &mut pads,
        &crate::api::chat::remote_image::RemoteImagePolicy::default(),
    )
    .expect("base64 URIs need no fetch policy");
    (
        media.iter().map(|i| (i.kind, i.uri.clone())).collect(),
        pads.len(),
    )
}

#[test]
fn video_before_image_is_collected_in_that_order() {
    // The exact request shape that failed the `video-before-image`
    // gate leg: the client sends the clip first and asks about it.
    // Collection used to walk the content twice — every image, then
    // every video — so the still was handed to the model first while
    // every count still agreed with every other one.
    let m = message(vec![
        vid("clip"),
        img("still"),
        ContentPart::Text("which came first?".into()),
    ]);
    let (media, pads) = collect(&m);
    assert_eq!(
        media,
        vec![
            (MediaKind::Video, "clip".to_string()),
            (MediaKind::Image, "still".to_string()),
        ]
    );
    assert_eq!(pads, 2, "one pad-count slot per media item, in that order");
}

#[test]
fn alternating_media_keeps_every_position() {
    // Three items, alternating: a fix that merely swapped the two
    // groups would still fail this.
    let m = message(vec![vid("v1"), img("i1"), vid("v2"), img("i2")]);
    let (media, _) = collect(&m);
    assert_eq!(
        media.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        vec![
            MediaKind::Video,
            MediaKind::Image,
            MediaKind::Video,
            MediaKind::Image
        ]
    );
}

/// The ordered collection and `MsgEntry::media` are built by two
/// different traversals of the same content, and the splice is only
/// correct while they agree — so pin them against each other.
#[test]
fn collected_order_matches_the_rendered_marker_order() {
    let m = message(vec![
        vid("v1"),
        ContentPart::Text("mid".into()),
        img("i1"),
        vid("v2"),
    ]);
    let (media, _) = collect(&m);
    assert_eq!(
        media.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        m.media_kinds(),
        "the encoder items and the template markers must describe the same sequence"
    );
}

/// Media accumulates ACROSS messages in conversation order too — the
/// pad-count vector is global, so a later message's items must follow
/// an earlier message's.
#[test]
fn media_accumulates_across_messages_in_order() {
    let first = message(vec![vid("v1")]);
    let second = message(vec![img("i1")]);
    let mut media = Vec::new();
    let mut pads = Vec::new();
    let policy = crate::api::chat::remote_image::RemoteImagePolicy::default();
    collect_message_media(&first, &mut media, &mut pads, &policy).expect("first");
    collect_message_media(&second, &mut media, &mut pads, &policy).expect("second");
    assert_eq!(
        media.iter().map(|i| i.kind).collect::<Vec<_>>(),
        vec![MediaKind::Video, MediaKind::Image]
    );
    assert_eq!(pads.len(), 2);
}
