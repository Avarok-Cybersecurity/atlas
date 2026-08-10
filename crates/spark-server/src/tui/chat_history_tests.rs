// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-turn history: which transcript entries reach the wire.
//!
//! Split from `chat_more_tests.rs` to keep that file under the 500-line cap.

use super::turn_tests::{body_of, pump_until_settled, runtime};
use super::*;
use crate::tui::chat_stream::fake::{serve, sse};

/// ★ An EARLIER answerless model turn must survive into history.
///
/// Regression: the placeholder was excluded by PROPERTY (`role == Model &&
/// text.is_empty()`) rather than by POSITION. Every cancelled reply is an
/// earlier empty model turn, and so is the documented `response_format` +
/// thinking case — all of them were silently dropped, leaving two consecutive
/// `user` messages on the wire, which some chat templates reject outright.
///
/// Keeping the empty assistant turn preserves the alternation templates need,
/// and it is truthful: the model really did answer nothing that turn.
#[test]
fn a_cancelled_earlier_turn_stays_in_history_so_roles_keep_alternating() {
    let f = serve(|s| sse(s, &[b"data: [DONE]\n\n"]));
    let rt = runtime();
    let mut s = ChatState::default();
    s.set_runtime(rt.handle().clone());
    // Turn 1 answered nothing — a cancel before the first token looks exactly
    // like this.
    s.transcript
        .push(ChatMessage::new(Role::User, "first".into()));
    s.transcript.push(ChatMessage::new(Role::Model, "".into()));
    s.input = "second".into();
    s.send(f.port);
    pump_until_settled(&mut s);

    let sent = &body_of(&f)["messages"];
    assert_eq!(
        sent,
        &serde_json::json!([
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "second"},
        ]),
        "the answerless turn must be preserved, not filtered out"
    );

    // The property that actually breaks templates, asserted directly so the
    // intent survives a rewrite of the expected value above.
    let roles: Vec<&str> = sent
        .as_array()
        .expect("messages is an array")
        .iter()
        .map(|m| m["role"].as_str().expect("role is a string"))
        .collect();
    assert!(
        roles.windows(2).all(|w| w[0] != w[1]),
        "no two consecutive turns may share a role: {roles:?}"
    );
}

/// The placeholder for THIS send is still excluded — it is the last element.
#[test]
fn the_placeholder_for_the_current_send_is_still_excluded() {
    let f = serve(|s| sse(s, &[b"data: [DONE]\n\n"]));
    let rt = runtime();
    let mut s = ChatState::default();
    s.set_runtime(rt.handle().clone());
    s.input = "only".into();
    s.send(f.port);
    pump_until_settled(&mut s);
    assert_eq!(
        body_of(&f)["messages"],
        serde_json::json!([{"role": "user", "content": "only"}]),
        "the just-pushed empty model placeholder must not be sent"
    );
}
