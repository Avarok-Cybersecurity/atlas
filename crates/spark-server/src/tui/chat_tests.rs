// SPDX-License-Identifier: AGPL-3.0-only

//! What the chat reducer does with deltas and view keys.
//!
//! Split from `chat.rs` at the per-file cap.

use std::sync::mpsc::Sender;

use super::*;

pub(super) fn streaming_state() -> (ChatState, Sender<ChatDelta>) {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut s = ChatState::default();
    s.transcript
        .push(ChatMessage::new(Role::Model, String::new()));
    s.streaming = true;
    s.rx = Some(rx);
    (s, tx)
}

pub(super) fn key(c: char) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
}

pub(super) fn chord(c: char, m: KeyModifiers) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(c), m)
}

#[test]
fn a_sender_that_dies_without_finishing_unsticks_the_pane() {
    // `send` refuses while `streaming` is set, so a dropped sender used to
    // lock the chat pane for the rest of the process.
    let (mut s, tx) = streaming_state();
    drop(tx);
    s.pump();
    assert!(!s.streaming, "the pane is usable again");
    assert!(s.rx.is_none());
    assert!(
        s.transcript
            .last()
            .expect("a message")
            .text
            .contains("without finishing"),
        "and says why"
    );
}

#[test]
fn a_delta_racing_the_disconnect_check_is_not_dropped() {
    // The check calls try_recv once and KEEPS what it gets: a token can
    // arrive between `try_iter` ending and that call.
    let (mut s, tx) = streaming_state();
    tx.send(ChatDelta::Token("hi".into())).expect("send");
    s.pump();
    assert!(s.streaming, "still streaming");
    assert_eq!(s.transcript.last().expect("a message").text, "hi");
}

#[test]
fn a_live_channel_with_nothing_pending_is_left_alone() {
    let (mut s, _tx) = streaming_state();
    s.pump();
    assert!(s.streaming, "empty is not dead");
    assert!(s.rx.is_some());
}

#[test]
fn reasoning_deltas_land_in_the_reasoning_half_not_the_answer() {
    // The bug: nothing in the TUI looked at `reasoning_content`, so 197-245
    // deltas streamed into a pane that stayed blank for ~18 s.
    let (mut s, tx) = streaming_state();
    tx.send(ChatDelta::Reasoning("mulling".into())).expect("tx");
    tx.send(ChatDelta::Reasoning(" it over".into()))
        .expect("tx");
    tx.send(ChatDelta::Token("Paris.".into())).expect("tx");
    s.pump();
    let m = s.transcript.last().expect("a message");
    assert_eq!(m.reasoning.text, "mulling it over");
    assert_eq!(m.reasoning.tokens, 2);
    assert_eq!(m.text, "Paris.", "the answer is uncontaminated");
    assert!(
        m.reasoning.seconds().is_some(),
        "the live timer started on the first reasoning delta"
    );
}

#[test]
fn the_thinking_clock_stops_when_the_answer_starts() {
    // Reasoning that keeps arriving after the answer began is normal; the
    // summary must not go on counting up under a reply already being read.
    let (mut s, tx) = streaming_state();
    tx.send(ChatDelta::Reasoning("mulling".into())).expect("tx");
    s.pump();
    tx.send(ChatDelta::Token("Paris.".into())).expect("tx");
    s.pump();
    let sealed = s
        .transcript
        .last()
        .expect("a message")
        .reasoning
        .seconds()
        .expect("a span");
    tx.send(ChatDelta::Reasoning(" more".into())).expect("tx");
    s.pump();
    assert_eq!(
        s.transcript.last().expect("a message").reasoning.seconds(),
        Some(sealed),
        "frozen at the moment the answer landed"
    );
}

#[test]
fn a_reply_that_only_thought_is_marked_answerless() {
    let (mut s, tx) = streaming_state();
    tx.send(ChatDelta::Reasoning("thinking".into()))
        .expect("tx");
    tx.send(ChatDelta::Done {
        ttft_ms: Some(412.0),
        answer_ttft_ms: None,
        think_ms: Some(18_200.0),
        tok_per_s: Some(12.9),
        tokens: 0,
        reasoning_tokens: 247,
    })
    .expect("tx");
    s.pump();
    let m = s.transcript.last().expect("a message");
    assert!(
        m.is_answerless(),
        "the pane must say so rather than look hung"
    );
    assert_eq!(m.reasoning.tokens, 247);
    assert_eq!(m.reasoning.seconds(), Some(18.2));
    assert_eq!(s.observed_thinking, Some(true));
}

#[test]
fn a_reply_with_no_reasoning_records_the_observation_too() {
    let (mut s, tx) = streaming_state();
    tx.send(ChatDelta::Token("Paris.".into())).expect("tx");
    tx.send(ChatDelta::Done {
        ttft_ms: Some(88.0),
        answer_ttft_ms: Some(88.0),
        think_ms: None,
        tok_per_s: Some(12.9),
        tokens: 1,
        reasoning_tokens: 0,
    })
    .expect("tx");
    s.pump();
    let m = s.transcript.last().expect("a message");
    assert!(m.reasoning.is_empty(), "nothing to draw");
    assert!(!m.is_answerless());
    assert_eq!(s.observed_thinking, Some(false));
}

#[test]
fn a_cancelled_reply_keeps_its_partial_text_and_claims_no_observation() {
    let (mut s, tx) = streaming_state();
    tx.send(ChatDelta::cancelled()).expect("tx");
    s.pump();
    assert!(!s.streaming);
    assert_eq!(
        s.transcript.last().expect("a message").text,
        "",
        "a cancel is not an error and must not overwrite the reply"
    );
    assert_eq!(
        s.observed_thinking, None,
        "a reply that never started observed nothing"
    );
}

#[test]
fn the_request_toggle_cycles_and_forgets_the_stale_observation() {
    let mut s = ChatState {
        observed_thinking: Some(true),
        ..ChatState::default()
    };
    assert_eq!(s.think_req, ThinkingRequest::Auto);
    assert!(s.on_view_key(key('t'), false).is_some());
    assert_eq!(s.think_req, ThinkingRequest::Off);
    assert_eq!(
        s.observed_thinking, None,
        "the old reply does not describe the new request"
    );
    s.on_view_key(key('t'), false);
    assert_eq!(s.think_req, ThinkingRequest::On);
    s.on_view_key(key('t'), false);
    assert_eq!(s.think_req, ThinkingRequest::Auto);
}

#[test]
fn a_bare_t_while_typing_is_text_and_the_chords_still_work() {
    let mut s = ChatState::default();
    assert!(
        s.on_view_key(key('t'), true).is_none(),
        "typing `t` must type a `t`"
    );
    assert_eq!(s.think_req, ThinkingRequest::Auto);
    assert!(
        s.on_view_key(chord('t', KeyModifiers::CONTROL), true)
            .is_some()
    );
    assert_eq!(s.think_req, ThinkingRequest::Off);
    assert!(s.on_view_key(chord('t', KeyModifiers::ALT), true).is_some());
    assert_eq!(s.think_view, ThinkingView::Expanded);
    assert_eq!(
        s.think_req,
        ThinkingRequest::Off,
        "the display key never touches the wire"
    );
}

#[test]
fn shift_t_cycles_the_view_without_touching_the_request() {
    let mut s = ChatState::default();
    assert!(s.on_view_key(key('T'), false).is_some());
    assert_eq!(s.think_view, ThinkingView::Expanded);
    s.on_view_key(key('T'), false);
    assert_eq!(s.think_view, ThinkingView::Hidden);
    s.on_view_key(key('T'), false);
    assert_eq!(s.think_view, ThinkingView::Collapsed);
    assert_eq!(s.think_req, ThinkingRequest::Auto);
}

#[test]
fn transcript_keys_still_scroll_and_unknown_keys_fall_through() {
    let mut s = ChatState::default();
    s.on_content_key(KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE));
    assert_eq!(s.scroll, Some(10));
    s.on_content_key(key('j'));
    assert_eq!(s.scroll, Some(9));
    s.on_content_key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE));
    assert_eq!(s.scroll, None, "End follows the tip again");
    assert!(s.on_content_key(key('z')).is_none());
}
