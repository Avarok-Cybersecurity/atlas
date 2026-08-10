// SPDX-License-Identifier: AGPL-3.0-only

//! Sending, cancelling, and the transcript viewport.
//!
//! A third chat file rather than a longer second one: `chat_tests` covers what
//! the reducer does with a delta, and these cover the turn AROUND it — the
//! request that goes out, the state the pane is left in, and the ways a reply
//! can stop. The whole-turn tests drive real HTTP over loopback, because
//! `send` spawns onto a runtime and a fake that skipped that would test a
//! different function.

use std::time::{Duration, Instant};

use super::pump_tests::{chord, key, streaming_state};
use super::*;
use crate::tui::chat_stream::fake::{Fake, serve, sse};

/// A runtime `send` can spawn onto. Multi-threaded on purpose: a
/// current-thread runtime nobody blocks on never polls the spawned task, so
/// the request would never leave.
pub(super) fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .expect("runtime")
}

/// Drive the event loop the way the TUI does, until the reply settles.
pub(super) fn pump_until_settled(s: &mut ChatState) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while s.streaming && Instant::now() < deadline {
        s.pump();
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(!s.streaming, "the reply settled");
}

pub(super) fn body_of(f: &Fake) -> serde_json::Value {
    let req = f.request.recv().expect("the server saw a request");
    serde_json::from_str(req.split("\r\n\r\n").nth(1).expect("a body")).expect("valid JSON")
}

#[test]
fn send_ignores_an_empty_or_whitespace_only_prompt() {
    let mut s = ChatState::default();
    s.send(1);
    assert!(
        s.transcript.is_empty(),
        "Enter on an empty box is not a turn"
    );
    s.input = "  \n\t ".into();
    s.send(1);
    assert!(
        s.transcript.is_empty(),
        "whitespace is not a message either"
    );
    assert_eq!(s.input, "  \n\t ", "and the buffer is left alone");
}

#[test]
fn send_without_a_runtime_says_so_instead_of_leaving_a_blank_reply() {
    // No runtime means no request will ever be made, so the placeholder would
    // sit there forever looking like a model that had stopped answering.
    let mut s = ChatState {
        input: "  hello  ".into(),
        ..ChatState::default()
    };
    s.send(1);
    assert_eq!(s.transcript.len(), 2);
    assert!(s.transcript[0].role == Role::User);
    assert_eq!(s.transcript[0].text, "hello", "the prompt is trimmed");
    let m = &s.transcript[1];
    assert!(m.text.contains("no runtime handle"), "{}", m.text);
    assert!(m.done, "and it is finished, not pending");
    assert!(!s.streaming, "the pane stays usable");
    assert!(s.input.is_empty(), "the box is cleared");
}

#[test]
fn send_refuses_while_a_reply_is_still_streaming() {
    let (mut s, _tx) = streaming_state();
    s.input = "second".into();
    s.send(1);
    assert_eq!(s.transcript.len(), 1, "no second turn was started");
    assert_eq!(
        s.input, "second",
        "the text is kept for when it can be sent"
    );
}

#[test]
fn send_snaps_the_transcript_back_to_the_live_tip() {
    // Sending is an explicit "show me the new reply"; leaving the viewport
    // parked would stream the answer off-screen.
    let mut s = ChatState {
        scroll: Some(12),
        input: "hi".into(),
        ..ChatState::default()
    };
    s.send(1);
    assert_eq!(s.scroll, None);
}

#[test]
fn a_whole_turn_goes_out_over_http_and_lands_back_in_the_transcript() {
    let f = serve(|s| {
        sse(
            s,
            &[
                b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hm\"}}]}\n\n",
                b"data: {\"choices\":[{\"delta\":{\"content\":\"Paris.\"}}]}\n\n",
                b"data: [DONE]\n\n",
            ],
        )
    });
    let rt = runtime();
    let mut s = ChatState::default();
    s.set_runtime(rt.handle().clone());
    s.think_req = ThinkingRequest::On;
    s.input = "capital of France?".into();
    s.send(f.port);
    assert!(
        s.streaming,
        "the pane is busy from the keypress, not from the reply"
    );
    pump_until_settled(&mut s);
    let m = s.transcript.last().expect("a reply");
    assert_eq!(m.text, "Paris.");
    assert_eq!(m.reasoning.text, "hm");
    assert_eq!(m.tokens, 1);
    assert!(m.done);
    assert!(m.ttft_ms.is_some() && m.answer_ttft_ms.is_some());
    assert_eq!(s.observed_thinking, Some(true));
    let body = body_of(&f);
    assert_eq!(
        body["chat_template_kwargs"],
        serde_json::json!({"enable_thinking": true}),
        "the request state at the moment of sending is what goes on the wire"
    );
}

#[test]
fn the_empty_model_placeholder_is_never_sent_back_as_history() {
    // The placeholder exists only so the stream has somewhere to land. On the
    // wire it would be an empty assistant turn, which some templates reject.
    let f = serve(|s| sse(s, &[b"data: [DONE]\n\n"]));
    let rt = runtime();
    let mut s = ChatState::default();
    s.set_runtime(rt.handle().clone());
    s.transcript
        .push(ChatMessage::new(Role::User, "first".into()));
    s.transcript
        .push(ChatMessage::new(Role::Model, "answer".into()));
    s.input = "second".into();
    s.send(f.port);
    pump_until_settled(&mut s);
    assert_eq!(
        body_of(&f)["messages"],
        serde_json::json!([
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second"},
        ])
    );
}

#[test]
fn cancelling_mid_reply_keeps_the_partial_text_and_frees_the_pane() {
    // A cancel arrives as a terminal delta with no measurements at all. It
    // must not read as an error, which would replace what already arrived.
    let f = serve(|s| {
        sse(
            s,
            &[b"data: {\"choices\":[{\"delta\":{\"content\":\"Par\"}}]}\n\n"],
        );
        std::thread::sleep(Duration::from_secs(10));
    });
    let rt = runtime();
    let mut s = ChatState::default();
    s.set_runtime(rt.handle().clone());
    s.input = "capital of France?".into();
    s.send(f.port);
    let deadline = Instant::now() + Duration::from_secs(20);
    while s.transcript.last().expect("a reply").text.is_empty() && Instant::now() < deadline {
        s.pump();
        std::thread::sleep(Duration::from_millis(5));
    }
    s.cancel();
    pump_until_settled(&mut s);
    let m = s.transcript.last().expect("a reply");
    assert_eq!(m.text, "Par", "the partial answer survives the cancel");
    assert!(m.done);
}

#[test]
fn cancel_without_a_running_stream_is_a_no_op() {
    let mut s = ChatState::default();
    s.cancel();
    assert!(!s.streaming);
    assert!(s.transcript.is_empty());
}

#[test]
fn pump_before_anything_has_been_sent_does_nothing() {
    let mut s = ChatState::default();
    s.pump();
    assert!(s.transcript.is_empty());
    assert!(!s.streaming);
}

#[test]
fn an_error_delta_replaces_the_reply_and_ends_the_stream() {
    let (mut s, tx) = streaming_state();
    tx.send(ChatDelta::Error("connect: refused".into()))
        .expect("tx");
    s.pump();
    let m = s.transcript.last().expect("a message");
    assert_eq!(m.text, "(error: connect: refused)");
    assert!(m.done);
    assert!(!s.streaming);
    assert_eq!(
        s.observed_thinking, None,
        "a reply that never started observed nothing about thinking"
    );
}

#[test]
fn deltas_queued_behind_a_terminal_delta_go_with_the_stream() {
    // Everything after the terminal delta belongs to a reply that has ended;
    // appending it would grow a message the footer already summarised.
    let (mut s, tx) = streaming_state();
    tx.send(ChatDelta::Token("a".into())).expect("tx");
    tx.send(ChatDelta::cancelled()).expect("tx");
    tx.send(ChatDelta::Token("b".into())).expect("tx");
    s.pump();
    assert_eq!(s.transcript.last().expect("a message").text, "a");
    assert!(!s.streaming);
    s.pump();
    assert_eq!(
        s.transcript.last().expect("a message").text,
        "a",
        "and pumping a settled pane is inert"
    );
}

#[test]
fn a_disconnect_after_a_partial_reply_keeps_what_arrived() {
    let (mut s, tx) = streaming_state();
    tx.send(ChatDelta::Token("half a sen".into())).expect("tx");
    s.pump();
    drop(tx);
    s.pump();
    assert_eq!(
        s.transcript.last().expect("a message").text,
        "half a sen",
        "a partial answer is more useful than an apology that replaces it"
    );
    assert!(!s.streaming);
}

#[test]
fn a_disconnect_after_reasoning_alone_keeps_the_trace() {
    // Reasoning with no answer is a real observed reply shape; the trace is
    // the only thing there is to show, so it must not be overwritten either.
    let (mut s, tx) = streaming_state();
    tx.send(ChatDelta::Reasoning("mulling".into())).expect("tx");
    s.pump();
    drop(tx);
    s.pump();
    let m = s.transcript.last().expect("a message");
    assert_eq!(m.reasoning.text, "mulling");
    assert!(m.text.is_empty(), "no apology over a trace that did arrive");
    assert_eq!(s.observed_thinking, Some(true));
}

#[test]
fn a_cancel_does_not_restart_the_sealed_thinking_clock() {
    let (mut s, tx) = streaming_state();
    tx.send(ChatDelta::Reasoning("hm".into())).expect("tx");
    s.pump();
    tx.send(ChatDelta::Token("Paris.".into())).expect("tx");
    s.pump();
    let sealed = s.transcript.last().expect("a message").reasoning.think_ms;
    assert!(sealed.is_some(), "the answer sealed it");
    tx.send(ChatDelta::cancelled()).expect("tx");
    s.pump();
    assert_eq!(
        s.transcript.last().expect("a message").reasoning.think_ms,
        sealed,
        "a cancel reports None for everything and must not un-seal a stopped clock"
    );
}

#[test]
fn the_servers_final_counts_replace_the_ones_counted_off_the_wire() {
    // One SSE frame is not one token. The server's counts are authoritative,
    // but only when it actually reported some.
    let (mut s, tx) = streaming_state();
    tx.send(ChatDelta::Reasoning("a".into())).expect("tx");
    tx.send(ChatDelta::Token("b".into())).expect("tx");
    tx.send(ChatDelta::Done {
        ttft_ms: Some(412.0),
        answer_ttft_ms: Some(18_200.0),
        think_ms: Some(17_788.0),
        tok_per_s: Some(12.9),
        tokens: 97,
        reasoning_tokens: 247,
    })
    .expect("tx");
    s.pump();
    let m = s.transcript.last().expect("a message");
    assert_eq!((m.tokens, m.reasoning.tokens), (97, 247));
    assert_eq!(m.ttft_ms, Some(412.0));
    assert_eq!(m.answer_ttft_ms, Some(18_200.0));
    assert_eq!(m.tok_per_s, Some(12.9));
    assert_eq!(m.reasoning.think_ms, Some(17_788.0));
}

#[test]
fn a_terminal_delta_reporting_no_counts_keeps_the_ones_already_counted() {
    let (mut s, tx) = streaming_state();
    tx.send(ChatDelta::Token("a".into())).expect("tx");
    tx.send(ChatDelta::Reasoning("r".into())).expect("tx");
    tx.send(ChatDelta::cancelled()).expect("tx");
    s.pump();
    let m = s.transcript.last().expect("a message");
    assert_eq!((m.tokens, m.reasoning.tokens), (1, 1));
}

#[test]
fn tokens_that_split_a_grapheme_across_deltas_concatenate_intact() {
    // The server frames per token, not per character: CJK and a ZWJ emoji
    // sequence arrive in pieces and must be joined byte-for-byte.
    let (mut s, tx) = streaming_state();
    for part in ["日", "本語 ", "👩\u{200d}", "👧"] {
        tx.send(ChatDelta::Token(part.into())).expect("tx");
    }
    s.pump();
    let m = s.transcript.last().expect("a message");
    assert_eq!(m.text, "日本語 👩\u{200d}👧");
    assert_eq!(m.tokens, 4, "four deltas, whatever they rendered as");
}

#[test]
fn scrolling_clamps_at_the_bottom_and_restores_follow() {
    let mut s = ChatState::default();
    s.scroll_by(-5);
    assert_eq!(s.scroll, None, "already following the tip");
    s.scroll_by(3);
    assert_eq!(s.scroll, Some(3));
    s.scroll_by(-3);
    assert_eq!(
        s.scroll, None,
        "landing exactly at the bottom follows again"
    );
    s.scroll_by(2);
    s.scroll_by(-1000);
    assert_eq!(s.scroll, None, "and overshooting never goes negative");
}

#[test]
fn every_transcript_key_moves_the_viewport() {
    let mut s = ChatState::default();
    s.on_content_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
    assert_eq!(s.scroll, Some(1));
    s.on_content_key(key('k'));
    assert_eq!(s.scroll, Some(2));
    s.on_content_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    assert_eq!(s.scroll, Some(1));
    s.on_content_key(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE));
    assert_eq!(s.scroll, None);
    s.on_content_key(KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE));
    assert_eq!(s.scroll, Some(10));
    s.on_content_key(key('G'));
    assert_eq!(s.scroll, None, "G follows the tip again");
}

#[test]
fn the_view_chords_still_reach_a_focused_transcript() {
    // `on_content_key` falls through to the view keys, so the reasoning
    // toggles work without moving focus back to the input box first.
    let mut s = ChatState::default();
    assert!(s.on_content_key(chord('t', KeyModifiers::ALT)).is_some());
    assert_eq!(s.think_view, ThinkingView::Expanded);
    assert!(s.on_content_key(key('t')).is_some());
    assert_eq!(s.think_req, ThinkingRequest::Off);
}

#[test]
fn a_shifted_t_typed_into_the_input_box_is_text() {
    let mut s = ChatState::default();
    assert!(
        s.on_view_key(key('T'), true).is_none(),
        "typing `T` must type a `T`"
    );
    assert_eq!(s.think_view, ThinkingView::Collapsed);
}

#[test]
fn the_display_toggle_never_forgets_what_was_observed() {
    // Collapsing a trace says nothing about whether the model produced one.
    let mut s = ChatState {
        observed_thinking: Some(true),
        ..ChatState::default()
    };
    assert_eq!(s.cycle_view(), ThinkingView::Expanded);
    assert_eq!(s.observed_thinking, Some(true));
}

#[test]
fn only_a_finished_empty_model_reply_reads_as_answerless() {
    let mut m = ChatMessage::new(Role::Model, String::new());
    assert!(!m.is_answerless(), "still streaming — not the same thing");
    m.done = true;
    assert!(m.is_answerless());
    m.text = "Paris.".into();
    assert!(!m.is_answerless());
    let mut u = ChatMessage::new(Role::User, String::new());
    u.done = true;
    assert!(!u.is_answerless(), "a user turn is never a missing answer");
}
