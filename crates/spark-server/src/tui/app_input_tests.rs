// SPDX-License-Identifier: AGPL-3.0-only

//! Text entry, driven through [`App::on_key`] rather than the buffers directly.
//!
//! Which buffer owns a keystroke is the whole job of this module, so every case
//! here starts from a section and a focus and types.

use super::*;
use crate::tui::app::MainSub;
use crate::tui::chat_thinking::{ThinkingRequest, ThinkingView};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

fn app() -> App {
    App::new(clap::Parser::parse_from(["spark", "org/m"]))
}

fn press(a: &mut App, c: char) {
    a.on_key(KeyEvent::from(KeyCode::Char(c)));
}

fn tap(a: &mut App, code: KeyCode) {
    a.on_key(KeyEvent::from(code));
}

fn chord(a: &mut App, c: char, m: KeyModifiers) {
    a.on_key(KeyEvent::new(KeyCode::Char(c), m));
}

fn type_str(a: &mut App, s: &str) {
    for c in s.chars() {
        press(a, c);
    }
}

/// Terminal ▸ Ops with the input line focused.
fn ops() -> App {
    let mut a = app();
    press(&mut a, '6');
    press(&mut a, 'i');
    a
}

/// Terminal ▸ Chat with the input line focused.
fn chat_input() -> App {
    let mut a = app();
    press(&mut a, '6');
    press(&mut a, '6');
    press(&mut a, 'i');
    a
}

/// Terminal ▸ Chat with the transcript focused.
fn chat_content() -> App {
    let mut a = app();
    press(&mut a, '6');
    press(&mut a, '6');
    a
}

// ── Ops line ────────────────────────────────────────────────────────────────

#[test]
fn the_ops_line_types_and_deletes_and_runs() {
    let mut a = ops();
    type_str(&mut a, "/help");
    assert_eq!(a.ops.input, "/help");
    tap(&mut a, KeyCode::Backspace);
    // Derived rather than spelled out: the literal is a deliberate partial
    // word, and writing it inline trips the spell checker.
    assert_eq!(a.ops.input, &"/help"[.."/help".len() - 1]);
    press(&mut a, 'p');
    tap(&mut a, KeyCode::Enter);
    assert_eq!(a.ops.input, "", "the line is consumed by running it");
    assert_eq!(a.ops.history, ["/help"]);
    assert!(a.ops.output.iter().any(|l| l.contains("/status")));
}

#[test]
fn backspacing_an_empty_ops_line_is_not_an_error() {
    let mut a = ops();
    for _ in 0..3 {
        tap(&mut a, KeyCode::Backspace);
    }
    assert_eq!(a.ops.input, "");
}

#[test]
fn a_blank_ops_line_is_not_recorded_or_run() {
    // Otherwise ⏎ on an empty prompt fills the history with nothing.
    let mut a = ops();
    tap(&mut a, KeyCode::Enter);
    type_str(&mut a, "   ");
    tap(&mut a, KeyCode::Enter);
    assert!(a.ops.history.is_empty());
    assert!(a.ops.output.is_empty());
}

#[test]
fn up_walks_back_through_the_ops_history() {
    let mut a = ops();
    for line in ["/status", "/cache", "/help"] {
        type_str(&mut a, line);
        tap(&mut a, KeyCode::Enter);
    }
    tap(&mut a, KeyCode::Up);
    assert_eq!(a.ops.input, "/help", "the most recent first");
    tap(&mut a, KeyCode::Up);
    assert_eq!(a.ops.input, "/cache");
    tap(&mut a, KeyCode::Up);
    assert_eq!(a.ops.input, "/status");
    tap(&mut a, KeyCode::Up);
    assert_eq!(a.ops.input, "/status", "and stops at the oldest");
}

#[test]
fn up_on_an_empty_history_leaves_the_line_alone() {
    let mut a = ops();
    type_str(&mut a, "half typed");
    tap(&mut a, KeyCode::Up);
    assert_eq!(a.ops.input, "half typed");
}

#[test]
fn the_ops_line_ignores_the_keys_it_has_no_binding_for() {
    let mut a = ops();
    type_str(&mut a, "abc");
    for code in [
        KeyCode::Down,
        KeyCode::Left,
        KeyCode::Right,
        KeyCode::Home,
        KeyCode::End,
        KeyCode::Delete,
        KeyCode::PageUp,
        KeyCode::F(2),
        KeyCode::Tab,
    ] {
        tap(&mut a, code);
    }
    assert_eq!(a.ops.input, "abc", "nothing above edits the line");
    assert!(a.focus == Focus::Input, "and none of them drops focus");
}

// ── Chat line ───────────────────────────────────────────────────────────────

#[test]
fn the_chat_line_sends_on_enter_and_clears() {
    let mut a = chat_input();
    type_str(&mut a, "hello");
    tap(&mut a, KeyCode::Enter);
    assert_eq!(a.chat.input, "");
    assert_eq!(a.chat.transcript.len(), 2, "the turn and its placeholder");
    assert_eq!(a.chat.transcript[0].text, "hello");
    // No tokio handle in a test, so the reply is the refusal rather than a
    // request — which is what keeps this case off the network.
    assert!(a.chat.transcript[1].text.contains("chat unavailable"));
    assert!(!a.chat.streaming);
}

#[test]
fn an_empty_chat_line_sends_nothing() {
    let mut a = chat_input();
    tap(&mut a, KeyCode::Enter);
    type_str(&mut a, "   ");
    tap(&mut a, KeyCode::Enter);
    assert!(a.chat.transcript.is_empty());
}

#[test]
fn a_trailing_backslash_continues_onto_a_new_line_instead_of_sending() {
    // Ctrl+⏎ is indistinguishable from ⏎ in legacy terminal protocols, so it
    // cannot be the only way to write a second line.
    let mut a = chat_input();
    type_str(&mut a, "first\\");
    tap(&mut a, KeyCode::Enter);
    assert_eq!(a.chat.input, "first\n");
    assert!(a.chat.transcript.is_empty(), "nothing was sent");
    type_str(&mut a, "second");
    tap(&mut a, KeyCode::Enter);
    assert_eq!(a.chat.transcript[0].text, "first\nsecond");
}

#[test]
fn esc_leaves_the_chat_line_and_the_transcript_survives() {
    let mut a = chat_input();
    type_str(&mut a, "draft");
    tap(&mut a, KeyCode::Esc);
    assert!(a.focus == Focus::Content);
    assert_eq!(a.chat.input, "draft", "Esc is not a discard here");
    press(&mut a, 'i');
    assert!(a.focus == Focus::Input);
}

#[test]
fn the_chat_line_scrolls_the_transcript_without_losing_focus() {
    // Up/Down are free here (unlike Ops, which spends them on history), and
    // scrollback has to stay live while a reply streams — that is where you are.
    let mut a = chat_input();
    tap(&mut a, KeyCode::Up);
    assert_eq!(a.chat.scroll, Some(1));
    tap(&mut a, KeyCode::PageUp);
    assert_eq!(a.chat.scroll, Some(11));
    tap(&mut a, KeyCode::Down);
    assert_eq!(a.chat.scroll, Some(10));
    tap(&mut a, KeyCode::PageDown);
    assert_eq!(a.chat.scroll, None, "back at the tip, following again");
    assert!(a.focus == Focus::Input);
    assert_eq!(a.chat.input, "", "none of that typed anything");

    tap(&mut a, KeyCode::PageUp);
    tap(&mut a, KeyCode::End);
    assert_eq!(a.chat.scroll, None, "End snaps to the live tip");
}

// ── Unicode ─────────────────────────────────────────────────────────────────

#[test]
fn backspace_deletes_a_whole_character_not_a_byte() {
    // `String::pop` is scalar-wise, so the buffer can never be left holding a
    // partial UTF-8 sequence — the failure mode would be a panic on the next
    // slice, not a visible one.
    for (typed, after_one) in [
        ("héllo", "héll"),
        ("日本語", "日本"),
        ("ok👍", "ok"),
        ("naïve", "naïv"),
    ] {
        let mut a = chat_input();
        type_str(&mut a, typed);
        assert_eq!(a.chat.input, typed);
        tap(&mut a, KeyCode::Backspace);
        assert_eq!(a.chat.input, after_one, "deleting from {typed:?}");
        assert_eq!(
            a.chat.input.chars().count() + 1,
            typed.chars().count(),
            "exactly one character, whatever its byte length"
        );
    }
}

#[test]
fn a_four_byte_emoji_goes_in_one_press_not_four() {
    let mut a = chat_input();
    press(&mut a, '👍');
    assert_eq!(a.chat.input.len(), 4, "four bytes on the wire");
    tap(&mut a, KeyCode::Backspace);
    assert_eq!(a.chat.input, "", "and one press to remove all of them");
}

#[test]
fn a_multi_scalar_grapheme_still_deletes_one_scalar_at_a_time() {
    // KNOWN LIMIT, pinned so it is a decision rather than a surprise: the
    // buffers delete by SCALAR, so a ZWJ sequence or a combining mark takes as
    // many presses as it has scalars. The invariant that does hold is that the
    // buffer stays valid UTF-8 and strictly shrinks.
    let mut a = chat_input();
    type_str(&mut a, "e\u{301}"); // e + combining acute
    assert_eq!(a.chat.input.chars().count(), 2);
    tap(&mut a, KeyCode::Backspace);
    assert_eq!(a.chat.input, "e", "the mark went, the letter stayed");
    tap(&mut a, KeyCode::Backspace);
    assert_eq!(a.chat.input, "");
}

#[test]
fn the_log_filter_takes_unicode_too() {
    let mut a = app();
    press(&mut a, 'f');
    type_str(&mut a, "модель");
    assert_eq!(a.log_filter, "модель");
    tap(&mut a, KeyCode::Backspace);
    assert_eq!(a.log_filter, "модел");
}

// ── Thinking: tri-state, and what it puts on the wire ────────────────────────

/// What the NEXT request would carry for `chat_template_kwargs.enable_thinking`
/// — `None` meaning the key is absent from the body entirely.
fn on_the_wire(a: &App) -> Option<bool> {
    a.chat.think_req.enable_thinking()
}

#[test]
fn ctrl_t_cycles_the_thinking_request_auto_off_on_auto() {
    // The order is load-bearing: Off sits one press from the default because
    // "stop thinking at me" is why anyone reaches for the key.
    let mut a = chat_input();
    assert_eq!(a.chat.think_req, ThinkingRequest::Auto);
    assert_eq!(on_the_wire(&a), None, "Auto sends no key at all");

    chord(&mut a, 't', KeyModifiers::CONTROL);
    assert_eq!(a.chat.think_req, ThinkingRequest::Off);
    assert_eq!(on_the_wire(&a), Some(false));

    chord(&mut a, 't', KeyModifiers::CONTROL);
    assert_eq!(a.chat.think_req, ThinkingRequest::On);
    assert_eq!(on_the_wire(&a), Some(true));

    chord(&mut a, 't', KeyModifiers::CONTROL);
    assert_eq!(a.chat.think_req, ThinkingRequest::Auto);
    assert_eq!(
        on_the_wire(&a),
        None,
        "Auto must OMIT the key, not send a guess at the model's default"
    );
    assert_eq!(a.chat.input, "", "the chord never typed a `t`");
}

#[test]
fn a_bare_t_in_the_chat_line_is_a_letter() {
    let mut a = chat_input();
    type_str(&mut a, "tot");
    assert_eq!(a.chat.input, "tot");
    assert_eq!(a.chat.think_req, ThinkingRequest::Auto, "untouched");
}

#[test]
fn a_bare_t_on_the_transcript_cycles_the_request() {
    // Bare letters are free when the input box does not have focus, so the
    // toggles get their unchorded forms there.
    let mut a = chat_content();
    press(&mut a, 't');
    assert_eq!(a.chat.think_req, ThinkingRequest::Off);
    assert_eq!(a.chat.input, "", "and typed nothing");
}

#[test]
fn alt_t_and_shift_t_move_the_display_only() {
    let mut a = chat_input();
    chord(&mut a, 't', KeyModifiers::ALT);
    assert_eq!(a.chat.think_view, ThinkingView::Expanded);
    chord(&mut a, 't', KeyModifiers::ALT);
    assert_eq!(a.chat.think_view, ThinkingView::Hidden);
    chord(&mut a, 't', KeyModifiers::ALT);
    assert_eq!(a.chat.think_view, ThinkingView::Collapsed);
    assert_eq!(
        on_the_wire(&a),
        None,
        "a view choice never reaches the request"
    );

    let mut a = chat_content();
    press(&mut a, 'T');
    assert_eq!(a.chat.think_view, ThinkingView::Expanded);
    assert_eq!(a.chat.think_req, ThinkingRequest::Auto);
}

#[test]
fn every_thinking_toggle_says_what_it_did() {
    // The state lives in a chip that is easy to miss; a toggle nobody can see
    // fire is indistinguishable from a dead key.
    let mut a = chat_input();
    chord(&mut a, 't', KeyModifiers::CONTROL);
    let said = &a.toasts.last().expect("a toast").text;
    assert!(said.contains("thinking off"), "got {said:?}");
    assert!(
        said.contains("next message"),
        "and says when it takes effect: {said:?}"
    );

    chord(&mut a, 't', KeyModifiers::ALT);
    let said = &a.toasts.last().expect("a toast").text;
    assert!(said.contains("reasoning expanded"), "got {said:?}");
}

#[test]
fn changing_the_request_forgets_what_the_last_reply_did() {
    // `observed_thinking` captions Auto with what actually happened; kept
    // across a change it would caption the new state with the old behaviour.
    let mut a = chat_input();
    a.chat.observed_thinking = Some(true);
    chord(&mut a, 't', KeyModifiers::CONTROL);
    assert_eq!(a.chat.observed_thinking, None);
}

#[test]
fn the_thinking_state_survives_leaving_and_re_entering_the_section() {
    // It is a session preference, not a property of one visit to the pane.
    let mut a = chat_input();
    chord(&mut a, 't', KeyModifiers::CONTROL);
    tap(&mut a, KeyCode::Esc);
    press(&mut a, '1');
    press(&mut a, '6');
    press(&mut a, '6');
    assert_eq!(a.chat.think_req, ThinkingRequest::Off);
    assert_eq!(on_the_wire(&a), Some(false));
}

// ── Routing ─────────────────────────────────────────────────────────────────

#[test]
fn the_log_filter_outranks_every_other_buffer() {
    // `f` can be pressed from either Main subsection, and while it is up the
    // section's own keys must not also fire.
    let mut a = app();
    a.main_sub = MainSub::Kernels;
    a.kernel_scroll_max.set(50);
    press(&mut a, 'f');
    type_str(&mut a, "jjj");
    assert_eq!(a.log_filter, "jjj");
    assert_eq!(a.kernel_scroll, 0, "`j` did not also scroll the table");
}
