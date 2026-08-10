// SPDX-License-Identifier: AGPL-3.0-only

//! What the chat pane actually draws, read back off a `TestBackend`.
//!
//! Widget snapshots rather than assertions on the intermediate `Vec<Line>`:
//! the bug this module exists to fix was invisible to every unit test in the
//! tree because the pane rendered a perfectly valid EMPTY screen for 18
//! seconds. Only the finished buffer can say "there is nothing there".

use ratatui::Terminal;
use ratatui::backend::TestBackend;

use super::*;
use crate::tui::app::{App, Focus, Section, TermSub};
use crate::tui::render::tests::app;

/// The rendered frame, one `String` per row, trailing blanks trimmed.
fn screen(app: &App, w: u16, h: u16) -> Vec<String> {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).expect("backend");
    terminal
        .draw(|f| crate::tui::render::draw(f, app))
        .expect("draw");
    let buf = terminal.backend().buffer();
    (0..h)
        .map(|y| {
            (0..w)
                .map(|x| buf[(x, y)].symbol())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect()
}

fn has(rows: &[String], needle: &str) -> bool {
    rows.iter().any(|r| r.contains(needle))
}

fn chat_app(transcript: Vec<ChatMessage>, streaming: bool, view: ThinkingView) -> App {
    let mut a = app();
    a.section = Section::Terminal;
    a.term_sub = TermSub::Chat;
    a.focus = Focus::Content;
    a.chat.transcript = transcript;
    a.chat.streaming = streaming;
    a.chat.think_view = view;
    a
}

fn user(text: &str) -> ChatMessage {
    ChatMessage::new(Role::User, text.to_string())
}

/// A reply mid-thought: reasoning arriving, nothing readable yet.
fn thinking_now() -> ChatMessage {
    let mut m = ChatMessage::new(Role::Model, String::new());
    m.reasoning.begin();
    m.reasoning.text = "The user wants the capital of France. Answer plainly.".into();
    m.reasoning.tokens = 12;
    m
}

/// A finished reply that thought first.
fn thought_then_answered() -> ChatMessage {
    let mut m = thinking_now();
    m.text = "The capital of France is Paris.".into();
    m.tokens = 312;
    m.reasoning.tokens = 247;
    m.reasoning.think_ms = Some(18_200.0);
    m.ttft_ms = Some(412.0);
    m.answer_ttft_ms = Some(18_600.0);
    m.tok_per_s = Some(12.887);
    m.done = true;
    m
}

#[test]
fn streaming_reasoning_is_visible_and_labelled_while_it_streams() {
    // The measured bug: 197-245 reasoning deltas streamed into a pane that
    // showed NOTHING for ~18 s, and the user read it as a hang.
    let rows = screen(
        &chat_app(
            vec![user("capital of France?"), thinking_now()],
            true,
            ThinkingView::Collapsed,
        ),
        100,
        30,
    );
    assert!(
        rows.iter()
            .any(|r| r.contains("thinking") && r.contains(theme::SPINNER[0])),
        "one header row, saying what is happening AND visibly moving: {rows:#?}"
    );
    assert!(
        has(&rows, "Answer plainly"),
        "the reasoning text itself is on screen, not just a placeholder"
    );
    assert!(has(&rows, "┆"), "behind the dashed reasoning rule");
    assert!(
        !has(&rows, "thought for"),
        "nothing is summarised until it is over"
    );
}

#[test]
fn a_finished_reply_collapses_to_one_quiet_line() {
    let rows = screen(
        &chat_app(
            vec![user("capital of France?"), thought_then_answered()],
            false,
            ThinkingView::Collapsed,
        ),
        100,
        30,
    );
    assert!(
        has(&rows, "▸ thought for 18.2s · 247 tokens"),
        "one line, and it says how long: {rows:#?}"
    );
    assert!(
        !has(&rows, "Answer plainly"),
        "the trace itself is out of the way — the answer dominates"
    );
    assert!(has(&rows, "The capital of France is Paris."));
    // The honest footer: `ttft` measures the first token of ANY kind, and the
    // separate `answer` number is the one that explains the blank pane.
    assert!(has(&rows, "ttft 412ms"), "{rows:#?}");
    assert!(has(&rows, "answer 18.6s"));
    assert!(has(&rows, "247 think + 312 tok"));
    assert!(has(&rows, "77.6 ms/tok"));
}

#[test]
fn expanded_puts_the_whole_trace_back() {
    let rows = screen(
        &chat_app(
            vec![user("capital of France?"), thought_then_answered()],
            false,
            ThinkingView::Expanded,
        ),
        100,
        30,
    );
    assert!(has(&rows, "▾ thought for 18.2s"), "the caret points down");
    assert!(has(&rows, "Answer plainly"), "and the trace is back");
    assert!(has(&rows, "The capital of France is Paris."));
}

#[test]
fn hidden_draws_no_reasoning_chrome_at_all() {
    let rows = screen(
        &chat_app(
            vec![user("capital of France?"), thought_then_answered()],
            false,
            ThinkingView::Hidden,
        ),
        100,
        30,
    );
    assert!(!has(&rows, "thought for"));
    assert!(!has(&rows, "┆"));
    assert!(has(&rows, "The capital of France is Paris."), "{rows:#?}");
}

#[test]
fn a_reply_with_no_reasoning_renders_exactly_as_it_always_did() {
    // No empty header, no wasted row, and — because there is nothing to draw —
    // the display toggle must be a no-op on it.
    let mut m = ChatMessage::new(Role::Model, "Paris.".into());
    m.tokens = 2;
    m.ttft_ms = Some(88.0);
    m.answer_ttft_ms = Some(88.0);
    m.tok_per_s = Some(12.887);
    m.done = true;
    // Rendered rows for all three view states must be IDENTICAL — compared on
    // the lines rather than the frame, because the frame carries a header
    // clock that ticks between two renders.
    let collapsed = message_lines(&m, false, ThinkingView::Collapsed, 0, 69);
    for view in [ThinkingView::Expanded, ThinkingView::Hidden] {
        assert_eq!(
            collapsed,
            message_lines(&m, false, view, 0, 69),
            "a view preference cannot change a reply that has no reasoning"
        );
    }
    let plain = screen(
        &chat_app(vec![user("capital?"), m], false, ThinkingView::Collapsed),
        100,
        30,
    );
    // The chip in the panel title legitimately contains the word "thinking",
    // so absence is asserted on the block's own glyphs.
    assert!(!has(&plain, "thought"), "{plain:#?}");
    assert!(!has(&plain, "┆"), "no dashed rule, no wasted row");
    assert!(has(&plain, "Paris."));
    assert!(has(&plain, "ttft 88ms · 2 tok · 77.6 ms/tok"), "{plain:#?}");
}

#[test]
fn a_reply_that_never_answered_says_so_instead_of_looking_broken() {
    // `response_format` + thinking on returned all reasoning and no content on
    // 2 of 4 requests. The pane must not look like the bug is in the pane.
    let mut m = thinking_now();
    m.reasoning.tokens = 247;
    m.reasoning.think_ms = Some(18_200.0);
    m.ttft_ms = Some(412.0);
    m.tok_per_s = Some(13.57);
    m.done = true;
    let rows = screen(
        &chat_app(
            vec![user("give me JSON"), m],
            false,
            ThinkingView::Collapsed,
        ),
        100,
        30,
    );
    assert!(
        has(&rows, "no answer"),
        "the user is told, in as many words: {rows:#?}"
    );
    assert!(
        has(&rows, "thought for 18.2s"),
        "and the reasoning it did produce is still reachable"
    );
}

#[test]
fn the_thinking_request_state_is_always_on_screen() {
    let mut a = chat_app(vec![], false, ThinkingView::Collapsed);
    assert!(has(&screen(&a, 100, 30), "thinking auto"), "the default");
    // Auto claims nothing until a reply has been observed.
    assert!(!has(&screen(&a, 100, 30), "auto ("));
    a.chat.observed_thinking = Some(true);
    assert!(has(&screen(&a, 100, 30), "thinking auto (thinking)"));
    a.chat.cycle_request();
    let off = screen(&a, 100, 30);
    assert!(has(&off, "thinking off"), "{off:#?}");
    assert!(
        !has(&off, "thinking off ("),
        "an explicit state is not an observation"
    );
}

#[test]
fn the_pane_survives_the_eighty_by_twentyfour_floor() {
    // A hard-split word behind a reasoning rule is where an off-by-one
    // overhang shows up first.
    let long = "supercalifragilisticexpialidocious ".repeat(20);
    for (w, h) in [(80u16, 24u16), (40, 12)] {
        let mut m = thinking_now();
        m.reasoning.text = long.clone();
        let rows = screen(
            &chat_app(vec![user("hi"), m], true, ThinkingView::Expanded),
            w,
            h,
        );
        assert_eq!(rows.len(), h as usize, "it drew a whole screen");
    }
}

#[test]
fn no_row_overhangs_the_body_it_was_wrapped_to() {
    let mut m = thought_then_answered();
    m.reasoning.text = "supercalifragilisticexpialidocious ".repeat(8);
    m.text = "antidisestablishmentarianism ".repeat(8);
    for body_w in [1usize, 7, 20, 69] {
        for line in message_lines(&m, true, ThinkingView::Expanded, 0, body_w) {
            let w: usize = line
                .spans
                .iter()
                .map(|s| UnicodeWidthStr::width(s.content.as_ref()))
                .sum();
            // gutter (2) + rule (1) + body + streaming cursor (1). The floor
            // of 7 is the thinking header's unwrappable lead — `⬢ ┆ ▾ …` —
            // which clips rather than reflows and which no real pane is
            // narrow enough to reach.
            let cap = (body_w + 4).max(7);
            assert!(w <= cap, "{w} columns over a {body_w}-wide body");
        }
    }
}

#[test]
fn a_zero_width_pane_neither_panics_nor_loops() {
    let m = thought_then_answered();
    let lines = message_lines(&m, false, ThinkingView::Expanded, 0, 0);
    assert!(!lines.is_empty());
}
