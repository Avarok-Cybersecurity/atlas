// SPDX-License-Identifier: AGPL-3.0-only

//! Column accounting in the chat pane.
//!
//! The pane slices its viewport on the rows `wrap_rows` returns, so a row that
//! is a column wider than it claims does not merely look wrong — it shifts
//! every cell after it and corrupts the frame. Everything here is asserted in
//! DISPLAY COLUMNS (`unicode-width`), never in bytes or `char`s, on the text
//! that makes the three disagree: CJK, emoji, ZWJ sequences and combining
//! marks.

use ratatui::Terminal;
use ratatui::backend::TestBackend;

use super::*;
use crate::tui::app::{App, Focus, Section, TermSub};

/// A CJK sentence, a ZWJ family, a skin-toned emoji, and a decomposed accent —
/// bytes, chars and columns all differ, and by different ratios.
const WIDE: &str = "日本語のテキスト 👨‍👩‍👧‍👦 👋🏽 cafe\u{0301} ﬁn";

fn columns(s: &str) -> usize {
    UnicodeWidthStr::width(s)
}

fn chat_app(text: &str, reasoning: &str) -> App {
    let mut a = crate::tui::render::tests::app();
    a.section = Section::Terminal;
    a.term_sub = TermSub::Chat;
    a.focus = Focus::Content;
    let mut m = ChatMessage::new(Role::Model, text.to_string());
    if !reasoning.is_empty() {
        m.reasoning.begin();
        m.reasoning.text = reasoning.to_string();
        m.reasoning.tokens = 42;
    }
    m.done = true;
    a.chat.transcript = vec![ChatMessage::new(Role::User, WIDE.to_string()), m];
    a.chat.think_view = ThinkingView::Expanded;
    a
}

/// The frame as raw cell symbols, one `Vec` per row — untrimmed, because the
/// question being asked is how many columns each row actually consumed.
fn cells(app: &App, w: u16, h: u16) -> Vec<Vec<String>> {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).expect("backend");
    terminal
        .draw(|f| crate::tui::render::draw(f, app))
        .expect("draw");
    let buf = terminal.backend().buffer();
    (0..h)
        .map(|y| (0..w).map(|x| buf[(x, y)].symbol().to_string()).collect())
        .collect()
}

#[test]
fn no_wrapped_row_is_wider_than_the_pane_it_was_measured_against() {
    // From two columns up: a single two-column glyph cannot be fitted into a
    // one-column pane by any wrapping, so that case is its own test below.
    let text = WIDE.repeat(12);
    for width in [2usize, 3, 7, 20, 41, 80] {
        for row in wrap_rows(&text, width) {
            assert!(
                columns(&row) <= width,
                "{} columns in a {width}-column pane: {row:?}",
                columns(&row)
            );
        }
    }
}

#[test]
fn a_one_column_pane_holds_at_most_one_glyph_per_row_and_terminates() {
    let rows = wrap_rows(WIDE, 1);
    assert!(!rows.is_empty());
    for row in &rows {
        assert!(
            row.chars()
                .filter(|c| UnicodeWidthChar::width(*c).unwrap_or(0) > 0)
                .count()
                <= 1,
            "more than one visible glyph in a one-column row: {row:?}"
        );
    }
}

#[test]
fn a_wide_glyph_that_would_straddle_the_last_column_moves_down_instead() {
    // Odd widths are where a two-column glyph has nowhere to sit: the row must
    // end a column short rather than overhang by one.
    let rows = wrap_rows(&"日".repeat(20), 5);
    assert!(!rows.is_empty());
    for row in &rows {
        assert!(columns(row) <= 4, "{row:?} in a 5-column pane");
    }
}

#[test]
fn a_combining_mark_stays_with_the_letter_it_modifies() {
    // The mark is zero-width, so it must never be the character that triggers
    // a break — a row starting with a bare accent is a rendering artefact.
    let rows = wrap_rows(&"cafe\u{0301} ".repeat(10), 5);
    for row in &rows {
        assert!(
            !row.starts_with('\u{0301}'),
            "a row began with a combining mark: {row:?}"
        );
        assert!(columns(row) <= 5, "{row:?}");
    }
}

#[test]
fn a_five_hundred_character_url_is_split_rather_than_left_to_overhang() {
    let url = format!("https://example.invalid/{}", "x".repeat(480));
    assert_eq!(url.chars().count(), 504);
    let rows = wrap_rows(&url, 40);
    for row in &rows {
        assert!(columns(row) <= 40, "{} columns", columns(row));
    }
    assert_eq!(
        rows.concat(),
        url,
        "hard-splitting must not drop or duplicate a character"
    );
}

#[test]
fn explicit_newlines_start_new_rows_even_when_everything_fits() {
    assert_eq!(wrap_rows("a\nb\nc", 40), vec!["a", "b", "c"]);
    // A trailing newline is a real empty row, not a rounding artefact.
    assert_eq!(wrap_rows("a\n", 40), vec!["a", ""]);
}

#[test]
fn the_reasoning_header_clips_wide_text_at_a_glyph_boundary() {
    // The header is the one unwrappable row in the pane, so it clips — and a
    // clip measured in bytes would cut a multi-byte glyph in half.
    let mut m = ChatMessage::new(Role::Model, "answer".into());
    m.reasoning.begin();
    m.reasoning.text = WIDE.into();
    m.reasoning.tokens = 1234;
    m.reasoning.think_ms = Some(18_200.0);
    m.done = true;
    for body_w in [8usize, 20, 34, 60] {
        for line in message_lines(&m, false, ThinkingView::Collapsed, 0, body_w) {
            let w: usize = line.spans.iter().map(|s| columns(&s.content)).sum();
            assert!(w <= (body_w + 4).max(7), "{w} columns over {body_w}");
        }
    }
}

#[test]
fn no_message_row_overhangs_its_body_for_text_of_undecidable_width() {
    let mut m = ChatMessage::new(Role::Model, WIDE.repeat(6));
    m.reasoning.begin();
    m.reasoning.text = WIDE.repeat(6);
    m.reasoning.tokens = 99;
    m.tokens = 120;
    m.ttft_ms = Some(412.0);
    m.tok_per_s = Some(12.887);
    m.done = true;
    for body_w in [1usize, 7, 20, 40, 69] {
        for view in [ThinkingView::Collapsed, ThinkingView::Expanded] {
            for line in message_lines(&m, true, view, 0, body_w) {
                let w: usize = line.spans.iter().map(|s| columns(&s.content)).sum();
                // gutter (2) + rule (1) + body + streaming cursor (1), floored
                // at the header's unwrappable lead.
                assert!(w <= (body_w + 4).max(7), "{w} columns over {body_w}");
            }
        }
    }
}

#[test]
fn wide_glyphs_do_not_shift_the_frame_that_encloses_them() {
    // The corruption this guards against is silent: a row that consumed more
    // columns than it claimed pushes the panel border sideways. So the border
    // column is compared against the same frame drawn with plain ASCII — the
    // transcript may differ, the chrome around it may not.
    let ascii = chat_app(&"paris ".repeat(60), &"thinking ".repeat(60));
    let wide = chat_app(&WIDE.repeat(8), &WIDE.repeat(8));
    for (w, h) in [(20u16, 12u16), (40, 20), (81, 24), (160, 48)] {
        let edge = |a: &App| -> Vec<String> {
            cells(a, w, h)
                .into_iter()
                .map(|row| row[w as usize - 1].clone())
                .collect()
        };
        assert_eq!(
            edge(&wide).len(),
            h as usize,
            "{w}x{h} drew a partial frame"
        );
        assert_eq!(
            edge(&wide),
            edge(&ascii),
            "the right-hand border moved at {w}x{h}"
        );
    }
}

#[test]
fn no_double_width_glyph_is_left_hanging_off_the_last_column() {
    let a = chat_app(&WIDE.repeat(8), &WIDE.repeat(8));
    for (w, h) in [(21u16, 12u16), (40, 20), (81, 24)] {
        for (y, row) in cells(&a, w, h).into_iter().enumerate() {
            let last = &row[w as usize - 1];
            assert!(
                columns(last) <= 1,
                "row {y} ends with a {last:?} that needs two columns at {w}x{h}"
            );
        }
    }
}

#[test]
fn a_streaming_reasoning_block_keeps_its_shape_while_the_spinner_turns() {
    // Only the spinner glyph may change between ticks: a block whose height
    // moved would drag the answer up and down under the reader's eyes.
    let mut m = ChatMessage::new(Role::Model, String::new());
    m.reasoning.begin();
    m.reasoning.text = WIDE.repeat(4);
    m.reasoning.tokens = 40;
    let shape = |tick: u64| -> Vec<usize> {
        message_lines(&m, true, ThinkingView::Collapsed, tick, 60)
            .iter()
            .map(|l| l.spans.iter().map(|s| columns(&s.content)).sum())
            .collect()
    };
    let first = shape(0);
    for tick in 1..12 {
        assert_eq!(shape(tick), first, "the block moved on tick {tick}");
    }
    assert!(
        (1..12).any(|t| {
            message_lines(&m, true, ThinkingView::Collapsed, t, 60)[0]
                != message_lines(&m, true, ThinkingView::Collapsed, 0, 60)[0]
        }),
        "but it must visibly move — a frozen spinner reads as a hang"
    );
}

#[test]
fn a_live_reasoning_trace_shows_its_tail_not_its_whole_history() {
    // Collapsed and streaming keeps a bounded preview, so a thousand-token
    // ramble cannot own the pane.
    let mut m = ChatMessage::new(Role::Model, String::new());
    m.reasoning.begin();
    m.reasoning.text = (0..200)
        .map(|i| format!("step-{i}"))
        .collect::<Vec<_>>()
        .join("\n");
    let lines = message_lines(&m, true, ThinkingView::Collapsed, 0, 60);
    let text: Vec<String> = lines
        .iter()
        .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
        .collect();
    assert!(lines.len() <= 8, "header + preview only: {text:#?}");
    assert!(
        text.iter().any(|r| r.contains("step-199")),
        "the newest reasoning is what is shown: {text:#?}"
    );
    assert!(
        !text.iter().any(|r| r.contains("step-0")),
        "and the oldest is not: {text:#?}"
    );
}

#[test]
fn a_reasoning_block_that_is_still_streaming_draws_no_empty_answer_rule() {
    // There is no answer yet, and a bare cyan rule under the thinking block
    // would just be a stray glyph.
    let mut m = ChatMessage::new(Role::Model, String::new());
    m.reasoning.begin();
    m.reasoning.text = "weighing the options".into();
    let lines = message_lines(&m, true, ThinkingView::Collapsed, 0, 60);
    let last: String = lines
        .last()
        .expect("rows")
        .spans
        .iter()
        .map(|s| s.content.as_ref())
        .collect();
    assert!(!last.contains('▏'), "no answer rule yet: {last:?}");
    assert!(last.ends_with('▍'), "the cursor rides the trace: {last:?}");
}
