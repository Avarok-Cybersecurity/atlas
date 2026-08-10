// SPDX-License-Identifier: AGPL-3.0-only

//! The Terminal section: the Ops REPL and the Chat pane it tab-switches with.
//!
//! Chat's transcript rendering is covered next door in `chat_lines_tests`;
//! what is asserted here is the frame around it — the title chips, the input
//! box, and the hints that change with focus and width.

use super::super::harness::{has, screen};
use super::chat_hints;
use crate::tui::app::{App, Focus, Section, TermSub};

fn term(sub: TermSub) -> App {
    let mut a = crate::tui::render::tests::app();
    a.section = Section::Terminal;
    a.term_sub = sub;
    a
}

#[test]
fn the_tab_strip_marks_which_of_the_two_panes_is_showing() {
    let ops = screen(&term(TermSub::Ops), 120, 40);
    assert!(has(&ops, "Ops"), "{ops:#?}");
    assert!(has(&ops, "Chat"));
    assert!(has(&ops, "(6 toggles)"), "and says how to swap them");
    assert!(has(&ops, "OPS ─ 0 lines"), "Ops is the pane on screen");

    let chat = screen(&term(TermSub::Chat), 120, 40);
    assert!(has(&chat, "CHAT ─"), "{chat:#?}");
    assert!(!has(&chat, "OPS ─"));
}

#[test]
fn an_unfocused_ops_input_says_how_to_take_the_keyboard() {
    let rows = screen(&term(TermSub::Ops), 120, 40);
    assert!(has(&rows, "(Enter to focus · /help)"), "{rows:#?}");
}

#[test]
fn a_focused_ops_input_ghosts_the_rest_of_the_command_it_can_finish() {
    let mut a = term(TermSub::Ops);
    a.focus = Focus::Input;
    a.ops.input = "/ker".into();
    let rows = screen(&a, 120, 40);
    assert!(
        has(&rows, "/kernels"),
        "the completion is shown:\n{rows:#?}"
    );
    assert!(has(&rows, "⇥ accept"), "and how to take it:\n{rows:#?}");

    // A command with nothing left to complete gets a cursor instead of a
    // ghost, so the line never looks like it is waiting on a suggestion.
    a.ops.input = "/quit".into();
    let done = screen(&a, 120, 40);
    assert!(!has(&done, "⇥ accept"), "{done:#?}");
    assert!(has(&done, "❯ /quit▏"), "{done:#?}");
}

#[test]
fn the_ops_pane_keeps_the_newest_output_when_it_overflows() {
    let mut a = term(TermSub::Ops);
    a.ops.output = (0..200).map(|i| format!("line-{i}")).collect();
    let rows = screen(&a, 120, 40);
    assert!(has(&rows, "OPS ─ 200 lines"), "{rows:#?}");
    assert!(
        has(&rows, "line-199"),
        "the newest is on screen:\n{rows:#?}"
    );
    assert!(!has(&rows, "line-0 "), "the oldest is not:\n{rows:#?}");
}

#[test]
fn an_echoed_command_is_marked_apart_from_the_output_it_produced() {
    let mut a = term(TermSub::Ops);
    a.ops.output = vec!["❯ /gpu".into(), "atlas 57.2 GB".into()];
    let rows = screen(&a, 120, 40);
    assert!(has(&rows, "❯ /gpu"), "{rows:#?}");
    assert!(has(&rows, "atlas 57.2 GB"));
}

#[test]
fn the_chat_hints_shrink_with_the_pane_but_never_vanish() {
    assert!(chat_hints(true, true).contains("Ctrl+T"));
    assert!(chat_hints(true, true).contains("newline"));
    // The narrow forms drop the toggles and keep the two keys that matter.
    assert_eq!(chat_hints(true, false), "─ ⏎ send · Esc cancel ─");
    assert_eq!(chat_hints(false, false), "─ ⏎ focus ─");
    assert!(chat_hints(false, true).contains("t thinking"));
    for wide in [true, false] {
        for focused in [true, false] {
            assert!(chat_hints(focused, wide).contains('⏎'));
        }
    }
}

#[test]
fn the_chat_title_names_the_model_and_the_thinking_state_it_will_ask_for() {
    let a = term(TermSub::Chat);
    let rows = screen(&a, 120, 40);
    assert!(has(&rows, "CHAT ─ nvidia/Qwen3.6-27B-NVFP4"), "{rows:#?}");
    assert!(has(&rows, "thinking auto"), "{rows:#?}");
}

#[test]
fn the_chat_title_reports_streaming_and_how_far_back_it_is_scrolled() {
    let mut a = term(TermSub::Chat);
    a.chat.streaming = true;
    assert!(has(&screen(&a, 120, 40), "streaming"));

    // Scrolled back, the title stops claiming to be live and says how to
    // return — the two states are mutually exclusive on purpose.
    a.chat.scroll = Some(12);
    let scrolled = screen(&a, 120, 40);
    assert!(has(&scrolled, "↑12 ─ End follows"), "{scrolled:#?}");
    assert!(!has(&scrolled, " streaming ─"), "{scrolled:#?}");
}

#[test]
fn the_chat_input_box_grows_with_its_content_and_stops_at_five_rows() {
    let mut a = term(TermSub::Chat);
    a.focus = Focus::Input;
    a.chat.input = (1..=10)
        .map(|i| format!("L{i}"))
        .collect::<Vec<_>>()
        .join("\n");
    let rows = screen(&a, 120, 40);
    assert!(has(&rows, "L1"), "{rows:#?}");
    assert!(
        has(&rows, "L5"),
        "five rows of input are visible:\n{rows:#?}"
    );
    assert!(
        !has(&rows, "L6"),
        "the box stops growing rather than eating the transcript:\n{rows:#?}"
    );
}

#[test]
fn a_focused_chat_input_carries_the_cursor() {
    let mut a = term(TermSub::Chat);
    a.chat.input = "why".into();
    assert!(!has(&screen(&a, 120, 40), "why▏"));
    a.focus = Focus::Input;
    assert!(has(&screen(&a, 120, 40), "why▏"));
}

#[test]
fn both_terminal_panes_survive_narrow_and_short_terminals() {
    for sub in [TermSub::Ops, TermSub::Chat] {
        let mut a = term(sub);
        a.ops.output = (0..40).map(|i| format!("line-{i}")).collect();
        a.chat.input = "one\ntwo\nthree".into();
        for (w, h) in [(20u16, 4u16), (20, 8), (20, 40), (40, 12), (200, 3)] {
            let rows = screen(&a, w, h);
            assert_eq!(rows.len(), h as usize, "{w}x{h} drew a partial frame");
        }
    }
}
