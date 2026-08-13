// SPDX-License-Identifier: AGPL-3.0-only

//! Scrolling, from the keyboard and from the wheel.
//!
//! `app_tests.rs` covers the wheel's ceilings; these cases are the ones the
//! KEYS reach — which used to be a second, unclamped implementation of the
//! same thing — plus the boundaries and the empty-pane cases.

use super::*;
use crate::tui::app::{BenchSub, Focus};
use crossterm::event::{KeyCode, KeyEvent};

fn app() -> App {
    App::new(clap::Parser::parse_from(["spark", "org/m"]))
}

fn press(a: &mut App, c: char) {
    a.on_key(KeyEvent::from(KeyCode::Char(c)));
}

fn tap(a: &mut App, code: KeyCode) {
    a.on_key(KeyEvent::from(code));
}

/// Main ▸ Overview with `lines` rows of scrollback above the fold.
fn log_pane(lines: usize) -> App {
    let mut a = app();
    a.section = Section::Main;
    a.main_sub = MainSub::Overview;
    a.log_scroll_max.set(lines);
    a
}

#[test]
fn the_keyboard_stops_at_the_oldest_line_just_as_the_wheel_does() {
    // The wheel was given a ceiling and the keys were not, so `k` walked the
    // offset past the end of the buffer: the pane went blank and coming back
    // cost exactly as many presses as had been spent going up.
    let mut a = log_pane(4);
    for _ in 0..20 {
        press(&mut a, 'k');
    }
    assert_eq!(a.log_scroll, Some(4), "clamped at the oldest line");
    for _ in 0..4 {
        press(&mut a, 'j');
    }
    assert_eq!(a.log_scroll, None, "and four presses bring it back");
}

#[test]
fn the_arrows_and_the_vi_keys_are_the_same_binding() {
    let mut a = log_pane(10);
    tap(&mut a, KeyCode::Up);
    tap(&mut a, KeyCode::Up);
    assert_eq!(a.log_scroll, Some(2));
    tap(&mut a, KeyCode::Down);
    assert_eq!(a.log_scroll, Some(1));
    press(&mut a, 'k');
    assert_eq!(a.log_scroll, Some(2));
    press(&mut a, 'j');
    assert_eq!(a.log_scroll, Some(1));
}

#[test]
fn a_log_shorter_than_the_viewport_cannot_be_scrolled_at_all() {
    // Nothing above the fold: the pane must refuse rather than pretend, or the
    // status chip reads `⏸ 3↑` over a view that never moved.
    let mut a = log_pane(0);
    for _ in 0..5 {
        press(&mut a, 'k');
    }
    assert_eq!(a.log_scroll, None, "still following the newest line");
    press(&mut a, 'j');
    assert_eq!(a.log_scroll, None);
}

#[test]
fn end_and_capital_g_return_to_following_from_any_depth() {
    for jump_key in [KeyCode::Char('G'), KeyCode::End] {
        let mut a = log_pane(200);
        for _ in 0..30 {
            press(&mut a, 'k');
        }
        assert_eq!(a.log_scroll, Some(30));
        tap(&mut a, jump_key);
        assert_eq!(a.log_scroll, None, "{jump_key:?} snaps to the tip");
        tap(&mut a, jump_key);
        assert_eq!(a.log_scroll, None, "and is idempotent there");
    }
}

#[test]
fn the_kernel_table_clamps_at_both_ends() {
    let mut a = app();
    a.section = Section::Main;
    a.main_sub = MainSub::Kernels;
    a.kernel_scroll_max.set(3);
    for _ in 0..10 {
        press(&mut a, 'j');
    }
    assert_eq!(a.kernel_scroll, 3, "cannot scroll past the last row");
    for _ in 0..10 {
        press(&mut a, 'k');
    }
    assert_eq!(a.kernel_scroll, 0, "nor above the first");

    press(&mut a, 'j');
    press(&mut a, 'g');
    assert_eq!(a.kernel_scroll, 0, "`g` is the way home");
}

#[test]
fn a_kernel_table_that_fits_on_screen_does_not_move() {
    let mut a = app();
    a.section = Section::Main;
    a.main_sub = MainSub::Kernels;
    for _ in 0..5 {
        press(&mut a, 'j');
    }
    assert_eq!(a.kernel_scroll, 0);
}

#[test]
fn a_growing_log_does_not_yank_a_reader_who_scrolled_up() {
    // The offset counts BACKWARDS from the newest line, so lines arriving
    // underneath leave the reader parked where they were. Only an explicit key
    // resumes following.
    let mut a = log_pane(10);
    for _ in 0..3 {
        press(&mut a, 'k');
    }
    assert_eq!(a.log_scroll, Some(3));
    a.log_scroll_max.set(400); // the renderer sees a much longer buffer
    a.on_tick();
    press(&mut a, 'z'); // an unbound key, i.e. a redraw with no input
    assert_eq!(a.log_scroll, Some(3), "still parked");
    tap(&mut a, KeyCode::End);
    assert_eq!(a.log_scroll, None);
}

#[test]
fn typing_while_scrolled_back_does_not_yank_the_chat_transcript() {
    let mut a = app();
    press(&mut a, '6');
    press(&mut a, '6');
    press(&mut a, 'i');
    a.chat_scroll_max.set(10); // the keys clamp now, like the wheel
    tap(&mut a, KeyCode::PageUp);
    assert_eq!(a.chat.scroll, Some(10));
    for c in "still reading".chars() {
        press(&mut a, c);
    }
    assert_eq!(a.chat.scroll, Some(10), "typing is not navigation");
    // Sending IS an explicit "show me the new reply", so that one does resume.
    tap(&mut a, KeyCode::Enter);
    assert_eq!(a.chat.scroll, None);
}

#[test]
fn the_chat_wheel_respects_the_ceiling_and_collapses_to_follow() {
    // Including the collapse back to follow when there is nothing above the
    // fold at all. (The keyboard paths share this clamp now — see the test
    // below — so wheel and keys can no longer disagree about position.)
    let mut a = app();
    a.section = Section::Terminal;
    a.term_sub = TermSub::Chat;
    a.chat_scroll_max.set(2);
    for _ in 0..10 {
        a.scroll(-3);
    }
    assert_eq!(a.chat.scroll, Some(2));
    a.chat_scroll_max.set(0);
    a.scroll(-3);
    assert_eq!(a.chat.scroll, None, "an empty transcript follows the tip");
}

#[test]
fn the_ops_wheel_moves_ops_output_and_never_the_main_log() {
    // The previous version of this test asserted the OPPOSITE, on the claim
    // that Ops "draws the same ring buffer" — it does not: Ops renders
    // `ops.output`, so routing its wheel to `log_scroll` moved an offset
    // this section does not even draw. Wheel in Ops: nothing visibly
    // happened, and Main was later found scrolled.
    let mut a = app();
    a.section = Section::Terminal;
    a.term_sub = TermSub::Ops;
    a.log_scroll_max.set(20);
    a.ops.scroll_max.set(20);
    a.scroll(-3);
    assert_eq!(a.ops.scroll_up, 3, "the pane under the wheel moved");
    assert_eq!(a.log_scroll, None, "the Main log did not");
}

#[test]
fn the_wheel_moves_the_benchmark_selection_and_stops_at_the_ends() {
    // Lists move their SELECTION, not a viewport — that is what the arrow keys
    // do here, and a wheel that scrolled past it would leave the two out of step.
    let n = atlas_plugin::registry::all().len();
    let mut a = app();
    a.section = Section::Benchmarks;
    a.bench_sub = BenchSub::Suite;
    for _ in 0..(n + 5) {
        a.scroll(1);
    }
    assert_eq!(
        a.bench.selected,
        n.saturating_sub(1),
        "the last benchmark, not past it"
    );
    for _ in 0..(n + 5) {
        a.scroll(-1);
    }
    assert_eq!(a.bench.selected, 0);
}

#[test]
fn sections_with_nothing_to_scroll_ignore_the_wheel_without_panicking() {
    let mut a = app();
    a.log_scroll_max.set(10);
    a.scroll(-3);
    let parked = a.log_scroll;
    for s in [Section::Stats, Section::Network, Section::Library] {
        a.section = s;
        a.scroll(3);
        a.scroll(-3);
    }
    assert_eq!(a.log_scroll, parked, "and touch nobody else's offset");
}

#[test]
fn scrolling_does_not_disturb_focus_or_the_section() {
    let mut a = app();
    press(&mut a, '6');
    press(&mut a, 'i');
    a.log_scroll_max.set(10);
    a.scroll(-3);
    assert!(a.focus == Focus::Input);
    assert_eq!(a.section, Section::Terminal);
}

#[test]
fn lowercase_g_and_home_jump_the_log_to_its_oldest_line() {
    // The help overlay advertises "g / G — top / bottom" globally; the log
    // pane implemented only the G half.
    for jump_key in [KeyCode::Char('g'), KeyCode::Home] {
        let mut a = log_pane(200);
        tap(&mut a, jump_key);
        assert_eq!(a.log_scroll, Some(200), "{jump_key:?} parks at the oldest");
    }
    // A log that fits has no "top" to jump to; following is the honest state.
    let mut a = log_pane(0);
    press(&mut a, 'g');
    assert_eq!(a.log_scroll, None);
}

#[test]
fn capital_g_and_end_jump_the_kernel_table_to_its_last_row() {
    let mut a = app();
    a.section = Section::Main;
    a.main_sub = MainSub::Kernels;
    a.kernel_scroll_max.set(7);
    for jump_key in [KeyCode::Char('G'), KeyCode::End] {
        a.kernel_scroll = 0;
        tap(&mut a, jump_key);
        assert_eq!(a.kernel_scroll, 7, "{jump_key:?} parks at the bottom");
    }
    tap(&mut a, KeyCode::Home);
    assert_eq!(a.kernel_scroll, 0, "Home is the same way back as `g`");
}

#[test]
fn chat_g_and_home_jump_to_the_oldest_row_in_both_focus_states() {
    let mut a = app();
    press(&mut a, '6');
    press(&mut a, '6'); // Terminal ▸ Chat, content focus
    a.chat_scroll_max.set(40);
    press(&mut a, 'g');
    assert_eq!(a.chat.scroll, Some(40), "content-focus g parks at the top");
    tap(&mut a, KeyCode::End);
    assert_eq!(a.chat.scroll, None);

    press(&mut a, 'i'); // input focus: `g` is text, Home is the jump
    tap(&mut a, KeyCode::Home);
    assert_eq!(a.chat.scroll, Some(40));
    press(&mut a, 'g');
    assert_eq!(a.chat.input, "g", "a bare g while typing stays a letter");
    assert_eq!(a.chat.scroll, Some(40), "and moves nothing");
}

#[test]
fn an_empty_chat_ignores_the_jump_rather_than_banking_it() {
    let mut a = app();
    press(&mut a, '6');
    press(&mut a, '6');
    press(&mut a, 'g');
    assert_eq!(a.chat.scroll, None, "nothing above the fold to park at");
}

#[test]
fn help_scroll_keys_move_the_key_list_and_anything_else_closes_it() {
    // At the 80x24 floor the key table is taller than the modal; j/k must
    // page it rather than dismiss it, or the tail entries stay unreadable.
    let mut a = app();
    press(&mut a, '?');
    assert!(a.help_open);
    a.help_scroll_max.set(2); // what the renderer would publish at 80x24
    press(&mut a, 'j');
    press(&mut a, 'j');
    press(&mut a, 'j');
    assert!(a.help_open, "scroll keys do not dismiss");
    assert_eq!(a.help_scroll, 2, "and clamp at the ceiling");
    press(&mut a, 'k');
    assert_eq!(a.help_scroll, 1);
    press(&mut a, 'G');
    assert_eq!(a.help_scroll, 2);
    press(&mut a, 'g');
    assert_eq!(a.help_scroll, 0);

    press(&mut a, 'G');
    let section = a.section;
    press(&mut a, '4');
    assert!(!a.help_open, "a non-scroll key closes it");
    assert_eq!(a.section, section, "and is swallowed, not acted on");
    assert_eq!(a.help_scroll, 0, "the next open starts at the top");
}

/// Both chat keyboard paths — the input-focus arrows and the content-focus
/// letters — used to call `scroll_by` bare, banking presses past the oldest
/// row that were paid back one dead key at a time. The tree's own doctrine
/// (`app_scroll`, `lib_modal`) names exactly this defect.
#[test]
fn the_chat_keys_clamp_against_the_same_ceiling_as_the_wheel() {
    let mut a = app();
    press(&mut a, '6');
    press(&mut a, '6');
    press(&mut a, 'i');
    a.chat_scroll_max.set(3);
    tap(&mut a, KeyCode::PageUp);
    assert_eq!(
        a.chat.scroll,
        Some(3),
        "PageUp lands on the ceiling, not 10"
    );

    tap(&mut a, KeyCode::Esc); // back to content focus
    for _ in 0..5 {
        press(&mut a, 'k');
    }
    assert_eq!(a.chat.scroll, Some(3), "content-focus k cannot bank either");
    press(&mut a, 'j');
    assert_eq!(a.chat.scroll, Some(2), "one press back means one row back");
}
