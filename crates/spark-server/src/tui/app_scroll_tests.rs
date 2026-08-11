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
fn the_chat_wheel_respects_the_ceiling_the_keys_do_not_see() {
    // The keys move the transcript blind — the ceiling is a render result — so
    // the wheel is where it is enforced, including the collapse back to follow
    // when there is nothing above the fold at all.
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
fn the_ops_pane_and_the_main_log_share_one_scrollback() {
    // They draw the same ring buffer, so a separate offset for each would let
    // the two panes disagree about where the reader is.
    let mut a = app();
    a.section = Section::Terminal;
    a.term_sub = TermSub::Ops;
    a.log_scroll_max.set(20);
    a.scroll(-3);
    assert_eq!(a.log_scroll, Some(3));
    a.section = Section::Main;
    a.main_sub = MainSub::Overview;
    a.scroll(3);
    assert_eq!(a.log_scroll, None);
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
