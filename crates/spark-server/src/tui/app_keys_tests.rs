// SPDX-License-Identifier: AGPL-3.0-only

//! Every global binding, driven through the real reducer.
//!
//! Synthetic `KeyEvent`s go into [`App::on_key`] and the assertions are on the
//! state a user would see on screen — the terminal equivalent of driving the
//! UI rather than calling the handlers it happens to be built from.
//!
//! Split from `app_tests.rs`, which owns the navigation-order and lifecycle
//! cases, to stay under the per-file cap.

use super::*;
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

/// Where the sidebar cursor is, spelled the way the sidebar draws it.
fn at(a: &App) -> String {
    match a.section.subs().get(a.sub_index(a.section)) {
        Some(sub) => format!("{}/{sub}", a.section.label()),
        None => a.section.label().to_string(),
    }
}

fn focus_of(a: &App) -> u8 {
    match a.focus {
        Focus::Sidebar => 0,
        Focus::Content => 1,
        Focus::Input => 2,
    }
}

/// Everything a keystroke could plausibly move, for the "does nothing" cases.
fn snapshot(a: &App) -> (String, u8, Option<usize>, usize, bool, bool, bool) {
    (
        at(a),
        focus_of(a),
        a.log_scroll,
        a.kernel_scroll,
        a.help_open,
        a.should_quit,
        a.log_filter_editing,
    )
}

#[test]
fn tab_walks_every_sidebar_row_in_order_and_wraps_home() {
    let mut a = app();
    let mut seen = vec![at(&a)];
    for _ in 1..App::nav_rows().len() {
        tap(&mut a, KeyCode::Tab);
        seen.push(at(&a));
    }
    assert_eq!(
        seen,
        [
            "Main/Overview",
            "Main/Kernels",
            "Stats",
            "Network",
            "Library",
            "Benchmarks/Suite",
            "Benchmarks/History",
            "Terminal/Ops",
            "Terminal/Chat",
        ]
    );
    tap(&mut a, KeyCode::Tab);
    assert_eq!(at(&a), "Main/Overview", "the last row wraps to the first");
}

#[test]
fn shift_tab_walks_the_same_rows_backwards() {
    let mut a = app();
    tap(&mut a, KeyCode::BackTab);
    assert_eq!(at(&a), "Terminal/Chat", "the first row wraps to the last");
    for expected in ["Terminal/Ops", "Benchmarks/History", "Benchmarks/Suite"] {
        tap(&mut a, KeyCode::BackTab);
        assert_eq!(at(&a), expected);
    }
    tap(&mut a, KeyCode::Tab);
    assert_eq!(at(&a), "Benchmarks/History", "and ⇥ undoes ⇧⇥");
}

#[test]
fn a_repeat_section_key_cycles_that_sections_subsections() {
    let mut a = app();
    // Away from Main first, so the press below is a plain arrival rather than
    // the repeat this test is about.
    press(&mut a, '3');
    press(&mut a, '1');
    assert_eq!(at(&a), "Main/Overview");
    press(&mut a, '1');
    assert_eq!(at(&a), "Main/Kernels");
    press(&mut a, '1');
    assert_eq!(
        at(&a),
        "Main/Overview",
        "two subsections, so it is a toggle"
    );

    // A section without subsections has nothing to cycle and must not move.
    press(&mut a, '2');
    press(&mut a, '2');
    assert_eq!(at(&a), "Stats");
}

#[test]
fn a_section_key_arriving_from_elsewhere_lands_on_the_current_subsection() {
    // Leaving Main on Kernels and coming back must not silently reset the view.
    let mut a = app();
    press(&mut a, '1'); // already on Main, so this cycles to Kernels
    assert_eq!(at(&a), "Main/Kernels");
    press(&mut a, '3');
    press(&mut a, '1');
    assert_eq!(at(&a), "Main/Kernels");
}

#[test]
fn help_opens_on_question_mark_and_the_next_key_only_closes_it() {
    let mut a = app();
    press(&mut a, '?');
    assert!(a.help_open);
    // The key that dismisses the modal must not ALSO act, or dismissing it
    // navigates somewhere the user did not ask for.
    press(&mut a, '4');
    assert!(!a.help_open);
    assert_eq!(at(&a), "Main/Overview", "the dismissing key was swallowed");
    press(&mut a, '4');
    assert_eq!(
        at(&a),
        "Library",
        "and works normally once the modal is gone"
    );
}

#[test]
fn help_is_reversible_with_the_same_key() {
    let mut a = app();
    press(&mut a, '?');
    press(&mut a, '?');
    assert!(!a.help_open);
}

#[test]
fn ctrl_c_quits_even_while_a_text_field_owns_the_keyboard() {
    // Raw mode swallows SIGINT, so this key IS the interrupt. It must outrank
    // every buffer — including the ones that would otherwise type a `c`.
    // (`shutdown::request` latches a process global; that is the point of it.)
    let mut a = app();
    press(&mut a, '6');
    press(&mut a, 'i');
    assert_eq!(focus_of(&a), 2, "the ops input has focus");
    chord(&mut a, 'c', KeyModifiers::CONTROL);
    assert!(a.should_quit);
    assert_eq!(a.ops.input, "", "the interrupt was not typed into the line");
}

#[test]
fn q_quits_only_when_no_text_field_owns_the_keyboard() {
    let mut a = app();
    press(&mut a, 'q');
    assert!(a.should_quit);

    let mut a = app();
    press(&mut a, '6');
    press(&mut a, 'i');
    press(&mut a, 'q');
    assert!(!a.should_quit, "a `q` in an input line is a letter");
    assert_eq!(a.ops.input, "q");
}

#[test]
fn esc_outside_a_text_field_drops_focus_and_resumes_following() {
    let mut a = app();
    a.log_scroll_max.set(50);
    press(&mut a, 'k');
    press(&mut a, 'k');
    assert_eq!(a.log_scroll, Some(2));
    a.focus = Focus::Sidebar;
    tap(&mut a, KeyCode::Esc);
    assert_eq!(focus_of(&a), 1, "focus returns to the content");
    assert_eq!(a.log_scroll, None, "and the log pane follows again");
}

#[test]
fn esc_in_the_benchmarks_section_steps_back_instead_of_dropping_focus() {
    // Benchmarks and the Library own Esc — there it means "back one step in the
    // flow", which is why the global Esc arm is matched AFTER their catch-all arms.
    use crate::tui::bench_state::View;
    let mut a = app();
    a.log_scroll_max.set(20);
    press(&mut a, 'k');
    press(&mut a, '5');
    a.bench.view = View::Params;
    tap(&mut a, KeyCode::Esc);
    assert_eq!(a.bench.view, View::List, "one step back in the flow");
    assert_eq!(
        a.log_scroll,
        Some(1),
        "and the global Esc did not also fire"
    );
}

#[test]
fn f_opens_the_log_filter_on_main_and_nowhere_else() {
    let mut a = app();
    press(&mut a, 'f');
    assert!(a.log_filter_editing);
    for c in "warn".chars() {
        press(&mut a, c);
    }
    assert_eq!(a.log_filter, "warn");
    // Enter commits: the filter stays, the keyboard is handed back.
    tap(&mut a, KeyCode::Enter);
    assert!(!a.log_filter_editing);
    assert_eq!(a.log_filter, "warn");
    // Esc is the discard: it clears what was typed, which is what makes `f`
    // reversible rather than a one-way trip.
    press(&mut a, 'f');
    tap(&mut a, KeyCode::Esc);
    assert!(!a.log_filter_editing);
    assert_eq!(a.log_filter, "");

    for section in ['2', '3', '6'] {
        let mut a = app();
        press(&mut a, section);
        press(&mut a, 'f');
        assert!(!a.log_filter_editing, "`f` is a Main binding");
    }
}

#[test]
fn the_filter_is_editable_on_the_kernels_subsection_too() {
    // Both Main subsections draw the log pane's filter chip, and `f` is matched
    // on the SECTION — a filter reachable from one subsection only would read
    // as a broken key.
    let mut a = app();
    press(&mut a, '1');
    assert_eq!(at(&a), "Main/Kernels");
    press(&mut a, 'f');
    assert!(a.log_filter_editing);
}

#[test]
fn entering_and_leaving_the_terminal_input_is_reversible() {
    let mut a = app();
    press(&mut a, '6');
    assert_eq!(focus_of(&a), 1);
    press(&mut a, 'i');
    assert_eq!(focus_of(&a), 2);
    tap(&mut a, KeyCode::Esc);
    assert_eq!(focus_of(&a), 1, "Esc hands the keyboard back");
    // Enter is the second way in — `i` is muscle memory, Enter is what a
    // terminal-shaped pane invites.
    tap(&mut a, KeyCode::Enter);
    assert_eq!(focus_of(&a), 2);
    tap(&mut a, KeyCode::Esc);
    assert_eq!(focus_of(&a), 1);
}

#[test]
fn leaving_the_terminal_by_section_key_does_not_strand_the_focus() {
    let mut a = app();
    press(&mut a, '6');
    press(&mut a, 'i');
    a.on_key(KeyEvent::new(KeyCode::Char('2'), KeyModifiers::NONE));
    // The digit is text while the input has focus, so the section key cannot
    // be the way out — Esc is, and this documents that it is the only one.
    assert_eq!(at(&a), "Terminal/Ops");
    assert_eq!(a.ops.input, "2");
    tap(&mut a, KeyCode::Esc);
    press(&mut a, '2');
    assert_eq!(at(&a), "Stats");
}

#[test]
fn the_network_pane_walks_its_ranks_and_stops_at_both_ends() {
    let mut a = app();
    a.args.world_size = 3;
    press(&mut a, '3');
    for (key, expected) in [('l', 1), ('l', 2), ('l', 2), ('h', 1), ('h', 0), ('h', 0)] {
        press(&mut a, key);
        assert_eq!(a.network_selected, expected, "after `{key}`");
    }
    tap(&mut a, KeyCode::Right);
    assert_eq!(a.network_selected, 1, "the arrows are the same binding");
    tap(&mut a, KeyCode::Left);
    assert_eq!(a.network_selected, 0);

    tap(&mut a, KeyCode::Enter);
    assert!(a.network_detail);
    tap(&mut a, KeyCode::Enter);
    assert!(!a.network_detail, "Enter is a toggle");
}

#[test]
fn a_single_rank_deployment_has_nowhere_to_walk() {
    let mut a = app();
    press(&mut a, '3');
    assert_eq!(a.args.world_size, 1);
    press(&mut a, 'l');
    assert_eq!(a.network_selected, 0, "there is no second rank to select");
}

#[test]
fn the_library_owns_its_own_keys_but_the_globals_still_win() {
    let mut a = app();
    press(&mut a, '4');
    // `/` belongs to the Library, and starting to type must take the keyboard
    // away from the global bindings — otherwise a model name with a digit in
    // it navigates out of the section mid-search.
    press(&mut a, '/');
    assert!(a.lib.filter_editing);
    press(&mut a, '3');
    press(&mut a, 'q');
    assert_eq!(a.lib.filter, "3q");
    assert_eq!(at(&a), "Library");
    assert!(!a.should_quit);
    // Esc backs out one step, and the globals answer again.
    tap(&mut a, KeyCode::Esc);
    assert!(!a.lib.filter_editing);
    assert_eq!(a.lib.filter, "", "Esc discards the search");
    press(&mut a, '3');
    assert_eq!(at(&a), "Network");
}

#[test]
fn a_gauge_pane_ignores_everything_that_is_not_a_global() {
    // Stats is a gauge, not a document: it has nothing to move, and every key
    // it does not own must be inert rather than half-handled.
    let mut a = app();
    press(&mut a, '2');
    let before = snapshot(&a);
    for code in [
        KeyCode::Down,
        KeyCode::Up,
        KeyCode::Enter,
        KeyCode::PageUp,
        KeyCode::PageDown,
        KeyCode::Home,
        KeyCode::End,
        KeyCode::Delete,
        KeyCode::Insert,
        KeyCode::F(1),
        KeyCode::Char('j'),
        KeyCode::Char('z'),
        KeyCode::Char('G'),
    ] {
        tap(&mut a, code);
        assert_eq!(snapshot(&a), before, "{code:?} must do nothing in Stats");
    }
}

#[test]
fn unbound_keys_on_main_leave_the_pane_where_it_was() {
    let mut a = app();
    a.log_scroll_max.set(50);
    press(&mut a, 'k');
    let before = snapshot(&a);
    for code in [
        KeyCode::Char('z'),
        KeyCode::Char('$'),
        KeyCode::F(5),
        KeyCode::Insert,
        KeyCode::Delete,
        KeyCode::Backspace,
    ] {
        tap(&mut a, code);
        assert_eq!(snapshot(&a), before, "{code:?} must do nothing on Main");
    }
}

#[test]
fn a_keystroke_ends_a_mouse_selection() {
    // The coordinates are SCREEN CELLS, so the moment anything redraws they
    // highlight different text than the user chose.
    let mut a = app();
    a.selection = Some(crate::tui::selection::Selection::new((4, 4)));
    press(&mut a, 'z');
    assert!(a.selection.is_none(), "even a key that does nothing else");
}

/// `q` on an idle dashboard is unchanged — nothing is lost, so nothing is
/// asked. The confirmation is a cost, and a cost paid for nothing trains the
/// user to dismiss it.
#[test]
fn q_still_quits_immediately_when_there_is_nothing_to_lose() {
    let mut a = app();
    assert!(a.work_in_flight().is_none());
    press(&mut a, 'q');
    assert!(a.should_quit);
    assert!(!a.confirm_quit);
}

/// ★ `q` DRAINS AND STOPS THE SERVER. A single stray keypress used to end a
/// multi-hour benchmark with no way back, from any screen, including one where
/// the user was only reading logs.
#[test]
fn q_asks_first_when_a_run_is_in_flight_and_a_second_q_confirms() {
    let mut a = app();
    a.chat.streaming = true;
    press(&mut a, 'q');
    assert!(a.confirm_quit, "the first press asks");
    assert!(!a.should_quit, "and does NOT quit");

    press(&mut a, 'q');
    assert!(a.should_quit, "the second press is the deliberate one");
    assert!(!a.confirm_quit);
}

#[test]
fn y_also_confirms_because_the_prompt_offers_it() {
    let mut a = app();
    a.chat.streaming = true;
    press(&mut a, 'q');
    press(&mut a, 'y');
    assert!(a.should_quit);
}

/// The safe reading of an ambiguous keystroke is "do not stop the server", so
/// everything that is not an affirmative cancels — and cancelling must not
/// also do the thing the key normally does, or dismissing the prompt would
/// jump the user to another section as a side effect.
#[test]
fn any_other_key_cancels_and_does_nothing_else() {
    for code in [
        KeyCode::Esc,
        KeyCode::Char('n'),
        KeyCode::Char('3'),
        KeyCode::Enter,
    ] {
        let mut a = app();
        a.chat.streaming = true;
        let before = a.section;
        press(&mut a, 'q');
        tap(&mut a, code);
        assert!(!a.should_quit, "{code:?} must not quit");
        assert!(!a.confirm_quit, "{code:?} must dismiss the prompt");
        assert_eq!(a.section, before, "{code:?} was swallowed by the prompt");
    }
}

/// Ctrl+C is the deliberate override and must stay one keystroke: it is what a
/// user falls back on when the UI is not responding, and a confirmation on it
/// would be a prompt they cannot see.
#[test]
fn ctrl_c_still_shuts_down_without_asking() {
    let mut a = app();
    a.chat.streaming = true;
    chord(&mut a, 'c', KeyModifiers::CONTROL);
    assert!(a.should_quit);
    assert!(!a.confirm_quit);
}
