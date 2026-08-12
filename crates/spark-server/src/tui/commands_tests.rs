// SPDX-License-Identifier: AGPL-3.0-only

//! The Ops REPL: completion, dispatch, and what every command puts in the pane.
//!
//! `/gpu` is deliberately absent — it queries the device, and these run on
//! boxes with a benchmark on it.

use super::*;

fn app() -> App {
    App::new(clap::Parser::parse_from(["spark", "org/m"]))
}

/// Everything the pane printed, as one blob to search.
fn out(a: &App) -> String {
    a.ops.output.join("\n")
}

#[test]
fn completion_only_fires_on_an_unambiguous_prefix() {
    assert_eq!(complete("/q"), Some("/quit"));
    assert_eq!(complete("/de"), Some("/detach"));
    assert_eq!(complete("/"), Some("/help"), "the first, not nothing");
    assert_eq!(complete("/zzz"), None, "no such command");
    assert_eq!(complete(""), None);
    assert_eq!(complete("/status"), None, "already complete");
}

#[test]
fn completion_stops_once_an_argument_is_being_typed() {
    // The ghost text is for command names; continuing to offer one while the
    // user types `/metrics decode` would suggest replacing what they wrote.
    assert_eq!(complete("/metrics "), None);
    assert_eq!(complete("/metrics dec"), None);
}

#[test]
fn completion_offers_the_bare_name_of_a_command_that_takes_an_argument() {
    assert_eq!(complete("/met"), Some("/metrics"));
    assert_eq!(complete("/ker"), Some("/kernels"));
}

#[test]
fn every_command_echoes_the_line_it_ran() {
    let mut a = app();
    execute("/cache", &mut a);
    assert_eq!(a.ops.output[0], "❯ /cache");
}

#[test]
fn a_line_is_trimmed_before_it_is_dispatched() {
    let mut a = app();
    execute("  /help  ", &mut a);
    assert_eq!(a.ops.output[0], "❯ /help");
    assert!(out(&a).contains("/quit"), "and it really ran");
}

#[test]
fn help_lists_every_command_it_can_run() {
    // The list and the dispatcher are the same table, so a command added to one
    // and not the other is the failure this guards.
    let mut a = app();
    execute("/help", &mut a);
    let printed = out(&a);
    for (name, description) in COMMANDS {
        assert!(printed.contains(name), "{name} missing from /help");
        assert!(printed.contains(description), "{description} missing");
    }
}

#[test]
fn bare_text_is_pointed_at_the_chat_tab_rather_than_guessed_at() {
    let mut a = app();
    execute("what is the capital of France", &mut a);
    assert!(out(&a).contains("Chat tab"), "got {:?}", a.ops.output);
    assert!(!a.should_quit);
}

#[test]
fn an_unknown_command_names_itself_and_points_at_help() {
    let mut a = app();
    execute("/nope", &mut a);
    let printed = out(&a);
    assert!(printed.contains("/nope"), "got {printed:?}");
    assert!(printed.contains("/help"));
}

#[test]
fn the_watchdog_needs_on_or_off_and_says_so() {
    for line in ["/watchdog", "/watchdog maybe", "/watchdog ON"] {
        let mut a = app();
        execute(line, &mut a);
        assert!(
            out(&a).contains("usage: /watchdog on|off"),
            "{line} should have been refused"
        );
    }
}

#[test]
fn status_says_so_when_the_scheduler_has_not_published_yet() {
    // The counters are process-wide and always available; the snapshot is not.
    let mut a = app();
    execute("/status", &mut a);
    let printed = out(&a);
    assert!(printed.contains("snapshot not yet published"));
    assert!(printed.contains("requests:"), "the counters still print");
}

#[test]
fn a_filter_that_matches_nothing_says_nothing_matched() {
    // Rather than printing an empty section, which reads as a broken command.
    for (line, expected) in [
        ("/metrics zzzz-no-such-metric", "(no metrics matched)"),
        (
            "/kernels zzzz-no-such-kernel",
            "(no kernel lookups matched)",
        ),
    ] {
        let mut a = app();
        execute(line, &mut a);
        assert!(out(&a).contains(expected), "{line} → {:?}", a.ops.output);
    }
}

#[test]
fn the_cache_command_reports_even_before_anything_is_cached() {
    let mut a = app();
    execute("/cache", &mut a);
    assert!(out(&a).contains("prefix cache:"));
}

#[test]
fn detach_leaves_the_tui_without_shutting_the_server_down() {
    // The two exits are different: `/detach` keeps serving with plain logs.
    let mut a = app();
    execute("/detach", &mut a);
    assert!(a.detach);
    assert!(!a.should_quit, "the process stays up");
}

#[test]
fn quit_asks_for_a_clean_shutdown() {
    // `shutdown::request` latches a process global on purpose — the drain is a
    // property of the process, not of this dashboard.
    let mut a = app();
    execute("/quit", &mut a);
    assert!(a.should_quit);
    assert!(!a.detach);
}

fn focused_ops(a: &mut App) {
    a.on_key(crossterm::event::KeyEvent::from(
        crossterm::event::KeyCode::Char('6'),
    ));
    a.on_key(crossterm::event::KeyEvent::from(
        crossterm::event::KeyCode::Char('i'),
    ));
}

fn tap(a: &mut App, code: crossterm::event::KeyCode) {
    a.on_key(crossterm::event::KeyEvent::from(code));
}

fn type_line(a: &mut App, s: &str) {
    for c in s.chars() {
        tap(a, crossterm::event::KeyCode::Char(c));
    }
}

/// The "⇥ accept" hint drawn beside the ghost text was a dead key: Tab fell
/// into the input reducer's catch-all while the global Tab handler sat
/// unreachable behind `in_input()`.
#[test]
fn tab_accepts_the_ghost_completion_and_never_leaves_the_section() {
    let mut a = app();
    focused_ops(&mut a);
    type_line(&mut a, "/de");
    tap(&mut a, crossterm::event::KeyCode::Tab);
    assert_eq!(a.ops.input, "/detach", "the ghost became the line");
    assert_eq!(a.section, crate::tui::app::Section::Terminal);

    // No ghost to accept: Tab must be inert — in particular it must NOT
    // reach `cycle_section` and throw the user out of the pane mid-word.
    type_line(&mut a, "zzz");
    let before = a.ops.input.clone();
    tap(&mut a, crossterm::event::KeyCode::Tab);
    assert_eq!(a.ops.input, before, "nothing to complete, nothing changed");
    assert_eq!(a.section, crate::tui::app::Section::Terminal, "still here");
    assert!(a.term_sub == crate::tui::app::TermSub::Ops);
}

/// Up walked history back with no way forward again — the asymmetry the
/// chat pane never had. Down is the way back, and past the newest entry the
/// line returns to empty.
#[test]
fn history_walks_both_ways_and_falls_off_the_newest_end_empty() {
    let mut a = app();
    focused_ops(&mut a);
    type_line(&mut a, "/help");
    tap(&mut a, crossterm::event::KeyCode::Enter);
    type_line(&mut a, "/status");
    tap(&mut a, crossterm::event::KeyCode::Enter);

    tap(&mut a, crossterm::event::KeyCode::Up);
    assert_eq!(a.ops.input, "/status");
    tap(&mut a, crossterm::event::KeyCode::Up);
    assert_eq!(a.ops.input, "/help");
    tap(&mut a, crossterm::event::KeyCode::Down);
    assert_eq!(a.ops.input, "/status");
    tap(&mut a, crossterm::event::KeyCode::Down);
    assert_eq!(a.ops.input, "", "past the newest entry is a fresh line");
    assert!(a.ops.history_pos.is_none());
    tap(&mut a, crossterm::event::KeyCode::Down);
    assert_eq!(a.ops.input, "", "and Down there stays put");
}

/// The Terminal footer has said "↑/↓ scroll" since the pane existed; until
/// now the arrows did nothing in Ops and the WHEEL silently moved the Main
/// tab's log offset — a pane this section does not even render.
#[test]
fn ops_output_scrolls_with_keys_and_wheel_against_the_published_ceiling() {
    let mut a = app();
    a.on_key(crossterm::event::KeyEvent::from(
        crossterm::event::KeyCode::Char('6'),
    ));
    a.ops.output = (0..100).map(|i| format!("line {i}")).collect();
    a.ops.scroll_max.set(50); // what the renderer would publish

    tap(&mut a, crossterm::event::KeyCode::Up);
    assert_eq!(a.ops.scroll_up, 1, "content-focus arrows move the output");
    a.scroll(-3); // wheel up
    assert_eq!(a.ops.scroll_up, 4);
    assert_eq!(
        a.log_scroll, None,
        "Main's log no longer moves as a side effect"
    );

    tap(&mut a, crossterm::event::KeyCode::Home);
    assert_eq!(a.ops.scroll_up, 50, "g/Home parks at the oldest");
    tap(&mut a, crossterm::event::KeyCode::Up);
    assert_eq!(a.ops.scroll_up, 50, "clamped at the ceiling, never banked");
    tap(&mut a, crossterm::event::KeyCode::End);
    assert_eq!(a.ops.scroll_up, 0, "G/End resumes following");

    // PageUp keeps working while the input line owns the keyboard.
    tap(&mut a, crossterm::event::KeyCode::Char('i'));
    tap(&mut a, crossterm::event::KeyCode::PageUp);
    assert_eq!(a.ops.scroll_up, 10);
}

#[test]
fn running_a_command_snaps_the_scrollback_to_its_output() {
    let mut a = app();
    a.ops.scroll_up = 30;
    execute("/help", &mut a);
    assert_eq!(a.ops.scroll_up, 0, "Enter means: show me the result");
}

/// The one buffer in the tree with no cap (log ring 10_000, bench log 500):
/// /metrics adds 40+ lines per call, and a dashboard serves for days.
#[test]
fn the_output_buffer_is_capped_oldest_first() {
    let mut a = app();
    a.ops.output = (0..1_500).map(|i| format!("old {i}")).collect();
    execute("/help", &mut a);
    assert_eq!(a.ops.output.len(), 1_000, "trimmed to the cap");
    assert!(
        a.ops.output.last().unwrap().contains("/quit"),
        "the newest output — the command just run — is what is kept"
    );
    assert!(
        !a.ops.output.iter().any(|l| l == "old 0"),
        "the oldest lines are what went"
    );
}
