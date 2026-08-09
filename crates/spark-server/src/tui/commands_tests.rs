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
