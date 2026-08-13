// SPDX-License-Identifier: AGPL-3.0-only

//! The Library actions the reducer cannot perform itself.
//!
//! Only the REFUSALS are driven here, and deliberately: every success path
//! starts a multi-gigabyte download or loads a model, so the cases worth a unit
//! test are the ones that must cost nothing.

use super::*;
use crossterm::event::{KeyCode, KeyEvent};

fn library() -> App {
    let mut a = App::new(clap::Parser::parse_from(["spark", "org/m"]));
    a.on_key(KeyEvent::from(KeyCode::Char('4')));
    a
}

fn key(c: char) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(c), crossterm::event::KeyModifiers::NONE)
}

fn last_toast(a: &App) -> (&str, bool) {
    let t = a.toasts.last().expect("a toast");
    (t.text.as_str(), t.error)
}

#[test]
fn downloading_with_nothing_selected_refuses_before_it_resolves_a_cache() {
    // The order matters: an empty list must not send a request at whatever
    // repository a stale selection points at.
    let mut a = library();
    a.download_selected_model();
    assert_eq!(last_toast(&a), ("no model selected", true));
}

#[test]
fn checking_freshness_with_nothing_selected_refuses_the_same_way() {
    let mut a = library();
    a.check_selected_model();
    assert_eq!(last_toast(&a), ("no model selected", true));
}

#[test]
fn launching_without_a_server_says_so_and_goes_nowhere() {
    // The dashboard can be built without a host — the launch has to fail
    // loudly rather than navigate to a Main pane tracking a load that is not
    // happening.
    let mut a = library();
    a.launch_selected_recipe();
    assert_eq!(
        last_toast(&a),
        ("no server attached to this dashboard", true)
    );
    assert_eq!(a.section, Section::Library, "still where the user was");
}

#[test]
fn the_library_download_keys_reach_these_actions() {
    // `d` / `u` / `x` are the Library's, and the reducer that owns them cannot
    // perform any of the three — this is the wiring between the two halves.
    for (key, expected) in [
        ('d', "no model selected"),
        ('u', "no model selected"),
        ('x', "nothing is downloading"),
    ] {
        let mut a = library();
        a.on_key(KeyEvent::from(KeyCode::Char(key)));
        assert_eq!(last_toast(&a).0, expected, "`{key}`");
    }
}

#[test]
fn cancelling_when_nothing_is_downloading_is_not_an_error() {
    // It is a mis-press, not a failure, so it must not raise a red toast that
    // sticks until dismissed.
    let mut a = library();
    a.on_key(KeyEvent::from(KeyCode::Char('x')));
    assert_eq!(last_toast(&a), ("nothing is downloading", false));
}

/// Asking for a SECOND download must ask, not refuse. The old behaviour was a
/// toast naming the running job — accurate, and a dead end: the way forward
/// was to know `x` stops it, find its row, press it, then press `d` again.
#[test]
fn a_second_download_opens_the_question_instead_of_refusing() {
    let mut a = library();
    let root = std::env::temp_dir().join("atlas-switch");
    std::fs::create_dir_all(&root).ok();
    a.download.start("org/first", root);
    a.download_switch = None;
    // Ask for a different model while the first is running.
    a.download_switch = Some(("org/first".to_string(), "org/second".to_string()));
    assert!(a.download_switch.is_some(), "the question is open");

    // Any key that is not affirmative keeps the running download.
    let consumed = a.answer_download_switch(key('n'));
    assert!(consumed, "the question owns the keyboard");
    assert!(a.download_switch.is_none(), "answered");
    assert!(a.pending_start.is_none(), "nothing was queued");
    assert!(a.download.job.is_some(), "the running download survives");
}

/// The affirmative queues the wanted model rather than starting it next to the
/// running one — the one-job invariant is the whole reason the question exists.
#[test]
fn the_affirmative_queues_the_second_download_it_does_not_race_it() {
    let mut a = library();
    let root = std::env::temp_dir().join("atlas-switch2");
    std::fs::create_dir_all(&root).ok();
    a.download.start("org/first", root);
    a.download_switch = Some(("org/first".to_string(), "org/second".to_string()));

    assert!(a.answer_download_switch(key('x')));
    assert_eq!(
        a.pending_start.as_deref(),
        Some("org/second"),
        "queued, not started"
    );
    // Still exactly one job tracked: B must not start until the slot is free.
    assert!(
        a.download
            .job
            .as_ref()
            .is_some_and(|j| j.repo == "org/first"),
        "B must not have displaced A"
    );
}
