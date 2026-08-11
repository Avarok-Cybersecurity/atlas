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
