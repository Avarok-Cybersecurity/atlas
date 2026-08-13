// SPDX-License-Identifier: AGPL-3.0-only

//! The quit guard's knowledge of in-flight model loads.
//!
//! In its own mount (not `app_keys_tests.rs`, which is at the per-file cap)
//! because these cases are about what [`App::work_in_flight`] names, not
//! about key routing.

use super::*;
use crossterm::event::{KeyCode, KeyEvent};

fn app() -> App {
    App::new(clap::Parser::parse_from(["spark", "org/m"]))
}

fn press(a: &mut App, c: char) {
    a.on_key(KeyEvent::from(KeyCode::Char(c)));
}

/// A load is minutes of shard reading the user cannot resume, and `q` used to
/// tear it down with no confirmation — the one long-running job the guard
/// did not know about.
#[test]
fn q_asks_first_while_the_boot_load_is_still_running() {
    let mut a = app(); // argv names a model; `ready` has not flipped yet
    assert_eq!(a.work_in_flight(), Some("a model is still loading"));
    press(&mut a, 'q');
    assert!(a.confirm_quit, "the first press asks");
    assert!(!a.should_quit);

    // Once serving, the same press quits clean.
    let mut a = app();
    a.progress.ready = true;
    press(&mut a, 'q');
    assert!(a.should_quit);
    assert!(!a.confirm_quit);
}

/// `spark serve` with no model has nothing loading: the Library boot must
/// not inherit the loading guard, or an idle dashboard costs a confirmation
/// that protects nothing.
#[test]
fn an_awaiting_model_boot_quits_without_the_loading_prompt() {
    let mut a = App::new(clap::Parser::parse_from(["spark"]));
    assert!(a.work_in_flight().is_none());
    press(&mut a, 'q');
    assert!(a.should_quit);
}
