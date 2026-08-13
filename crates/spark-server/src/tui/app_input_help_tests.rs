// SPDX-License-Identifier: AGPL-3.0-only

//! Input routing for the Help section, split from `app_input_tests.rs` at the
//! 500-line cap.
//!
//! These are ROUTING tests and that is the point: `HelpState`'s own tests sit
//! one layer below `App::on_key` and pass against a build where the section
//! never receives a keystroke at all. Only the real entry point can see that.

// One definition of each fixture, shared with `app_input_tests` --
// the convention `render_tests`/`start_tests` already use.
use super::tests::{app, press, tap};
use super::*;
use crossterm::event::KeyCode;

/// The report title accepts typing through `App::on_key`.
///
/// This is a ROUTING test, and it exists because the unit tests one layer
/// down could not see the bug: `HelpState::on_key` was always correct, but
/// `App` never called it. `in_input()` is consulted before the Help arm in
/// `on_key`, so the moment the title went into edit mode every keystroke was
/// handed to `on_input_key` -- which sent it to `on_help_overlay_key`, the
/// `?` modal's scroll handler, whose `_ =>` arm discards the key. Typing did
/// nothing and nothing on screen moved, which reads as a frozen dashboard.
#[test]
fn typing_a_report_title_reaches_the_composer_and_does_not_look_frozen() {
    let mut a = app();
    a.jump(Section::Help);
    a.help.sub = crate::tui::help_state::HelpSub::Report;

    // Enter on the Title row starts editing; the composer now owns the keys.
    tap(&mut a, KeyCode::Enter);
    assert!(
        a.help.is_editing(),
        "Enter on Title must start editing, else the field can never be filled"
    );

    for c in "gpu oom".chars() {
        press(&mut a, c);
    }
    assert_eq!(
        a.help.title, "gpu oom",
        "every typed character must reach the title buffer"
    );

    // Backspace edits rather than being swallowed.
    tap(&mut a, KeyCode::Backspace);
    assert_eq!(a.help.title, "gpu oo");

    // Esc leaves edit mode and KEEPS the text -- a draft title is the user's
    // work, and the filter-box "Esc clears" grammar would destroy it.
    tap(&mut a, KeyCode::Esc);
    assert!(!a.help.is_editing());
    assert_eq!(a.help.title, "gpu oo");
}
