// SPDX-License-Identifier: AGPL-3.0-only

//! Render tests for the starting-point picker (`lib_start`).
//!
//! Split from `library_tests.rs` at the 500-LoC cap; the fixtures stay there,
//! which is the SSOT for them. State-level honesty rules are in
//! `lib_start_tests`; these cover what actually reaches the screen.

use crate::tui::render::harness::{has, screen};

use super::tests::{lib, local, recipe};

/// The starting-point picker must never read as a list of measured recipes:
/// the chip, the pane title, and the description all say what it is.
#[test]
fn starting_points_are_marked_as_guesses_everywhere_they_render() {
    let donor = recipe("qwen3.6-35b-a3b-fp8-mtp");
    let mut a = lib(vec![donor], vec![local("org/orphan", true)]);
    // Select the no-recipe row (the join sorts it after the recipe row).
    a.lib.selected = a
        .lib
        .visible()
        .iter()
        .position(|e| e.model == "org/orphan")
        .expect("row");
    a.lib.open_cards().expect("opens on starting points");
    let rows = screen(&a, 200, 50);
    assert!(has(&rows, " starting point "), "the chip:\n{rows:#?}");
    assert!(
        has(&rows, "starting points ─"),
        "the pane title uses the honest noun:\n{rows:#?}"
    );
    assert!(
        has(&rows, "not a measurement"),
        "the description says what it is:\n{rows:#?}"
    );
    assert!(
        !has(&rows, "THE FLAGSHIP"),
        "the donor's measured rationale must not survive onto the guess:\n{rows:#?}"
    );

    // And the form behind the card carries the warning to the launch screen.
    a.lib.open_config().expect("configurable");
    let rows = screen(&a, 200, 50);
    assert!(
        has(&rows, "starting point — "),
        "the form is the last honest moment:\n{rows:#?}"
    );
    assert!(has(&rows, "unverified on this model"), "{rows:#?}");
}

/// The list row itself must say the dead end is gone.
#[test]
fn a_no_recipe_row_names_the_way_forward() {
    let a = lib(Vec::new(), vec![local("org/orphan", true)]);
    let rows = screen(&a, 200, 50);
    assert!(has(&rows, "no recipe — ⏎ starting points"), "{rows:#?}");
    assert!(has(&rows, "⏎ pick a starting point"), "{rows:#?}");
}
