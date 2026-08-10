// SPDX-License-Identifier: AGPL-3.0-only

//! The key map's remaining reflexes: the arrow-key aliases, the search field's
//! own editing keys, and the three places where the SAME letter deliberately
//! means different things in different panes.
//!
//! A second file rather than more of `lib_keys_tests.rs`, which pins the happy
//! path through `List → Cards → Config`. These are the edges: a key that must
//! do nothing, a key that must not navigate, and a key whose meaning is
//! pane-dependent by design and would otherwise drift.

use super::*;
use crate::recipe::Recipe;
use crate::tui::data::library::LibraryEntry;

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::from(code)
}

fn typed(s: &mut LibState, text: &str) {
    for c in text.chars() {
        s.on_key(key(KeyCode::Char(c)));
    }
}

fn recipe() -> Recipe {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml");
    Recipe::parse(
        "qwen3.6/flagship",
        &std::fs::read_to_string(path).expect("fixture"),
    )
    .expect("parses")
}

fn local(id: &str) -> LibraryEntry {
    LibraryEntry {
        id: id.into(),
        snapshot_dir: Default::default(),
        size_bytes: 1024,
        has_weights: true,
        model_type: "qwen3_6_moe".into(),
        quant: "fp8".into(),
        layers: 40,
        hidden: 4096,
        heads: 32,
        experts: 128,
        context: 65536,
        optimized: true,
    }
}

fn state() -> LibState {
    let r = recipe();
    let weights = vec![local(&r.model), local("org/orphan")];
    let mut s = LibState::default();
    s.index = crate::recipe::fetch::Index {
        recipes: vec![r],
        ..Default::default()
    };
    s.rebuild(&weights);
    s
}

#[test]
fn the_arrow_keys_do_what_hjkl_do_in_every_pane() {
    // Both sets are taught in the help modal, so a pane that answers one and
    // not the other is a dead key the user has been told about.
    let mut s = state();
    s.on_key(key(KeyCode::Down));
    assert_eq!(s.selected, 1);
    s.on_key(key(KeyCode::Up));
    assert_eq!(s.selected, 0);

    s.on_key(key(KeyCode::Right));
    assert_eq!(s.view, View::Cards, "Right descends like Enter");
    s.on_key(key(KeyCode::Left));
    assert_eq!(s.view, View::List, "Left steps back like Esc");

    s.on_key(key(KeyCode::Char('l')));
    assert_eq!(s.view, View::Cards);
    s.on_key(key(KeyCode::Char('h')));
    assert_eq!(s.view, View::List);
}

#[test]
fn a_key_the_library_does_not_bind_changes_nothing() {
    let mut s = state();
    for code in [
        KeyCode::Tab,
        KeyCode::PageDown,
        KeyCode::Home,
        KeyCode::F(1),
    ] {
        assert_eq!(s.on_key(key(code)), Outcome::None, "{code:?}");
        assert_eq!(s.view, View::List, "{code:?} must not navigate");
        assert_eq!(s.selected, 0, "{code:?} must not move the cursor");
    }
}

#[test]
fn backspace_shortens_the_search_and_enter_keeps_what_was_typed() {
    // Esc CLEARS; Enter commits. Confusing the two loses the filter the reader
    // just typed, or keeps one they meant to abandon.
    let mut s = state();
    s.on_key(key(KeyCode::Char('/')));
    typed(&mut s, "orphanx");
    s.on_key(key(KeyCode::Backspace));
    assert_eq!(s.filter, "orphan");

    s.on_key(key(KeyCode::Enter));
    assert!(!s.filter_editing, "the field gives up the keyboard");
    assert_eq!(s.filter, "orphan", "and keeps the filter");
    assert!(!s.is_editing(), "global bindings resume");
    assert_eq!(s.visible().len(), 1);
}

#[test]
fn backspace_on_an_empty_search_is_not_an_error() {
    let mut s = state();
    s.on_key(key(KeyCode::Char('/')));
    s.on_key(key(KeyCode::Backspace));
    assert!(s.filter.is_empty());
    assert!(s.filter_editing, "and the field is still open");
}

#[test]
fn every_search_keystroke_puts_the_cursor_back_on_the_first_match() {
    // The selection indexes the FILTERED list, so leaving it where it was
    // points past the end as soon as the filter narrows.
    let mut s = state();
    s.selected = 1;
    s.on_key(key(KeyCode::Char('/')));
    typed(&mut s, "o");
    assert_eq!(s.selected, 0);
    s.on_key(key(KeyCode::Backspace));
    assert_eq!(s.selected, 0, "and on the way back out too");
}

#[test]
fn a_key_the_search_field_does_not_use_leaves_the_filter_alone() {
    let mut s = state();
    s.on_key(key(KeyCode::Char('/')));
    typed(&mut s, "orph");
    s.selected = 0;
    assert_eq!(s.on_key(key(KeyCode::Down)), Outcome::None);
    assert_eq!(s.filter, "orph", "unchanged");
    assert!(s.filter_editing, "and the field keeps the keyboard");
}

#[test]
fn r_does_not_start_a_second_fetch_while_one_is_running() {
    // The key is on the render tick's path via a held keyboard repeat; a second
    // fetch would be eight more workers against a 60/hour API budget.
    let mut s = state();
    s.fetching = true;
    assert_eq!(s.on_key(key(KeyCode::Char('r'))), Outcome::None);
    assert!(!s.mark_dirty, "and it does not queue a rescan either");
}

#[test]
fn r_asks_for_a_local_rescan_as_well_as_a_fetch() {
    // "Bring this list up to date" has to include a model that appeared on disk
    // since the last scan, or pressing it on a finished download does nothing.
    let mut s = state();
    s.on_key(key(KeyCode::Char('r')));
    assert!(s.mark_dirty, "the event loop drains this into a rescan");
}

#[test]
fn d_means_restore_defaults_in_the_form_and_download_everywhere_else() {
    // The one letter with two meanings, and the reason the module doc warns
    // about it: in Config it must NOT start a download.
    let mut s = state();
    s.on_key(key(KeyCode::Enter));
    s.on_key(key(KeyCode::Enter));
    assert_eq!(s.view, View::Config);

    match s.on_key(key(KeyCode::Char('d'))) {
        Outcome::Toast { text, error } => {
            assert!(!error);
            assert!(text.contains("restored"), "{text}");
        }
        other => panic!("d in the form must restore defaults, got {other:?}"),
    }
    assert_eq!(s.view, View::Config, "and it does not navigate");
}

#[test]
fn s_launches_only_from_the_form() {
    // Elsewhere `s` is an ordinary unbound letter; starting a model from the
    // list would skip the screen that shows what is about to be started.
    let mut s = state();
    assert_eq!(s.on_key(key(KeyCode::Char('s'))), Outcome::None);
    s.view = View::Cards;
    assert_eq!(s.on_key(key(KeyCode::Char('s'))), Outcome::None);
    s.view = View::Config;
    assert_eq!(s.on_key(key(KeyCode::Char('s'))), Outcome::Launch);
}

#[test]
fn leaving_the_form_clears_the_error_it_was_showing() {
    // The error describes a value in a form that is no longer open; carrying it
    // to the card view attaches a complaint to the wrong screen.
    let mut s = state();
    s.on_key(key(KeyCode::Enter));
    s.on_key(key(KeyCode::Enter));
    s.row = s
        .config_rows()
        .iter()
        .position(|(k, _, _)| k == "scheduling_policy")
        .expect("scheduling_policy");
    s.on_key(key(KeyCode::Enter));
    for _ in 0..40 {
        s.on_key(key(KeyCode::Backspace));
    }
    typed(&mut s, "nonsense");
    s.on_key(key(KeyCode::Enter));
    assert!(s.error.is_some(), "the commit was rejected");

    s.on_key(key(KeyCode::Esc));
    assert_eq!(s.view, View::Cards);
    assert!(s.error.is_none(), "and the complaint goes with the form");
}

#[test]
fn cancelling_an_edit_keeps_an_error_the_previous_commit_earned() {
    // Abandoning a keystroke does not make an earlier rejection untrue.
    let mut s = state();
    s.on_key(key(KeyCode::Enter));
    s.on_key(key(KeyCode::Enter));
    s.error = Some("earlier complaint".into());
    s.on_key(key(KeyCode::Enter));
    assert!(s.editing);
    typed(&mut s, "x");
    s.on_key(key(KeyCode::Esc));

    assert!(!s.editing);
    assert!(s.edit_buffer.is_empty(), "the buffer is discarded");
    assert_eq!(s.error.as_deref(), Some("earlier complaint"));
}

#[test]
fn a_field_being_edited_owns_the_keyboard() {
    let mut s = state();
    s.on_key(key(KeyCode::Enter));
    s.on_key(key(KeyCode::Enter));
    assert!(!s.is_editing(), "not until a row is opened");
    s.on_key(key(KeyCode::Enter));
    assert!(s.is_editing(), "global bindings must stand down");

    // While editing, the pane's own movement keys are text.
    let before = s.row;
    typed(&mut s, "j");
    assert_eq!(s.row, before, "j is a character, not a move");
    assert!(s.edit_buffer.ends_with('j'));
}

#[test]
fn backspace_on_an_empty_edit_buffer_is_not_an_error() {
    let mut s = state();
    s.on_key(key(KeyCode::Enter));
    s.on_key(key(KeyCode::Enter));
    s.on_key(key(KeyCode::Enter));
    for _ in 0..40 {
        s.on_key(key(KeyCode::Backspace));
    }
    assert!(s.edit_buffer.is_empty());
    assert!(s.editing, "still editing, just empty");
}

#[test]
fn moving_between_form_rows_clamps_at_both_ends() {
    let mut s = state();
    s.on_key(key(KeyCode::Enter));
    s.on_key(key(KeyCode::Enter));
    let rows = s.config_rows().len();
    assert!(rows > 1, "the fixture has several settings");

    s.on_key(key(KeyCode::Char('k')));
    assert_eq!(s.row, 0, "clamped at the first row");
    for _ in 0..rows + 5 {
        s.on_key(key(KeyCode::Char('j')));
    }
    assert_eq!(s.row, rows - 1, "and at the last");
}

#[test]
fn the_card_keys_do_not_move_the_model_selection() {
    let mut s = state();
    s.on_key(key(KeyCode::Enter));
    let selected = s.selected;
    s.on_key(key(KeyCode::Char('j')));
    s.on_key(key(KeyCode::Char('k')));
    assert_eq!(s.selected, selected);
    assert_eq!(s.view, View::Cards);
}

#[test]
fn moving_in_an_empty_card_list_is_a_no_op_rather_than_a_panic() {
    // A row can lose its recipes to a refresh while the cards are open.
    let mut s = LibState::default();
    s.view = View::Cards;
    assert!(s.cards().is_empty());
    s.on_key(key(KeyCode::Char('j')));
    s.on_key(key(KeyCode::Char('k')));
    assert_eq!(s.card, 0);
    match s.on_key(key(KeyCode::Enter)) {
        Outcome::Toast { error, .. } => assert!(error, "it explains rather than opening"),
        other => panic!("expected a toast, got {other:?}"),
    }
}

#[test]
fn moving_in_an_empty_form_is_a_no_op_rather_than_a_panic() {
    let mut s = LibState::default();
    s.view = View::Config;
    assert!(s.config_rows().is_empty());
    s.on_key(key(KeyCode::Char('j')));
    assert_eq!(s.row, 0);
    s.on_key(key(KeyCode::Enter));
    assert!(!s.editing, "there is no row to edit");
}

#[test]
fn enter_on_a_non_atlas_card_explains_instead_of_opening_the_form() {
    let mut r = recipe();
    r.runtime = Some("vllm".into());
    let weights = vec![local(&r.model)];
    let mut s = LibState::default();
    s.index = crate::recipe::fetch::Index {
        recipes: vec![r],
        ..Default::default()
    };
    s.rebuild(&weights);

    s.on_key(key(KeyCode::Enter));
    assert_eq!(s.view, View::Cards, "the card still opens");
    match s.on_key(key(KeyCode::Enter)) {
        Outcome::Toast { text, error } => {
            assert!(error);
            assert!(text.contains("vllm"), "names the runtime: {text}");
        }
        other => panic!("expected a toast, got {other:?}"),
    }
    assert_eq!(s.view, View::Cards, "and it stays put");
}
