// SPDX-License-Identifier: AGPL-3.0-only

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
        model_type: "qwen3_5_moe".into(),
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
fn j_and_k_move_within_bounds() {
    let mut s = state();
    assert_eq!(s.selected, 0);
    s.on_key(key(KeyCode::Char('k')));
    assert_eq!(s.selected, 0, "cannot go above the first row");
    s.on_key(key(KeyCode::Char('j')));
    assert_eq!(s.selected, 1);
    for _ in 0..10 {
        s.on_key(key(KeyCode::Char('j')));
    }
    assert_eq!(s.selected, s.visible().len() - 1, "clamped to the last row");
}

#[test]
fn enter_opens_the_config_for_a_recipe_row() {
    let mut s = state();
    assert_eq!(s.view, View::List);
    let outcome = s.on_key(key(KeyCode::Enter));
    assert_eq!(outcome, Outcome::None);
    assert_eq!(s.view, View::Config);
}

#[test]
fn enter_on_a_row_without_a_recipe_explains_rather_than_opening_a_blank_form() {
    let mut s = state();
    // The orphan sorts last.
    s.selected = s.visible().len() - 1;
    let outcome = s.on_key(key(KeyCode::Enter));
    assert_eq!(s.view, View::List, "stays put");
    match outcome {
        Outcome::Toast { text, error } => {
            assert!(error);
            assert!(text.contains("no recipe"), "{text}");
        }
        other => panic!("expected a toast, got {other:?}"),
    }
}

#[test]
fn esc_steps_back_one_level_rather_than_dropping_focus() {
    let mut s = state();
    s.on_key(key(KeyCode::Enter));
    assert_eq!(s.view, View::Config);
    s.on_key(key(KeyCode::Esc));
    assert_eq!(s.view, View::List);
}

#[test]
fn the_search_filters_and_esc_clears_it() {
    let mut s = state();
    s.on_key(key(KeyCode::Char('/')));
    assert!(s.filter_editing);
    assert!(s.is_editing(), "global bindings must stand down");
    typed(&mut s, "orphan");
    assert_eq!(s.visible().len(), 1);
    assert_eq!(s.visible()[0].model, "org/orphan");

    s.on_key(key(KeyCode::Esc));
    assert!(!s.filter_editing);
    assert!(s.filter.is_empty(), "Esc clears rather than keeping");
    assert!(s.visible().len() > 1);
}

#[test]
fn a_digit_typed_into_the_search_does_not_jump_sections() {
    // The section jump keys are 1-6; while a filter is open they are text.
    let mut s = state();
    s.on_key(key(KeyCode::Char('/')));
    typed(&mut s, "3");
    assert_eq!(s.filter, "3");
    assert!(s.is_editing());
}

#[test]
fn editing_seeds_the_buffer_with_the_current_value() {
    // Adjusting a setting is the common case; retyping it from scratch is not.
    let mut s = state();
    s.on_key(key(KeyCode::Enter));
    let (key_name, value, _) = s.config_rows().into_iter().next().expect("a row");
    s.on_key(key(KeyCode::Enter));
    assert!(s.editing);
    assert_eq!(s.edit_buffer, value, "seeded from {key_name}");
}

#[test]
fn a_committed_edit_shows_in_the_form_and_the_command() {
    let mut s = state();
    s.on_key(key(KeyCode::Enter));
    s.row = s
        .config_rows()
        .iter()
        .position(|(k, _, _)| k == "port")
        .expect("port");
    s.on_key(key(KeyCode::Enter));
    for _ in 0..8 {
        s.on_key(key(KeyCode::Backspace));
    }
    typed(&mut s, "9100");
    s.on_key(key(KeyCode::Enter));

    assert!(!s.editing);
    assert!(s.error.is_none(), "{:?}", s.error);
    let argv = s.preview_argv().expect("renders");
    let i = argv.iter().position(|a| a == "--port").expect("present");
    assert_eq!(argv[i + 1], "9100");
}

#[test]
fn cancelling_an_edit_keeps_the_committed_value() {
    let mut s = state();
    s.on_key(key(KeyCode::Enter));
    s.on_key(key(KeyCode::Enter));
    typed(&mut s, "garbage");
    s.on_key(key(KeyCode::Esc));
    assert!(!s.editing);
    assert!(s.overrides.is_empty(), "nothing was committed");
}

#[test]
fn d_restores_the_recipes_own_values() {
    let mut s = state();
    s.on_key(key(KeyCode::Enter));
    s.row = s
        .config_rows()
        .iter()
        .position(|(k, _, _)| k == "port")
        .expect("port");
    s.on_key(key(KeyCode::Enter));
    for _ in 0..8 {
        s.on_key(key(KeyCode::Backspace));
    }
    typed(&mut s, "9100");
    s.on_key(key(KeyCode::Enter));
    assert_eq!(s.overrides.len(), 1);

    match s.on_key(key(KeyCode::Char('d'))) {
        Outcome::Toast { error, .. } => assert!(!error),
        other => panic!("expected a toast, got {other:?}"),
    }
    assert!(s.overrides.is_empty());
}

#[test]
fn r_says_so_when_there_is_no_store_rather_than_faking_a_fetch() {
    // Without a store `refresh` is a no-op. Announcing "fetching recipes…"
    // anyway would leave the user waiting for something that never started.
    let mut s = state();
    match s.on_key(key(KeyCode::Char('r'))) {
        Outcome::Toast { text, error } => {
            assert!(error, "it is a failure, not progress");
            assert!(text.contains("store"), "{text}");
        }
        other => panic!("expected a toast, got {other:?}"),
    }
    assert!(!s.fetching);
}

#[test]
fn j_in_the_config_moves_rows_not_the_model_list() {
    let mut s = state();
    s.on_key(key(KeyCode::Enter));
    let before = s.selected;
    s.on_key(key(KeyCode::Char('j')));
    assert_eq!(s.row, 1);
    assert_eq!(s.selected, before, "the model selection is untouched");
}
