// SPDX-License-Identifier: AGPL-3.0-only

//! The pickers' shared keyboard dialect, plus the add-picker's second scroll
//! surface. The list verbs (pick, add, remove, borrow) are exercised end to
//! end in `lib_config_tests` and `lib_borrow_tests`; what lives here is the
//! help panel's scroll — its clamp, its reset on cursor moves, and its
//! refusal to leak into pickers that have no panel.

use crossterm::event::{KeyCode, KeyEvent};

use super::{ConfigModal, HELP_PANEL_TEXT_W};
use crate::recipe::Recipe;
use crate::recipe::fetch::Index;
use crate::tui::data::library::LibraryEntry;
use crate::tui::lib_state::LibState;

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::from(code)
}

fn open_form() -> LibState {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml");
    let r = Recipe::parse(
        "qwen3.6/flagship",
        &std::fs::read_to_string(path).expect("fixture"),
    )
    .expect("parses");
    let weights = vec![LibraryEntry {
        id: r.model.clone(),
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
    }];
    let mut s = LibState::default();
    s.index = Index {
        recipes: vec![r],
        ..Index::default()
    };
    s.rebuild(&weights);
    s.open_cards().expect("cards open");
    s.open_config().expect("form opens");
    s
}

/// The add picker, cursor parked on a flag whose help runs to paragraphs.
fn add_picker_on_long_help() -> LibState {
    let mut s = open_form();
    s.on_key(key(KeyCode::Char('a')));
    let Some(ConfigModal::Add { fields, .. }) = &s.modal else {
        panic!("add picker open");
    };
    let target = fields
        .iter()
        .position(|f| f.key == "check_kernels")
        .expect("check_kernels is addable and long-documented");
    for _ in 0..target {
        s.on_key(key(KeyCode::Char('j')));
    }
    s
}

fn help_scroll(s: &LibState) -> usize {
    match &s.modal {
        Some(ConfigModal::Add { help_scroll, .. }) => *help_scroll,
        other => panic!("expected the add picker, got {other:?}"),
    }
}

#[test]
fn shift_j_scrolls_the_help_panel_and_shift_k_scrolls_it_back() {
    let mut s = add_picker_on_long_help();
    assert_eq!(help_scroll(&s), 0);
    s.on_key(key(KeyCode::Char('J')));
    s.on_key(key(KeyCode::Char('J')));
    assert_eq!(help_scroll(&s), 2, "J moves the panel, not the list");
    s.on_key(key(KeyCode::Char('K')));
    assert_eq!(help_scroll(&s), 1);
    // And the LIST cursor did not move: the shifted pair is the panel's.
    let Some(ConfigModal::Add {
        fields, selected, ..
    }) = &s.modal
    else {
        panic!("still the add picker");
    };
    assert_eq!(fields[*selected].key, "check_kernels");
}

#[test]
fn the_panel_scroll_is_clamped_at_both_ends() {
    let mut s = add_picker_on_long_help();
    s.on_key(key(KeyCode::Char('K')));
    assert_eq!(help_scroll(&s), 0, "no scroll above the first line");
    // Bank far more J than the text has lines: the clamp is what keeps the
    // next K responsive instead of paying back dead presses one at a time.
    let wrapped = {
        let Some(ConfigModal::Add {
            fields, selected, ..
        }) = &s.modal
        else {
            unreachable!()
        };
        crate::tui::format::wrap_help(&fields[*selected].help_full, HELP_PANEL_TEXT_W).len()
    };
    for _ in 0..wrapped + 50 {
        s.on_key(key(KeyCode::Char('J')));
    }
    assert_eq!(help_scroll(&s), wrapped - 1, "clamped at the last line");
    s.on_key(key(KeyCode::Char('K')));
    assert_eq!(help_scroll(&s), wrapped - 2, "K acts immediately");
}

#[test]
fn moving_the_cursor_resets_the_panel_to_the_top_of_the_new_help() {
    let mut s = add_picker_on_long_help();
    s.on_key(key(KeyCode::Char('J')));
    assert_eq!(help_scroll(&s), 1);
    s.on_key(key(KeyCode::Char('j')));
    assert_eq!(
        help_scroll(&s),
        0,
        "a panel still scrolled into the previous flag's help would caption \
         one flag with another's paragraphs"
    );
}

#[test]
fn the_shifted_pair_is_inert_on_pickers_without_a_panel() {
    let mut s = open_form();
    s.row = s
        .config_rows()
        .iter()
        .position(|r| r.key == "kv_cache_dtype")
        .expect("row");
    s.on_key(key(KeyCode::Enter));
    let before = format!("{:?}", s.modal);
    s.on_key(key(KeyCode::Char('J')));
    s.on_key(key(KeyCode::Char('K')));
    assert_eq!(
        format!("{:?}", s.modal),
        before,
        "the option picker has no panel for J/K to move"
    );
}
