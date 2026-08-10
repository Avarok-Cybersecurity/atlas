// SPDX-License-Identifier: AGPL-3.0-only

//! What the Library refuses to do, and what it says instead.
//!
//! A second test file for `lib_state` rather than a first for `lib_keys`,
//! because these reach private state — and because the claims they pin are
//! claims the UI makes about ITSELF. A refusal that says "press Esc, then d"
//! is an assertion about navigation, and it was wrong when first written: the
//! config pane escapes to Cards, not to the list, and `d` did nothing there.
//! The advice and the key map live in different files, so nothing but a test
//! keeps them honest.

use super::*;
use crate::recipe::Recipe;
use crate::recipe::fetch::Index;
use crate::tui::data::library::LibraryEntry;

fn real_recipe() -> Recipe {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml");
    let text = std::fs::read_to_string(&path).expect("fixture");
    Recipe::parse("qwen3.6/flagship", &text).expect("parses")
}

fn with_weights(model: &str) -> LibraryEntry {
    LibraryEntry {
        id: model.into(),
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

fn state_with_recipe() -> LibState {
    let recipe = real_recipe();
    let local = vec![with_weights(&recipe.model)];
    let mut s = LibState {
        index: Index {
            recipes: vec![recipe],
            ..Index::default()
        },
        ..LibState::default()
    };
    s.rebuild(&local);
    s
}

#[test]
fn starting_a_recipe_whose_weights_are_missing_is_refused_before_anything_is_torn_down() {
    // The reported flow: press start, the swap tears down whatever was
    // serving, the checklist resets, the dashboard jumps to Main — and only
    // then does `resolve_model_dir` report that the model is not there, with
    // nothing to do about it. Now that weights are downloadable, the honest
    // answer is to refuse up front and name the action.
    // A recipe with NO local weights — the case the Library now exists to fix.
    let recipe = real_recipe();
    let mut s = LibState {
        index: Index {
            recipes: vec![recipe],
            ..Index::default()
        },
        ..LibState::default()
    };
    s.rebuild(&[]);
    s.view = View::Config;
    assert!(!s.selected_has_weights(), "nothing is on disk");

    let err = s
        .launch(std::sync::Arc::new(
            crate::main_modules::model_host::ModelHost::empty(),
        ))
        .expect_err("a model that is not downloaded cannot be started");
    assert!(err.contains("not downloaded"), "{err}");
    assert!(err.contains('d'), "and names the download key: {err}");

    // And with weights present it gets past the check (whatever the swap then
    // does is the swap's business, not this guard's).
    let mut ok = state_with_recipe();
    ok.view = View::Config;
    assert!(ok.selected_has_weights(), "the fixture has weights");
}

#[test]
fn the_download_advice_matches_the_number_of_escapes_it_actually_takes() {
    // The refusal message tells the reader to "press Esc, then d". That is a
    // claim about navigation, and it was WRONG when written: Config escapes to
    // Cards, not to the list, and `d` did nothing in Cards — so following the
    // instruction literally did nothing at all. Pinned here because the advice
    // and the key map live in different files and will drift again otherwise.
    use crossterm::event::{KeyCode, KeyEvent};
    let mut s = state_with_recipe();
    s.view = View::Config;

    // One Esc, exactly as the message says.
    s.on_key(KeyEvent::from(KeyCode::Esc));
    assert_eq!(s.view, View::Cards, "one Esc leaves the form");

    // ...and `d` must do something HERE, where that Esc landed.
    let outcome = s.on_key(KeyEvent::from(KeyCode::Char('d')));
    assert!(
        matches!(outcome, crate::tui::lib_keys::Outcome::Download),
        "d must start a download from the pane one Esc away from the form"
    );
}

#[test]
fn the_download_keys_work_in_both_panes_that_show_a_model() {
    use crossterm::event::{KeyCode, KeyEvent};
    for view in [View::List, View::Cards] {
        let mut s = state_with_recipe();
        s.view = view;
        for (key, want) in [
            ('d', crate::tui::lib_keys::Outcome::Download),
            ('x', crate::tui::lib_keys::Outcome::CancelDownload),
            ('u', crate::tui::lib_keys::Outcome::CheckFresh),
        ] {
            let got = s.on_key(KeyEvent::from(KeyCode::Char(key)));
            assert_eq!(
                std::mem::discriminant(&got),
                std::mem::discriminant(&want),
                "{key:?} in {view:?}"
            );
            assert_eq!(s.view, view, "{key:?} must not navigate");
        }
    }
}
