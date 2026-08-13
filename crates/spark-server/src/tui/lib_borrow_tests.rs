// SPDX-License-Identifier: AGPL-3.0-only

//! Borrowing, followed into the state it changes and the state it must not.
//!
//! The defect classes under test: values overwritten without a preview,
//! provenance presented as measurement, a donor list offering a recipe that
//! cannot serve here, and an undo (`d`) that leaves the provenance line
//! standing over values it no longer describes.

use crossterm::event::{KeyCode, KeyEvent};

use crate::recipe::Recipe;
use crate::recipe::fetch::Index;
use crate::tui::data::library::LibraryEntry;
use crate::tui::lib_keys::Outcome;
use crate::tui::lib_modal::ConfigModal;
use crate::tui::lib_state::LibState;

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::from(code)
}

fn fixture(stem: &str) -> Recipe {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6")
        .join(format!("{stem}.yaml"));
    Recipe::parse(
        format!("qwen3.6/{stem}"),
        &std::fs::read_to_string(path).expect("fixture"),
    )
    .expect("parses")
}

fn weights(model: &str) -> LibraryEntry {
    LibraryEntry {
        id: model.to_string(),
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

/// The flagship's form open, with the 27B recipe in the index as a donor.
fn form_with_donor() -> LibState {
    let flagship = fixture("qwen3.6-35b-a3b-fp8-mtp");
    let donor = fixture("qwen3.6-27b-nvfp4");
    let locals = vec![weights(&flagship.model)];
    let mut s = LibState::default();
    s.index = Index {
        recipes: vec![flagship, donor],
        ..Index::default()
    };
    s.rebuild(&locals);
    // Both recipes are in the index but only the flagship's model is local,
    // so the list has one row and its cards are the flagship's.
    s.open_cards().expect("cards open");
    s.open_config().expect("form opens");
    s
}

fn effective(s: &LibState, key: &str) -> String {
    s.config_rows()
        .into_iter()
        .find(|r| r.key == key)
        .unwrap_or_else(|| panic!("{key} is not on the form"))
        .value
}

#[test]
fn b_lists_donors_without_the_forms_own_recipe() {
    let mut s = form_with_donor();
    s.on_key(key(KeyCode::Char('b')));
    let Some(ConfigModal::Borrow { donors, .. }) = &s.modal else {
        panic!("b opens the borrow picker, got {:?}", s.modal);
    };
    assert_eq!(donors.len(), 1, "only the 27B donor");
    assert_eq!(donors[0].id, "qwen3.6/qwen3.6-27b-nvfp4");
    // The donor's own model rides along: it is the provenance the preview
    // and the form line will warn with.
    assert_eq!(donors[0].model, "nvidia/Qwen3.6-27B-NVFP4");
}

#[test]
fn vllm_and_multi_node_recipes_are_never_offered_as_donors() {
    let mut s = form_with_donor();
    let mut vllm = fixture("qwen3.6-27b-fp8");
    vllm.id = "qwen3.6/vllm-donor".into();
    vllm.runtime = Some("vllm".into());
    let mut ep2 = fixture("qwen3.6-27b-fp8");
    ep2.id = "qwen3.6/ep2-donor".into();
    ep2.min_nodes = 2;
    s.index.recipes.push(vllm);
    s.index.recipes.push(ep2);
    s.on_key(key(KeyCode::Char('b')));
    let Some(ConfigModal::Borrow { donors, .. }) = &s.modal else {
        panic!("borrow picker open");
    };
    // The same exclusion `lib_start` applies, through the same function: a
    // vLLM donor cannot serve from here and an EP donor's config the
    // single-node launcher refuses — settings from either are a dead end.
    assert!(
        donors.iter().all(|d| d.id.contains("27b-nvfp4")),
        "only loadable donors offered: {:?}",
        donors.iter().map(|d| &d.id).collect::<Vec<_>>()
    );
}

#[test]
fn enter_on_a_donor_previews_and_commits_nothing() {
    let mut s = form_with_donor();
    let before = s.config_rows();
    s.on_key(key(KeyCode::Char('b')));
    s.on_key(key(KeyCode::Enter));
    let Some(ConfigModal::Preview { changes, .. }) = &s.modal else {
        panic!("Enter on a donor opens the preview, got {:?}", s.modal);
    };
    // Only differing keys are listed: identical rows would bury the ones the
    // user is being asked to approve.
    assert!(
        changes.iter().any(|c| c.key == "max_model_len"),
        "the donor's 32768 differs from the recipe's 65536: {changes:#?}"
    );
    assert!(
        !changes.iter().any(|c| c.key == "port"),
        "both recipes pin port 8888 — no change to approve: {changes:#?}"
    );
    // A key the recipe does not carry reads "not set", not an invented value.
    let parser = changes
        .iter()
        .find(|c| c.key == "tool_call_parser")
        .expect("the donor adds tool_call_parser");
    assert_eq!(parser.from, "not set");
    assert_eq!(parser.to, "qwen3_coder");
    // And the form is untouched until the preview's own Enter.
    assert_eq!(s.config_rows(), before, "preview commits nothing");
    assert!(s.overrides.is_empty());
    assert!(s.borrowed.is_none());
}

#[test]
fn a_user_edit_the_donor_would_overwrite_is_shown_before_it_is_lost() {
    let mut s = form_with_donor();
    s.overrides
        .insert("gpu_memory_utilization".into(), "0.75".into());
    s.on_key(key(KeyCode::Char('b')));
    s.on_key(key(KeyCode::Enter));
    let Some(ConfigModal::Preview { changes, .. }) = &s.modal else {
        panic!("preview open");
    };
    let gpu = changes
        .iter()
        .find(|c| c.key == "gpu_memory_utilization")
        .expect("the edited row is in the preview");
    // The FROM is the user's edit, not the recipe value: what the preview
    // shows being replaced must be what is actually on the form.
    assert_eq!(gpu.from, "0.75");
    assert_eq!(gpu.to, "0.85");
}

#[test]
fn applying_the_preview_takes_the_donor_values_and_records_provenance() {
    let mut s = form_with_donor();
    // A removed row the donor names: borrowing must re-pin it, because
    // "removed" plus an override is a state the form forbids.
    s.on_key(key(KeyCode::Char('b')));
    s.on_key(key(KeyCode::Enter));
    let outcome = s.on_key(key(KeyCode::Enter));
    assert!(
        matches!(&outcome, Outcome::Toast { error: false, text } if text.contains("borrowed")),
        "apply reports what it did: {outcome:?}"
    );
    assert!(s.modal.is_none(), "the modal closed on apply");
    assert_eq!(effective(&s, "max_model_len"), "32768");
    assert_eq!(effective(&s, "tool_call_parser"), "qwen3_coder");
    // Provenance names the donor AND the model it was measured on — the two
    // halves of "these values are not a measurement for this model".
    let borrowed = s.borrowed.as_deref().expect("provenance recorded");
    assert!(borrowed.contains("qwen3.6/qwen3.6-27b-nvfp4"), "{borrowed}");
    assert!(borrowed.contains("nvidia/Qwen3.6-27B-NVFP4"), "{borrowed}");
    // And the launch line agrees with the form: the borrow's whole point.
    let argv = s.preview_argv().expect("launchable").join(" ");
    assert!(argv.contains("--max-seq-len 32768"), "{argv}");
    assert!(argv.contains("--tool-call-parser qwen3_coder"), "{argv}");
}

#[test]
fn borrowing_re_pins_a_removed_row_rather_than_stranding_it() {
    let mut s = form_with_donor();
    let i = s
        .config_rows()
        .iter()
        .position(|r| r.key == "max_model_len")
        .expect("row");
    s.row = i;
    s.on_key(key(KeyCode::Char('x')));
    assert!(s.removed.contains("max_model_len"), "removed first");
    s.on_key(key(KeyCode::Char('b')));
    s.on_key(key(KeyCode::Enter));
    s.on_key(key(KeyCode::Enter));
    assert!(
        !s.removed.contains("max_model_len"),
        "the donor names a value for it, so the flag is passed again"
    );
    assert_eq!(effective(&s, "max_model_len"), "32768");
}

#[test]
fn d_undoes_a_borrow_completely_values_and_provenance_together() {
    let mut s = form_with_donor();
    s.on_key(key(KeyCode::Char('b')));
    s.on_key(key(KeyCode::Enter));
    s.on_key(key(KeyCode::Enter));
    assert!(s.borrowed.is_some(), "borrow applied");
    s.on_key(key(KeyCode::Char('d')));
    assert!(s.overrides.is_empty(), "values back to the recipe");
    assert!(
        s.borrowed.is_none(),
        "a provenance line outliving the values it described would warn about \
         settings that are no longer on the form"
    );
    assert_eq!(effective(&s, "max_model_len"), "65536");
}

#[test]
fn esc_from_the_preview_returns_to_the_donor_list_not_the_form() {
    let mut s = form_with_donor();
    s.on_key(key(KeyCode::Char('b')));
    s.on_key(key(KeyCode::Enter));
    assert!(matches!(s.modal, Some(ConfigModal::Preview { .. })));
    s.on_key(key(KeyCode::Esc));
    assert!(
        matches!(s.modal, Some(ConfigModal::Borrow { .. })),
        "one Esc backs out of the preview, keeping the choice in progress"
    );
    s.on_key(key(KeyCode::Esc));
    assert!(s.modal.is_none(), "the second Esc closes the picker");
    assert!(s.is_editing() || s.modal.is_none());
}

#[test]
fn an_invalid_donor_is_refused_whole_and_the_form_is_untouched() {
    let mut s = form_with_donor();
    // A donor carrying a key that is not a serve flag: rendering it would
    // change nothing, which `argv_edited` refuses by design.
    let mut bad = fixture("qwen3.6-27b-nvfp4");
    bad.id = "qwen3.6/bad-donor".into();
    bad.defaults
        .insert("mystery_knob".into(), "definitely".into());
    s.index.recipes.push(bad);
    s.on_key(key(KeyCode::Char('b')));
    // The donor list is sorted; put the cursor on the bad donor.
    let Some(ConfigModal::Borrow { donors, .. }) = &s.modal else {
        panic!("borrow picker open");
    };
    let i = donors
        .iter()
        .position(|d| d.id == "qwen3.6/bad-donor")
        .expect("bad donor listed");
    for _ in 0..i {
        s.on_key(key(KeyCode::Char('j')));
    }
    s.on_key(key(KeyCode::Enter));
    s.on_key(key(KeyCode::Enter));
    assert!(s.error.is_some(), "the refusal says why");
    assert!(s.overrides.is_empty(), "nothing applied");
    assert!(
        s.borrowed.is_none(),
        "no provenance for a borrow that did not happen"
    );
}

#[test]
fn a_second_enter_on_an_identical_donor_says_nothing_to_change() {
    let mut s = form_with_donor();
    // A donor identical to the recipe's own settings.
    let mut twin = fixture("qwen3.6-35b-a3b-fp8-mtp");
    twin.id = "qwen3.6/twin".into();
    s.index.recipes.push(twin);
    s.on_key(key(KeyCode::Char('b')));
    let Some(ConfigModal::Borrow { donors, .. }) = &s.modal else {
        panic!("borrow picker open");
    };
    let i = donors
        .iter()
        .position(|d| d.id == "qwen3.6/twin")
        .expect("twin listed");
    for _ in 0..i {
        s.on_key(key(KeyCode::Char('j')));
    }
    let outcome = s.on_key(key(KeyCode::Enter));
    assert!(
        matches!(&outcome, Outcome::Toast { error: false, text } if text.contains("already matches")),
        "an empty preview would be a box the user has to interpret: {outcome:?}"
    );
}
