// SPDX-License-Identifier: AGPL-3.0-only

//! Which recipe the Library is holding, and that it is the one that launches.
//!
//! The binding is load-bearing rather than cosmetic. Two recipes for the same
//! checkpoint differ in the numbers their measurements were taken under — the
//! 27B pair below differ in context (262144 vs 60000), KV dtype (fp8 vs bf16)
//! and speculation (off vs K=1) — so a card whose settings are read from a
//! DIFFERENT recipe than the one that reaches `serve_args` produces a run whose
//! thresholds were never measured. Every test here follows a value from the
//! card the cursor is on all the way into argv.

use super::*;
use crate::recipe::Recipe;
use crate::tui::data::library::LibraryEntry;

/// Two real recipes for ONE checkpoint. Both are shipped fixtures rather than
/// hand-built structs: their difference is the fact under test.
fn fixture(stem: &str) -> Recipe {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6")
        .join(format!("{stem}.yaml"));
    let text = std::fs::read_to_string(&path).expect("fixture");
    Recipe::parse(format!("qwen3.6/{stem}"), &text).expect("parses")
}

fn plain() -> Recipe {
    fixture("qwen3.6-27b-fp8")
}

fn mtp() -> Recipe {
    fixture("qwen3.6-27b-fp8-mtp")
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

fn state(recipes: Vec<Recipe>) -> LibState {
    let mut models: Vec<String> = recipes.iter().map(|r| r.model.clone()).collect();
    models.sort();
    models.dedup();
    let local: Vec<LibraryEntry> = models.iter().map(|m| with_weights(m)).collect();
    let mut s = LibState {
        index: Index {
            recipes,
            ..Index::default()
        },
        ..LibState::default()
    };
    s.rebuild(&local);
    s
}

/// The value a flag carries in the argv this form would launch.
fn flag(s: &LibState, name: &str) -> Option<String> {
    let argv = s.preview_argv()?;
    let i = argv.iter().position(|a| a == name)?;
    argv.get(i + 1).cloned()
}

#[test]
fn several_recipes_for_one_checkpoint_are_one_row_with_several_cards() {
    // One row per MODEL: repeating the id once per recipe put the choice in a
    // list that has no room to explain it.
    let s = state(vec![mtp(), plain()]);
    assert_eq!(s.rows.len(), 1, "one row: {:?}", s.rows[0].model);
    assert_eq!(s.cards().len(), 2, "both recipes are behind it");
    let ids: Vec<&str> = s.cards().iter().map(|r| r.id.as_str()).collect();
    assert_eq!(
        ids,
        ["qwen3.6/qwen3.6-27b-fp8", "qwen3.6/qwen3.6-27b-fp8-mtp"],
        "in id order, regardless of the order the fetch returned them"
    );
}

#[test]
fn the_card_the_cursor_is_on_is_the_recipe_that_reaches_the_serve_args() {
    // The load-bearing claim. `--max-seq-len` differs by 200k between these two
    // and `--speculative` is present in only one of them, so reading the wrong
    // card is a run whose measurements do not apply.
    let mut s = state(vec![mtp(), plain()]);
    s.open_cards().expect("cards open");

    s.card = 0;
    s.open_config().expect("form opens");
    assert_eq!(
        s.config_recipe().expect("recipe").id,
        "qwen3.6/qwen3.6-27b-fp8"
    );
    assert_eq!(flag(&s, "--max-seq-len").as_deref(), Some("262144"));
    assert_eq!(flag(&s, "--kv-cache-dtype").as_deref(), Some("fp8"));
    let plain_argv = s.preview_argv().expect("renders");
    assert!(
        !plain_argv.iter().any(|a| a == "--speculative"),
        "this recipe measures speculation OFF: {plain_argv:?}"
    );

    s.view = View::Cards;
    s.card = 1;
    s.open_config().expect("form opens");
    assert_eq!(
        s.config_recipe().expect("recipe").id,
        "qwen3.6/qwen3.6-27b-fp8-mtp"
    );
    assert_eq!(flag(&s, "--max-seq-len").as_deref(), Some("60000"));
    assert_eq!(flag(&s, "--kv-cache-dtype").as_deref(), Some("bf16"));
    let mtp_argv = s.preview_argv().expect("renders");
    assert!(
        mtp_argv.iter().any(|a| a == "--speculative"),
        "and this one measures it ON: {mtp_argv:?}"
    );
    assert_eq!(flag(&s, "--num-drafts").as_deref(), Some("1"));
}

#[test]
fn the_form_rows_come_from_the_selected_card_not_the_first_one() {
    // The rows are what the reader edits. Showing recipe A's values while
    // launching recipe B is the same bug one screen earlier.
    let mut s = state(vec![mtp(), plain()]);
    s.open_cards().expect("cards open");
    s.card = 1;
    s.open_config().expect("form opens");

    let rows = s.config_rows();
    let value = |key: &str| {
        rows.iter()
            .find(|(k, _, _)| k == key)
            .map(|(_, v, _)| v.clone())
    };
    assert_eq!(value("max_model_len").as_deref(), Some("60000"));
    assert_eq!(value("speculative").as_deref(), Some("true"));
    assert_eq!(
        rows.len(),
        s.config_recipe().expect("recipe").defaults.len(),
        "every key the recipe carries, and no more"
    );
}

#[test]
fn the_serve_args_a_card_validates_to_match_the_card_it_came_from() {
    // `preview_argv` renders; `serve_args` parses and VALIDATES. Both must
    // describe the same recipe, or the confirm line and the launch disagree.
    let mut s = state(vec![mtp(), plain()]);
    s.open_cards().expect("cards open");
    for (card, want_len, want_spec) in [(0usize, 262144usize, false), (1, 60000, true)] {
        s.view = View::Cards;
        s.card = card;
        s.open_config().expect("form opens");
        let args = s
            .config_recipe()
            .expect("recipe")
            .serve_args(&s.overrides)
            .expect("the recipe is valid");
        assert_eq!(args.max_seq_len, want_len, "card {card}");
        assert_eq!(args.speculative, want_spec, "card {card}");
    }
}

#[test]
fn opening_another_card_starts_from_that_recipes_own_values() {
    // Overrides are keyed as the RECIPE keys them, so carrying them across a
    // card change would apply one recipe's edits to another's settings.
    let mut s = state(vec![mtp(), plain()]);
    s.open_cards().expect("cards open");
    s.card = 0;
    s.open_config().expect("form opens");
    s.row = s
        .config_rows()
        .iter()
        .position(|(k, _, _)| k == "max_model_len")
        .expect("max_model_len");
    s.editing = true;
    s.edit_buffer = "8192".into();
    s.commit_edit();
    assert_eq!(flag(&s, "--max-seq-len").as_deref(), Some("8192"));

    s.view = View::Cards;
    s.card = 1;
    s.open_config().expect("the other form opens");
    assert!(
        s.overrides.is_empty(),
        "the edits did not follow the cursor"
    );
    assert_eq!(
        flag(&s, "--max-seq-len").as_deref(),
        Some("60000"),
        "the second recipe's own value"
    );
}

#[test]
fn a_model_with_no_recipe_has_no_cards_and_no_form() {
    let mut s = LibState::default();
    s.rebuild(&[with_weights("org/local-only")]);
    assert!(s.cards().is_empty());
    assert!(s.selected_card().is_none());
    assert!(s.config_recipe().is_none());
    assert!(s.config_rows().is_empty(), "an empty form, not a blank one");
    assert!(s.preview_argv().is_none());
    assert!(s.open_cards().is_err());
    assert!(s.open_config().is_err(), "and nothing to configure");
}

#[test]
fn committing_an_edit_with_nothing_selected_is_a_no_op() {
    let mut s = LibState {
        view: View::Config,
        edit_buffer: "9100".into(),
        editing: true,
        ..Default::default()
    };
    s.commit_edit();
    assert!(s.overrides.is_empty());
    assert!(s.error.is_none(), "no row means nothing to complain about");
}

#[test]
fn a_non_atlas_recipe_is_listed_and_readable_but_refuses_the_form() {
    // A vLLM recipe still carries the description and params worth rendering;
    // it just cannot be launched from here, and the refusal has to say which
    // runtime it is rather than "no".
    let mut vllm = plain();
    vllm.runtime = Some("vllm".into());
    let mut s = state(vec![vllm]);
    assert!(s.current().expect("a row").has_recipe());
    assert!(
        !s.current().expect("a row").runnable_now(),
        "weights plus a non-atlas recipe is not runnable here"
    );
    s.open_cards().expect("the card still opens");

    let err = s.open_config().expect_err("the form does not");
    assert!(err.contains("vllm"), "names the runtime: {err}");
    assert_eq!(s.view, View::Cards, "and it stays on the card");
    assert!(s.preview_argv().is_none(), "nothing to launch");
}

#[test]
fn the_row_is_described_by_its_atlas_recipe_when_it_has_a_choice() {
    // `primary()` picks the recipe the list row and its detail pane describe.
    // An Atlas recipe wins because it is the one that can actually be started.
    let mut vllm = mtp();
    vllm.runtime = Some("vllm".into());
    let s = state(vec![vllm, plain()]);
    let entry = s.current().expect("a row");
    assert_eq!(
        entry.primary().expect("a primary").id,
        "qwen3.6/qwen3.6-27b-fp8"
    );
    assert!(entry.runnable_now(), "one launchable recipe is enough");
}

#[test]
fn an_override_for_a_key_the_recipe_does_not_have_is_visible_and_unlaunchable() {
    // The form can only produce keys the recipe carries, but the map outlives a
    // rebuild; a stray key must fail loudly rather than be dropped from argv.
    //
    // It is now SHOWN rather than suppressed. `argv` stopped rejecting a key
    // absent from `defaults:` — that is also the shape of a legitimate addition
    // — so the preview renders the stray flag and the operator can SEE the
    // thing that is wrong, instead of an empty preview that says only that
    // something, somewhere, is. Launching still refuses: clap owns the flag
    // surface and rejects the flag by name.
    let mut s = state(vec![plain()]);
    s.open_cards().expect("cards open");
    s.open_config().expect("form opens");
    s.overrides
        .insert("not_a_setting".into(), "whatever".into());
    let preview = s
        .preview_argv()
        .expect("the stray key is shown, not hidden");
    assert!(
        preview.iter().any(|a| a == "--not-a-setting"),
        "the operator must be able to see WHICH key is wrong: {preview:?}"
    );
    assert!(
        s.config_recipe()
            .expect("recipe")
            .serve_args(&s.overrides)
            .is_err(),
        "and it must still be unlaunchable"
    );
}

#[test]
fn a_recipe_that_needs_more_nodes_still_emits_its_world_size() {
    // `min_nodes` is not a `defaults:` key, so a launcher reading only the form
    // rows builds a config the validator rejects.
    let mut wide = plain();
    wide.min_nodes = 2;
    let mut s = state(vec![wide]);
    s.open_cards().expect("cards open");
    s.open_config().expect("form opens");
    assert!(
        !s.config_rows().iter().any(|(k, _, _)| k == "min_nodes"),
        "it is not a form row"
    );
    assert_eq!(flag(&s, "--world-size").as_deref(), Some("2"));
}
