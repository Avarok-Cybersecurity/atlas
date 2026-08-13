// SPDX-License-Identifier: AGPL-3.0-only

//! The Config form's three new verbs — pick, add, remove — followed all the
//! way into the argv, because every one of them exists to change what
//! launches. A test that stops at the form state would pass while the
//! command line quietly kept the old value; the launch command is what these
//! edits are FOR.

use crossterm::event::{KeyCode, KeyEvent};

use crate::recipe::Recipe;
use crate::recipe::fetch::Index;
use crate::tui::data::library::LibraryEntry;
use crate::tui::lib_modal::ConfigModal;
use crate::tui::lib_state::{LibState, View};

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::from(code)
}

fn typed(s: &mut LibState, text: &str) {
    for c in text.chars() {
        s.on_key(key(KeyCode::Char(c)));
    }
}

/// The shipped flagship fixture: carries an enumerated field with a non-first
/// value (`kv_cache_dtype: bf16`), a numeric one (`port: 8888`), a policy
/// (`scheduling_policy: slai`) and `speculative: true` — every shape the form
/// dispatches on.
fn recipe() -> Recipe {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml");
    Recipe::parse(
        "qwen3.6/flagship",
        &std::fs::read_to_string(path).expect("fixture"),
    )
    .expect("parses")
}

fn open_form() -> LibState {
    let r = recipe();
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

fn row_of(s: &LibState, key: &str) -> usize {
    s.config_rows()
        .iter()
        .position(|r| r.key == key)
        .unwrap_or_else(|| panic!("{key} is not on the form"))
}

fn flag_value(s: &LibState, flag: &str) -> Option<String> {
    let argv = s.preview_argv()?;
    let i = argv.iter().position(|a| a == flag)?;
    argv.get(i + 1).cloned()
}

// ── the picker ──

#[test]
fn enter_on_an_enumerated_field_opens_the_picker_not_the_editor() {
    let mut s = open_form();
    s.row = row_of(&s, "kv_cache_dtype");
    s.on_key(key(KeyCode::Enter));
    assert!(!s.editing, "a closed set is picked, not typed");
    let Some(ConfigModal::Options {
        key: k,
        options,
        selected,
    }) = &s.modal
    else {
        panic!("no picker opened: {:?}", s.modal);
    };
    assert_eq!(k, "kv_cache_dtype");
    // The whole point: the list IS the enum, not a copy of it.
    let expected: Vec<String> = spark_runtime::kv_cache::KvCacheDtype::ALL
        .iter()
        .map(|d| d.name().to_string())
        .collect();
    assert_eq!(options, &expected);
    assert_eq!(
        options[*selected], "bf16",
        "the cursor starts on the value the form currently has"
    );
}

#[test]
fn picking_a_value_lands_in_the_form_and_the_command() {
    let mut s = open_form();
    s.row = row_of(&s, "kv_cache_dtype");
    s.on_key(key(KeyCode::Enter));
    // g = top of the list, which is bf16's neighbourhood; walk to fp8.
    s.on_key(key(KeyCode::Char('g')));
    s.on_key(key(KeyCode::Char('j')));
    s.on_key(key(KeyCode::Enter));
    assert!(s.modal.is_none(), "the picker closes on selection");
    assert!(s.error.is_none(), "{:?}", s.error);
    assert_eq!(flag_value(&s, "--kv-cache-dtype").as_deref(), Some("fp8"));
    let row = &s.config_rows()[row_of(&s, "kv_cache_dtype")];
    assert!(row.changed, "the row is marked changed");
}

#[test]
fn esc_closes_the_picker_without_touching_the_value() {
    let mut s = open_form();
    s.row = row_of(&s, "kv_cache_dtype");
    s.on_key(key(KeyCode::Enter));
    s.on_key(key(KeyCode::Char('j')));
    s.on_key(key(KeyCode::Esc));
    assert!(s.modal.is_none());
    assert!(s.overrides.is_empty(), "cancel commits nothing");
    assert_eq!(flag_value(&s, "--kv-cache-dtype").as_deref(), Some("bf16"));
}

#[test]
fn a_boolean_field_is_a_two_entry_picker() {
    let mut s = open_form();
    s.row = row_of(&s, "speculative");
    s.on_key(key(KeyCode::Enter));
    let Some(ConfigModal::Options { options, .. }) = &s.modal else {
        panic!("no picker for a boolean: {:?}", s.modal);
    };
    assert_eq!(options, &["true", "false"]);
}

// ── free typing stays free ──

#[test]
fn enter_on_a_numeric_field_opens_the_text_editor() {
    let mut s = open_form();
    s.row = row_of(&s, "port");
    s.on_key(key(KeyCode::Enter));
    assert!(s.editing);
    assert!(s.modal.is_none());
    assert_eq!(s.edit_buffer, "8888", "seeded with the current value");
}

#[test]
fn garbage_in_a_numeric_field_is_rejected_at_commit_not_at_launch() {
    let mut s = open_form();
    s.row = row_of(&s, "port");
    s.on_key(key(KeyCode::Enter));
    for _ in 0..8 {
        s.on_key(key(KeyCode::Backspace));
    }
    typed(&mut s, "not-a-port");
    s.on_key(key(KeyCode::Enter));
    assert!(
        s.error.is_some(),
        "the bad value is named now, not at launch"
    );
    assert!(s.overrides.is_empty(), "and it never enters the form");
    assert_eq!(flag_value(&s, "--port").as_deref(), Some("8888"));
}

// ── adding ──

#[test]
fn a_opens_the_add_list_and_a_default_carrying_flag_lands_at_its_default() {
    let mut s = open_form();
    s.on_key(key(KeyCode::Char('a')));
    let Some(ConfigModal::Add { fields, .. }) = &s.modal else {
        panic!("no add list: {:?}", s.modal);
    };
    let target = fields
        .iter()
        .position(|f| f.key == "block_size")
        .expect("block_size is addable");
    for _ in 0..target {
        s.on_key(key(KeyCode::Char('j')));
    }
    s.on_key(key(KeyCode::Enter));
    assert!(s.modal.is_none());
    let row = &s.config_rows()[row_of(&s, "block_size")];
    assert!(row.added, "marked as an addition, not a recipe value");
    // 16 is clap's declared default, read from clap — not written here twice.
    assert_eq!(
        row.value,
        crate::tui::lib_fields::spec_for_key("block_size")
            .unwrap()
            .default
            .clone()
            .unwrap()
    );
    assert_eq!(flag_value(&s, "--block-size").as_deref(), Some("16"));
    assert_eq!(
        s.row,
        row_of(&s, "block_size"),
        "the cursor follows the add"
    );
}

#[test]
fn the_add_list_omits_flags_already_on_the_form_including_renames() {
    let s = {
        let mut s = open_form();
        s.on_key(key(KeyCode::Char('a')));
        s
    };
    let Some(ConfigModal::Add { fields, .. }) = &s.modal else {
        panic!("no add list");
    };
    for f in fields {
        assert_ne!(f.key, "kv_cache_dtype", "already a recipe row");
        // The recipe pins `max_model_len`, the vLLM spelling of
        // `--max-seq-len`; offering `max_seq_len` too would render the same
        // flag twice and fail the parse at commit.
        assert_ne!(f.flag, "max-seq-len", "covered by max_model_len");
    }
}

#[test]
fn adding_a_flag_with_no_default_asks_for_a_value_first() {
    let mut s = open_form();
    s.on_key(key(KeyCode::Char('a')));
    let Some(ConfigModal::Add { fields, .. }) = &s.modal else {
        panic!("no add list");
    };
    let target = fields
        .iter()
        .position(|f| f.key == "model_name")
        .expect("model_name is addable");
    for _ in 0..target {
        s.on_key(key(KeyCode::Char('j')));
    }
    s.on_key(key(KeyCode::Enter));
    assert_eq!(s.pending_add.as_deref(), Some("model_name"));
    assert!(s.editing, "the value is asked for, not invented");
    assert!(s.overrides.is_empty(), "nothing exists until the commit");

    // Esc: the half-typed addition evaporates without trace.
    typed(&mut s, "half");
    s.on_key(key(KeyCode::Esc));
    assert!(s.pending_add.is_none());
    assert!(s.overrides.is_empty());
    assert!(!s.config_rows().iter().any(|r| r.key == "model_name"));

    // And the committed path lands on the form and in the command.
    s.on_key(key(KeyCode::Char('a')));
    for _ in 0..target {
        s.on_key(key(KeyCode::Char('j')));
    }
    s.on_key(key(KeyCode::Enter));
    typed(&mut s, "prod-alias");
    s.on_key(key(KeyCode::Enter));
    assert!(s.error.is_none(), "{:?}", s.error);
    assert_eq!(
        flag_value(&s, "--model-name").as_deref(),
        Some("prod-alias")
    );
}

#[test]
fn x_on_an_added_setting_simply_unadds_it() {
    let mut s = open_form();
    s.overrides.insert("block_size".into(), "32".into());
    s.row = row_of(&s, "block_size");
    s.on_key(key(KeyCode::Char('x')));
    assert!(!s.config_rows().iter().any(|r| r.key == "block_size"));
    assert_eq!(flag_value(&s, "--block-size"), None);
    assert!(s.row < s.config_rows().len(), "the cursor is re-clamped");
}

// ── removing ──

#[test]
fn x_removes_a_recipe_setting_and_its_flag_leaves_the_command() {
    let mut s = open_form();
    s.row = row_of(&s, "scheduling_policy");
    s.on_key(key(KeyCode::Char('x')));
    let row = &s.config_rows()[row_of(&s, "scheduling_policy")];
    assert!(row.removed, "the row stays visible, marked removed");
    assert_eq!(
        flag_value(&s, "--scheduling-policy"),
        None,
        "removed means NOT PASSED — the server default applies"
    );
}

#[test]
fn x_again_restores_the_recipes_value() {
    let mut s = open_form();
    s.row = row_of(&s, "scheduling_policy");
    s.on_key(key(KeyCode::Char('x')));
    s.on_key(key(KeyCode::Char('x')));
    let row = &s.config_rows()[row_of(&s, "scheduling_policy")];
    assert!(!row.removed);
    assert_eq!(
        flag_value(&s, "--scheduling-policy").as_deref(),
        Some("slai")
    );
}

#[test]
fn enter_on_a_removed_row_restores_it_too() {
    let mut s = open_form();
    s.row = row_of(&s, "scheduling_policy");
    s.on_key(key(KeyCode::Char('x')));
    s.on_key(key(KeyCode::Enter));
    assert!(s.modal.is_none(), "restore, not a picker over a ghost row");
    assert!(!s.editing);
    assert!(!s.config_rows()[row_of(&s, "scheduling_policy")].removed);
}

#[test]
fn removing_a_changed_row_drops_its_override_and_restore_returns_the_recipe_value() {
    let mut s = open_form();
    s.overrides.insert("port".into(), "9100".into());
    s.row = row_of(&s, "port");
    s.on_key(key(KeyCode::Char('x')));
    assert!(s.overrides.is_empty(), "removal un-pins, it does not park");
    assert_eq!(flag_value(&s, "--port"), None);
    s.on_key(key(KeyCode::Char('x')));
    assert_eq!(
        flag_value(&s, "--port").as_deref(),
        Some("8888"),
        "restore returns to the RECIPE, not to the dropped edit"
    );
}

#[test]
fn d_restores_removals_along_with_edits() {
    let mut s = open_form();
    s.row = row_of(&s, "scheduling_policy");
    s.on_key(key(KeyCode::Char('x')));
    s.on_key(key(KeyCode::Char('d')));
    assert!(s.removed.is_empty());
    assert_eq!(
        flag_value(&s, "--scheduling-policy").as_deref(),
        Some("slai")
    );
}

#[test]
fn a_removal_that_breaks_the_config_is_refused_with_the_reason() {
    let mut s = open_form();
    // `--num-drafts 2` is only legal beside a speculative method; the recipe
    // pins `speculative: true`, so the pair is valid…
    s.overrides.insert("num_drafts".into(), "2".into());
    assert!(s.preview_argv().is_some());
    // …and removing `speculative` would strand it. The removal must bounce,
    // not launch a config the validator rejects.
    s.row = row_of(&s, "speculative");
    s.on_key(key(KeyCode::Char('x')));
    assert!(
        !s.config_rows()[row_of(&s, "speculative")].removed,
        "the removal was refused"
    );
    assert!(s.error.is_some(), "and the reason is on the form");
}

#[test]
fn view_state_resets_between_recipes() {
    let mut s = open_form();
    s.row = row_of(&s, "scheduling_policy");
    s.on_key(key(KeyCode::Char('x')));
    s.on_key(key(KeyCode::Esc));
    assert_eq!(s.view, View::Cards);
    s.open_config().expect("form reopens");
    assert!(
        s.removed.is_empty(),
        "a reopened form starts from the recipe, not from stale removals"
    );
}
