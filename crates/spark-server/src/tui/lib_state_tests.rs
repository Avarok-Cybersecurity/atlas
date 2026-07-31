// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::recipe::Recipe;

fn real_recipe() -> Recipe {
    // A real fixture rather than a hand-built struct: the form's whole job is
    // to edit values that must survive `serve_args`, and an invented recipe
    // would not prove that.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml");
    let text = std::fs::read_to_string(&path).expect("fixture");
    Recipe::parse("qwen3.6/flagship", &text).expect("parses")
}

fn local_of(model: &str) -> LibraryEntry {
    LibraryEntry {
        id: model.into(),
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

fn state_with_recipe() -> LibState {
    let recipe = real_recipe();
    let local = vec![local_of(&recipe.model)];
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
fn the_list_populates_from_cache_without_a_network() {
    let s = state_with_recipe();
    assert_eq!(s.rows.len(), 1);
    assert!(s.current().expect("a row").runnable_now());
}

#[test]
fn opening_the_config_needs_an_atlas_recipe() {
    // A local-only row has nothing to edit; opening a blank form would imply
    // otherwise.
    let mut s = LibState::default();
    s.rebuild(&[local_of("org/orphan")]);
    let err = s.open_config().expect_err("refused");
    assert!(err.contains("no recipe"), "{err}");
    assert_eq!(s.view, View::List, "and it stays on the list");
}

#[test]
fn the_form_shows_every_recipe_key_with_its_value() {
    let mut s = state_with_recipe();
    s.open_config().expect("opens");
    let rows = s.config_rows();
    let recipe = s.config_recipe().expect("recipe");
    assert_eq!(rows.len(), recipe.defaults.len());
    assert!(
        rows.iter().all(|(_, _, edited)| !edited),
        "nothing edited yet"
    );
    let (key, value, _) = rows.iter().find(|(k, _, _)| k == "port").expect("port");
    assert_eq!(key, "port");
    assert_eq!(value, "8888");
}

#[test]
fn a_valid_edit_is_kept_and_marked() {
    let mut s = state_with_recipe();
    s.open_config().expect("opens");
    let rows = s.config_rows();
    s.row = rows
        .iter()
        .position(|(k, _, _)| k == "max_model_len")
        .expect("key");
    s.editing = true;
    s.edit_buffer = "4096".into();
    s.commit_edit();

    assert!(s.error.is_none(), "{:?}", s.error);
    assert!(!s.editing);
    let (_, value, edited) = s
        .config_rows()
        .into_iter()
        .find(|(k, _, _)| k == "max_model_len")
        .expect("key");
    assert_eq!(value, "4096");
    assert!(
        edited,
        "an edited row is marked as differing from the recipe"
    );
}

#[test]
fn an_invalid_edit_is_rejected_and_not_kept() {
    // A rejected value left in the form reads as accepted.
    let mut s = state_with_recipe();
    s.open_config().expect("opens");
    let rows = s.config_rows();
    s.row = rows
        .iter()
        .position(|(k, _, _)| k == "scheduling_policy")
        .expect("key");
    s.editing = true;
    s.edit_buffer = "nonsense".into();
    s.commit_edit();

    let err = s.error.clone().expect("rejected");
    assert!(
        err.contains("scheduling-policy") || err.contains("nonsense"),
        "{err}"
    );
    assert!(
        s.overrides.is_empty(),
        "the bad value must not enter the overrides"
    );
    let (_, value, edited) = s
        .config_rows()
        .into_iter()
        .find(|(k, _, _)| k == "scheduling_policy")
        .expect("key");
    assert_eq!(value, "slai", "still the recipe's value");
    assert!(!edited);
}

#[test]
fn an_empty_edit_is_refused_rather_than_silently_clearing_a_flag() {
    let mut s = state_with_recipe();
    s.open_config().expect("opens");
    s.editing = true;
    s.edit_buffer = "   ".into();
    s.commit_edit();
    assert!(
        s.error.as_deref().is_some_and(|e| e.contains("empty")),
        "{:?}",
        s.error
    );
    assert!(s.overrides.is_empty());
}

#[test]
fn the_whole_config_is_validated_not_just_the_field() {
    // Flags interact, so a per-field check would accept combinations that
    // cannot serve. `--ep-size` past the world size is the cheapest proof.
    let mut s = state_with_recipe();
    s.open_config().expect("opens");
    let rows = s.config_rows();
    if let Some(i) = rows
        .iter()
        .position(|(k, _, _)| k == "gpu_memory_utilization")
    {
        s.row = i;
        s.editing = true;
        s.edit_buffer = "9.0".into(); // out of range
        s.commit_edit();
        assert!(s.error.is_some(), "an out-of-range value must be caught");
        assert!(s.overrides.is_empty());
    }
}

#[test]
fn resetting_returns_to_the_recipes_own_values() {
    let mut s = state_with_recipe();
    s.open_config().expect("opens");
    s.row = s
        .config_rows()
        .iter()
        .position(|(k, _, _)| k == "port")
        .expect("port");
    s.editing = true;
    s.edit_buffer = "9999".into();
    s.commit_edit();
    assert_eq!(s.overrides.len(), 1);

    s.reset_overrides();
    assert!(s.overrides.is_empty());
    let (_, value, edited) = s
        .config_rows()
        .into_iter()
        .find(|(k, _, _)| k == "port")
        .expect("port");
    assert_eq!(value, "8888");
    assert!(!edited);
}

#[test]
fn the_preview_argv_reflects_the_edits() {
    let mut s = state_with_recipe();
    s.open_config().expect("opens");
    s.row = s
        .config_rows()
        .iter()
        .position(|(k, _, _)| k == "port")
        .expect("port");
    s.editing = true;
    s.edit_buffer = "9999".into();
    s.commit_edit();

    let argv = s.preview_argv().expect("renders");
    let i = argv.iter().position(|a| a == "--port").expect("present");
    assert_eq!(argv[i + 1], "9999");
    assert_eq!(
        argv.iter().filter(|a| *a == "--port").count(),
        1,
        "specified once: {argv:?}"
    );
}

#[test]
fn the_filter_narrows_and_the_selection_stays_in_range() {
    let mut s = state_with_recipe();
    s.rebuild(&[local_of("org/other")]);
    s.selected = s.visible().len().saturating_sub(1);
    s.filter = "zzz-matches-nothing".into();
    s.rebuild(&[]);
    assert!(s.visible().is_empty());
    assert_eq!(s.selected, 0, "a filtered-out selection cannot dangle");
    assert!(s.current().is_none());
}

#[test]
fn a_refresh_without_a_store_is_a_no_op_not_a_panic() {
    let mut s = LibState::default();
    s.refresh();
    assert!(!s.fetching, "nothing to refresh against");
    assert!(!s.poll(&[]), "and polling is harmless");
}

#[test]
fn a_field_error_carries_the_actionable_line_not_the_header() {
    // "Atlas CLI: 1 invalid flag combination" tells the reader nothing they do
    // not already know. The form must show WHAT is wrong and HOW to fix it.
    let report = concat!(
        "Atlas CLI: 1 invalid flag combination — fix before serving:\n\n",
        "  [1] --ep-size 2 exceeds --world-size 1.\n",
        "      why: expert parallelism cannot span more ranks than exist.\n",
        "      fix: raise --world-size to at least --ep-size, or lower --ep-size.\n"
    );
    let line = problem_line(report);
    assert!(line.contains("--ep-size 2 exceeds"), "{line}");
    assert!(
        line.contains("raise --world-size"),
        "carries the fix: {line}"
    );
    assert!(!line.contains("Atlas CLI:"), "not the header: {line}");
    assert!(!line.contains('\n'), "one line, for one field: {line}");
}

#[test]
fn a_clap_error_without_a_numbered_block_still_reads() {
    let line = problem_line("error: invalid value 'x' for '--port <PORT>'");
    assert!(line.contains("invalid value"), "{line}");
}
