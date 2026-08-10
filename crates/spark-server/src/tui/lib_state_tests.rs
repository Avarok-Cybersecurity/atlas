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
fn a_model_with_no_recipe_never_reaches_the_cards() {
    // Unchanged behaviour, asked for explicitly: a local-only checkpoint is
    // listed and serveable by hand, but there is nothing to show a card for.
    let mut s = LibState::default();
    s.rebuild(&[local_of("org/orphan")]);
    let err = s.open_cards().expect_err("refused");
    assert!(err.contains("no recipe"), "{err}");
    assert_eq!(s.view, View::List, "and it stays on the list");
}

#[test]
fn the_form_shows_every_recipe_key_with_its_value() {
    let mut s = state_with_recipe();
    s.open_cards().expect("opens the cards");
    s.open_config().expect("opens the form");
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
    s.open_cards().expect("opens the cards");
    s.open_config().expect("opens the form");
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
    s.open_cards().expect("opens the cards");
    s.open_config().expect("opens the form");
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
    s.open_cards().expect("opens the cards");
    s.open_config().expect("opens the form");
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
    s.open_cards().expect("opens the cards");
    s.open_config().expect("opens the form");
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
    s.open_cards().expect("opens the cards");
    s.open_config().expect("opens the form");
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
    s.open_cards().expect("opens the cards");
    s.open_config().expect("opens the form");
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

#[test]
fn a_refresh_keeps_the_selection_on_the_same_model_not_the_same_row() {
    // The join sorts runnable rows first. A background fetch that makes an
    // earlier-sorting model runnable shifts every row below it, so an index
    // kept across the rebuild points at a DIFFERENT model — and the cards, the
    // form and what `s` launches all change under the user.
    let mut s = state_with_recipe();
    let mine = s.current().expect("a row").model.clone();
    s.rebuild(&[local_of(&mine), local_of("aaa/sorts-first")]);
    let before = s.current().expect("a row").model.clone();
    assert_eq!(before, mine, "still the model I had selected");

    // And again with the newcomer genuinely ahead of it in the order.
    s.rebuild(&[local_of("aaa/sorts-first"), local_of(&mine)]);
    assert_eq!(
        s.current().expect("a row").model,
        mine,
        "a reshuffle must not move the selection to another model"
    );
}

#[test]
fn a_refresh_that_removes_the_open_model_steps_back_to_the_list() {
    // Staying in Cards would render another model's recipes under the heading
    // of the one that vanished.
    let mut s = state_with_recipe();
    s.open_cards().expect("opens");
    assert_eq!(s.view, View::Cards);
    s.index = Index::default(); // the fetch came back with nothing
    s.rebuild(&[]);
    assert_eq!(
        s.view,
        View::List,
        "cannot stay in a vanished model's cards"
    );
}

#[test]
fn a_shorter_recipe_list_cannot_leave_the_card_index_dangling() {
    let mut s = state_with_recipe();
    s.open_cards().expect("opens");
    s.card = 5; // as if a longer list had been shown
    s.rebuild(&[local_of(&real_recipe().model)]);
    assert!(
        s.card < s.cards().len().max(1),
        "card {} out of {} ",
        s.card,
        s.cards().len()
    );
}

#[test]
fn edits_do_not_survive_onto_a_recipe_the_user_never_opened() {
    // If the recipe being edited disappears in a refresh, its overrides are
    // edits to nothing. Carrying them onto whichever recipe inherits the index
    // would launch a configuration the user never typed.
    let mut s = state_with_recipe();
    s.open_cards().expect("opens");
    s.open_config().expect("opens");
    s.row = s
        .config_rows()
        .iter()
        .position(|(k, _, _)| k == "port")
        .expect("port");
    s.editing = true;
    s.edit_buffer = "9100".into();
    s.commit_edit();
    assert_eq!(s.overrides.len(), 1);

    s.index = Index::default(); // the recipe is gone
    s.rebuild(&[local_of(&real_recipe().model)]);
    assert!(
        s.overrides.is_empty(),
        "edits to a vanished recipe are dropped"
    );
    assert!(!s.editing);
}

#[test]
fn a_multi_node_recipe_can_still_be_opened_and_read() {
    // It cannot be STARTED here — `model_swap::swap` refuses world_size > 1 —
    // and the card says so. But the settings are still worth reading, so the
    // form opens. An earlier version of this refused at `open_config`, which
    // was a restriction invented to fix a bug that did not exist: a harness
    // that rendered argv from `defaults:` alone reproduced the very failure
    // `Recipe::serve_args` already prevents by emitting `--world-size` from
    // `min_nodes`.
    let mut recipe = real_recipe();
    recipe.min_nodes = 2;
    let local = vec![local_of(&recipe.model)];
    let mut s = LibState {
        index: Index {
            recipes: vec![recipe],
            ..Index::default()
        },
        ..LibState::default()
    };
    s.rebuild(&local);
    s.open_cards().expect("cards open");
    s.open_config().expect("and so does the form");

    // And the argv it renders is the valid one, not the one that trips the
    // world-size check.
    let argv = s.preview_argv().expect("renders");
    let i = argv
        .iter()
        .position(|a| a == "--world-size")
        .expect("world-size is emitted from min_nodes");
    assert_eq!(argv[i + 1], "2");
}

#[test]
fn a_launch_that_fails_after_spawning_is_reported_not_swallowed() {
    // `launch` returns when the loader THREAD starts, not when the swap
    // succeeds. The dashboard resets its checklist and sets the pill to
    // LOADING at that moment, so a failure afterwards — no compiled kernels
    // for that model, say — left both showing a load that had given up.
    let mut s = state_with_recipe();
    assert!(s.poll_launch().is_none(), "nothing in flight yet");

    // Stand in for the loader thread: it reports its error on this channel.
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    s.launch_result = Some(rx);
    assert!(
        s.poll_launch().is_none(),
        "still loading — nothing said yet"
    );

    tx.send("this build has no compiled kernels for qwen3_6_moe".into())
        .expect("send");
    let got = s.poll_launch().expect("the failure surfaces");
    assert!(got.contains("no compiled kernels"), "{got}");
    assert!(
        s.poll_launch().is_none(),
        "and it is reported once, not every tick"
    );
}

#[test]
fn a_launch_that_succeeds_reports_nothing() {
    // The thread drops its sender on success; a disconnect with no message is
    // not an error to show the user.
    let mut s = state_with_recipe();
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    s.launch_result = Some(rx);
    drop(tx);
    assert!(s.poll_launch().is_none(), "silence means it worked");
}
