// SPDX-License-Identifier: AGPL-3.0-only

//! The list itself: what order it is in, what the filter hides, and what
//! happens to the cursor when a refresh changes the list under it.
//!
//! A background fetch lands on whatever screen the user is on and re-joins the
//! whole catalogue, so the list can get longer, shorter or re-sorted between
//! one frame and the next. Every off-by-one that produces — a selection past
//! the end, a cursor that slides onto a different model, a card index into a
//! shorter recipe list — is a wrong model launched from a screen that looked
//! correct.

use super::*;
use crate::recipe::Recipe;
use crate::tui::data::library::LibraryEntry;

fn recipe_for(id: &str, model: &str) -> Recipe {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml");
    let text = std::fs::read_to_string(&path).expect("fixture");
    let mut r = Recipe::parse(id, &text).expect("parses");
    r.model = model.to_string();
    r
}

fn entry(model: &str, has_weights: bool) -> LibraryEntry {
    LibraryEntry {
        id: model.into(),
        snapshot_dir: Default::default(),
        size_bytes: 1024,
        has_weights,
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

fn state(recipes: Vec<Recipe>, local: &[LibraryEntry]) -> LibState {
    let mut s = LibState {
        index: Index {
            recipes,
            ..Index::default()
        },
        ..LibState::default()
    };
    s.rebuild(local);
    s
}

fn models(s: &LibState) -> Vec<String> {
    s.visible().iter().map(|e| e.model.clone()).collect()
}

#[test]
fn what_can_be_run_right_now_sorts_to_the_top() {
    // Three genuinely different states, in the order a reader needs them:
    // runnable, downloadable, and servable-by-hand.
    let s = state(
        vec![
            recipe_for("fam/ready", "org/ready"),
            recipe_for("fam/needs-download", "org/needs-download"),
        ],
        &[entry("org/ready", true), entry("org/local-only", true)],
    );
    assert_eq!(
        models(&s),
        ["org/ready", "org/needs-download", "org/local-only"]
    );
}

#[test]
fn rows_of_equal_rank_are_ordered_by_id_so_the_list_does_not_shuffle() {
    let s = state(
        vec![],
        &[
            entry("zzz/last", true),
            entry("aaa/first", true),
            entry("mmm/middle", true),
        ],
    );
    assert_eq!(models(&s), ["aaa/first", "mmm/middle", "zzz/last"]);
}

#[test]
fn a_half_finished_download_ranks_with_the_downloadable_not_the_runnable() {
    // `has_weights` is the resolver's predicate, so a partial checkpoint must
    // not sort as ready — the row it displaces is one the user could run.
    let s = state(
        vec![
            recipe_for("fam/partial", "org/partial"),
            recipe_for("fam/whole", "org/whole"),
        ],
        &[entry("org/partial", false), entry("org/whole", true)],
    );
    assert_eq!(models(&s), ["org/whole", "org/partial"]);
    assert!(
        !s.rows
            .iter()
            .any(|r| r.model == "org/partial" && r.runnable_now())
    );
}

#[test]
fn the_filter_matches_the_model_the_recipe_id_and_the_architecture() {
    // All three are things people type; a filter that only matched the id sent
    // readers looking for a recipe stem to an empty list.
    let mut s = state(
        vec![recipe_for("qwen3.6/flagship-mtp", "org/qwen")],
        &[entry("org/qwen", true), entry("org/other", true)],
    );
    for needle in ["qwen", "flagship-mtp", "qwen3_6_moe"] {
        s.filter = needle.into();
        assert!(
            models(&s).contains(&"org/qwen".to_string()),
            "{needle:?} must find the row: {:?}",
            models(&s)
        );
    }
    s.filter = "FLAGSHIP".into();
    assert!(!s.visible().is_empty(), "matching is case-insensitive");
    s.filter = "no-such-thing".into();
    assert!(s.visible().is_empty());
    s.filter.clear();
    assert_eq!(s.visible().len(), 2, "an empty filter hides nothing");
}

#[test]
fn a_filter_that_hides_the_selected_row_leaves_no_dangling_selection() {
    let mut s = state(
        vec![],
        &[
            entry("org/a", true),
            entry("org/b", true),
            entry("org/c", true),
        ],
    );
    s.selected = 2;
    s.filter = "org/a".into();
    // The index is into the FILTERED list, so it dangles until something
    // re-clamps it — `current` has to answer "nothing" rather than panic.
    assert!(s.current().is_none());

    s.rebuild(&[
        entry("org/a", true),
        entry("org/b", true),
        entry("org/c", true),
    ]);
    assert_eq!(s.selected, 0);
    assert_eq!(s.current().expect("a row").model, "org/a");
}

#[test]
fn a_refresh_that_shortens_the_list_cannot_leave_the_selection_past_the_end() {
    let mut s = state(
        vec![],
        &[
            entry("org/a", true),
            entry("org/b", true),
            entry("org/c", true),
            entry("org/d", true),
        ],
    );
    s.selected = 3;
    assert_eq!(s.current().expect("a row").model, "org/d");

    s.rebuild(&[entry("org/a", true)]);
    assert!(
        s.selected < s.visible().len(),
        "selected {} of {}",
        s.selected,
        s.visible().len()
    );
    assert!(s.current().is_some(), "the cursor still points at a row");
}

#[test]
fn a_refresh_that_empties_the_list_leaves_the_cursor_at_zero_and_nothing_selected() {
    let mut s = state(vec![], &[entry("org/a", true), entry("org/b", true)]);
    s.selected = 1;
    s.rebuild(&[]);
    assert_eq!(s.selected, 0, "not `len() - 1` on an empty list");
    assert!(s.current().is_none());
    assert!(s.cards().is_empty());
}

#[test]
fn a_refresh_that_lengthens_the_list_keeps_the_cursor_on_the_same_model() {
    // The newcomers sort AHEAD of the selected row, so an index kept across the
    // rebuild would slide onto a different model.
    let mut s = state(vec![], &[entry("org/z", true)]);
    assert_eq!(s.current().expect("a row").model, "org/z");
    s.rebuild(&[
        entry("org/a", true),
        entry("org/b", true),
        entry("org/z", true),
    ]);
    assert_eq!(s.selected, 2, "moved with its model");
    assert_eq!(s.current().expect("a row").model, "org/z");
}

#[test]
fn a_row_becoming_runnable_moves_the_cursor_with_it() {
    // A finished download re-ranks a row from "needs download" to "runnable",
    // which moves it up past every other downloadable row.
    let mut s = state(
        vec![
            recipe_for("fam/aaa", "org/aaa"),
            recipe_for("fam/zzz", "org/zzz"),
        ],
        &[],
    );
    s.selected = 1;
    assert_eq!(s.current().expect("a row").model, "org/zzz");

    s.rebuild(&[entry("org/zzz", true)]);
    assert_eq!(s.selected, 0, "runnable rows sort first");
    assert_eq!(s.current().expect("a row").model, "org/zzz");
}

#[test]
fn the_filter_is_applied_before_the_selection_is_anchored() {
    // `rebuild` anchors on the visible list, so a refresh while a filter is open
    // must not re-point the cursor at a row the filter hides.
    let mut s = state(vec![], &[entry("org/aaa", true), entry("org/zzz", true)]);
    s.filter = "zzz".into();
    s.selected = 0;
    assert_eq!(s.current().expect("a row").model, "org/zzz");

    s.rebuild(&[
        entry("org/aaa", true),
        entry("org/mmm", true),
        entry("org/zzz", true),
    ]);
    assert_eq!(models(&s), ["org/zzz"], "still filtered");
    assert_eq!(s.current().expect("a row").model, "org/zzz");
}

#[test]
fn moving_the_selection_clamps_at_both_ends_and_is_a_no_op_when_empty() {
    let mut s = state(vec![], &[entry("org/a", true), entry("org/b", true)]);
    s.move_selection(-1);
    assert_eq!(s.selected, 0);
    s.move_selection(50);
    assert_eq!(s.selected, 1, "clamped to the last row, not wrapped");
    s.move_selection(-50);
    assert_eq!(s.selected, 0);

    let mut empty = LibState::default();
    empty.move_selection(1);
    assert_eq!(empty.selected, 0, "nothing to move to");
}

#[test]
fn a_refresh_that_shortens_the_form_cannot_leave_the_row_cursor_dangling() {
    // `row` indexes the config form, which a recipe with fewer keys shortens.
    let mut s = state(
        vec![recipe_for("fam/one", "org/m")],
        &[entry("org/m", true)],
    );
    s.open_cards().expect("cards open");
    s.open_config().expect("form opens");
    s.row = s.config_rows().len() - 1;

    let mut short = recipe_for("fam/one", "org/m");
    short.defaults.retain(|k, _| k.as_str() == "port");
    s.index = Index {
        recipes: vec![short],
        ..Index::default()
    };
    s.rebuild(&[entry("org/m", true)]);
    assert!(
        s.row < s.config_rows().len(),
        "row {} of {}",
        s.row,
        s.config_rows().len()
    );
}

#[test]
fn an_index_refresh_that_changes_nothing_leaves_the_cursor_exactly_where_it_was() {
    // The common case: the fetch returns what the cache already had. A rebuild
    // that moves anything here is a visible jump for no reason.
    let recipes = vec![
        recipe_for("fam/a", "org/a"),
        recipe_for("fam/b", "org/b"),
        recipe_for("fam/c", "org/c"),
    ];
    let local = [
        entry("org/a", true),
        entry("org/b", true),
        entry("org/c", true),
    ];
    let mut s = state(recipes.clone(), &local);
    s.selected = 1;
    s.open_cards().expect("cards open");
    s.open_config().expect("form opens");
    s.row = 2;

    s.index = Index {
        recipes,
        ..Index::default()
    };
    s.rebuild(&local);
    assert_eq!(s.selected, 1);
    assert_eq!(s.row, 2);
    assert_eq!(s.view, View::Config, "and the pane does not step back");
    assert_eq!(s.current().expect("a row").model, "org/b");
}

#[test]
fn polling_with_no_fetch_in_flight_changes_nothing() {
    let mut s = state(vec![], &[entry("org/a", true)]);
    assert!(!s.poll(&[]), "no fetch, no change");
    assert_eq!(models(&s), ["org/a"], "and the rows are left alone");
}

#[test]
fn a_landed_fetch_replaces_the_index_and_re_joins_the_list() {
    let mut s = state(vec![], &[entry("org/local", true)]);
    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(Index {
        recipes: vec![recipe_for("fam/new", "org/local")],
        ..Index::default()
    })
    .expect("send");
    s.pending = Some(rx);
    s.fetching = true;

    assert!(s.poll(&[entry("org/local", true)]), "the list changed");
    assert!(!s.fetching, "the spinner stops");
    assert!(
        s.current().expect("a row").runnable_now(),
        "the recipe joined onto the local checkpoint"
    );
}

#[test]
fn a_dead_fetch_thread_stops_the_spinner_and_keeps_the_cached_list() {
    let mut s = state(vec![], &[entry("org/a", true)]);
    let (tx, rx) = std::sync::mpsc::channel::<Index>();
    drop(tx);
    s.pending = Some(rx);
    s.fetching = true;

    assert!(!s.poll(&[]), "nothing arrived");
    assert!(!s.fetching, "but the spinner must not spin forever");
    assert_eq!(models(&s), ["org/a"], "the cache is still on screen");
}

#[test]
fn cancelling_a_refresh_that_was_never_started_is_harmless() {
    LibState::default().cancel_refresh();
}
