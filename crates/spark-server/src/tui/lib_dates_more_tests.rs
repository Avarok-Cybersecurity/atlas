// SPDX-License-Identifier: AGPL-3.0-only

//! The rate-limit budget, from the other side.
//!
//! `lib_dates_tests.rs` pins what one lookup does. These pin what the dashboard
//! must NOT do: GitHub allows 60 unauthenticated calls an hour and the Library
//! itself spends some of them, so every path that could turn a render tick into
//! a request is a path that can exhaust the budget the list depends on.

use super::*;
use crate::recipe::fetch::Index;

fn recipe_with(id: &str, updated: &str) -> crate::recipe::Recipe {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml");
    let text = std::fs::read_to_string(&path).expect("fixture");
    let mut r = crate::recipe::Recipe::parse(id, &text).expect("parses");
    r.updated = updated.to_string();
    r
}

fn state(recipes: Vec<crate::recipe::Recipe>) -> LibState {
    let mut s = LibState::default();
    s.index = Index {
        recipes,
        ..Default::default()
    };
    s.rebuild(&[]);
    s
}

#[test]
fn a_recipe_the_index_has_never_heard_of_is_not_looked_up() {
    // The id comes from the row on screen, which a refresh can replace between
    // the tick that read it and the tick that acts on it.
    let mut s = state(vec![recipe_with("fam/stem", "")]);
    s.want_date_for("fam/vanished");
    assert!(s.dating.is_none(), "no request for a recipe that is gone");
    assert!(s.pending_date.is_none());
}

#[test]
fn a_lookup_in_flight_blocks_a_lookup_for_a_different_recipe() {
    // One at a time, deliberately: the alternative is one request per row the
    // cursor passes over.
    let mut s = state(vec![
        recipe_with("fam/first", ""),
        recipe_with("fam/second", ""),
    ]);
    // Stand in for the worker thread, so this stays a test of the guard.
    let (_tx, rx) = std::sync::mpsc::channel::<(String, Option<String>)>();
    s.pending_date = Some(rx);
    s.dating = Some("fam/first".into());
    s.dated.insert("fam/first".into());

    s.want_date_for("fam/second");
    assert_eq!(
        s.dating.as_deref(),
        Some("fam/first"),
        "the second must wait rather than replace it"
    );
    assert!(
        !s.dated.contains("fam/second"),
        "and is not marked as asked"
    );
}

#[test]
fn nothing_is_visible_to_date_before_a_row_is_selected() {
    let s = LibState::default();
    assert!(s.visible_recipe_id().is_none());

    // Same in the panes one and two levels in, which read the selected CARD.
    let mut s = LibState::default();
    s.view = View::Cards;
    assert!(s.visible_recipe_id().is_none());
    s.view = View::Config;
    assert!(s.visible_recipe_id().is_none());
}

#[test]
fn the_pane_decides_which_of_a_models_recipes_gets_the_request() {
    // The list describes the row's `primary()`; the card panes describe the
    // card. Dating anything else spends a request on a recipe nobody is
    // reading.
    let mut first = recipe_with("fam/aaa", "");
    let mut second = recipe_with("fam/zzz", "");
    first.runtime = Some("vllm".into());
    second.runtime = Some("atlas".into());
    let mut s = state(vec![first, second]);

    s.view = View::List;
    assert_eq!(
        s.visible_recipe_id().as_deref(),
        Some("fam/zzz"),
        "the list describes the launchable recipe"
    );

    s.view = View::Cards;
    s.card = 0;
    assert_eq!(s.visible_recipe_id().as_deref(), Some("fam/aaa"));
    s.card = 1;
    assert_eq!(s.visible_recipe_id().as_deref(), Some("fam/zzz"));
}

#[test]
fn a_recipes_own_date_wins_over_one_fetched_for_it() {
    // `metadata.updated` is what the author wrote; a commit timestamp dates the
    // last edit to the FILE. If both exist the file's own claim is the answer.
    let mut s = state(vec![recipe_with("fam/stem", "2026-08-01")]);
    s.fetched_dates
        .insert("fam/stem".into(), "1999-01-01".into());
    assert_eq!(
        s.date_text(&recipe_with("fam/stem", "2026-08-01")),
        "2026-08-01"
    );
}

#[test]
fn a_skeleton_never_outranks_a_date_that_is_already_known() {
    // Otherwise a row that has its answer would flicker back to a placeholder
    // the moment a lookup started for it.
    let mut s = state(vec![recipe_with("fam/stem", "")]);
    s.fetched_dates
        .insert("fam/stem".into(), "2026-07-04".into());
    s.dating = Some("fam/stem".into());
    assert_eq!(s.date_text(&recipe_with("fam/stem", "")), "2026-07-04");
}

#[test]
fn an_empty_date_from_the_worker_is_treated_as_no_answer() {
    // The worker echoes the id back whatever happens; an empty string is a
    // failure that would otherwise be stored and rendered as a blank date.
    let mut s = state(vec![recipe_with("fam/stem", "")]);
    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(("fam/stem".to_string(), Some(String::new())))
        .expect("send");
    s.pending_date = Some(rx);
    s.dating = Some("fam/stem".into());

    assert!(!s.poll_date(), "nothing usable arrived");
    assert!(s.fetched_dates.is_empty(), "and nothing was stored");
    assert!(s.dating.is_none(), "the skeleton still clears");
}

#[test]
fn polling_with_no_lookup_in_flight_is_quiet() {
    let mut s = LibState::default();
    assert!(!s.poll_date());
    assert!(s.pending_date.is_none());
}

#[test]
fn a_lookup_that_has_not_answered_yet_leaves_the_skeleton_up() {
    let mut s = state(vec![recipe_with("fam/stem", "")]);
    let (_tx, rx) = std::sync::mpsc::channel::<(String, Option<String>)>();
    s.pending_date = Some(rx);
    s.dating = Some("fam/stem".into());

    assert!(!s.poll_date(), "nothing has landed");
    assert_eq!(s.dating.as_deref(), Some("fam/stem"), "still in flight");
    assert!(s.pending_date.is_some());
}

#[test]
fn a_failed_lookup_is_never_retried_this_session() {
    // An offline box would otherwise spend a thread per frame rediscovering
    // that it is offline — which is why `dated` holds failed ids too.
    let mut s = state(vec![recipe_with("fam/stem", "")]);
    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(("fam/stem".to_string(), None)).expect("send");
    s.pending_date = Some(rx);
    s.dating = Some("fam/stem".into());
    s.dated.insert("fam/stem".into()); // as `want_date_for` records it
    assert!(!s.poll_date(), "the lookup failed");

    s.want_date_for("fam/stem");
    assert!(s.dating.is_none(), "the failure is remembered");
    assert!(s.pending_date.is_none());
}
