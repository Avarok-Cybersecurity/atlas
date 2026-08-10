// SPDX-License-Identifier: AGPL-3.0-only

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

fn state(updated: &str) -> LibState {
    let mut s = LibState::default();
    s.index = Index {
        recipes: vec![recipe_with("fam/stem", updated)],
        ..Default::default()
    };
    s.rebuild(&[]);
    s
}

#[test]
fn a_recipe_that_states_its_own_date_is_never_looked_up() {
    // `metadata.updated` is what the author wrote; a commit timestamp dates the
    // last edit to the FILE, which is a different fact. Overriding one with the
    // other would also spend a rate-limited request to get a worse answer.
    let mut s = state("2026-08-01");
    s.want_date_for("fam/stem");
    assert!(s.dating.is_none(), "no lookup should have started");
    assert!(!s.dated.contains("fam/stem"), "and nothing was recorded");
}

#[test]
fn an_undated_recipe_is_looked_up_exactly_once() {
    let mut s = state("");
    s.want_date_for("fam/stem");
    assert_eq!(s.dating.as_deref(), Some("fam/stem"), "the lookup started");

    // A second ask while one is in flight must not spawn another thread — this
    // runs on the render tick, so "every frame" is the failure mode.
    s.want_date_for("fam/stem");
    assert_eq!(s.dating.as_deref(), Some("fam/stem"));

    // And once it has been asked about, never again this session — including
    // after a failure, which is what `dated` holding failed ids buys.
    s.dating = None;
    s.pending_date = None;
    s.want_date_for("fam/stem");
    assert!(
        s.dating.is_none(),
        "an already-asked recipe is not re-asked"
    );
}

#[test]
fn the_date_cell_is_empty_a_skeleton_or_the_date() {
    let known = recipe_with("fam/stem", "2026-08-01");
    let unknown = recipe_with("fam/stem", "");

    let s = state("");
    assert_eq!(s.date_text(&known), "2026-08-01", "a known date is shown");
    assert!(
        s.date_text(&unknown).is_empty(),
        "an unknown date draws no row, rather than a blank one"
    );

    let mut s = state("");
    s.dating = Some("fam/stem".into());
    let cell = s.date_text(&unknown);
    assert!(!cell.is_empty(), "a lookup in flight shows a placeholder");
    assert!(
        cell.chars().all(|c| c == '░'),
        "and it is a skeleton, not a fake date: {cell:?}"
    );
}

#[test]
fn a_skeleton_is_only_shown_for_the_recipe_actually_being_looked_up() {
    // One lookup runs at a time; every other row must stay quiet rather than
    // shimmer in sympathy.
    let mut s = state("");
    s.dating = Some("other/recipe".into());
    assert!(s.date_text(&recipe_with("fam/stem", "")).is_empty());
}

#[test]
fn only_the_recipe_on_screen_is_dated() {
    // Which recipe that is depends on the pane, and getting it wrong spends a
    // rate-limited request on something nobody is reading.
    let mut s = state("");
    s.view = View::List;
    assert_eq!(s.visible_recipe_id().as_deref(), Some("fam/stem"));
    s.view = View::Cards;
    assert_eq!(s.visible_recipe_id().as_deref(), Some("fam/stem"));

    // With nothing listed there is nothing to date.
    let empty = LibState::default();
    assert!(empty.visible_recipe_id().is_none());
}

#[test]
fn a_date_that_arrives_reaches_the_rows_the_render_actually_reads() {
    // `rows` holds CLONES of the index's recipes, so writing the date into the
    // index alone leaves the screen unchanged — the bug this re-join prevents.
    let mut s = state("");
    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(("fam/stem".to_string(), Some("2026-07-04".to_string())))
        .unwrap();
    s.pending_date = Some(rx);

    assert!(s.poll_date(), "a landed date is a change");
    assert_eq!(
        s.date_text(&recipe_with("fam/stem", "")),
        "2026-07-04",
        "the render must be able to see it"
    );
    assert!(s.dating.is_none(), "the skeleton is cleared");

    // The load-bearing part: a background index refresh landing afterwards
    // replaces `index` wholesale. A date written into the old index vanished
    // here — invisible to every test, and the reason the row stayed blank on
    // real hardware.
    s.index = Index {
        recipes: vec![recipe_with("fam/stem", "")],
        ..Default::default()
    };
    s.rebuild(&[]);
    assert_eq!(
        s.date_text(&recipe_with("fam/stem", "")),
        "2026-07-04",
        "a fetched date must survive an index refresh"
    );
}

#[test]
fn a_failed_lookup_clears_the_skeleton_and_changes_nothing() {
    let mut s = state("");
    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(("fam/stem".to_string(), None)).unwrap();
    s.pending_date = Some(rx);
    s.dating = Some("fam/stem".into());

    assert!(!s.poll_date(), "nothing changed");
    assert!(s.dating.is_none(), "but the skeleton must not spin forever");
    assert!(s.date_text(&recipe_with("fam/stem", "")).is_empty());
}

#[test]
fn a_dead_lookup_thread_clears_the_skeleton() {
    // Disconnected without a message: the same guarantee the index fetch makes.
    let mut s = state("");
    let (tx, rx) = std::sync::mpsc::channel::<(String, Option<String>)>();
    drop(tx);
    s.pending_date = Some(rx);
    s.dating = Some("fam/stem".into());

    assert!(!s.poll_date());
    assert!(s.dating.is_none());
    assert!(s.pending_date.is_none());
}

#[test]
fn a_date_for_a_recipe_that_vanished_is_dropped_quietly() {
    // The index can be replaced by a refresh while a lookup is in flight. The
    // id is echoed back precisely so a late answer cannot be applied to
    // whatever happens to be selected by then.
    let mut s = state("");
    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(("gone/recipe".to_string(), Some("2026-07-04".into())))
        .unwrap();
    s.pending_date = Some(rx);

    assert!(s.poll_date(), "the answer was stored under its own id");
    assert!(
        s.date_text(&recipe_with("fam/stem", "")).is_empty(),
        "the visible recipe must not inherit another recipe's date"
    );
}

/// Network test: the whole lazy-date flow against the real cached index and
/// the real GitHub API, exactly as the dashboard drives it.
#[test]
#[ignore = "network"]
fn the_real_flow_dates_a_real_recipe() {
    let root = std::path::PathBuf::from("/workspace/.atlas");
    let mut s = LibState::default();
    s.attach(root, &[]);
    assert!(!s.index.recipes.is_empty(), "cached index must be present");

    let id = s
        .index
        .recipes
        .iter()
        .find(|r| r.updated.is_empty())
        .map(|r| r.id.clone())
        .expect("some recipe lacks a date");
    eprintln!("dating {id}");

    s.want_date_for(&id);
    assert_eq!(s.dating.as_deref(), Some(id.as_str()), "lookup started");

    for _ in 0..100 {
        if s.poll_date() {
            let got = s.fetched_dates.get(&id).expect("stored under its id");
            eprintln!("got date {got:?}");
            assert_eq!(got.len(), 10, "YYYY-MM-DD");
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    panic!("no date arrived; dating={:?}", s.dating);
}
