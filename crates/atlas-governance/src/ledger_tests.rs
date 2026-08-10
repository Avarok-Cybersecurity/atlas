// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the journey ledger.
//!
//! No mocking: `lattice_core`'s storage is an injected trait and the engine runs
//! fully in memory, so these exercise the real materialisation path rather than
//! a stand-in for it. The filesystem cases use a real temporary directory.

use super::event::{Event, EventKind, Verdict};
use super::ledger::{self, Journey, append, materialize, path_for, read_all};

fn ev(sha: &str, attempt: u32, kind: EventKind) -> Event {
    Event {
        pr: 389,
        head_sha: sha.to_string(),
        run_id: "run-1".to_string(),
        attempt,
        at: 1_786_200_000,
        kind,
    }
}

fn gate(id: &str, verdict: Verdict) -> EventKind {
    EventKind::Gate {
        id: id.to_string(),
        verdict,
        invalidated_by: Vec::new(),
        detail: None,
    }
}

fn tmpdir() -> std::path::PathBuf {
    let base = std::env::temp_dir().join(format!(
        "atlas-governance-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).expect("temp dir");
    base
}

#[test]
fn appending_then_reading_round_trips() {
    let dir = tmpdir();
    let path = path_for(&dir, 389);
    append(&path, &ev("aaa", 0, gate("bfcl-subset", Verdict::Missing))).unwrap();
    append(
        &path,
        &ev(
            "bbb",
            0,
            EventKind::State {
                to: "gates".to_string(),
            },
        ),
    )
    .unwrap();

    let journey = read_all(&path).unwrap();
    assert_eq!(journey.events.len(), 2);
    assert_eq!(journey.events[0].head_sha, "aaa");
    assert_eq!(journey.events[1].pr, 389);
}

#[test]
fn append_creates_the_directory() {
    let dir = tmpdir();
    let path = path_for(&dir.join("nested").join("deeper"), 7);
    append(&path, &ev("aaa", 0, gate("ttft-warm-gate", Verdict::Pass))).unwrap();
    assert!(path.exists());
}

/// ★ The file is a SET, not a log.
///
/// Replaying a CI job appends the same records again. A reader that counted
/// them twice would report a gate as having been evaluated more often than it
/// was — and the whole point of the ledger is to answer questions like "how
/// many times did this gate re-open?" correctly.
#[test]
fn replayed_events_collapse() {
    let e = ev("aaa", 0, gate("bfcl-subset", Verdict::Pass));
    let journey = Journey {
        events: vec![e.clone(), e.clone(), e],
    }
    .deduplicated();
    assert_eq!(journey.events.len(), 1);
}

/// ★ …but a genuine RE-RUN is not a duplicate.
///
/// `attempt` is part of the identity precisely so a re-run after a flake is
/// kept. Collapsing those would erase the flakiness, which is one of the few
/// things the history is uniquely able to show.
#[test]
fn a_second_attempt_is_a_distinct_event() {
    let journey = Journey {
        events: vec![
            ev("aaa", 0, gate("bfcl-subset", Verdict::Fail)),
            ev("aaa", 1, gate("bfcl-subset", Verdict::Pass)),
        ],
    }
    .deduplicated();
    assert_eq!(journey.events.len(), 2, "a re-run must survive dedup");
}

/// The timestamp is data, not identity — otherwise nothing would ever dedup.
#[test]
fn the_timestamp_does_not_affect_identity() {
    let mut a = ev("aaa", 0, gate("bfcl-subset", Verdict::Pass));
    let mut b = a.clone();
    a.at = 1;
    b.at = 999_999;
    assert_eq!(a.identity(), b.identity());
}

/// Different kinds at the same commit and attempt are distinct.
#[test]
fn different_kinds_do_not_collide() {
    let a = ev("aaa", 0, gate("bfcl-subset", Verdict::Pass));
    let b = ev("aaa", 0, gate("ttft-warm-gate", Verdict::Pass));
    let c = ev(
        "aaa",
        0,
        EventKind::Category {
            value: "numerics".into(),
            status: "ok".into(),
        },
    );
    assert_ne!(a.identity(), b.identity());
    assert_ne!(a.identity(), c.identity());
}

/// ★ Dedup is order-independent — the property that makes concurrent appends
/// safe. Two CI jobs writing in either order must yield the same set, which is
/// what "grow-only set" buys and why `merge=union` is sound.
#[test]
fn dedup_is_order_independent() {
    let a = ev("aaa", 0, gate("bfcl-subset", Verdict::Pass));
    let b = ev("bbb", 0, gate("ttft-cold-gate", Verdict::Fail));
    let one = Journey {
        events: vec![a.clone(), b.clone(), a.clone()],
    }
    .deduplicated();
    let two = Journey {
        events: vec![b.clone(), a.clone(), b],
    }
    .deduplicated();

    let mut ids_one: Vec<String> = one.events.iter().map(Event::identity).collect();
    let mut ids_two: Vec<String> = two.events.iter().map(Event::identity).collect();
    ids_one.sort();
    ids_two.sort();
    assert_eq!(ids_one, ids_two);
}

/// A malformed line is an error, not a silent skip. A reader that quietly
/// dropped what it could not parse would report a partial history as a full one.
#[test]
fn a_corrupt_line_is_refused() {
    let dir = tmpdir();
    let path = path_for(&dir, 1);
    append(&path, &ev("aaa", 0, gate("bfcl-subset", Verdict::Pass))).unwrap();
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(f, "{{not json").unwrap();
    }
    let err = read_all(&path).unwrap_err().to_string();
    assert!(err.contains("line 2"), "{err}");
}

#[test]
fn blank_lines_are_tolerated() {
    let dir = tmpdir();
    let path = path_for(&dir, 2);
    append(&path, &ev("aaa", 0, gate("bfcl-subset", Verdict::Pass))).unwrap();
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(f).unwrap();
    }
    assert_eq!(read_all(&path).unwrap().events.len(), 1);
}

/// `Missing` and `Fail` are different facts and must stay distinguishable:
/// "we have not measured this" is not "we measured it and it regressed".
#[test]
fn missing_and_fail_are_distinct_on_the_wire() {
    let m = serde_json::to_string(&ev("a", 0, gate("g", Verdict::Missing))).unwrap();
    let f = serde_json::to_string(&ev("a", 0, gate("g", Verdict::Fail))).unwrap();
    assert!(m.contains("\"missing\""), "{m}");
    assert!(f.contains("\"fail\""), "{f}");
    assert_ne!(m, f);
}

/// The materialised graph holds one node per commit plus one per event, with
/// an edge from each commit to what was observed at it.
#[test]
fn materialize_builds_commit_and_event_nodes() {
    let journey = Journey {
        events: vec![
            ev("aaa", 0, gate("bfcl-subset", Verdict::Missing)),
            ev("aaa", 0, gate("ttft-warm-gate", Verdict::Pass)),
            ev("bbb", 0, gate("bfcl-subset", Verdict::Pass)),
        ],
    };
    let engine = materialize(&journey).unwrap();
    // 2 commits + 3 events
    assert_eq!(engine.point_ids().unwrap().len(), 5);

    // Commit `aaa` sorts first, so it is node 0 and observed two events.
    let edges = engine.get_edges(0).unwrap();
    assert_eq!(edges.len(), 2, "commit aaa observed two events");
    assert!(edges.iter().all(|e| e.relation == "observed"));
}

/// ★ Commit node ids must not shift as events are appended.
///
/// Ids are assigned commits-first precisely so that materialising again after
/// more events arrive leaves each commit where it was. If they moved, an edge
/// recorded by an earlier materialisation would point at a different node.
#[test]
fn commit_ids_are_stable_as_events_are_appended() {
    let first = Journey {
        events: vec![ev("aaa", 0, gate("bfcl-subset", Verdict::Missing))],
    };
    let engine_a = materialize(&first).unwrap();
    let sha_before = engine_a.get_point(0).unwrap().unwrap();

    let mut later = first.clone();
    later
        .events
        .push(ev("aaa", 1, gate("bfcl-subset", Verdict::Pass)));
    let engine_b = materialize(&later).unwrap();
    let sha_after = engine_b.get_point(0).unwrap().unwrap();

    assert_eq!(
        sha_before.payload.get("sha"),
        sha_after.payload.get("sha"),
        "commit node 0 moved when an event was appended"
    );
}

#[test]
fn an_empty_journey_materializes_to_an_empty_graph() {
    let engine = materialize(&Journey::default()).unwrap();
    assert!(engine.point_ids().unwrap().is_empty());
}

/// The traversal the graph exists for: a gate's history at one glance.
#[test]
fn gate_history_selects_only_that_gate() {
    let journey = Journey {
        events: vec![
            ev("aaa", 0, gate("bfcl-subset", Verdict::Missing)),
            ev("aaa", 0, gate("ttft-warm-gate", Verdict::Pass)),
            ev("bbb", 1, gate("bfcl-subset", Verdict::Pass)),
        ],
    };
    let hits: Vec<&Event> = journey.gate_history("bfcl-subset").collect();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].head_sha, "aaa");
    assert_eq!(hits[1].head_sha, "bbb");
}

/// The advisory category is recorded with its status, so an abstention is never
/// mistaken for a confident answer when the log is audited later.
#[test]
fn an_abstained_category_is_distinguishable_from_an_answer() {
    let ok = ev(
        "aaa",
        0,
        EventKind::Category {
            value: "numerics".into(),
            status: "ok".into(),
        },
    );
    let abstain = ev(
        "aaa",
        0,
        EventKind::Category {
            value: "numerics".into(),
            status: "abstain".into(),
        },
    );
    assert_ne!(ok.identity(), abstain.identity());
}

#[test]
fn the_path_is_one_file_per_pull_request() {
    let root = std::path::Path::new("/repo");
    assert_eq!(
        ledger::path_for(root, 389),
        std::path::Path::new("/repo/governance/pr-389.jsonl")
    );
    assert_ne!(ledger::path_for(root, 389), ledger::path_for(root, 390));
}
