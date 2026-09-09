// SPDX-License-Identifier: AGPL-3.0-only

//! The one session-gate predicate, and the guard that keeps it one.

use super::super::snapshot::{SnapshotEntry, session_gate_blocks};

fn entry(is_tail: bool, session_hash: u64) -> SnapshotEntry {
    sibling_entry(is_tail, false, session_hash)
}

fn sibling_entry(is_tail: bool, is_tail_sibling: bool, session_hash: u64) -> SnapshotEntry {
    SnapshotEntry {
        snapshot_id: 1,
        session_hash,
        token_count: 64,
        prefix_hash: 0xdead_beef,
        last_access: 0,
        tiered: false,
        is_tail,
        is_tail_sibling,
    }
}

/// PRODUCTION BEHAVIOUR IS UNCHANGED. A non-tail entry from another session
/// still matches, which is the whole reason the blanket gate was removed —
/// `session_hash` is unstable across turns of a growing conversation, and
/// gating non-tails rejected every valid prior-turn anchor (the warm-TTFT
/// climb). If this test ever needs "fixing", `--hermetic` has leaked into the
/// default path.
#[test]
fn without_hermetic_a_non_tail_entry_crosses_sessions() {
    assert!(!session_gate_blocks(&entry(false, 11), 22, false));
}

/// And under hermetic it does not. This is the M3 channel: which non-tail
/// anchor a request restores depends on pool history, so a KAT must not
/// restore one another request produced.
#[test]
fn under_hermetic_a_non_tail_entry_is_confined_to_its_session() {
    assert!(session_gate_blocks(&entry(false, 11), 22, true));
}

/// Hermetic must not be a blanket refusal: a request may still restore its
/// OWN state. A gate that blocked everything would look like it worked and
/// would cost every snapshot restore in the run.
#[test]
fn hermetic_still_allows_an_entry_from_the_same_session() {
    assert!(!session_gate_blocks(&entry(false, 42), 42, true));
    assert!(!session_gate_blocks(&entry(true, 42), 42, true));
}

/// A tail is gated in BOTH regimes — that rule predates hermetic and hermetic
/// must not relax it.
#[test]
fn a_tail_from_another_session_is_blocked_either_way() {
    assert!(session_gate_blocks(&entry(true, 11), 22, false));
    assert!(session_gate_blocks(&entry(true, 11), 22, true));
}

/// `session_hash == 0` means "no session", and an unsessioned lookup must not
/// be handed a tail. Under hermetic it must not be handed anything.
#[test]
fn an_unsessioned_lookup_gets_nothing_it_should_not() {
    assert!(session_gate_blocks(&entry(true, 11), 0, false));
    assert!(!session_gate_blocks(&entry(false, 11), 0, false));
    assert!(session_gate_blocks(&entry(false, 11), 0, true));
}

/// `is_tail_sibling` carries the SAME claim in its own doc — "Exact-prefix
/// keyed — NOT session-gated in lookups (safe cross-session, unlike
/// `is_tail`)" — and it is the same claim, so hermetic must gate it too. It
/// is a mid-chunk capture, which makes it if anything MORE path-dependent
/// than an exact one, not less.
#[test]
fn hermetic_also_confines_the_tail_sibling() {
    assert!(
        !session_gate_blocks(&sibling_entry(false, true, 11), 22, false),
        "production must keep serving warm turns from the sibling"
    );
    assert!(
        session_gate_blocks(&sibling_entry(false, true, 11), 22, true),
        "a KAT must not restore a mid-chunk capture from another request"
    );
}

/// ★ THE GUARD. This condition existed as two copies — in `lookup` and in
/// `lookup_tiered` — and only the tier-aware one is on the serving path, so a
/// change to the other would have looked right and changed nothing that runs.
/// No source file may spell the raw condition again.
#[test]
fn the_session_gate_is_not_spelled_out_anywhere_but_the_predicate() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/radix_tree");
    let mut offenders = Vec::new();
    let mut scanned = 0usize;
    let mut walk = vec![dir.clone()];
    while let Some(d) = walk.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk.push(p);
                continue;
            }
            if p.extension().is_none_or(|x| x != "rs") {
                continue;
            }
            let rel = p
                .strip_prefix(&dir)
                .unwrap_or(&p)
                .to_string_lossy()
                .to_string();
            // The predicate itself, and this test, are where it belongs.
            if rel == "snapshot.rs" || rel.ends_with("snapshot_session_gate.rs") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            scanned += 1;
            for (n, line) in text.lines().enumerate() {
                // Match the SHAPE of the raw gate, not a name: a re-spelling
                // would use the same two field reads in the same condition.
                let code = line.split("//").next().unwrap_or("");
                if code.contains("entry.session_hash") && code.contains("session_hash ==") {
                    offenders.push(format!("{rel}:{}", n + 1));
                }
            }
        }
    }
    assert!(
        scanned > 5,
        "the scan visited {scanned} files — it is not scanning the tree"
    );
    assert!(
        offenders.is_empty(),
        "the session gate must be read through `session_gate_blocks`, not \
         re-spelled — otherwise --hermetic reaches one copy and not the other:\n  {}",
        offenders.join("\n  ")
    );
}
