// SPDX-License-Identifier: AGPL-3.0-only

//! The one-time 2026-08-16 amnesty must excuse EXACTLY the pinned bytes,
//! fail closed on everything else, and demand its own removal.
//!
//! `the_table_is_exactly_the_2026_08_16_grant` is deliberately RED while the
//! table holds `"PENDING"` OIDs: the pin phase (compute the landed blob OIDs
//! with `git hash-object` once content is final) is what turns it green, so
//! the grant cannot ship half-armed by accident.

use super::amnesty::{AMNESTY_EPOCH, AmnestyEntry, ONE_TIME_AMNESTY, excused, excused_by};
use super::check::invalidating_paths;
use super::coverage_tests::{any_gate, scratch_repo};
use super::tests::tempdir;
use super::{REQUIRED_GATES, read_record, records_newest_first};

const TAXONOMY: &str = ".github/pr-taxonomy.json";

fn blob_oid(root: &std::path::Path, head: &str, path: &str) -> String {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", &format!("{head}:{path}")])
        .output()
        .expect("git runs");
    assert!(out.status.success(), "rev-parse {head}:{path}: {out:?}");
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// A test-only entry pinning a real blob. The leak is fine: a handful of
/// 40-byte strings for the life of the test binary.
fn entry(path: &'static str, oid: String) -> AmnestyEntry {
    AmnestyEntry {
        path,
        head_blob_oid: Box::leak(oid.into_boxed_str()),
        grant: "test grant",
    }
}

/// The grant's core behaviour: the pinned content is excused AT the commit
/// that carries it, and a later edit to the same path re-invalidates because
/// the blob OID moved.
#[test]
fn pinned_content_is_excused_and_a_later_edit_reinvalidates() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    scratch_repo::commit(root, TAXONOMY, r#"{ "a": {}, "b": {} }"#, "the landing");
    let landed = scratch_repo::head(root);
    let table = [entry(TAXONOMY, blob_oid(root, &landed, TAXONOMY))];

    assert!(
        excused_by(root, &landed, TAXONOMY, &table),
        "the exact landed bytes must be excused at the landing commit"
    );

    scratch_repo::commit(
        root,
        TAXONOMY,
        r#"{ "a": {}, "b": {}, "c": {} }"#,
        "a later edit",
    );
    let later = scratch_repo::head(root);
    assert!(
        !excused_by(root, &later, TAXONOMY, &table),
        "an edit after the grant changes the blob OID — the amnesty must not \
         stretch to cover bytes nobody reviewed"
    );
}

/// A path the table does not list is never excused, whatever its content.
#[test]
fn an_unlisted_path_is_never_excused() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    scratch_repo::commit(root, TAXONOMY, "{}", "taxonomy");
    scratch_repo::commit(root, "crates/engine.rs", "// code", "engine");
    let head = scratch_repo::head(root);
    // The table pins engine.rs's own real OID under the TAXONOMY path name,
    // so even a colliding-content probe cannot sneak an unlisted path in.
    let table = [entry(TAXONOMY, blob_oid(root, &head, TAXONOMY))];
    assert!(
        !excused_by(root, &head, "crates/engine.rs", &table),
        "only listed paths participate; content is checked second, not instead"
    );
}

/// Every way git can fail must read as "not excused" — never as forgiveness.
#[test]
fn git_failure_fails_closed() {
    // Not a git repository at all.
    let bare = tempdir::Dir::new();
    let table = [entry(TAXONOMY, "0".repeat(40))];
    assert!(
        !excused_by(bare.path(), "HEAD", TAXONOMY, &table),
        "no repo means no answer, and no answer must not excuse"
    );

    // A real repo, but the commit is unknown and the path absent at head.
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    let head = scratch_repo::head(root);
    assert!(
        !excused_by(root, "ffffffffff", TAXONOMY, &table),
        "an unknown commit must fail closed"
    );
    assert!(
        !excused_by(root, &head, TAXONOMY, &table),
        "a path absent at head has no blob to match — fail closed"
    );
}

/// ★ The wiring, not just the table: `check.rs::invalidating_paths` really
/// consults the grant. Content matching no pin must survive the filter and
/// invalidate; the REAL repo's taxonomy bytes — the only content the live
/// table can possibly pin — must be dropped from the list exactly when
/// [`excused`] says they are the grant. While the pin is live this exercises
/// the excuse arm; after any later taxonomy edit both sides of the
/// equivalence flip together and it keeps proving the keep arm.
#[test]
fn invalidating_paths_drops_exactly_what_the_grant_excuses() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    scratch_repo::commit(root, TAXONOMY, "{}", "baseline");
    let record_sha = scratch_repo::head(root);

    scratch_repo::commit(root, TAXONOMY, r#"{ "not": "the grant" }"#, "hostile edit");
    let hostile = scratch_repo::head(root);
    let kept = invalidating_paths(root, &hostile, &record_sha, &any_gate())
        .expect("the diff runs in a scratch repo");
    assert!(
        kept.iter().any(|p| p == TAXONOMY),
        "content matching no pin must keep invalidating, got {kept:?}"
    );

    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root is two levels above the crate");
    let real = std::fs::read_to_string(repo.join(TAXONOMY)).expect("the real taxonomy reads");
    scratch_repo::commit(root, TAXONOMY, &real, "the amnestied landing");
    let head = scratch_repo::head(root);
    let after = invalidating_paths(root, &head, &record_sha, &any_gate())
        .expect("the diff runs in a scratch repo");
    assert_eq!(
        !after.iter().any(|p| p == TAXONOMY),
        excused(root, &head, TAXONOMY),
        "invalidating_paths must drop the surviving path exactly when the \
         grant excuses it; the filter and the table may never disagree"
    );
}

/// ★ RETIRED 2026-08-17 — the grant is spent and the table is empty.
///
/// This test used to pin the exact two paths of the 2026-08-16 grant so that
/// adding a third was a visible, authorization-requiring act. The grant has
/// since been fully re-earned: every required gate carries a record newer than
/// [`AMNESTY_EPOCH`], cut at sha 4012c9b7e1 (all ten PASS, including
/// bfcl-subset 84.22/84.12 and bfcl-subset-echolp 86.25/86.61), so
/// `amnesty_expires_once_every_gate_has_a_fresh_record` required the table be
/// emptied.
///
/// The assertion is now strictly STRONGER than the one it replaces: the table
/// must be EMPTY. Any future entry — including a re-add of either original
/// path — is a new grant and needs its own authorization, its own pinned blob
/// OID, and its own expiry story.
#[test]
#[allow(clippy::const_is_empty)]
fn the_table_is_empty_the_grant_is_spent() {
    let paths: Vec<&str> = ONE_TIME_AMNESTY.iter().map(|e| e.path).collect();
    assert!(
        paths.is_empty(),
        "the one-time amnesty is spent and must stay empty; found {paths:?}. \
         Adding a path is a NEW grant: it needs explicit authorization, a \
         pinned 40-hex head_blob_oid, and a reason it will expire."
    );
}

/// ★ The grant must not outlive its purpose. Once every required gate's
/// newest committed record postdates [`AMNESTY_EPOCH`], every record was
/// earned against the amnestied content and the table protects nothing —
/// this fails until someone empties it.
#[test]
#[allow(clippy::const_is_empty)]
fn amnesty_expires_once_every_gate_has_a_fresh_record() {
    if ONE_TIME_AMNESTY.is_empty() {
        return; // The grant has been removed; nothing left to expire.
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root is two levels above the crate");
    let stale: Vec<&str> = REQUIRED_GATES
        .iter()
        .copied()
        .filter(|id| {
            let newest = records_newest_first(root, id)
                .first()
                .and_then(|p| read_record(p).ok())
                .map(|r| r.recorded_at)
                .unwrap_or(0);
            newest <= AMNESTY_EPOCH
        })
        .collect();
    assert!(
        !stale.is_empty(),
        "every required gate now has a record newer than AMNESTY_EPOCH \
         (end of 2026-08-16 UTC): the one-time grant has been fully re-earned \
         and protects nothing. EMPTY THE TABLE in \
         crates/atlas-plugin/src/gate/amnesty.rs — the amnesty must not \
         outlive the records it existed to protect."
    );
}
