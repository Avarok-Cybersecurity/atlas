// SPDX-License-Identifier: AGPL-3.0-only

//! The one-time PR #648 amnesty must excuse exactly the pinned bytes,
//! fail closed on everything else, and demand its own removal.
//!
//! `the_table_is_exactly_the_pr_648_grant` is deliberately red while the
//! table holds `"PENDING"` OIDs: the pin phase (compute the landed blob OIDs
//! with `git hash-object` once content is final) is what turns it green, so
//! the grant cannot ship half-armed by accident.

use super::amnesty::{AMNESTY_EPOCH, AmnestyEntry, ONE_TIME_AMNESTY, excused, excused_by};
use super::check::invalidating_paths;
use super::coverage_tests::{any_gate, scratch_repo};
use super::tests::tempdir;
use super::{REQUIRED_GATES, read_record, records_newest_first};

const TAXONOMY: &str = ".github/pr-taxonomy.json";
const GRANTED_COVERAGE: &str = "crates/atlas-plugin/src/gate/coverage.rs";

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
/// invalidate; the real repo's coverage bytes, the content the live
/// table can possibly pin — must be dropped from the list exactly when
/// [`excused`] says they are the grant. While the pin is live this exercises
/// the excuse arm; after any later coverage edit both sides of the
/// equivalence flip together and it keeps proving the keep arm.
#[test]
fn invalidating_paths_drops_exactly_what_the_grant_excuses() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    scratch_repo::commit(root, GRANTED_COVERAGE, "// baseline", "baseline");
    let record_sha = scratch_repo::head(root);

    scratch_repo::commit(root, GRANTED_COVERAGE, "// not the grant", "hostile edit");
    let hostile = scratch_repo::head(root);
    let kept = invalidating_paths(root, &hostile, &record_sha, &any_gate())
        .expect("the diff runs in a scratch repo");
    assert!(
        kept.iter().any(|p| p == GRANTED_COVERAGE),
        "content matching no pin must keep invalidating, got {kept:?}"
    );

    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root is two levels above the crate");
    let real =
        std::fs::read_to_string(repo.join(GRANTED_COVERAGE)).expect("the real coverage map reads");
    scratch_repo::commit(root, GRANTED_COVERAGE, &real, "the amnestied landing");
    let head = scratch_repo::head(root);
    let after = invalidating_paths(root, &head, &record_sha, &any_gate())
        .expect("the diff runs in a scratch repo");
    assert_eq!(
        !after.iter().any(|p| p == GRANTED_COVERAGE),
        excused(root, &head, GRANTED_COVERAGE),
        "invalidating_paths must drop the surviving path exactly when the \
         grant excuses it; the filter and the table may never disagree"
    );
}

/// The PR #648 grant is exactly the one KV-budget file in that change.
/// Placeholder OIDs keep this test red until the final-content pin commit.
/// (The PR #701 grant — three boundary files — completed this same lifecycle:
/// pinned, re-earned, emptied.)
#[test]
fn the_table_is_exactly_the_pr_648_grant() {
    let paths: Vec<&str> = ONE_TIME_AMNESTY.iter().map(|e| e.path).collect();
    // The grant has exactly two legal shapes: PR #648's single accounting-fix
    // file, or EMPTY once `amnesty_expires_once_every_gate_has_a_fresh_record`
    // has demanded its removal. Anything else is the grant growing, which is
    // what this test exists to prevent. Removal is the designed end of a
    // one-time grant, so it must not read as a violation of it.
    if !paths.is_empty() {
        assert_eq!(
            paths,
            vec!["crates/spark-model/src/factory/build.rs"],
            "the grant must not grow beyond PR #648's KV-budget fix"
        );
    }
    for entry in &ONE_TIME_AMNESTY {
        assert_eq!(
            entry.head_blob_oid.len(),
            40,
            "{} is not pinned",
            entry.path
        );
        assert!(
            entry.head_blob_oid.chars().all(|c| c.is_ascii_hexdigit()),
            "{} has a non-hex blob OID",
            entry.path
        );
        assert!(
            entry.grant.contains("PR #648"),
            "{} lacks its grant",
            entry.path
        );
    }
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
         (end of 2026-08-27 UTC): the one-time grant has been fully re-earned \
         and protects nothing. EMPTY THE TABLE in \
         crates/atlas-plugin/src/gate/amnesty.rs — the amnesty must not \
         outlive the records it existed to protect."
    );
}
