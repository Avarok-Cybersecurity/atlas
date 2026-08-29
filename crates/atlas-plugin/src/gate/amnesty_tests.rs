// SPDX-License-Identifier: AGPL-3.0-only

//! The one-time PR #648 amnesty must excuse exactly the pinned bytes,
//! fail closed on everything else, and demand its own removal.
//!
//! `the_table_is_exactly_the_pr_648_grant` is deliberately red while the
//! table holds `"PENDING"` OIDs: the pin phase (compute the landed blob OIDs
//! with `git hash-object` once content is final) is what turns it green, so
//! the grant cannot ship half-armed by accident.

use super::amnesty::{AMNESTY_EPOCH, AmnestyEntry, ONE_TIME_AMNESTY, excused_by};
use super::check::invalidating_paths_with_amnesty;
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
    scratch_repo::commit(root, "crates/engine.rs", "{}", "engine");
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

/// ★ The wiring, not just the table: the invalidating-path filter really
/// consults the grant. An explicit test table keeps both arms executable after
/// the one-time production table has been emptied.
#[test]
fn invalidating_paths_drops_exactly_what_the_grant_excuses() {
    let dir = tempdir::Dir::new();
    let root = dir.path();
    scratch_repo::init(root);
    scratch_repo::commit(root, GRANTED_COVERAGE, "// baseline", "baseline");
    let record_sha = scratch_repo::head(root);

    scratch_repo::commit(root, GRANTED_COVERAGE, "// granted bytes", "the grant");
    let granted = scratch_repo::head(root);
    let table = [entry(
        GRANTED_COVERAGE,
        blob_oid(root, &granted, GRANTED_COVERAGE),
    )];
    let dropped = invalidating_paths_with_amnesty(root, &granted, &record_sha, &any_gate(), &table)
        .expect("the granted diff runs");
    assert_eq!(dropped, Vec::<String>::new());

    scratch_repo::commit(root, GRANTED_COVERAGE, "// later edit", "later edit");
    let later = scratch_repo::head(root);
    let kept = invalidating_paths_with_amnesty(root, &later, &record_sha, &any_gate(), &table)
        .expect("the later diff runs");
    assert_eq!(
        kept,
        vec![GRANTED_COVERAGE.to_string()],
        "editing the granted path must restore invalidation"
    );
}

/// The PR #816 grant is exactly the final coverage-policy blob.
#[test]
fn the_table_is_exactly_the_pr_816_grant() {
    let paths: Vec<&str> = ONE_TIME_AMNESTY.iter().map(|e| e.path).collect();
    // The grant has exactly two legal shapes: PR #816's single policy
    // file, or EMPTY once `amnesty_expires_once_every_gate_has_a_fresh_record`
    // has demanded its removal. Anything else is the grant growing, which is
    // what this test exists to prevent. Removal is the designed end of a
    // one-time grant, so it must not read as a violation of it.
    if !paths.is_empty() {
        assert_eq!(
            paths,
            vec!["crates/atlas-plugin/src/gate/coverage.rs"],
            "the grant must not grow beyond PR #816's coverage-policy blob"
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
            entry.grant.contains("PR #816"),
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
fn amnesty_expires_once_every_gate_has_a_fresh_record() {
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
    if ONE_TIME_AMNESTY.is_empty() {
        assert!(
            stale.is_empty(),
            "the PR #648 grant was removed before every required gate had a fresh record: {stale:?}"
        );
    } else {
        assert!(
            !stale.is_empty(),
            "every required gate now has a record newer than AMNESTY_EPOCH \
             (end of 2026-08-27 UTC): empty the fully re-earned one-time grant"
        );
    }
}
