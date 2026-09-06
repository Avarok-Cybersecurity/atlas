// SPDX-License-Identifier: AGPL-3.0-only
//! `committed_signers` must answer "is this key in the TREE", not "is it on
//! disk" — the distinction the 2026-09-05 three-box campaign turned on.

use super::signing::committed_signers;
use super::tests::tempdir;
use std::process::Command;

fn git(dir: &std::path::Path, args: &[&str]) {
    let ok = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .expect("git runs")
        .status
        .success();
    assert!(ok, "git {args:?} failed in {}", dir.display());
}

fn repo() -> tempdir::Dir {
    let d = tempdir::Dir::new();
    let p = d.path();
    git(p, &["init", "-q"]);
    git(p, &["config", "user.email", "t@example.invalid"]);
    git(p, &["config", "user.name", "t"]);
    std::fs::create_dir_all(p.join(".github/record-signers")).expect("mkdir");
    d
}

fn put(dir: &std::path::Path, fp: &str) {
    std::fs::write(
        dir.join(".github/record-signers").join(format!("{fp}.pub")),
        "# header\nAAAA\n",
    )
    .expect("write key");
}

/// The whole point: a key `register` dropped on disk but nobody committed is
/// NOT a reviewed signer. If this ever passes, the pre-run warning goes silent
/// on exactly the box it exists to catch.
#[test]
fn an_auto_registered_but_uncommitted_key_does_not_count() {
    let d = repo();
    put(d.path(), "aaaaaaaaaaaaaaaa");
    let got = committed_signers(d.path()).expect("reads");
    assert!(
        got.is_empty(),
        "an untracked .pub must not read as a committed signer, got {got:?}"
    );
}

#[test]
fn a_committed_key_counts() {
    let d = repo();
    put(d.path(), "bbbbbbbbbbbbbbbb");
    git(d.path(), &["add", ".github/record-signers"]);
    git(d.path(), &["commit", "-qm", "register"]);
    assert_eq!(
        committed_signers(d.path()).expect("reads"),
        vec!["bbbbbbbbbbbbbbbb".to_string()]
    );
}

/// Committed and uncommitted side by side: only the committed one is returned.
/// A test with a single key cannot tell "filters correctly" from "returns
/// everything it finds".
#[test]
fn only_the_committed_one_of_two_is_returned() {
    let d = repo();
    put(d.path(), "cccccccccccccccc");
    git(d.path(), &["add", ".github/record-signers"]);
    git(d.path(), &["commit", "-qm", "register"]);
    put(d.path(), "dddddddddddddddd");
    let got = committed_signers(d.path()).expect("reads");
    assert_eq!(got, vec!["cccccccccccccccc".to_string()], "got {got:?}");
}

/// Not a directory git can answer for -> Err, never a silent empty list. An
/// empty list means "no signers are committed", which the caller renders as a
/// warning; "I could not look" must not wear that costume.
#[test]
fn an_unreadable_tree_is_an_error_not_an_empty_list() {
    let d = tempdir::Dir::new();
    assert!(
        committed_signers(d.path()).is_err(),
        "a non-repo must not report zero committed signers"
    );
}

// ── The message itself ───────────────────────────────────────────────────

use super::signing::signer_notice;

#[test]
fn a_committed_signer_gets_no_notice() {
    assert!(
        signer_notice(&["aaaa".to_string(), "bbbb".to_string()], "bbbb").is_none(),
        "the ordinary case must be silent, or operators learn to ignore it"
    );
}

/// The notice has one job: make the operator stop before the GPU-hours. It
/// must name the fingerprint AND the consequence — a notice that says only
/// "unknown key" reads as bookkeeping and gets skipped, which is precisely
/// what happened on 2026-09-05.
#[test]
fn an_uncommitted_signer_is_named_along_with_the_consequence() {
    let msg = signer_notice(&["aaaa".to_string()], "zzzz").expect("must warn");
    assert!(msg.contains("zzzz"), "must name the fingerprint: {msg}");
    assert!(
        msg.contains(".github/record-signers"),
        "must name where to commit it: {msg}"
    );
    assert!(
        msg.contains("same") || msg.contains("SAME"),
        "must say every record needs the same signer: {msg}"
    );
    assert!(
        msg.contains("split across boxes") || msg.contains("spanning signers"),
        "must name the campaign-splitting consequence: {msg}"
    );
}

#[test]
fn an_empty_registry_still_warns_rather_than_waving_through() {
    assert!(
        signer_notice(&[], "zzzz").is_some(),
        "no committed signers must not read as permission"
    );
}
