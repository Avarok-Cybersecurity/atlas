// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use std::process::Command;

fn git(root: &std::path::Path, args: &[&str]) {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .env("GIT_AUTHOR_NAME", "test")
        .env("GIT_AUTHOR_EMAIL", "test@example.com")
        .env("GIT_COMMITTER_NAME", "test")
        .env("GIT_COMMITTER_EMAIL", "test@example.com")
        .output()
        .expect("git runs");
    assert!(output.status.success(), "git {args:?}: {output:?}");
}

#[test]
fn unreadable_dirty_state_aborts_provenance_capture() {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!("atlas-provenance-{unique}"));
    std::fs::create_dir_all(&root).unwrap();
    git(&root, &["init", "-q"]);
    std::fs::write(root.join("README.md"), "fixture").unwrap();
    git(&root, &["add", "."]);
    git(&root, &["commit", "-q", "-m", "fixture"]);

    // HEAD remains readable, but status cannot parse this index. This reaches
    // the dirty-state failure rather than failing the earlier sha lookup.
    std::fs::write(root.join(".git/index"), "not a git index").unwrap();
    let error = capture_provenance_at(&root).expect_err("unknown dirt is not clean");
    assert_eq!(
        format!("{error:#}"),
        format!(
            "reading the working tree state before the gate run: git status failed in {} — \
             cannot tell whether the measured binary matches the commit being stamped",
            root.display()
        )
    );

    std::fs::remove_dir_all(root).unwrap();
}
