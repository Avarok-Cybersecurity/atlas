// SPDX-License-Identifier: AGPL-3.0-only

//! Which changed paths invalidate a gate record.
//!
//! Split out of `check.rs` to keep that file under the repository's 500-LoC
//! cap. The diff is taken with NO pathspec and filtered in Rust so the
//! filter is unit-testable without a git fixture — see the doc on
//! [`invalidating_paths`].

use std::path::Path;

/// The changed paths that invalidate `gate` between two commits.
///
/// `None` means the question could not be answered — git failed, or one of the
/// two commits is not in this clone. Every such case is treated as "not
/// covered" by the caller, keeping the fail-closed doctrine: a gate check that
/// cannot see the trees must never read as a pass.
///
/// # This deliberately does NOT require ancestry
///
/// It used to. `merge-base --is-ancestor record_sha head` gated the diff, and
/// that was wrong in a way that took main down: **Atlas squash-merges.** A
/// record is written on a PR branch, against a commit on that branch; the
/// squash lands a brand-new commit on main with a different sha and no parent
/// link to the branch. Every record the PR paid GPU hours for stops being an
/// ancestor of anything the instant it merges.
///
/// It did exactly that. `.benchmarks/*/2026-08-09-b0be4ba0e6.json` are five
/// real passing records for #389 — `b0be4ba0e` being the branch's merge of
/// #417 — and after #389 squash-landed as `dd2ac46d5` the gate reported
/// "not an ancestor of this commit" for all five. Main went red, and every PR
/// opened afterwards inherited it and demanded 5 fresh GPU legs to fix a
/// typo.
///
/// Ancestry was never what the check needed. `git diff A B` compares TREES; it
/// is defined for any two commits and needs no history relationship. The
/// question a gate record answers is "was the perf-relevant code the same when
/// this was measured?", and the diff answers exactly that. Ancestry only added
/// an assumption about the shape of history — one this repo's merge strategy
/// violates by design.
///
/// The obvious worry — "then a record from an unrelated branch could cover
/// main" — is answered by the diff itself. An unrelated branch differs on the
/// perf paths and is rejected. If it does NOT differ on them, it measured the
/// same code, and the record is valid; that is the whole content-not-ancestry
/// doctrine, and it is why an identical squash lands covered.
///
/// The one thing ancestry incidentally caught was a missing commit (a shallow
/// clone). `git diff` fails outright there, so that case still returns `None`.
/// The gate job checks out with `fetch-depth: 0`.
///
/// The diff is taken with NO pathspec and filtered in Rust. Two reasons, both
/// practical: the filter is then unit-testable without a git fixture, and
/// git's exclude-pathspec precedence rules are subtle enough that expressing
/// per-gate exclusions in them would move the policy somewhere nobody reviews.
pub fn invalidating_paths(
    root: &Path,
    head: &str,
    record_sha: &str,
    gate: &super::coverage::GateCoverage,
) -> Option<Vec<String>> {
    invalidating_paths_with(root, head, record_sha, gate, |path| {
        super::amnesty::excused(root, head, path)
    })
}

fn invalidating_paths_with(
    root: &Path,
    head: &str,
    record_sha: &str,
    gate: &super::coverage::GateCoverage,
    mut is_excused: impl FnMut(&str) -> bool,
) -> Option<Vec<String>> {
    if head == record_sha {
        return Some(Vec::new());
    }
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["diff", "--name-only", record_sha, head])
        .stdin(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .filter(|p| super::coverage::invalidates(gate, p))
            // ★ A one-time content-pinned amnesty: a surviving path whose blob
            // at `head` is exactly the grant's pinned content is excused,
            // loudly. Content-pinned, so any later edit to the file changes
            // the OID and invalidates as before. See `amnesty.rs` for the
            // grant, the fail-closed rule, and the removal condition.
            .filter(|p| {
                if is_excused(p) {
                    tracing::warn!(
                        "amnesty: {p} would re-open {} but its content at {head} is the \
                         pinned one-time grant; excused (see gate/amnesty.rs)",
                        gate.id
                    );
                    return false;
                }
                true
            })
            .map(str::to_string)
            .collect(),
    )
}

#[cfg(test)]
pub(crate) fn invalidating_paths_with_amnesty(
    root: &Path,
    head: &str,
    record_sha: &str,
    gate: &super::coverage::GateCoverage,
    table: &[super::amnesty::AmnestyEntry],
) -> Option<Vec<String>> {
    invalidating_paths_with(root, head, record_sha, gate, |path| {
        super::amnesty::excused_by(root, head, path, table)
    })
}
