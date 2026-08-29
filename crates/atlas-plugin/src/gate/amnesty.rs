// SPDX-License-Identifier: AGPL-3.0-only

//! Content-pinned amnesty for a boundary-policy bootstrap.
//!
//! # The grant
//!
//! PR #816 corrects the coverage assigned to the flat concurrency benchmark
//! driver. The policy file is itself a verdict boundary, so its final reviewed
//! blob would otherwise invalidate all ten records before the corrected rule
//! could classify later changes. The grant covers that one blob and no other
//! path.
//!
//! PR #701 and PR #648 completed this same lifecycle: pinned, re-earned, then
//! emptied after every required gate had a fresh record. The expiry test below
//! requires the same removal for this grant.
//!
//! A grant covers only the final reviewed blobs of the listed files. It is
//! the same mechanism accepted for the 2026-08-16 governance bootstrap:
//! a table anyone can read, a pin no later edit can inherit, and a test that
//! demands removal after all ten records have been re-earned.
//!
//! # Why content-pinned rather than waived
//!
//! Each entry pins the exact blob OID this PR lands. `git rev-parse <head>:<path>`
//! names the CONTENT of the file at the commit being checked,
//! so the grant covers precisely the reviewed bytes: the moment anyone edits
//! either file again, the OID changes and invalidation applies exactly as
//! before. There is no time window and no path-level waiver to inherit —
//! a second edit to the taxonomy pays full price, as it should.
//!
//! # Fail-closed
//!
//! Every uncertainty is "not excused": a path not in the table, git missing
//! or failing, an unknown commit, the path absent at `head`, output that is
//! not a 40-hex OID, an OID that is not the pinned one. A false "not excused"
//! costs a re-run; a false "excused" would be a shipped regression behind a
//! green gate — the same asymmetry the rest of the gate is built on.
//!
//! # Residual risk, stated plainly
//!
//! This file cannot itself be in `BOUNDARY_FILES` without circularity: its
//! own landing would then invalidate everything it exists to protect. It is
//! covered only by `GATE_MACHINERY`'s cargo-test rationale, like the rest of
//! the gate bookkeeping. Compensations: the table's exact contents are pinned
//! by `the_table_is_exactly_the_pr_816_grant` (paths, OID format, grant
//! text), every application is logged loudly by `check.rs`, CODEOWNERS
//! review covers the gate directory, and the gate already executes
//! PR-checkout code — so this adds no new attack class, only a reviewed
//! single-file exception to one rule.
//!
//! # Removal condition
//!
//! EMPTY THE TABLE once every required gate has a record newer than
//! `AMNESTY_EPOCH`. At that point the grant protects nothing because every
//! record postdates the grant day and was earned against the amnestied
//! content. `amnesty_expires_once_every_gate_has_a_fresh_record` fails with
//! instructions when that day arrives, so the table cannot quietly outlive
//! its purpose.

use std::path::Path;

/// One excused path: the file, the exact blob its grant covers, and why.
#[derive(Debug, Clone, Copy)]
pub struct AmnestyEntry {
    pub path: &'static str,
    /// The 40-hex blob OID of the file AS THIS PR LANDS IT — computed with
    /// `git rev-parse <head>:<path>` (equivalently `git hash-object <path>`)
    /// once the content is final, in the pin phase. Until pinned it holds
    /// `"PENDING"`, which matches no blob and keeps the grant inert.
    pub head_blob_oid: &'static str,
    pub grant: &'static str,
}

/// End of the PR #816 grant day: 2026-08-30T00:00:00Z. A record counts as
/// fresh only when it postdates the whole grant day.
pub const AMNESTY_EPOCH: u64 = 1_788_048_000;

/// The PR #816 grant: exactly the reviewed coverage-policy blob.
///
/// This does not claim that a benchmark passed. It prevents the old mapping
/// from demanding unrelated campaigns to validate a deterministic policy
/// correction. Any later edit changes the blob OID and restores all-ten
/// invalidation.
pub const ONE_TIME_AMNESTY: [AmnestyEntry; 1] = [AmnestyEntry {
    path: "crates/atlas-plugin/src/gate/coverage.rs",
    head_blob_oid: "ff3f5109bc3159555bb518b7c4d132d5335ad3fd",
    grant: "PR #816 scopes the flat concurrency driver to its own benchmark gate",
}];

/// Whether the one-time grant excuses `path` at `head`.
pub fn excused(root: &Path, head: &str, path: &str) -> bool {
    excused_by(root, head, path, &ONE_TIME_AMNESTY)
}

/// [`excused`] against an explicit table, so tests can pin real OIDs.
///
/// True iff `path` is in `table` AND the blob at `<head>:<path>` is exactly
/// the pinned 40-hex OID. Anything else — including any git failure — is
/// `false`.
pub fn excused_by(root: &Path, head: &str, path: &str, table: &[AmnestyEntry]) -> bool {
    let Some(entry) = table.iter().find(|e| e.path == path) else {
        return false;
    };
    let Ok(out) = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", &format!("{head}:{path}")])
        .stdin(std::process::Stdio::null())
        .output()
    else {
        return false;
    };
    if !out.status.success() {
        return false;
    }
    let oid = String::from_utf8_lossy(&out.stdout).trim().to_string();
    oid.len() == 40 && oid.chars().all(|c| c.is_ascii_hexdigit()) && oid == entry.head_blob_oid
}
