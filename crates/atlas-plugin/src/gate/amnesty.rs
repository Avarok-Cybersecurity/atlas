// SPDX-License-Identifier: AGPL-3.0-only

//! A one-time, content-pinned amnesty for the 2026-08-16 governance PR.
//!
//! # The grant
//!
//! `.github/pr-taxonomy.json` and `check.rs` are
//! `coverage::BOUNDARY_FILES` entries: touching either invalidates every
//! standing gate record, at ~4h19m of GPU to re-earn. The 2026-08-16
//! governance PR has to touch exactly those two files — the taxonomy to fill
//! its empty `_benches` leaves, `check.rs` to add the hook that consults this
//! table — and the user authorized a one-time bypass for that landing alone
//! ("just this one time", 2026-08-16). This module is that bypass, made
//! auditable: a table anyone can read, a pin nobody can stretch, a test that
//! demands its own removal.
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
//! by `the_table_is_exactly_the_2026_08_16_grant` (entry count, paths, OID
//! format), every application is logged loudly by `check.rs`, CODEOWNERS
//! review covers the gate directory, and the gate already executes
//! PR-checkout code — so this adds no new attack class, only a reviewed
//! two-file exception to one rule.
//!
//! # Removal condition
//!
//! EMPTY THE TABLE once every required gate has a record newer than
//! `AMNESTY_EPOCH`. At that point the grant protects nothing — every
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

/// When the grant was made: end of 2026-08-16 UTC (`1786924800` =
/// 2026-08-17T00:00:00Z). End of day rather than midnight because the
/// standing records this amnesty protects were themselves earned earlier on
/// 2026-08-16 — a record only counts as "fresh" against the grant if it
/// postdates the whole grant day.
pub const AMNESTY_EPOCH: u64 = 1_786_924_800;

/// The whole grant. When this table is empty the module is inert.
pub const ONE_TIME_AMNESTY: [AmnestyEntry; 0] = [
    // EMPTIED 2026-08-17. Every required gate now carries a record newer than
    // AMNESTY_EPOCH: the ten-gate suite was re-cut at sha 4012c9b7e1 (vision,
    // video, ttft-warm, ttft-cold, ssm-state-poisoning, decode-floor,
    // concurrency-sweep, agentic-webserver, bfcl-subset 84.22/84.12,
    // bfcl-subset-echolp 86.25/86.61 — all PASS). The grant has been fully
    // re-earned by measurement, so it protects nothing and the module is inert,
    // exactly as `amnesty_expires_once_every_gate_has_a_fresh_record` demands.
];

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
