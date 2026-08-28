// SPDX-License-Identifier: AGPL-3.0-only

//! Content-pinned amnesty for a boundary-policy bootstrap.
//!
//! # The grant
//!
//! The current grant is documented on `ONE_TIME_AMNESTY` itself: PR #648's
//! KV-budget accounting fix, one pinned file. (The prior PR #701 grant —
//! three coverage-policy boundary files — completed the full lifecycle here:
//! pinned, re-earned, emptied.)
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
//! by `the_table_is_exactly_the_pr_648_grant` (paths, OID format, grant
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

/// End of the PR #648 grant day: 2026-08-28T00:00:00Z. A record counts as
/// fresh only when it postdates the whole grant day.
/// (The PR #701 grant used 1_787_356_800, end of 2026-08-21 UTC; its table
/// was emptied once every gate re-recorded past that epoch.)
pub const AMNESTY_EPOCH: u64 = 1_787_875_200;

/// The PR #648 grant: one file, the KV-budget accounting fix.
///
/// `factory/build.rs`'s self-relative (auto) KV budget charged the weight
/// loader's transient footprint (checkpoint mapping/staging, ~the size of the
/// safetensors file) as if it were permanent — measured free-delta ~61 GB vs
/// ~27 GB actual steady state on a 27B NVFP4 load (bug shipped in #281). At
/// 0.85 util the phantom alone overruns the decode-floor budget on any box,
/// making the gate unpassable while the fix — being a `crates/` change —
/// would invalidate the nine records already earned on this branch. Exactly
/// the bootstrap shape PR #701 established; same mechanism, one pinned file.
///
/// The prior PR #701 grant covered three boundary files whose landing
/// invalidated all ten GPU records before the narrower test-only rule could
/// help. Every required gate was re-recorded at 2026-08-22, past that grant's
/// epoch, and its table was emptied —
/// `amnesty_expires_once_every_gate_has_a_fresh_record`
/// asserts exactly that and demands such removal, which is the designed end of
/// a one-time grant rather than a change of policy.
///
/// The mechanism is deliberately kept rather than deleted: the module docs
/// record why a content-pinned grant was preferred to a path waiver, and
/// `excused_by` plus its tests stay exercised so the next bootstrap does not
/// have to re-derive it. An empty table is fail-closed by construction — every
/// lookup falls through to "not excused".
pub const ONE_TIME_AMNESTY: [AmnestyEntry; 1] = [AmnestyEntry {
    path: "crates/spark-model/src/factory/build.rs",
    // Pinned in the pin phase: `git hash-object crates/spark-model/src/factory/build.rs`
    // over the fix's final content, immediately before the landing commit.
    // Re-pinned 2026-08-28 after a comment-only wording fix for the typos CI
    // check (OVERcount -> overstate); same commit updates file and pin so the
    // grant never covers unreviewed bytes.
    head_blob_oid: "3d79c147db23ce7d81d35815b37e17a00bd4c488",
    grant: "PR #648 KV-budget fix: stop charging the weight loader's transient \
            (checkpoint staging, ~34 GB on 27B NVFP4) as permanent in the \
            self-relative (auto) budget — the #281 accounting bug made \
            decode-floor unpassable at 0.85 util on any box, and the fix would \
            otherwise invalidate the nine records already earned on this branch",
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
