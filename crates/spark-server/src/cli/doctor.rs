// SPDX-License-Identifier: AGPL-3.0-only
//! `spark doctor` — is this box ready to run a benchmark?
//!
//! Every finding here is a condition that has actually cost hours, and every
//! one of them used to present as the same useless symptom: `recipe "..." is
//! not in the local index (0 cached)`.
//!
//! * `~/.atlas` owned by uid 1000 while the process was uid 996, so the recipe
//!   index was unwritable. Nothing in the tree checked ownership.
//! * `sync-recipes` never run on a fresh box.
//! * A signing identity minted into a scratch `ATLAS_HOME`, so a campaign's
//!   records carried a key nobody had committed — discovered only in CI, after
//!   the GPU-hours.
//!
//! Design rule for this file: **every line must be able to go red.** A check
//! that has only ever been seen green is indistinguishable from one that cannot
//! fail, and this repo has shipped four of those before.

use anyhow::Result;
use atlas_plugin::artifacts::{AtlasHome, HomeFault};
use atlas_plugin::gate;

/// One line of the report.
pub struct Finding {
    pub label: &'static str,
    pub problem: bool,
    pub detail: String,
    /// What to do about it. Empty when there is nothing to do.
    pub remedy: String,
}

impl Finding {
    fn ok(label: &'static str, detail: impl Into<String>) -> Self {
        Self {
            label,
            problem: false,
            detail: detail.into(),
            remedy: String::new(),
        }
    }
    fn bad(label: &'static str, detail: impl Into<String>, remedy: impl Into<String>) -> Self {
        Self {
            label,
            problem: true,
            detail: detail.into(),
            remedy: remedy.into(),
        }
    }
}

/// Where the home is, and whether it came from the environment or the default.
pub fn check_home() -> Finding {
    match AtlasHome::resolve() {
        Err(e) => Finding::bad(
            "home",
            format!("cannot be resolved: {e:#}"),
            "set ATLAS_HOME, or ensure HOME is set and non-empty.",
        ),
        Ok(h) => Finding::ok("home", h.describe()),
    }
}

/// Can this process write there? Probed by writing, not by reading a mode bit.
pub fn check_writable() -> Finding {
    let Ok(h) = AtlasHome::resolve() else {
        return Finding::bad(
            "writable",
            "skipped — the home could not be resolved",
            "fix `home` first.",
        );
    };
    match h.fault() {
        None => Finding::ok("writable", format!("{} is writable", h.root.display())),
        Some(f @ HomeFault::NotWritable { .. }) => Finding::bad(
            "writable",
            format!("{} {f}", h.root.display()),
            "every gate writes its run frames and provisioned artifacts here; \
             an unwritable home fails the run minutes in, reported as an empty \
             recipe index.",
        ),
        Some(f) => Finding::bad(
            "writable",
            format!("{} {f}", h.root.display()),
            "point ATLAS_HOME at a directory this user can create and write.",
        ),
    }
}

/// The signing identity, and — the part that matters — whether it is COMMITTED.
///
/// A key `signing::register` dropped on disk proves this box once signed
/// something, not that anyone vouched for it. `git ls-files` is the question
/// worth asking, and asking the filesystem instead is the mistake that hid a
/// three-box campaign's problem until CI.
pub fn check_identity(repo_root: Option<&std::path::Path>) -> Finding {
    let Ok(h) = AtlasHome::resolve() else {
        return Finding::bad(
            "identity",
            "skipped — the home could not be resolved",
            "fix `home` first.",
        );
    };
    let key = h.root.join("identity").join("ed25519.pk8");
    if !key.exists() {
        return Finding::ok(
            "identity",
            "no signing key yet — one is minted on this box's first gate record",
        );
    }
    let Some(root) = repo_root else {
        return Finding::bad(
            "identity",
            format!("{} exists, but this is not a git repo", key.display()),
            "run from inside the atlas checkout so the committed signer list \
             can be read.",
        );
    };
    let Ok(identity) = gate::signing::load_or_create(&h.root) else {
        return Finding::bad(
            "identity",
            format!("{} is present but unusable", key.display()),
            "delete it and let the next gate run mint a fresh one.",
        );
    };
    let fp = identity.fingerprint().to_string();
    match gate::signing::committed_signers(root) {
        Err(e) => Finding::bad(
            "identity",
            format!("{fp}; could not read .github/record-signers/: {e:#}"),
            "run from inside the checkout.",
        ),
        Ok(list) if list.contains(&fp) => Finding::ok("identity", format!("{fp}, committed")),
        Ok(list) => Finding::bad(
            "identity",
            format!(
                "{fp} is NOT committed in .github/record-signers/ ({} signer(s) are)",
                list.len()
            ),
            "commit the one-line .pub beside the records this box produces, and \
             remember every record one PR adds for a SPEED-class gate must carry \
             the same fingerprint — a campaign split across boxes cannot be \
             certified.",
        ),
    }
}

/// Has the recipe index been populated? An empty index is the single most
/// common cause of a gate dying seconds after it starts.
pub fn check_recipes() -> Finding {
    let Ok(h) = AtlasHome::resolve() else {
        return Finding::bad(
            "recipes",
            "skipped — the home could not be resolved",
            "fix `home` first.",
        );
    };
    let index = h.root.join("atlas-recipes").join("index.json");
    match std::fs::read_to_string(&index) {
        Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(v) => {
                let n = v
                    .get("recipes")
                    .and_then(|r| r.as_array().map(|a| a.len()))
                    .or_else(|| v.as_array().map(|a| a.len()))
                    .unwrap_or(0);
                if n == 0 {
                    Finding::bad(
                        "recipes",
                        format!("{} parses but lists none", index.display()),
                        "run `spark sync-recipes`.",
                    )
                } else {
                    Finding::ok("recipes", format!("{n} cached in {}", index.display()))
                }
            }
            Err(e) => Finding::bad(
                "recipes",
                format!("{} is not valid JSON: {e}", index.display()),
                "delete it and run `spark sync-recipes`.",
            ),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Finding::bad(
            "recipes",
            format!("{} has never been written", index.display()),
            "run `spark sync-recipes`.",
        ),
        // The distinction that matters: unreadable is NOT absent. Telling an
        // operator to sync when the file is there but unreadable sends them to
        // the network and then fails on the same path.
        Err(e) => Finding::bad(
            "recipes",
            format!("{} exists but cannot be read: {e}", index.display()),
            "this is a permission problem, not a missing sync — check `writable` \
             above; `spark sync-recipes` would fail on the same path.",
        ),
    }
}

/// Run every check. Returns the findings and whether any is a problem.
pub fn run(repo_root: Option<&std::path::Path>) -> (Vec<Finding>, bool) {
    let findings = vec![
        check_home(),
        check_writable(),
        check_identity(repo_root),
        check_recipes(),
    ];
    let bad = findings.iter().any(|f| f.problem);
    (findings, bad)
}

/// `spark doctor`. Exit code 1 when anything is wrong, so a script can rely on it.
pub fn dispatch() -> Result<i32> {
    let repo_root = super::bench_run::repo_root().ok();
    let (findings, bad) = run(repo_root.as_deref());
    for f in &findings {
        let mark = if f.problem { "PROBLEM" } else { "ok" };
        println!("{:>8}  {:<9} {}", mark, f.label, f.detail);
        if f.problem && !f.remedy.is_empty() {
            println!("          {:<9} {}", "", f.remedy);
        }
    }
    println!();
    if bad {
        println!(
            "{} problem(s) found.",
            findings.iter().filter(|f| f.problem).count()
        );
    } else {
        println!("no problems found.");
    }
    Ok(i32::from(bad))
}

#[cfg(test)]
#[path = "doctor_tests.rs"]
mod doctor_tests;
