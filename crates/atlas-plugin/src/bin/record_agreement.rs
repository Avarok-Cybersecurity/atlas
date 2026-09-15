// SPDX-License-Identifier: AGPL-3.0-only
//! `record_agreement <record.json>...` — do the records a PR adds agree?
//!
//! Invoked from `.github/workflows/ci.yml`'s record-agreement step. The
//! workflow collects the added files; this decides the verdict: it has the
//! registry (each benchmark's `Sensitivity`) and, run from the checkout, the
//! repository at the PR head — so each record's STANDING at that head is
//! judged here too, by the same rule the gate check uses
//! (`gate::record_standing`) — by content, never ancestry, because the
//! repository squash-merges. The head is `HEAD` of the current directory's
//! repository, or `RECORD_AGREEMENT_HEAD` when set.
//!
//! Exit 0 if they agree, 1 with a GitHub `::error` annotation per disagreement.

use std::path::Path;

use atlas_plugin::gate::Standing;
use atlas_plugin::gate::agreement::{AddedRecord, check, standing_at};

fn field(v: &serde_json::Value, key: &str) -> Option<String> {
    v.get(key).and_then(|x| x.as_str()).map(str::to_owned)
}

/// The benchmark id is taken from the record's `benchmark_id`, falling back to
/// the directory name (`.benchmarks/<id>/<date>-<sha>.json`) when the field is
/// absent — pre-schema records on old branches have no field, and refusing them
/// would fail a PR for a record it did not add.
fn benchmark_id_of(path: &Path, v: &serde_json::Value) -> Option<String> {
    field(v, "benchmark_id").or_else(|| {
        path.parent()?
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
    })
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        println!("this PR adds no records — nothing to agree on.");
        return std::process::ExitCode::SUCCESS;
    }

    let root = match std::env::current_dir()
        .ok()
        .and_then(|d| atlas_plugin::gate::git_rev_parse_toplevel(&d).ok())
    {
        Some(r) => r,
        None => {
            println!("::error title=No repository::record_agreement must run inside the checkout");
            return std::process::ExitCode::FAILURE;
        }
    };
    let head = match std::env::var("RECORD_AGREEMENT_HEAD")
        .ok()
        .or_else(|| atlas_plugin::gate::git_sha(&root).ok())
    {
        Some(h) => h,
        None => {
            println!("::error title=No head::cannot resolve the head to judge standing against");
            return std::process::ExitCode::FAILURE;
        }
    };
    println!("head {head}");

    let mut added = Vec::new();
    for a in &args {
        let path = Path::new(a);
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                println!("::error title=Unreadable record::{a}: {e}");
                return std::process::ExitCode::FAILURE;
            }
        };
        let v: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                println!("::error title=Malformed record::{a}: {e}");
                return std::process::ExitCode::FAILURE;
            }
        };
        let sig_path = format!("{a}.sig");
        let signer = std::fs::read_to_string(&sig_path)
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|s| field(&s, "key"))
            .unwrap_or_else(|| "NO_SIDECAR".into());
        let Some(benchmark_id) = benchmark_id_of(path, &v) else {
            println!("::error title=Unattributable record::{a} names no benchmark.");
            return std::process::ExitCode::FAILURE;
        };
        let git_sha = field(&v, "git_sha").unwrap_or_else(|| "MISSING".into());
        // The hardware capture, when the record parses as a full GateRecord.
        // A pre-schema record has none, and is then equivalent to nothing —
        // the one-box rule applies to it exactly as before.
        let parsed = atlas_plugin::gate::read_record(path).ok();
        let hardware = parsed
            .as_ref()
            .map(atlas_plugin::hardware::equivalence::HardwareFingerprint::from_record);
        let hardware_class = parsed
            .as_ref()
            .map_or_else(|| "unknown".to_string(), |r| r.hardware.gate_key());
        // A record that does not even parse cannot be shown to stand.
        let standing = parsed
            .as_ref()
            .map_or(Standing::Unknown, |r| standing_at(&root, &head, r));
        println!(
            "  {a:<58} gate={benchmark_id} sha={git_sha} signer={signer} standing={standing:?}"
        );
        added.push(AddedRecord {
            path: a.clone(),
            benchmark_id,
            git_sha,
            signer,
            hardware,
            hardware_class,
            standing,
        });
    }

    let problems = check(&root, &added);
    if problems.is_empty() {
        println!(
            "all {} added record(s) agree: each stands at {head}, and signer agreement \
             holds for every speed-class gate.",
            added.len()
        );
        return std::process::ExitCode::SUCCESS;
    }
    for p in &problems {
        println!("::error title=Records do not agree::{p}");
    }
    std::process::ExitCode::FAILURE
}
