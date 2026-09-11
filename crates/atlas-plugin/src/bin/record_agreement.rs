// SPDX-License-Identifier: AGPL-3.0-only
//! `record_agreement <record.json>...` — do the records a PR adds agree?
//!
//! Invoked from `.github/workflows/ci.yml`'s "One PR, one commit, one signer"
//! step. The workflow collects the added files (it has git and the working
//! tree); this decides the verdict (it has the registry, and therefore each
//! benchmark's `Sensitivity`).
//!
//! Exit 0 if they agree, 1 with a GitHub `::error` annotation per disagreement.

use std::path::Path;

use atlas_plugin::gate::agreement::{AddedRecord, check};

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
        println!("  {a:<58} gate={benchmark_id} sha={git_sha} signer={signer}");
        added.push(AddedRecord {
            path: a.clone(),
            benchmark_id,
            git_sha,
            signer,
        });
    }

    let problems = check(&added);
    if problems.is_empty() {
        println!(
            "all {} added record(s) agree: one commit, and signer agreement \
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
