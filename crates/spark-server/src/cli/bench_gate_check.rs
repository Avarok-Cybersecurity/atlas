// SPDX-License-Identifier: AGPL-3.0-only

//! `--pull-request-gate-check` — the no-endpoint half of `spark benchmark`.
//!
//! Split from [`super::bench_run`] (exact piecewise copy) at the 500-line
//! boundary; it shares only `repo_root` with the run path and touches no GPU.

use anyhow::Result;
use atlas_plugin::gate;

use super::bench_run::repo_root;

/// `--pull-request-gate-check`: does THIS commit have a passing record for
/// every required gate? Prints a line per bench and exits 1 until they all
/// pass. Pure filesystem reads — fast enough to run on every PR in CI.
pub(super) fn gate_check_cmd(pr: Option<u64>) -> Result<i32> {
    let root = repo_root()?;
    let sha = gate::git_sha(&root)?;
    let gates = gate::check_gates(&root, &sha);
    println!("gate check for {sha} ({})", root.display());
    let mut open = Vec::new();
    for id in gate::REQUIRED_GATES {
        let status = &gates[id];
        match status {
            gate::GateStatus::Pass => println!("  PASS  {id}"),
            gate::GateStatus::Fail(reasons) => {
                println!("  FAIL  {id}");
                for reason in reasons {
                    println!("        - {reason}");
                }
                open.push(id);
            }
            gate::GateStatus::Missing(reason) => {
                println!("  NONE  {id} — {reason}");
                open.push(id);
            }
        }
    }
    // ── ADVISORY: what the classified intent would have asked for ──
    //
    // ★ Printed AFTER the verdict and consulted by nothing. `gate::exit_code`
    // takes only the statuses, by signature, so this cannot reach the exit code
    // without a visible change to that function. The owner's decision is that
    // intent stays advisory until it is proven stable; this is the reporting
    // half of it.
    //
    // `atlas-governance`'s own doctrine says the ledger is advisory
    // "permanently — adding a ledger read would make [the gate] depend on a
    // file any job can append to". Reading it to PRINT is not that; reading it
    // to DECIDE would be, and would need that paragraph rewritten first.
    let roots = gate::pr_taxonomy::load(&root);
    let source = gate::required::intent_source(&root, pr);
    println!();
    match (&source, &roots) {
        (gate::required::IntentSource::NotRequested, _) => {
            println!("intent: not evaluated (no --pr)");
        }
        (gate::required::IntentSource::NotRecorded { ledger }, _) => {
            println!("intent: nothing recorded ({})", ledger.display());
        }
        (gate::required::IntentSource::Degraded { reason }, _) => {
            println!("intent: DEGRADED — {reason}");
        }
        (_, Err(e)) => println!("intent: taxonomy unreadable — {e:#}"),
        (
            gate::required::IntentSource::Recorded {
                categories,
                skipped,
            },
            Ok(roots),
        ) => {
            let report = gate::required::report(&[], source.clone(), roots);
            println!(
                "intent: {} classification(s){}",
                categories.len(),
                if *skipped > 0 {
                    format!(", {skipped} abstained/errored (not counted)")
                } else {
                    String::new()
                }
            );
            for c in categories {
                println!("        {}", c.join("/"));
            }
            let implied = report.set.by_intent;
            println!(
                "        implies: {}",
                if implied.is_empty() {
                    "(nothing)".to_string()
                } else {
                    implied.iter().cloned().collect::<Vec<_>>().join(", ")
                }
            );
            println!("        advisory — does not change the verdict above or the exit code");
        }
    }

    if open.is_empty() {
        println!("all {} required gates pass", gate::REQUIRED_GATES.len());
        Ok(0)
    } else {
        println!(
            "{} bench(es) still need a passing gate record: {}",
            open.len(),
            open.join(", ")
        );
        Ok(1)
    }
}
