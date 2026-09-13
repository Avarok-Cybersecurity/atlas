// SPDX-License-Identifier: AGPL-3.0-only

//! `spark bench certify` — the certification campaign, in the binary.
//!
//! Replaces the shell driver (`campaign_pr.sh` + `post_and_bank.sh`) that ran
//! every certification through 2026-09-13, keeping each of its rules: the
//! plan comes from `gate::check_gates` (the same SSOT as the gate check), each
//! gate is a child `spark benchmark run … --pull-request-gate`, the drift
//! guard runs before every unit and on a timer during it, the lockfile is
//! heartbeated, the evidence is the record on disk, and the last word is the
//! gate table plus the record-agreement rule.
//!
//! Local mode runs one unit at a time on this box. `--with-nodes` (the next
//! layer) plans across nodes.

pub mod args;
pub mod guard;
pub mod lockfile;
pub mod plan;
pub mod preflight;
pub mod report;
pub mod runner;
pub mod state;
mod text;

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use atlas_plugin::{ArtifactStore, gate, history};

use self::args::CertifyArgs;
use self::runner::{GateRunner, RunCtx, RunOutcome};
use self::state::Campaign;
use self::text::{SummaryJson, describe, human, print_plan, print_summary};

/// How often the drift guard runs while a unit is in flight.
pub const GUARD_EVERY: Duration = Duration::from_secs(60);
/// Build time allowed on top of a unit's expected duration (a cold box builds
/// the binary first; a warm one takes a minute).
pub const BUILD_ALLOWANCE: Duration = Duration::from_secs(1800);

/// Where the campaign's words go: human lines, or one JSON object per line.
pub struct Emit {
    json: bool,
}

impl Emit {
    fn event(&self, kind: &str, fields: serde_json::Value) {
        if !self.json {
            return;
        }
        let mut v = fields;
        if let Some(o) = v.as_object_mut() {
            o.insert("event".into(), kind.into());
            o.insert("at".into(), lockfile::rfc3339(lockfile::now_unix()).into());
        }
        println!("{v}");
    }
    fn say(&self, line: &str) {
        if !self.json {
            eprintln!("certify: {line}");
        }
    }
}

pub async fn certify_cmd(args: CertifyArgs) -> Result<i32> {
    if let Err(msg) = args.validate() {
        bail!("{msg}");
    }
    if !args.with_nodes.is_empty() {
        bail!("--with-nodes is not available in this build yet");
    }
    let emit = Emit { json: args.json };
    let root = super::bench_run::repo_root()?;
    let head = gate::git_sha(&root)?;
    let anchor = args.anchor.clone().unwrap_or_else(|| head.clone());

    // ── the plan, from the SSOT ──
    let statuses = gate::check_gates(&root, &anchor);
    let gates = plan::remaining(&statuses, &args.gates)?;
    let store = ArtifactStore::discover().context("locating ATLAS_HOME")?;
    let measured = |id: &str| measured_secs(&store, id);
    let units = plan::order_local(plan::units(&gates, &measured)?);
    let hardware = match &args.hardware {
        Some(h) => h.clone(),
        None => {
            let k = atlas_plugin::hardware::Hardware::probe().gate_key();
            if k == "unknown" {
                bail!("cannot probe this box's hardware class; pass --hardware (e.g. gb10)");
            }
            k
        }
    };
    let guard_ref = args
        .guard_ref
        .clone()
        .or_else(|| guard::upstream_of_head(&root));
    let serial = plan::serial_estimate_secs(&units);
    emit.event(
        "plan",
        serde_json::json!({
            "anchor": anchor, "hardware": hardware, "guard_ref": guard_ref,
            "gates": gates, "serial_estimate_secs": serial,
            "units": units.iter().map(|u| serde_json::json!({
                "id": u.id, "group": u.group, "class": format!("{:?}", u.class),
                "expected_secs": u.secs(),
                "estimate": match u.estimate {
                    plan::Estimate::Declared(_) => "declared",
                    plan::Estimate::Measured { .. } => "measured",
                },
            })).collect::<Vec<_>>(),
        }),
    );
    if !args.json {
        print_plan(
            &anchor,
            &hardware,
            guard_ref.as_deref(),
            &gates,
            &units,
            serial,
        );
    }

    // ── preflight ──
    let needs: Vec<&'static str> = units
        .iter()
        .filter(|u| u.needs_confirmation)
        .map(|u| u.id)
        .collect();
    let facts = preflight::gather(
        &root,
        &anchor,
        guard_ref.clone(),
        args.no_guard,
        needs,
        args.yes,
    )?;
    let findings = preflight::evaluate(&facts);
    emit.event(
        "preflight",
        serde_json::json!({
            "signer": facts.signer, "atlas_home": facts.atlas_home,
            "findings": findings.iter().map(|f| f.0.clone()).collect::<Vec<_>>(),
        }),
    );
    for f in &findings {
        emit.say(&format!("preflight: {f}"));
    }
    if args.dry_run {
        emit.say(if findings.is_empty() {
            "dry run: preflight clean; nothing was run"
        } else {
            "dry run: preflight would refuse; nothing was run"
        });
        return Ok(i32::from(!findings.is_empty()));
    }
    if !findings.is_empty() {
        bail!(
            "preflight refused ({} finding(s) above); nothing was run",
            findings.len()
        );
    }
    if units.is_empty() {
        emit.say("nothing remaining to run at this commit");
        return finish(&emit, &root, &anchor, None);
    }

    // ── the lock ──
    let now = lockfile::now_unix();
    let branch = current_branch(&root);
    let mut lock = lockfile::LockGuard::claim(
        &root,
        lockfile::LockOwner {
            session_id: format!("certify-{}-{now}", std::process::id()),
            hostname: hostname(),
            user: std::env::var("USER").unwrap_or_default(),
            cwd: root.display().to_string(),
        },
        lockfile::Campaign {
            pr: args.pr,
            branch: branch.clone(),
            anchor_sha: anchor.clone(),
            atlas_home: facts.atlas_home.clone(),
            driver_pid: Some(std::process::id()),
            driver_cmdline: Some(std::env::args().collect::<Vec<_>>().join(" ")),
            started_at: Some(lockfile::rfc3339(now)),
            current_gate: None,
            gates_done: vec![],
            heartbeat_at: Some(lockfile::rfc3339(now)),
            guard_last_rc: None,
        },
        now,
        &pid_alive,
    )?;

    // ── the loop ──
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let flag = cancel.clone();
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                flag.store(true, Ordering::SeqCst);
            }
        });
    }
    let log_dir = args
        .out
        .clone()
        .unwrap_or_else(|| root.join(".certify").join(&anchor));
    std::fs::create_dir_all(&log_dir).with_context(|| format!("creating {}", log_dir.display()))?;
    let exe = std::env::current_exe().context("locating this binary")?;
    let mut runner = runner::LocalChild {
        exe,
        records: Box::new(runner::RepoRecords),
        cancel: cancel.clone(),
        extra_args: vec![],
    };
    let git = guard::GitCli { root: root.clone() };
    let guard_ref_for_loop = if args.no_guard {
        None
    } else {
        guard_ref.clone()
    };
    let mut campaign = Campaign::new(units, args.keep_going);
    let factor = args.timeout_factor;

    while let Some(i) = campaign.next_to_start() {
        let unit = campaign.units[i].clone();
        // Guard before every start.
        let mut guard_rc = 0;
        if let Some(r) = &guard_ref_for_loop {
            let result = guard::drift(&git, &anchor, r).map_err(|e| format!("{e:#}"));
            guard_rc = match &result {
                Ok(guard::Drift::PerfPathMoved { .. }) => 1,
                Ok(_) => 0,
                Err(_) => 2,
            };
            emit.event(
                "guard",
                serde_json::json!({ "unit": unit.id, "result": format!("{result:?}") }),
            );
            if let Some(why) = campaign.guard(result) {
                emit.say(&format!("ABORT: {why}"));
                break;
            }
        }
        lock.beat(unit.id, guard_rc, lockfile::now_unix())?;
        emit.event(
            "start",
            serde_json::json!({ "unit": unit.id, "expected_secs": unit.secs() }),
        );
        emit.say(&format!("▶ {} (expected ~{})", unit.id, human(unit.secs())));
        let deadline = Duration::from_secs((unit.secs() as f64 * factor) as u64) + BUILD_ALLOWANCE;
        let ctx = RunCtx {
            root: &root,
            anchor: &anchor,
            hardware: &hardware,
            yes: args.yes,
            deadline,
            log_dir: &log_dir,
        };
        // Guard on a timer while the unit runs: drift sets the cancel flag,
        // and the reason is read back below so the abort names the paths.
        let drift_seen: Arc<std::sync::Mutex<Option<Result<guard::Drift, String>>>> =
            Arc::new(std::sync::Mutex::new(None));
        let stop_ticker = Arc::new(AtomicBool::new(false));
        let ticker = guard_ref_for_loop.as_ref().map(|r| {
            let (r, root, anchor) = (r.clone(), root.clone(), anchor.clone());
            let (cancel, seen, stop) = (cancel.clone(), drift_seen.clone(), stop_ticker.clone());
            std::thread::spawn(move || {
                let git = guard::GitCli { root };
                let mut waited = Duration::ZERO;
                while !stop.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(500));
                    waited += Duration::from_millis(500);
                    if waited < GUARD_EVERY {
                        continue;
                    }
                    waited = Duration::ZERO;
                    let result = guard::drift(&git, &anchor, &r).map_err(|e| format!("{e:#}"));
                    let bad = !matches!(
                        result,
                        Ok(guard::Drift::Unmoved) | Ok(guard::Drift::MovedHarmlessly { .. })
                    );
                    if bad {
                        *seen.lock().unwrap_or_else(|p| p.into_inner()) = Some(result);
                        cancel.store(true, Ordering::SeqCst);
                        return;
                    }
                }
            })
        });
        let started = std::time::Instant::now();
        let mut on_line = |line: &str| {
            if args.json {
                emit.event("line", serde_json::json!({ "unit": unit.id, "text": line }));
            } else if line.starts_with("  [") || line.contains("Pass:") || line.contains("Fail:") {
                eprintln!("  {} {}", unit.id, line.trim_end());
            }
        };
        let outcome = runner.run(&unit, &ctx, &mut on_line);
        stop_ticker.store(true, Ordering::SeqCst);
        if let Some(t) = ticker {
            let _ = t.join();
        }
        let elapsed = started.elapsed().as_secs();
        emit.event(
            "done",
            serde_json::json!({ "unit": unit.id, "outcome": format!("{outcome:?}"), "elapsed_secs": elapsed }),
        );
        emit.say(&format!(
            "■ {} → {} after {}",
            unit.id,
            describe(&outcome),
            human(elapsed)
        ));
        let drift = drift_seen.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let (RunOutcome::Cancelled, Some(result)) = (&outcome, drift) {
            campaign.finished(i, outcome);
            if let Some(why) = campaign.guard(result) {
                emit.say(&format!("ABORT: {why}"));
            }
            break;
        }
        let retry = campaign.finished(i, outcome.clone());
        if matches!(
            outcome,
            RunOutcome::Passed { .. } | RunOutcome::MemberDone { .. }
        ) {
            lock.done(unit.id)?;
        }
        if retry {
            emit.say(&format!("retrying {} once", unit.id));
        }
    }

    let summary = campaign.summary();
    emit.event(
        "summary",
        serde_json::to_value(SummaryJson::from(&summary))?,
    );
    if !args.json {
        print_summary(&summary);
    }
    lock.release_as(
        if summary.aborted.is_some() {
            "aborted"
        } else {
            "campaign_done"
        },
        lockfile::now_unix(),
    )?;
    finish(&emit, &root, &anchor, Some(&campaign))
}

/// The final gate table + agreement, and the exit code.
fn finish(emit: &Emit, root: &Path, anchor: &str, campaign: Option<&Campaign>) -> Result<i32> {
    let f = report::evaluate(root, anchor)?;
    emit.event(
        "final",
        serde_json::json!({
            "certified": f.certified(), "open": f.open,
            "added": f.added.iter().map(|a| serde_json::json!({
                "path": a.path, "gate": a.benchmark_id, "sha": a.git_sha, "signer": a.signer
            })).collect::<Vec<_>>(),
            "disagreements": f.disagreements.iter().map(ToString::to_string).collect::<Vec<_>>(),
        }),
    );
    if !emit.json {
        report::print(&f, anchor, root);
    }
    Ok(match campaign {
        Some(c) => c.exit_code(f.certified()),
        None => i32::from(!f.certified()) * 2,
    })
}

/// `(secs, recorded_at)` of the newest COMPLETED run of `id` in the history.
fn measured_secs(store: &ArtifactStore, id: &str) -> Option<(u64, u64)> {
    history::load(store, id)
        .into_iter()
        .find(|r| r.frame.status == atlas_plugin::result::RunStatus::Completed)
        .map(|r| (r.frame.elapsed.as_secs(), r.recorded_at))
}

fn pid_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

fn hostname() -> String {
    std::fs::read_to_string("/etc/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

fn current_branch(root: &Path) -> String {
    std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .stdin(std::process::Stdio::null())
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "HEAD".into())
}
