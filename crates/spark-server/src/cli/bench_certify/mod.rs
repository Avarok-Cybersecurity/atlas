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
pub mod local;
pub mod lockfile;
pub mod plan;
pub mod preflight;
pub mod remote;
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
use self::local::drive_local;
use self::state::Campaign;
use self::text::{SummaryJson, print_plan, print_summary};

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
    let emit = Emit { json: args.json };
    let root = super::bench_run::repo_root()?;
    let head = gate::git_sha(&root)?;
    let anchor = args.anchor.clone().unwrap_or_else(|| head.clone());

    // ── the plan, from the SSOT ──
    let statuses = gate::check_gates(&root, &anchor);
    let gates = plan::remaining(&statuses, &args.gates)?;
    let store = ArtifactStore::discover().context("locating ATLAS_HOME")?;
    let measured = |id: &str| measured_secs(&store, id);
    let boxes = args.with_nodes.len() + usize::from(!args.remote_only);
    let wanted = args.shards.unwrap_or_else(|| plan::shard_count(boxes));
    let owed =
        |g: &'static gate::group::BenchmarkGroup| gate::shards_owed(&root, g, &anchor, wanted);
    let units = plan::order_local(plan::units(&gates, &measured, &owed)?);
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
            "gates": gates, "shards": wanted, "serial_estimate_secs": serial,
            "units": units.iter().map(|u| serde_json::json!({
                "id": u.label(), "group": u.group, "shard": u.shard, "class": format!("{:?}", u.class),
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
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    // A server a dead campaign left leased on this box is ours to stop, and
    // it would otherwise read as "another spark process" below.
    if let Some(l) = super::bench_lease::release_if_orphaned(&store)? {
        emit.say(&format!(
            "released the leased server of a campaign that is gone (pid {}, port {}, {})",
            l.pid, l.port, l.model
        ));
    }
    let facts = preflight::gather(
        &root,
        &anchor,
        guard_ref.clone(),
        args.no_guard,
        needs,
        args.yes,
        args.remote_only,
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

    // ── the fleet (--with-nodes) ──
    // A node builds the anchor from the trusted remote, and the wire names
    // a commit by its full 40-hex sha; the campaign's own `anchor` may be
    // the abbreviation `git_sha` prints.
    let anchor_full = gate::git_rev_parse(&root, &anchor)?;
    let fleet = if args.with_nodes.is_empty() {
        None
    } else {
        let atlasctl = remote::atlasctl::SubprocessAtlasctl::locate(args.atlasctl.as_deref())?;
        let wanted = remote::node::Wanted {
            hardware: &hardware,
            committed_signers: &facts.committed_signers,
            anchor: &anchor_full,
            min_free_fraction: preflight::MIN_FREE_FRACTION,
        };
        let f = remote::assemble(
            &atlasctl,
            &args.with_nodes,
            args.remote_only,
            &wanted,
            &facts.signer,
        )?;
        let plan = remote::schedule::simulate(&units, &f.nodes, &f.mode, BUILD_ALLOWANCE.as_secs());
        emit.event("fleet", remote::fleet_json(&f, &units, &plan));
        if !args.json {
            remote::print_fleet(&f, &units, &plan);
        }
        Some(f)
    };
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
    let guard_ref_for_loop = if args.no_guard {
        None
    } else {
        guard_ref.clone()
    };
    let campaign = Campaign::new(units, args.keep_going);
    let campaign = match &fleet {
        None => drive_local(
            &args,
            &root,
            &anchor,
            &hardware,
            guard_ref_for_loop.as_deref(),
            cancel.clone(),
            &log_dir,
            &emit,
            &mut lock,
            campaign,
        )?,
        Some(f) => {
            let atlasctl: Arc<dyn remote::atlasctl::Atlasctl> = Arc::new(
                remote::atlasctl::SubprocessAtlasctl::locate(args.atlasctl.as_deref())?,
            );
            let run_id = format!("{}-{}", &anchor[..anchor.len().min(10)], now);
            let runners = remote::runners(
                f,
                atlasctl.clone(),
                &run_id,
                &anchor_full,
                cancel.clone(),
                &log_dir,
                args.no_serve_reuse,
            )?;
            let thermal = remote::thermal::FleetProbe {
                atlasctl: atlasctl.clone(),
            };
            let shared = remote::Shared {
                root: &root,
                anchor: &anchor,
                hardware: &hardware,
                yes: args.yes,
                log_dir: &log_dir,
                timeout_factor: args.timeout_factor,
                emit: &emit,
                cancel: cancel.clone(),
                thermal: &thermal,
            };
            remote::drive(
                campaign,
                f,
                runners,
                &shared,
                guard_ref_for_loop.as_deref(),
                &mut lock,
            )?
        }
    };
    // Whatever the last unit left serving is not needed any more.
    if let Some(l) = super::bench_lease::release(&store)? {
        emit.say(&format!(
            "released the leased server (pid {}, port {}, {})",
            l.pid, l.port, l.model
        ));
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
            lockfile::TERMINAL_STATUSES[1]
        } else {
            lockfile::TERMINAL_STATUSES[0]
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
/// The newest completed run of `id`, as WHOLE-DRAW seconds: a shard run
/// (`--param shard=i/n`) is scaled back up by its count, since the planner
/// divides by the count it wants. Without this the first campaign after
/// arbitrary shards planned every shard from the last shard's time divided
/// by six again — 5 min for a 28 min unit — and the deadline would have
/// killed them (2026-09-15, caught in a dry run).
fn measured_secs(store: &ArtifactStore, id: &str) -> Option<(u64, u64)> {
    history::load(store, id)
        .into_iter()
        .find(|r| r.frame.status == atlas_plugin::result::RunStatus::Completed)
        .map(|r| {
            (
                whole_draw_secs(r.frame.elapsed.as_secs(), &r.params),
                r.recorded_at,
            )
        })
}

/// `elapsed` of a run scaled to the whole draw: `× n` for `shard=i/n`.
pub fn whole_draw_secs(elapsed: u64, params: &std::collections::BTreeMap<String, String>) -> u64 {
    let count = params
        .get("shard")
        .and_then(|s| s.split_once('/'))
        .and_then(|(_, n)| n.trim().parse::<u64>().ok())
        .filter(|n| *n > 1);
    match count {
        Some(n) => elapsed.saturating_mul(n),
        None => elapsed,
    }
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
