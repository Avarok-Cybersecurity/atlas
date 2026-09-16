// SPDX-License-Identifier: AGPL-3.0-only
//! `--with-nodes`: the same campaign, run on several machines at once.
//!
//! The plan comes from the same SSOT as the local run and the verdict is
//! read the same way; what this module adds is a set of nodes (this box
//! unless `--remote-only`, plus every address that avarokctl reaches and
//! [`node::admit`] accepts), one worker per node that keeps taking the
//! longest eligible unit until nothing is left, a guard tick on the main
//! thread that cancels every worker on drift, and the Speed-mode decision
//! that keeps a box-dependent number on one box unless the boxes are one.

pub mod avarokctl;
pub mod node;
pub mod place;
pub mod runner;
pub mod schedule;
mod text;
pub mod thermal;
pub mod wire;
pub use text::{fleet_json, print_fleet};

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use avarok_plugin::gate::signing;
use avarok_plugin::hardware::{Hardware, HardwareState};

use super::guard;
use super::lockfile::LockGuard;
use super::plan::Unit;
use super::runner::{GateRunner, LocalChild, RepoRecords, RunCtx, RunOutcome};
use super::state::{Campaign, Phase};
use super::{Emit, GUARD_EVERY};
use avarok_plugin::hardware::equivalence::EquivalencePolicy;
use avarok_plugin::hardware::limits::ThermalEnvelope;
use node::Node;
use schedule::SpeedMode;

/// The admitted fleet and the mode it runs in.
pub struct Fleet {
    pub nodes: Vec<Node>,
    pub rejected: Vec<node::Rejection>,
    pub mode: SpeedMode,
    /// The class's declared thermal envelope; `None` only under
    /// `--dangerous-ignore-thermals`.
    pub envelope: Option<ThermalEnvelope>,
}

/// Ask every address, admit what qualifies, decide the Speed mode.
///
/// # Errors
/// When avarokctl cannot be run, or no node at all is admitted.
pub fn assemble(
    avarokctl: &dyn avarokctl::Avarokctl,
    addrs: &[String],
    remote_only: bool,
    wanted: &node::Wanted,
    local_signer: &str,
    envelope: Option<ThermalEnvelope>,
    policy: Option<EquivalencePolicy>,
) -> Result<Fleet> {
    let rows = avarokctl.nodes(addrs)?;
    let mut nodes = Vec::new();
    let mut rejected = Vec::new();
    if !remote_only {
        let hw = Hardware::probe();
        let state = HardwareState::collect();
        nodes.push(node::local(local_signer, &hw, &state));
    }
    for addr in addrs {
        match rows.iter().find(|r| &r.node == addr) {
            Some(row) => match node::admit(row, wanted) {
                Ok(n) => nodes.push(n),
                Err(r) => rejected.push(r),
            },
            None => rejected.push(node::Rejection {
                addr: addr.clone(),
                why: "avarokctl returned no row for it".into(),
            }),
        }
    }
    if nodes.is_empty() {
        bail!(
            "no node admitted: {}",
            rejected
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ")
        );
    }
    let mode = schedule::speed_mode(&nodes, policy);
    Ok(Fleet {
        nodes,
        rejected,
        mode,
        envelope,
    })
}

/// The pieces the workers share.
pub struct Shared<'a> {
    pub root: &'a std::path::Path,
    /// The anchor as the campaign names it (may be abbreviated).
    pub anchor: &'a str,
    pub hardware: &'a str,
    pub yes: bool,
    pub log_dir: &'a std::path::Path,
    pub timeout_factor: f64,
    pub emit: &'a Emit,
    pub cancel: Arc<AtomicBool>,
    /// Live chassis readings for the cool-down (`thermal`).
    pub thermal: &'a dyn thermal::Probe,
    /// `--dangerous-ignore-thermals`: warn instead of parking.
    pub ignore_thermals: bool,
    /// The class's envelope; `None` (only under the flag) parks nothing.
    pub envelope: Option<ThermalEnvelope>,
    /// Build time a node may spend on the anchor before a unit's deadline
    /// counts (`[benchmarks.limits.timing] build_allowance_s`).
    pub build_allowance: Duration,
}

/// One runner per node: this box's child spawner, or a remote driver.
pub fn runners(
    fleet: &Fleet,
    avarokctl: Arc<dyn avarokctl::Avarokctl>,
    run_id: &str,
    anchor_full: &str,
    cancel: Arc<AtomicBool>,
    log_dir: &std::path::Path,
    no_serve_reuse: bool,
) -> Result<Vec<Box<dyn GateRunner + Send>>> {
    let exe = std::env::current_exe().context("locating this binary")?;
    fleet
        .nodes
        .iter()
        .map(|n| -> Result<Box<dyn GateRunner + Send>> {
            if n.local {
                Ok(Box::new(LocalChild {
                    exe: exe.clone(),
                    records: Box::new(RepoRecords),
                    cancel: cancel.clone(),
                    extra_args: LocalChild::reuse_args(no_serve_reuse),
                }))
            } else {
                Ok(Box::new(runner::RemoteRunner {
                    avarokctl: avarokctl.clone(),
                    node: n.clone(),
                    run_id: run_id.to_owned(),
                    anchor_full: anchor_full.to_owned(),
                    cancel: cancel.clone(),
                    scratch: log_dir.join("remote").join(n.addr.replace([':', '/'], "-")),
                }))
            }
        })
        .collect()
}

/// This machine's signing fingerprint, for the local node.
pub fn local_signer(avarok_home: &std::path::Path) -> Result<String> {
    Ok(signing::load_or_create(avarok_home)?
        .fingerprint()
        .to_owned())
}

struct Board {
    campaign: Campaign,
    placed: Vec<Option<usize>>,
}

/// Run the campaign across the fleet. Returns when every unit is settled or
/// the campaign is stopped. The guard runs on this thread; drift cancels
/// every worker and aborts.
pub fn drive(
    campaign: Campaign,
    fleet: &Fleet,
    mut runners: Vec<Box<dyn GateRunner + Send>>,
    shared: &Shared,
    guard_ref: Option<&str>,
    lock: &mut LockGuard,
) -> Result<Campaign> {
    let n = campaign.units.len();
    let board = Arc::new(Mutex::new(Board {
        campaign,
        placed: vec![None; n],
    }));
    let mode = fleet.mode.clone();
    let workers: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = runners
            .iter_mut()
            .enumerate()
            .map(|(k, runner)| {
                let board = board.clone();
                let mode = mode.clone();
                let node = fleet.nodes[k].clone();
                scope.spawn(move || worker(k, &node, runner.as_mut(), &board, &mode, shared))
            })
            .collect();
        // The guard, on this thread, until every worker is done.
        let git = guard::GitCli {
            root: shared.root.to_path_buf(),
        };
        let mut last_guard = std::time::Instant::now();
        let mut poll = guard::Poll::default();
        loop {
            if handles.iter().all(|h| h.is_finished()) {
                break;
            }
            std::thread::sleep(Duration::from_millis(500));
            if last_guard.elapsed() < GUARD_EVERY {
                continue;
            }
            last_guard = std::time::Instant::now();
            let running: Vec<String> = {
                let b = board.lock().unwrap_or_else(|p| p.into_inner());
                b.campaign
                    .units
                    .iter()
                    .zip(&b.campaign.phase)
                    .filter(|(_, p)| **p == Phase::Running)
                    .map(|(u, _)| u.label())
                    .collect()
            };
            let mut guard_rc = 0;
            if let Some(r) = guard_ref {
                let result = guard::drift(&git, shared.anchor, r).map_err(|e| format!("{e:#}"));
                guard_rc = match &result {
                    Ok(guard::Drift::PerfPathMoved { .. }) => 1,
                    Ok(_) => 0,
                    Err(_) => 2,
                };
                match poll.judge(result) {
                    guard::Judgement::Fine => {}
                    guard::Judgement::Blind(n) => shared.emit.say(&format!(
                        "guard: could not answer ({n} of {} allowed in a row); retrying",
                        guard::BLIND_TICKS_ALLOWED
                    )),
                    guard::Judgement::Stop(result) => {
                        let mut b = board.lock().unwrap_or_else(|p| p.into_inner());
                        if let Some(why) = b.campaign.guard(result) {
                            shared.emit.say(&format!("ABORT: {why}"));
                        }
                        shared.cancel.store(true, Ordering::SeqCst);
                    }
                }
            }
            let _ = lock.beat(&running.join(","), guard_rc, super::lockfile::now_unix());
        }
        handles.into_iter().map(|h| h.join()).collect()
    });
    for w in workers {
        w.map_err(|_| anyhow::anyhow!("a node worker panicked"))?;
    }
    let board = Arc::try_unwrap(board)
        .map_err(|_| anyhow::anyhow!("a worker outlived the campaign"))?
        .into_inner()
        .unwrap_or_else(|p| p.into_inner());
    Ok(board.campaign)
}

/// Retryable harness failures in a row after which a node is retired: the
/// second one says the node, not the unit, is the problem.
pub const STRIKES_TO_RETIRE: u32 = 2;

/// One node's loop: take the next eligible unit, run it, report, repeat.
fn worker(
    k: usize,
    node: &Node,
    runner: &mut dyn GateRunner,
    board: &Mutex<Board>,
    mode: &SpeedMode,
    shared: &Shared,
) {
    let mut strikes = 0;
    let mut cool = thermal::Gate::default();
    loop {
        // A box that warmed past its baseline takes nothing more until it
        // is back near it; the others keep working (`thermal`).
        if !cool.may_take(
            node,
            shared.thermal,
            shared.envelope,
            shared.ignore_thermals,
            &|s| {
                shared.emit.event(
                    "thermal",
                    serde_json::json!({ "node": node.addr, "text": s }),
                );
                shared.emit.say(s);
            },
        ) {
            let stopped = board
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .campaign
                .stopped();
            if stopped {
                return;
            }
            std::thread::sleep(thermal::RECHECK);
            continue;
        }
        let picked = {
            let mut b = board.lock().unwrap_or_else(|p| p.into_inner());
            if b.campaign.stopped() {
                return;
            }
            let pending: Vec<bool> = b
                .campaign
                .phase
                .iter()
                .map(|p| *p == Phase::Pending)
                .collect();
            match schedule::next_for(k, &b.campaign.units, &pending, &b.placed, mode) {
                Some(i) => {
                    b.campaign.start(i);
                    b.placed[i] = Some(k);
                    Some((i, b.campaign.units[i].clone()))
                }
                None => {
                    // Nothing this node may take. If something is still
                    // running elsewhere it may come back as a retry, so wait;
                    // if nothing is, this node is done.
                    let any_running = b.campaign.phase.contains(&Phase::Running);
                    if !any_running {
                        return;
                    }
                    None
                }
            }
        };
        let Some((i, unit)) = picked else {
            std::thread::sleep(Duration::from_secs(2));
            continue;
        };
        let outcome = run_one(node, runner, board, shared, &unit, i);
        match outcome {
            RunOutcome::Harness {
                retryable: true, ..
            } => {
                strikes += 1;
                if strikes >= STRIKES_TO_RETIRE {
                    shared.emit.say(&format!(
                        "retiring {} after {strikes} harness failures in a row",
                        node.addr
                    ));
                    return;
                }
            }
            RunOutcome::Passed { .. } | RunOutcome::MemberDone { .. } => strikes = 0,
            _ => {}
        }
    }
}

fn run_one(
    node: &Node,
    runner: &mut dyn GateRunner,
    board: &Mutex<Board>,
    shared: &Shared,
    unit: &Unit,
    i: usize,
) -> RunOutcome {
    let emit = shared.emit;
    let local_deadline = unit.deadline(shared.timeout_factor);
    let deadline = runner::deadline_for(local_deadline, node, shared.build_allowance);
    emit.event(
        "start",
        serde_json::json!({ "unit": unit.label(), "node": node.addr, "expected_secs": unit.secs() }),
    );
    emit.say(&format!(
        "▶ {} on {} (expected ~{})",
        unit.label(),
        node.addr,
        super::text::human(unit.secs())
    ));
    let ctx = RunCtx {
        root: shared.root,
        anchor: shared.anchor,
        hardware: shared.hardware,
        yes: shared.yes,
        deadline,
        log_dir: shared.log_dir,
    };
    let started = std::time::Instant::now();
    let mut on_line = |line: &str| {
        if emit.json {
            emit.event(
                "line",
                serde_json::json!({ "unit": unit.label(), "node": node.addr, "text": line }),
            );
        } else if line.starts_with("  [") || line.contains("Pass:") || line.contains("Fail:") {
            eprintln!("  {}@{} {}", unit.label(), node.addr, line.trim_end());
        }
    };
    let outcome = runner.run(unit, &ctx, &mut on_line);
    let elapsed = started.elapsed().as_secs();
    emit.event(
        "done",
        serde_json::json!({ "unit": unit.label(), "node": node.addr, "outcome": format!("{outcome:?}"), "elapsed_secs": elapsed }),
    );
    emit.say(&format!(
        "■ {} on {} → {} after {}",
        unit.label(),
        node.addr,
        super::text::describe(&outcome),
        super::text::human(elapsed)
    ));
    let mut b = board.lock().unwrap_or_else(|p| p.into_inner());
    let retry = b.campaign.finished(i, outcome.clone());
    if retry {
        // Pending again; another node may take it (anti-affinity prefers
        // one that has not failed it, and a strike here keeps it away).
        b.placed[i] = None;
        emit.say(&format!(
            "retrying {} once (last on {})",
            unit.label(),
            node.addr
        ));
    }
    if matches!(outcome, RunOutcome::Cancelled) {
        shared.cancel.store(true, Ordering::SeqCst);
    }
    outcome
}

/// The scratch dir for a node's fetched files.
pub fn scratch_for(log_dir: &std::path::Path, addr: &str) -> PathBuf {
    log_dir.join("remote").join(addr.replace([':', '/'], "-"))
}
