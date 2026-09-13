// SPDX-License-Identifier: AGPL-3.0-only
//! `--with-nodes`: the same campaign, run on several machines at once.
//!
//! The plan comes from the same SSOT as the local run and the verdict is
//! read the same way; what this module adds is a set of nodes (this box
//! unless `--remote-only`, plus every address that atlasctl reaches and
//! [`node::admit`] accepts), one worker per node that keeps taking the
//! longest eligible unit until nothing is left, a guard tick on the main
//! thread that cancels every worker on drift, and the Speed-mode decision
//! that keeps a box-dependent number on one box unless the boxes are one.

pub mod atlasctl;
pub mod node;
pub mod place;
pub mod runner;
pub mod schedule;
mod text;
pub mod wire;
pub use text::{fleet_json, print_fleet};

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use atlas_plugin::gate::signing;
use atlas_plugin::hardware::{Hardware, HardwareState};

use super::guard;
use super::lockfile::LockGuard;
use super::plan::Unit;
use super::runner::{GateRunner, LocalChild, RepoRecords, RunCtx, RunOutcome};
use super::state::{Campaign, Phase};
use super::{BUILD_ALLOWANCE, Emit, GUARD_EVERY};
use node::Node;
use schedule::SpeedMode;

/// The admitted fleet and the mode it runs in.
pub struct Fleet {
    pub nodes: Vec<Node>,
    pub rejected: Vec<node::Rejection>,
    pub mode: SpeedMode,
}

/// Ask every address, admit what qualifies, decide the Speed mode.
///
/// # Errors
/// When atlasctl cannot be run, or no node at all is admitted.
pub fn assemble(
    atlasctl: &dyn atlasctl::Atlasctl,
    addrs: &[String],
    remote_only: bool,
    wanted: &node::Wanted,
    local_signer: &str,
) -> Result<Fleet> {
    let rows = atlasctl.nodes(addrs)?;
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
                why: "atlasctl returned no row for it".into(),
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
    let mode = schedule::speed_mode(&nodes);
    Ok(Fleet {
        nodes,
        rejected,
        mode,
    })
}

/// The pieces the workers share.
pub struct Shared<'a> {
    pub root: &'a std::path::Path,
    pub anchor: &'a str,
    pub hardware: &'a str,
    pub yes: bool,
    pub log_dir: &'a std::path::Path,
    pub timeout_factor: f64,
    pub emit: &'a Emit,
    pub cancel: Arc<AtomicBool>,
}

/// One runner per node: this box's child spawner, or a remote driver.
pub fn runners(
    fleet: &Fleet,
    atlasctl: Arc<dyn atlasctl::Atlasctl>,
    run_id: &str,
    cancel: Arc<AtomicBool>,
    log_dir: &std::path::Path,
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
                    extra_args: vec![],
                }))
            } else {
                Ok(Box::new(runner::RemoteRunner {
                    atlasctl: atlasctl.clone(),
                    node: n.clone(),
                    run_id: run_id.to_owned(),
                    cancel: cancel.clone(),
                    scratch: log_dir.join("remote").join(n.addr.replace([':', '/'], "-")),
                }))
            }
        })
        .collect()
}

/// This machine's signing fingerprint, for the local node.
pub fn local_signer(atlas_home: &std::path::Path) -> Result<String> {
    Ok(signing::load_or_create(atlas_home)?
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
        loop {
            if handles.iter().all(|h| h.is_finished()) {
                break;
            }
            std::thread::sleep(Duration::from_millis(500));
            if last_guard.elapsed() < GUARD_EVERY {
                continue;
            }
            last_guard = std::time::Instant::now();
            let running: Vec<&'static str> = {
                let b = board.lock().unwrap_or_else(|p| p.into_inner());
                b.campaign
                    .units
                    .iter()
                    .zip(&b.campaign.phase)
                    .filter(|(_, p)| **p == Phase::Running)
                    .map(|(u, _)| u.id)
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
                let bad = !matches!(
                    result,
                    Ok(guard::Drift::Unmoved) | Ok(guard::Drift::MovedHarmlessly { .. })
                );
                if bad {
                    let mut b = board.lock().unwrap_or_else(|p| p.into_inner());
                    if let Some(why) = b.campaign.guard(result) {
                        shared.emit.say(&format!("ABORT: {why}"));
                    }
                    shared.cancel.store(true, Ordering::SeqCst);
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
    loop {
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
    let local_deadline = Duration::from_secs((unit.secs() as f64 * shared.timeout_factor) as u64);
    let deadline = runner::deadline_for(local_deadline, node, BUILD_ALLOWANCE);
    emit.event(
        "start",
        serde_json::json!({ "unit": unit.id, "node": node.addr, "expected_secs": unit.secs() }),
    );
    emit.say(&format!(
        "▶ {} on {} (expected ~{})",
        unit.id,
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
                serde_json::json!({ "unit": unit.id, "node": node.addr, "text": line }),
            );
        } else if line.starts_with("  [") || line.contains("Pass:") || line.contains("Fail:") {
            eprintln!("  {}@{} {}", unit.id, node.addr, line.trim_end());
        }
    };
    let outcome = runner.run(unit, &ctx, &mut on_line);
    let elapsed = started.elapsed().as_secs();
    emit.event(
        "done",
        serde_json::json!({ "unit": unit.id, "node": node.addr, "outcome": format!("{outcome:?}"), "elapsed_secs": elapsed }),
    );
    emit.say(&format!(
        "■ {} on {} → {} after {}",
        unit.id,
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
            unit.id, node.addr
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
