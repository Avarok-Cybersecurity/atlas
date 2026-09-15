// SPDX-License-Identifier: AGPL-3.0-only
//! The local driver: the campaign one unit at a time on this box, with the
//! drift guard on a timer beside each unit.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};

use super::args::CertifyArgs;
use super::runner::{GateRunner, RunCtx, RunOutcome};
use super::state::Campaign;
use super::text::{describe, human};
use super::{Emit, GUARD_EVERY, guard, lockfile, runner};

/// The serial loop: one unit at a time on this box.
#[allow(clippy::too_many_arguments)]
pub(super) fn drive_local(
    args: &CertifyArgs,
    root: &Path,
    anchor: &str,
    hardware: &str,
    guard_ref: Option<&str>,
    cancel: Arc<AtomicBool>,
    log_dir: &Path,
    emit: &Emit,
    lock: &mut lockfile::LockGuard,
    mut campaign: Campaign,
) -> Result<Campaign> {
    let root = root.to_path_buf();
    let anchor = anchor.to_owned();
    let hardware = hardware.to_owned();
    let exe = std::env::current_exe().context("locating this binary")?;
    let mut runner = runner::LocalChild {
        exe,
        records: Box::new(runner::RepoRecords),
        cancel: cancel.clone(),
        extra_args: vec![],
    };
    let git = guard::GitCli { root: root.clone() };
    let guard_ref_for_loop = guard_ref.map(str::to_owned);
    let factor = args.timeout_factor;
    let mut poll = guard::Poll::default();

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
                serde_json::json!({ "unit": unit.label(), "result": format!("{result:?}") }),
            );
            match poll.judge(result) {
                guard::Judgement::Fine => {}
                guard::Judgement::Blind(n) => emit.say(&format!(
                    "guard: could not answer ({n} of {} allowed in a row); retrying",
                    guard::BLIND_TICKS_ALLOWED
                )),
                guard::Judgement::Stop(result) => {
                    if let Some(why) = campaign.guard(result) {
                        emit.say(&format!("ABORT: {why}"));
                    }
                    break;
                }
            }
        }
        lock.beat(&unit.label(), guard_rc, lockfile::now_unix())?;
        emit.event(
            "start",
            serde_json::json!({ "unit": unit.label(), "expected_secs": unit.secs() }),
        );
        emit.say(&format!(
            "▶ {} (expected ~{})",
            unit.label(),
            human(unit.secs())
        ));
        let deadline = unit.deadline(factor);
        let ctx = RunCtx {
            root: &root,
            anchor: &anchor,
            hardware: &hardware,
            yes: args.yes,
            deadline,
            log_dir,
        };
        // Guard on a timer while the unit runs: drift sets the cancel flag,
        // and the reason is read back below so the abort names the paths.
        let drift_seen: Arc<std::sync::Mutex<Option<Result<guard::Drift, String>>>> =
            Arc::new(std::sync::Mutex::new(None));
        let stop_ticker = Arc::new(AtomicBool::new(false));
        let ticker = guard_ref_for_loop.as_ref().map(|r| {
            let mut ticker_poll = guard::Poll::default();
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
                    if let guard::Judgement::Stop(result) = ticker_poll.judge(result) {
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
                emit.event(
                    "line",
                    serde_json::json!({ "unit": unit.label(), "text": line }),
                );
            } else if line.starts_with("  [") || line.contains("Pass:") || line.contains("Fail:") {
                eprintln!("  {} {}", unit.label(), line.trim_end());
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
            serde_json::json!({ "unit": unit.label(), "outcome": format!("{outcome:?}"), "elapsed_secs": elapsed }),
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
            lock.done(&unit.label())?;
        }
        if retry {
            emit.say(&format!("retrying {} once", unit.label()));
        }
    }
    Ok(campaign)
}
