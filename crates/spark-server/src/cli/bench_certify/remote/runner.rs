// SPDX-License-Identifier: AGPL-3.0-only
//! One unit on one node: submit, follow, fetch, place, classify.
//!
//! The outcome vocabulary is the local runner's ([`RunOutcome`]) and the
//! final word on a record is the local runner's [`classify`], so a record
//! that came over the wire is judged exactly as one written here. What is
//! remote-specific is the retry shape: a lost stream is re-attached from the
//! last seq (bounded), a busy or queue-full refusal is a retryable harness
//! failure the campaign will try once more elsewhere, and a cancel here
//! cancels the job there before returning.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::super::plan::Unit;
use super::super::runner::{GateRunner, RunCtx, RunOutcome, classify};
use super::avarokctl::{Avarokctl, AttachEnd, Exit, StreamEvent, SubmitSpec};
use super::node::Node;
use super::place::{Expect, place};

/// Re-attaches allowed per unit before the stream is given up on. Each one
/// is itself a bounded reconnect inside avarokctl; ten of them is a link that
/// keeps dying, not a blip.
pub const MAX_REATTACH: u32 = 10;

/// avarokctl's `JobKey` bound: `[A-Za-z0-9._-]{1,64}`.
pub const JOB_KEY_MAX: usize = 64;

pub struct RemoteRunner {
    pub avarokctl: Arc<dyn Avarokctl>,
    pub node: Node,
    /// Distinguishes this campaign's jobs from a previous one's on the node.
    pub run_id: String,
    /// The anchor as a full 40-hex commit: the node builds THIS, and the
    /// wire refuses an abbreviation (`ctx.anchor` may be one).
    pub anchor_full: String,
    pub cancel: Arc<AtomicBool>,
    /// Where fetched files land before placement: `<log_dir>/<node>/<job>/`.
    pub scratch: PathBuf,
}

impl RemoteRunner {
    /// `certify-<run>-<node>-<gate>[-s<i>of<n>]`: the same unit on the same
    /// node in the same campaign is the same job, so a retry after a lost
    /// driver resumes rather than duplicates. avarokctl caps a key at 64
    /// characters; when the whole does not fit, the gate's name is cut from
    /// the FRONT so the shard tail — the part that tells sibling units
    /// apart — always survives.
    pub fn job_key(&self, unit: &Unit) -> String {
        let node = self.node.node_id.chars().take(8).collect::<String>();
        let node = if node.is_empty() {
            self.node.addr.replace(['.', ':'], "-")
        } else {
            node
        };
        let head = format!("certify-{}-{node}-", self.run_id);
        let stem = unit.file_stem();
        let room = JOB_KEY_MAX.saturating_sub(head.chars().count());
        let tail: String = stem
            .chars()
            .skip(stem.chars().count().saturating_sub(room))
            .collect();
        format!("{head}{tail}")
    }

    fn harness(reason: String, retryable: bool) -> RunOutcome {
        RunOutcome::Harness { reason, retryable }
    }

    fn refused(
        &self,
        what: &str,
        exit: Exit,
        err: Option<super::avarokctl::ErrorObj>,
    ) -> RunOutcome {
        let retryable = matches!(exit, Exit::Unreachable | Exit::StreamLost)
            || err.as_ref().is_some_and(|e| e.retryable);
        Self::harness(
            format!(
                "{} on {}: {}",
                what,
                self.node.addr,
                err.map_or_else(|| format!("avarokctl exited {exit:?}"), |e| e.to_string())
            ),
            retryable,
        )
    }
}

/// Render an event for the campaign's line stream.
pub fn event_line(ev: &StreamEvent) -> Vec<String> {
    match ev.kind.as_str() {
        "progress" => vec![format!(
            "  [{}] {}",
            ev.phase.as_deref().unwrap_or(""),
            ev.detail.as_deref().unwrap_or("")
        )],
        "log" => ev.lines.clone(),
        "build" => vec![format!(
            "build: {}",
            if ev.cached == Some(true) {
                "cached"
            } else {
                "compiling"
            }
        )],
        "verdict" => vec![
            ev.verdict
                .as_ref()
                .and_then(|v| {
                    Some(format!(
                        "{}: {}",
                        v.get("kind")?.as_str()?,
                        v.get("text")?.as_str()?
                    ))
                })
                .unwrap_or_default(),
        ],
        "done" => vec![format!(
            "done: {}{}",
            ev.outcome.as_deref().unwrap_or("?"),
            ev.reason
                .as_deref()
                .map(|r| format!(" — {r}"))
                .unwrap_or_default()
        )],
        _ => vec![],
    }
}

impl GateRunner for RemoteRunner {
    fn run(&mut self, unit: &Unit, ctx: &RunCtx, on_line: &mut dyn FnMut(&str)) -> RunOutcome {
        let started = Instant::now();
        let spec = SubmitSpec {
            job_key: self.job_key(unit),
            sha: self.anchor_full.clone(),
            gate: unit.id.to_owned(),
            params: unit.shard_param().into_iter().collect(),
            hardware: ctx.hardware.to_owned(),
            max_run_s: u32::try_from(ctx.deadline.as_secs()).ok(),
            note: format!("spark bench certify run {}", self.run_id),
        };
        let submitted = match self.avarokctl.submit(&self.node.addr, &spec) {
            Ok(Ok(s)) => s,
            Ok(Err((exit, err))) => return self.refused("submit", exit, err),
            Err(e) => return Self::harness(format!("submit on {}: {e:#}", self.node.addr), true),
        };
        on_line(&format!(
            "remote {} job {}{}",
            self.node.addr,
            submitted.job_id,
            if submitted.existing { " (resumed)" } else { "" }
        ));

        // Follow, re-attaching from the last seq on a lost stream.
        let mut from_seq = 1;
        let mut reattached = 0;
        let end = loop {
            let left = ctx.deadline.saturating_sub(started.elapsed());
            if left.is_zero() {
                let _ = self.avarokctl.cancel(&self.node.addr, &submitted.job_id);
                return RunOutcome::TimedOut;
            }
            let mut on_event = |ev: &StreamEvent| {
                for l in event_line(ev) {
                    on_line(&l);
                }
            };
            match self.avarokctl.attach(
                &self.node.addr,
                &submitted.job_id,
                from_seq,
                left,
                &self.cancel,
                &mut on_event,
            ) {
                Ok(AttachEnd::StreamLost { last_seq }) => {
                    reattached += 1;
                    if reattached > MAX_REATTACH {
                        break AttachEnd::StreamLost { last_seq };
                    }
                    from_seq = last_seq + 1;
                    on_line(&format!(
                        "stream to {} lost at seq {last_seq}; re-attaching ({reattached}/{MAX_REATTACH})",
                        self.node.addr
                    ));
                }
                Ok(other) => break other,
                Err(e) => {
                    return Self::harness(format!("attach on {}: {e:#}", self.node.addr), true);
                }
            }
        };
        if self.cancel.load(Ordering::SeqCst) {
            let _ = self.avarokctl.cancel(&self.node.addr, &submitted.job_id);
            return RunOutcome::Cancelled;
        }
        let exit_code = match end {
            AttachEnd::Passed { exit_code } => exit_code,
            AttachEnd::JobFailed {
                outcome,
                exit_code,
                detail,
            } => match outcome.as_deref() {
                // A completed run whose verdict was not a pass still left a
                // record; fetch it so the failure is on file, and let
                // `classify` say what it is (a shard's Info is MemberDone).
                Some("completed") => exit_code.or(Some(2)),
                Some("timed_out") => return RunOutcome::TimedOut,
                Some("cancelled") => return RunOutcome::Cancelled,
                // Failed while preparing/building/collecting, or orphaned:
                // nothing was measured; another node may do better.
                _ => return Self::harness(format!("{} on {}", detail, self.node.addr), true),
            },
            AttachEnd::Cancelled => {
                let _ = self.avarokctl.cancel(&self.node.addr, &submitted.job_id);
                return RunOutcome::Cancelled;
            }
            AttachEnd::StreamLost { last_seq } => {
                return Self::harness(
                    format!(
                        "lost the stream from {} at seq {last_seq} {MAX_REATTACH} times",
                        self.node.addr
                    ),
                    true,
                );
            }
            AttachEnd::Failed { exit, error } => return self.refused("attach", exit, error),
        };

        // Fetch and place.
        let scratch = self.scratch.join(&submitted.job_id);
        let files = match self
            .avarokctl
            .fetch(&self.node.addr, &submitted.job_id, &scratch)
        {
            Ok(Ok(f)) => f,
            Ok(Err((exit, err))) => return self.refused("fetch", exit, err),
            Err(e) => return Self::harness(format!("fetch on {}: {e:#}", self.node.addr), true),
        };
        let placed = match place(
            ctx.root,
            ctx.log_dir,
            &files,
            &Expect {
                unit_id: unit.id,
                shard: unit.shard,
                log_stem: &unit.file_stem(),
                anchor: ctx.anchor,
                hardware: ctx.hardware,
            },
        ) {
            Ok(p) => p,
            Err(e) => {
                return Self::harness(
                    format!("record from {} refused: {e:#}", self.node.addr),
                    false,
                );
            }
        };
        let facts = avarok_plugin::gate::read_record(&placed.record)
            .ok()
            .filter(|r| r.benchmark_id == unit.id && r.shard() == unit.shard)
            .map(|r| super::super::runner::facts_of(placed.record.clone(), &r));
        classify(unit, ctx.anchor, exit_code, facts, None)
    }
}

/// How long a remote unit may take: the local deadline plus a build, unless
/// the node already has the anchor built.
pub fn deadline_for(local: Duration, node: &Node, build_allowance: Duration) -> Duration {
    if node.built {
        local
    } else {
        local + build_allowance
    }
}

#[cfg(test)]
#[path = "runner_tests.rs"]
mod runner_tests;
