// SPDX-License-Identifier: AGPL-3.0-only

//! SSM state poisoning detector: replay the same conversation script and
//! demand byte-identical transcripts every round.
//!
//! # The invariant
//!
//! Atlas is bitwise-deterministic at batch 1 and temperature 0 — the agentic
//! gate's design doc pins that property, and the sampler short-circuits to
//! argmax. Given that, a conversation replayed from scratch against the same
//! server MUST produce the same bytes whether the server is fresh or has
//! served this exact script eleven times already. Anything else is
//! engine-state corruption by construction; there is no stochastic term
//! that could legitimately differ.
//!
//! # What it catches
//!
//! The 2026-08-11 batch4 regression: a prefix-cache / Marconi SSM-snapshot
//! restore bug. Runs 0–7 of the agentic gate accumulated checkpoints for
//! their shared prompt prefix; runs 8–9 then RESTORED a poisoned recurrent
//! state and degenerated to early-EOS (3–5 turns, empty sandbox). This probe
//! replays a 4-turn script 12 times against one server with the flagship
//! recipe's `enable_prefix_caching: true` — the exact path that was poisoned
//! — and fails on the first replay that returns different bytes. The recipe
//! serves with prefix caching ON deliberately: turning it off would make the
//! gate blind to the class of bug it exists to police.
//!
//! # Round structure (one `next()` per round)
//!
//! 0. probe — reachability.
//! 1. baseline — round 0 replays the script once; its transcripts are the
//!    reference every later round is compared to.
//! 2. replays — rounds 1..=N each replay the script from scratch and compare
//!    turn-by-turn against the reference.
//! 3. score — verdict from the round verdicts.
//!
//! Transport failures become [`super::compare::RoundVerdict::Unmeasured`]
//! rather than aborting the run: a dropped connection costs one round, not
//! the eleven that already measured.

use crate::hardware::Sensitivity;
use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::benchmarks::{one_line, transcript::Transcript};
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{BenchmarkResult, LogLine, RunStatus, Verdict};

use super::compare::{self, RoundVerdict};
use super::probe;
use super::score::RoundRecord;

const SUMMARY: &str = "Replayed conversations must come back byte-identical";
pub const METADATA: PluginMetadata = PluginMetadata::atlas(SUMMARY);

/// Rounds pinned at 12, same way the agentic gate pins iterations=10: the
/// incident's poisoning manifested at rounds 8–9 of 10, so the gate carries
/// margin past that without paying for an agentic run's wall. BENCH.toml
/// pins the exact count; this is the driver default.
pub const DEFAULT_ROUNDS: usize = 12;

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "ssm-state-poisoning-gate",
    name: "SSM State Poisoning Gate",
    summary: SUMMARY,
    detail: "Replays a fixed 4-turn conversation script 12 times against one prefix-cached \
             server at temperature 0, comparing every turn against the first round. It splits \
             two failure classes: restore JITTER (same finish reason, length within bounds) is \
             a healthy engine property — Marconi restores the same token from alternating \
             anchors between rounds, so accumulation differs and turns 2-4 come back reworded — \
             and is recorded but passed; restore POISONING collapses the output (early-EOS \
             stubs or runaway, the exact signature the batch4 stack shipped 2026-08-11) and \
             FAILS the gate. Any collapsed or unmeasured round fails; jitter only records. \
             Two self-honesty checks keep the gate from passing vacuously: the reference \
             round must satisfy the script's semantic anchors (poisoning deterministic from \
             round 0 would otherwise certify as Invariant), and every replay's first turn \
             must attest a nonzero prefix-cache restore (with caching off, the gate would \
             green-light a path it never exercised). Serves with --serve-override \
             ssm_cache_slots=256 so the snapshot pool cannot evict mid-run (eviction churn \
             is noise, not the poisoning class). ~8-10 min.",
    duration_hint: "~8–10 min",
    updated: "2026-08-13",
    needs_confirmation: false,
    // The invariant is a property of the ENGINE (deterministic replay must
    // hold for any served model), but the thresholds and recipe binding are
    // defined on the flagship — same footing as the contamination detector,
    // which runs anywhere, with its baseline measured on one checkpoint.
    intended_for: None,
    threshold_params: &[],
    // State-poisoning is a bit-level property of the SSM rollback path; a hot
    // box must never be able to block it.
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(SsmPoison::default()),
};

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    #[default]
    Baseline,
    Replay,
    Score,
    Done,
}

#[derive(Default)]
pub struct SsmPoison {
    handle: Option<PluginHandle>,
    phase: Phase,
    rounds: usize,
    max_tokens: usize,
    timeout: Duration,
    started: Option<Instant>,
    probed: bool,
    /// Round 0's transcripts, one per turn — the reference.
    reference: Vec<Transcript>,
    /// One record per replay round: verdict plus the turn-1 cache
    /// attestation the vacuity check reads.
    replays: Vec<RoundRecord>,
}

impl SsmPoison {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    /// probe + baseline + N replays + score.
    fn total_steps(&self) -> u64 {
        self.rounds as u64 + 3
    }

    fn steps_done(&self) -> u64 {
        match self.phase {
            Phase::Baseline => 1,
            Phase::Replay => 2 + self.replays.len() as u64,
            Phase::Score => 2 + self.rounds as u64,
            Phase::Done => self.total_steps(),
        }
    }

    fn frame(&self, phase: &str, line: Option<LogLine>) -> BenchmarkResult {
        let mut f = BenchmarkResult::running(phase, self.elapsed())
            .with_progress(self.steps_done(), self.total_steps());
        if let Some(line) = line {
            f = f.log_line(line);
        }
        f
    }

    /// One chat turn against the endpoint; an error is returned as text so
    /// the caller decides between a failed round and a failed run.
    async fn turn(&self, messages: &[Value]) -> Result<Transcript> {
        let handle = self.handle()?;
        let target = handle.target();
        let body = probe::request_body(&target.model, messages, self.max_tokens);
        let outcome = http::chat_stream(target, &body, self.timeout)
            .await
            .context("chat request failed")?;
        Ok(Transcript::from(&outcome))
    }

    /// Replay the whole script from scratch. Returns the per-turn
    /// transcripts, or the first error that stopped the replay.
    async fn replay_script(&self, label: &str) -> Result<Vec<Transcript>> {
        let handle = self.handle()?.clone();
        let mut messages: Vec<Value> = Vec::with_capacity(probe::TURNS.len() * 2);
        let mut transcripts = Vec::with_capacity(probe::TURNS.len());
        for (i, turn) in probe::TURNS.iter().enumerate() {
            handle.check_cancelled()?;
            let content = if i == 0 {
                probe::first_turn()
            } else {
                turn.to_string()
            };
            messages.push(json!({"role": "user", "content": content}));
            handle.status(format!(
                "{label} · turn {}/{turns}",
                i + 1,
                turns = probe::TURNS.len()
            ));
            let t = self.turn(&messages).await?;
            messages.push(json!({"role": "assistant", "content": t.text.clone()}));
            transcripts.push(t);
        }
        Ok(transcripts)
    }

    /// The final decision, pure over collected state — the reduction and the
    /// zero-tolerance rule live in `score.rs`, which the tests exercise
    /// without a server.
    pub(super) fn scored(&self) -> (super::score::Score, Verdict) {
        let s = super::score::score(&self.replays);
        let v = super::score::verdict(&s, self.rounds);
        (s, v)
    }
}

impl Plugin for SsmPoison {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.handle = Some(handle);
        self.started = Some(Instant::now());
        async { Ok(()) }
    }
}

impl Benchmark for SsmPoison {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new(
                "rounds",
                "Replay rounds",
                "How many times the script is replayed after the reference round. The incident \
                 poisoned rounds 8-9 of 10; BENCH.toml pins the gate at 12.",
                ParamKind::Int { min: 3, max: 30 },
                ParamValue::Int(DEFAULT_ROUNDS as i64),
            ),
            ParamSpec::new(
                "max_tokens",
                "Max tokens per turn",
                "Output budget per turn. Sized so the COLLAPSE_RATIO_CEIL (2.0) stays \
                 reachable: the script's longest answer is a short paragraph (~a few hundred \
                 tokens), and a runaway replay must be able to reach twice the reference \
                 length before the budget clamps it. The old 256 clamped runaways below the \
                 ceiling on any turn past 128 tokens. A reference turn that hits this budget \
                 fails the reference anchors, so a budget still too small is loud, not silent.",
                ParamKind::Int { min: 32, max: 4096 },
                ParamValue::Int(1024),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Request timeout",
                "Seconds before a single turn is abandoned; the round scores Unmeasured.",
                ParamKind::Int { min: 10, max: 3600 },
                ParamValue::Int(300),
            ),
        ]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        self.rounds = values.usize("rounds")?;
        self.max_tokens = values.usize("max_tokens")?;
        self.timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        self.probed = false;
        self.phase = Phase::Baseline;
        self.reference.clear();
        self.replays.clear();
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;

        if !self.probed {
            self.probed = true;
            http::probe(handle.target(), Duration::from_secs(10))
                .await
                .context("endpoint probe failed — check the target URL and port")?;
            if self.rounds < 3 {
                bail!("need at least 3 replay rounds");
            }
            return Ok(self.frame(
                "probe",
                Some(LogLine::info(format!(
                    "{} · model {} · {} replay rounds, prefix caching under test",
                    handle.target().base_url,
                    handle.target().model,
                    self.rounds
                ))),
            ));
        }

        match self.phase {
            Phase::Baseline => {
                self.reference = self.replay_script("reference").await?;
                // The replays are only held to round 0, so round 0 must be
                // held to the script: poisoning that is deterministic from
                // the first round produces the same wrong bytes every time
                // and would otherwise certify as Invariant.
                let violations = probe::validate_reference(&self.reference);
                if !violations.is_empty() {
                    self.phase = Phase::Done;
                    let reason = format!(
                        "reference round failed the script anchors ({}) — replays compared \
                         against it would certify nothing",
                        violations.join("; ")
                    );
                    let (s, _) = self.scored();
                    let mut metrics = super::report::metrics(&s);
                    metrics.insert("sum_wall_s".to_string(), self.elapsed().as_secs_f64());
                    return Ok(BenchmarkResult {
                        status: RunStatus::Completed,
                        ..BenchmarkResult::running("reference", self.elapsed())
                    }
                    .with_metrics(metrics)
                    .with_verdict(Verdict::fail(reason.clone()))
                    .log_line(LogLine::error(one_line(reason))));
                }
                self.phase = Phase::Replay;
                Ok(self.frame(
                    "reference",
                    Some(LogLine::info(format!(
                        "reference round captured: {} turns, script anchors hold",
                        self.reference.len()
                    ))),
                ))
            }
            Phase::Replay => {
                let n = self.replays.len() + 1;
                // Turn 1 of every replay is the cross-round prefix restore
                // under test; its cached-token attestation is what separates
                // a measured round from a cold rerun of the script.
                let (v, turn1_cached) = match self.replay_script(&format!("replay {n}")).await {
                    Ok(replay) => {
                        let cached = replay.first().map(|t| t.cached_prompt_tokens);
                        (compare::compare_round(&self.reference, &replay), cached)
                    }
                    Err(e) => (
                        RoundVerdict::Unmeasured {
                            reason: one_line(format!("{e:#}")),
                        },
                        None,
                    ),
                };
                let line = match &v {
                    RoundVerdict::Collapsed { turns } => Some(LogLine::error(format!(
                        "replay {n} COLLAPSED — restored state produced degenerate output on \
                         turns {:?} (the SSM state poisoning signature)",
                        turns.iter().map(|t| t.turn).collect::<Vec<_>>()
                    ))),
                    RoundVerdict::Jittered { turns } => Some(LogLine::info(format!(
                        "replay {n} jittered (healthy) on turns {:?} — restore anchor \
                         selection varies between rounds",
                        turns.iter().map(|t| t.turn).collect::<Vec<_>>()
                    ))),
                    _ => None,
                };
                // A collapse or jitter line keeps priority; the vacuity line
                // only fills silence, since collapse already fails the round.
                let line = if line.is_none() && turn1_cached == Some(0) {
                    Some(LogLine::error(format!(
                        "replay {n} restored 0 cached prompt tokens on turn 1 — the round \
                         never exercised the prefix restore path"
                    )))
                } else {
                    line
                };
                self.replays.push(RoundRecord {
                    round: n,
                    verdict: v,
                    turn1_cached,
                });
                if self.replays.len() >= self.rounds {
                    self.phase = Phase::Score;
                }
                Ok(self.frame(&format!("replay {n}"), line))
            }
            Phase::Score => {
                let (s, v) = self.scored();
                self.phase = Phase::Done;
                let line = LogLine::info(one_line(format!(
                    "{} replays: {} invariant · {} jittered · {} collapsed · {} unmeasured",
                    s.rounds, s.invariant, s.jittered, s.collapsed, s.unmeasured
                )));
                // sum_wall_s is runtime state (the Score is pure), so it is
                // added here rather than in `report::metrics`. BENCH.toml
                // bounds it as a blowup detector, same as the agentic gate.
                let mut metrics = super::report::metrics(&s);
                metrics.insert("sum_wall_s".to_string(), self.elapsed().as_secs_f64());
                Ok(BenchmarkResult {
                    status: RunStatus::Completed,
                    ..BenchmarkResult::running("done", self.elapsed())
                }
                .with_progress(self.total_steps(), self.total_steps())
                .with_summary(super::report::summary(&s))
                .with_table(super::report::table(&s))
                .with_metrics(metrics)
                .with_verdict(v)
                .log_line(line))
            }
            Phase::Done => bail!("next() was called after the run finished"),
        }
    }
}

#[cfg(test)]
#[path = "driver_tests.rs"]
mod driver_tests;
