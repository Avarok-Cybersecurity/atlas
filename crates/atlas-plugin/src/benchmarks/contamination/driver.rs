// SPDX-License-Identifier: AGPL-3.0-only

//! The four-leg state machine. One leg per `next()`, so the pane paints as
//! each leg lands and cancellation takes effect between legs, not at the end.
//!
//! Leg order (see `score.rs` for why it is four legs and not two):
//!
//! 1. **prime** — every probe once, solo, unmeasured. All later legs run
//!    cache-warm, so cache state is equal by construction.
//! 2. **ref ×2** — every probe solo, twice. A probe whose two solo runs
//!    disagree is `AloneUnstable` (#435) and cannot speak to contamination.
//! 3. **rungs** — the probes together inside one concurrent batch per
//!    configured concurrency, padded with copies of the probes themselves —
//!    identical concurrent prompts are exactly the prefix-share pressure a
//!    collision needs.
//! 4. **post** — every probe solo again. A divergence HERE is `Persistent`:
//!    state survived the concurrent episode.
//!
//! Transport failures inside a leg become [`RequestOutcome::Error`] rather
//! than aborting the run — the scorer counts them as `Unmeasured`, which
//! fails the verdict without discarding the legs that did measure.

use crate::hardware::Sensitivity;
use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::benchmarks::one_line;
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{BenchmarkResult, LogLine, RunStatus, Verdict};

use super::prompts::{self, Probe};
use super::report;
use super::score::{Legs, Score, score, verdict};
use super::transcript::{RequestOutcome, Transcript};

const SUMMARY: &str = "Concurrent requests must not change each other's output";
pub const METADATA: PluginMetadata = PluginMetadata::atlas(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "cross-contamination",
    name: "Cross-Contamination Detector",
    summary: SUMMARY,
    detail: "Four legs at temperature 0: prime every probe cache-warm, record each probe's solo \
             output twice, re-run the probes inside concurrent batches at several concurrency \
             rungs, then re-run them solo afterwards. Each probe carries a lexical canary; a \
             foreign canary in a reply is leakage on its own evidence, any other token-level \
             difference from the solo reference is a divergence, and a divergence in the closing \
             solo leg means corrupted state SURVIVED the batch. Zero tolerance — token identity \
             has no noise term to allow for.",
    duration_hint: "~2–5 min",
    updated: "2026-08-09",
    needs_confirmation: false,
    // Determinism under concurrency is a property of the ENGINE, not of a
    // checkpoint; any served model is a valid subject.
    intended_for: None,
    threshold_params: &[],
    // Cross-request state bleed is a correctness property: a throttled box
    // produces the same tokens, slower.
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(CrossContamination::default()),
};

/// Where the state machine is. One leg per `next()`.
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    #[default]
    Prime,
    RefA,
    RefB,
    Rung,
    Post,
    Score,
    Done,
}

#[derive(Default)]
pub struct CrossContamination {
    handle: Option<PluginHandle>,
    phase: Phase,
    concurrencies: Vec<usize>,
    rung_cursor: usize,
    ref_a: Vec<RequestOutcome>,
    ref_b: Vec<RequestOutcome>,
    rungs: Vec<(String, Vec<RequestOutcome>)>,
    post: Vec<RequestOutcome>,
    max_tokens: usize,
    min_completion_tokens: usize,
    timeout: Duration,
    started: Option<Instant>,
    probed: bool,
}

/// The request every leg sends. Greedy and streamed — the transcript equality
/// the scorer asserts is only meaningful at temperature 0.
pub(super) fn request_body(model: &str, prompt: &str, max_tokens: usize) -> Value {
    json!({
        "model": model,
        "stream": true,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    })
}

/// Which probe each slot of a rung batch runs. The first `n_probes` slots are
/// the MEASURED ones, one per probe in probe order — `Legs.rungs` indexes
/// outcomes by prompt, so that identity is load-bearing. The remaining slots
/// are ballast cycling through the probes, which puts every measured request
/// beside both copies of the OTHER prompt (foreign-state pressure) and copies
/// of ITSELF (identical concurrent prompts — prefix-share collision pressure).
pub(super) fn rung_slots(n_probes: usize, conc: usize) -> Vec<usize> {
    (0..conc.max(n_probes)).map(|s| s % n_probes).collect()
}

/// Fold one request's result into what the scorer consumes. An error is data
/// (`Unmeasured` downstream), never an abort — one reset connection must not
/// discard three finished legs.
pub(super) fn to_outcome(r: Result<http::ChatOutcome>) -> RequestOutcome {
    match r {
        Ok(o) => RequestOutcome::Ok(Box::new(Transcript::from(&o))),
        Err(e) => RequestOutcome::Error(one_line(format!("{e:#}"))),
    }
}

impl CrossContamination {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    /// prime + ref A + ref B + rungs + post + score.
    fn total_steps(&self) -> u64 {
        self.concurrencies.len() as u64 + 5
    }

    fn steps_done(&self) -> u64 {
        match self.phase {
            Phase::Prime => 0,
            Phase::RefA => 1,
            Phase::RefB => 2,
            Phase::Rung => 3 + self.rung_cursor as u64,
            Phase::Post => 3 + self.concurrencies.len() as u64,
            Phase::Score => 4 + self.concurrencies.len() as u64,
            Phase::Done => self.total_steps(),
        }
    }

    async fn one(&self, probe: &Probe) -> RequestOutcome {
        let Ok(handle) = self.handle() else {
            return RequestOutcome::Error("benchmark was not loaded".into());
        };
        let target = handle.target();
        let body = request_body(&target.model, &probe.prompt(), self.max_tokens);
        to_outcome(http::chat_stream(target, &body, self.timeout).await)
    }

    /// One solo leg: every probe once, strictly sequentially — concurrency 1
    /// is the leg's whole definition.
    async fn solo_leg(&self, label: &str) -> Result<Vec<RequestOutcome>> {
        let handle = self.handle()?.clone();
        let mut out = Vec::with_capacity(prompts::PROBES.len());
        for probe in &prompts::PROBES {
            handle.check_cancelled()?;
            handle.status(format!("{label} · probe {} solo", probe.name));
            out.push(self.one(probe).await);
        }
        Ok(out)
    }

    /// One rung: the whole batch in flight at once; only the first slot per
    /// probe is measured (see [`rung_slots`]).
    async fn run_rung(&self, conc: usize) -> Vec<RequestOutcome> {
        let slots = rung_slots(prompts::PROBES.len(), conc);
        let futures: Vec<_> = slots
            .iter()
            .map(|&p| self.one(&prompts::PROBES[p]))
            .collect();
        let mut outcomes = futures::future::join_all(futures).await;
        outcomes.truncate(prompts::PROBES.len());
        outcomes
    }

    /// Assemble the recorded legs and score them. Pure over collected state —
    /// this is the driver's decision logic, and it is what the tests exercise
    /// without a server.
    pub(super) fn scored(&self) -> (Score, Verdict) {
        let canaries = prompts::canaries();
        let legs = Legs {
            ref_a: &self.ref_a,
            ref_b: &self.ref_b,
            rungs: &self.rungs,
            post: &self.post,
            canaries: &canaries,
            min_completion_tokens: self.min_completion_tokens,
        };
        let s = score(&legs);
        let v = verdict(&s);
        (s, v)
    }

    fn frame(&self, phase: &str, line: Option<LogLine>) -> BenchmarkResult {
        let mut f = BenchmarkResult::running(phase, self.elapsed())
            .with_progress(self.steps_done(), self.total_steps());
        if let Some(line) = line {
            f = f.log_line(line);
        }
        f
    }
}

impl Plugin for CrossContamination {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.handle = Some(handle);
        self.started = Some(Instant::now());
        async { Ok(()) }
    }
}

impl Benchmark for CrossContamination {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new(
                "concurrencies",
                "Concurrency rungs",
                "Batch sizes for the concurrent leg; the probes ride inside each batch.",
                ParamKind::IntList { min: 2, max: 128 },
                ParamValue::IntList(vec![2, 4, 8]),
            ),
            ParamSpec::new(
                "max_tokens",
                "Max tokens",
                "Output budget per request. The probes answer in well under this.",
                ParamKind::Int { min: 32, max: 4096 },
                ParamValue::Int(256),
            ),
            ParamSpec::new(
                "min_completion_tokens",
                "Completion-token floor",
                "Replies shorter than this are Unmeasured: two empty replies are equal and prove nothing.",
                ParamKind::Int { min: 1, max: 4096 },
                ParamValue::Int(16),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Request timeout",
                "Seconds before a single request is abandoned and scored as Unmeasured.",
                ParamKind::Int { min: 10, max: 3600 },
                ParamValue::Int(300),
            ),
        ]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        let floor = values.usize("min_completion_tokens")?;
        let max_tokens = values.usize("max_tokens")?;
        // Cross-field: a floor at or above the budget makes every leg
        // Unmeasured by construction — a run that cannot pass and looks like a
        // server fault. Reject it against the field instead.
        if floor >= max_tokens {
            bail!(
                "Completion-token floor: {floor} must be below the max_tokens budget \
                 ({max_tokens}), or every reply is Unmeasured by construction"
            );
        }
        self.concurrencies = values
            .int_list("concurrencies")?
            .iter()
            .map(|c| *c as usize)
            .collect();
        self.max_tokens = max_tokens;
        self.min_completion_tokens = floor;
        self.timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        self.phase = Phase::Prime;
        self.probed = false;
        self.rung_cursor = 0;
        self.ref_a.clear();
        self.ref_b.clear();
        self.rungs.clear();
        self.post.clear();
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
            if self.concurrencies.is_empty() {
                bail!("no concurrency rungs configured");
            }
            return Ok(self.frame(
                "probe",
                Some(LogLine::info(format!(
                    "{} · model {} · {} probes × {} rungs",
                    handle.target().base_url,
                    handle.target().model,
                    prompts::PROBES.len(),
                    self.concurrencies.len()
                ))),
            ));
        }

        match self.phase {
            Phase::Prime => {
                // Unmeasured on purpose: this leg exists so every LATER leg
                // runs cache-warm — equality of cache state by construction.
                let primes = self.solo_leg("prime").await?;
                let failed = primes
                    .iter()
                    .filter(|o| matches!(o, RequestOutcome::Error(_)))
                    .count();
                self.phase = Phase::RefA;
                let line = if failed > 0 {
                    LogLine::warn(format!(
                        "{failed} prime request(s) failed — later legs may run cache-cold"
                    ))
                } else {
                    LogLine::info("primed — all measured legs run cache-warm")
                };
                Ok(self.frame("prime", Some(line)))
            }
            Phase::RefA => {
                self.ref_a = self.solo_leg("reference 1/2").await?;
                self.phase = Phase::RefB;
                Ok(self.frame("reference 1/2", None))
            }
            Phase::RefB => {
                self.ref_b = self.solo_leg("reference 2/2").await?;
                self.phase = Phase::Rung;
                Ok(self.frame("reference 2/2", None))
            }
            Phase::Rung => {
                let conc = self.concurrencies[self.rung_cursor];
                handle.status(format!("concurrency {conc} · batch in flight"));
                let measured = self.run_rung(conc).await;
                self.rungs.push((format!("c{conc}"), measured));
                self.rung_cursor += 1;
                if self.rung_cursor >= self.concurrencies.len() {
                    self.phase = Phase::Post;
                }
                handle.progress(self.steps_done(), self.total_steps());
                Ok(self.frame(format!("concurrency {conc}").as_str(), None))
            }
            Phase::Post => {
                self.post = self.solo_leg("post-check").await?;
                self.phase = Phase::Score;
                Ok(self.frame("post-check", None))
            }
            Phase::Score => {
                let (s, v) = self.scored();
                self.phase = Phase::Done;
                let line = LogLine::info(one_line(format!(
                    "{} comparisons: {} identical · {} diverged · {} contaminated · \
                     {} persistent · {} unmeasured",
                    s.compared, s.identical, s.diverged, s.contaminated, s.persistent, s.unmeasured
                )));
                Ok(BenchmarkResult {
                    status: RunStatus::Completed,
                    ..BenchmarkResult::running("done", self.elapsed())
                }
                .with_progress(self.total_steps(), self.total_steps())
                .with_summary(report::summary(&s))
                .with_table(report::table(&s))
                .with_metrics(report::metrics(&s))
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
