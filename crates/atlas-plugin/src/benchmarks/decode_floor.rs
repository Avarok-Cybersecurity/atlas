// SPDX-License-Identifier: AGPL-3.0-only

//! Decode Floor Gate — the pinned single-user DECODE throughput floor.
//!
//! `quick-speed-bench` is deliberately a measurement tool: knobs everywhere,
//! no baseline, excused from the gate set. This driver is its opposite: every
//! generation knob is PINNED so that two runs are comparable by construction,
//! and the one number it reports — the MEDIAN server decode rate across three
//! runs — is judged against a committed BENCH.toml threshold under
//! `--pull-request-gate`. A promotion candidate (`gate::coverage::
//! PROMOTION_CANDIDATES`), not REQUIRED: promotion waits on a >=10-run sigma
//! calibration of the floor.
//!
//! # The pins (the benchmark's definition, not parameters)
//!
//! * **Prompt**: one MinHeap-class code prompt (`MINHEAP_PROMPT`, committed
//!   below). Prompt class moves the accept rate — counting prompts accept
//!   drafts near ceiling and inflate tok/s, natural code text accepts ~2–2.5
//!   per verify — so the class is part of the metric's identity.
//! * **Request**: `temperature 0.0, seed 0, max_tokens 1500`, and
//!   `reasoning_effort: "none"` IN THE BODY. Thinking-off is per-request (it
//!   works since the medium-default change), NOT a serve flag — the gate
//!   serve needs no operator flags beyond the recipe. This matters because
//!   speculative dispatch is hard-gated OFF inside `<think>`: a thinking-on
//!   run measures the serial floor, not the engine.
//! * **Runs**: exactly 3, no warmup knob. The metric is the MEDIAN
//!   `usage."response_token/s"`, so one cold or one lucky run cannot carry
//!   the verdict.
//!
//! # Vacuity pins — INCONCLUSIVE, never PASS
//!
//! A decode floor measured on a run that decoded almost nothing, or with
//! speculation silently disengaged, is not a measurement. Each pin failing
//! makes the run INCONCLUSIVE (rendered as a failing verdict, like the video
//! gate's — a run that measured nothing must not read as green):
//!
//! * every run's `completion_tokens >= 1200` (of the 1500 cap);
//! * every run reports the server decode rate (`usage."response_token/s"`);
//! * `accept_len_mean >= 1.5`, derived from
//!   `usage.completion_tokens_details.accepted_prediction_tokens` — the
//!   accept-stats instrumentation. Per run, `completion / (completion −
//!   accepted)` is emitted-tokens-per-decode-step: `1 + accepted/steps`, the
//!   closest honest derivation of accept depth from the wire field (verify
//!   steps are not on the wire; serial steps make this a LOWER bound on the
//!   per-verify accept length, so the pin cannot be flattered). If the field
//!   is absent or zero the run says so by name: it depends on the
//!   accept-stats commit, or the serve is not speculating.

use std::collections::BTreeMap;
use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::json;

use crate::benchmark::{Benchmark, BenchmarkDescriptor, ModelExpectation};
use crate::benchmarks::stats;
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{
    BenchmarkResult, Cell, CellStyle, Column, LogLine, ResultTable, RunStatus, Stat, Verdict,
};

const SUMMARY: &str = "Pinned decode-rate floor: 3 fixed runs of one code prompt, \
                       median server decode tok/s vs a committed threshold";
pub const METADATA: PluginMetadata = PluginMetadata::atlas(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "decode-floor",
    name: "Decode Floor Gate",
    summary: SUMMARY,
    detail: "Three timed streaming runs of one committed MinHeap-class code prompt, with every \
             generation knob pinned (temperature 0, seed 0, max_tokens 1500, thinking off via \
             a per-request reasoning_effort \"none\" — no serve flag needed). The metric is the \
             MEDIAN server decode rate (usage.\"response_token/s\"), judged against the \
             BENCH.toml floor under --pull-request-gate. Vacuity pins make the run \
             INCONCLUSIVE rather than PASS when it measured nothing: every run must emit \
             >=1200 of the 1500-token budget, report the server rate, and show \
             accept_len_mean >= 1.5 derived from \
             usage.completion_tokens_details.accepted_prediction_tokens (requires the \
             accept-stats instrumentation; a serve that is not speculating cannot pass this \
             gate's floor honestly). A PROMOTION CANDIDATE: runs as debt on release cuts; \
             REQUIRED status waits on a >=10-run sigma calibration.",
    duration_hint: "~3–6 min",
    updated: "2026-08-15",
    // The floor in BENCH.toml is measured on the dense Qwen3.8-27B NVFP4
    // checkpoint; the driver measures whatever it is pointed at, but only
    // that family has a committed baseline to judge against.
    intended_for: Some(ModelExpectation {
        families: &["qwen3.8-27b"],
        note: "The decode floor is recorded for unsloth/Qwen3.8-27B-NVFP4 (n=3 basis, \
               2026-08-15). Other checkpoints run fine but have no committed floor to be \
               judged against — a number with no baseline gates nothing.",
    }),
    threshold_params: &[],
    needs_confirmation: false,
    ctor: || Box::new(DecodeFloor::default()),
};

/// Timed runs. PINNED — the median-of-3 is the metric's definition, and a
/// different run count is a different benchmark.
pub(crate) const RUNS: usize = 3;
/// Output budget. PINNED at the measured basis (MinHeap 1500).
pub(crate) const MAX_TOKENS: usize = 1500;
/// Vacuity floor on every run's `completion_tokens`.
pub(crate) const MIN_OUTPUT_TOKENS: usize = 1200;
/// Vacuity floor on the derived tokens-per-decode-step.
pub(crate) const MIN_ACCEPT_LEN: f64 = 1.5;

/// The committed code prompt. MinHeap-class on purpose: a structured,
/// code-shaped generation whose accept behaviour is the documented middle of
/// the road (~2–2.5 per verify), unlike counting prompts which accept near
/// ceiling and flatter the rate. Owned here — benchmark drivers must not
/// import each other, so nothing is borrowed from quick-speed's fixtures.
pub(crate) const MINHEAP_PROMPT: &str = "Implement a complete, production-quality MinHeap class in Python. Include the methods \
     insert, extract_min, peek, heapify (bottom-up from an arbitrary list), decrease_key, \
     delete_at_index, merge (with another MinHeap), __len__ and __iter__. Every method needs a \
     full docstring with time-complexity analysis. Then write a comprehensive pytest test \
     suite covering the empty heap, a single element, duplicate keys, and long interleaved \
     insert/extract sequences. Finish with a line-by-line explanation of the sift_up and \
     sift_down invariants. Be exhaustive and do not stop early.";

/// One timed run, reduced to what the pins and the metric need.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RunObs {
    pub completion_tokens: usize,
    pub server_tps: Option<f64>,
    /// `None` = the server reported no details object (no instrumentation);
    /// `Some(0)` = instrumented but nothing accepted. The pins treat the two
    /// differently only in the message — both are INCONCLUSIVE.
    pub accepted_prediction_tokens: Option<usize>,
    pub e2e_ms: f64,
}

impl RunObs {
    pub(crate) fn from_outcome(o: &http::ChatOutcome) -> Self {
        Self {
            completion_tokens: o.completion_tokens,
            server_tps: o.server_tps,
            accepted_prediction_tokens: o.accepted_prediction_tokens,
            e2e_ms: o.e2e_ms,
        }
    }

    /// Emitted tokens per decode step, `1 + accepted/steps` — the honest
    /// accept-depth lower bound derivable from the wire (see module docs).
    /// `None` when it cannot be derived (no accept field, or a corrupt
    /// `accepted >= completion` which would divide by zero or go negative).
    pub(crate) fn accept_len(&self) -> Option<f64> {
        let accepted = self.accepted_prediction_tokens?;
        (accepted < self.completion_tokens && self.completion_tokens > 0)
            .then(|| self.completion_tokens as f64 / (self.completion_tokens - accepted) as f64)
    }
}

/// What three runs add up to. Pure — the whole verdict is unit-testable
/// without an endpoint.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Evaluation {
    /// A vacuity pin failed; the message names which one and why. Never PASS.
    Inconclusive(String),
    Measured {
        /// MEDIAN server decode tok/s across the runs — THE metric.
        median_decode_tok_s: f64,
        /// Minimum `completion_tokens` across runs, so the BENCH.toml
        /// `output_tokens >= 1200` bound means "every run", not "on average".
        min_output_tokens: usize,
        /// Mean of the per-run derived accept lengths.
        accept_len_mean: f64,
    },
}

pub(crate) fn evaluate(samples: &[RunObs]) -> Evaluation {
    if samples.len() != RUNS {
        return Evaluation::Inconclusive(format!(
            "{} run(s) completed, the pinned count is {RUNS}",
            samples.len()
        ));
    }
    for (i, s) in samples.iter().enumerate() {
        if s.completion_tokens < MIN_OUTPUT_TOKENS {
            return Evaluation::Inconclusive(format!(
                "run {} emitted {} tokens, below the {MIN_OUTPUT_TOKENS}-token vacuity floor \
                 (of the {MAX_TOKENS} budget) — too short a decode to measure a floor on",
                i + 1,
                s.completion_tokens
            ));
        }
        if s.server_tps.is_none() {
            return Evaluation::Inconclusive(format!(
                "run {} reported no server decode rate (usage.\"response_token/s\") — without \
                 the server's own clock there is no defensible per-token number",
                i + 1
            ));
        }
        match s.accepted_prediction_tokens {
            None => {
                return Evaluation::Inconclusive(format!(
                    "run {} reported no usage.completion_tokens_details.\
                     accepted_prediction_tokens — this gate depends on the accept-stats \
                     instrumentation (the commit wiring real MTP accept counts into usage); \
                     serve a binary that has it",
                    i + 1
                ));
            }
            Some(0) => {
                return Evaluation::Inconclusive(format!(
                    "run {} accepted 0 draft tokens — either the serve is not speculating or \
                     the accept-stats instrumentation is not live; a serial-floor number must \
                     not be recorded as the decode floor",
                    i + 1
                ));
            }
            Some(_) => {}
        }
    }
    let mut accept_lens = Vec::with_capacity(samples.len());
    for (i, s) in samples.iter().enumerate() {
        match s.accept_len() {
            Some(l) => accept_lens.push(l),
            None => {
                return Evaluation::Inconclusive(format!(
                    "run {}: accepted ({}) >= completion_tokens ({}) — corrupt accounting, \
                     nothing derivable",
                    i + 1,
                    s.accepted_prediction_tokens.unwrap_or(0),
                    s.completion_tokens
                ));
            }
        }
    }
    let accept_len_mean = accept_lens.iter().sum::<f64>() / accept_lens.len() as f64;
    if accept_len_mean < MIN_ACCEPT_LEN {
        return Evaluation::Inconclusive(format!(
            "accept_len_mean {accept_len_mean:.2} < {MIN_ACCEPT_LEN} — speculation is not \
             engaged at gate depth, so this run measures the serial floor, not the engine"
        ));
    }
    let tps: Vec<f64> = samples.iter().filter_map(|s| s.server_tps).collect();
    // stats::median, NOT stats::percentile(_, 50): the nearest-rank p50 of
    // three samples is the maximum, and the floor must not ride the best run.
    let median = stats::median(&tps).unwrap_or(0.0);
    Evaluation::Measured {
        median_decode_tok_s: median,
        min_output_tokens: samples
            .iter()
            .map(|s| s.completion_tokens)
            .min()
            .unwrap_or(0),
        accept_len_mean,
    }
}

#[derive(Default)]
pub struct DecodeFloor {
    handle: Option<PluginHandle>,
    timeout: Duration,
    samples: Vec<RunObs>,
    started: Option<Instant>,
    probed: bool,
}

impl DecodeFloor {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    async fn one_run(&self) -> Result<http::ChatOutcome> {
        let handle = self.handle()?;
        let target = handle.target();
        // The pinned request. `reasoning_effort: "none"` is the per-request
        // thinking-off switch — deliberately in the body rather than the
        // serve config, so the gate needs no operator flags.
        let body = json!({
            "model": target.model,
            "stream": true,
            "temperature": 0.0,
            "seed": 0,
            "max_tokens": MAX_TOKENS,
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": MINHEAP_PROMPT}],
        });
        http::chat_stream(target, &body, self.timeout).await
    }

    fn table(&self) -> ResultTable {
        let mut t = ResultTable::new(
            "DECODE FLOOR",
            vec![
                Column::right("Run", 4),
                Column::right("Out tok", 8),
                Column::right("Decode tok/s (srv)", 18),
                Column::right("Accepted", 9),
                Column::right("E2E ms", 9),
            ],
        );
        for (i, s) in self.samples.iter().enumerate() {
            t.push(vec![
                Cell::new((i + 1).to_string()),
                Cell::new(s.completion_tokens.to_string()),
                Cell::styled(
                    s.server_tps
                        .map(|v| format!("{v:.1}"))
                        .unwrap_or_else(|| "—".into()),
                    CellStyle::Accent,
                ),
                Cell::new(
                    s.accepted_prediction_tokens
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "—".into()),
                ),
                Cell::new(format!("{:.0}", s.e2e_ms)),
            ]);
        }
        t
    }
}

impl Plugin for DecodeFloor {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.handle = Some(handle);
        self.started = Some(Instant::now());
        async { Ok(()) }
    }
}

impl Benchmark for DecodeFloor {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        // The generation knobs are PINS, not parameters (module docs). Only
        // the transport timeout is tunable, and it cannot move the metric.
        vec![ParamSpec::new(
            "request_timeout_s",
            "Request timeout",
            "Seconds before a single request is abandoned. Transport-side only — it cannot \
             change the measured decode rate.",
            ParamKind::Int { min: 30, max: 3600 },
            ParamValue::Int(300),
        )]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        self.timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        self.samples.clear();
        self.probed = false;
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;
        let total = RUNS as u64;

        if !self.probed {
            self.probed = true;
            http::probe(handle.target(), Duration::from_secs(10))
                .await
                .context("endpoint probe failed — check the target URL and port")?;
            return Ok(BenchmarkResult::running("probe", self.elapsed())
                .with_progress(0, total)
                .log_line(LogLine::info(format!(
                    "{} · MinHeap code prompt · max_tokens {MAX_TOKENS} · temp 0 · seed 0 · \
                     reasoning_effort none · {RUNS} pinned runs",
                    handle.target().base_url
                ))));
        }

        if self.samples.len() < RUNS {
            handle.status(format!("run {}/{RUNS}", self.samples.len() + 1));
            let outcome = self.one_run().await?;
            let obs = RunObs::from_outcome(&outcome);
            let line = LogLine::info(format!(
                "run {}/{RUNS}: {} tok · decode {} tok/s (server) · accepted {} · E2E {:.0} ms",
                self.samples.len() + 1,
                obs.completion_tokens,
                obs.server_tps
                    .map(|v| format!("{v:.1}"))
                    .unwrap_or_else(|| "—".into()),
                obs.accepted_prediction_tokens
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "—".into()),
                obs.e2e_ms,
            ));
            self.samples.push(obs);
            let done = self.samples.len() as u64;
            handle.progress(done, total);
            return Ok(BenchmarkResult::running("timed", self.elapsed())
                .with_progress(done, total)
                .with_table(self.table())
                .log_line(line));
        }

        if self.samples.iter().all(|s| s.completion_tokens == 0) {
            bail!("no run produced any output token — nothing to measure");
        }

        let mut metrics = BTreeMap::new();
        metrics.insert("runs".to_string(), self.samples.len() as f64);
        let (summary, verdict) = match evaluate(&self.samples) {
            Evaluation::Inconclusive(why) => {
                let v = Verdict::fail(format!("INCONCLUSIVE: {why}"));
                (Vec::new(), v)
            }
            Evaluation::Measured {
                median_decode_tok_s,
                min_output_tokens,
                accept_len_mean,
            } => {
                metrics.insert("server_decode_tok_s".to_string(), median_decode_tok_s);
                metrics.insert("output_tokens".to_string(), min_output_tokens as f64);
                metrics.insert("accept_len_mean".to_string(), accept_len_mean);
                let summary = vec![
                    Stat::new(
                        "Decode tok/s (server, median)",
                        format!("{median_decode_tok_s:.1}"),
                        "tok/s",
                    )
                    .with_style(CellStyle::Good),
                    Stat::new("Accept len (mean)", format!("{accept_len_mean:.2}"), ""),
                    Stat::new(
                        "Output tok (min run)",
                        format!("{min_output_tokens} / {MAX_TOKENS} cap"),
                        "",
                    ),
                ];
                let verdict = Verdict::info(format!(
                    "median decode {median_decode_tok_s:.1} tok/s over {RUNS} pinned runs \
                     (accept_len_mean {accept_len_mean:.2}) — judged against the BENCH.toml \
                     floor under --pull-request-gate"
                ));
                (summary, verdict)
            }
        };
        Ok(BenchmarkResult {
            status: RunStatus::Completed,
            ..BenchmarkResult::running("done", self.elapsed())
        }
        .with_progress(total, total)
        .with_summary(summary)
        .with_table(self.table())
        .with_metrics(metrics)
        .with_verdict(verdict))
    }
}

#[cfg(test)]
#[path = "decode_floor_tests.rs"]
mod tests;
