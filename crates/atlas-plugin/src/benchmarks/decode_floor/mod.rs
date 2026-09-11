// SPDX-License-Identifier: AGPL-3.0-only

//! Decode Floor Gate — the pinned single-user DECODE throughput floor.
//!
//! `quick-speed-bench` is deliberately a measurement tool: knobs everywhere,
//! no baseline, excused from the gate set. This driver is its opposite: every
//! generation knob is PINNED so that two runs are comparable by construction,
//! and the one number it reports — the MEDIAN server decode rate across three
//! runs — is judged against a committed BENCH.toml threshold under
//! `--pull-request-gate`. REQUIRED (`gate::coverage::REQUIRED`) since
//! 2026-08-15. The BENCH.toml floor is 22.7 (effective 22.2 after noise),
//! set 2026-09-06 from ten consecutive gate runs on dgx2 — mean 22.78, sigma
//! 0.063, range 22.7-22.9.
//!
//! It is NOT set from the 12-run 28.03 / sigma-0.05 set that justified the
//! promotion. That set ran on a hand-built serve with the default lm head;
//! this gate serves the recipe, which pins lm_head_dtype bf16, and that head
//! costs ~20%. These lines claimed otherwise until 2026-09-06.
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
//! * every run's `completion_tokens >= 750` (of the 1500 cap — the calibrated
//!   instrument's deterministic natural stop is 915, see `MIN_OUTPUT_TOKENS`);
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

use crate::hardware::Sensitivity;
use std::collections::BTreeMap;
use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::json;

use crate::benchmark::{Benchmark, BenchmarkDescriptor, ModelExpectation};
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{
    BenchmarkResult, Cell, CellStyle, Column, LogLine, ResultTable, RunStatus, Stat,
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
             >=750 of the 1500-token budget (the calibrated instrument's natural stop is a \
             deterministic 915), report the server rate, and show accept_len_mean >= 1.5 \
             derived from usage.completion_tokens_details.accepted_prediction_tokens \
             (requires the accept-stats instrumentation; a serve that is not speculating \
             cannot pass this gate's floor honestly). REQUIRED since 2026-08-15; the \
             floor in force is 22.7, from a 10-run set on the gate's own serve (2026-09-06).",
    duration_hint: "~3–6 min",
    updated: "2026-08-15",
    // The floor in BENCH.toml is measured on the dense Qwen3.8-27B NVFP4
    // checkpoint; the driver measures whatever it is pointed at, but only
    // that family has a committed baseline to judge against.
    intended_for: Some(ModelExpectation {
        families: &["qwen3.8-27b"],
        note: "The decode floor is recorded for unsloth/Qwen3.8-27B-NVFP4 (10-run \
               calibration on the gate's own serve, 2026-09-06). Other checkpoints run fine \
               but have no committed floor to be judged against — a number with no baseline \
               gates nothing.",
    }),
    // Under --pull-request-gate `min_tok_s` is auto-filled from the variant's
    // BENCH.toml floor, so a Measured run self-verdicts PASS/FAIL (see
    // `verdict_for`) — load-bearing now that this gate is REQUIRED, since
    // gate machinery accepts nothing short of a PASS run verdict.
    threshold_params: &[("min_tok_s", "server_decode_tok_s")],
    needs_confirmation: false,
    // The metric IS a rate. A box that throttled during the run reports a
    // floor miss that the code did not cause.
    sensitivity: Sensitivity::Speed,
    ctor: || Box::new(DecodeFloor::default()),
};

mod score;
pub(crate) use score::{
    Evaluation, MAX_TOKENS, MINHEAP_PROMPT, RUNS, RunObs, evaluate, verdict_for,
};

#[derive(Default)]
pub struct DecodeFloor {
    handle: Option<PluginHandle>,
    timeout: Duration,
    /// Verdict floor (tok/s); 0.0 = info verdict. Gate-filled per variant.
    min_tok_s: f64,
    samples: Vec<RunObs>,
    started: Option<Instant>,
    probed: bool,
}

impl DecodeFloor {
    fn request_body(model: &str) -> serde_json::Value {
        json!({
            "model": model,
            "stream": true,
            "temperature": 0.0,
            "seed": 0,
            "max_tokens": MAX_TOKENS,
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": MINHEAP_PROMPT}],
        })
    }

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
        let body = Self::request_body(&target.model);
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
        // the transport timeout and the verdict floor are tunable, and
        // neither can move the measured rate itself.
        vec![
            ParamSpec::new(
                "request_timeout_s",
                "Request timeout",
                "Seconds before a single request is abandoned. Transport-side only — it \
                 cannot change the measured decode rate.",
                ParamKind::Int { min: 30, max: 3600 },
                ParamValue::Int(300),
            ),
            ParamSpec::new(
                "min_tok_s",
                "Decode floor",
                "Run-verdict floor on the median server decode rate. 0 disables (a \
                 standalone run reports an info verdict); under --pull-request-gate this \
                 is auto-filled from the variant's BENCH.toml server_decode_tok_s `min` \
                 bound. Vacuous runs stay INCONCLUSIVE regardless.",
                ParamKind::Float {
                    min: 0.0,
                    max: 10_000.0,
                },
                // 0.0 is the documented OFF state, not an implicit bar (PCND).
                ParamValue::Float(0.0),
            ),
        ]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        self.timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        self.min_tok_s = values.float("min_tok_s")?;
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
        let eval = evaluate(&self.samples);
        let verdict = verdict_for(&eval, self.min_tok_s);
        let summary = match eval {
            Evaluation::Inconclusive(_) => Vec::new(),
            Evaluation::Measured {
                median_decode_tok_s,
                min_output_tokens,
                accept_len_mean,
            } => {
                metrics.insert("server_decode_tok_s".to_string(), median_decode_tok_s);
                metrics.insert("output_tokens".to_string(), min_output_tokens as f64);
                metrics.insert("accept_len_mean".to_string(), accept_len_mean);
                vec![
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
                ]
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
