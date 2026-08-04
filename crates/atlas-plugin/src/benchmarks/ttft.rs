// SPDX-License-Identifier: AGPL-3.0-only

//! Warm and Cold TTFT regression gates.
//!
//! One state machine, two modes, because the only difference between the gates
//! is whether the prefix cache is allowed to hit:
//!
//! * **Warm** — every sample at a given prompt length uses a bit-identical
//!   prompt, and each measured request is preceded by a priming request. The
//!   measurement is the cached-prefix path.
//! * **Cold** — every sample gets a unique prefix_tag, so no two requests share a
//!   prefix and every prefill is done from scratch.
//!
//! Both gate the way `benchmark-pr` Gate C does — median ≤3 %, p90 ≤5 % against
//! a **same-box** baseline, never an absolute stored number, because box and
//! run variance is ~±1 % and a cross-box baseline manufactures wins.

use std::collections::BTreeMap;
use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::json;

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::benchmarks::stats::{self, PromptMode};
use crate::benchmarks::{baseline, one_line};
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{
    BenchmarkResult, Cell, CellStyle, Column, LogLine, ResultTable, RunStatus, Stat, Verdict,
};

const WARM_SUMMARY: &str = "Cached-prefix TTFT vs the stored same-box baseline";
const COLD_SUMMARY: &str = "Uncached prefill TTFT vs the stored same-box baseline";
pub const WARM_METADATA: PluginMetadata = PluginMetadata::atlas(WARM_SUMMARY);
pub const COLD_METADATA: PluginMetadata = PluginMetadata::atlas(COLD_SUMMARY);

pub const WARM_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "ttft-warm-gate",
    name: "Warm TTFT Regression Gate",
    summary: WARM_SUMMARY,
    detail: "Measures time-to-first-token on the WARM path: each sample repeats a bit-identical \
             prompt so the prefix cache hits. Gates at median ≤3% and p90 ≤5% against a baseline \
             recorded on this box — the guard that catches an optimization silently falling back \
             to a slow path while the correctness gates stay green.",
    duration_hint: "~3–6 min",
    updated: "2026-07-31",
    needs_confirmation: false,
    // A TTFT gate compares against a baseline recorded on the SAME box and
    // model, which it stores itself — so it is meaningful for any checkpoint
    // and constrains none.
    intended_for: None,
    ctor: || Box::new(TtftGate::new(Mode::Warm)),
};

pub const COLD_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "ttft-cold-gate",
    name: "Cold TTFT Regression Gate",
    summary: COLD_SUMMARY,
    detail: "Measures time-to-first-token with the prefix cache guaranteed to MISS: every sample \
             carries a unique prefix_tag, so each request pays a full prefill. This is the prefill path \
             on its own, with the cache's contribution removed — the warm gate cannot see a \
             prefill regression that caching is hiding.",
    duration_hint: "~3–6 min",
    updated: "2026-07-31",
    needs_confirmation: false,
    // A TTFT gate compares against a baseline recorded on the SAME box and
    // model, which it stores itself — so it is meaningful for any checkpoint
    // and constrains none.
    intended_for: None,
    ctor: || Box::new(TtftGate::new(Mode::Cold)),
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Warm,
    Cold,
}

impl Mode {
    fn descriptor(self) -> &'static BenchmarkDescriptor {
        match self {
            Mode::Warm => &WARM_DESCRIPTOR,
            Mode::Cold => &COLD_DESCRIPTOR,
        }
    }
    fn metadata(self) -> &'static PluginMetadata {
        match self {
            Mode::Warm => &WARM_METADATA,
            Mode::Cold => &COLD_METADATA,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Mode::Warm => "warm",
            Mode::Cold => "cold",
        }
    }
}

struct LengthRow {
    prompt_tokens: usize,
    samples: Vec<f64>,
    cached_tokens: usize,
}

pub struct TtftGate {
    mode: Mode,
    handle: Option<PluginHandle>,
    lengths: Vec<usize>,
    repeats: usize,
    median_limit_pct: f64,
    p90_limit_pct: f64,
    update_baseline: bool,
    timeout: Duration,
    cursor: usize,
    rows: Vec<LengthRow>,
    started: Option<Instant>,
    probed: bool,
}

impl TtftGate {
    pub fn new(mode: Mode) -> Self {
        Self {
            mode,
            handle: None,
            lengths: Vec::new(),
            repeats: 0,
            median_limit_pct: 3.0,
            p90_limit_pct: 5.0,
            update_baseline: true,
            timeout: Duration::from_secs(300),
            cursor: 0,
            rows: Vec::new(),
            started: None,
            probed: false,
        }
    }

    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    /// Every measured TTFT sample, across all prompt lengths.
    fn all_samples(&self) -> Vec<f64> {
        self.rows.iter().flat_map(|r| r.samples.clone()).collect()
    }

    async fn measure(&self, tokens: usize, prefix_tag: &str) -> Result<http::ChatOutcome> {
        let handle = self.handle()?;
        let target = handle.target();
        let body = json!({
            "model": target.model,
            "stream": true,
            // Enough tokens for a first-token timestamp; the decode length is
            // irrelevant to TTFT and a longer reply only costs wall time.
            "max_tokens": 8,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": stats::make_prompt(tokens, PromptMode::Natural, prefix_tag),
            }],
        });
        http::chat_stream(target, &body, self.timeout).await
    }

    async fn run_length(&mut self, tokens: usize) -> Result<LengthRow> {
        let handle = self.handle()?.clone();
        let mut samples = Vec::with_capacity(self.repeats);
        let mut cached_tokens = 0usize;
        for i in 0..self.repeats {
            handle.check_cancelled()?;
            handle.status(format!(
                "{} · {tokens} tok · sample {}/{}",
                self.mode.label(),
                i + 1,
                self.repeats
            ));
            let prefix_tag = match self.mode {
                // Constant per length: identical prompt every time, so the
                // prefix cache can hit.
                Mode::Warm => format!("warm-{tokens}"),
                // Unique per sample: no prefix is ever shared, so every
                // request pays a full prefill.
                Mode::Cold => crate::benchmarks::unique_prefix_tag(
                    &format!("cold-{tokens}-{i}"),
                    handle.run_id(),
                ),
            };
            if self.mode == Mode::Warm {
                // Prime, then measure. The first request populates the cache;
                // only the second is a warm-path measurement.
                let _ = self.measure(tokens, &prefix_tag).await;
                handle.check_cancelled()?;
            }
            let outcome = self.measure(tokens, &prefix_tag).await?;
            cached_tokens = cached_tokens.max(outcome.cached_prompt_tokens);
            match outcome.ttft_ms {
                Some(ms) => samples.push(ms),
                None => handle.warn(format!("{tokens} tok sample {i}: no token was emitted")),
            }
        }
        Ok(LengthRow {
            prompt_tokens: tokens,
            samples,
            cached_tokens,
        })
    }

    fn table(&self) -> ResultTable {
        let mut t = ResultTable::new(
            format!("{} TTFT", self.mode.label().to_uppercase()),
            vec![
                Column::right("Prompt tok", 11),
                Column::right("n", 4),
                Column::right("median", 9),
                Column::right("p90", 9),
                Column::right("p99", 9),
                Column::right("cached tok", 11),
            ],
        );
        for r in &self.rows {
            let p = stats::Percentiles::of(&r.samples);
            t.push(vec![
                Cell::new(r.prompt_tokens.to_string()),
                Cell::new(r.samples.len().to_string()),
                Cell::styled(stats::fmt_ms(p.p50), CellStyle::Accent),
                Cell::new(stats::fmt_ms(p.p90)),
                Cell::new(stats::fmt_ms(p.p99)),
                // The cache-hit evidence: a warm gate reporting 0 cached tokens
                // measured a cold path and its verdict means nothing.
                Cell::styled(
                    r.cached_tokens.to_string(),
                    match (self.mode, r.cached_tokens) {
                        (Mode::Warm, 0) => CellStyle::Bad,
                        (Mode::Cold, 0) => CellStyle::Good,
                        (Mode::Warm, _) => CellStyle::Good,
                        (Mode::Cold, _) => CellStyle::Warn,
                    },
                ),
            ]);
        }
        t
    }

    /// Compare against the stored baseline and decide. `None` baseline is
    /// `Info`, never a green PASS — a gate with nothing to compare against has
    /// not passed anything.
    fn verdict(&self, median: Option<f64>, p90: Option<f64>) -> (Verdict, Vec<Stat>) {
        let store = match self.handle() {
            Ok(h) => h.artifacts().clone(),
            Err(_) => return (Verdict::info("no handle"), Vec::new()),
        };
        let id = self.mode.descriptor().id;
        let stored = baseline::load(&store, id);
        let mut summary = vec![
            Stat::new("Median TTFT", stats::fmt_ms(median), "ms").with_style(CellStyle::Accent),
            Stat::new("p90 TTFT", stats::fmt_ms(p90), "ms"),
        ];
        let Some(base) = stored else {
            summary.push(Stat::new("Baseline", "none", "").with_style(CellStyle::Dim));
            return (
                Verdict::info("no baseline on this box yet — this run is recorded as the baseline"),
                summary,
            );
        };
        let target_now = self
            .handle()
            .map(|h| h.target().base_url.clone())
            .unwrap_or_default();
        let model_now = self
            .handle()
            .map(|h| h.target().model.clone())
            .unwrap_or_default();
        if base.target != target_now || base.model != model_now {
            summary.push(Stat::new("Baseline", "other target", "").with_style(CellStyle::Warn));
            return (
                Verdict::info(format!(
                    "baseline was recorded against {} / {} — not comparable, reporting only",
                    base.target, base.model
                )),
                summary,
            );
        }
        let dm = stats::pct_delta(median, base.get("median_ms"));
        let dp = stats::pct_delta(p90, base.get("p90_ms"));
        summary.push(
            Stat::new(
                "vs baseline",
                dm.map(|d| format!("{d:+.1}")).unwrap_or_else(|| "—".into()),
                format!("% median · {}", base.age_text()),
            )
            .with_style(match dm {
                Some(d) if d > self.median_limit_pct => CellStyle::Bad,
                Some(d) if d < 0.0 => CellStyle::Good,
                _ => CellStyle::Neutral,
            }),
        );
        let median_bad = dm.is_some_and(|d| d > self.median_limit_pct);
        let p90_bad = dp.is_some_and(|d| d > self.p90_limit_pct);
        let detail = format!(
            "median {} (limit +{:.1}%) · p90 {} (limit +{:.1}%)",
            dm.map(|d| format!("{d:+.1}%"))
                .unwrap_or_else(|| "—".into()),
            self.median_limit_pct,
            dp.map(|d| format!("{d:+.1}%"))
                .unwrap_or_else(|| "—".into()),
            self.p90_limit_pct,
        );
        if median_bad || p90_bad {
            (Verdict::fail(format!("REGRESSED — {detail}")), summary)
        } else {
            (Verdict::pass(detail), summary)
        }
    }
}

impl Plugin for TtftGate {
    fn metadata(&self) -> &'static PluginMetadata {
        self.mode.metadata()
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.handle = Some(handle);
        self.started = Some(Instant::now());
        async { Ok(()) }
    }
}

impl Benchmark for TtftGate {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        self.mode.descriptor()
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new(
                "prompt_lengths",
                "Prompt lengths",
                "Prompt sizes in tokens; one table row each.",
                ParamKind::IntList {
                    min: 16,
                    max: 131_072,
                },
                ParamValue::IntList(vec![256, 1024, 4096]),
            ),
            ParamSpec::new(
                "repeats",
                "Samples per length",
                "More samples narrow the median; each costs one request (two in warm mode).",
                ParamKind::Int { min: 1, max: 200 },
                ParamValue::Int(12),
            ),
            ParamSpec::new(
                "median_limit_pct",
                "Median limit",
                "Percent the median may rise over the baseline before this gate fails.",
                ParamKind::Float {
                    min: 0.0,
                    max: 100.0,
                },
                ParamValue::Float(3.0),
            ),
            ParamSpec::new(
                "p90_limit_pct",
                "p90 limit",
                "Percent p90 may rise over the baseline before this gate fails.",
                ParamKind::Float {
                    min: 0.0,
                    max: 100.0,
                },
                ParamValue::Float(5.0),
            ),
            ParamSpec::new(
                "update_baseline",
                "Record as baseline",
                "Store this run's numbers as the new baseline. Turn off to compare without moving the bar.",
                ParamKind::Bool,
                ParamValue::Bool(true),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Request timeout",
                "Seconds before a single request is abandoned.",
                ParamKind::Int { min: 10, max: 3600 },
                ParamValue::Int(300),
            ),
        ]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        self.lengths = values
            .int_list("prompt_lengths")?
            .iter()
            .map(|v| *v as usize)
            .collect();
        self.repeats = values.usize("repeats")?;
        self.median_limit_pct = values.float("median_limit_pct")?;
        self.p90_limit_pct = values.float("p90_limit_pct")?;
        self.update_baseline = values.bool("update_baseline")?;
        self.timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        self.cursor = 0;
        self.rows.clear();
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;
        let total = self.lengths.len() as u64;

        if !self.probed {
            self.probed = true;
            http::probe(handle.target(), Duration::from_secs(10))
                .await
                .context("endpoint probe failed — check the target URL and port")?;
            if total == 0 {
                bail!("no prompt lengths to measure");
            }
            return Ok(BenchmarkResult::running("probe", self.elapsed())
                .with_progress(0, total)
                .log_line(LogLine::info(format!(
                    "{} mode · {} · {} length(s) × {} samples",
                    self.mode.label(),
                    handle.target().base_url,
                    total,
                    self.repeats
                ))));
        }

        if self.cursor >= self.lengths.len() {
            let samples = self.all_samples();
            if samples.is_empty() {
                bail!("no TTFT samples were collected — the endpoint emitted no tokens");
            }
            let p = stats::Percentiles::of(&samples);
            let (verdict, summary) = self.verdict(p.p50, p.p90);
            if self.update_baseline {
                let mut metrics = BTreeMap::new();
                if let Some(v) = p.p50 {
                    metrics.insert("median_ms".to_string(), v);
                }
                if let Some(v) = p.p90 {
                    metrics.insert("p90_ms".to_string(), v);
                }
                let target = handle.target();
                baseline::save(
                    handle.artifacts(),
                    self.mode.descriptor().id,
                    &target.base_url,
                    &target.model,
                    metrics,
                )
                .context("recording baseline")?;
            }
            return Ok(BenchmarkResult {
                status: RunStatus::Completed,
                ..BenchmarkResult::running("done", self.elapsed())
            }
            .with_progress(total, total)
            .with_summary(summary)
            .with_table(self.table())
            .with_verdict(verdict));
        }

        let tokens = self.lengths[self.cursor];
        let row = self.run_length(tokens).await?;
        let p = stats::Percentiles::of(&row.samples);
        let line = LogLine::info(one_line(format!(
            "{} {tokens} tok: median {} ms · p90 {} ms · cached {} tok",
            self.mode.label(),
            stats::fmt_ms(p.p50),
            stats::fmt_ms(p.p90),
            row.cached_tokens
        )));
        self.rows.push(row);
        self.cursor += 1;
        handle.progress(self.cursor as u64, total);
        Ok(BenchmarkResult::running(
            format!("{} · {tokens} tok", self.mode.label()),
            self.elapsed(),
        )
        .with_progress(self.cursor as u64, total)
        .with_table(self.table())
        .log_line(line))
    }
}

#[cfg(test)]
#[path = "ttft_tests.rs"]
mod tests;
