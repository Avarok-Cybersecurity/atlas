// SPDX-License-Identifier: AGPL-3.0-only

//! Quick Speed Bench — the fast single-user speed probe.
//!
//! Port of `bench/quick_bench.py` (now deleted): warmup + N timed streaming
//! runs of one prompt against the served endpoint, reporting the server's
//! decode rate, TTFT and end-to-end latency. A measurement tool, not a gate:
//! it stores no baseline, reads no threshold, and is deliberately excused from
//! the PR gate set (`gate::coverage::NOT_REQUIRED`).
//!
//! # Which tok/s to quote
//!
//! Two rates are reported and they answer different questions:
//!
//! * **Decode tok/s (server)** — the headline. `usage."response_token/s"`,
//!   computed by the server as `(completion_tokens − 1) / decode_time`. It
//!   excludes prefill, client buffering and the network, so it is the only
//!   number here that can back a per-token throughput claim.
//! * **E2E tok/s (client)** — `completion_tokens / wall`, INCLUDING prefill
//!   and transport. Always lower; useful for "how long until my answer", never
//!   for kernel comparisons.
//!
//! The Python original also printed a client-side TPOT from first-delta to
//! last-delta timestamps. That number was a buffering artifact: buffered reads
//! plus MTP burst emission compress the client's inter-token spacing, and the
//! script reported TPOTs (9.9 ms ⇒ 101 tok/s) that are bandwidth-impossible on
//! this hardware. It is NOT reproduced here — TPOT is derived from the
//! server's own decode rate (`1000 / server tok/s`) or omitted when the server
//! does not report one. No client-clock TPOT is ever emitted.
//!
//! # TTFT semantics (session-scoped SSM snapshots, 2026-03-27)
//!
//! Atlas uses Marconi prefix caching with per-session SSM snapshot isolation.
//! SSM snapshots are tagged with a session hash (hash of the first 64 prompt
//! tokens); cross-session snapshot restore is rejected and the SSM state is
//! recomputed. Therefore, with the same prompt every run (this benchmark's
//! behaviour):
//!
//! * Run 1 (cold): full prefill, no cache. TTFT = prefill time.
//! * Run 2+ (same session/prompt): prefix-cache HIT + SSM snapshot HIT.
//!   TTFT ≈ 50–100 ms — near-instant, just LM head + first token. This is why
//!   repeated runs look implausibly fast: they are measuring the warm
//!   intra-session path, which the warmup run deliberately primes.
//! * A NEW session (different first user message) hits the prefix cache for a
//!   shared system prompt but MISSES the SSM snapshot (different session
//!   hash): TTFT = SSM recompute for the shared prefix — slower than an
//!   intra-session hit, faster than fully cold.
//!
//! To measure cross-session or cold TTFT properly, use the dedicated
//! `ttft-warm-gate` / `ttft-cold-gate` benchmarks, which control the prefix
//! deliberately. This probe's TTFT is the warm intra-session figure.

use crate::hardware::Sensitivity;
use std::collections::BTreeMap;
use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::json;

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::benchmarks::stats::{self, PromptMode};
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{
    BenchmarkResult, Cell, CellStyle, Column, LogLine, ResultTable, RunStatus, Stat, Verdict,
};

const SUMMARY: &str = "Fast single-user speed probe: warmup + N timed runs, \
                       server decode tok/s, TTFT and E2E";
pub const METADATA: PluginMetadata = PluginMetadata::atlas(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "quick-speed-bench",
    name: "Quick Speed Bench",
    summary: SUMMARY,
    detail: "One warmup plus N timed streaming runs of a single prompt. The headline is the \
             SERVER-side decode-only rate ((completion_tokens − 1) / decode_time, from the \
             response usage); the client E2E rate is shown separately and includes prefill and \
             transport. TPOT is derived from the server's decode rate — the client-clock TPOT \
             the old Python probe printed was a buffering artifact and is not reproduced. \
             Because the warmup primes the Marconi prefix cache and per-session SSM snapshot, \
             the timed TTFT is the warm INTRA-SESSION figure (~50–100 ms) — use the TTFT gates \
             for controlled cold/warm measurements. A measurement tool: it gates nothing and \
             stores no baseline.",
    duration_hint: "~1–3 min",
    updated: "2026-08-15",
    // A speed probe measures whatever it is pointed at; nothing here compares
    // against a checkpoint-specific number.
    intended_for: None,
    threshold_params: &[],
    needs_confirmation: false,
    // Decode tok/s, TTFT and E2E — every headline it prints is a speed number.
    sensitivity: Sensitivity::Speed,
    ctor: || Box::new(QuickSpeed::default()),
};

/// Committed prompt fixtures, keyed by the ISL they were cut for. Real text
/// rather than synthesized filler — at large ISLs a natural document exercises
/// prefill more honestly than a looped filler corpus. Any other ISL falls back
/// to [`stats::make_prompt`]'s synthesized filler (the same corpus and rule the
/// Python used).
const FIXTURES: &[(usize, &str)] = &[
    (
        128,
        include_str!("../../../../tests/fixtures/bench_prompt_128.txt"),
    ),
    (
        512,
        include_str!("../../../../tests/fixtures/bench_prompt_512.txt"),
    ),
    (
        1024,
        include_str!("../../../../tests/fixtures/bench_prompt_1024.txt"),
    ),
    (
        4096,
        include_str!("../../../../tests/fixtures/bench_prompt_4096.txt"),
    ),
];

/// The forcing suffix every prompt ends with, fixture or synthesized — without
/// it the model answers in a handful of tokens and the run measures scheduling
/// overhead rather than decode.
const COUNT_SUFFIX: &str = "Count from 1 upward, one number per line, until told to stop.";

/// The prompt for a requested input length: a committed fixture when one
/// exists for exactly that ISL, synthesized filler otherwise.
pub(crate) fn prompt_for(isl: usize) -> String {
    match FIXTURES.iter().find(|(n, _)| *n == isl) {
        Some((_, text)) => format!("{}\n{COUNT_SUFFIX}", text.trim()),
        None => stats::make_prompt(isl, PromptMode::Count, ""),
    }
}

/// One timed run, reduced to the numbers this benchmark aggregates.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RunSample {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub e2e_ms: f64,
    pub server_ttft_ms: Option<f64>,
    pub server_tps: Option<f64>,
}

impl RunSample {
    pub(crate) fn from_outcome(o: &http::ChatOutcome) -> Self {
        Self {
            prompt_tokens: o.prompt_tokens,
            completion_tokens: o.completion_tokens,
            e2e_ms: o.e2e_ms,
            server_ttft_ms: o.server_ttft_ms,
            server_tps: o.server_tps,
        }
    }

    /// Client-observed rate INCLUDING prefill and transport — never the
    /// headline, see the module docs.
    pub(crate) fn client_e2e_tok_s(&self) -> Option<f64> {
        (self.completion_tokens > 0 && self.e2e_ms.is_finite() && self.e2e_ms > 0.0)
            .then(|| self.completion_tokens as f64 / (self.e2e_ms / 1000.0))
    }

    /// Per-token decode latency derived from the SERVER's decode rate — the
    /// only TPOT this benchmark reports (module docs, "Which tok/s to quote").
    pub(crate) fn server_tpot_ms(&self) -> Option<f64> {
        self.server_tps
            .filter(|t| t.is_finite() && *t > 0.0)
            .map(|t| 1000.0 / t)
    }
}

fn mean(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let v: Vec<f64> = values.into_iter().filter(|x| x.is_finite()).collect();
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

/// Cross-run averages. Each field averages only the runs that reported the
/// underlying value — a run without server timings never contributes a zero.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Averages {
    pub prompt_tokens: Option<f64>,
    pub output_tokens: Option<f64>,
    pub server_decode_tok_s: Option<f64>,
    pub client_e2e_tok_s: Option<f64>,
    pub server_ttft_ms: Option<f64>,
    pub server_tpot_ms: Option<f64>,
    pub e2e_ms: Option<f64>,
}

impl Averages {
    pub(crate) fn of(samples: &[RunSample]) -> Self {
        Self {
            prompt_tokens: mean(samples.iter().map(|s| s.prompt_tokens as f64)),
            output_tokens: mean(samples.iter().map(|s| s.completion_tokens as f64)),
            server_decode_tok_s: mean(
                samples
                    .iter()
                    .filter_map(|s| s.server_tps)
                    .filter(|v| v.is_finite() && *v > 0.0),
            ),
            client_e2e_tok_s: mean(samples.iter().filter_map(RunSample::client_e2e_tok_s)),
            server_ttft_ms: mean(
                samples
                    .iter()
                    .filter_map(|s| s.server_ttft_ms)
                    .filter(|v| v.is_finite() && *v >= 0.0),
            ),
            server_tpot_ms: mean(samples.iter().filter_map(RunSample::server_tpot_ms)),
            e2e_ms: mean(
                samples
                    .iter()
                    .map(|s| s.e2e_ms)
                    .filter(|v| v.is_finite() && *v >= 0.0),
            ),
        }
    }
}

fn fmt(v: Option<f64>, digits: usize) -> String {
    v.map(|x| format!("{x:.digits$}"))
        .unwrap_or_else(|| "—".into())
}

#[derive(Default)]
pub struct QuickSpeed {
    handle: Option<PluginHandle>,
    isl: usize,
    osl: usize,
    runs: usize,
    warmup: usize,
    timeout: Duration,
    warmups_done: usize,
    samples: Vec<RunSample>,
    started: Option<Instant>,
    probed: bool,
}

impl QuickSpeed {
    fn request_body(&self, model: &str) -> serde_json::Value {
        json!({
            "model": model,
            "stream": true,
            "max_tokens": self.osl,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt_for(self.isl)}],
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
        let body = self.request_body(&target.model);
        http::chat_stream(target, &body, self.timeout).await
    }

    fn table(&self) -> ResultTable {
        let mut t = ResultTable::new(
            "QUICK SPEED",
            vec![
                Column::right("Run", 4),
                Column::right("Out tok", 8),
                Column::right("Decode tok/s (srv)", 18),
                Column::right("TTFT ms (srv)", 13),
                Column::right("E2E ms", 9),
            ],
        );
        for (i, s) in self.samples.iter().enumerate() {
            t.push(vec![
                Cell::new((i + 1).to_string()),
                Cell::new(s.completion_tokens.to_string()),
                Cell::styled(fmt(s.server_tps, 1), CellStyle::Accent),
                Cell::new(fmt(s.server_ttft_ms, 0)),
                Cell::new(format!("{:.0}", s.e2e_ms)),
            ]);
        }
        t
    }

    fn summary(&self, avg: &Averages) -> Vec<Stat> {
        vec![
            // The headline, and it says so in its label — the whole defect the
            // Python had was that its two tok/s lines were quotable
            // interchangeably.
            Stat::new(
                "Decode tok/s (server)",
                fmt(avg.server_decode_tok_s, 1),
                "tok/s",
            )
            .with_style(CellStyle::Good),
            Stat::new(
                "E2E tok/s (client, incl. prefill)",
                fmt(avg.client_e2e_tok_s, 1),
                "tok/s",
            )
            .with_style(CellStyle::Dim),
            Stat::new("TTFT (server prefill)", fmt(avg.server_ttft_ms, 1), "ms")
                .with_style(CellStyle::Accent),
            Stat::new("TPOT (server decode)", fmt(avg.server_tpot_ms, 2), "ms"),
            Stat::new(
                "Output tok",
                format!("{} / {} cap", fmt(avg.output_tokens, 0), self.osl),
                "",
            ),
        ]
    }

    fn metrics(&self, avg: &Averages) -> BTreeMap<String, f64> {
        let mut m = BTreeMap::new();
        m.insert("runs".to_string(), self.samples.len() as f64);
        let pairs = [
            ("prompt_tokens", avg.prompt_tokens),
            ("output_tokens", avg.output_tokens),
            ("server_decode_tok_s", avg.server_decode_tok_s),
            ("client_e2e_tok_s", avg.client_e2e_tok_s),
            ("server_ttft_ms", avg.server_ttft_ms),
            ("server_tpot_ms", avg.server_tpot_ms),
            ("e2e_ms", avg.e2e_ms),
        ];
        for (k, v) in pairs {
            if let Some(v) = v {
                m.insert(k.to_string(), v);
            }
        }
        m
    }
}

impl Plugin for QuickSpeed {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.handle = Some(handle);
        self.started = Some(Instant::now());
        async { Ok(()) }
    }
}

impl Benchmark for QuickSpeed {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new(
                "isl",
                "Input length",
                "Prompt size in tokens. 128 / 512 / 1024 / 4096 load a committed text fixture; \
                 any other size synthesizes a filler prompt.",
                ParamKind::Int {
                    min: 16,
                    max: 131_072,
                },
                // The Python default: a short ~60-token prompt.
                ParamValue::Int(60),
            ),
            ParamSpec::new(
                "osl",
                "Output tokens",
                "A CEILING, not a target — the model may hit EOS earlier, and a 49-token reply \
                 against a 128 cap is the model stopping, not a defect. The counting suffix \
                 pushes toward the cap but does not guarantee it.",
                ParamKind::Int { min: 2, max: 8192 },
                ParamValue::Int(128),
            ),
            ParamSpec::new(
                "runs",
                "Timed runs",
                "Measured requests; the report averages across them.",
                ParamKind::Int { min: 1, max: 100 },
                ParamValue::Int(5),
            ),
            ParamSpec::new(
                "warmup",
                "Warm-up runs",
                "Unmeasured priming requests with the SAME prompt. They populate the Marconi \
                 prefix cache and the per-session SSM snapshot, so the timed TTFT is the warm \
                 intra-session figure — use the TTFT gates to measure cold prefill.",
                ParamKind::Int { min: 0, max: 10 },
                ParamValue::Int(1),
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
        self.isl = values.usize("isl")?;
        self.osl = values.usize("osl")?;
        self.runs = values.usize("runs")?;
        self.warmup = values.usize("warmup")?;
        self.timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        self.warmups_done = 0;
        self.samples.clear();
        self.probed = false;
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;
        let total = (self.warmup + self.runs) as u64;
        let done = (self.warmups_done + self.samples.len()) as u64;

        if !self.probed {
            self.probed = true;
            http::probe(handle.target(), Duration::from_secs(10))
                .await
                .context("endpoint probe failed — check the target URL and port")?;
            return Ok(BenchmarkResult::running("probe", self.elapsed())
                .with_progress(0, total)
                .log_line(LogLine::info(format!(
                    "{} · isl {} · osl {} (cap) · {} warmup + {} timed",
                    handle.target().base_url,
                    self.isl,
                    self.osl,
                    self.warmup,
                    self.runs
                ))));
        }

        if self.warmups_done < self.warmup {
            handle.status(format!("warmup {}/{}", self.warmups_done + 1, self.warmup));
            let outcome = self.one_run().await?;
            self.warmups_done += 1;
            return Ok(BenchmarkResult::running("warmup", self.elapsed())
                .with_progress(done + 1, total)
                .log_line(LogLine::info(format!(
                    "warmup {}/{}: {} prompt tokens (prefix cache + SSM snapshot primed)",
                    self.warmups_done, self.warmup, outcome.prompt_tokens
                ))));
        }

        if self.samples.len() < self.runs {
            handle.status(format!("run {}/{}", self.samples.len() + 1, self.runs));
            let outcome = self.one_run().await?;
            let sample = RunSample::from_outcome(&outcome);
            let line = LogLine::info(format!(
                "run {}/{}: {} tok · decode {} tok/s (server) · TTFT {} ms · E2E {:.0} ms",
                self.samples.len() + 1,
                self.runs,
                sample.completion_tokens,
                fmt(sample.server_tps, 1),
                fmt(sample.server_ttft_ms, 0),
                sample.e2e_ms,
            ));
            self.samples.push(sample);
            handle.progress(done + 1, total);
            return Ok(BenchmarkResult::running("timed", self.elapsed())
                .with_progress(done + 1, total)
                .with_table(self.table())
                .log_line(line));
        }

        if self.samples.iter().all(|s| s.completion_tokens == 0) {
            bail!("no run produced any output token — nothing to measure");
        }
        let avg = Averages::of(&self.samples);
        let verdict = Verdict::info(format!(
            "measured {} run(s): decode {} tok/s (server), TTFT {} ms — a measurement, \
             not a gate",
            self.samples.len(),
            fmt(avg.server_decode_tok_s, 1),
            fmt(avg.server_ttft_ms, 1),
        ));
        Ok(BenchmarkResult {
            status: RunStatus::Completed,
            ..BenchmarkResult::running("done", self.elapsed())
        }
        .with_progress(total, total)
        .with_summary(self.summary(&avg))
        .with_table(self.table())
        .with_metrics(self.metrics(&avg))
        .with_verdict(verdict))
    }
}

#[cfg(test)]
#[path = "quick_speed_tests.rs"]
mod tests;
