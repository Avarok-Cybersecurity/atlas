// SPDX-License-Identifier: AGPL-3.0-only

//! Concurrency Sweep — the latency/throughput curve.
//!
//! Port of `bench/bench_concurrency.py`: for every (ISL × concurrency) cell,
//! fire `conc` streaming requests at once and report client TTFT / TPOT / E2E
//! as p50/p90/p99 plus the aggregate output throughput of the batch. One cell
//! per `next()`, so the pane paints a row as soon as it exists and cancellation
//! lands within one cell rather than at the end of the sweep.

use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::json;

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::benchmarks::stats::{self, Percentiles, PromptMode};
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{
    BenchmarkResult, Cell, CellStyle, Column, LogLine, ResultTable, RunStatus, Stat, Verdict,
};

const SUMMARY: &str = "Latency/throughput curve across concurrency 1 → 16";
pub const METADATA: PluginMetadata = PluginMetadata::atlas(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "concurrency-sweep",
    name: "Concurrency Sweep",
    summary: SUMMARY,
    detail: "Fires N concurrent streaming requests per (input-length × concurrency) cell and \
             reports client TTFT, TPOT and end-to-end latency as p50/p90/p99, plus the batch's \
             aggregate output throughput. This is the curve the GB10 concurrency campaign is \
             measured on — C=1 is where Atlas leads, C=16 is the bar.",
    duration_hint: "~10–30 min",
    updated: "2026-07-31",
    needs_confirmation: false,
    // A latency/throughput curve is meaningful for any served model; there is
    // no threshold here tied to a checkpoint.
    intended_for: None,
    ctor: || Box::new(ConcurrencySweep::default()),
};

#[derive(Default)]
struct CellRow {
    isl: usize,
    conc: usize,
    ttft: Percentiles,
    tpot: Percentiles,
    e2e_p50: Option<f64>,
    throughput: f64,
    errors: usize,
}

#[derive(Default)]
pub struct ConcurrencySweep {
    handle: Option<PluginHandle>,
    cells: Vec<(usize, usize)>,
    cursor: usize,
    osl: usize,
    warmup: usize,
    mode: PromptMode,
    timeout: Duration,
    rows: Vec<CellRow>,
    started: Option<Instant>,
    probed: bool,
}

impl ConcurrencySweep {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    /// One request. Returns `Err` only for transport failures — a completed
    /// request with zero tokens is a data point, not an error.
    async fn one(&self, isl: usize, prefix_tag: String) -> Result<http::ChatOutcome> {
        let handle = self.handle()?;
        let target = handle.target();
        let body = json!({
            "model": target.model,
            "stream": true,
            "max_tokens": self.osl,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": stats::make_prompt(isl, self.mode, &prefix_tag)}],
        });
        http::chat_stream(target, &body, self.timeout).await
    }

    async fn run_cell(&mut self, isl: usize, conc: usize) -> Result<CellRow> {
        let handle = self.handle()?.clone();
        for w in 0..self.warmup {
            handle.check_cancelled()?;
            handle.status(format!(
                "isl {isl} · conc {conc} · warmup {}/{}",
                w + 1,
                self.warmup
            ));
            // Warm-up uses the SAME prompt the measured batch will use, which is
            // the point: it primes the prefix cache so the timed requests measure
            // the warm path rather than a one-off cold prefill.
            let _ = self.one(isl, "warm".to_string()).await;
        }
        handle.check_cancelled()?;
        handle.status(format!("isl {isl} · conc {conc} · {conc} in flight"));

        let batch_start = Instant::now();
        let futures: Vec<_> = (0..conc).map(|i| self.one(isl, format!("c{i}"))).collect();
        let outcomes = futures::future::join_all(futures).await;
        let wall = batch_start.elapsed().as_secs_f64().max(1e-6);

        let mut ttft = Vec::new();
        let mut tpot = Vec::new();
        let mut e2e = Vec::new();
        let mut tokens = 0usize;
        let mut errors = 0usize;
        for outcome in outcomes {
            match outcome {
                Ok(o) => {
                    if let Some(v) = o.ttft_ms {
                        ttft.push(v);
                    }
                    if let Some(v) = o.tpot_ms {
                        tpot.push(v);
                    }
                    e2e.push(o.e2e_ms);
                    tokens += o.completion_tokens;
                }
                Err(e) => {
                    errors += 1;
                    handle.warn(format!("isl {isl} conc {conc}: {e:#}"));
                }
            }
        }
        Ok(CellRow {
            isl,
            conc,
            ttft: Percentiles::of(&ttft),
            tpot: Percentiles::of(&tpot),
            e2e_p50: stats::percentile(&e2e, 50),
            throughput: tokens as f64 / wall,
            errors,
        })
    }

    fn table(&self) -> ResultTable {
        let mut t = ResultTable::new(
            "LATENCY / THROUGHPUT",
            vec![
                Column::right("ISL", 6),
                Column::right("Conc", 5),
                Column::right("TTFT p50", 9),
                Column::right("p90", 8),
                Column::right("p99", 8),
                Column::right("TPOT p50", 9),
                Column::right("p90", 8),
                Column::right("E2E p50", 9),
                Column::right("tok/s", 8),
                Column::right("err", 4),
            ],
        );
        for r in &self.rows {
            t.push(vec![
                Cell::new(r.isl.to_string()),
                Cell::new(r.conc.to_string()),
                Cell::styled(stats::fmt_ms(r.ttft.p50), CellStyle::Accent),
                Cell::new(stats::fmt_ms(r.ttft.p90)),
                Cell::new(stats::fmt_ms(r.ttft.p99)),
                Cell::styled(stats::fmt_ms(r.tpot.p50), CellStyle::Accent),
                Cell::new(stats::fmt_ms(r.tpot.p90)),
                Cell::new(stats::fmt_ms(r.e2e_p50)),
                Cell::styled(format!("{:.1}", r.throughput), CellStyle::Good),
                Cell::styled(
                    r.errors.to_string(),
                    if r.errors == 0 {
                        CellStyle::Dim
                    } else {
                        CellStyle::Bad
                    },
                ),
            ]);
        }
        t
    }

    fn summary(&self) -> Vec<Stat> {
        let peak = self
            .rows
            .iter()
            .max_by(|a, b| a.throughput.total_cmp(&b.throughput));
        let best_ttft = self
            .rows
            .iter()
            .filter_map(|r| r.ttft.p50)
            .fold(f64::INFINITY, f64::min);
        vec![
            Stat::new(
                "Peak throughput",
                peak.map(|r| format!("{:.1}", r.throughput))
                    .unwrap_or_else(|| "—".into()),
                "tok/s",
            )
            .with_style(CellStyle::Good),
            Stat::new(
                "at concurrency",
                peak.map(|r| r.conc.to_string())
                    .unwrap_or_else(|| "—".into()),
                "",
            ),
            Stat::new(
                "Best TTFT p50",
                if best_ttft.is_finite() {
                    format!("{best_ttft:.0}")
                } else {
                    "—".into()
                },
                "ms",
            )
            .with_style(CellStyle::Accent),
            Stat::new(
                "Cells",
                format!("{}/{}", self.rows.len(), self.cells.len()),
                "",
            ),
        ]
    }
}

impl Plugin for ConcurrencySweep {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.handle = Some(handle);
        self.started = Some(Instant::now());
        async { Ok(()) }
    }
}

impl Benchmark for ConcurrencySweep {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new(
                "concurrencies",
                "Concurrency levels",
                "How many requests are in flight at once, one sweep column each.",
                ParamKind::IntList { min: 1, max: 256 },
                ParamValue::IntList(vec![1, 2, 4, 8, 16]),
            ),
            ParamSpec::new(
                "isls",
                "Input lengths",
                "Prompt sizes in tokens. Must fit inside the server's --max-seq-len with the output.",
                ParamKind::IntList {
                    min: 16,
                    max: 131_072,
                },
                ParamValue::IntList(vec![128, 512, 1024, 2048]),
            ),
            ParamSpec::new(
                "osl",
                "Output tokens",
                "Max tokens per request.",
                ParamKind::Int { min: 1, max: 8192 },
                ParamValue::Int(128),
            ),
            ParamSpec::new(
                "warmup",
                "Warm-up requests",
                "Unmeasured requests per cell, priming the prefix cache.",
                ParamKind::Int { min: 0, max: 8 },
                ParamValue::Int(1),
            ),
            ParamSpec::new(
                "prompt_mode",
                "Prompt mode",
                "count forces the full output budget so TPOT is real; natural lets the model stop early.",
                ParamKind::Choice(&["count", "natural"]),
                ParamValue::Text("count".into()),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Request timeout",
                "Seconds before a single request is abandoned and counted as an error.",
                ParamKind::Int { min: 10, max: 3600 },
                ParamValue::Int(600),
            ),
        ]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        let concurrencies = values.int_list("concurrencies")?.to_vec();
        let isls = values.int_list("isls")?.to_vec();
        // ISL-major so the sweep walks a full concurrency curve at one prompt
        // size before changing prompt size — that is the curve people read.
        self.cells = isls
            .iter()
            .flat_map(|isl| {
                concurrencies
                    .iter()
                    .map(move |c| (*isl as usize, *c as usize))
            })
            .collect();
        self.osl = values.usize("osl")?;
        self.warmup = values.usize("warmup")?;
        self.mode = PromptMode::parse(values.text("prompt_mode")?)
            .context("prompt_mode must be count or natural")?;
        self.timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        self.cursor = 0;
        self.rows.clear();
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;

        // Step 0: reachability. A wrong port otherwise produces a whole
        // sweep of transport errors that reads like a broken server.
        if !self.probed {
            self.probed = true;
            http::probe(handle.target(), Duration::from_secs(10))
                .await
                .context("endpoint probe failed — check the target URL and port")?;
            let total = self.cells.len() as u64;
            if total == 0 {
                bail!("no cells to run — check the concurrency and input-length lists");
            }
            return Ok(BenchmarkResult::running("probe", self.elapsed())
                .with_progress(0, total)
                .log_line(LogLine::info(format!(
                    "{} · model {} · {total} cells",
                    handle.target().base_url,
                    handle.target().model
                ))));
        }

        if self.cursor >= self.cells.len() {
            let errors: usize = self.rows.iter().map(|r| r.errors).sum();
            let verdict = if errors == 0 {
                Verdict::info(format!("{} cells, no request errors", self.rows.len()))
            } else {
                // Errors do not fail the sweep — they invalidate the cells
                // they landed in, and saying so is more useful than a FAIL.
                Verdict::fail(format!(
                    "{errors} request(s) failed — affected rows are not comparable"
                ))
            };
            let mut frame = BenchmarkResult {
                status: RunStatus::Completed,
                ..BenchmarkResult::running("done", self.elapsed())
            }
            .with_progress(self.cells.len() as u64, self.cells.len() as u64)
            .with_summary(self.summary())
            .with_table(self.table())
            .with_verdict(verdict);
            // A "—" in the TPOT column is a measurement limit, not a broken
            // number, and it is worth saying which: the endpoint delivered the
            // whole reply in ONE SSE delta, so there is no inter-token interval
            // to time. Atlas batches short replies that way, so this is common
            // at small output budgets and reads like a bug if left unexplained.
            let unmeasured = self.rows.iter().filter(|r| r.tpot.p50.is_none()).count();
            if unmeasured > 0 {
                frame = frame.log_line(LogLine::warn(format!(
                    "TPOT unmeasured in {unmeasured} cell(s): the endpoint sent the whole reply \
                     in one SSE delta, so there is no inter-token interval to time. Raise the \
                     output-token budget to measure decode."
                )));
            }
            return Ok(frame);
        }

        let (isl, conc) = self.cells[self.cursor];
        let row = self.run_cell(isl, conc).await?;
        let line = LogLine::info(format!(
            "isl {isl} conc {conc}: ttft p50 {} ms · tpot p50 {} ms · {:.1} tok/s",
            stats::fmt_ms(row.ttft.p50),
            stats::fmt_ms(row.tpot.p50),
            row.throughput
        ));
        self.rows.push(row);
        self.cursor += 1;
        handle.progress(self.cursor as u64, self.cells.len() as u64);
        Ok(
            BenchmarkResult::running(format!("isl {isl} · conc {conc}"), self.elapsed())
                .with_progress(self.cursor as u64, self.cells.len() as u64)
                .with_summary(self.summary())
                .with_table(self.table())
                .log_line(line),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configured(concs: Vec<i64>, isls: Vec<i64>) -> ConcurrencySweep {
        let mut b = ConcurrencySweep::default();
        let mut v = ParamValues::defaults(&b.parameters());
        v.set("concurrencies", ParamValue::IntList(concs));
        v.set("isls", ParamValue::IntList(isls));
        b.configure(&v).unwrap();
        b
    }

    #[test]
    fn cells_are_isl_major() {
        let b = configured(vec![1, 2], vec![128, 512]);
        assert_eq!(b.cells, vec![(128, 1), (128, 2), (512, 1), (512, 2)]);
    }

    #[test]
    fn defaults_are_the_campaign_sweep() {
        let b = ConcurrencySweep::default();
        let v = ParamValues::defaults(&b.parameters());
        assert_eq!(v.int_list("concurrencies").unwrap(), &[1, 2, 4, 8, 16]);
        assert_eq!(v.usize("osl").unwrap(), 128);
    }

    #[test]
    fn an_out_of_range_parameter_is_rejected_before_the_run() {
        let mut b = ConcurrencySweep::default();
        let mut v = ParamValues::defaults(&b.parameters());
        v.set("osl", ParamValue::Int(0));
        let err = b.configure(&v).unwrap_err().to_string();
        assert!(err.contains("Output tokens"), "{err}");
    }

    #[test]
    fn reconfiguring_clears_prior_rows() {
        let mut b = configured(vec![1], vec![128]);
        b.rows.push(CellRow::default());
        let mut v = ParamValues::defaults(&b.parameters());
        v.set("isls", ParamValue::IntList(vec![256]));
        b.configure(&v).unwrap();
        assert!(b.rows.is_empty() && b.cursor == 0);
    }
}
