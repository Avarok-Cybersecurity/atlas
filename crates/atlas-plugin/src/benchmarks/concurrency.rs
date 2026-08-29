// SPDX-License-Identifier: AGPL-3.0-only

//! Concurrency Sweep — the latency/throughput curve.
//!
//! Port of `bench/bench_concurrency.py`: for every (ISL × concurrency) cell,
//! fire `conc` streaming requests at once and report client TTFT / TPOT / E2E
//! as p50/p90/p99 plus the aggregate output throughput of the batch. One cell
//! per `next()`, so the pane paints a row as soon as it exists and cancellation
//! lands within one cell rather than at the end of the sweep.
//!
//! ★ A cell's tok/s is only as real as the tokens behind it. The 2026-08-15
//! re-scope: the counting prompt produced C=1 cells of 49-token bursts and
//! C≥4 cells with 0–1 output tokens (E2E==TTFT, TPOT 0.0, aggregate
//! DECREASING with C) on a serve where a natural code-generation prompt
//! completed the full 800-token budget at every C (C=1: 31.9 → C=16: 170.1
//! aggregate tok/s). The instrument was broken, not the server. Hence: the
//! natural code-generation fixture is the default, every request's delivered
//! evidence is recorded, and cells below the vacuity floor are flagged
//! non-comparable instead of silently reported.

use crate::hardware::Sensitivity;
use std::collections::BTreeMap;
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
    BenchmarkResult, Cell, CellStyle, Column, LogLine, ResultTable, RunStatus, Stat,
};

const SUMMARY: &str = "Latency/throughput curve across concurrency 1 → 128";
pub const METADATA: PluginMetadata = PluginMetadata::atlas(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "concurrency-sweep",
    name: "Concurrency Sweep",
    summary: SUMMARY,
    detail: "Fires N concurrent streaming requests per (input-length × concurrency) cell and \
             reports client TTFT, TPOT and end-to-end latency as p50/p90/p99, plus the batch's \
             aggregate output throughput. This is the curve the GB10 concurrency campaign is \
             measured on — C=1 is where Atlas leads, C=32 is where time-to-answer starts \
             inverting in Atlas's favour, and C=128 is the widest rung the published ladder \
             quotes. Requests pin temperature 0.0 / \
             seed 0 and send reasoning_effort \"none\" so the ladder measures decode, not \
             thinking. A cell where any request delivers under 80% of the output budget is \
             flagged vacuous and its tok/s marked non-comparable. REQUIRED gate since \
             2026-08-15: under --pull-request-gate the run serves the calibrated instrument \
             (C=1..128, isl 512, osl 320 via the variant's param_overrides) and \
             self-verdicts against gate-filled per-rung floors; a sweep with any vacuous \
             cell or request error never passes, whatever the floors say.",
    duration_hint: "~25–90 min",
    updated: "2026-08-29",
    needs_confirmation: false,
    // A latency/throughput curve is meaningful for any served model; there is
    // no threshold here tied to a checkpoint.
    intended_for: None,
    // Under --pull-request-gate each floor is auto-filled from the selected
    // variant's BENCH.toml `min` bound, so a run that clears its committed
    // ladder self-verdicts PASS — which gate machinery requires now that this
    // gate is REQUIRED (`gate::coverage::REQUIRED`). The gated ladder's SHAPE
    // (C=1..128, isl 512, osl 320) arrives via the same entry's
    // `[benchmarks.param_overrides]`, not from these schema defaults.
    threshold_params: GATE_THRESHOLD_PARAMS,
    // Latency and throughput at every rung; a thermal event mid-sweep moves
    // the later rungs and not the earlier ones, which reads as a shape change.
    sensitivity: Sensitivity::Speed,
    ctor: || Box::new(ConcurrencySweep::default()),
};

/// The same ladder, served with the DFlash2 drafter armed.
///
/// A SECOND gate id rather than a second BENCH.toml variant, deliberately, and
/// for the same reason `bfcl-subset-echolp` exists: `check_record`'s
/// `record_is_required_subject` refuses a non-default checkpoint for the
/// required verdict, so a variant can never satisfy a mandatory gate. The two
/// entries share this driver, this instrument and these floor params; what
/// differs is the recipe each one's `[[benchmarks]]` block names, and
/// therefore whether the served engine speculates.
///
/// It exists because nothing else gates the speculative path. Every other
/// required gate serves a no-drafter recipe, which is how a DFlash2 change can
/// only be judged by hand — the 2026-08 C4 hunt spent days establishing "not
/// DFlash2" from source reading, because no committed record could answer it.
///
/// Its numbers are NOT comparable to `concurrency-sweep`'s and must never be
/// crossed: speculation moves the aggregate at low C by more than any
/// regression this gate is meant to catch, so each carries its own bars.
pub const DFLASH2_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "concurrency-sweep-dflash2",
    name: "Concurrency Sweep (DFlash2)",
    summary: DFLASH2_SUMMARY,
    detail: "The concurrency ladder with the DFlash2 block-diffusion drafter armed \
             (`--dflash --draft-model incoai/Qwen3.8-27B-DFlash2 --dflash-gamma 8`), pinned by \
             the variant's serve_overrides. Same fixture, same rungs and same vacuity rule as \
             `concurrency-sweep`; the only difference is that the served engine speculates, \
             which is exactly the path no other required gate exercises. Expect the two curves \
             to converge at the wide rungs: DFlash2's verify batches are bounded, so above the \
             point where speculation self-limits this measures the base engine and says so \
             rather than pretending otherwise.",
    duration_hint: "~25–90 min",
    updated: "2026-08-29",
    needs_confirmation: false,
    // The drafter is checkpoint-specific — a drafter trained on other weights
    // degrades to near-zero acceptance rather than failing loudly, so this
    // names the target it was trained against.
    intended_for: Some(crate::benchmark::ModelExpectation {
        families: &["qwen3.8-27b"],
        note: "DFlash2 drafters are trained against one target's hidden states \
               (incoai/Qwen3.8-27B-DFlash2 consumes target layers [5,19,33,47,61] of \
               Qwen3.8-27B). Pointing this gate at another checkpoint measures a mismatched \
               drafter, which is slow rather than wrong and therefore easy to misread.",
    }),
    threshold_params: GATE_THRESHOLD_PARAMS,
    sensitivity: Sensitivity::Speed,
    ctor: || Box::new(ConcurrencySweep::default()),
};

const DFLASH2_SUMMARY: &str =
    "Latency/throughput curve across concurrency 1 → 128, DFlash2 armed";

/// The gated ladder, declared ONCE: `(C, floor param, metric key, label)`.
///
/// Three things are derived from this and nothing else — the descriptor's
/// `threshold_params` pairs, the floor `ParamSpec`s, and `configure`'s
/// `Floors::per_c`. Before this table those were three hand-maintained lists
/// of the same rungs; a rung present in one and missing from another is
/// silent, because `apply_threshold_params` skips a metric with no bound and
/// `sweep_verdict` only judges the rungs `per_c` names.
///
/// C=2 through C=128 joined on 2026-08-29 so the certification covers the same
/// span as the published ladder. The rungs are declarable, not measured: a
/// rung with no `[benchmarks.metrics.c<N>_aggregate_tok_s]` block in the
/// variant's BENCH.toml keeps the 0.0 default and gates nothing, which is how
/// a new rung is recorded before it is bounded.
const RUNGS: [(usize, &str, &str, &str); 8] = [
    (1, "min_c1", "c1_aggregate_tok_s", "C=1 aggregate floor"),
    (2, "min_c2", "c2_aggregate_tok_s", "C=2 aggregate floor"),
    (4, "min_c4", "c4_aggregate_tok_s", "C=4 aggregate floor"),
    (8, "min_c8", "c8_aggregate_tok_s", "C=8 aggregate floor"),
    (16, "min_c16", "c16_aggregate_tok_s", "C=16 aggregate floor"),
    (32, "min_c32", "c32_aggregate_tok_s", "C=32 aggregate floor"),
    (64, "min_c64", "c64_aggregate_tok_s", "C=64 aggregate floor"),
    (128, "min_c128", "c128_aggregate_tok_s", "C=128 aggregate floor"),
];

/// The peak floor, which is not a rung: it bounds `peak_aggregate_tok_s`,
/// whichever C produced it.
const PEAK_FLOOR: (&str, &str, &str) = (
    "min_peak",
    "peak_aggregate_tok_s",
    "Peak aggregate floor",
);

/// The baseline-coupled verdict params: each is paired to the metric the
/// BENCH.toml floor is written on (`min` bounds; see `bench_resolve::
/// apply_threshold_params`). Float, default 0.0 = non-gating, so a standalone
/// run keeps its info verdict. Derived from [`RUNGS`] in a const fn so the
/// pairing cannot drift from the ParamSpecs that must back it — a
/// `threshold_params` entry with no matching spec is a hard error in
/// `bench_resolve::apply_threshold_params`.
const GATE_THRESHOLD_PARAMS: &[(&str, &str)] = &gate_threshold_params();

const fn gate_threshold_params() -> [(&'static str, &'static str); RUNGS.len() + 1] {
    let mut out = [("", ""); RUNGS.len() + 1];
    let mut i = 0;
    while i < RUNGS.len() {
        out[i] = (RUNGS[i].1, RUNGS[i].2);
        i += 1;
    }
    out[RUNGS.len()] = (PEAK_FLOOR.0, PEAK_FLOOR.1);
    out
}

/// Natural code-generation fixture (own constant — no cross-driver imports).
/// Appended after the ISL padding so the filler reads as context and this
/// reads as the ask. A code-generation task of this shape reliably fills a
/// several-hundred-token output budget on a thinking-off serve, which is what
/// makes the cell's TPOT and aggregate tok/s measurements of decode at all.
const CODE_TASK: &str = "Ignore the reference text above. Task: write a complete, \
    production-quality MinHeap class in Python with insert, peek_min, extract_min, \
    decrease_key and heapify methods — full docstrings, input validation and a worked \
    usage example — followed by a unit-test class covering every method, including \
    empty-heap, duplicate-key and single-element edge cases. Write every method and \
    every test out in full; do not summarize or elide any code.";

/// Fraction of the output budget below which one request's completion makes
/// its whole cell vacuous. 80%: a natural stop a few tokens early is fine; a
/// 49-token burst against a 512-token budget is not a throughput measurement.
const VACUITY_FLOOR: f64 = 0.8;
// `make_prompt` puts the distinguishing tag at byte zero of message content;
// a wrong tag can share only the chat-template prefix, about 12 tokens against
// the minimum 128-token ISL. An exact warmed prompt is block-aligned near the
// full length, so 80% separates substantial reuse from incidental template KV.
const WARM_CACHE_FLOOR: f64 = 0.8;

/// What one completed request actually delivered — the evidence a cell's
/// tok/s stands on. `http::ChatOutcome` already parses all of this from the
/// stream's `usage`; the old sweep summed `completion_tokens` and discarded
/// the rest.
#[derive(Clone, Debug, Default)]
struct RequestEvidence {
    completion_tokens: usize,
    prompt_tokens: usize,
    cached_prompt_tokens: usize,
    finish_reason: Option<String>,
    server_ttft_ms: Option<f64>,
    server_tps: Option<f64>,
}

/// A cell is vacuous when ANY of its requests finished materially short of
/// the output budget — the aggregate then divides missing tokens' wall time
/// into real tokens and the number comparable to nothing.
fn cell_is_vacuous(requests: &[RequestEvidence], osl: usize) -> bool {
    requests
        .iter()
        .any(|r| (r.completion_tokens as f64) < VACUITY_FLOOR * osl as f64)
}

fn cache_is_uncontrolled(requests: &[RequestEvidence], warmup: usize) -> bool {
    warmup > 0
        && requests.iter().any(|request| {
            request.prompt_tokens == 0
                || (request.cached_prompt_tokens as f64)
                    < WARM_CACHE_FLOOR * request.prompt_tokens as f64
        })
}

/// The prompt identities one cell executes before and during measurement.
/// Keeping the policy in one value makes the cache-state claim testable
/// without substituting a fake HTTP implementation for the benchmark.
#[derive(Debug, PartialEq, Eq)]
struct PromptPlan {
    warmup_rounds: Vec<Vec<String>>,
    measured: Vec<String>,
}

fn prompt_plan(conc: usize, warmup: usize) -> PromptPlan {
    let measured: Vec<String> = (0..conc).map(|i| format!("c{i}")).collect();
    PromptPlan {
        warmup_rounds: vec![measured.clone(); warmup],
        measured,
    }
}

#[derive(Default)]
struct CellRow {
    isl: usize,
    conc: usize,
    ttft: Percentiles,
    tpot: Percentiles,
    e2e_p50: Option<f64>,
    throughput: f64,
    errors: usize,
    requests: Vec<RequestEvidence>,
    vacuous: bool,
    cache_uncontrolled: bool,
}

impl CellRow {
    fn min_completion(&self) -> Option<usize> {
        self.requests.iter().map(|r| r.completion_tokens).min()
    }
    fn min_cached_prompt(&self) -> Option<usize> {
        self.requests.iter().map(|r| r.cached_prompt_tokens).min()
    }
    fn min_cached_prompt_pct(&self) -> Option<f64> {
        self.requests
            .iter()
            .map(|request| {
                if request.prompt_tokens == 0 {
                    0.0
                } else {
                    request.cached_prompt_tokens as f64 / request.prompt_tokens as f64 * 100.0
                }
            })
            .reduce(f64::min)
    }
    /// Clean and above the vacuity floor — the only rows metrics may quote.
    fn comparable(&self) -> bool {
        self.errors == 0 && !self.vacuous && !self.cache_uncontrolled
    }
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
    /// Verdict floors (tok/s); all 0.0 = info verdict. Gate-filled per
    /// variant via `GATE_THRESHOLD_PARAMS`.
    floors: verdict::Floors,
}

impl ConcurrencySweep {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    /// One verdict-floor spec. The generation/sweep knobs are the benchmark;
    /// these are the gate's bars and cannot move a measured number.
    fn floor_spec(key: &'static str, label: &'static str) -> ParamSpec {
        ParamSpec::new(
            key,
            label,
            "Run-verdict floor on this rung's aggregate tok/s (comparable cells only). 0 \
             disables (a standalone run reports an info verdict); under --pull-request-gate \
             it is auto-filled from the variant's BENCH.toml `min` bound. Vacuous or errored \
             sweeps never PASS regardless.",
            ParamKind::Float {
                min: 0.0,
                max: 100_000.0,
            },
            // 0.0 is the documented OFF state, not an implicit bar (PCND).
            ParamValue::Float(0.0),
        )
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    /// The prompt for one cell: ISL padding first (the prefix-cache
    /// mechanics live in [`stats::make_prompt`]), the mode's task last.
    fn cell_prompt(&self, isl: usize, prefix_tag: &str) -> String {
        match self.mode {
            PromptMode::Count => stats::make_prompt(isl, PromptMode::Count, prefix_tag),
            PromptMode::Natural => {
                let mut p = stats::make_prompt(isl, PromptMode::Natural, prefix_tag);
                p.push(' ');
                p.push_str(CODE_TASK);
                p
            }
        }
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
            // Pinned sampling: the ladder is a performance instrument, and an
            // unpinned draw is run-to-run noise in the one thing it measures.
            "temperature": 0.0,
            "seed": 0,
            // Ladders measure DECODE, not thinking. On a thinking-on serve
            // the output budget otherwise disappears into <think> at whatever
            // effort the template defaults to, and the cell measures
            // reasoning length instead of decode.
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": self.cell_prompt(isl, &prefix_tag)}],
        });
        http::chat_stream(target, &body, self.timeout).await
    }

    async fn run_cell(&mut self, isl: usize, conc: usize) -> Result<CellRow> {
        let handle = self.handle()?.clone();
        let plan = prompt_plan(conc, self.warmup);
        for (w, tags) in plan.warmup_rounds.iter().enumerate() {
            handle.check_cancelled()?;
            handle.status(format!(
                "isl {isl} · conc {conc} · warmup round {}/{}",
                w + 1,
                self.warmup
            ));
            // Prime every exact prompt the measured batch will use. A single
            // unrelated `warm` tag leaves all measured prompts cold, while
            // reusing c0/c1/... across later rungs produces a history-dependent
            // mixture of cached and new prompts. A failed warm-up invalidates
            // the declared setup instead of silently changing the instrument.
            for tag in tags {
                self.one(isl, tag.clone()).await.with_context(|| {
                    format!("isl {isl} conc {conc}: warm-up prompt {tag} failed")
                })?;
            }
        }
        handle.check_cancelled()?;
        handle.status(format!("isl {isl} · conc {conc} · {conc} in flight"));

        let batch_start = Instant::now();
        let futures: Vec<_> = plan
            .measured
            .into_iter()
            .map(|tag| self.one(isl, tag))
            .collect();
        let outcomes = futures::future::join_all(futures).await;
        let wall = batch_start.elapsed().as_secs_f64().max(1e-6);

        let mut ttft = Vec::new();
        let mut tpot = Vec::new();
        let mut e2e = Vec::new();
        let mut requests = Vec::new();
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
                    requests.push(RequestEvidence {
                        completion_tokens: o.completion_tokens,
                        prompt_tokens: o.prompt_tokens,
                        cached_prompt_tokens: o.cached_prompt_tokens,
                        finish_reason: o.finish_reason.clone(),
                        server_ttft_ms: o.server_ttft_ms,
                        server_tps: o.server_tps,
                    });
                }
                Err(e) => {
                    errors += 1;
                    handle.warn(format!("isl {isl} conc {conc}: {e:#}"));
                }
            }
        }
        let vacuous = cell_is_vacuous(&requests, self.osl);
        // Matching prompt bytes reach the intended setup, but the endpoint's
        // usage is the oracle for whether the measured request actually used
        // the warmed prompt. A small shared chat-template prefix is not enough:
        // require a material cached fraction of each measured prompt.
        let cache_uncontrolled = cache_is_uncontrolled(&requests, self.warmup);
        handle.info(evidence_line(isl, conc, &requests));
        if vacuous {
            handle.warn(format!(
                "isl {isl} conc {conc}: a request delivered under {:.0}% of the {}-token \
                 budget (min {} tok) — this cell's tok/s is NOT comparable",
                VACUITY_FLOOR * 100.0,
                self.osl,
                requests
                    .iter()
                    .map(|r| r.completion_tokens)
                    .min()
                    .unwrap_or(0),
            ));
        }
        if cache_uncontrolled {
            handle.warn(format!(
                "isl {isl} conc {conc}: warm-up was requested but at least one measured request \
                 reported less than {:.0}% of its prompt as cached — this cell is NOT comparable",
                WARM_CACHE_FLOOR * 100.0,
            ));
        }
        Ok(CellRow {
            isl,
            conc,
            ttft: Percentiles::of(&ttft),
            tpot: Percentiles::of(&tpot),
            e2e_p50: stats::percentile(&e2e, 50),
            throughput: tokens as f64 / wall,
            errors,
            requests,
            vacuous,
            cache_uncontrolled,
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
                Column::right("min tok", 7),
                Column::right("min cache%", 10),
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
                // A vacuous cell's tok/s is printed struck with a marker, not
                // hidden: the number is evidence of the failure, it is just
                // not comparable to anything.
                if r.vacuous || r.cache_uncontrolled {
                    Cell::styled(format!("{:.1}*", r.throughput), CellStyle::Bad)
                } else {
                    Cell::styled(format!("{:.1}", r.throughput), CellStyle::Good)
                },
                Cell::new(
                    r.min_completion()
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "—".into()),
                ),
                Cell::new(
                    r.min_cached_prompt_pct()
                        .map(|v| format!("{v:.0}"))
                        .unwrap_or_else(|| "—".into()),
                ),
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
            .filter(|r| r.comparable())
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

    /// The gate/dashboard channel. Vacuous or errored cells are EXCLUDED from
    /// every throughput/TTFT key — a future threshold must never be minted
    /// from a cell whose tokens were not delivered — while
    /// `min_completion_tokens` spans ALL requests, because it is the evidence
    /// the exclusion decision rests on.
    fn metrics(&self) -> BTreeMap<String, f64> {
        let mut m = BTreeMap::new();
        // Per-C: the best comparable aggregate across ISLs at that rung (the
        // sustained curve the ladder is quoted on); TTFT p50 from that cell.
        let mut per_c: BTreeMap<usize, &CellRow> = BTreeMap::new();
        for r in self.rows.iter().filter(|r| r.comparable()) {
            let slot = per_c.entry(r.conc).or_insert(r);
            if r.throughput > slot.throughput {
                *slot = r;
            }
        }
        for (c, r) in &per_c {
            m.insert(format!("c{c}_aggregate_tok_s"), r.throughput);
            if let Some(t) = r.ttft.p50 {
                m.insert(format!("c{c}_ttft_p50_ms"), t);
            }
        }
        if let Some(peak) = per_c.values().map(|r| r.throughput).max_by(f64::total_cmp) {
            m.insert("peak_aggregate_tok_s".to_string(), peak);
        }
        if let Some(min) = self.rows.iter().filter_map(CellRow::min_completion).min() {
            m.insert("min_completion_tokens".to_string(), min as f64);
        }
        if let Some(min) = self
            .rows
            .iter()
            .filter_map(CellRow::min_cached_prompt)
            .min()
        {
            m.insert("min_cached_prompt_tokens".to_string(), min as f64);
        }
        if let Some(min) = self
            .rows
            .iter()
            .filter_map(CellRow::min_cached_prompt_pct)
            .reduce(f64::min)
        {
            m.insert("min_cached_prompt_pct".to_string(), min);
        }
        m.insert(
            "vacuous_cells".to_string(),
            self.rows.iter().filter(|r| r.vacuous).count() as f64,
        );
        m.insert(
            "cache_uncontrolled_cells".to_string(),
            self.rows.iter().filter(|r| r.cache_uncontrolled).count() as f64,
        );
        m
    }
}

/// One compact per-request evidence line for the run log: delivered tokens in
/// request order, a finish-reason histogram, and the server's own prefill/
/// decode numbers when the stream carried `usage`.
fn evidence_line(isl: usize, conc: usize, requests: &[RequestEvidence]) -> String {
    let toks: Vec<String> = requests
        .iter()
        .map(|r| r.completion_tokens.to_string())
        .collect();
    let cached: Vec<String> = requests
        .iter()
        .map(|r| format!("{}/{}", r.cached_prompt_tokens, r.prompt_tokens))
        .collect();
    let mut finish: BTreeMap<&str, usize> = BTreeMap::new();
    for r in requests {
        *finish
            .entry(r.finish_reason.as_deref().unwrap_or("?"))
            .or_default() += 1;
    }
    let finish: Vec<String> = finish.iter().map(|(k, n)| format!("{k}×{n}")).collect();
    let sttft: Vec<f64> = requests.iter().filter_map(|r| r.server_ttft_ms).collect();
    let stps: Vec<f64> = requests.iter().filter_map(|r| r.server_tps).collect();
    let mut line = format!(
        "evidence isl {isl} conc {conc}: tok [{}] · cached [{}] · finish [{}]",
        toks.join(","),
        cached.join(","),
        finish.join(",")
    );
    if let Some(v) = stats::percentile(&sttft, 50) {
        line.push_str(&format!(" · server ttft p50 {v:.0} ms"));
    }
    if let Some(v) = stats::percentile(&stps, 50) {
        line.push_str(&format!(" · server decode p50 {v:.1} tok/s"));
    }
    line
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
        let mut specs = vec![
            ParamSpec::new(
                "concurrencies",
                "Concurrency levels",
                "How many requests are in flight at once, one sweep column each.",
                ParamKind::IntList { min: 1, max: 256 },
                // 32 is the top rung on purpose: it is where the campaign
                // measured time-to-answer INVERTING in Atlas's favour (C=32
                // -4.47% vs vLLM, C=128 -10.84%), so a sweep that stops at 16
                // reports the regime where Atlas trails and omits the one
                // where it wins. C=64/128 are deliberately NOT default — they
                // need bs=64 preflight headroom that not every recipe has.
                ParamValue::IntList(vec![1, 2, 4, 8, 16, 32]),
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
                "Warm-up rounds",
                "Unmeasured rounds per cell. Each round runs every exact prompt in the measured \
                 batch so prefix-cache state is controlled before timing.",
                ParamKind::Int { min: 0, max: 8 },
                ParamValue::Int(1),
            ),
            ParamSpec::new(
                "prompt_mode",
                "Prompt mode",
                // ★ Honest phrasing: NEITHER mode forces the output budget —
                // the server has no ignore_eos, so nothing can. Short
                // completions are caught by the vacuity floor instead of
                // being promised away here.
                "natural (default) poses a code-generation task that reliably fills the output \
                 budget; count appends a counting instruction the model may still stop early on. \
                 Neither forces the budget — under-budget cells are flagged vacuous.",
                ParamKind::Choice(&["natural", "count"]),
                ParamValue::Text("natural".into()),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Request timeout",
                "Seconds before a single request is abandoned and counted as an error.",
                ParamKind::Int { min: 10, max: 3600 },
                ParamValue::Int(600),
            ),
        ];
        // One floor per gated rung, plus the peak — both from RUNGS/PEAK_FLOOR,
        // so a rung added there arrives here without a second edit.
        specs.extend(RUNGS.iter().map(|(_, key, _, label)| Self::floor_spec(key, label)));
        specs.push(Self::floor_spec(PEAK_FLOOR.0, PEAK_FLOOR.2));
        specs
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
            .context("prompt_mode must be natural or count")?;
        self.timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        let mut per_c = Vec::with_capacity(RUNGS.len());
        for (c, key, _, _) in RUNGS {
            per_c.push((c, values.float(key)?));
        }
        self.floors = verdict::Floors {
            per_c,
            peak: values.float(PEAK_FLOOR.0)?,
        };
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
            let vacuous = self.rows.iter().filter(|r| r.vacuous).count();
            let cache_uncontrolled = self.rows.iter().filter(|r| r.cache_uncontrolled).count();
            // The verdict is computed over the SAME metrics map the gate
            // record carries (see `verdict::sweep_verdict`), so the two can
            // never disagree about a rung's value. Floors all-zero keeps the
            // pre-gate info verdicts verbatim.
            let metrics = self.metrics();
            let verdict = verdict::sweep_verdict(
                &metrics,
                self.rows.len(),
                errors,
                vacuous,
                cache_uncontrolled,
                VACUITY_FLOOR * 100.0,
                &self.floors,
            );
            let mut frame = BenchmarkResult {
                status: RunStatus::Completed,
                ..BenchmarkResult::running("done", self.elapsed())
            }
            .with_progress(self.cells.len() as u64, self.cells.len() as u64)
            .with_summary(self.summary())
            .with_table(self.table())
            .with_metrics(metrics)
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
            "isl {isl} conc {conc}: ttft p50 {} ms · tpot p50 {} ms · {:.1} tok/s{}",
            stats::fmt_ms(row.ttft.p50),
            stats::fmt_ms(row.tpot.p50),
            row.throughput,
            if row.vacuous { " (vacuous)" } else { "" }
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

#[path = "concurrency_verdict.rs"]
mod verdict;

#[cfg(test)]
#[path = "concurrency_tests.rs"]
mod concurrency_tests;
