// SPDX-License-Identifier: AGPL-3.0-only

//! Expert Categorization — which MoE experts does each kind of prompt need?
//!
//! Sends a corpus of short prompts grouped by category, reads the per-layer
//! expert routing off each response (`usage.expert_activation`, requires a
//! serve started with `--expert-telemetry`), and reduces it to one table:
//! for every category, the smallest set of experts per layer that carries
//! `coverage` of that layer's routing mass.
//!
//! That table is the input to selective expert loading: pasted into the
//! model's MODEL.toml, `spark serve --expert-category <name>` loads only
//! those experts. The memory saving is exactly the fraction of experts a
//! category does NOT need, which is why the run reports mean experts per
//! layer alongside the sets.
//!
//! # Not a gate
//!
//! There is no threshold and no baseline. This is a measurement tool that
//! produces a config artifact, not a pass/fail number — the same posture
//! `quick-speed-bench` has. Judging a category's expert count against a
//! committed bound would gate the MODEL, not the change under test.
//!
//! # What makes a run trustworthy
//!
//! Generation is pinned (temperature 0, seed 0, `reasoning_effort: "none"`)
//! so two runs of the same corpus are comparable by construction, and the
//! prompt — not the continuation — carries the category signal, which is
//! also why `max_tokens` is small. Every response is checked against the
//! server's own conservation identity before it is folded (see `usage.rs`);
//! a response that fails stops the run rather than averaging in.

use std::collections::BTreeMap;
use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::json;

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::hardware::Sensitivity;
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{BenchmarkResult, Cell, Column, LogLine, ResultTable, RunStatus, Stat};

pub mod aggregate;
pub mod corpus;
pub mod report;
pub mod usage;

const SUMMARY: &str = "Maps prompt categories to the MoE experts they route to, \
                       for selective expert loading";
pub const METADATA: PluginMetadata = PluginMetadata::atlas(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "expert-categories",
    name: "Expert Categorization",
    summary: SUMMARY,
    detail: "Sends a committed corpus of short prompts grouped by category (python, rust, \
             sql, math, translation, creative writing, science, chat, tool calling, legal) \
             and reads each response's per-layer MoE expert routing from \
             usage.expert_activation. Per category and layer it keeps the smallest set of \
             experts covering `coverage` of the routing MASS — not the most frequently \
             chosen, since an expert picked often at trivial weight contributes less than \
             one picked rarely at high weight. Emits a paste-ready \
             [expert_categories] block for the model's MODEL.toml plus a stats artifact \
             carrying the full per-expert distribution and the cross-category overlap. \
             REQUIRES a serve started with --expert-telemetry on an MoE checkpoint; a \
             response without the routing report stops the run rather than averaging a \
             zero. Not a gate: it produces a config artifact, not a pass/fail number.",
    duration_hint: "~10-20 min",
    updated: "2026-08-30",
    // No committed baseline anywhere: the table describes whatever model it
    // is pointed at, and every MoE has a different expert space.
    intended_for: None,
    threshold_params: &[],
    needs_confirmation: false,
    // Routing is deterministic at temperature 0; a thermally throttled box
    // reports the same experts, just later.
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(ExpertCategories::default()),
};

#[derive(Default)]
pub struct ExpertCategories {
    handle: Option<PluginHandle>,
    timeout: Duration,
    coverage: f64,
    per_category: usize,
    max_tokens: usize,
    selection: Vec<String>,
    rows: Vec<corpus::Row>,
    next_row: usize,
    acc: aggregate::Accumulator,
    started: Option<Instant>,
    probed: bool,
}

impl ExpertCategories {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    /// The pinned request. `report_expert_metadata` is the whole point;
    /// temperature/seed are pinned so a re-run measures the same routing.
    fn body(&self, model: &str, prompt: &str) -> serde_json::Value {
        json!({
            "model": model,
            "stream": true,
            "temperature": 0.0,
            "seed": 0,
            "max_tokens": self.max_tokens,
            "reasoning_effort": "none",
            "report_expert_metadata": true,
            "messages": [{"role": "user", "content": prompt}],
        })
    }

    fn table(&self) -> ResultTable {
        let mut t = ResultTable::new(
            "EXPERT CATEGORIES",
            vec![
                Column::left("Category", 18),
                Column::right("Prompts", 8),
                Column::right("Tokens routed", 14),
                Column::right("Experts/layer", 14),
            ],
        );
        for b in self.acc.budgets(self.coverage) {
            t.push(vec![
                Cell::new(b.category.clone()),
                Cell::new(b.totals.prompts.to_string()),
                Cell::new(b.totals.tokens_routed.to_string()),
                Cell::new(format!("{:.1}", report::mean_experts(&b))),
            ]);
        }
        t
    }
}

impl Plugin for ExpertCategories {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.handle = Some(handle);
        self.started = Some(Instant::now());
        async { Ok(()) }
    }
}

impl Benchmark for ExpertCategories {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new(
                "coverage",
                "Routing-mass coverage",
                "Fraction of each layer's routing mass the kept expert set must cover. \
                 Lower keeps fewer experts (less memory, more routing that falls outside \
                 the loaded set); 0.90 is the default the emitted table records alongside \
                 the ids, so a table is never separable from the threshold that made it.",
                ParamKind::Float {
                    min: 0.5,
                    max: 0.999,
                },
                ParamValue::Float(0.90),
            ),
            ParamSpec::new(
                "prompts_per_category",
                "Prompts per category",
                "How many of each category's corpus rows to send, taken in file order. 32 \
                 gives roughly 20k weighted expert draws per layer per category, enough \
                 for the budgeted set to be stable run to run. Small values are for \
                 smoke-testing the wiring — the tail of the set churns below about 8.",
                ParamKind::Int { min: 1, max: 64 },
                ParamValue::Int(32),
            ),
            ParamSpec::new(
                "max_tokens",
                "Max tokens",
                "Generation cap per prompt. Small on purpose: routing is attributed to the \
                 PROMPT, so decode only costs wall-clock here.",
                ParamKind::Int { min: 1, max: 256 },
                ParamValue::Int(48),
            ),
            ParamSpec::new(
                "categories",
                "Categories",
                "Comma-separated category ids to measure, or `all`. An unknown id is an \
                 error naming the corpus's categories rather than a silently empty run.",
                ParamKind::Text,
                ParamValue::Text("all".to_string()),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Request timeout",
                "Seconds before a single request is abandoned. Transport-side only.",
                ParamKind::Int { min: 30, max: 600 },
                ParamValue::Int(120),
            ),
        ]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        self.coverage = values.float("coverage")?;
        self.per_category = values.usize("prompts_per_category")?;
        self.max_tokens = values.usize("max_tokens")?;
        self.selection = corpus::parse_selection(values.text("categories")?);
        self.timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);

        let all = corpus::load()?;
        self.rows = corpus::draw(&all, self.per_category, &self.selection)?;
        self.next_row = 0;
        self.acc = aggregate::Accumulator::new();
        self.probed = false;
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;
        let total = self.rows.len() as u64;

        if !self.probed {
            self.probed = true;
            http::probe(handle.target(), Duration::from_secs(10))
                .await
                .context("endpoint probe failed — check the target URL and port")?;
            let cats = corpus::categories(&self.rows);
            return Ok(BenchmarkResult::running("probe", self.elapsed())
                .with_progress(0, total)
                .log_line(LogLine::info(format!(
                    "{} · {} prompts · {} categories · coverage {:.2} · max_tokens {}",
                    handle.target().base_url,
                    total,
                    cats.len(),
                    self.coverage,
                    self.max_tokens,
                ))));
        }

        // One prompt per step: a whole category is a minute of work, and
        // `next()` must return promptly or the run pane stops repainting and
        // cancel stops responding.
        if self.next_row < self.rows.len() {
            let row = self.rows[self.next_row].clone();
            handle.status(format!("{} · {}", row.category, row.id));
            let target = handle.target();
            let body = self.body(&target.model, &row.prompt);
            let outcome = http::chat_stream(target, &body, self.timeout)
                .await
                .with_context(|| format!("request failed for prompt '{}'", row.id))?;

            let raw = outcome
                .expert_activation
                .as_ref()
                .ok_or_else(|| usage::missing_report_error(&row.id))?;
            let act = usage::parse(raw, &row.id)?;
            self.acc.feed(&row.category, &act)?;

            self.next_row += 1;
            let done = self.next_row as u64;
            handle.progress(done, total);
            let line = LogLine::info(format!(
                "{} {}: {} layers · {} routed positions",
                row.category,
                row.id,
                act.layers.len(),
                act.tokens_routed,
            ));
            return Ok(BenchmarkResult::running("measuring", self.elapsed())
                .with_progress(done, total)
                .with_table(self.table())
                .log_line(line));
        }

        let budgets = self.acc.budgets(self.coverage);
        if budgets.is_empty() {
            bail!("no category produced any routing — nothing to categorize");
        }

        let model = handle.target().model.clone();
        let sha = report::corpus_sha256(include_str!(
            "../../../assets/expert-categories/corpus.jsonl"
        ));
        let toml = report::toml_block(&model, &sha, self.max_tokens, &budgets);
        let stats = report::stats_json(&model, &sha, self.coverage, &self.acc, &budgets);

        let dir = handle.artifacts().runs_dir(DESCRIPTOR.id)?;
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let toml_path = dir.join("expert_categories.toml");
        let stats_path = dir.join("stats.json");
        std::fs::write(&toml_path, &toml)
            .with_context(|| format!("writing {}", toml_path.display()))?;
        std::fs::write(&stats_path, serde_json::to_vec_pretty(&stats)?)
            .with_context(|| format!("writing {}", stats_path.display()))?;

        let mut metrics = BTreeMap::new();
        metrics.insert("categories".to_string(), budgets.len() as f64);
        let mean_experts: f64 =
            budgets.iter().map(report::mean_experts).sum::<f64>() / budgets.len() as f64;
        metrics.insert("mean_experts_per_layer".to_string(), mean_experts);
        let pairs = report::overlap_pairs(&budgets);
        if let Some((_, _, worst)) = pairs.first() {
            metrics.insert("max_pair_jaccard".to_string(), *worst);
        }

        let num_experts = self.acc.num_experts() as f64;
        let mut stats_out = vec![
            Stat::new("Categories", budgets.len().to_string(), ""),
            Stat::new(
                "Experts kept per layer (mean)",
                format!("{mean_experts:.1} of {}", self.acc.num_experts()),
                "",
            ),
        ];
        if num_experts > 0.0 {
            stats_out.push(Stat::new(
                "Routed-expert memory",
                format!("{:.0}%", 100.0 * mean_experts / num_experts),
                "of full",
            ));
        }
        if let Some((a, b, j)) = pairs.first() {
            // The most-similar pair bounds how much categorization can buy:
            // two categories that route alike cannot be given different sets.
            stats_out.push(Stat::new(
                "Most-similar pair",
                format!("{a} / {b} — {j:.2} Jaccard"),
                "",
            ));
        }

        let mut logs = vec![
            LogLine::info(format!("wrote {}", toml_path.display())),
            LogLine::info(format!("wrote {}", stats_path.display())),
            LogLine::info(
                "paste the [expert_categories] block into the model's MODEL.toml and REBUILD \
                 (it is read at build time), then serve with --expert-category <name>"
                    .to_string(),
            ),
        ];
        let unattributed: u64 = budgets.iter().map(|b| b.totals.unattributed_rows).sum();
        if unattributed > 0 {
            // Surfaced, never swallowed: it means the table describes a
            // prefix of some prompts.
            logs.push(LogLine::warn(format!(
                "{unattributed} token positions were not attributed — some prompts exceeded \
                 the server's telemetry staging width, so their tables cover a prefix"
            )));
        }

        Ok(BenchmarkResult {
            status: RunStatus::Completed,
            metrics,
            summary: stats_out,
            log: logs,
            ..BenchmarkResult::running("done", self.elapsed())
        }
        .with_table(self.table()))
    }
}
