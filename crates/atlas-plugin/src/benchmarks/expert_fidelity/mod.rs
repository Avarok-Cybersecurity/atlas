// SPDX-License-Identifier: AGPL-3.0-only

//! Expert Fidelity — how far a restricted serve has moved from the full model.
//!
//! The instrument for every expert-pruning decision. Selective expert
//! loading trades memory for quality, and until that quality cost is a
//! NUMBER, every choice about it — coverage, which experts to protect,
//! whether a compensation scheme helped — is an argument about anecdotes.
//!
//! ## Two runs, one comparison
//!
//! `capture` runs against the FULL model and writes a baseline: for each
//! prompt, the model's own greedy continuation and the log-probability it
//! assigned to every token of it.
//!
//! `measure` runs against a RESTRICTED serve (`--expert-category ...`) and
//! teacher-forces those same sequences — the restricted model never chooses
//! the text, it only scores it. That is what keeps the two comparable: under
//! free generation one flipped token diverges everything after it, and the
//! comparison stops being about the same positions.
//!
//! The headline is **ΔCE**, nats per token of extra surprise. 0.0 means the
//! restricted serve finds the full model's output exactly as natural.
//! **Top-1 agreement** is reported beside it because they answer different
//! questions: agreement predicts whether greedy generation diverges, ΔCE
//! measures how close the run came to diverging.
//!
//! Not a gate. It is a measurement tool whose whole purpose is to compare
//! configurations of the same model against each other.

use std::collections::BTreeMap;
use std::future::Future;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::json;

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::hardware::Sensitivity;
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{
    BenchmarkResult, Cell, CellStyle, Column, LogLine, ResultTable, RunStatus, Stat,
};

use super::expert_categories::corpus;

pub mod score;
use score::{Reference, Scored};

const SUMMARY: &str = "Teacher-forced divergence of a restricted serve from the full model";
pub const METADATA: PluginMetadata = PluginMetadata::atlas(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "expert-fidelity",
    name: "Expert Fidelity",
    summary: SUMMARY,
    detail: "Measures what selective expert loading costs in quality, as a number. Run once \
             in `capture` mode against the full model to record its own greedy continuations \
             and the log-probability it assigns to each token of them; then in `measure` \
             mode against a serve started with --expert-category, which teacher-forces those \
             same sequences rather than generating. The headline is delta-CE, nats per token \
             of extra surprise the restricted serve assigns to the full model's output — 0.0 \
             is indistinguishable. Top-1 agreement is reported beside it: agreement predicts \
             whether greedy generation diverges at all, delta-CE measures how close it came. \
             Both are needed, because a configuration can hold agreement while losing \
             confidence everywhere, which is invisible to greedy output and predicts \
             fragility under sampling. Byte comparison of generated text cannot do this job: \
             one flipped token diverges everything after it, and a valid paraphrase scores \
             the same as a corruption. Per-category results separate the category a serve \
             holds experts for from the traffic it does not. Not a gate — it compares \
             configurations of one model, so there is no cross-model baseline to gate on.",
    duration_hint: "~5-15 min",
    updated: "2026-08-30",
    intended_for: None,
    threshold_params: &[],
    needs_confirmation: false,
    // Teacher-forced scoring is deterministic; a throttled box returns the
    // same log-probabilities, just later.
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(ExpertFidelity::default()),
};

#[derive(Default, PartialEq, Clone, Copy)]
enum Mode {
    #[default]
    Capture,
    Measure,
}

#[derive(Default)]
pub struct ExpertFidelity {
    handle: Option<PluginHandle>,
    mode: Mode,
    timeout: Duration,
    max_tokens: usize,
    top_logprobs: u8,
    rows: Vec<corpus::Row>,
    next_row: usize,
    /// capture mode: what we are building. measure mode: what we compare to.
    references: Vec<Reference>,
    by_id: BTreeMap<String, usize>,
    scored: Vec<Scored>,
    dropped: Vec<String>,
    started: Option<Instant>,
    probed: bool,
}

impl ExpertFidelity {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    fn baseline_path(&self) -> Result<PathBuf> {
        Ok(self
            .handle()?
            .artifacts()
            .runs_dir(DESCRIPTOR.id)?
            .join("fidelity_baseline.json"))
    }

    /// The full model's own greedy continuation. Pinned hard: the baseline is
    /// only a baseline if a re-capture reproduces it.
    async fn generate(&self, prompt: &str) -> Result<String> {
        let handle = self.handle()?;
        let target = handle.target();
        let body = json!({
            "model": target.model,
            "temperature": 0.0,
            "seed": 0,
            "max_tokens": self.max_tokens,
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": prompt}],
        });
        let out = http::chat_blocking(target, &body, self.timeout).await?;
        Ok(out.choices.first().cloned().unwrap_or_default())
    }

    /// Score a prompt+continuation without generating anything.
    async fn teacher_force(
        &self,
        prompt: &str,
        continuation: &str,
    ) -> Result<(Vec<f32>, Vec<String>)> {
        let handle = self.handle()?;
        let text = format!("{prompt}{continuation}");
        let v =
            http::teacher_force(handle.target(), &text, self.top_logprobs, self.timeout).await?;
        score::extract(&v, prompt.len())
    }

    fn table(&self) -> ResultTable {
        let mut t = ResultTable::new(
            "EXPERT FIDELITY",
            vec![
                Column::left("Category", 20),
                Column::right("Prompts", 8),
                Column::right("ΔCE nats/tok", 13),
                Column::right("Top-1 agree", 12),
            ],
        );
        if let Some(f) = score::aggregate(&self.scored) {
            for (cat, (ce, ag, n)) in &f.per_category {
                t.push(vec![
                    Cell::new(cat.clone()),
                    Cell::new(n.to_string()),
                    Cell::new(format!("{ce:+.4}")),
                    Cell::new(format!("{:.1}%", 100.0 * ag)),
                ]);
            }
        }
        t
    }
}

impl Plugin for ExpertFidelity {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.handle = Some(handle);
        self.started = Some(Instant::now());
        async { Ok(()) }
    }
}

impl Benchmark for ExpertFidelity {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new(
                "mode",
                "Mode",
                "`capture` records the full model's continuations and its log-probabilities \
                 for them — run it against an UNRESTRICTED serve. `measure` teacher-forces \
                 that baseline against the serve you are testing. There is no default that \
                 is safe for both, so capture is the default and measure fails fast when no \
                 baseline exists.",
                ParamKind::Text,
                ParamValue::Text("capture".to_string()),
            ),
            ParamSpec::new(
                "prompts_per_category",
                "Prompts per category",
                "Rows to take from each category of the expert-categories corpus, in file \
                 order. The metric is position-weighted, so 25 prompts at 48 tokens is about \
                 1200 scored positions per category — enough to separate configurations that \
                 a five-prompt byte comparison ties.",
                ParamKind::Int { min: 1, max: 100 },
                ParamValue::Int(25),
            ),
            ParamSpec::new(
                "max_tokens",
                "Continuation length",
                "How many tokens of the full model's own output to score against. Longer \
                 continuations drift further from the prompt and are where restriction \
                 damage shows up, but cost capture time linearly.",
                ParamKind::Int { min: 8, max: 512 },
                ParamValue::Int(48),
            ),
            ParamSpec::new(
                "categories",
                "Categories",
                "Comma-separated category ids, or `all`. Measuring a category-restricted \
                 serve on ALL categories is the point — the cost of restriction shows up in \
                 the categories it does not hold experts for.",
                ParamKind::Text,
                ParamValue::Text("all".to_string()),
            ),
            ParamSpec::new(
                "top_logprobs",
                "Top-k logprobs",
                "Alternatives requested per position; only the argmax is used, for the \
                 agreement number. Larger values cost response size, not accuracy.",
                ParamKind::Int { min: 1, max: 20 },
                ParamValue::Int(1),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Request timeout",
                "Seconds before a single request is abandoned. Transport-side only.",
                ParamKind::Int { min: 30, max: 600 },
                ParamValue::Int(180),
            ),
        ]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        self.mode = match values.text("mode")?.trim().to_ascii_lowercase().as_str() {
            "capture" => Mode::Capture,
            "measure" => Mode::Measure,
            other => bail!("unknown mode '{other}' — expected `capture` or `measure`"),
        };
        self.max_tokens = values.usize("max_tokens")?;
        self.top_logprobs = values.usize("top_logprobs")? as u8;
        self.timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);

        let all = corpus::load()?;
        let selection = corpus::parse_selection(values.text("categories")?);
        self.rows = corpus::draw(&all, values.usize("prompts_per_category")?, &selection)?;
        self.next_row = 0;
        self.references.clear();
        self.by_id.clear();
        self.scored.clear();
        self.dropped.clear();
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
            if self.mode == Mode::Measure {
                let path = self.baseline_path()?;
                let raw = std::fs::read_to_string(&path).with_context(|| {
                    format!(
                        "no fidelity baseline at {}.\nWHAT: `measure` compares against the \
                         FULL model's own output.\nWHY: nothing has captured it yet.\nFIX: \
                         run this benchmark once with --param mode=capture against a serve \
                         started WITHOUT --expert-category, then re-run measure.",
                        path.display()
                    )
                })?;
                self.references = serde_json::from_str(&raw)
                    .with_context(|| format!("parsing {}", path.display()))?;
                self.by_id = self
                    .references
                    .iter()
                    .enumerate()
                    .map(|(i, r)| (r.id.clone(), i))
                    .collect();
                // Measuring rows the baseline never captured would silently
                // score nothing; measuring a subset of it is fine.
                self.rows.retain(|r| self.by_id.contains_key(&r.id));
                if self.rows.is_empty() {
                    bail!(
                        "the baseline at {} shares no prompts with the current selection — \
                         it was captured from a different corpus or category set",
                        path.display()
                    );
                }
            }
            let mode = if self.mode == Mode::Capture {
                "capture (full model)"
            } else {
                "measure (teacher-forced vs baseline)"
            };
            return Ok(BenchmarkResult::running("probe", self.elapsed())
                .with_progress(0, total)
                .log_line(LogLine::info(format!(
                    "{} · {mode} · {} prompts · {} tokens each",
                    handle.target().base_url,
                    self.rows.len(),
                    self.max_tokens,
                ))));
        }

        if self.next_row < self.rows.len() {
            let row = self.rows[self.next_row].clone();
            handle.status(format!("{} · {}", row.category, row.id));

            let line = match self.mode {
                Mode::Capture => {
                    let cont = self.generate(&row.prompt).await?;
                    if cont.is_empty() {
                        bail!(
                            "prompt '{}' produced no output on the full model — a baseline of \
                             empty continuations would score every restricted serve as perfect",
                            row.id
                        );
                    }
                    let (lp, argmax) = self.teacher_force(&row.prompt, &cont).await?;
                    let n = lp.len();
                    self.references.push(Reference {
                        id: row.id.clone(),
                        category: row.category.clone(),
                        prompt: row.prompt.clone(),
                        continuation: cont,
                        logprobs: lp,
                        argmax,
                    });
                    LogLine::info(format!("{} {}: {n} scored positions", row.category, row.id))
                }
                Mode::Measure => {
                    let idx = self.by_id[&row.id];
                    let reference = self.references[idx].clone();
                    let (lp, argmax) = self
                        .teacher_force(&reference.prompt, &reference.continuation)
                        .await?;
                    match score::score_one(&reference, &lp, &argmax) {
                        Some(s) => {
                            let l = LogLine::info(format!(
                                "{} {}: ΔCE {:+.4} · top-1 {:.0}%",
                                row.category,
                                row.id,
                                s.delta_ce,
                                100.0 * s.top1_agreement
                            ));
                            self.scored.push(s);
                            l
                        }
                        None => {
                            // Tokenized differently from the baseline. Dropped
                            // rather than averaged over misaligned positions,
                            // and counted so the drop rate is visible.
                            self.dropped.push(row.id.clone());
                            LogLine::warn(format!(
                                "{} {}: position count differs from the baseline, dropped",
                                row.category, row.id
                            ))
                        }
                    }
                }
            };

            self.next_row += 1;
            let done = self.next_row as u64;
            handle.progress(done, total);
            return Ok(BenchmarkResult::running("scoring", self.elapsed())
                .with_progress(done, total)
                .with_table(self.table())
                .log_line(line));
        }

        let mut metrics = BTreeMap::new();
        let mut stats = Vec::new();
        let mut logs = Vec::new();

        if self.mode == Mode::Capture {
            let path = self.baseline_path()?;
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            std::fs::write(&path, serde_json::to_vec_pretty(&self.references)?)
                .with_context(|| format!("writing {}", path.display()))?;
            let positions: usize = self.references.iter().map(|r| r.logprobs.len()).sum();
            metrics.insert("baseline_prompts".to_string(), self.references.len() as f64);
            metrics.insert("baseline_positions".to_string(), positions as f64);
            stats.push(Stat::new(
                "Baseline captured",
                format!("{} prompts · {positions} positions", self.references.len()),
                "",
            ));
            logs.push(LogLine::info(format!("wrote {}", path.display())));
            logs.push(LogLine::info(
                "now restart the serve with --expert-category and re-run with \
                 --param mode=measure"
                    .to_string(),
            ));
        } else {
            let Some(f) = score::aggregate(&self.scored) else {
                bail!("no prompt scored — nothing to compare");
            };
            metrics.insert("delta_ce_nats".to_string(), f.delta_ce);
            metrics.insert("top1_agreement".to_string(), f.top1_agreement);
            metrics.insert("scored_positions".to_string(), f.positions as f64);
            stats.push(
                Stat::new("ΔCE", format!("{:+.4}", f.delta_ce), "nats/token")
                    .with_style(CellStyle::Good),
            );
            stats.push(Stat::new(
                "Top-1 agreement",
                format!("{:.1}%", 100.0 * f.top1_agreement),
                "",
            ));
            stats.push(Stat::new(
                "Scored",
                format!("{} positions · {} prompts", f.positions, f.prompts),
                "",
            ));
            if let Some(w) = f.worst.first() {
                stats.push(Stat::new(
                    "Worst prompt",
                    format!("{} ({}) ΔCE {:+.3}", w.id, w.category, w.delta_ce),
                    "",
                ));
            }
            if !self.dropped.is_empty() {
                // Surfaced, never swallowed: a high drop rate means the two
                // runs are not tokenizing the same text and the aggregate
                // describes whatever happened to align.
                logs.push(LogLine::warn(format!(
                    "{} of {} prompts dropped for position-count mismatch against the baseline",
                    self.dropped.len(),
                    self.rows.len()
                )));
                metrics.insert("dropped_prompts".to_string(), self.dropped.len() as f64);
            }
        }

        Ok(BenchmarkResult {
            status: RunStatus::Completed,
            metrics,
            summary: stats,
            log: logs,
            ..BenchmarkResult::running("done", self.elapsed())
        }
        .with_table(self.table()))
    }
}
