// SPDX-License-Identifier: AGPL-3.0-only

//! BFCL v4 single-turn — full and subset.
//!
//! The one benchmark that is not pure Rust, because its ground truth and its
//! AST checker live in `bfcl-eval`. The split is:
//!
//! * **Python** materializes the dataset and scores the responses. Both scripts
//!   are committed in `assets/bfcl/`, written into `~/.atlas/artifacts/bfcl`,
//!   and run from a venv provisioned during `load()`.
//! * **Rust** owns the draw, the generation, the streaming and the presentation
//!   — so the pane can show the resulting `n` before the run starts, and a run
//!   is cancellable between samples rather than only between phases.
//!
//! Generation is single-stream (`max_batch_size 1` semantics) and greedy, which
//! is what the recorded scores were produced with. Concurrency would change the
//! numbers.

pub mod aggregate;
pub mod dataset;
pub mod draw;
pub mod provision;
pub mod report;

use std::collections::BTreeMap;
use std::future::Future;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::json;

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::benchmarks::one_line;
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{BenchmarkResult, LogLine, RunStatus};

use draw::DrawSpec;

pub use report::{
    MLPERF_FLOOR_CHECKPOINTS, MLPERF_FLOOR_NORMALIZED, MLPERF_FLOOR_OVERALL,
    is_mlperf_submission_checkpoint,
};

mod descriptors;
pub use descriptors::{
    ECHOLP_A, ECHOLP_B, ECHOLP_C, ECHOLP_D, ECHOLP_METADATA, FULL_DESCRIPTOR, FULL_METADATA,
    SUBSET_A, SUBSET_B, SUBSET_C, SUBSET_D, SUBSET_DESCRIPTOR, SUBSET_ECHOLP_DESCRIPTOR,
    SUBSET_METADATA,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Subset,
    SubsetEcholp,
    Full,
}

/// What the scorer prints.
#[derive(Debug, Deserialize)]
struct Scores {
    overall_accuracy: f64,
    normalized_single_turn_score: f64,
    category_scores: BTreeMap<String, f64>,
    subset_scores: BTreeMap<String, f64>,
    total_samples: usize,
    unmatched_responses: usize,
    /// Per-subset `(hits, n)` from `score.py`. The only thing that recombines
    /// exactly across shards — see `benchmarks::bfcl::aggregate`. `default` so
    /// a record written before the scorer emitted these still deserializes.
    #[serde(default)]
    subset_totals: BTreeMap<String, (u64, u64)>,
}

/// Where the state machine is.
enum Phase {
    Provision,
    Generate,
    Score,
    Done,
}

pub struct Bfcl {
    variant: Variant,
    /// Which slice of the draw this run measures, if it is a shard of a group.
    /// `None` is the whole draw and is byte-for-byte the pre-shard behaviour.
    shard: Option<dataset::Shard>,
    handle: Option<PluginHandle>,
    phase: Phase,
    artifacts: Option<provision::Artifacts>,
    samples: Vec<dataset::Sample>,
    cursor: usize,
    responses: Vec<serde_json::Value>,
    responses_path: Option<PathBuf>,
    scores: Option<Scores>,
    // Parameters.
    spec: DrawSpec,
    max_new_tokens: usize,
    temperature: f64,
    request_timeout: Duration,
    started: Option<Instant>,
    tool_call_samples: usize,
    /// Samples whose request failed at the transport, scored as "no call".
    /// Published as a metric so a group can refuse a degraded member.
    transport_errors: usize,
    /// The served model, captured at `load()` from the target endpoint.
    /// Decides whether the MLPerf floor VERDICT applies (`report.rs`) — the
    /// floor rides on the Qwen3.6-27B submission checkpoints and does not
    /// transfer to other weights (BENCH.toml doctrine).
    target_model: Option<String>,
    /// Baseline floors for non-MLPerf checkpoints (`report::BaselineMins`) —
    /// gate-filled from the variant's BENCH.toml; both 0.0 = info verdict.
    baseline_mins: report::BaselineMins,
}

/// The `shard` parameter value that means "leave the constructor's slice
/// alone". See the spec's own note for why this is a word and not an empty
/// string or `0/1`.
pub const INHERIT_SHARD: &str = "inherit";

/// The per-sample output budget the MLPerf-edge config uses, and the budget
/// every committed BFCL record was measured at.
///
/// Exported because `kat_equality` replays this exact request body to ask
/// whether BFCL's own conditions are order-independent. Two literals would let
/// the equality gate certify a generation regime BFCL never runs — it shipped
/// that way once, at 512 against BFCL's 1024.
pub const MAX_NEW_TOKENS: usize = 1024;

impl Bfcl {
    /// The sample count THIS run should produce: the variant's pinned draw, or
    /// this shard's slice of it.
    ///
    /// Derived with `draw::shard_take` over the same plan the loader uses, so a
    /// shard cannot disagree with the rows it was handed. Restating a per-shard
    /// number here is the `default_floor` mistake this file already documents.
    fn expected_samples(&self) -> Option<usize> {
        let whole = self.variant.expected_samples()?;
        match self.shard {
            None => Some(whole),
            Some(sh) => {
                let totals = draw::reference_subset_totals();
                let plan = draw::plan(&self.variant.spec(), &totals);
                Some(
                    plan.iter()
                        .map(|(_, take)| draw::shard_take(*take, sh.index, sh.count))
                        .sum(),
                )
            }
        }
    }

    pub fn new(variant: Variant) -> Self {
        Self::maybe_sharded(variant, None)
    }

    /// One shard of `variant`'s draw. The group aggregates the members.
    pub fn sharded(variant: Variant, index: usize, count: usize) -> Self {
        Self::maybe_sharded(variant, Some(dataset::Shard { index, count }))
    }

    fn maybe_sharded(variant: Variant, shard: Option<dataset::Shard>) -> Self {
        Self {
            variant,
            shard,
            handle: None,
            phase: Phase::Provision,
            artifacts: None,
            samples: Vec::new(),
            cursor: 0,
            responses: Vec::new(),
            responses_path: None,
            scores: None,
            spec: variant.spec(),
            max_new_tokens: MAX_NEW_TOKENS,
            temperature: 0.0,
            request_timeout: Duration::from_secs(600),
            started: None,
            tool_call_samples: 0,
            transport_errors: 0,
            target_model: None,
            baseline_mins: report::BaselineMins::default(),
        }
    }

    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }
}

impl Plugin for Bfcl {
    fn metadata(&self) -> &'static PluginMetadata {
        self.variant.metadata()
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.started = Some(Instant::now());
        self.target_model = Some(handle.target().model.clone());
        self.handle = Some(handle.clone());
        async move {
            let artifacts = provision::ensure(handle.artifacts(), &handle).await?;
            self.artifacts = Some(artifacts);
            Ok(())
        }
    }
}

impl Benchmark for Bfcl {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        self.variant.descriptor()
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        let v = self.variant;
        let mut specs = vec![
            ParamSpec::new(
                "non_live_pct",
                "non_live %",
                "Percentage of each non_live subset to draw. 62 is the golden MLPerf draw.",
                ParamKind::Float {
                    min: 0.01,
                    max: 100.0,
                },
                ParamValue::Float(v.default_pct("non_live")),
            ),
            ParamSpec::new(
                "live_pct",
                "live %",
                "Percentage of each live subset to draw. 10 is the golden MLPerf draw.",
                ParamKind::Float {
                    min: 0.01,
                    max: 100.0,
                },
                ParamValue::Float(v.default_pct("live")),
            ),
            ParamSpec::new(
                "hallucination_pct",
                "hallucination %",
                "Percentage of each hallucination subset to draw. 10 is the golden MLPerf draw.",
                ParamKind::Float {
                    min: 0.01,
                    max: 100.0,
                },
                ParamValue::Float(v.default_pct("hallucination")),
            ),
            ParamSpec::new(
                "subset_floor",
                "Subset floor",
                "Subsets this small are taken whole, so tiny ones do not collapse to noise.",
                ParamKind::Int {
                    min: 0,
                    max: 10_000,
                },
                ParamValue::Int(v.default_floor() as i64),
            ),
            ParamSpec::new(
                "max_new_tokens",
                "Max new tokens",
                "Output budget per sample. The MLPerf config uses 1024.",
                ParamKind::Int {
                    min: 16,
                    max: 32_768,
                },
                ParamValue::Int(MAX_NEW_TOKENS as i64),
            ),
            ParamSpec::new(
                "temperature",
                "Temperature",
                "0 is greedy, which is what every recorded BFCL score was produced with.",
                ParamKind::Float { min: 0.0, max: 2.0 },
                ParamValue::Float(0.0),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Request timeout",
                "Seconds before one sample is abandoned and scored as no tool call.",
                ParamKind::Int { min: 10, max: 3600 },
                ParamValue::Int(600),
            ),
            ParamSpec::new(
                "shard",
                "Shard",
                "Run one Nth of the draw, as `index/count` with a 0-based index \
                 (`2/7`; the whole draw is `0/1`). `inherit` runs whatever this \
                 benchmark id already selects: the whole draw, or — for a \
                 registered shard member like `bfcl-subset-a` — its own quarter.",
                ParamKind::Text,
                // ★ THE DEFAULT IS `inherit`, NOT `0/1`. The registered shard
                // members set their slice in the constructor, and a default
                // that meant "the whole draw" would overwrite it on every
                // `configure` — which the TUI and every gate run call — turning
                // all four members into four copies of the whole draw. The
                // union would be 4 x 995 rows with every sample scored four
                // times. See
                // `a_shard_member_keeps_its_slice_under_default_parameters`.
                //
                // A word rather than an empty string because `ParamKind::Text`
                // refuses an empty value outright, so "" could never be the
                // default that reaches `configure`.
                ParamValue::Text(INHERIT_SHARD.to_string()),
            ),
        ];
        specs.extend(report::BaselineMins::specs());
        specs
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        let floor = values.usize("subset_floor")?;
        self.spec = DrawSpec {
            categories: draw::CATEGORIES.iter().map(|c| c.to_string()).collect(),
            category_pct: [
                ("non_live".to_string(), values.float("non_live_pct")?),
                ("live".to_string(), values.float("live_pct")?),
                (
                    "hallucination".to_string(),
                    values.float("hallucination_pct")?,
                ),
            ]
            .into_iter()
            .collect(),
            subset_floor: (floor > 0).then_some(floor),
        };
        // `inherit` leaves `self.shard` exactly as the constructor set it.
        let shard = values.text("shard")?;
        if shard.trim() != INHERIT_SHARD {
            self.shard = Some(dataset::Shard::parse(shard.trim()).map_err(anyhow::Error::msg)?);
        }
        self.max_new_tokens = values.usize("max_new_tokens")?;
        self.temperature = values.float("temperature")?;
        self.request_timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        self.baseline_mins = report::BaselineMins::from_values(values)?;
        self.phase = Phase::Provision;
        self.cursor = 0;
        self.responses.clear();
        self.samples.clear();
        self.scores = None;
        self.tool_call_samples = 0;
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;
        match self.phase {
            Phase::Provision => {
                http::probe(handle.target(), Duration::from_secs(10))
                    .await
                    .context("endpoint probe failed — check the target URL and port")?;
                let artifacts = self
                    .artifacts
                    .clone()
                    .context("artifacts were not provisioned")?;
                self.samples = dataset::load_shard(&artifacts.dataset, &self.spec, self.shard)?;
                self.phase = Phase::Generate;
                let n = self.samples.len();
                let mut frame = BenchmarkResult::running("draw", self.elapsed())
                    .with_progress(0, n as u64)
                    .log_line(LogLine::info(format!(
                        "drew {n} samples across {} subsets",
                        self.samples
                            .iter()
                            .map(|s| s.subset.as_str())
                            .collect::<std::collections::BTreeSet<_>>()
                            .len()
                    )));
                // The single most useful thing to say up front: whether this
                // is the MLPerf-comparable draw or something else.
                if let Some(want) = self.expected_samples()
                    && n != want
                {
                    let of = match self.shard {
                        None => String::new(),
                        Some(sh) => format!(" (shard {} of {})", sh.index, sh.count),
                    };
                    frame = frame.log_line(LogLine::warn(format!(
                        "n={n}, not the pinned {want}{of} — this run is NOT comparable to \
                         this draw's baseline"
                    )));
                }
                Ok(frame)
            }
            Phase::Generate => {
                let total = self.samples.len() as u64;
                if self.cursor >= self.samples.len() {
                    self.phase = Phase::Score;
                    handle.status("scoring with bfcl-eval");
                    return Ok(BenchmarkResult::running("scoring", self.elapsed())
                        .with_progress(total, total)
                        .with_summary(self.summary())
                        .log_line(LogLine::info(format!(
                            "generated {} responses; running the AST checker",
                            self.responses.len()
                        ))));
                }
                let subset = self.samples[self.cursor].subset.clone();
                self.generate_one().await?;
                let done = self.cursor as u64;
                handle.progress(done, total);
                handle.status(format!("{subset} · {done}/{total}"));
                Ok(BenchmarkResult::running(subset, self.elapsed())
                    .with_progress(done, total)
                    .with_summary(self.summary()))
            }
            Phase::Score => {
                let scores = self.score().await?;
                if scores.unmatched_responses > 0 {
                    handle.warn(format!(
                        "{} response(s) did not match a dataset sample",
                        scores.unmatched_responses
                    ));
                }
                self.scores = Some(scores);
                self.phase = Phase::Done;
                let total = self.samples.len() as u64;
                let mut frame = BenchmarkResult {
                    status: RunStatus::Completed,
                    ..BenchmarkResult::running("done", self.elapsed())
                }
                .with_progress(total, total)
                .with_summary(self.summary())
                .with_metrics(self.metrics())
                .with_verdict(self.verdict());
                if let Some(t) = self.table() {
                    frame = frame.with_table(t);
                }
                Ok(frame)
            }
            Phase::Done => bail!("next() was called after the run finished"),
        }
    }
}

#[path = "variant.rs"]
mod variant_impl;

#[path = "exec.rs"]
mod exec;

#[cfg(test)]
#[path = "bfcl_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "bfcl_shard_tests.rs"]
mod shard_tests;

#[cfg(test)]
#[path = "aggregate_tests.rs"]
mod aggregate_tests;
