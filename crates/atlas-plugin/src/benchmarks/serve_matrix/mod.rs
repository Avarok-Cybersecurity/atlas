// SPDX-License-Identifier: AGPL-3.0-only

//! Serve Matrix — boot every checkpoint the box can serve, probe it, score it.
//!
//! Port of `tests/run_all_models.py` + `tests/gate_results.py`: one round per
//! model×quant, each booting the checkpoint, running the coherence / codegen /
//! tool-call / long-context / throughput probes, and recording a row. The
//! verdict mirrors `gate_results.py`'s bars, so the pane shows a real PASS/FAIL
//! rather than a wall of numbers.
//!
//! Four defects in the Python are fixed here rather than carried:
//!
//! 1. **False READY.** `wait_listening()` grepped the container log for
//!    `Listening on`, which matches `Listening on 127.0.0.1:8888` — the server
//!    binding loopback INSIDE a bridged namespace, unreachable from every
//!    probe. Readiness here is the endpoint ANSWERING, and the round carries a
//!    bar of its own for identity: `/v1/models` must name the checkpoint this
//!    round loaded. Atlas answers a completion under whatever model name it is
//!    sent, so a swap that failed and restored the previous model still passes
//!    every content probe — only the model list can see it, and only a bar can
//!    stop it becoming a green row under a checkpoint that never loaded.
//! 2. **Cross-checkout hardcodes.** `RESULTS_DIR`/`SUITE` were absolute paths
//!    into a DIFFERENT checkout still holding results from months earlier.
//!    Nothing here has a path: results are frames, and the baseline lives in
//!    the run's own `~/.atlas/runs/serve-matrix/`.
//! 3. **`repetition_penalty=1.05`** in the codegen probe — see `probes.rs`.
//! 4. **An inert tps bar.** `tests/baselines/` is empty repo-wide, so the
//!    Python gate has never compared throughput against anything. This one says
//!    "no baseline" where there is none, instead of implying a check passed.
//!
//! And the roster is **derived, not restated**: `ROUNDS` in the Python is a
//! Qwen3.5-era list of twelve checkpoints, none of which are in this box's
//! cache. Here the plan comes from the host's own view of the box — cached
//! weights ∩ compiled kernels — so it cannot go stale.

pub mod host;
pub mod plan;
pub mod probes;
pub mod report;
pub mod schema;
pub mod score;

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::benchmarks::{baseline, one_line, unique_prefix_tag};
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamSpec, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{BenchmarkResult, LogLine, RunStatus};

use host::{ServeHost, ServeOptions};
use plan::{Plan, Round};
use score::{Outcome, RoundResult, Signals};

/// Tokens the long-context probe needs BEYOND its filler: the needle sentence,
/// the question, the chat template, and the 32-token answer budget. Explicit
/// rather than implied, because "it fits" is the difference between a recall
/// measurement and a context overflow that looks like one.
const LONG_CTX_HEADROOM: usize = 256;

const SUMMARY: &str = "Boot every servable checkpoint, probe it, gate the matrix";
pub const METADATA: PluginMetadata = PluginMetadata::atlas(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "serve-matrix",
    name: "Serve Matrix",
    summary: SUMMARY,
    detail: "The release serve gate. For every checkpoint this box can actually serve — cached \
             weights intersected with the kernels this build compiled — it loads the model, \
             checks coherence, code generation, tool calling and long-context recall, measures \
             decode throughput, and moves on. Coverage is enforced: a checkpoint that is \
             planned and fails to boot is a FAIL, never a silent absence. Booting replaces \
             whatever model is currently serving; the one that was serving when the run started \
             is restored at the end.",
    duration_hint: "~5–10 min per checkpoint",
    updated: "2026-08-02",
    // Every round tears down the model the box is serving. That is a side
    // effect well beyond load on an endpoint, and the operator gets to say so.
    needs_confirmation: true,
    // The matrix is defined on whatever the box holds — that IS the measurement.
    intended_for: None,
    ctor: || Box::new(ServeMatrix::default()),
};

#[derive(Default)]
pub struct ServeMatrix {
    handle: Option<PluginHandle>,
    host: Option<Arc<dyn ServeHost>>,
    started: Option<Instant>,
    include: String,
    opts: Option<ServeOptions>,
    long_ctx_tokens: usize,
    tps_tokens: usize,
    probe_budget: usize,
    timeout: Duration,
    update_baselines: bool,
    /// The roster, classified. SSOT: the cursor indexes `plan.planned()` and
    /// so does the scoring, so there is no second list to fall out of step.
    plan: Plan,
    planned_built: bool,
    cursor: usize,
    results: Vec<RoundResult>,
    baseline: Option<baseline::Baseline>,
}

impl ServeMatrix {
    /// Construct against an explicit host. Tests use this; the registry's
    /// `ctor` cannot, since it takes no arguments.
    pub fn with_host(host: Arc<dyn ServeHost>) -> Self {
        Self {
            host: Some(host),
            ..Self::default()
        }
    }

    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn host(&self) -> Result<Arc<dyn ServeHost>> {
        self.host.clone().context(host::NO_HOST)
    }

    fn options(&self) -> Result<ServeOptions> {
        self.opts.context("benchmark was not configured")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    fn baseline_for(&self, label: &str) -> Option<f64> {
        self.baseline.as_ref()?.get(&format!("tps:{label}"))
    }

    /// Boot one checkpoint and probe it. Never returns `Err`: a round that
    /// fails is a ROW, not the end of the matrix — the Python orchestrator got
    /// that right and it is why a broken model does not cost the other eleven.
    async fn run_round(&self, round: &Round) -> Outcome {
        let Ok(handle) = self.handle().cloned() else {
            return Outcome::NotReached;
        };
        let (Ok(host), Ok(opts)) = (self.host(), self.options()) else {
            return Outcome::NotReached;
        };
        handle.status(format!("{} · booting", round.label()));
        let target = match host.serve(&round.model, opts).await {
            Ok(t) => t,
            Err(e) => return Outcome::BootFailed(one_line(format!("{e:#}"))),
        };
        // Assert the endpoint ANSWERS. `serve` promises this; asserting it here
        // is what makes a host implementation that only watched a log line
        // unable to manufacture a green round.
        if let Err(e) = http::probe(&target, Duration::from_secs(30)).await {
            return Outcome::BootFailed(format!(
                "came up but did not answer: {}",
                one_line(format!("{e:#}"))
            ));
        }

        let mut s = Signals::default();
        handle.status(format!("{} · coherence", round.label()));
        let (coherence, concern) = probes::coherence_probe(&target, self.timeout).await;
        s.coherence_pass = coherence.passed;
        s.coherence_total = coherence.total;
        s.identity = coherence.identity;
        if !concern.is_empty() {
            handle.warn(format!("{}: {concern}", round.label()));
        }

        handle.status(format!("{} · codegen", round.label()));
        s.codegen = probes::codegen_probe(&target, self.timeout, self.probe_budget).await;
        handle.status(format!("{} · tool call", round.label()));
        s.tool_call = probes::tool_call_probe(&target, self.timeout, self.probe_budget).await;

        if self.long_ctx_tokens > 0 {
            handle.status(format!("{} · long context", round.label()));
            let tag = unique_prefix_tag(&format!("sm-lc-{}", self.cursor), handle.run_id());
            s.long_ctx =
                probes::long_context_probe(&target, self.timeout, self.long_ctx_tokens, &tag).await;
        }

        handle.status(format!("{} · throughput", round.label()));
        let tag = unique_prefix_tag(&format!("sm-tps-{}", self.cursor), handle.run_id());
        let (tps, err) = probes::tps_probe(&target, self.timeout, self.tps_tokens, &tag).await;
        s.tps = tps;
        if let Some(e) = err {
            handle.warn(format!("{}: throughput probe failed: {e}", round.label()));
        }
        Outcome::Probed(Box::new(s))
    }

    /// Ask the host for the box's roster and classify it.
    fn build_plan(&mut self) -> Result<()> {
        let roster = self.host()?.roster().context("reading the box's roster")?;
        self.plan = Plan::build(&roster, &self.include);
        self.planned_built = true;
        Ok(())
    }

    /// Bless the measured throughput as the new bar.
    ///
    /// Deliberate and opt-in — a gate that blesses its own numbers every run
    /// cannot regress — and only from the rounds that PASSED. `baseline::save`
    /// says it plainly: numbers from a failed leg poison every later run, and
    /// a round whose model answered nonsense while still emitting tokens has
    /// exactly the tok/s you must not adopt as the floor.
    fn write_baselines(&self) -> Result<usize> {
        let handle = self.handle()?;
        let mut metrics = std::collections::BTreeMap::new();
        for r in self.results.iter().filter(|r| r.bars().is_empty()) {
            if let Some(tps) = r.signals().and_then(|s| s.tps).filter(|v| *v > 0.0) {
                metrics.insert(format!("tps:{}", r.label), (tps * 10.0).round() / 10.0);
            }
        }
        let written = metrics.len();
        baseline::save(
            handle.artifacts(),
            DESCRIPTOR.id,
            &handle.target().base_url,
            &handle.target().model,
            metrics,
        )?;
        Ok(written)
    }

    fn finish(&self) -> Result<BenchmarkResult> {
        let tally = score::tally(&self.plan, &self.results);
        let total = self.plan.planned_count() as u64;
        let mut frame = BenchmarkResult {
            status: RunStatus::Completed,
            ..BenchmarkResult::running("done", self.elapsed())
        }
        .with_progress(total, total)
        .with_summary(report::summary(&tally))
        .with_table(report::table(&self.plan, &self.results))
        .with_verdict(report::verdict(&tally, &self.plan));
        if self.update_baselines {
            match self.write_baselines() {
                Ok(n) => {
                    frame = frame.log_line(LogLine::info(format!(
                        "{n} throughput baseline(s) refreshed from the rounds that PASSED — later \
                         runs gate against these. Rounds below bar were not blessed."
                    )))
                }
                Err(e) => {
                    frame = frame.log_line(LogLine::warn(format!("baselines not written: {e:#}")))
                }
            }
        } else if self.baseline.is_none() {
            frame = frame.log_line(LogLine::warn(
                "no throughput baseline on this box: the tok/s column is liveness only, not a \
                 regression check. Re-run with `update baselines` on to record one.",
            ));
        }
        Ok(frame)
    }
}

impl Plugin for ServeMatrix {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        // An explicitly-supplied host (tests) wins; otherwise take the one the
        // server installed. Absent means NOT RUNNABLE here, and `load`'s
        // contract is that the message names what is missing.
        if self.host.is_none() {
            self.host = host::installed();
        }
        self.baseline = baseline::load(handle.artifacts(), DESCRIPTOR.id);
        self.handle = Some(handle);
        self.started = Some(Instant::now());
        let ok = self.host.is_some();
        async move {
            if ok {
                Ok(())
            } else {
                anyhow::bail!(host::NO_HOST)
            }
        }
    }
}

impl Benchmark for ServeMatrix {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        schema::specs()
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        let include = values.text("include")?.trim().to_string();
        // `all` is the schema's word for "no filter"; ParamKind::Text refuses an
        // empty field, so there has to be one.
        self.include = if include.eq_ignore_ascii_case("all") {
            String::new()
        } else {
            include
        };
        let max_seq_len = values.usize("max_seq_len")?;
        self.long_ctx_tokens = values.usize("long_ctx_tokens")?;
        if self.long_ctx_tokens + LONG_CTX_HEADROOM > max_seq_len {
            // Stated as a bound, not as `>=`: the probe's prompt is the filler
            // PLUS the needle sentence and the question, and the reply needs
            // room too. A prompt admitted at exactly max_seq_len overflows the
            // context and returns a `Fail` that reads like a recall regression.
            anyhow::bail!(
                "Long-context probe: {} tokens plus {LONG_CTX_HEADROOM} tokens of needle, \
                 question and answer does not fit in a {max_seq_len}-token context",
                self.long_ctx_tokens
            );
        }
        self.opts = Some(ServeOptions {
            max_seq_len,
            speculative: values.bool("speculative")?,
        });
        self.tps_tokens = values.usize("tps_tokens")?;
        self.probe_budget = values.usize("probe_budget")?;
        self.timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        self.update_baselines = values.bool("update_baselines")?;
        self.plan = Plan::default();
        self.planned_built = false;
        self.cursor = 0;
        self.results.clear();
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;

        if !self.planned_built {
            self.build_plan()?;
            let total = self.plan.planned_count() as u64;
            let skipped: Vec<String> = self
                .plan
                .skipped()
                .map(|(r, why)| format!("{} ({})", r.model, why.reason()))
                .collect();
            let mut frame = BenchmarkResult::running("plan", self.elapsed())
                .with_progress(0, total.max(1))
                .with_table(report::table(&self.plan, &self.results))
                .log_line(LogLine::info(format!(
                    "{total} checkpoint(s) planned · {} not runnable on this box · {} outside the \
                     filter",
                    self.plan.skipped().count(),
                    self.plan.excluded_count()
                )));
            if !skipped.is_empty() {
                frame = frame.log_line(LogLine::info(format!("skipping {}", skipped.join(", "))));
            }
            return Ok(frame);
        }

        let total = self.plan.planned_count();
        if self.cursor >= total {
            return self.finish();
        }

        // Indexing the plan's own iterator rather than a copy of it: one list,
        // so the cursor and the scoring cannot disagree about round four.
        let round = self
            .plan
            .planned()
            .nth(self.cursor)
            .cloned()
            .context("the plan changed under the cursor")?;
        let outcome = self.run_round(&round).await;
        let result = RoundResult {
            label: round.label(),
            baseline_tps: self.baseline_for(&round.label()),
            outcome,
        };
        let line = report::round_line(&result);
        let failed = !result.bars().is_empty();
        self.results.push(result);
        self.cursor += 1;
        let total = total as u64;
        handle.progress(self.cursor as u64, total);
        let tally = score::tally(&self.plan, &self.results);
        Ok(BenchmarkResult::running(round.label(), self.elapsed())
            .with_progress(self.cursor as u64, total)
            .with_summary(report::summary(&tally))
            .with_table(report::table(&self.plan, &self.results))
            .log_line(if failed {
                LogLine::error(line)
            } else {
                LogLine::info(line)
            }))
    }

    /// Put the box back where it was found — on every exit path, cancellation
    /// included. Without this a cancelled matrix leaves the operator's server
    /// on whatever checkpoint round four happened to load.
    async fn cleanup(&mut self) -> Result<()> {
        let Some(host) = self.host.clone() else {
            return Ok(());
        };
        if !self.planned_built {
            return Ok(());
        }
        host.restore().await.context("restoring the previous model")
    }
}

#[cfg(test)]
#[path = "serve_matrix_tests.rs"]
mod tests;
