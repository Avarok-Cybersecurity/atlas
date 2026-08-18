// SPDX-License-Identifier: AGPL-3.0-only

//! Agentic Webserver Test — the flagship end-to-end agentic probe.
//!
//! N iterations of one task: *write a Rust Axum project with a ping/pong
//! endpoint, add tests, run them, run the server, curl it, tear it down.* Each
//! iteration gets a fresh sandbox; afterwards the scorer builds and runs what
//! the agent left behind and asks `/ping` for a `pong`.
//!
//! **This is a different measurement from the recorded Gate A history.** Those
//! numbers come from driving the `opencode` CLI; this drives our own agent loop
//! against the same endpoint, so it is a different client and starts its own
//! baseline series. The *task* and the *pass criteria* are identical — the
//! prompt is verbatim from `bench/fp8_dgx2_drift/harness/run_tier.sh` and the
//! scoring is a port of `score_run.py`.
//!
//! It executes model-authored shell; see [`agent`] for the containment.
//!
//! ## Canonical serve recipe (Gate A)
//!
//! The recorded 10/10 Gate A history (321ws Σ951s, 307A Σ978s, fixA Σ1158s)
//! was measured against THIS serve configuration. A Gate A run is only
//! comparable to that band when served exactly like this (no docker wrapper —
//! run the binary directly):
//!
//! The canonical serve command, and why `--mtp-gate force` belongs in it,
//! are documented on `DESCRIPTOR` in `descriptors.rs` — beside the thresholds
//! they justify, rather than duplicated here where they would drift.
//!
//!
//! Why `--mtp-gate force` is a DETERMINISM pin — and why the non-neutrality
//! it works around is a BUG with an open fix — is documented on `DESCRIPTOR`
//! in `descriptors.rs`, beside the thresholds it protects.
//!
//! The client
//! side of this gate is pinned to the bone (temp 0.0, seed 0, constant prompt,
//! normalized tool output) and none of that helps while the SERVER is choosing
//! numeric paths by stopwatch.
//!
//! Proof it was really happening: two runs of the SAME binary (8b7de2638),
//! same box, back to back — iteration 9 of the failing run left
//! `/tmp/agent_server.log` built `--release`, while iteration 9 of the passing
//! run wrote `/tmp/server.log` built dev. Identical inputs, divergent
//! trajectories. Roughly one wandering trajectory in ten ends without
//! evidencing its final directive, which is the 9/10.
//!
//! The 2026-07-22 campaign already declared this pin mandatory for gates. This
//! gate never adopted it. The alternative — widening the threshold to tolerate
//! the flip — was explicitly rejected: it would convert a determinism bug into
//! permanent noise the gate can never see through again.
//!
//! Two further pins that must not drift (see `descriptors.rs`).

pub mod agent;
mod params;
pub mod preflight;
mod render;
pub mod score;
mod verdict;
pub mod warm;

use std::future::Future;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::benchmarks::one_line;
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamSpec, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{BenchmarkResult, LogLine, RunStatus, Verdict};

/// The task, verbatim from `run_tier.sh`. Changing a word changes the
/// benchmark, so it is a single constant and not assembled from pieces.
pub const PROMPT: &str = "Please create a pure rust Axum project here in the current working \
directory. Just have a ping/pong endpoint. The server MUST bind to the port from the \
ATLAS_HARNESS_PORT env var (default 3001) — use `let port: u16 = \
std::env::var(\"ATLAS_HARNESS_PORT\").unwrap_or_else(|_| \"3001\".to_string()).parse().unwrap();` \
then bind to `0.0.0.0:port`. Add tests, run them and prove all tests pass, then run the server and \
use curl to prove it works. Whenever you run the server or any long-lived process in the \
background, always start it detached with its output redirected to a file (for example `setsid \
cargo run > /tmp/server.log 2>&1 &`) so your shell never blocks waiting on it, and wrap any \
command that might hang, such as curl checks or process kills, in a short `timeout 15`. Finally, \
tear down the server by killing whatever is listening on its port rather than guessing the process \
name, always wrapped in a short timeout so it can never stall your shell, for example `timeout 5 \
fuser -k ${ATLAS_HARNESS_PORT:-3001}/tcp 2>/dev/null || true`.";

mod descriptors;
pub use descriptors::{DESCRIPTOR, METADATA};

#[derive(Default)]
struct IterationRow {
    index: usize,
    /// Iteration wall INCLUDING the scorer's build and probe — the blowup
    /// detector's number.
    wall_s: f64,
    /// The agent's own wall, scorer excluded. The speed bound divides THIS by
    /// turns: the scorer is a per-ITERATION cost, so charging it to a per-TURN
    /// ratio adds a term that shrinks as turns grow — it would make a long
    /// trajectory look faster per turn purely because it amortised the build.
    agent_wall_s: f64,
    webserver_ok: bool,
    directions: score::Directions,
    turns: usize,
    tool_calls: usize,
    completion_tokens: usize,
    note: String,
}

#[derive(Default)]
pub struct AgenticWebserver {
    handle: Option<PluginHandle>,
    iterations: usize,
    max_turns: usize,
    command_timeout: Duration,
    request_timeout: Duration,
    build_timeout: Duration,
    serve_timeout: Duration,
    max_tokens: usize,
    wall_budget_s: f64,
    s_per_turn_budget: f64,
    cursor: usize,
    rows: Vec<IterationRow>,
    sandbox_root: Option<PathBuf>,
    cargo_target_dir: Option<PathBuf>,
    started: Option<Instant>,
    probed: bool,
}

impl AgenticWebserver {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    fn total_wall(&self) -> f64 {
        self.rows.iter().map(|r| r.wall_s).sum()
    }

    /// The agent's own seconds, scorer excluded — the speed numerator.
    fn total_agent_wall(&self) -> f64 {
        self.rows.iter().map(|r| r.agent_wall_s).sum()
    }

    fn total_turns(&self) -> usize {
        verdict::total_turns(&self.rows)
    }

    fn total_tool_calls(&self) -> usize {
        self.rows.iter().map(|r| r.tool_calls).sum()
    }

    fn total_completion_tokens(&self) -> usize {
        self.rows.iter().map(|r| r.completion_tokens).sum()
    }

    /// `None` when the tier took no turns — see [`verdict::seconds_per_turn`].
    fn seconds_per_turn(&self) -> Option<f64> {
        verdict::seconds_per_turn(self.total_agent_wall(), self.total_turns())
    }

    async fn run_iteration(&mut self, index: usize) -> Result<IterationRow> {
        let handle = self.handle()?.clone();
        let root = self
            .sandbox_root
            .clone()
            .context("sandbox root was not prepared")?;
        let sandbox = root.join(format!("run-{index:02}"));
        // A fresh directory per iteration: leftovers from the previous run
        // would let a later agent "pass" on code it never wrote.
        let _ = std::fs::remove_dir_all(&sandbox);
        std::fs::create_dir_all(&sandbox)
            .with_context(|| format!("creating sandbox {}", sandbox.display()))?;

        let cfg = agent::AgentConfig {
            sandbox: sandbox.clone(),
            max_turns: self.max_turns,
            command_timeout: self.command_timeout,
            request_timeout: self.request_timeout,
            max_tokens: self.max_tokens,
            cargo_target_dir: self.cargo_target_dir.clone(),
        };

        let started = Instant::now();
        let transcript = agent::run_task(&handle, &cfg, PROMPT).await?;
        // Taken HERE, before the scorer runs. The comment below used to claim
        // the scorer was not charged to the model while the only wall recorded
        // was taken after it; both numbers now exist and each says what it is.
        let agent_wall_s = started.elapsed().as_secs_f64();
        handle.status(format!("run {index}: scoring"));
        let web = score::webserver_test(
            &sandbox,
            self.cargo_target_dir.as_deref(),
            self.build_timeout,
            self.serve_timeout,
        )
        .await;
        // Total, scorer included: `sum_wall_s` is a blowup detector, and a
        // scorer build that runs away is exactly a blowup worth detecting.
        let wall_s = started.elapsed().as_secs_f64();
        let directions = score::followed_directions(&transcript.commands, &sandbox);

        let mut note = web.error.clone();
        if transcript.hit_turn_cap {
            note = format!("turn cap ({}) reached; {note}", self.max_turns);
        }
        // ★ NAME the unevidenced directives. `directions.met()` renders "5/6"
        // and nothing recorded WHICH one — so a 9/10 was undiagnosable once the
        // next run truncated the trajectory. The names were in `steps` the
        // whole time; only the count was ever surfaced.
        let missing = directions.missing();
        if !missing.is_empty() {
            note = format!("missing: {}; {note}", missing.join(", "));
        }
        Ok(IterationRow {
            index,
            wall_s,
            agent_wall_s,
            webserver_ok: web.webserver_ok,
            directions,
            turns: transcript.turns,
            tool_calls: transcript.tool_calls,
            completion_tokens: transcript.completion_tokens,
            note: one_line(note),
        })
    }

    /// Raw gate numbers for `--pull-request-gate` (same source the summary
    /// tiles and the verdict read from — the three cannot disagree).
    fn metrics(&self) -> std::collections::BTreeMap<String, f64> {
        let n = self.rows.len();
        let mut m = std::collections::BTreeMap::new();
        m.insert("iterations".to_string(), n as f64);
        m.insert(
            "webserver_ok".to_string(),
            self.rows.iter().filter(|r| r.webserver_ok).count() as f64,
        );
        m.insert(
            "followed_directions".to_string(),
            self.rows.iter().filter(|r| r.directions.overall()).count() as f64,
        );
        m.insert("sum_wall_s".to_string(), self.total_wall());
        m.insert("sum_agent_wall_s".to_string(), self.total_agent_wall());
        m.insert("sum_turns".to_string(), self.total_turns() as f64);
        m.insert("sum_tool_calls".to_string(), self.total_tool_calls() as f64);
        m.insert(
            "sum_completion_tokens".to_string(),
            self.total_completion_tokens() as f64,
        );
        // Absent, not 0.0, for a zero-turn tier: `check_record` compares
        // numbers, and a 0.0 here would read as the best speed ever recorded.
        if let Some(spt) = self.seconds_per_turn() {
            m.insert("s_per_turn".to_string(), spt);
        }
        // Recorded, never gated — see the `decode_tps` note on the ParamSpec.
        // Tokens are the denominator a speed claim should ultimately use, but
        // no variant has a measured bound yet, and inventing one is the exact
        // mistake this change exists to undo.
        if self.total_agent_wall() > 0.0 {
            m.insert(
                "decode_tps".to_string(),
                self.total_completion_tokens() as f64 / self.total_agent_wall(),
            );
        }

        m.extend(score::per_step_tallies(
            &self.rows.iter().map(|r| &r.directions).collect::<Vec<_>>(),
        ));
        m
    }

    fn verdict(&self) -> Verdict {
        verdict::verdict(
            &self.rows,
            self.total_wall(),
            self.total_agent_wall(),
            self.wall_budget_s,
            self.s_per_turn_budget,
        )
    }
}

impl Plugin for AgenticWebserver {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.started = Some(Instant::now());
        let store = handle.artifacts().clone();
        self.handle = Some(handle.clone());
        async move {
            // `cargo` has to exist or nothing can be scored — say so now rather
            // than after the model has spent five minutes writing code.
            crate::python::run(std::path::Path::new("cargo"), &["--version"], None)
                .await
                .context(
                    "cargo is not on PATH — this benchmark builds the code the model writes",
                )?;
            let root = store.runs_dir(DESCRIPTOR.id)?.join("sandbox");
            std::fs::create_dir_all(&root)?;
            self.sandbox_root = Some(root);
            // Pre-warm (not merely allocate) the shared target dir the agent AND
            // the scorer build in. Creating an empty dir is not the harness's
            // behaviour: `run_tier.sh:75-96` warms BOTH profiles up front
            // because a tier drives `cargo test` 141× and the cold dep build
            // "was the entire 92s↔305s wall variance". See [`warm`].
            self.cargo_target_dir = Some(warm::prepare(&handle).await?);
            Ok(())
        }
    }
}

impl Benchmark for AgenticWebserver {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        params::parameters()
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        self.iterations = values.usize("iterations")?;
        self.wall_budget_s = values.float("wall_budget_s")?;
        self.s_per_turn_budget = values.float("s_per_turn_budget")?;
        self.max_turns = values.usize("max_turns")?;
        self.command_timeout = Duration::from_secs(values.usize("command_timeout_s")? as u64);
        self.build_timeout = Duration::from_secs(values.usize("build_timeout_s")? as u64);
        self.serve_timeout = Duration::from_secs(values.usize("serve_timeout_s")? as u64);
        self.max_tokens = values.usize("max_tokens")?;
        self.request_timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        self.cursor = 0;
        self.rows.clear();
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;
        let total = self.iterations as u64;

        if !self.probed {
            self.probed = true;
            http::probe(handle.target(), Duration::from_secs(10))
                .await
                .context("endpoint probe failed — check the target URL and port")?;
            // run_tier.sh halts on a bad 2+2 rather than after 25 min of tier:
            // a 200 from /v1/models proves a server is listening, not that the
            // checkpoint still decodes.
            preflight::sanity_check(&handle, Duration::from_secs(60)).await?;
            let root = self.sandbox_root.clone().context("no sandbox root")?;
            return Ok(BenchmarkResult::running("probe", self.elapsed())
                .with_progress(0, total)
                .log_line(LogLine::info(format!(
                    "{total} iteration(s) · sandbox {}",
                    root.display()
                )))
                .log_line(LogLine::warn(
                    "this benchmark executes model-authored shell inside the sandbox",
                )));
        }

        if self.cursor >= self.iterations {
            if self.rows.is_empty() {
                bail!("no iterations ran");
            }
            return Ok(BenchmarkResult {
                status: RunStatus::Completed,
                ..BenchmarkResult::running("done", self.elapsed())
            }
            .with_progress(total, total)
            .with_summary(self.summary())
            .with_table(self.table())
            .with_metrics(self.metrics())
            .with_verdict(self.verdict()));
        }

        let index = self.cursor;
        handle.status(format!("run {index}/{}", self.iterations));
        let row = self.run_iteration(index).await?;
        let line = LogLine::info(format!(
            "run {index}: {} · {}/6 steps · {:.1}s · {} turns{}",
            if row.webserver_ok {
                "webserver_ok"
            } else {
                "FAILED"
            },
            row.directions.met(),
            row.wall_s,
            row.turns,
            if row.note.is_empty() {
                String::new()
            } else {
                format!(" · {}", row.note)
            }
        ));
        self.rows.push(row);
        self.cursor += 1;
        handle.progress(self.cursor as u64, total);
        Ok(
            BenchmarkResult::running(format!("run {index}"), self.elapsed())
                .with_progress(self.cursor as u64, total)
                .with_summary(self.summary())
                .with_table(self.table())
                .log_line(line),
        )
    }

    /// Sandboxes are left in place on purpose — after a failed iteration the
    /// code the model wrote is the only evidence of why. They are wiped at the
    /// start of the next run of the same index, so they cannot accumulate.
    async fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[path = "agentic_tests.rs"]
mod tests;
