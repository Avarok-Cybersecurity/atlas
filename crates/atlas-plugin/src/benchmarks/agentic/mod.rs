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

pub mod agent;
pub mod score;

use std::future::Future;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::benchmarks::one_line;
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{
    BenchmarkResult, Cell, CellStyle, Column, LogLine, ResultTable, RunStatus, Stat, Verdict,
};

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

const SUMMARY: &str = "N agentic runs: build a working Axum server, then verify it";
pub const METADATA: PluginMetadata = PluginMetadata::atlas(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "agentic-webserver",
    name: "Agentic Webserver Test",
    summary: SUMMARY,
    detail: "Runs the flagship agentic task N times: the model writes a Rust Axum ping/pong \
             server, tests it, runs it and tears it down, using bash/write_file/read_file tools \
             in a fresh sandbox. Each run is scored on OUTCOME (the scorer builds it and gets a \
             'pong') and on PROCESS (did the agent do all six things the prompt asked?), plus \
             wall time. RUNS MODEL-AUTHORED SHELL inside the sandbox directory.",
    duration_hint: "~5 min per iteration",
    updated: "2026-07-31",
    needs_confirmation: true,
    // Gate A. The webserver_ok thresholds (10/10 and Σ wall ≤ 1300 s) were
    // measured on the 35B MoE flagship and mean nothing against another
    // checkpoint. FP8 and NVFP4 are both the same family and both valid.
    intended_for: Some(crate::benchmark::ModelExpectation {
        families: &["qwen3.6-35b-a3b"],
        note: "Gate A is defined on the 35B MoE flagship (Qwen3.6-35B-A3B, FP8 or NVFP4). \
               The dense 27B is a different gate (C2/D) with different thresholds, so a \
               run here would produce numbers that compare to nothing.",
    }),
    ctor: || Box::new(AgenticWebserver::default()),
};

#[derive(Default)]
struct IterationRow {
    index: usize,
    wall_s: f64,
    webserver_ok: bool,
    directions: score::Directions,
    turns: usize,
    tool_calls: usize,
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
        handle.status(format!("run {index}: scoring"));
        let web = score::webserver_test(
            &sandbox,
            self.cargo_target_dir.as_deref(),
            self.build_timeout,
            self.serve_timeout,
        )
        .await;
        // The agent's own wall time is the measurement; the scorer's build and
        // probe are harness cost and must not be charged to the model.
        let wall_s = started.elapsed().as_secs_f64();
        let directions = score::followed_directions(&transcript.commands, &sandbox);

        let mut note = web.error.clone();
        if transcript.hit_turn_cap {
            note = format!("turn cap ({}) reached; {note}", self.max_turns);
        }
        Ok(IterationRow {
            index,
            wall_s,
            webserver_ok: web.webserver_ok,
            directions,
            turns: transcript.turns,
            tool_calls: transcript.tool_calls,
            note: one_line(note),
        })
    }

    fn table(&self) -> ResultTable {
        let mut t = ResultTable::new(
            "ITERATIONS",
            vec![
                Column::right("Run", 4),
                Column::right("wall s", 8),
                Column::left("ws_ok", 6),
                Column::right("steps", 6),
                Column::right("turns", 6),
                Column::right("tools", 6),
                Column::left("note", 40),
            ],
        );
        for r in &self.rows {
            t.push(vec![
                Cell::new(r.index.to_string()),
                Cell::new(format!("{:.1}", r.wall_s)),
                Cell::styled(
                    if r.webserver_ok { "pass" } else { "FAIL" },
                    if r.webserver_ok {
                        CellStyle::Good
                    } else {
                        CellStyle::Bad
                    },
                ),
                Cell::styled(
                    format!("{}/6", r.directions.met()),
                    if r.directions.overall() {
                        CellStyle::Good
                    } else {
                        CellStyle::Warn
                    },
                ),
                Cell::new(r.turns.to_string()),
                Cell::new(r.tool_calls.to_string()),
                Cell::styled(r.note.clone(), CellStyle::Dim),
            ]);
        }
        t
    }

    fn summary(&self) -> Vec<Stat> {
        let ok = self.rows.iter().filter(|r| r.webserver_ok).count();
        let fd = self.rows.iter().filter(|r| r.directions.overall()).count();
        let n = self.rows.len();
        vec![
            Stat::new("webserver_ok", format!("{ok}/{n}"), "").with_style(if n > 0 && ok == n {
                CellStyle::Good
            } else {
                CellStyle::Warn
            }),
            Stat::new("followed_directions", format!("{fd}/{n}"), "").with_style(
                if n > 0 && fd == n {
                    CellStyle::Good
                } else {
                    CellStyle::Warn
                },
            ),
            Stat::new("Σ wall", format!("{:.0}", self.total_wall()), "s").with_style(
                if self.total_wall() <= self.wall_budget_s {
                    CellStyle::Good
                } else {
                    CellStyle::Warn
                },
            ),
        ]
    }

    fn verdict(&self) -> Verdict {
        let n = self.rows.len();
        let ok = self.rows.iter().filter(|r| r.webserver_ok).count();
        let fd = self.rows.iter().filter(|r| r.directions.overall()).count();
        let wall = self.total_wall();
        let mut failures = Vec::new();
        if ok < n {
            failures.push(format!("webserver_ok {ok}/{n}"));
        }
        if fd < n {
            failures.push(format!("followed_directions {fd}/{n}"));
        }
        if wall > self.wall_budget_s {
            failures.push(format!("Σwall {wall:.0}s > {:.0}s", self.wall_budget_s));
        }
        if failures.is_empty() {
            Verdict::pass(format!(
                "{ok}/{n} webserver_ok · {fd}/{n} followed_directions · Σwall {wall:.0}s ≤ {:.0}s",
                self.wall_budget_s
            ))
        } else {
            Verdict::fail(failures.join(" · "))
        }
    }
}

impl Plugin for AgenticWebserver {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.started = Some(Instant::now());
        let store = handle.artifacts().clone();
        self.handle = Some(handle);
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
            // Share one warm target dir across every iteration, matching the
            // harness's ATLAS_WARM_TARGET_DIR. Without it each iteration
            // cold-compiles the axum/tokio tree and the wall time measures
            // dependency compilation instead of the model.
            let warm = match std::env::var_os("ATLAS_WARM_TARGET_DIR") {
                Some(p) => PathBuf::from(p),
                None => store.root().join("cargo-warm-target"),
            };
            std::fs::create_dir_all(&warm)?;
            self.cargo_target_dir = Some(warm);
            Ok(())
        }
    }
}

impl Benchmark for AgenticWebserver {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new(
                "iterations",
                "Iterations",
                "How many independent agent runs. The gate tier is 10; use 1 for a smoke test.",
                ParamKind::Int { min: 1, max: 50 },
                ParamValue::Int(10),
            ),
            ParamSpec::new(
                "wall_budget_s",
                "Σ wall budget",
                "Total agent seconds across all iterations before the gate fails.",
                ParamKind::Float {
                    min: 1.0,
                    max: 100_000.0,
                },
                ParamValue::Float(1300.0),
            ),
            ParamSpec::new(
                "max_turns",
                "Max turns",
                "Tool-calling rounds per iteration before the agent is stopped.",
                ParamKind::Int { min: 1, max: 200 },
                ParamValue::Int(40),
            ),
            ParamSpec::new(
                "command_timeout_s",
                "Command timeout",
                "Seconds a single agent shell command may run before it is killed.",
                ParamKind::Int { min: 5, max: 3600 },
                ParamValue::Int(180),
            ),
            ParamSpec::new(
                "build_timeout_s",
                "Scorer build timeout",
                "Seconds the scorer's cargo build may take. A cold dependency tree is slow.",
                ParamKind::Int { min: 30, max: 3600 },
                ParamValue::Int(600),
            ),
            ParamSpec::new(
                "serve_timeout_s",
                "Ping timeout",
                "Seconds to wait for /ping to answer 'pong' after the server is started.",
                ParamKind::Int { min: 5, max: 300 },
                ParamValue::Int(30),
            ),
            ParamSpec::new(
                "max_tokens",
                "Max tokens per turn",
                "Output budget for one model turn.",
                ParamKind::Int {
                    min: 256,
                    max: 32_768,
                },
                ParamValue::Int(8192),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Request timeout",
                "Seconds before a single model call is abandoned.",
                ParamKind::Int { min: 10, max: 3600 },
                ParamValue::Int(900),
            ),
        ]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        self.iterations = values.usize("iterations")?;
        self.wall_budget_s = values.float("wall_budget_s")?;
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
