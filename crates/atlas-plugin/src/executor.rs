// SPDX-License-Identifier: AGPL-3.0-only

//! [`drive`] — the single loop that turns a benchmark into a stream of frames —
//! and [`BenchmarkExecutor`], which runs the whole lifecycle on the server's
//! tokio runtime and publishes to the render thread.
//!
//! Threading contract (the same one `tui::chat` uses): the benchmark runs as a
//! tokio task; frames and plugin events cross to the render thread over
//! `std::sync::mpsc`, which the event loop drains once per tick. The render
//! thread never awaits, and a slow benchmark can never stall a redraw.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

use anyhow::Result;
use futures::{Stream, StreamExt};

use crate::artifacts::ArtifactStore;
use crate::benchmark::BenchmarkDescriptor;
use crate::coherence::CoherencePolicy;
use crate::dynamic::DynBenchmark;
use crate::params::ParamValues;
use crate::plugin::{PluginEvent, PluginHandle, TargetEndpoint};
use crate::result::{BenchmarkResult, LogLine, RunStatus};

/// Drive `bench` by calling `next()` until it reports a terminal status or
/// errors. This is the ONLY driver: `Benchmark::run`'s default body calls it,
/// and so does the executor, so a benchmark cannot behave differently depending
/// on how it was started.
pub fn drive(bench: &mut dyn DynBenchmark) -> impl Stream<Item = Result<BenchmarkResult>> + '_ {
    futures::stream::unfold(Some(bench), |state| async move {
        let bench = state?;
        match bench.next().await {
            Ok(frame) => {
                let finished = frame.status.is_terminal();
                Some((Ok(frame), if finished { None } else { Some(bench) }))
            }
            // An `Err` is itself terminal — a benchmark that cannot take a step
            // cannot take the next one either.
            Err(e) => Some((Err(e), None)),
        }
    })
}

/// What the render thread receives from a run.
///
/// `Frame` is boxed: a terminal [`BenchmarkResult`] is several times larger
/// than an [`PluginEvent`], and without indirection every event in flight
/// pays the frame's full size.
pub enum ExecutorMessage {
    Event(PluginEvent),
    Frame(Box<BenchmarkResult>),
}

/// The render thread's control surface for one in-flight run.
pub struct RunHandle {
    events: Receiver<PluginEvent>,
    frames: Receiver<BenchmarkResult>,
    cancel: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
}

impl RunHandle {
    /// Non-blocking drain, events before frames so a "provisioning…" status
    /// lands before the frame that reports it done.
    pub fn drain(&self) -> Vec<ExecutorMessage> {
        let mut out: Vec<ExecutorMessage> =
            self.events.try_iter().map(ExecutorMessage::Event).collect();
        out.extend(
            self.frames
                .try_iter()
                .map(|f| ExecutorMessage::Frame(Box::new(f))),
        );
        out
    }

    /// Ask the run to stop. The benchmark observes this at its own await points
    /// (`PluginHandle::check_cancelled`), and the driver stops between frames.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// True once `cleanup()` has returned — the run owns no resources after it.
    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Relaxed)
    }
}

/// Starts benchmark runs on an existing tokio runtime.
#[derive(Clone)]
pub struct BenchmarkExecutor {
    runtime: tokio::runtime::Handle,
    artifacts: ArtifactStore,
    /// Run ids handed to `PluginHandle`. One executor per dashboard, so
    /// counting here is what makes ids unique across the process without a
    /// global — see `PluginHandle::run_id`.
    next_run_id: Arc<std::sync::atomic::AtomicU64>,
}

impl BenchmarkExecutor {
    pub fn new(runtime: tokio::runtime::Handle, artifacts: ArtifactStore) -> Self {
        Self {
            runtime,
            artifacts,
            next_run_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
        }
    }

    /// The runtime this executor drives work on, so a caller can run a short
    /// async check (an endpoint pre-flight) on the same one rather than
    /// building a second.
    pub fn runtime(&self) -> &tokio::runtime::Handle {
        &self.runtime
    }

    pub fn artifacts(&self) -> &ArtifactStore {
        &self.artifacts
    }

    /// Build, load, configure, drive, clean up. Every exit path runs
    /// `cleanup()` — including cancellation and a `load()` that failed, so a
    /// half-provisioned run cannot leak a child process or a sandbox dir.
    pub fn start(
        &self,
        descriptor: &'static BenchmarkDescriptor,
        values: ParamValues,
        target: TargetEndpoint,
        coherence: CoherencePolicy,
    ) -> RunHandle {
        let (event_tx, event_rx) = channel();
        let (frame_tx, frame_rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let handle = PluginHandle::new(
            self.next_run_id
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            target,
            self.artifacts.clone(),
            event_tx.clone(),
            cancel.clone(),
        );
        let task = RunTask {
            descriptor,
            values,
            coherence,
            handle,
            events: event_tx,
            frames: frame_tx,
            cancel: cancel.clone(),
            finished: finished.clone(),
        };
        self.runtime.spawn(task.execute());
        RunHandle {
            events: event_rx,
            frames: frame_rx,
            cancel,
            finished,
        }
    }
}

/// The probe asks for at most 32 tokens. A model that cannot answer "2+2" in
/// 30 s is not going to complete a sweep either.
const COHERENCE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

struct RunTask {
    descriptor: &'static BenchmarkDescriptor,
    values: ParamValues,
    handle: PluginHandle,
    events: Sender<PluginEvent>,
    frames: Sender<BenchmarkResult>,
    cancel: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
    coherence: CoherencePolicy,
}

impl RunTask {
    /// Ask the endpoint two known-answer questions before committing to a run.
    ///
    /// **Advisory only.** A model that answers oddly is worth saying out loud,
    /// but it is not grounds to refuse: a latency sweep against a base
    /// checkpoint is a real thing to measure, and so is a benchmark aimed at a
    /// model it was not written for. The warning goes in the run log, where it
    /// stays attached to the numbers it qualifies.
    async fn probe_coherence(&self) {
        if self.coherence == CoherencePolicy::Skip {
            return;
        }
        self.handle.status("checking the endpoint".to_string());
        let report = crate::coherence::probe_for(
            self.handle.target(),
            self.descriptor.intended_for,
            COHERENCE_TIMEOUT,
        )
        .await;
        match report.concern(self.handle.target()) {
            Some(concern) => self.handle.warn(concern),
            None => {
                for a in &report.answers {
                    self.handle
                        .info(format!("endpoint check {}: {:?}", a.label, a.answer.trim()));
                }
            }
        }
    }

    async fn execute(self) {
        let started = Instant::now();
        self.handle.set_glow(true);
        let mut bench = self.descriptor.build();

        // Phase 0 — probe, then load, then configure. All pre-run steps, so a
        // failure is reported as a terminal frame rather than a silent dead pane.
        //
        // The coherence probe runs FIRST, before `load()`. That ordering is the
        // whole point: BFCL's load builds a venv, pip-installs a pinned
        // bfcl-eval and materializes a dataset before it ever contacts the
        // endpoint, so a wrong --model used to cost minutes of setup and then
        // hours of uniformly-failing samples. Two short completions up front
        // turn that into a two-second error.
        self.probe_coherence().await;
        let setup = async {
            bench.load(self.handle.clone()).await?;
            bench.configure(&self.values)
        }
        .await;
        if let Err(e) = setup {
            self.emit_failed("setup", e, started.elapsed());
            self.teardown(bench.as_mut()).await;
            return;
        }

        // Phase 1 — drive.
        {
            let stream = drive(bench.as_mut());
            futures::pin_mut!(stream);
            while let Some(item) = stream.next().await {
                match item {
                    Ok(frame) => {
                        if self.frames.send(frame).is_err() {
                            break; // TUI gone
                        }
                    }
                    Err(e) => {
                        self.emit_failed("run", e, started.elapsed());
                        break;
                    }
                }
                if self.cancel.load(Ordering::Relaxed) {
                    let _ = self.frames.send(BenchmarkResult::failed(
                        "cancelled",
                        "cancelled by user",
                        started.elapsed(),
                    ));
                    break;
                }
            }
        }

        self.teardown(bench.as_mut()).await;
    }

    fn emit_failed(&self, phase: &str, error: anyhow::Error, elapsed: Duration) {
        // `{:#}` unrolls the anyhow context chain, which is where the actionable
        // half of these messages lives ("pip install failed — needs network").
        let _ = self.frames.send(BenchmarkResult::failed(
            phase,
            format!("{error:#}"),
            elapsed,
        ));
    }

    async fn teardown(&self, bench: &mut dyn DynBenchmark) {
        if let Err(e) = bench.cleanup().await {
            let _ = self
                .events
                .send(PluginEvent::Log(LogLine::warn(format!("cleanup: {e:#}"))));
        }
        let _ = self.events.send(PluginEvent::Glow(false));
        self.finished.store(true, Ordering::Relaxed);
    }
}

/// True when a frame ends the run. Exposed so the TUI can stop its spinner on
/// the same rule the driver stops on.
pub fn is_final(frame: &BenchmarkResult) -> bool {
    frame.status.is_terminal()
}

/// Convenience for benchmarks whose terminal frame is just "everything above".
pub fn terminal_status(ok: bool) -> RunStatus {
    if ok {
        RunStatus::Completed
    } else {
        RunStatus::Failed
    }
}

#[cfg(test)]
#[path = "executor_tests.rs"]
mod tests;
