// SPDX-License-Identifier: AGPL-3.0-only

//! Driving a benchmark without a terminal.
//!
//! The dashboard pumps a run from its render tick. A script has no tick, so
//! this is the same pump as a blocking call: start, drain until finished,
//! record. Both paths end at [`crate::history::save`], so a run means the same
//! thing and lands in the same place however it was started.
//!
//! Lives here rather than in the server because this crate has no GPU
//! dependency — the whole loop is exercised against a mock endpoint in
//! `cargo test -p atlas-plugin`, on any machine.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;

use crate::benchmark::BenchmarkDescriptor;
use crate::coherence::CoherencePolicy;
use crate::executor::{BenchmarkExecutor, ExecutorMessage};
use crate::history::{self, RunRecord, RunSource};
use crate::params::ParamValues;
use crate::plugin::{PluginEvent, TargetEndpoint};
use crate::result::{BenchmarkResult, VerdictKind};

/// How the driver behaves, as opposed to what it runs.
#[derive(Clone, Debug)]
pub struct HeadlessOptions {
    /// Drain cadence. 250 ms keeps a CI log readable while still landing a
    /// Ctrl-C inside a quarter second.
    pub poll: Duration,
    /// Write the run to `~/.atlas/runs`.
    pub save: bool,
    pub source: RunSource,
    /// Recorded on the run so a result can be traced to the build.
    pub atlas_version: String,
    /// Whether to require a coherent endpoint before measuring.
    pub coherence: CoherencePolicy,
}

impl HeadlessOptions {
    pub fn cli(atlas_version: impl Into<String>) -> Self {
        Self {
            poll: Duration::from_millis(250),
            save: true,
            source: RunSource::Cli,
            atlas_version: atlas_version.into(),
            coherence: CoherencePolicy::Probe,
        }
    }
}

/// What to run, and against what.
pub struct RunRequest {
    pub descriptor: &'static BenchmarkDescriptor,
    pub values: ParamValues,
    pub target: TargetEndpoint,
    pub options: HeadlessOptions,
}

/// What a driver does with the run as it happens.
///
/// Every method defaults to nothing, so a caller that only wants the final
/// record implements none of them.
pub trait RunReporter {
    fn started(&mut self, _request: &RunRequest) {}
    fn event(&mut self, _event: &PluginEvent) {}
    fn frame(&mut self, _frame: &BenchmarkResult) {}
}

/// A reporter that says nothing. For tests and `--quiet`.
pub struct SilentReporter;
impl RunReporter for SilentReporter {}

#[derive(Debug)]
pub struct RunOutcome {
    pub record: RunRecord,
    /// Where it was written, when `options.save`.
    pub saved_to: Option<PathBuf>,
    pub cancelled: bool,
}

impl RunOutcome {
    /// `0` clean · `1` the run itself failed or was cancelled · `2` the run was
    /// fine and the gate said no.
    ///
    /// Those are distinct because a script has to tell "the harness broke" from
    /// "the model missed the bar" — collapsing them makes a red build ambiguous.
    pub fn exit_code(&self) -> i32 {
        if self.cancelled
            || !matches!(
                self.record.frame.status,
                crate::result::RunStatus::Completed
            )
        {
            return 1;
        }
        match self.record.verdict_kind() {
            Some(VerdictKind::Fail) => 2,
            _ => 0,
        }
    }
}

/// Run to completion, pumping the executor's channels. **Blocks the calling
/// thread** — from async, use `spawn_blocking`.
///
/// `should_cancel` is polled each tick so a caller can wire a signal handler
/// without this module knowing about signals.
pub fn run_blocking(
    executor: &BenchmarkExecutor,
    request: RunRequest,
    reporter: &mut dyn RunReporter,
    should_cancel: &dyn Fn() -> bool,
) -> Result<RunOutcome> {
    // Fail on a bad parameter before a run directory is created or a single
    // request is issued.
    let specs = request.descriptor.build().parameters();
    request.values.validate_against(&specs)?;

    reporter.started(&request);
    let run = executor.start(
        request.descriptor,
        request.values.clone(),
        request.target.clone(),
        request.options.coherence,
    );

    let mut terminal: Option<BenchmarkResult> = None;
    loop {
        for message in run.drain() {
            dispatch(message, reporter, &mut terminal);
        }
        if should_cancel() && !run.is_cancelled() {
            run.cancel();
            reporter.event(&PluginEvent::Status(
                "cancelling — stopping after the request in flight".into(),
            ));
        }
        if run.is_finished() {
            break;
        }
        std::thread::sleep(request.options.poll);
    }

    // `finished` is stored AFTER the last send, so seeing it true does not mean
    // the last frame is dequeuable yet. Drain until a drain comes back empty
    // rather than draining once and hoping.
    loop {
        let messages = run.drain();
        if messages.is_empty() {
            break;
        }
        for message in messages {
            dispatch(message, reporter, &mut terminal);
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let cancelled = run.is_cancelled();
    // A run that ends with no terminal frame still gets recorded: a missing run
    // is harder to diagnose than a recorded failure.
    let frame = terminal.unwrap_or_else(|| {
        BenchmarkResult::failed(
            "run",
            "the run ended without a terminal frame",
            Duration::ZERO,
        )
    });

    let mut record = RunRecord::new(
        request.descriptor,
        &request.values,
        &request.target,
        request.options.source,
        &request.options.atlas_version,
        frame,
    );
    let saved_to = if request.options.save {
        Some(history::save(executor.artifacts(), &mut record)?)
    } else {
        None
    };

    Ok(RunOutcome {
        record,
        saved_to,
        cancelled,
    })
}

fn dispatch(
    message: ExecutorMessage,
    reporter: &mut dyn RunReporter,
    terminal: &mut Option<BenchmarkResult>,
) {
    match message {
        ExecutorMessage::Event(e) => reporter.event(&e),
        ExecutorMessage::Frame(f) => {
            reporter.frame(&f);
            if f.status.is_terminal() {
                *terminal = Some(*f);
            }
        }
    }
}

#[cfg(test)]
#[path = "headless_tests.rs"]
mod tests;
