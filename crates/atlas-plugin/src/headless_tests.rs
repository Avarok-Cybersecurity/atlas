// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the headless driver.
//!
//! Uses the real registry's `concurrency-sweep` with an unreachable endpoint:
//! its first `next()` probes the target and terminates on failure, which
//! exercises the whole loop — start, drain, terminal frame, record, save —
//! in milliseconds and without a socket.

use super::*;
use crate::artifacts::ArtifactStore;
use crate::registry;
use crate::result::RunStatus;

struct Dir(std::path::PathBuf);
impl Dir {
    fn new() -> Self {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let p = std::env::temp_dir().join(format!(
            "atlas-headless-{n}-{:?}",
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&p).expect("scratch dir");
        Self(p)
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Counts what the driver reported, so a test can assert the caller saw the run
/// rather than only its result.
#[derive(Default)]
struct Counting {
    started: usize,
    events: usize,
    frames: usize,
    terminal_frames: usize,
    glows: Vec<bool>,
    statuses: Vec<String>,
    terminal: Option<BenchmarkResult>,
}
impl RunReporter for Counting {
    fn started(&mut self, _r: &RunRequest) {
        self.started += 1;
    }
    fn event(&mut self, e: &PluginEvent) {
        self.events += 1;
        if let PluginEvent::Glow(on) = e {
            self.glows.push(*on);
        }
        if let PluginEvent::Status(status) = e {
            self.statuses.push(status.clone());
        }
    }
    fn frame(&mut self, f: &BenchmarkResult) {
        self.frames += 1;
        if f.status.is_terminal() {
            self.terminal_frames += 1;
            self.terminal = Some(f.clone());
        }
    }
}

/// A request pointed at a closed port, so the probe fails fast.
fn request(options: HeadlessOptions) -> RunRequest {
    let descriptor = registry::find("concurrency-sweep").expect("registered");
    let specs = descriptor.build().parameters();
    let values = ParamValues::from_overrides(
        &specs,
        [
            ("concurrencies", "1"),
            ("isls", "128"),
            ("warmup", "0"),
            ("osl", "8"),
        ],
    )
    .expect("overrides parse");
    RunRequest {
        descriptor,
        values,
        // Port 1 is reserved and never listening.
        target: TargetEndpoint::new("http://127.0.0.1:1", "unreachable"),
        options,
    }
}

fn run(dir: &Dir, options: HeadlessOptions, reporter: &mut dyn RunReporter) -> RunOutcome {
    let store = ArtifactStore::with_root(&dir.0);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let executor = BenchmarkExecutor::new(rt.handle().clone(), store);
    run_blocking(&executor, request(options), reporter, &|| false).expect("drives")
}

#[test]
fn a_run_produces_a_record_and_writes_it() {
    let dir = Dir::new();
    let mut reporter = Counting::default();
    let outcome = run(
        &dir,
        HeadlessOptions::cli("1.0.0-beta-preview"),
        &mut reporter,
    );

    assert_eq!(reporter.started, 1);
    assert!(reporter.frames >= 1, "the caller saw the run happen");
    assert_eq!(reporter.terminal_frames, 1, "exactly one terminal frame");
    assert_eq!(reporter.glows, [true, false], "run lifecycle events");

    let path = outcome.saved_to.as_ref().expect("saved");
    assert!(path.exists(), "the file is really on disk");
    assert_eq!(outcome.record.benchmark_id, "concurrency-sweep");
    assert_eq!(outcome.record.source, RunSource::Cli);
    assert_eq!(outcome.record.atlas_version, "1.0.0-beta-preview");
    assert!(!outcome.record.run_id.is_empty(), "stamped by save");
    let reported = reporter.terminal.as_ref().expect("reported terminal frame");
    assert_eq!(reported.status, outcome.record.frame.status);
    assert_eq!(reported.phase, outcome.record.frame.phase);
    assert_eq!(
        reported.verdict.as_ref().map(|v| v.reason.as_str()),
        outcome
            .record
            .frame
            .verdict
            .as_ref()
            .map(|v| v.reason.as_str()),
        "the persisted record keeps the terminal frame the caller saw"
    );

    // Re-read through the public reader: writer and reader must agree.
    let store = ArtifactStore::with_root(&dir.0);
    let back = crate::history::load(&store, "concurrency-sweep");
    assert_eq!(back.len(), 1);
    assert_eq!(back[0].run_id, outcome.record.run_id);
}

#[test]
fn an_unreachable_endpoint_is_a_recorded_failure_not_a_lost_run() {
    // The endpoint is down, so the run fails — but it must still be recorded.
    // A run that vanishes is harder to diagnose than one that says it failed.
    let dir = Dir::new();
    let outcome = run(&dir, HeadlessOptions::cli("v"), &mut SilentReporter);
    assert_eq!(outcome.record.frame.status, RunStatus::Failed);
    assert!(outcome.saved_to.is_some(), "a failure is still history");
    assert_eq!(outcome.exit_code(), 1, "the harness could not measure");
}

#[test]
fn no_save_leaves_nothing_on_disk() {
    let dir = Dir::new();
    let mut options = HeadlessOptions::cli("v");
    options.save = false;
    let outcome = run(&dir, options, &mut SilentReporter);

    assert!(outcome.saved_to.is_none());
    let store = ArtifactStore::with_root(&dir.0);
    assert!(crate::history::load(&store, "concurrency-sweep").is_empty());
}

#[test]
fn cancellation_is_recorded_and_returns_a_harness_failure() {
    let dir = Dir::new();
    let store = ArtifactStore::with_root(&dir.0);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let executor = BenchmarkExecutor::new(rt.handle().clone(), store);
    let mut reporter = Counting::default();

    let outcome = run_blocking(
        &executor,
        request(HeadlessOptions::cli("v")),
        &mut reporter,
        &|| true,
    )
    .expect("drives cancellation");

    assert!(outcome.cancelled, "the cancellation reaches the run handle");
    assert_eq!(outcome.exit_code(), 1, "a cancelled result is not usable");
    assert!(
        reporter
            .statuses
            .iter()
            .any(|s| s == "cancelling — stopping after the request in flight"),
        "the caller sees cancellation progress: {:?}",
        reporter.statuses
    );
}

#[test]
fn an_invalid_parameter_fails_before_anything_runs() {
    // Validation must happen before a run directory exists, so a typo does not
    // leave a half-run behind.
    let dir = Dir::new();
    let store = ArtifactStore::with_root(&dir.0);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let executor = BenchmarkExecutor::new(rt.handle().clone(), store.clone());

    let descriptor = registry::find("concurrency-sweep").expect("registered");
    let mut values = ParamValues::defaults(&descriptor.build().parameters());
    values.set("osl", crate::params::ParamValue::Int(-5)); // below the spec's min

    let mut reporter = Counting::default();
    let err = run_blocking(
        &executor,
        RunRequest {
            descriptor,
            values,
            target: TargetEndpoint::new("http://127.0.0.1:1", "m"),
            options: HeadlessOptions::cli("v"),
        },
        &mut reporter,
        &|| false,
    )
    .expect_err("rejected");
    assert!(
        err.to_string().contains("Output tokens"),
        "says which: {err}"
    );
    assert!(crate::history::load(&store, "concurrency-sweep").is_empty());
    assert_eq!(reporter.started, 0, "validation precedes start reporting");
    assert_eq!(reporter.events, 0, "validation precedes executor events");
    assert_eq!(reporter.frames, 0, "validation precedes benchmark frames");
}

#[test]
fn exit_codes_separate_a_broken_harness_from_a_failed_gate() {
    use crate::result::Verdict;
    let descriptor = registry::find("concurrency-sweep").expect("registered");
    let values = ParamValues::defaults(&descriptor.build().parameters());
    let target = TargetEndpoint::new("http://127.0.0.1:1", "m");
    let mk = |frame| RunOutcome {
        record: RunRecord::new(descriptor, &values, &target, RunSource::Cli, "v", frame),
        saved_to: None,
        cancelled: false,
    };

    let mut ok = BenchmarkResult::completed("done", std::time::Duration::ZERO);
    ok.verdict = Some(Verdict::pass("under the bar"));
    assert_eq!(mk(ok).exit_code(), 0);

    let mut failed_gate = BenchmarkResult::completed("done", std::time::Duration::ZERO);
    failed_gate.verdict = Some(Verdict::fail("over the bar"));
    assert_eq!(
        mk(failed_gate).exit_code(),
        2,
        "the run worked; the gate said no"
    );

    let broken = BenchmarkResult::failed("run", "endpoint refused", std::time::Duration::ZERO);
    assert_eq!(mk(broken).exit_code(), 1, "the harness could not measure");

    let mut cancelled = mk(BenchmarkResult::completed(
        "done",
        std::time::Duration::ZERO,
    ));
    cancelled.cancelled = true;
    assert_eq!(
        cancelled.exit_code(),
        1,
        "a cancelled measurement is unusable"
    );
}
