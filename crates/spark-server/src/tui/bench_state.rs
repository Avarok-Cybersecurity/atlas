// SPDX-License-Identifier: AGPL-3.0-only

//! Benchmarks-section state: what is selected, what the parameters are set to,
//! and what the running benchmark has reported so far.
//!
//! The section is a three-step flow — **Suite → Parameters → Run** — plus a
//! History pane over `~/.atlas/runs`. Nothing here awaits: the executor owns
//! the tokio side and this drains its channels once per tick, exactly like
//! [`crate::tui::chat`].

use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

use atlas_plugin::{
    BenchmarkDescriptor, BenchmarkResult, ExecutorMessage, LogLine, ParamSpec, ParamValues,
    PluginEvent, RunHandle, RunStatus, TargetEndpoint, registry,
};

/// How many log lines the run pane keeps.
const LOG_CAPACITY: usize = 500;

/// Which step of the flow the Suite subsection is showing.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum View {
    #[default]
    List,
    Params,
    Run,
}

#[derive(Default)]
pub struct BenchState {
    /// Index into [`registry::all`].
    pub selected: usize,
    pub view: View,
    /// Provenance of the selected benchmark. Cached at selection time — the
    /// detail pane redraws at 10 Hz and must not construct a plugin per frame.
    meta: Option<&'static atlas_plugin::PluginMetadata>,
    /// Schema of the selected benchmark, and the values being edited.
    pub specs: Vec<ParamSpec>,
    pub values: ParamValues,
    /// One edit buffer per row. Rows past `specs.len()` are the target fields,
    /// so the endpoint is edited with the same keys as everything else.
    pub edit: Vec<String>,
    pub row: usize,
    pub editing: bool,
    /// Per-field validation messages, shown under the field.
    pub errors: BTreeMap<String, String>,
    pub target: TargetEndpoint,
    /// True once the operator has typed a model into the target field.
    ///
    /// Until then the target follows whatever the server is actually serving.
    /// It was captured once at dashboard start from the boot argv, which is
    /// empty for `spark serve` with no model — so runs recorded a blank model
    /// — and went stale the moment a model was loaded from the Library or a
    /// request swapped one in. With --auto-swap that is worse than cosmetic: a
    /// benchmark request carrying the old name is exactly the trigger that
    /// swaps the server back, mid-run.
    pub target_model_pinned: bool,
    /// Set for a benchmark whose descriptor demands confirmation.
    pub confirm_open: bool,

    executor: Option<atlas_plugin::BenchmarkExecutor>,
    run: Option<RunHandle>,
    /// The benchmark the in-flight (or last) run belongs to.
    pub running_id: Option<&'static str>,
    /// The descriptor of the run in flight — a record needs its name, not
    /// just its id.
    running_descriptor: Option<&'static atlas_plugin::BenchmarkDescriptor>,
    /// Whether to require a coherent endpoint before measuring. Defaults to
    /// requiring it; `p` in the form toggles.
    pub coherence: atlas_plugin::CoherencePolicy,
    pub frame: Option<BenchmarkResult>,
    pub log: VecDeque<LogLine>,
    pub status: String,
    pub progress: Option<(u64, u64)>,
    pub glow: bool,
    pub started: Option<Instant>,
    pub table_scroll: usize,

    pub history: Vec<atlas_plugin::RunRecord>,
    pub history_row: usize,
    history_loaded: bool,
    /// Set once a terminal frame has been persisted, so it is written once.
    persisted: bool,
    /// The endpoint check between pressing START and the run beginning.
    pub preflight: Option<crate::tui::bench_preflight::Preflight>,
}

impl BenchState {
    /// Wire in the executor and the default target. Called once at TUI start,
    /// when the tokio handle exists.
    /// Point the target at the model that is actually serving.
    ///
    /// A no-op once the operator has typed one in: benchmarking a different
    /// endpoint on purpose is a real thing to want.
    pub fn follow_live_model(&mut self, live: &str) {
        if self.target_model_pinned || live.is_empty() || self.target.model == live {
            return;
        }
        self.target = TargetEndpoint::new(self.target.base_url.clone(), live);
    }

    pub fn attach(&mut self, executor: atlas_plugin::BenchmarkExecutor, target: TargetEndpoint) {
        self.executor = Some(executor);
        self.target = target;
        self.select(0);
    }

    pub fn descriptor(&self) -> Option<&'static BenchmarkDescriptor> {
        registry::all().get(self.selected).copied()
    }

    /// True when a text buffer owns the keyboard — the app must not treat
    /// digits as section jumps while a value is being typed.
    pub fn is_editing(&self) -> bool {
        self.editing
    }

    pub fn is_running(&self) -> bool {
        self.run.as_ref().is_some_and(|r| !r.is_finished())
    }

    /// Load a benchmark's schema into the form. Defaults come from the specs,
    /// so the form and the benchmark can never disagree about them.
    pub fn select(&mut self, index: usize) {
        let all = registry::all();
        if all.is_empty() {
            return;
        }
        self.selected = index.min(all.len() - 1);
        let Some(descriptor) = self.descriptor() else {
            return;
        };
        let bench = descriptor.build();
        self.meta = Some(bench.metadata());
        self.specs = bench.parameters();
        self.values = ParamValues::defaults(&self.specs);
        self.edit = self
            .specs
            .iter()
            .map(|s| s.default.to_edit_string())
            .chain([self.target.base_url.clone(), self.target.model.clone()])
            .collect();
        self.errors.clear();
        self.row = 0;
        self.editing = false;
    }

    /// Provenance of the selected benchmark.
    pub fn plugin_metadata(&self) -> &'static atlas_plugin::PluginMetadata {
        // A benchmark is always selected once `attach` has run; the fallback
        // keeps the renderer total rather than making it handle an Option.
        self.meta.unwrap_or(&FALLBACK_METADATA)
    }

    /// Total form rows: one per parameter, then the two target fields.
    pub fn row_count(&self) -> usize {
        self.specs.len() + 2
    }

    /// Label/hint for a row, whether it is a parameter or a target field.
    pub fn row_meta(&self, row: usize) -> (&str, &str, String) {
        match self.specs.get(row) {
            Some(spec) => (spec.label, spec.help, spec.kind.domain_hint()),
            None if row == self.specs.len() => (
                "Endpoint URL",
                "Which server to benchmark. Defaults to this one.",
                "http://host:port".to_string(),
            ),
            _ => (
                "Model",
                "The `model` field sent in each request.",
                "model id".to_string(),
            ),
        }
    }

    /// Parse and store the row's edit buffer. Errors stay attached to the field
    /// rather than becoming a run failure.
    pub fn commit_row(&mut self, row: usize) {
        let raw = self.edit.get(row).cloned().unwrap_or_default();
        match self.specs.get(row) {
            Some(spec) => {
                let key = spec.key.to_string();
                match spec.kind.parse(&raw) {
                    Ok(value) => {
                        self.values.set(key.clone(), value);
                        self.errors.remove(&key);
                    }
                    Err(e) => {
                        self.errors.insert(key, e.to_string());
                    }
                }
            }
            None if row == self.specs.len() => {
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    self.errors
                        .insert("__url".into(), "must not be empty".into());
                } else {
                    self.target = TargetEndpoint::new(trimmed, self.target.model.clone());
                    self.errors.remove("__url");
                    // `new` normalises the trailing slash; show what will be used.
                    self.edit[row] = self.target.base_url.clone();
                }
            }
            _ => {
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    self.errors
                        .insert("__model".into(), "must not be empty".into());
                } else {
                    self.target = TargetEndpoint::new(self.target.base_url.clone(), trimmed);
                    self.target_model_pinned = true;
                    self.errors.remove("__model");
                }
            }
        }
    }

    /// Validation message for a row, if it has one.
    pub fn row_error(&self, row: usize) -> Option<&str> {
        let key = match self.specs.get(row) {
            Some(spec) => spec.key,
            None if row == self.specs.len() => "__url",
            _ => "__model",
        };
        self.errors.get(key).map(String::as_str)
    }

    /// Check the endpoint, then start. Refuses up front for the same reasons
    /// `start` does, so a doomed run never reaches the check.
    ///
    /// With the probe switched off this is `start` — asking nothing is the
    /// point of that setting.
    pub fn begin_start(&mut self) -> Result<(), String> {
        if self.is_running() {
            return Err("a benchmark is already running".into());
        }
        if !self.errors.is_empty() {
            return Err(format!("{} field(s) need fixing", self.errors.len()));
        }
        if self.coherence == atlas_plugin::CoherencePolicy::Skip {
            return self.start();
        }
        let executor = self
            .executor
            .as_ref()
            .ok_or("the benchmark executor is unavailable")?;
        let expectation = self.descriptor().and_then(|d| d.intended_for);
        self.preflight = Some(crate::tui::bench_preflight::Preflight::begin(
            executor.runtime(),
            self.target.clone(),
            expectation,
            std::time::Duration::from_secs(30),
        ));
        Ok(())
    }

    /// Drain the pre-flight. Called once per tick; starts the run itself when
    /// the endpoint checks out, so a clean check is invisible beyond a flicker.
    pub fn poll_preflight(&mut self) {
        let Some(pre) = self.preflight.as_mut() else {
            return;
        };
        // Some(true) starts now; a concern keeps the modal up for the user to
        // decide, and None means the check is still in flight.
        if pre.poll(&self.target) == Some(true) {
            self.preflight = None;
            let _ = self.start();
        }
    }

    /// Proceed past a reported concern.
    pub fn accept_preflight(&mut self) -> Result<(), String> {
        self.preflight = None;
        self.start()
    }

    /// Abandon the run and go back to the form.
    pub fn cancel_preflight(&mut self) {
        self.preflight = None;
    }

    /// Start the selected benchmark. Refuses while a run is in flight and while
    /// any field is invalid — an invalid form is the user's to fix, not
    /// something to discover three hours in.
    pub fn start(&mut self) -> Result<(), String> {
        if self.is_running() {
            return Err("a benchmark is already running".into());
        }
        if !self.errors.is_empty() {
            return Err(format!("{} field(s) need fixing", self.errors.len()));
        }
        let descriptor = self.descriptor().ok_or("no benchmark selected")?;
        let executor = self
            .executor
            .as_ref()
            .ok_or("the benchmark executor is unavailable")?;
        self.log.clear();
        self.frame = None;
        self.status = "starting".into();
        self.progress = None;
        self.table_scroll = 0;
        self.persisted = false;
        self.started = Some(Instant::now());
        self.running_id = Some(descriptor.id);
        self.running_descriptor = Some(descriptor);
        self.run = Some(executor.start(
            descriptor,
            self.values.clone(),
            self.target.clone(),
            self.coherence,
        ));
        self.view = View::Run;
        Ok(())
    }

    pub fn cancel(&mut self) {
        if let Some(run) = &self.run {
            run.cancel();
            self.status = "cancelling — the server keeps serving".into();
        }
    }

    /// Drain the executor's channels. Called once per tick from the event loop.
    pub fn pump(&mut self) {
        // Drain first, then release the borrow: the handlers below mutate
        // `self`, and holding `&self.run` across them would not compile.
        let Some((messages, finished)) = self
            .run
            .as_ref()
            .map(|run| (run.drain(), run.is_finished()))
        else {
            return;
        };
        for message in messages {
            match message {
                ExecutorMessage::Event(PluginEvent::Log(line)) => self.push_log(line),
                ExecutorMessage::Event(PluginEvent::Status(text)) => self.status = text,
                ExecutorMessage::Event(PluginEvent::Progress { done, total }) => {
                    self.progress = Some((done, total));
                }
                ExecutorMessage::Event(PluginEvent::Glow(on)) => self.glow = on,
                ExecutorMessage::Frame(frame) => {
                    for line in &frame.log {
                        self.push_log(line.clone());
                    }
                    if let Some(p) = frame.progress {
                        self.progress = Some(p);
                    }
                    if frame.status.is_terminal() {
                        self.status = match frame.status {
                            RunStatus::Completed => "completed".into(),
                            _ => "failed".into(),
                        };
                        self.persist(&frame);
                    } else {
                        self.status = frame.phase.clone();
                    }
                    self.frame = Some(*frame);
                }
            }
        }
        // The glow follows the executor's own signal, but a run that died
        // without emitting one must not leave the ring pulsing forever.
        if finished {
            self.glow = false;
        }
    }

    fn push_log(&mut self, line: LogLine) {
        if self.log.len() >= LOG_CAPACITY {
            self.log.pop_front();
        }
        self.log.push_back(line);
    }

    /// Record the terminal frame, once.
    ///
    /// Goes through `atlas_plugin::history`, the same writer the CLI uses, so a
    /// run started here and a run started headlessly land in one store with one
    /// format — and both carry their parameters and target, which the old
    /// frame-only write did not.
    fn persist(&mut self, frame: &BenchmarkResult) {
        if self.persisted {
            return;
        }
        self.persisted = true;
        let (Some(executor), Some(descriptor)) = (&self.executor, self.running_descriptor) else {
            return;
        };
        let mut record = atlas_plugin::RunRecord::new(
            descriptor,
            &self.values,
            &self.target,
            atlas_plugin::RunSource::Tui,
            crate::cli::ATLAS_VERSION,
            frame.clone(),
        );
        if let Err(e) = atlas_plugin::history::save(executor.artifacts(), &mut record) {
            tracing::warn!("could not record this run: {e:#}");
        }
        // The next visit to History re-reads the directory rather than trying
        // to keep an in-memory list in sync with the filesystem.
        self.history_loaded = false;
    }

    /// Populate the History pane. Lazy and re-run after each persisted frame.
    ///
    /// Sorted newest-first across ALL benchmarks rather than grouped by
    /// benchmark: every row already prints its own age and id, and a single
    /// chronological list is what "what ran recently" actually asks for.
    pub fn load_history(&mut self) {
        if self.history_loaded {
            return;
        }
        self.history_loaded = true;
        self.history = match &self.executor {
            Some(executor) => atlas_plugin::history::load_all(executor.artifacts()),
            None => Vec::new(),
        };
        self.history_row = self.history_row.min(self.history.len().saturating_sub(1));
    }

    pub fn elapsed_text(&self) -> String {
        let secs = self.started.map(|s| s.elapsed().as_secs()).unwrap_or(0);
        format!(
            "{:02}:{:02}:{:02}",
            secs / 3600,
            (secs / 60) % 60,
            secs % 60
        )
    }
}

/// Shown only before `attach` has selected anything.
///
/// STATIC, DELIBERATELY — compile-time data. A `const`-constructed literal
/// with no interior mutability and nothing derived from a model; it is a
/// string table that happens to need a stable address to be borrowed from.
const FALLBACK_METADATA: atlas_plugin::PluginMetadata =
    atlas_plugin::PluginMetadata::atlas("no benchmark selected");

#[cfg(test)]
#[path = "bench_state_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "bench_state_more_tests.rs"]
mod more_tests;
