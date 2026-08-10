// SPDX-License-Identifier: AGPL-3.0-only

//! [`Plugin`] — the general abstraction a [`crate::Benchmark`] specialises.
//!
//! A plugin's first step is [`Plugin::load`], which receives a [`PluginHandle`]:
//! the controlled seam onto the host terminal (status line, log, progress, the
//! run-glow) plus the artifact store and the endpoint it is pointed at. The
//! plugin never touches the TUI's state directly — it emits [`PluginEvent`]s
//! that the render thread drains on its own tick, so nothing a plugin does can
//! block or race the redraw.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;

use anyhow::{Result, bail};

use crate::artifacts::ArtifactStore;
use crate::metadata::PluginMetadata;
use crate::result::{LogLevel, LogLine};

/// The served endpoint a benchmark drives. Defaults to the server hosting the
/// TUI; the Benchmarks pane can retarget it (another box, or a reference
/// engine) without any benchmark knowing the difference.
///
/// `Default` is the empty, unusable endpoint — the "not attached yet" state a
/// UI holds before the runtime hands it a port. It is never given to a
/// benchmark: `probe` rejects it before any measurement starts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TargetEndpoint {
    /// Base URL with no trailing slash, e.g. `http://127.0.0.1:8888`.
    pub base_url: String,
    /// The `model` field sent in requests.
    pub model: String,
}

impl TargetEndpoint {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let base = base_url.into();
        Self {
            base_url: base.trim_end_matches('/').to_string(),
            model: model.into(),
        }
    }

    /// The local server this TUI is attached to.
    pub fn local(port: u16, model: impl Into<String>) -> Self {
        Self::new(format!("http://127.0.0.1:{port}"), model)
    }

    /// Split into `(host, port)` for a raw TCP connect.
    pub fn host_port(&self) -> Result<(String, u16)> {
        let rest = self.base_url.strip_prefix("http://").ok_or_else(|| {
            anyhow::anyhow!("only http:// targets are supported: {}", self.base_url)
        })?;
        let authority = rest.split('/').next().unwrap_or(rest);
        match authority.rsplit_once(':') {
            Some((host, port)) => {
                let port: u16 = port
                    .parse()
                    .map_err(|_| anyhow::anyhow!("bad port in {}", self.base_url))?;
                Ok((host.to_string(), port))
            }
            None => Ok((authority.to_string(), 80)),
        }
    }
}

/// A message from a running plugin to the terminal.
#[derive(Clone, Debug)]
pub enum PluginEvent {
    Log(LogLine),
    /// Replace the one-line status shown beside the spinner.
    Status(String),
    Progress {
        done: u64,
        total: u64,
    },
    /// Pulse the terminal's inner border while work is in flight.
    Glow(bool),
}

/// The plugin's view of its host.
#[derive(Clone)]
pub struct PluginHandle {
    /// Distinguishes this run from every other run in the process.
    ///
    /// The cold-TTFT gate depends on it: two requests sharing a prefix means
    /// the second hits the cache and the "cold" number is warm. Within one
    /// run the benchmark's own indices make its prompts unique; ACROSS runs
    /// nothing would, which is what this supplies — without a process-global
    /// counter, and without two runs of the same benchmark silently warming
    /// each other's cold leg.
    run_id: u64,
    target: TargetEndpoint,
    artifacts: ArtifactStore,
    events: Sender<PluginEvent>,
    cancel: Arc<AtomicBool>,
}

impl PluginHandle {
    /// This run's id — see the field doc for what it guarantees.
    pub fn run_id(&self) -> u64 {
        self.run_id
    }

    pub fn new(
        run_id: u64,
        target: TargetEndpoint,
        artifacts: ArtifactStore,
        events: Sender<PluginEvent>,
        cancel: Arc<AtomicBool>,
    ) -> Self {
        Self {
            run_id,
            target,
            artifacts,
            events,
            cancel,
        }
    }

    pub fn target(&self) -> &TargetEndpoint {
        &self.target
    }

    pub fn artifacts(&self) -> &ArtifactStore {
        &self.artifacts
    }

    /// Emit an event. A closed channel means the TUI is gone; that is not an
    /// error for the plugin, which is about to be cancelled anyway.
    fn emit(&self, event: PluginEvent) {
        let _ = self.events.send(event);
    }

    pub fn log(&self, level: LogLevel, text: impl Into<String>) {
        self.emit(PluginEvent::Log(LogLine {
            level,
            text: text.into(),
        }));
    }
    pub fn info(&self, text: impl Into<String>) {
        self.log(LogLevel::Info, text);
    }
    pub fn warn(&self, text: impl Into<String>) {
        self.log(LogLevel::Warn, text);
    }
    pub fn status(&self, text: impl Into<String>) {
        self.emit(PluginEvent::Status(text.into()));
    }
    pub fn progress(&self, done: u64, total: u64) {
        self.emit(PluginEvent::Progress { done, total });
    }
    pub fn set_glow(&self, on: bool) {
        self.emit(PluginEvent::Glow(on));
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Bail out of a long inner loop when the user has cancelled. Call it at
    /// every await point a benchmark controls — cancellation that only lands
    /// between phases leaves a 3-hour BFCL run un-stoppable.
    pub fn check_cancelled(&self) -> Result<()> {
        if self.is_cancelled() {
            bail!("cancelled");
        }
        Ok(())
    }
}

/// The general plugin abstraction. Benchmarks are the first implementors; the
/// registry that dispatches them is deliberately not benchmark-specific.
pub trait Plugin {
    /// Who wrote this plugin, where it came from, and where to report it.
    /// Rendered before the user commits to running anything.
    fn metadata(&self) -> &'static PluginMetadata;

    /// First step. Acquire resources, provision artifacts, verify the host has
    /// what this plugin needs. Returning `Err` here means "not runnable on this
    /// box" and the message is shown to the user in place of the Start button,
    /// so it must name what is missing.
    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_loses_its_trailing_slash() {
        let t = TargetEndpoint::new("http://box:8888/", "m");
        assert_eq!(t.base_url, "http://box:8888");
    }

    #[test]
    fn host_port_splits_ipv4_and_defaults_to_80() {
        assert_eq!(
            TargetEndpoint::local(8888, "m").host_port().unwrap(),
            ("127.0.0.1".to_string(), 8888)
        );
        assert_eq!(
            TargetEndpoint::new("http://dgx3", "m").host_port().unwrap(),
            ("dgx3".to_string(), 80)
        );
        assert!(TargetEndpoint::new("https://x", "m").host_port().is_err());
    }

    #[test]
    fn cancellation_is_visible_through_the_handle() {
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let h = PluginHandle::new(
            1,
            TargetEndpoint::local(8888, "m"),
            ArtifactStore::with_root("/tmp/atlas-test"),
            tx,
            cancel.clone(),
        );
        h.check_cancelled().unwrap();
        h.status("warming up");
        cancel.store(true, Ordering::Relaxed);
        assert!(h.check_cancelled().is_err());
        assert!(matches!(rx.try_recv(), Ok(PluginEvent::Status(s)) if s == "warming up"));
    }
}
