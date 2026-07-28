// SPDX-License-Identifier: AGPL-3.0-only

//! Structured log capture for the TUI log pane.
//!
//! A `tracing_subscriber::Layer` that records every event as a typed
//! [`LogLine`] (timestamp, level, target, message) into a process-global ring
//! — no re-parsing of formatted text, so the pane can color/dim columns
//! independently. The ring is global (not owned by the TUI thread) so the
//! panic hook can dump it even when the TUI thread is the one panicking.
//!
//! Capacity-bounded (`CAP` lines); oldest lines drop first. Writers never
//! block: the ring lock is uncontended in practice (log emission and one
//! 10 Hz reader).

use std::collections::VecDeque;
use std::io::Write;
use std::time::SystemTime;

use parking_lot::Mutex;
use tracing::Level;
use tracing::field::{Field, Visit};

/// Ring capacity in lines. ~10k lines ≈ a few MB; enough scrollback to cover
/// a full model load plus serving history.
const CAP: usize = 10_000;

/// One captured log event.
#[derive(Clone, Debug)]
pub struct LogLine {
    pub at: SystemTime,
    pub level: Level,
    /// Event target, e.g. `spark_model::weight_loader`.
    pub target: String,
    pub message: String,
}

static RING: Mutex<VecDeque<LogLine>> = Mutex::new(VecDeque::new());
/// Monotone counter of lines ever pushed — lets the pane detect "n new lines
/// while scrolled up" without diffing contents.
static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn push(line: LogLine) {
    let mut ring = RING.lock();
    if ring.len() == CAP {
        ring.pop_front();
    }
    ring.push_back(line);
    SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Total lines ever captured (for new-line badges while scrolled).
pub fn seq() -> u64 {
    SEQ.load(std::sync::atomic::Ordering::Relaxed)
}

/// Clone out the newest `n` lines (oldest-first order preserved).
pub fn tail(n: usize) -> Vec<LogLine> {
    let ring = RING.lock();
    let skip = ring.len().saturating_sub(n);
    ring.iter().skip(skip).cloned().collect()
}

/// Panic-hook dump: newest `n` lines, plain text, to `w`. Signature is a
/// plain `fn` so `terminal_guard::install_panic_hook` can hold it without a
/// type dependency on this module.
pub fn dump_to(w: &mut dyn Write, n: usize) {
    for l in tail(n) {
        let _ = writeln!(w, "{:>5} {} {}", l.level, l.target, l.message);
    }
}

/// Extracts the `message` field of an event.
#[derive(Default)]
struct MessageVisitor(String);

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            use std::fmt::Write as _;
            let _ = write!(self.0, "{value:?}");
        }
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.0.push_str(value);
        }
    }
}

/// The capture layer. Attach with the same `EnvFilter` semantics as the fmt
/// layer so the pane shows exactly what plain-mode stdout would have shown.
pub struct LogRingLayer;

impl<S> tracing_subscriber::Layer<S> for LogRingLayer
where
    S: tracing::Subscriber,
{
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        // Progress events have their own channel; keep them out of the pane.
        if event.metadata().target() == spark_runtime::progress::TARGET {
            return;
        }
        let mut v = MessageVisitor::default();
        event.record(&mut v);
        push(LogLine {
            at: SystemTime::now(),
            level: *event.metadata().level(),
            target: event.metadata().target().to_string(),
            message: v.0,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_caps_and_tails() {
        for i in 0..(CAP + 10) {
            push(LogLine {
                at: SystemTime::now(),
                level: Level::INFO,
                target: "t".into(),
                message: format!("m{i}"),
            });
        }
        let t = tail(5);
        assert_eq!(t.len(), 5);
        assert_eq!(t.last().unwrap().message, format!("m{}", CAP + 9));
        assert!(seq() >= (CAP + 10) as u64);
    }
}
