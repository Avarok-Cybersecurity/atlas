// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the CUDA-graph cache counters of the `graph_exec_update`
//! row (port sprint lane L09), their `/metrics` text, and the per-model
//! topology-signature tables the batched graph caches use.
//!
//! The cache logic itself is `model/trait_impl/graph_family.rs`; this module
//! is public so `spark-server`'s `api/port_metrics.rs` can render the series.
//!
//! The counters move only when the row is on or `AVAROK_GRAPH_COUNTERS=1`
//! (house VALUE convention: the literal `1` and nothing else), so a default
//! serve on every target renders all five series as zero. The variable is a
//! host-side diagnostic only: it adds relaxed atomic increments at capture
//! time and changes no launch, no capture and no cache decision, which is why
//! an A/B can set it in both arms to read the row-off arm's churn.

use std::collections::HashMap;
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

/// Graph-cache counters, one process-wide set ([`GRAPH_COUNTERS`]).
#[derive(Debug, Default)]
pub struct GraphCounters {
    captures: AtomicU64,
    instantiations: AtomicU64,
    updates: AtomicU64,
    update_failures: AtomicU64,
    evictions: AtomicU64,
}

/// A point-in-time copy of [`GraphCounters`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphCounts {
    /// Captures that ended with a graph (stream capture closed, non-null).
    pub captures: u64,
    /// Executables instantiated from a capture.
    pub instantiations: u64,
    /// Successful `cuGraphExecUpdate` calls that replaced an instantiation.
    pub updates: u64,
    /// Updates the driver refused (topology or parameter mismatch) or failed.
    pub update_failures: u64,
    /// Least-recently-used cache entries removed at the cache cap, whether
    /// their executable was destroyed or recycled by an update.
    pub evictions: u64,
}

impl GraphCounters {
    /// All zero; `const` so the process-wide set is a plain static.
    pub const fn new() -> Self {
        Self {
            captures: AtomicU64::new(0),
            instantiations: AtomicU64::new(0),
            updates: AtomicU64::new(0),
            update_failures: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    /// One capture closed with a graph.
    pub fn add_capture(&self) {
        self.captures.fetch_add(1, Ordering::Relaxed);
    }

    /// One executable instantiated.
    pub fn add_instantiation(&self) {
        self.instantiations.fetch_add(1, Ordering::Relaxed);
    }

    /// One executable updated in place.
    pub fn add_update(&self) {
        self.updates.fetch_add(1, Ordering::Relaxed);
    }

    /// One update attempt that did not apply.
    pub fn add_update_failure(&self) {
        self.update_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// One LRU entry removed at the cap.
    pub fn add_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// The current values.
    pub fn snapshot(&self) -> GraphCounts {
        GraphCounts {
            captures: self.captures.load(Ordering::Relaxed),
            instantiations: self.instantiations.load(Ordering::Relaxed),
            updates: self.updates.load(Ordering::Relaxed),
            update_failures: self.update_failures.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }
}

/// The process-wide graph-cache counters.
pub static GRAPH_COUNTERS: GraphCounters = GraphCounters::new();

/// The diagnostic switch that arms the counters with the row off.
pub const COUNTERS_ENV: &str = "AVAROK_GRAPH_COUNTERS";

/// Whether the counters move: the row is on, or [`COUNTERS_ENV`] is exactly
/// `1`. Read once per process.
pub fn counting_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        crate::layers::ops::port_rows::graph_exec_update::row()
            || counters_env_armed(std::env::var(COUNTERS_ENV).ok().as_deref())
    })
}

/// House VALUE convention for [`COUNTERS_ENV`]: armed by the literal `"1"`.
pub(crate) fn counters_env_armed(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Per-model topology-signature tables, one per batched graph cache, keyed by
/// executable handle (`GraphHandle.0`). A field of
/// [`crate::port::state::ModelPortState`]; empty unless the row is on.
///
/// The value is the topology signature of the capture the executable was
/// instantiated from. An update keeps the topology, so the entry stays valid
/// across updates; an executable destroyed outside the graph-family path (a
/// LoRA drain, a stale-slot removal) leaves a dead entry that the next insert
/// prunes once the table outgrows its cache.
#[derive(Debug, Default)]
pub struct FamilyState {
    /// The batched-decode cache (`batch_decode_graphs`).
    pub decode_sigs: Mutex<HashMap<u64, u64>>,
    /// The batched-verify cache (`verify_batched_graphs`).
    pub verify_sigs: Mutex<HashMap<u64, u64>>,
}

/// Append the five `atlas_graph_*` counters to the Prometheus text.
pub fn render_prometheus(out: &mut String) {
    render_counts(out, &GRAPH_COUNTERS.snapshot());
}

/// [`render_prometheus`] over given values (pure, for tests).
pub fn render_counts(out: &mut String, c: &GraphCounts) {
    let series: [(&str, &str, u64); 5] = [
        (
            "atlas_graph_captures_total",
            "CUDA graph captures closed by the decode and verify graph caches.",
            c.captures,
        ),
        (
            "atlas_graph_instantiations_total",
            "CUDA graph executables instantiated by the decode and verify graph caches.",
            c.instantiations,
        ),
        (
            "atlas_graph_updates_total",
            "CUDA graph executables updated in place (cuGraphExecUpdate) instead of instantiated.",
            c.updates,
        ),
        (
            "atlas_graph_update_failures_total",
            "CUDA graph exec updates that did not apply; the capture was instantiated instead.",
            c.update_failures,
        ),
        (
            "atlas_graph_evictions_total",
            "Least-recently-used batched graph cache entries removed at the cache cap.",
            c.evictions,
        ),
    ];
    for (name, help, v) in series {
        let _ = writeln!(
            out,
            "# HELP {name} {help}\n# TYPE {name} counter\n{name} {v}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_add_and_snapshot() {
        let c = GraphCounters::new();
        c.add_capture();
        c.add_capture();
        c.add_instantiation();
        c.add_update();
        c.add_update_failure();
        c.add_eviction();
        c.add_eviction();
        assert_eq!(
            c.snapshot(),
            GraphCounts {
                captures: 2,
                instantiations: 1,
                updates: 1,
                update_failures: 1,
                evictions: 2,
            }
        );
    }

    #[test]
    fn render_writes_five_counters_in_order() {
        let mut s = String::from("x 1\n");
        let c = GraphCounts {
            captures: 5,
            instantiations: 4,
            updates: 3,
            update_failures: 2,
            evictions: 1,
        };
        render_counts(&mut s, &c);
        assert!(s.starts_with("x 1\n"));
        let values: Vec<&str> = s.lines().filter(|l| !l.starts_with('#')).collect();
        assert_eq!(
            values,
            [
                "x 1",
                "atlas_graph_captures_total 5",
                "atlas_graph_instantiations_total 4",
                "atlas_graph_updates_total 3",
                "atlas_graph_update_failures_total 2",
                "atlas_graph_evictions_total 1",
            ]
        );
        assert_eq!(s.matches("# TYPE ").count(), 5);
        assert!(
            s.lines()
                .filter(|l| l.starts_with("# TYPE"))
                .all(|l| l.ends_with(" counter"))
        );
        assert!(s.ends_with('\n'));
    }

    #[test]
    fn zero_counts_render_zero_series() {
        let mut s = String::new();
        render_counts(&mut s, &GraphCounts::default());
        assert_eq!(s.lines().filter(|l| l.ends_with(" 0")).count(), 5);
    }

    #[test]
    fn the_counters_env_uses_the_value_convention() {
        assert!(counters_env_armed(Some("1")));
        for off in [None, Some("0"), Some(""), Some("true"), Some(" 1")] {
            assert!(!counters_env_armed(off), "{off:?}");
        }
    }

    #[test]
    fn family_state_starts_empty() {
        let s = FamilyState::default();
        assert!(s.decode_sigs.lock().is_empty());
        assert!(s.verify_sigs.lock().is_empty());
    }
}
