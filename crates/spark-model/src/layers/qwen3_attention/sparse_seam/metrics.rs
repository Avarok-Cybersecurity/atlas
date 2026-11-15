// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the seam's `/metrics` series (`atlas_sparse_*`) and the counters
//! behind them.
//!
//! The counters are host-side: a plan is counted when a hook runs, which is
//! every eager step and every capture, not every graph replay. Selected pages
//! are reported by the selector ([`record_selected_pages`]), because only it
//! knows them. `render_prometheus` writes the same `# HELP` / `# TYPE` text as
//! spark-server's `port_metrics` helpers; on a serve with no selector every
//! series renders as zero (the GB10-visible change of this lane).

use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

/// The plan kinds `atlas_sparse_plan_total{kind}` counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlanKind {
    Dense,
    View,
    Custom,
    PrefillDense,
    PrefillView,
}

const KINDS: [(PlanKind, &str); 5] = [
    (PlanKind::Dense, "dense"),
    (PlanKind::View, "view"),
    (PlanKind::Custom, "custom"),
    (PlanKind::PrefillDense, "prefill_dense"),
    (PlanKind::PrefillView, "prefill_view"),
];

/// One set of seam counters. [`GLOBAL`] is the serve's; tests own theirs.
#[derive(Default)]
pub(crate) struct Counters {
    plans: [AtomicU64; 5],
    selected_pages: AtomicU64,
    graph_bypass: AtomicU64,
    view_ptr_changes: AtomicU64,
}

impl Counters {
    pub(crate) const fn new() -> Self {
        Self {
            plans: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            selected_pages: AtomicU64::new(0),
            graph_bypass: AtomicU64::new(0),
            view_ptr_changes: AtomicU64::new(0),
        }
    }

    fn slot(kind: PlanKind) -> usize {
        KINDS
            .iter()
            .position(|(k, _)| *k == kind)
            .expect("every PlanKind is listed in KINDS")
    }

    pub(crate) fn count_plan(&self, kind: PlanKind) {
        self.plans[Self::slot(kind)].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn plans(&self, kind: PlanKind) -> u64 {
        self.plans[Self::slot(kind)].load(Ordering::Relaxed)
    }

    pub(crate) fn add_selected_pages(&self, n: u64) {
        self.selected_pages.fetch_add(n, Ordering::Relaxed);
    }

    pub(crate) fn selected_pages(&self) -> u64 {
        self.selected_pages.load(Ordering::Relaxed)
    }

    pub(crate) fn count_graph_bypass(&self) {
        self.graph_bypass.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn graph_bypass(&self) -> u64 {
        self.graph_bypass.load(Ordering::Relaxed)
    }

    pub(crate) fn count_view_ptr_change(&self) {
        self.view_ptr_changes.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn view_ptr_changes(&self) -> u64 {
        self.view_ptr_changes.load(Ordering::Relaxed)
    }
}

pub(crate) static GLOBAL: Counters = Counters::new();

/// Selector-side report: `n` pages (KV blocks) the active selector chose for
/// attention, summed over rows. Feeds `atlas_sparse_selected_pages_total`.
pub fn record_selected_pages(n: u64) {
    GLOBAL.add_selected_pages(n);
}

/// Append the seam's series to the `/metrics` text.
pub fn render_prometheus(out: &mut String) {
    let sel = super::active().map(|s| (s.name(), s.reserved_bytes()));
    render_with(out, sel, &GLOBAL);
}

/// [`render_prometheus`] over explicit inputs: the active selector's name and
/// reservation, and the counters.
pub(crate) fn render_with(out: &mut String, sel: Option<(&str, u64)>, c: &Counters) {
    let (label, active, reserved) = match sel {
        Some((name, bytes)) => (name, 1u64, bytes),
        None => ("none", 0u64, 0u64),
    };
    let _ = writeln!(
        out,
        "# HELP atlas_sparse_active Sparse-attention selector registered for this serve (1) or none (0).\n\
         # TYPE atlas_sparse_active gauge\n\
         atlas_sparse_active{{selector=\"{label}\"}} {active}"
    );
    let _ = writeln!(
        out,
        "# HELP atlas_sparse_plan_total Sparse-attention plans by kind (host-side: eager steps and graph captures).\n\
         # TYPE atlas_sparse_plan_total counter"
    );
    for (kind, name) in KINDS {
        let _ = writeln!(
            out,
            "atlas_sparse_plan_total{{kind=\"{name}\"}} {}",
            c.plans(kind)
        );
    }
    let _ = writeln!(
        out,
        "# HELP atlas_sparse_selected_pages_total KV pages the active selector chose for attention, summed over rows.\n\
         # TYPE atlas_sparse_selected_pages_total counter\n\
         atlas_sparse_selected_pages_total {}",
        c.selected_pages()
    );
    let _ = writeln!(
        out,
        "# HELP atlas_sparse_graph_bypass_total Custom plans that are not graph-safe, replaced by the dense tables while capturing.\n\
         # TYPE atlas_sparse_graph_bypass_total counter\n\
         atlas_sparse_graph_bypass_total {}",
        c.graph_bypass()
    );
    let _ = writeln!(
        out,
        "# HELP atlas_sparse_view_ptr_changes_total Captured View plans whose table pointers differed from an earlier capture of the same layer and row count.\n\
         # TYPE atlas_sparse_view_ptr_changes_total counter\n\
         atlas_sparse_view_ptr_changes_total {}",
        c.view_ptr_changes()
    );
    let _ = writeln!(
        out,
        "# HELP atlas_sparse_reserved_bytes Device bytes the active selector reserved, charged to the KV-block budget.\n\
         # TYPE atlas_sparse_reserved_bytes gauge\n\
         atlas_sparse_reserved_bytes {reserved}"
    );
}
