// SPDX-License-Identifier: AGPL-3.0-only

//! History-pane reads for [`BenchState`], split out of `bench_state.rs` at the
//! 500-LoC cap. Exact piecewise move — no logic changed. These two are the
//! only members that read `~/.atlas/runs` rather than driving a live run, so
//! they form the natural seam.

use super::BenchState;

impl BenchState {
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

impl BenchState {
    /// `c` in History — write a result card for the selected run.
    ///
    /// Reports through `status` rather than a modal: it is a one-line outcome,
    /// and the path is the only thing the operator needs next.
    pub fn export_card(&mut self) -> crate::tui::bench_keys::Outcome {
        let Some(run) = self.history.get(self.history_row) else {
            return crate::tui::bench_keys::Outcome::None;
        };
        self.status =
            match crate::cli::bench_run::render_card_for_benchmark(&run.benchmark_id, None) {
                Ok(path) => format!("card written to {}", path.display()),
                // The common cause is a benchmark that has never been committed as a
                // gate record — say that, rather than "failed".
                Err(e) => format!("no card: {e}"),
            };
        crate::tui::bench_keys::Outcome::None
    }
}
