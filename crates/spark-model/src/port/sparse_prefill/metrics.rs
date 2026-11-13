// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the block-sparse prefill `/metrics` series (lane L35's `atlas_prefill_sparse_*`
//! namespace, master plan 4.2):
//!
//! * `atlas_prefill_sparse_layers_total` (counter): attention layer calls of a prefill chunk that ran
//!   block-sparse;
//! * `atlas_prefill_sparse_blocks_kept_ratio` (summary, `_sum` and `_count`): kept key blocks over
//!   candidate key blocks, one observation per sampled layer call (every `launch::SAMPLE_EVERY`-th call
//!   and the first, which wait for the stream to read the kernel's counters).
//!
//! Both render as zero with the row off, on every target. spark-model cannot reach spark-server's
//! `api::port_metrics` helpers, so [`render_prometheus`] writes the same `# HELP` / `# TYPE` lines;
//! spark-server calls it from `port_metrics::render`.

use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

static LAYERS: AtomicU64 = AtomicU64::new(0);
/// (sum of observed ratios, observations).
static RATIO: Mutex<(f64, u64)> = Mutex::new((0.0, 0));

/// Count one block-sparse layer call; returns the running count (1 for the first).
pub fn note_layer() -> u64 {
    LAYERS.fetch_add(1, Ordering::Relaxed) + 1
}

/// Record one sampled call's kept and candidate block totals.
pub fn note_sample(kept: u64, candidates: u64) {
    if candidates == 0 {
        return;
    }
    let mut g = RATIO.lock();
    g.0 += kept as f64 / candidates as f64;
    g.1 += 1;
}

/// Append both series to the Prometheus text.
pub fn render_prometheus(out: &mut String) {
    let (sum, count) = *RATIO.lock();
    render_values(out, LAYERS.load(Ordering::Relaxed), sum, count);
}

/// [`render_prometheus`] over explicit values (pure, for the tests).
pub fn render_values(out: &mut String, layers: u64, ratio_sum: f64, ratio_count: u64) {
    let _ = writeln!(
        out,
        "# HELP atlas_prefill_sparse_layers_total Prefill attention layer calls that ran block-sparse \
         (row prefill_sparse_attn, APPROXIMATE).\n\
         # TYPE atlas_prefill_sparse_layers_total counter\n\
         atlas_prefill_sparse_layers_total {layers}"
    );
    let _ = writeln!(
        out,
        "# HELP atlas_prefill_sparse_blocks_kept_ratio Kept over candidate key blocks of a sampled \
         block-sparse prefill layer call.\n\
         # TYPE atlas_prefill_sparse_blocks_kept_ratio summary\n\
         atlas_prefill_sparse_blocks_kept_ratio_sum {ratio_sum}\n\
         atlas_prefill_sparse_blocks_kept_ratio_count {ratio_count}"
    );
}
