// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the Quest selector's `/metrics` series (port lane L30):
//! `atlas_attn_sparse_steps_total` (sequence decode or verify steps that
//! attended a sparse view, counted at the first attention layer) and
//! `atlas_attn_sparse_pages_selected_total` (pages those steps kept, same
//! layer). Both are zero on every serve with the row off.
//!
//! Decode steps replay CUDA graphs, where no host code runs, so the counts are
//! kept on the device by the compaction kernel and harvested into the host
//! atomics below at eager hooks, at most once per second (one 16-byte copy
//! ordered after the stream). A new arena (model swap) starts its device
//! counters at zero; the host keeps the earlier totals as a base.

use std::fmt::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub const STEPS: &str = "atlas_attn_sparse_steps_total";
pub const PAGES: &str = "atlas_attn_sparse_pages_selected_total";

/// Host mirror of one arena's device counters, plus the totals of earlier arenas.
pub struct Counters {
    base: [AtomicU64; 2],
    last: [AtomicU64; 2],
    harvested_at: Mutex<Option<Instant>>,
}

impl Counters {
    pub const fn new() -> Self {
        Self {
            base: [AtomicU64::new(0), AtomicU64::new(0)],
            last: [AtomicU64::new(0), AtomicU64::new(0)],
            harvested_at: Mutex::new(None),
        }
    }

    /// Record a harvested device snapshot `(steps, pages)`.
    pub fn observe(&self, steps: u64, pages: u64) {
        self.last[0].store(steps, Ordering::Relaxed);
        self.last[1].store(pages, Ordering::Relaxed);
    }

    /// A new arena replaces the device counters: fold the last snapshot into the base.
    pub fn rebase(&self) {
        for (last, base) in self.last.iter().zip(&self.base) {
            base.fetch_add(last.swap(0, Ordering::Relaxed), Ordering::Relaxed);
        }
    }

    pub fn totals(&self) -> (u64, u64) {
        (
            self.base[0].load(Ordering::Relaxed) + self.last[0].load(Ordering::Relaxed),
            self.base[1].load(Ordering::Relaxed) + self.last[1].load(Ordering::Relaxed),
        )
    }

    /// Harvest the device counters at `stats` when the last harvest is older
    /// than `every` (or `force`). Eager hooks only.
    pub fn harvest(
        &self,
        gpu: &dyn GpuBackend,
        stats: DevicePtr,
        stream: u64,
        every: Duration,
        force: bool,
    ) -> Result<()> {
        {
            let mut at = self
                .harvested_at
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let now = Instant::now();
            if !force && at.is_some_and(|t| now.duration_since(t) < every) {
                return Ok(());
            }
            *at = Some(now);
        }
        let mut raw = [0u8; 16];
        gpu.copy_d2h_on_stream(stats, &mut raw, stream)?;
        let steps = u64::from_le_bytes(raw[..8].try_into().expect("8 bytes"));
        let pages = u64::from_le_bytes(raw[8..].try_into().expect("8 bytes"));
        self.observe(steps, pages);
        Ok(())
    }
}

impl Default for Counters {
    fn default() -> Self {
        Self::new()
    }
}

pub static GLOBAL: Counters = Counters::new();

/// Append the two series to the `/metrics` text.
pub fn render_with(out: &mut String, c: &Counters) {
    let (steps, pages) = c.totals();
    let _ = writeln!(
        out,
        "# HELP {STEPS} Sequence decode or verify steps that attended a Quest page-sparse view \
         (attn_page_sparse), counted at the first attention layer.\n# TYPE {STEPS} counter\n{STEPS} {steps}"
    );
    let _ = writeln!(
        out,
        "# HELP {PAGES} Pages (64 tokens) the Quest page-sparse steps kept: sinks, top-k and the \
         recent window, at the first attention layer.\n# TYPE {PAGES} counter\n{PAGES} {pages}"
    );
}

/// The `port_metrics::render` entry point.
pub fn render_prometheus(out: &mut String) {
    render_with(out, &GLOBAL);
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn a_fresh_serve_renders_zeros() {
        let c = Counters::new();
        let mut s = String::new();
        render_with(&mut s, &c);
        assert!(s.contains(
            "# TYPE atlas_attn_sparse_steps_total counter\natlas_attn_sparse_steps_total 0\n"
        ));
        assert!(s.contains("atlas_attn_sparse_pages_selected_total 0\n"));
        assert!(!s.contains('\u{2014}'));
    }

    #[test]
    fn rebase_keeps_totals_across_arenas() {
        let c = Counters::new();
        c.observe(5, 170);
        c.rebase();
        c.observe(2, 68);
        assert_eq!(c.totals(), (7, 238));
    }

    #[test]
    fn harvest_reads_the_device_counters_and_rate_limits() {
        let gpu = MockGpuBackend::new();
        let p = gpu.alloc(16).unwrap();
        let mut raw = Vec::new();
        raw.extend_from_slice(&3u64.to_le_bytes());
        raw.extend_from_slice(&99u64.to_le_bytes());
        gpu.copy_h2d(&raw, p).unwrap();
        let c = Counters::new();
        c.harvest(&gpu, p, 0, Duration::from_secs(3600), false)
            .unwrap();
        assert_eq!(c.totals(), (3, 99));
        let mut raw2 = Vec::new();
        raw2.extend_from_slice(&4u64.to_le_bytes());
        raw2.extend_from_slice(&120u64.to_le_bytes());
        gpu.copy_h2d(&raw2, p).unwrap();
        c.harvest(&gpu, p, 0, Duration::from_secs(3600), false)
            .unwrap();
        assert_eq!(c.totals(), (3, 99), "rate limited");
        c.harvest(&gpu, p, 0, Duration::from_secs(3600), true)
            .unwrap();
        assert_eq!(c.totals(), (4, 120));
    }
}
