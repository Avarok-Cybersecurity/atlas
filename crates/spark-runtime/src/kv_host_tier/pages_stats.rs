// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the page API's cumulative counters (lane L36). One [`PageStats`] may be
//! shared by every page API a caller creates (the window desk keeps one across model swaps, so
//! its `/metrics` counters never go backwards); [`PageSnapshot`] is a plain copy for rendering.

use std::sync::atomic::{AtomicU64, Ordering};

use super::PageOp;

/// Cumulative counters of one or more page APIs.
#[derive(Debug, Default)]
pub struct PageStats {
    demote_pages: AtomicU64,
    demote_bytes: AtomicU64,
    demote_micros: AtomicU64,
    fetch_pages: AtomicU64,
    fetch_bytes: AtomicU64,
    fetch_micros: AtomicU64,
    host_bytes: AtomicU64,
    refused_pages: AtomicU64,
    failed_copies: AtomicU64,
}

/// A copy of [`PageStats`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PageSnapshot {
    pub demote_pages: u64,
    pub demote_bytes: u64,
    pub demote_seconds: f64,
    pub fetch_pages: u64,
    pub fetch_bytes: u64,
    pub fetch_seconds: f64,
    /// Bytes of host pages held now (a gauge).
    pub host_bytes: u64,
    pub refused_pages: u64,
    pub failed_copies: u64,
}

impl PageStats {
    pub fn snapshot(&self) -> PageSnapshot {
        let r = |a: &AtomicU64| a.load(Ordering::Relaxed);
        PageSnapshot {
            demote_pages: r(&self.demote_pages),
            demote_bytes: r(&self.demote_bytes),
            demote_seconds: r(&self.demote_micros) as f64 / 1e6,
            fetch_pages: r(&self.fetch_pages),
            fetch_bytes: r(&self.fetch_bytes),
            fetch_seconds: r(&self.fetch_micros) as f64 / 1e6,
            host_bytes: r(&self.host_bytes),
            refused_pages: r(&self.refused_pages),
            failed_copies: r(&self.failed_copies),
        }
    }

    /// One copy completed: `pages` pages, `bytes` bytes in `secs` seconds.
    pub(super) fn add_done(&self, op: PageOp, pages: usize, bytes: u64, secs: f64) {
        let micros = (secs.max(0.0) * 1e6) as u64;
        let (p, b, m) = match op {
            PageOp::Demote => (&self.demote_pages, &self.demote_bytes, &self.demote_micros),
            PageOp::Fetch => (&self.fetch_pages, &self.fetch_bytes, &self.fetch_micros),
        };
        p.fetch_add(pages as u64, Ordering::Relaxed);
        b.fetch_add(bytes, Ordering::Relaxed);
        m.fetch_add(micros, Ordering::Relaxed);
    }

    pub(super) fn add_refused(&self, pages: usize) {
        self.refused_pages
            .fetch_add(pages as u64, Ordering::Relaxed);
    }

    pub(super) fn add_failed(&self) {
        self.failed_copies.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn hold(&self, bytes: u64) {
        self.host_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub(super) fn drop_held(&self, bytes: u64) {
        let _ = self
            .host_bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(bytes))
            });
    }
}
