// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the restore queue's `/metrics` series (lane L27, `atlas_kv_restore_*`, port
//! plan 4.2). Process-wide atomics every scheduler thread adds to; each thread's queue publishes
//! its gauge values as deltas, so the gauges are the sum over schedulers.

use std::sync::atomic::{AtomicU64, Ordering};

use super::park::{Exit, Refusal, Report};

static PARKED: AtomicU64 = AtomicU64::new(0);
static DRAINING: AtomicU64 = AtomicU64::new(0);
static PARKED_TOTAL: AtomicU64 = AtomicU64::new(0);
static PARKED_US_TOTAL: AtomicU64 = AtomicU64::new(0);
static EXITS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
static REFUSED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn bump(a: &AtomicU64) {
    a.fetch_add(1, Ordering::Relaxed);
}

pub(super) fn parked_one() {
    bump(&PARKED_TOTAL);
}

/// Count every exit of one poll and the time each request spent parked.
pub(super) fn record<I>(report: &Report<I>) {
    for (exit, waited) in &report.exits {
        if let Some(i) = Exit::ALL.iter().position(|x| x == exit) {
            bump(&EXITS[i]);
        }
        PARKED_US_TOTAL.fetch_add(waited.as_micros() as u64, Ordering::Relaxed);
    }
}

pub(super) fn refused(r: Refusal) {
    if let Some(i) = Refusal::ALL.iter().position(|x| *x == r) {
        bump(&REFUSED[i]);
    }
}

/// One scheduler thread's share of the two gauges.
#[derive(Default)]
pub(super) struct Shown {
    parked: u64,
    draining: u64,
}

impl Shown {
    /// Publish this thread's current counts as deltas against what it published last.
    pub(super) fn sync(&mut self, parked: usize, draining: usize) {
        shift(&PARKED, &mut self.parked, parked as u64);
        shift(&DRAINING, &mut self.draining, draining as u64);
    }
}

impl Drop for Shown {
    fn drop(&mut self) {
        self.sync(0, 0);
    }
}

fn shift(gauge: &AtomicU64, shown: &mut u64, now: u64) {
    if now > *shown {
        gauge.fetch_add(now - *shown, Ordering::Relaxed);
    } else {
        gauge.fetch_sub(*shown - now, Ordering::Relaxed);
    }
    *shown = now;
}

/// Append the `atlas_kv_restore_*` series, through the `port_metrics` helpers.
pub(crate) fn render_prometheus(out: &mut String) {
    use crate::api::port_metrics as pm;
    pm::gauge(
        out,
        "atlas_kv_restore_parked",
        "Requests parked on an asynchronous host-tier KV restore (kv_restore_async).",
        PARKED.load(Ordering::Relaxed),
    );
    pm::gauge(
        out,
        "atlas_kv_restore_draining",
        "Restore tickets polled to completion after their request left the queue.",
        DRAINING.load(Ordering::Relaxed),
    );
    pm::counter(
        out,
        "atlas_kv_restore_parked_total",
        "Requests parked on an asynchronous KV restore.",
        PARKED_TOTAL.load(Ordering::Relaxed),
    );
    pm::counter(
        out,
        "atlas_kv_restore_parked_seconds_total",
        "Seconds requests spent parked on an asynchronous KV restore.",
        PARKED_US_TOTAL.load(Ordering::Relaxed) as f64 / 1e6,
    );
    let exits: Vec<(String, u64)> = Exit::ALL
        .iter()
        .zip(&EXITS)
        .map(|(e, a)| {
            (
                format!("outcome=\"{}\"", e.label()),
                a.load(Ordering::Relaxed),
            )
        })
        .collect();
    let exits: Vec<(&str, u64)> = exits.iter().map(|(l, v)| (l.as_str(), *v)).collect();
    pm::labeled_counter(
        out,
        "atlas_kv_restore_exits_total",
        "Parked requests by how they left the queue.",
        &exits,
    );
    let refused: Vec<(String, u64)> = Refusal::ALL
        .iter()
        .zip(&REFUSED)
        .map(|(r, a)| {
            (
                format!("reason=\"{}\"", r.label()),
                a.load(Ordering::Relaxed),
            )
        })
        .collect();
    let refused: Vec<(&str, u64)> = refused.iter().map(|(l, v)| (l.as_str(), *v)).collect();
    pm::labeled_counter(
        out,
        "atlas_kv_restore_park_refused_total",
        "Restores handled synchronously because parking was refused, by reason.",
        &refused,
    );
}
