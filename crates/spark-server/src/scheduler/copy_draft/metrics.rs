// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the copy drafter's `/metrics` series (`atlas_copy_*`).
//!
//! Plain atomics, rendered by [`render_prometheus`] through the port sprint's
//! `api::port_metrics` helpers. With the drafter off every series reads 0.
//!
//! | series | meaning |
//! |---|---|
//! | `atlas_copy_rounds_total{kind="chain"}` | verify rounds whose chain came from a copy |
//! | `atlas_copy_rounds_total{kind="lookup"}` | lookup rounds (a copy verified alone) |
//! | `atlas_copy_tokens_proposed_total` | copy tokens sent to verification |
//! | `atlas_copy_tokens_accepted_total` | copy tokens the target committed |
//! | `atlas_copy_skipped_matches_total` | qualifying copy sources the yield gate refused |
//! | `atlas_copy_probe_rounds_total` | copy chains taken as a probe after a refusal streak |

use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) struct CopyMetrics {
    pub chain_rounds: AtomicU64,
    pub lookup_rounds: AtomicU64,
    pub proposed: AtomicU64,
    pub accepted: AtomicU64,
    pub skipped: AtomicU64,
    pub probes: AtomicU64,
}

pub(crate) static METRICS: CopyMetrics = CopyMetrics {
    chain_rounds: AtomicU64::new(0),
    lookup_rounds: AtomicU64::new(0),
    proposed: AtomicU64::new(0),
    accepted: AtomicU64::new(0),
    skipped: AtomicU64::new(0),
    probes: AtomicU64::new(0),
};

#[inline]
pub(crate) fn add(c: &AtomicU64, v: usize) {
    c.fetch_add(v as u64, Ordering::Relaxed);
}

fn get(c: &AtomicU64) -> u64 {
    c.load(Ordering::Relaxed)
}

/// Append the `atlas_copy_*` series to the `/metrics` text.
pub(crate) fn render_prometheus(out: &mut String) {
    use crate::api::port_metrics::{counter, labeled_counter};
    let m = &METRICS;
    labeled_counter(
        out,
        "atlas_copy_rounds_total",
        "Speculative rounds whose drafts came from the copy drafter, by plan kind.",
        &[
            ("kind=\"chain\"", get(&m.chain_rounds)),
            ("kind=\"lookup\"", get(&m.lookup_rounds)),
        ],
    );
    counter(
        out,
        "atlas_copy_tokens_proposed_total",
        "Copy-drafted tokens sent to target verification.",
        get(&m.proposed),
    );
    counter(
        out,
        "atlas_copy_tokens_accepted_total",
        "Copy-drafted tokens the target verification committed.",
        get(&m.accepted),
    );
    counter(
        out,
        "atlas_copy_skipped_matches_total",
        "Qualifying copy sources the yield gate left to the MTP drafter.",
        get(&m.skipped),
    );
    counter(
        out,
        "atlas_copy_probe_rounds_total",
        "Copy chains taken as a probe after a streak of yield-gate refusals.",
        get(&m.probes),
    );
}
