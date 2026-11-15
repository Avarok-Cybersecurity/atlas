// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the host tier's counters and gauges and their Prometheus text
//! (`atlas_kv_tier_*`; the `_disk_*` names belong to the disk tier). Each tier owns one
//! [`TierStats`]; the tier created for serving registers it, and [`render_prometheus`] (called
//! from `api/port_metrics.rs`) prints it, or zeros when no tier exists, so
//! the series are present on every target.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Live counters of one tier. Counters only grow; gauges are set.
#[derive(Default)]
pub struct TierStats {
    resident_bytes: AtomicU64,
    capacity_bytes: AtomicU64,
    writeback_blocks: AtomicU64,
    writeback_dropped: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    restore_blocks: AtomicU64,
    restore_tokens: AtomicU64,
    restore_nanos: AtomicU64,
    gate_declined: AtomicU64,
    evictions: AtomicU64,
    cost_model_valid: AtomicBool,
    restore_failed: AtomicU64,
    cold_demoted: AtomicU64,
    cold_promoted: AtomicU64,
}

/// A copy of one tier's numbers (tests, logs and the boot summary).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TierSnapshot {
    pub resident_bytes: u64,
    pub capacity_bytes: u64,
    pub writeback_blocks: u64,
    pub writeback_dropped: u64,
    pub hits: u64,
    pub misses: u64,
    pub restore_blocks: u64,
    pub restore_tokens: u64,
    pub restore_seconds: f64,
    pub gate_declined: u64,
    pub evictions: u64,
    pub cost_model_valid: bool,
    pub restore_failed: u64,
    pub cold_demoted: u64,
    pub cold_promoted: u64,
    /// Entries in the index (staging and ready).
    pub entries: usize,
    pub staging_bytes: u64,
    pub leased_bytes: u64,
    /// Blocks buffered by the lazy store.
    pub lazy_pending_blocks: usize,
}

fn add(c: &AtomicU64, n: u64) {
    c.fetch_add(n, Ordering::Relaxed);
}

fn get(c: &AtomicU64) -> u64 {
    c.load(Ordering::Relaxed)
}

impl TierStats {
    pub(crate) fn set_resident(&self, b: u64) {
        self.resident_bytes.store(b, Ordering::Relaxed);
    }
    pub(crate) fn set_capacity(&self, b: u64) {
        self.capacity_bytes.store(b, Ordering::Relaxed);
    }
    pub(crate) fn add_writeback_blocks(&self, n: u64) {
        add(&self.writeback_blocks, n);
    }
    pub(crate) fn add_writeback_dropped(&self, n: u64) {
        add(&self.writeback_dropped, n);
    }
    pub(crate) fn add_hit(&self) {
        add(&self.hits, 1);
    }
    pub(crate) fn add_miss(&self) {
        add(&self.misses, 1);
    }
    pub(crate) fn add_restore(&self, blocks: u64, tokens: u64, secs: f64) {
        add(&self.restore_blocks, blocks);
        add(&self.restore_tokens, tokens);
        add(&self.restore_nanos, (secs.max(0.0) * 1e9) as u64);
    }
    pub(crate) fn add_restore_failed(&self) {
        add(&self.restore_failed, 1);
    }
    pub(crate) fn add_gate_declined(&self) {
        add(&self.gate_declined, 1);
    }
    pub(crate) fn add_eviction(&self) {
        add(&self.evictions, 1);
    }
    pub(crate) fn add_cold_demoted(&self) {
        add(&self.cold_demoted, 1);
    }
    pub(crate) fn add_cold_promoted(&self) {
        add(&self.cold_promoted, 1);
    }
    pub(crate) fn set_cost_valid(&self, v: bool) {
        self.cost_model_valid.store(v, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(
        &self,
        entries: usize,
        staging_bytes: u64,
        leased_bytes: u64,
        lazy_pending_blocks: usize,
    ) -> TierSnapshot {
        TierSnapshot {
            resident_bytes: get(&self.resident_bytes),
            capacity_bytes: get(&self.capacity_bytes),
            writeback_blocks: get(&self.writeback_blocks),
            writeback_dropped: get(&self.writeback_dropped),
            hits: get(&self.hits),
            misses: get(&self.misses),
            restore_blocks: get(&self.restore_blocks),
            restore_tokens: get(&self.restore_tokens),
            restore_seconds: get(&self.restore_nanos) as f64 / 1e9,
            gate_declined: get(&self.gate_declined),
            evictions: get(&self.evictions),
            cost_model_valid: self.cost_model_valid.load(Ordering::Relaxed),
            restore_failed: get(&self.restore_failed),
            cold_demoted: get(&self.cold_demoted),
            cold_promoted: get(&self.cold_promoted),
            entries,
            staging_bytes,
            leased_bytes,
            lazy_pending_blocks,
        }
    }

    /// The `/metrics` text of this tier.
    pub fn render(&self, out: &mut String) {
        render_values(out, Some(self));
    }
}

static ACTIVE: RwLock<Option<Arc<TierStats>>> = RwLock::new(None);

/// Make `stats` the process's `/metrics` source (the newest tier wins).
pub(crate) fn register(stats: Arc<TierStats>) {
    let mut g = match ACTIVE.write() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    *g = Some(stats);
}

/// Append every `atlas_kv_tier_*` series to `out` (zeros when no tier is registered).
pub fn render_prometheus(out: &mut String) {
    let g = match ACTIVE.read() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    render_values(out, g.as_deref());
}

fn series(out: &mut String, name: &str, kind: &str, help: &str, value: &dyn std::fmt::Display) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} {kind}");
    let _ = writeln!(out, "{name} {value}");
}

fn render_values(out: &mut String, s: Option<&TierStats>) {
    let v = |f: fn(&TierStats) -> &AtomicU64| s.map(|s| get(f(s))).unwrap_or(0);
    let gauges: [(&str, &str, u64); 2] = [
        (
            "atlas_kv_tier_resident_bytes",
            "Host bytes held by the KV host tier (staging and ready blocks).",
            v(|s| &s.resident_bytes),
        ),
        (
            "atlas_kv_tier_capacity_bytes",
            "Host byte budget of the KV host tier (--kv-host-tier-gb).",
            v(|s| &s.capacity_bytes),
        ),
    ];
    for (name, help, val) in gauges {
        series(out, name, "gauge", help, &val);
    }
    let counters: [(&str, &str, u64); 8] = [
        (
            "atlas_kv_tier_writeback_blocks_total",
            "KV blocks copied to the host tier and published.",
            v(|s| &s.writeback_blocks),
        ),
        (
            "atlas_kv_tier_writeback_dropped_total",
            "KV blocks not written back (step budget, in-flight cap, no host room, evicted before a lazy drain, failed copy).",
            v(|s| &s.writeback_dropped),
        ),
        (
            "atlas_kv_tier_hits_total",
            "Host tier lookups that found at least one leading block.",
            v(|s| &s.hits),
        ),
        (
            "atlas_kv_tier_misses_total",
            "Host tier lookups that found no leading block.",
            v(|s| &s.misses),
        ),
        (
            "atlas_kv_tier_restore_blocks_total",
            "KV blocks restored from the host tier into GPU blocks.",
            v(|s| &s.restore_blocks),
        ),
        (
            "atlas_kv_tier_restore_tokens_total",
            "Prompt tokens whose KV was restored from the host tier instead of recomputed.",
            v(|s| &s.restore_tokens),
        ),
        (
            "atlas_kv_tier_gate_declined_total",
            "Restores the cost gate declined (too short, or recompute estimated cheaper).",
            v(|s| &s.gate_declined),
        ),
        (
            "atlas_kv_tier_evictions_total",
            "Host tier blocks evicted to make room.",
            v(|s| &s.evictions),
        ),
    ];
    for (name, help, val) in counters {
        series(out, name, "counter", help, &val);
    }
    let secs = s.map(|s| get(&s.restore_nanos) as f64 / 1e9).unwrap_or(0.0);
    series(
        out,
        "atlas_kv_tier_restore_seconds_total",
        "counter",
        "Seconds spent in host tier restore copies (device time when timing events work).",
        &secs,
    );
    let valid = s
        .map(|s| s.cost_model_valid.load(Ordering::Relaxed))
        .unwrap_or(false);
    series(
        out,
        "atlas_kv_tier_cost_model_valid",
        "gauge",
        "1 when the restore cost model passed its held-out accuracy check and gates restores.",
        &u8::from(valid),
    );
}
