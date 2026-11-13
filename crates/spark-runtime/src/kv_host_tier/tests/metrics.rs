// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the `/metrics` text of the tier: every `atlas_kv_tier_*` series of the lane's
//! fixed list appears once with its HELP and TYPE lines, zero-valued when no tier exists, with
//! the live values of a tier's own stats otherwise.

use super::*;
use crate::kv_host_tier::{TierPolicy, TierStats, render_prometheus};

const SERIES: [(&str, &str); 12] = [
    ("atlas_kv_tier_resident_bytes", "gauge"),
    ("atlas_kv_tier_capacity_bytes", "gauge"),
    ("atlas_kv_tier_writeback_blocks_total", "counter"),
    ("atlas_kv_tier_writeback_dropped_total", "counter"),
    ("atlas_kv_tier_hits_total", "counter"),
    ("atlas_kv_tier_misses_total", "counter"),
    ("atlas_kv_tier_restore_blocks_total", "counter"),
    ("atlas_kv_tier_restore_tokens_total", "counter"),
    ("atlas_kv_tier_restore_seconds_total", "counter"),
    ("atlas_kv_tier_gate_declined_total", "counter"),
    ("atlas_kv_tier_evictions_total", "counter"),
    ("atlas_kv_tier_cost_model_valid", "gauge"),
];

fn value(text: &str, name: &str) -> String {
    let prefix = format!("{name} ");
    text.lines()
        .find(|l| l.starts_with(&prefix))
        .map(|l| l[prefix.len()..].to_string())
        .unwrap_or_else(|| panic!("{name} missing"))
}

#[test]
fn every_series_has_help_type_and_a_value() {
    let mut out = String::new();
    TierStats::default().render(&mut out);
    for (name, kind) in SERIES {
        assert_eq!(out.matches(&format!("# HELP {name} ")).count(), 1, "{name}");
        assert!(out.contains(&format!("# TYPE {name} {kind}\n")), "{name}");
        assert_eq!(value(&out, name), "0", "{name}");
    }
    assert!(!out.contains("disk"), "the disk series are lane L27's");
    // The process-wide render always has the same series (zeros or a registered tier).
    let mut all = String::new();
    render_prometheus(&mut all);
    for (name, _) in SERIES {
        assert_eq!(all.matches(&format!("# TYPE {name} ")).count(), 1, "{name}");
    }
}

#[test]
fn a_tier_reports_its_live_values() {
    let r = rig(8, TierPolicy::Lru);
    let p = Prompt::new(NS, 0, 2);
    store_published(&r, &p, &[device_block(&r.gpu, 1), device_block(&r.gpu, 2)]);
    r.tier.lookup(&p.keys, &p.chunks());
    let q = Prompt::new(NS, 99, 1);
    r.tier.lookup(&q.keys, &q.chunks());
    let s = r.tier.snapshot();
    assert_eq!(s.capacity_bytes, 8 * BLOCK_BYTES as u64);
    assert_eq!(s.resident_bytes, 2 * BLOCK_BYTES as u64);
    assert_eq!((s.hits, s.misses, s.writeback_blocks), (1, 1, 2));
    assert_eq!(s.entries, 2);
    let mut out = String::new();
    r.tier.sh.stats.render(&mut out);
    assert_eq!(
        value(&out, "atlas_kv_tier_resident_bytes"),
        (2 * BLOCK_BYTES).to_string()
    );
    assert_eq!(value(&out, "atlas_kv_tier_hits_total"), "1");
    assert_eq!(value(&out, "atlas_kv_tier_misses_total"), "1");
    assert_eq!(value(&out, "atlas_kv_tier_writeback_blocks_total"), "2");
    assert_eq!(value(&out, "atlas_kv_tier_cost_model_valid"), "0");
}
