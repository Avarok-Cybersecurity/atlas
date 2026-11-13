// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: configuration parsing and validation, block keys, and the restore cost gate:
//! the gate always restores until both rate models pass the held-out check (p95 relative error
//! within the bound, no ordering inversion), then restores only when the copy is estimated under
//! `ratio` x the recompute, and it switches back to "always restore" when the model degrades.

use super::*;
use crate::kv_host_tier::config::{self, StoreMode};
use crate::kv_host_tier::cost::{CostModel, RateModel};
use crate::kv_host_tier::{GateDecision, TierPolicy};

#[test]
fn policies_and_store_modes_parse() {
    for p in TierPolicy::ALL {
        assert_eq!(p.as_str().parse::<TierPolicy>().unwrap(), p);
        assert_eq!(p.to_string(), p.as_str());
    }
    assert_eq!(" ARC ".parse::<TierPolicy>().unwrap(), TierPolicy::Arc);
    assert!("fifo".parse::<TierPolicy>().is_err());
    assert_eq!("lazy".parse::<StoreMode>().unwrap(), StoreMode::Lazy);
    assert_eq!("Eager".parse::<StoreMode>().unwrap(), StoreMode::Eager);
    assert!("later".parse::<StoreMode>().is_err());
}

#[test]
fn configs_validate() {
    assert!(
        config::validate(&TierConfig::default()).is_ok(),
        "off is valid"
    );
    assert!(config::validate(&TierConfig::with_budget_gib(4)).is_ok());
    let mut c = TierConfig::with_budget_gib(1);
    c.restore_gate_ratio = 0.0;
    assert!(config::validate(&c).is_err());
    c.restore_gate_ratio = f32::NAN;
    assert!(config::validate(&c).is_err());
    let tiny = TierConfig {
        budget_bytes: 1000,
        ..TierConfig::default()
    };
    assert!(config::validate(&tiny).is_err());
    let no_wb = TierConfig {
        writeback_bytes_per_step: 0,
        ..TierConfig::with_budget_gib(1)
    };
    assert!(config::validate(&no_wb).is_err());
    assert!(TierOptions::default().validate().is_ok());
    let bad = TierOptions {
        high_watermark: 1.5,
        ..TierOptions::default()
    };
    assert!(bad.validate().is_err());
    assert_eq!(TierConfig::with_budget_gib(2).budget_bytes, 2 << 30);
}

#[test]
fn block_keys_chain_and_cover_only_full_blocks() {
    let toks: Vec<u32> = (0..10).collect();
    let keys = BlockKey::chain_blocks(NS, 0, &toks, 4);
    assert_eq!(keys.len(), 2, "the 2-token tail gets no key");
    assert_eq!(keys[0], BlockKey::chain(NS, 0, None, &toks[..4]));
    assert_eq!(keys[1], BlockKey::chain(NS, 0, Some(&keys[0]), &toks[4..8]));
    // Same tokens, other parent: another key (a block is identified by its whole prefix).
    assert_ne!(keys[1], BlockKey::chain(NS, 0, None, &toks[4..8]));
    assert_ne!(keys[0], BlockKey::chain(NS, 1, None, &toks[..4]));
    assert_ne!(keys[0], BlockKey::chain(NS + 1, 0, None, &toks[..4]));
    assert_eq!(BlockKey::from_hex(&keys[0].to_hex()), Some(keys[0]));
    assert_eq!(BlockKey::from_hex("zz"), None);
    assert!(BlockKey::chain_blocks(NS, 0, &toks, 0).is_empty());
}

fn opts() -> TierOptions {
    TierOptions {
        cost_window: 16,
        cost_min_observations: 8,
        ..test_opts()
    }
}

#[test]
fn a_model_is_valid_only_after_enough_accurate_observations() {
    let mut m = RateModel::new(&opts());
    assert!(!m.valid());
    assert_eq!(m.predict(1.0), None);
    // Exactly 10 GB/s, sizes varying: predictions are exact once the rate is known.
    for i in 0..9 {
        let bytes = 1e6 * (1 + i % 4) as f64;
        m.observe(bytes, bytes / 10e9);
    }
    // The first observation only seeds the rate; eight held-out predictions follow.
    assert!(m.valid());
    assert!(m.p95_error().unwrap() < 1e-9);
    assert!(!m.has_inversion());
    let secs = m.predict(2e6).unwrap();
    assert!((secs - 2e-4).abs() < 1e-12);
}

#[test]
fn noisy_observations_invalidate_the_model() {
    let mut m = RateModel::new(&opts());
    for i in 0..9 {
        m.observe(1e6, 1e-4 * (1.0 + 0.01 * i as f64));
    }
    assert!(m.valid());
    // Large swings: the held-out p95 error passes 25%.
    for i in 0..16 {
        let secs = if i % 2 == 0 { 1e-4 } else { 5e-4 };
        m.observe(1e6, secs);
    }
    assert!(!m.valid());
    assert!(m.p95_error().unwrap() > 0.25);
}

#[test]
fn an_ordering_inversion_invalidates_the_model() {
    let mut m = RateModel::new(&TierOptions {
        cost_p95_limit: 10.0,
        ..opts()
    });
    // Big transfers that are fast and small ones that are slow: a size-proportional model
    // predicts them in reverse order, whatever its error bound.
    for i in 0..12 {
        if i % 2 == 0 {
            m.observe(8e6, 1e-4);
        } else {
            m.observe(1e6, 4e-4);
        }
    }
    assert!(m.has_inversion());
    assert!(!m.valid());
}

#[test]
fn the_gate_restores_until_validated_then_decides_by_cost() {
    let o = opts();
    let mut c = CostModel::new(&o);
    assert_eq!(c.decide(10, 1_000, 0.5, 256), GateDecision::DeclineShort);
    assert_eq!(
        c.decide(1_000, 1_000, 0.5, 256),
        GateDecision::RestoreUnvalidated
    );
    for i in 0..9 {
        let n = (1 + i % 3) as f64;
        c.observe_copy(1e8 * n, 1e-2 * n); // 10 GB/s
        c.observe_prefill(1_000.0 * n, n); // 1,000 tok/s
    }
    assert!(c.valid());
    // 4,000 tokens: recompute 4 s; 1 GB copy 0.1 s < 0.5 x 4 s.
    assert!(matches!(
        c.decide(4_000, 1 << 30, 0.5, 256),
        GateDecision::Restore { .. }
    ));
    // 300 tokens: recompute 0.3 s; 4 GB copy 0.43 s > 0.15 s.
    assert!(matches!(
        c.decide(300, 4 << 30, 0.5, 256),
        GateDecision::DeclineCost { .. }
    ));
}

#[test]
fn the_tier_gate_counts_declines_and_exports_validity() {
    let mut cfg = cfg_blocks(64, TierPolicy::Lru);
    cfg.min_tokens = 8;
    let r = rig_with(cfg, opts());
    assert_eq!(r.tier.gate(4, 1), GateDecision::DeclineShort);
    assert_eq!(r.tier.gate(64, 1), GateDecision::RestoreUnvalidated);
    assert_eq!(r.tier.snapshot().gate_declined, 1);
    assert!(!r.tier.cost_model_valid());
    for _ in 0..9 {
        r.tier.observe_prefill(1_000, 1.0);
    }
    // Only the prefill side is validated; copies have not been measured (no timing here).
    assert!(!r.tier.cost_model_valid());
    assert!(!r.tier.snapshot().cost_model_valid);
}

#[test]
fn the_row_gate_needs_the_row_and_a_budget() {
    // `active()` reads process-wide cells; this is the only test that touches them, and it
    // leaves the row off (the default) when it ends.
    config::set_enabled(false);
    assert!(config::active().is_none());
    assert!(!config::row_enabled());
    config::set_enabled(true);
    assert!(config::row_enabled());
    // No configuration published in unit tests: still inactive.
    if config::get().is_none() {
        assert!(config::active().is_none());
    }
    config::set_enabled(false);
}
