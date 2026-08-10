// SPDX-License-Identifier: AGPL-3.0-only

//! Type-surface seam tests over hand-built (no-GPU) [`LoraWeights`]: name→slot
//! resolution + active-slot mirror, slot-generation freshening of the adapter
//! id, and the ref_count acquire/release balance + busy gate. Types resolve
//! through the `crate::lora` facade.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize};

use atlas_core::config::PeftAdapterConfig;
use spark_runtime::gpu::DevicePtr;

use crate::lora::*;

#[test]
fn adapter_names_and_slot_resolve() {
    // A hand-built LoraWeights (no GPU) exercising the name→slot resolver
    // and the active-slot mirror the rotation control path relies on.
    let peft = PeftAdapterConfig {
        r: 4,
        lora_alpha: 8.0,
        target_modules: vec!["k_proj".into()],
        use_rslora: false,
        layers_to_transform: None,
    };
    let mk_slot = |name: &str| AdapterSlot {
        name: name.to_string(),
        adapter_config: peft.clone(),
        layers: Vec::new(),
        generation: 0,
    };
    let lw = LoraWeights {
        name: "alpha".into(),
        adapter_config: peft.clone(),
        max_rank: 4,
        max_loras: 8,
        pool: DevicePtr(0),
        pool_bytes: 0,
        slots: vec![mk_slot("alpha"), mk_slot("beta")],
        active: 0,
        tables: BTreeMap::new(),
        scale_table: DevicePtr(0),
        ref_counts: (0..8).map(|_| AtomicUsize::new(0)).collect(),
        pinned: 2,
        last_used: (0..8).map(|_| AtomicU64::new(0)).collect(),
        lru_tick: AtomicU64::new(0),
    };
    assert_eq!(lw.adapter_names(), vec!["alpha", "beta"]);
    assert_eq!(lw.slot_of("beta"), Some(1));
    assert_eq!(lw.slot_of("missing"), None);

    // Task #24: stable adapter_id resolution. Name-derived, `-1 -> active`.
    // Task #25: gen 0 keeps these byte-identical to the #24 name-only value.
    let id_alpha = adapter_id_hash("alpha", 0);
    let id_beta = adapter_id_hash("beta", 0);
    assert_ne!(id_alpha, id_beta, "distinct names must not collide");
    assert_ne!(
        id_alpha, 0,
        "a real adapter must never alias the base sentinel"
    );
    // slot >= 0 keys under that slot's name.
    assert_eq!(lw.adapter_id_for_slot(0), id_alpha);
    assert_eq!(lw.adapter_id_for_slot(1), id_beta);
    // slot == -1 defers to the active adapter (slot 0 = alpha here).
    assert_eq!(lw.adapter_id_for_slot(-1), id_alpha);
    // Out-of-range slot falls back to the base sentinel.
    assert_eq!(lw.adapter_id_for_slot(99), 0);
}

#[test]
fn slot_generation_bump_freshens_adapter_id() {
    // A slot whose contents were re-staged (generation bumped) must yield a
    // DIFFERENT adapter_id than at first load — the #24 residual: reloading
    // different weights under the SAME name no longer warm-hits stale KV.
    let peft = PeftAdapterConfig {
        r: 4,
        lora_alpha: 8.0,
        target_modules: vec!["k_proj".into()],
        use_rslora: false,
        layers_to_transform: None,
    };
    let mut lw = LoraWeights {
        name: "sol".into(),
        adapter_config: peft.clone(),
        max_rank: 4,
        max_loras: 4,
        pool: DevicePtr(0),
        pool_bytes: 0,
        slots: vec![AdapterSlot {
            name: "sol".into(),
            adapter_config: peft.clone(),
            layers: Vec::new(),
            generation: 0,
        }],
        active: 0,
        tables: BTreeMap::new(),
        scale_table: DevicePtr(0),
        ref_counts: (0..4).map(|_| AtomicUsize::new(0)).collect(),
        pinned: 1,
        last_used: (0..4).map(|_| AtomicU64::new(0)).collect(),
        lru_tick: AtomicU64::new(0),
    };
    let id_v1 = lw.adapter_id_for_slot(0);
    assert_eq!(id_v1, adapter_id_hash("sol", 0));
    // Simulate a same-name content swap: bump generation (what the two
    // content-replacing swaps do), name unchanged.
    lw.slots[0].generation = lw.slots[0].generation.wrapping_add(1);
    let id_v2 = lw.adapter_id_for_slot(0);
    assert_ne!(id_v1, id_v2, "re-staged slot must yield a fresh id");
    assert_eq!(id_v2, adapter_id_hash("sol", 1));
}

#[test]
fn ref_count_acquire_release_balance_and_busy_gate() {
    // Task #25 ref_count invariants on a hand-built (no-GPU) LoraWeights.
    let peft = PeftAdapterConfig {
        r: 4,
        lora_alpha: 8.0,
        target_modules: vec!["k_proj".into()],
        use_rslora: false,
        layers_to_transform: None,
    };
    let mk_slot = |name: &str| AdapterSlot {
        name: name.to_string(),
        adapter_config: peft.clone(),
        layers: Vec::new(),
        generation: 0,
    };
    let lw = LoraWeights {
        name: "alpha".into(),
        adapter_config: peft.clone(),
        max_rank: 4,
        max_loras: 4,
        pool: DevicePtr(0),
        pool_bytes: 0,
        slots: vec![mk_slot("alpha"), mk_slot("beta")],
        active: 1, // active != 0 so we can prove `-1 -> active` resolution
        tables: BTreeMap::new(),
        scale_table: DevicePtr(0),
        ref_counts: (0..4).map(|_| AtomicUsize::new(0)).collect(),
        pinned: 2,
        last_used: (0..4).map(|_| AtomicU64::new(0)).collect(),
        lru_tick: AtomicU64::new(0),
    };

    // acquire(0) returns the resolved index 0 and increments its counter.
    assert_eq!(lw.acquire_slot(0), 0);
    assert_eq!(lw.slot_ref_count(0), 1);
    // The busy gate (exact read the swap bail uses) now fires for slot 0.
    assert!(lw.slot_ref_count(0) > 0);
    assert_eq!(lw.slot_ref_count(1), 0, "other slots untouched");

    // -1 resolves to active (=1) and increments slot 1.
    assert_eq!(lw.acquire_slot(-1), 1);
    assert_eq!(lw.slot_ref_count(1), 1);

    // Two seqs on slot 0.
    assert_eq!(lw.acquire_slot(0), 0);
    assert_eq!(lw.slot_ref_count(0), 2);

    // Release by the RESOLVED index; balance returns each to 0.
    lw.release_slot(0);
    assert_eq!(lw.slot_ref_count(0), 1);
    lw.release_slot(0);
    assert_eq!(lw.slot_ref_count(0), 0);
    assert!(lw.slot_ref_count(0) == 0, "gate clears after full release");
    lw.release_slot(1);
    assert_eq!(lw.slot_ref_count(1), 0);

    // Saturating: a stray double-release cannot wrap below 0.
    lw.release_slot(0);
    assert_eq!(lw.slot_ref_count(0), 0);

    // Out-of-range / nothing-acquired paths are no-ops.
    assert_eq!(lw.acquire_slot(99), -1, "bad slot acquires nothing");
    lw.release_slot(-1); // no-op
    assert_eq!(lw.slot_ref_count(99), 0);
}
