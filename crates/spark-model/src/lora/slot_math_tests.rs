// SPDX-License-Identifier: AGPL-3.0-only

//! Slot/offset-math seam tests: pool-slot byte layout + offsets, per-step
//! `seq_slot` build, scale-table values, victim selection, and the
//! routed-prefill predicate + pair selection. Cross-seam bodies resolve types
//! through the `crate::lora` facade.

use atlas_core::config::{LayerType, PeftAdapterConfig};
use spark_runtime::weights::WeightStore;

use crate::lora::test_support::*;
use crate::lora::*;

#[test]
fn slot_base_is_k_times_slot_bytes() {
    let cfg = cfg();
    let mr = 16;
    let sb = pool_slot_bytes(&cfg, mr);
    for k in 0..8 {
        assert_eq!(slot_base_offset(k, &cfg, mr), k * sb);
    }
}

#[test]
fn pool_slot_bytes_absolute_golden() {
    // Absolute byte count for the factory config @ max_rank=16. Pins the
    // layout absolutely (a dims/pad change is caught), not just the
    // slot_base == k·slot_bytes relation.
    //
    // CHANGED from 16_515_072 when the pool stopped being "full-attention
    // layers × ALL modules". That shape reserved q/k/v/o on the layers that
    // have them but reserved NOTHING for the dense FFN a hybrid carries on
    // its linear-attention layers, so those pairs had nowhere to land and
    // were silently dropped — loaded, never packed, never applied. The pool
    // is now Σ over EVERY layer of the modules that layer can actually carry
    // (`LoraModule::applies_to_layer`), which also makes room for the GDN
    // `out_proj` on linear-attention layers.
    //
    // Smaller, not larger, on this config: the linear-attention layers gain
    // gate/up/down + out_proj but the full-attention layers no longer reserve
    // out_proj space they cannot use, and the old shape was reserving all
    // seven modules on all twelve attention layers.
    let cfg = cfg();
    assert_eq!(pool_slot_bytes(&cfg, 16), 15_335_424);
}

#[test]
fn module_offsets_walk_matches_pack_loop_and_fill_exactly_one_slot() {
    // Reproduce the pack loop's cumulative A-then-B walk (the frozen
    // layout) and assert module_slot_offsets agrees at every step, and the
    // running end lands exactly on pool_slot_bytes (one full slot).
    let cfg = cfg();
    let mr = 16;
    let mut off = 0usize;
    for layer in 0..cfg.num_hidden_layers {
        for module in LoraModule::ALL {
            if !module.applies_to_layer(&cfg, layer) {
                // The offset table must agree that this module is absent here.
                assert_eq!(
                    module_slot_offsets(&cfg, mr, layer, module),
                    None,
                    "layer {layer} {module:?} is not applicable and must have no slot"
                );
                continue;
            }
            let (out, inp) = module.dims(&cfg);
            let a_off = off;
            let b_off = off + mr * inp * BF16_BYTES;
            off = b_off + out * mr * BF16_BYTES;
            assert_eq!(
                module_slot_offsets(&cfg, mr, layer, module),
                Some((a_off, b_off)),
                "layer {layer} {module:?}"
            );
            assert!(a_off < b_off, "A precedes B within a module region");
        }
    }
    assert_eq!(
        off,
        pool_slot_bytes(&cfg, mr),
        "one pass fills exactly one slot"
    );
}

#[test]
fn module_offsets_none_for_non_full_attention_layer() {
    let cfg = cfg();
    assert_eq!(cfg.layer_type(0), LayerType::LinearAttention);
    assert_eq!(module_slot_offsets(&cfg, 16, 0, LoraModule::KProj), None);
}

#[test]
fn slot_boundaries_do_not_overlap() {
    let cfg = cfg();
    let mr = 16;
    let sb = pool_slot_bytes(&cfg, mr);
    // The LAST module in the walk ends exactly at slot_bytes, i.e. flush
    // against slot 1's base. Derived rather than hardcoded: which module comes
    // last depends on the last layer's type (a linear-attention tail layer
    // ends on out_proj, an attention one on down_proj), and hardcoding it
    // would make this test a statement about one config instead of about the
    // packing invariant.
    let (last_layer, last_module) = (0..cfg.num_hidden_layers)
        .flat_map(|l| LoraModule::ALL.iter().map(move |m| (l, *m)))
        .rfind(|(l, m)| m.applies_to_layer(&cfg, *l))
        .expect("some module is packed");
    let (_, b_off) = module_slot_offsets(&cfg, mr, last_layer, last_module).unwrap();
    let (out, _) = last_module.dims(&cfg);
    assert_eq!(b_off + out * mr * BF16_BYTES, sb);
    assert_eq!(slot_base_offset(1, &cfg, mr), sb);
}

#[test]
fn scale_table_values_per_slot_and_padded() {
    // scaling() = alpha/r (no rslora); alpha/sqrt(r) under rslora. The
    // scale table carries one f32 per slot, 0.0 for unpacked slots, in
    // slot order — exactly what bgmv indexes by seq_slot.
    let store = WeightStore::empty();
    let mk = |alpha: f64, r: usize, rslora: bool| LoraAdapterInput {
        name: String::new(),
        store: &store,
        peft: PeftAdapterConfig {
            r,
            lora_alpha: alpha,
            target_modules: vec!["k_proj".into()],
            target_modules_pattern: None,
            use_rslora: rslora,
            layers_to_transform: None,
            trainable_token_indices: Vec::new(),
            modules_to_save: Vec::new(),
            lora_embedding: false,
        },
    };
    let adapters = [mk(16.0, 8, false), mk(16.0, 4, true)];
    let v = scale_table_values(&adapters, 8);
    assert_eq!(v.len(), 8);
    assert_eq!(v[0], (16.0_f64 / 8.0) as f32); // alpha/r
    assert_eq!(v[1], (16.0_f64 / (4.0_f64).sqrt()) as f32); // rslora: alpha/sqrt(r)
    assert!(v[2..].iter().all(|&s| s == 0.0)); // unpacked slots
}

#[test]
fn seq_slot_host_defers_negatives_and_pads() {
    // Two real seqs on explicit slots 1 and 0, one defaulting (-1 -> active=2),
    // padded to 4 (pad rows -1 = base/no delta).
    let slots = [1i32, -1, 0];
    let v = build_seq_slot_host(&slots, 4, 2);
    assert_eq!(v, vec![1, 2, 0, -1]);
}

#[test]
fn seq_slot_uniform_prefill_fills_and_resolves() {
    // Pure core of `TransformerModel::upload_seq_slot_uniform`
    // (single-seq decode count=1, verify count=K, prefill count=m): every
    // row = resolve(adapter_slot, active). Covered across representative
    // counts {1, 4, 32}.
    for &count in &[1usize, 4, 32] {
        // Explicit slot ≥ 0 → every row is that slot (no active fallback).
        let v = build_seq_slot_host(&vec![3i32; count], count, 7);
        assert_eq!(v, vec![3i32; count], "count={count} explicit slot B");
        // Deferred (-1, request has no `adapter` field) → resolves to active,
        // so a single-adapter / no-field run applies the active slot on
        // every row — byte-identical delta to the installed-pair path.
        let v = build_seq_slot_host(&vec![-1i32; count], count, 5);
        assert_eq!(v, vec![5i32; count], "count={count} deferred → active");
        // Explicit slot 0 (naming the active adapter) stays 0.
        let v = build_seq_slot_host(&vec![0i32; count], count, 2);
        assert_eq!(v, vec![0i32; count], "count={count} slot 0");
    }
}

// ── Task #27: pure victim-selection policy ──

#[test]
fn victim_free_first_before_lru() {
    // Cache region starts at slot 2. Slot 3 is a never-filled placeholder;
    // it must be chosen BEFORE evicting any filled slot, even a very-idle one.
    let cache = vec![
        (2, view(true, 0, 1)),   // filled, idle, oldest tick
        (3, view(false, 0, 99)), // never filled → free-first despite newest tick
        (4, view(true, 0, 9)),   // filled, idle
    ];
    assert_eq!(select_victim_slot(&cache), Ok(3));
}

#[test]
fn victim_lru_idle_when_all_filled() {
    // No free slot: evict the idle slot with the smallest last_used tick.
    let cache = vec![
        (2, view(true, 0, 50)), // idle but recently used
        (3, view(true, 1, 5)),  // BUSY — never a victim despite oldest tick
        (4, view(true, 0, 12)), // idle, older than slot 2 → LRU winner
    ];
    assert_eq!(select_victim_slot(&cache), Ok(4));
}

#[test]
fn victim_pool_full_when_all_busy() {
    // Every cache slot has ref_count>0 → retryable PoolFull, never an evict.
    let cache = vec![(2, view(true, 1, 1)), (3, view(true, 2, 2))];
    assert_eq!(select_victim_slot(&cache), Err(VictimError::PoolFull));
}

// ── #30 routed-prefill precision: the two pure correctness invariants
// (right predicate → Some only for a non-active slot; right pair selection
// by GLOBAL layer index + module) — locked without a GPU.

#[test]
fn routed_prefill_slot_predicate() {
    // active = 0, pool of 2 slots.
    assert_eq!(
        routed_prefill_slot_of(-1, 0, 2),
        None,
        "-1 defers to active"
    );
    assert_eq!(
        routed_prefill_slot_of(0, 0, 2),
        None,
        "names the active slot"
    );
    assert_eq!(
        routed_prefill_slot_of(1, 0, 2),
        Some(1),
        "routes to a non-active slot"
    );
    assert_eq!(routed_prefill_slot_of(5, 0, 2), None, "out of range");
    // active = 1: now slot 0 is the non-active one, slot 1 defers.
    assert_eq!(routed_prefill_slot_of(0, 1, 2), Some(0));
    assert_eq!(routed_prefill_slot_of(1, 1, 2), None);
    assert_eq!(routed_prefill_slot_of(-1, 1, 2), None);
}

#[test]
fn select_routed_pair_by_global_index_and_module() {
    // A GLOBAL-layer-indexed slice: only global layer 3 is adapted, with a
    // distinct pair per module so a wrong (layer, module) selection is caught.
    let mut layers: Vec<Option<LoraLayerWeights>> = (0..8).map(|_| None).collect();
    let k = dummy_pair(100, 1024, 512);
    let v = dummy_pair(200, 1024, 512);
    let o = dummy_pair(300, 2048, 1024);
    layers[3] = Some(LoraLayerWeights {
        k_proj: Some(k),
        v_proj: Some(v),
        o_proj: Some(o),
        ..LoraLayerWeights::empty(3)
    });

    // Right pair for the adapted (layer, module) triples.
    assert_eq!(
        select_routed_pair(&layers, 3, LoraModule::KProj).map(|p| p.a.weight.0),
        Some(100)
    );
    assert_eq!(
        select_routed_pair(&layers, 3, LoraModule::VProj).map(|p| p.a.weight.0),
        Some(200)
    );
    assert_eq!(
        select_routed_pair(&layers, 3, LoraModule::OProj).map(|p| p.a.weight.0),
        Some(300)
    );

    // A module the slot does NOT adapt on this layer → None (caller falls
    // back to base for that module, never mis-applies another module's pair).
    assert!(select_routed_pair(&layers, 3, LoraModule::GateProj).is_none());
    // An unadapted layer → None (would be the WRONG layer if indexed by an
    // attention-only counter; this test pins GLOBAL-index selection).
    assert!(select_routed_pair(&layers, 2, LoraModule::KProj).is_none());
    // Out-of-range global index → None, never a panic.
    assert!(select_routed_pair(&layers, 99, LoraModule::KProj).is_none());
}
