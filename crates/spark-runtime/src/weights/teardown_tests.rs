// SPDX-License-Identifier: AGPL-3.0-only

//! `WeightStore` teardown + FP8 KV-scale-count tests — hoisted from
//! `weights.rs` to keep it under the 500 LoC cap.

use super::*;
use crate::gpu::mock::MockGpuBackend;
use avarok_core::scope::{ModelResource, Teardown};
use std::collections::HashMap;

fn store_with(gpu: &dyn GpuBackend, n: usize) -> WeightStore {
    let mut map = HashMap::new();
    for i in 0..n {
        map.insert(
            format!("w{i}"),
            WeightTensor {
                ptr: gpu.alloc(1024).expect("alloc"),
                shape: vec![16, 16],
                dtype: WeightDtype::BF16,
            },
        );
    }
    WeightStore::from_map(map)
}

#[test]
fn releasing_frees_every_tensor() {
    let gpu = MockGpuBackend::new();
    let mut store = store_with(&gpu, 8);
    assert_eq!(gpu.alloc_count(), 8);
    store.release(&gpu).expect("released");
    assert_eq!(gpu.alloc_count(), 0, "every weight was freed");
    assert_eq!(store.len(), 0, "and the map does not hold dead pointers");
}

/// The contract says idempotent: the host calls it, and a `Drop` backstop
/// may call it again. A second call must not double-free.
#[test]
fn releasing_twice_is_harmless() {
    let gpu = MockGpuBackend::new();
    let mut store = store_with(&gpu, 4);
    store.release(&gpu).expect("first");
    store.release(&gpu).expect("second");
    assert_eq!(gpu.alloc_count(), 0);
}

/// `fp8_kv_scale_count` counts exactly the `*.k_scale` tensors — one per
/// attention layer in checkpoints that ship calibrated FP8 KV scales —
/// and ignores `v_scale` (paired 1:1 with `k_scale`, counting both would
/// double-report) and lookalike suffixes.
#[test]
fn fp8_kv_scale_count_counts_only_k_scale_tensors() {
    let gpu = MockGpuBackend::new();
    let tensor = || WeightTensor {
        ptr: gpu.alloc(1024).expect("alloc"),
        shape: vec![1],
        dtype: WeightDtype::BF16,
    };
    let mut map = HashMap::new();
    for name in [
        "model.layers.0.self_attn.k_scale",
        "model.layers.0.self_attn.v_scale",
        "model.layers.7.self_attn.k_scale",
        "model.layers.7.self_attn.v_scale",
        "model.layers.0.self_attn.q_proj.weight",
        // Lookalikes that must NOT count: no dot before the suffix, and a
        // different scale kind entirely.
        "model.layers.0.self_attn.attnk_scale",
        "model.layers.0.mlp.weight_scale",
    ] {
        map.insert(name.to_string(), tensor());
    }
    let store = WeightStore::from_map(map);
    assert_eq!(store.fp8_kv_scale_count(), 2);
}

/// A checkpoint without shipped KV scales reports zero — the case where
/// serve logs the "needs calibration or a non-FP8 KV dtype" warning.
#[test]
fn fp8_kv_scale_count_zero_without_scales() {
    let gpu = MockGpuBackend::new();
    let store = store_with(&gpu, 4);
    assert_eq!(store.fp8_kv_scale_count(), 0);
}

/// Reverse order, and one failure does not abandon the rest — the whole
/// reason `Teardown` exists rather than `Drop`.
#[test]
fn teardown_releases_in_reverse_registration_order() {
    let gpu = MockGpuBackend::new();
    let mut teardown: Teardown<dyn GpuBackend> = Teardown::new();
    teardown.push(Box::new(store_with(&gpu, 3)));
    teardown.push(Box::new(store_with(&gpu, 5)));
    assert_eq!(gpu.alloc_count(), 8);
    teardown.release_all(&gpu).expect("released");
    assert_eq!(gpu.alloc_count(), 0);
    assert!(teardown.is_empty());
}

/// #736/#915: a buffer a loader DERIVED from these tensors must be released
/// here, not left for `AvarokCudaBackend::sweep_unreleased` to reclaim unowned.
///
/// The mock backend's live-allocation count is the same instrument the CUDA
/// ledger is: "every allocation this backend made and nobody released".
#[test]
fn releasing_frees_adopted_derived_buffers_too() {
    let gpu = MockGpuBackend::new();
    let mut store = store_with(&gpu, 4);
    // Two derived copies per tensor, the shape the dense loader produces:
    // a fused concat and its widened block-scale grid.
    for _ in 0..4 {
        store
            .derived()
            .adopt("fused concat", gpu.alloc(2048).expect("alloc"), 2048);
        store
            .derived()
            .adopt("block scale", gpu.alloc(64).expect("alloc"), 64);
    }
    assert_eq!(gpu.alloc_count(), 12);
    assert_eq!(store.derived().len(), 8);
    assert_eq!(store.derived().bytes(), 4 * (2048 + 64));

    store.release(&gpu).expect("released");
    assert_eq!(
        gpu.alloc_count(),
        0,
        "an owned derived buffer must leave nothing for the teardown sweep"
    );
    assert!(store.derived().is_empty());
}

/// The negative control, and the pre-#915 state: a derived buffer nobody
/// adopted survives `release` and is exactly what the H100 sweep reported as
/// "28.01 GB ... had no owner".
#[test]
fn an_unadopted_derived_buffer_is_what_the_sweep_would_report() {
    let gpu = MockGpuBackend::new();
    let mut store = store_with(&gpu, 2);
    let orphan = gpu.alloc(4096).expect("alloc");
    store.release(&gpu).expect("released");
    assert_eq!(
        gpu.alloc_count(),
        1,
        "the orphan outlives teardown — adopt it via `store.derived()`"
    );
    gpu.free(orphan).expect("freed");
}

/// Releasing twice must not double-free an adopted buffer either.
#[test]
fn releasing_twice_is_harmless_for_derived_buffers() {
    let gpu = MockGpuBackend::new();
    let mut store = store_with(&gpu, 1);
    store
        .derived()
        .adopt("twin", gpu.alloc(128).expect("alloc"), 128);
    store.release(&gpu).expect("released");
    store.release(&gpu).expect("released again");
    assert_eq!(gpu.alloc_count(), 0);
}

// ── release-on-consume (`release_tensor`) ────────────────────────────────
//
// The invariant every one of these guards: a released tensor's pointer is
// freed memory. It must never be handed out, never be freed a second time,
// and never be counted as resident.

#[test]
fn release_tensor_frees_the_allocation_and_reports_the_bytes() {
    let gpu = MockGpuBackend::new();
    let store = store_with(&gpu, 3);
    assert_eq!(gpu.alloc_count(), 3);
    // 16 x 16 BF16 = 512 bytes, not the 1024 the mock was asked for: the
    // ledger's number is the SHAPE, which is what the residency table adds up.
    assert_eq!(store.release_tensor(&gpu, "w1").expect("released"), 512);
    assert_eq!(gpu.alloc_count(), 2, "exactly one allocation went away");
    assert_eq!(store.released_count(), 1);
    assert_eq!(store.released_bytes(), 512);
}

#[test]
fn a_released_tensor_is_reported_as_released_not_as_missing() {
    let gpu = MockGpuBackend::new();
    let store = store_with(&gpu, 2);
    store.release_tensor(&gpu, "w0").expect("released");
    let msg = match store.get("w0") {
        Ok(_) => panic!("a freed pointer must not be handed out"),
        Err(e) => format!("{e}"),
    };
    assert!(msg.contains("RELEASED"), "got: {msg}");
    assert!(
        msg.contains("AVAROK_LOAD_RELEASE_SOURCES"),
        "the message must name the knob that turns this off: {msg}"
    );
    // And a genuinely absent name still says so, so the two faults stay
    // distinguishable in a log.
    let missing = match store.get("nope") {
        Ok(_) => panic!("an absent name must not resolve"),
        Err(e) => format!("{e}"),
    };
    assert!(missing.contains("not found in store"), "got: {missing}");
}

#[test]
fn a_released_tensor_leaves_contains_len_and_resident_bytes() {
    let gpu = MockGpuBackend::new();
    let store = store_with(&gpu, 4);
    assert_eq!(store.resident_bytes(), 4 * 512);
    store.release_tensor(&gpu, "w2").expect("released");
    assert!(
        !store.contains("w2"),
        "a caller deciding whether to read it gets 'no'"
    );
    assert!(store.contains("w3"));
    assert_eq!(store.len(), 3);
    assert_eq!(store.resident_bytes(), 3 * 512);
    assert_eq!(store.total_bytes(), 3 * 512);
    let names: Vec<&str> = store.names().collect();
    assert_eq!(names.len(), 3);
    assert!(!names.contains(&"w2"));
}

#[test]
fn releasing_the_same_name_twice_frees_once() {
    let gpu = MockGpuBackend::new();
    let store = store_with(&gpu, 2);
    assert_eq!(store.release_tensor(&gpu, "w0").expect("first"), 512);
    assert_eq!(
        store.release_tensor(&gpu, "w0").expect("second"),
        0,
        "the second call must be a no-op, not a double free"
    );
    assert_eq!(gpu.alloc_count(), 1);
    assert_eq!(
        store.released_bytes(),
        512,
        "and must not be double counted"
    );
}

#[test]
fn releasing_an_unknown_name_is_a_no_op() {
    let gpu = MockGpuBackend::new();
    let store = store_with(&gpu, 1);
    assert_eq!(store.release_tensor(&gpu, "not-here").expect("ok"), 0);
    assert_eq!(gpu.alloc_count(), 1);
    assert_eq!(store.released_count(), 0);
}

/// Teardown is where a double free would actually fire: the map still holds
/// the entry, so a `release` that did not filter would free it again.
#[test]
fn teardown_does_not_free_a_tensor_that_was_released_on_consume() {
    let gpu = MockGpuBackend::new();
    let mut store = store_with(&gpu, 5);
    store.release_tensor(&gpu, "w0").expect("released");
    store.release_tensor(&gpu, "w4").expect("released");
    assert_eq!(gpu.alloc_count(), 3);
    store.release(&gpu).expect("teardown");
    assert_eq!(gpu.alloc_count(), 0, "the remaining three, and only those");
}

/// `prune_after_load` builds its doomed set from checkpoint NAMES and has no
/// way to know a release site already claimed one.
#[test]
fn free_matching_skips_a_tensor_that_was_released_on_consume() {
    let gpu = MockGpuBackend::new();
    let mut store = store_with(&gpu, 4);
    store.release_tensor(&gpu, "w1").expect("released");
    let (count, bytes) = store.free_matching(&gpu, |_| true).expect("pruned");
    assert_eq!(count, 3, "w1 was already gone");
    assert_eq!(bytes, 3 * 512);
    assert_eq!(gpu.alloc_count(), 0);
}
