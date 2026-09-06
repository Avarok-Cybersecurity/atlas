// SPDX-License-Identifier: AGPL-3.0-only

//! `WeightStore` teardown + FP8 KV-scale-count tests — hoisted from
//! `weights.rs` to keep it under the 500 LoC cap.

use super::*;
use crate::gpu::mock::MockGpuBackend;
use atlas_core::scope::{ModelResource, Teardown};
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

/// A store holding `n` per-tensor entries plus one arena with `members`
/// 1 KiB views at 256-B slots. Returns the store and the arena base.
fn store_with_arena(gpu: &dyn GpuBackend, n: usize, members: usize) -> (WeightStore, DevicePtr) {
    let mut store = store_with(gpu, n);
    let bytes = members * 1024;
    let base = gpu.alloc(bytes).expect("arena alloc");
    store.adopt_arena(WeightArena {
        base,
        bytes,
        label: "test-arena",
    });
    for i in 0..members {
        store.insert(
            format!("pooled{i}"),
            WeightTensor {
                ptr: base.offset(i * 1024),
                shape: vec![512],
                dtype: WeightDtype::BF16,
            },
        );
    }
    (store, base)
}

/// Arena members are views: release must free the arena ONCE and never
/// pass a member (interior pointer — the mock rejects those, like CUDA) to
/// `gpu.free`. Non-members are still freed per entry.
#[test]
fn release_frees_arena_once_and_members_never() {
    let gpu = MockGpuBackend::new();
    let (mut store, base) = store_with_arena(&gpu, 3, 4);
    assert_eq!(gpu.alloc_count(), 4, "3 per-tensor + 1 arena");
    assert!(store.is_pooled(base.offset(1024)));
    assert!(!store.is_pooled(store.get("w0").unwrap().ptr));
    assert_eq!(store.arena_count(), 1);
    assert_eq!(store.pooled_bytes(), 4096);
    store.release(&gpu).expect("released");
    assert_eq!(gpu.alloc_count(), 0);
    assert_eq!(store.len(), 0);
    assert_eq!(store.arena_count(), 0);
    // Idempotent: the arena list was drained too.
    store.release(&gpu).expect("second");
}

/// Removing a member (the materialize pass does this) and then releasing
/// must not double-free: `release_tensor` is a no-op for the member, and
/// the arena is freed exactly once at release.
#[test]
fn removed_member_is_not_freed_and_release_frees_arena_once() {
    let gpu = MockGpuBackend::new();
    let (mut store, _base) = store_with_arena(&gpu, 2, 3);
    let t = store.remove("pooled1").expect("present");
    assert!(
        !store.release_tensor(&gpu, t).expect("no-op"),
        "member: nothing freed"
    );
    assert_eq!(gpu.alloc_count(), 3, "arena + 2 per-tensor still live");
    // `remove_and_free` reports the same fate.
    assert_eq!(store.remove_and_free(&gpu, "pooled2").unwrap(), Some(false));
    assert_eq!(store.remove_and_free(&gpu, "absent").unwrap(), None);
    // A non-member IS freed.
    assert_eq!(store.remove_and_free(&gpu, "w0").unwrap(), Some(true));
    assert_eq!(gpu.alloc_count(), 2);
    store.release(&gpu).expect("released");
    assert_eq!(gpu.alloc_count(), 0);
}

/// `release_ptr` frees a per-tensor pointer the caller took from the store
/// (the `.weight`-class loader sites) and ignores null.
#[test]
fn release_ptr_frees_non_members_only() {
    let gpu = MockGpuBackend::new();
    let (store, base) = store_with_arena(&gpu, 1, 2);
    assert!(!store.release_ptr(&gpu, DevicePtr::NULL).unwrap());
    assert!(!store.release_ptr(&gpu, base.offset(1024)).unwrap());
    assert!(
        !store.release_ptr(&gpu, base).unwrap(),
        "the base itself is pooled too"
    );
    let w0 = store.get("w0").unwrap().ptr;
    assert!(store.release_ptr(&gpu, w0).unwrap());
    assert_eq!(gpu.alloc_count(), 1, "only the arena remains");
    // A pointer just past the arena is NOT a member.
    assert!(store.arena_of(base.offset(2048)).is_none());
}

/// Two arenas: membership resolves to the right one at both ends.
#[test]
fn arena_of_resolves_between_adjacent_arenas() {
    let gpu = MockGpuBackend::new();
    let mut store = WeightStore::empty();
    let a = gpu.alloc(1024).unwrap();
    let b = gpu.alloc(1024).unwrap();
    for (base, label) in [(a, "a"), (b, "b")] {
        store.adopt_arena(WeightArena {
            base,
            bytes: 1024,
            label,
        });
    }
    assert_eq!(store.arena_of(a.offset(1023)).unwrap().label, "a");
    assert_eq!(store.arena_of(b).unwrap().label, "b");
    assert_eq!(store.arena_of(b.offset(512)).unwrap().label, "b");
    store.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
#[should_panic(expected = "overlaps")]
fn adopting_an_overlapping_arena_panics() {
    let gpu = MockGpuBackend::new();
    let mut store = WeightStore::empty();
    let a = gpu.alloc(4096).unwrap();
    store.adopt_arena(WeightArena {
        base: a,
        bytes: 4096,
        label: "a",
    });
    store.adopt_arena(WeightArena {
        base: a.offset(1024),
        bytes: 1024,
        label: "inside a",
    });
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
