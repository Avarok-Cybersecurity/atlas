// SPDX-License-Identifier: AGPL-3.0-only

//! The soundness rule of [`super::consumed_fp8_source`], pinned on a store
//! built by hand. No device, no checkpoint: this is the predicate that decides
//! whether a `gpu.free` lands on a buffer a layer still reads, and it is a
//! pure function of the on-disk dtype.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::WeightTensor;
use std::collections::HashMap;

fn store_with(gpu: &dyn GpuBackend, entries: &[(&str, WeightDtype, &[usize])]) -> WeightStore {
    let mut map = HashMap::new();
    for (name, dtype, shape) in entries {
        let elems: usize = shape.iter().product();
        map.insert(
            (*name).to_owned(),
            WeightTensor {
                ptr: gpu.alloc(elems.max(1) * dtype.byte_size().max(1)).unwrap(),
                shape: shape.to_vec(),
                dtype: *dtype,
            },
        );
    }
    WeightStore::from_map(map)
}

#[test]
fn only_an_fp8_source_is_claimed() {
    let gpu = MockGpuBackend::new();
    let store = store_with(
        &gpu,
        &[
            ("a.q_proj.weight", WeightDtype::FP8E4M3, &[128, 64]),
            // BF16: `dense_auto` hands the STORE pointer through, so a layer
            // may alias it. Never claimed.
            ("a.in_proj_a.weight", WeightDtype::BF16, &[8, 64]),
            // Packed NVFP4 from disk: bound zero-copy by `quantized_v2` and
            // read by every decode step.
            ("a.gate_proj.weight_packed", WeightDtype::UInt8, &[128, 32]),
        ],
    );
    assert!(consumed_fp8_source(&store, "a.q_proj"));
    assert!(!consumed_fp8_source(&store, "a.in_proj_a"));
    assert!(
        !consumed_fp8_source(&store, "a.gate_proj"),
        "a packed-NVFP4 projection has no `.weight` at all"
    );
    assert!(!consumed_fp8_source(&store, "a.does_not_exist"));
}

#[test]
fn release_projections_frees_the_weight_and_keeps_the_scale() {
    let gpu = MockGpuBackend::new();
    let store = store_with(
        &gpu,
        &[
            ("a.q_proj.weight", WeightDtype::FP8E4M3, &[128, 64]),
            ("a.q_proj.weight_scale", WeightDtype::BF16, &[128, 1]),
            ("a.k_proj.weight", WeightDtype::FP8E4M3, &[32, 64]),
            ("a.k_proj.weight_scale", WeightDtype::BF16, &[32, 1]),
            // Not claimed, so it must survive untouched.
            ("a.in_proj_a.weight", WeightDtype::BF16, &[8, 64]),
        ],
    );
    let before = gpu.alloc_count();
    let mut r = SourceReleaser {
        enabled: true,
        store_bytes: 0,
        store_count: 0,
        leaked_bytes: 0,
    };
    let names = ["a.q_proj".to_owned(), "a.k_proj".to_owned()];
    r.release_projections(&store, &gpu, 0, &names).unwrap();
    assert_eq!(
        gpu.alloc_count(),
        before - 2,
        "the two weights, and only those"
    );
    assert_eq!(r.store_count, 2);
    assert_eq!(r.store_bytes, (128 * 64 + 32 * 64) as u64);
    assert!(!store.contains("a.q_proj.weight"));
    assert!(
        store.contains("a.q_proj.weight_scale"),
        "the scale stays: `store.contains` on it is how several predicates \
         decide what a checkpoint IS, and a release must not move that answer"
    );
    assert!(store.contains("a.in_proj_a.weight"));
}

#[test]
fn the_policy_off_path_frees_nothing() {
    let gpu = MockGpuBackend::new();
    let store = store_with(
        &gpu,
        &[("a.q_proj.weight", WeightDtype::FP8E4M3, &[128, 64])],
    );
    let before = gpu.alloc_count();
    let mut r = SourceReleaser {
        enabled: false,
        store_bytes: 0,
        store_count: 0,
        leaked_bytes: 0,
    };
    r.release_projections(&store, &gpu, 0, &["a.q_proj".to_owned()])
        .unwrap();
    assert_eq!(gpu.alloc_count(), before);
    assert_eq!(r.store_bytes, 0);
    assert!(store.contains("a.q_proj.weight"));
}

#[test]
fn releasing_the_same_projection_twice_frees_once() {
    let gpu = MockGpuBackend::new();
    let store = store_with(
        &gpu,
        &[("a.q_proj.weight", WeightDtype::FP8E4M3, &[128, 64])],
    );
    let mut r = SourceReleaser {
        enabled: true,
        store_bytes: 0,
        store_count: 0,
        leaked_bytes: 0,
    };
    let names = ["a.q_proj".to_owned()];
    r.release_projections(&store, &gpu, 0, &names).unwrap();
    // The second call cannot even see it: `consumed_fp8_source` goes through
    // `store.get`, which refuses a released name. Belt to `release_tensor`'s
    // own braces.
    r.release_projections(&store, &gpu, 0, &names).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
    assert_eq!(r.store_count, 1);
    assert_eq!(r.store_bytes, (128 * 64) as u64);
}

#[test]
fn the_summary_names_both_halves_of_the_residency() {
    let gpu = MockGpuBackend::new();
    let store = store_with(
        &gpu,
        &[("a.q_proj.weight", WeightDtype::FP8E4M3, &[128, 64])],
    );
    let mut r = SourceReleaser {
        enabled: true,
        store_bytes: 1_000_000_000,
        store_count: 7,
        leaked_bytes: 0,
    };
    let s = r.summary(&store, &gpu);
    assert!(s.contains("store"), "got: {s}");
    assert!(
        s.contains("released on consume across 7 tensors"),
        "got: {s}"
    );
    assert!(s.contains("layer-owned"), "got: {s}");
    // The leak line appears only when something was actually freed.
    assert!(!s.contains("leaked"), "got: {s}");
    r.note_leaked_free(2_000_000_000);
    assert!(r.summary(&store, &gpu).contains("leaked"));
}
