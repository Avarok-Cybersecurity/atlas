// SPDX-License-Identifier: AGPL-3.0-only

//! Layout + allocation tests for the EXL3 weight pool (mock backend).

use std::collections::HashSet;

use super::*;
use crate::gpu::mock::MockGpuBackend;
use crate::weights::WeightDtype;

fn meta(name: &str, dtype: WeightDtype, shape: Vec<usize>, len: usize, off: u64) -> TensorMeta {
    TensorMeta {
        name: name.to_string(),
        dtype,
        from_f16: false,
        shape,
        abs_offset: off,
        len,
    }
}

/// One K=4 [2560 -> 640] quartet in file order, followed by a bystander.
fn quartet(prefix: &str, base_off: u64) -> Vec<TensorMeta> {
    vec![
        meta(
            &format!("{prefix}.trellis"),
            WeightDtype::UInt16,
            vec![160, 40, 64],
            819_200,
            base_off,
        ),
        meta(
            &format!("{prefix}.suh"),
            WeightDtype::F16,
            vec![2560],
            5120,
            base_off + 819_200,
        ),
        meta(
            &format!("{prefix}.svh"),
            WeightDtype::F16,
            vec![640],
            1280,
            base_off + 824_320,
        ),
        meta(
            &format!("{prefix}.mul1"),
            WeightDtype::Int32,
            vec![],
            4,
            base_off + 825_600,
        ),
    ]
}

fn set(prefixes: &[&str]) -> HashSet<String> {
    prefixes.iter().map(|s| s.to_string()).collect()
}

#[test]
fn exl3_prefix_strips_exactly_the_four_suffixes() {
    assert_eq!(exl3_prefix("a.b.trellis"), Some("a.b"));
    assert_eq!(exl3_prefix("a.b.suh"), Some("a.b"));
    assert_eq!(exl3_prefix("a.b.svh"), Some("a.b"));
    assert_eq!(exl3_prefix("a.b.mul1"), Some("a.b"));
    assert_eq!(exl3_prefix("a.b.weight"), None);
    assert_eq!(exl3_prefix("a.b.trellis_bias"), None);
}

/// The GB10 model measured in `granularity.md`: 512-B rounding under 2 MiB,
/// 2 MiB / floor(2 MiB / size) per object, dedicated at >= 2 MiB.
#[test]
fn measured_footprint_matches_the_granularity_table() {
    const MIB2: usize = 2 * 1024 * 1024;
    assert_eq!(measured_alloc_footprint(4), 512);
    assert_eq!(measured_alloc_footprint(1280), 1536);
    // 409 objects per chunk; the 7-byte share of the chunk tail is charged.
    assert_eq!(measured_alloc_footprint(5120), MIB2 / 409);
    assert_eq!(measured_alloc_footprint(819_200), MIB2 / 2); // K=4: two per chunk
    assert_eq!(measured_alloc_footprint(614_400), MIB2 / 3); // K=3: three per chunk
    assert_eq!(measured_alloc_footprint(409_600), MIB2 / 5); // K=2: five per chunk
    assert_eq!(measured_alloc_footprint(1_228_800), MIB2); // K=6 shared_expert: one
    assert_eq!(measured_alloc_footprint(MIB2), MIB2);
    assert_eq!(measured_alloc_footprint(MIB2 + 4096), MIB2 + 4096);
}

#[test]
fn plan_pools_only_admitted_prefixes_with_aligned_slots() {
    let mut tensors = quartet("model.layers.0.mlp.experts.0.gate_proj", 0);
    tensors.extend(quartet(
        "model.layers.0.mlp.shared_expert.up_proj",
        1_000_000,
    ));
    tensors.push(meta(
        "model.layers.0.norm.weight",
        WeightDtype::BF16,
        vec![2560],
        5120,
        2_000_000,
    ));
    let plan = PoolPlan::build(&tensors, &set(&["model.layers.0.mlp.experts.0.gate_proj"]));
    assert_eq!(plan.pooled_tensors, 4);
    assert_eq!(plan.pooled_bytes, 819_200 + 5120 + 1280 + 4);
    assert_eq!(plan.arena_count(), 2, "trellis + aux classes");
    // K=4 trellis: 224 KiB of chunk tail; aux: 512-B rounding + tail share.
    assert_eq!(
        plan.padding_avoided,
        (1_048_576 - 819_200) + (2_097_152 / 409 - 5120) + (1536 - 1280) + (512 - 4)
    );
    let trellis = &plan.classes[0];
    assert_eq!(trellis.class, PoolClass::Trellis);
    assert_eq!(trellis.slots, vec![(0, 0)]);
    assert_eq!(trellis.bytes, 819_200); // already a multiple of 256
    let aux = &plan.classes[1];
    assert_eq!(aux.class, PoolClass::Aux);
    assert_eq!(aux.slots, vec![(1, 0), (2, 5120), (3, 5120 + 1280)]);
    assert_eq!(
        aux.bytes,
        5120 + 1280 + 256,
        "the 4-byte flag rounds up to one slot"
    );
}

#[test]
fn empty_prefix_set_pools_nothing() {
    let tensors = quartet("lm_head", 0);
    let plan = PoolPlan::build(&tensors, &HashSet::new());
    assert!(plan.is_empty());
    assert_eq!(plan.pooled_tensors, 0);
    let gpu = MockGpuBackend::new();
    let mut logged = false;
    assert!(
        ShardArenas::alloc(&gpu, &plan, tensors.len(), &mut logged)
            .unwrap()
            .is_none()
    );
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn arenas_are_one_alloc_per_class_with_slot_views() {
    let tensors = quartet("lm_head", 0);
    let plan = PoolPlan::build(&tensors, &set(&["lm_head"]));
    let gpu = MockGpuBackend::new();
    let mut logged = false;
    let arenas = ShardArenas::alloc(&gpu, &plan, tensors.len(), &mut logged)
        .unwrap()
        .expect("arenas");
    assert_eq!(gpu.alloc_count(), 2);
    assert!(!logged);
    let trellis_base = arenas.arenas[0].base;
    let aux_base = arenas.arenas[1].base;
    assert_eq!(arenas.slot(0), Some(trellis_base));
    assert_eq!(arenas.slot(1), Some(aux_base));
    assert_eq!(arenas.slot(2), Some(aux_base.offset(5120)));
    assert_eq!(arenas.slot(3), Some(aux_base.offset(6400)));
    assert_eq!(arenas.slot(4), None);
    let labels: Vec<_> = arenas.into_arenas().into_iter().map(|a| a.label).collect();
    assert_eq!(labels, vec!["exl3-trellis", "exl3-aux"]);
}

/// A failed arena allocation falls back to the per-tensor path for the
/// shard: `None`, logged once, and any arena already allocated is freed.
#[test]
fn arena_alloc_failure_frees_partial_and_falls_back() {
    // Make the aux class the larger one so the trellis arena succeeds first.
    let tensors = vec![
        meta("p.trellis", WeightDtype::UInt16, vec![1, 1, 64], 256, 0),
        meta("p.suh", WeightDtype::F16, vec![2048], 4096, 256),
        meta("p.svh", WeightDtype::F16, vec![2048], 4096, 4352),
    ];
    let plan = PoolPlan::build(&tensors, &set(&["p"]));
    let gpu = MockGpuBackend::new();
    gpu.set_max_allocation_bytes(1024);
    let mut logged = false;
    let r = ShardArenas::alloc(&gpu, &plan, tensors.len(), &mut logged).unwrap();
    assert!(r.is_none());
    assert!(logged, "fallback logged once");
    assert_eq!(gpu.alloc_count(), 0, "the trellis arena was rolled back");
    // Second failure does not log again.
    let mut again = logged;
    let _ = ShardArenas::alloc(&gpu, &plan, tensors.len(), &mut again).unwrap();
    assert!(again);
}
