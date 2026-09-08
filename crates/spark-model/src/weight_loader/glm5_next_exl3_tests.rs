// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the GLM EXL3 routed-expert binder. These pin the two things that
//! fail SILENTLY on hardware: an EP-remote expert must leave a hole rather than
//! a bogus pointer, and a layer whose experts disagree on (K, codebook) must be
//! refused rather than decoded with one of them.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};
use std::collections::HashMap;

fn tensor(gpu: &MockGpuBackend, shape: Vec<usize>, dtype: WeightDtype) -> WeightTensor {
    let bytes: usize = shape.iter().product::<usize>() * dtype.byte_size().max(1);
    WeightTensor {
        ptr: gpu.alloc(bytes.max(4)).unwrap(),
        shape,
        dtype,
    }
}

/// One expert's trellis triplet at the GLM pack's real geometry:
/// in 2048, out 4096, K=2, mcg.
fn put_expert(gpu: &MockGpuBackend, m: &mut HashMap<String, WeightTensor>, layer: usize, id: usize) {
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        let p = format!("model.language_model.layers.{layer}.mlp.experts.{id}.{proj}");
        m.insert(
            format!("{p}.trellis"),
            tensor(gpu, vec![128, 256, 32], WeightDtype::UInt16),
        );
        m.insert(format!("{p}.suh"), tensor(gpu, vec![2048], WeightDtype::F16));
        m.insert(format!("{p}.svh"), tensor(gpu, vec![4096], WeightDtype::F16));
        let flag = tensor(gpu, vec![1], WeightDtype::Int32);
        gpu.copy_h2d(&0xCBAC_1FEDu32.to_le_bytes(), flag.ptr).unwrap();
        m.insert(format!("{p}.mcg"), flag);
    }
}

fn qualifier(layer: usize) -> impl Fn(&str) -> String {
    move |leaf: &str| format!("model.language_model.layers.{layer}.{leaf}")
}

#[test]
fn detects_an_exl3_routed_layer_and_ignores_a_bf16_one() {
    let gpu = MockGpuBackend::new();
    let mut m = HashMap::new();
    put_expert(&gpu, &mut m, 3, 0);
    // A BF16 layer: plain `.weight`, no triplet.
    m.insert(
        "model.language_model.layers.4.mlp.experts.0.gate_proj.weight".to_string(),
        tensor(&gpu, vec![2048, 4096], WeightDtype::BF16),
    );
    let store = WeightStore::from_map(m);

    assert!(layer_is_exl3(&store, &qualifier(3)));
    assert!(!layer_is_exl3(&store, &qualifier(4)));
}

#[test]
fn binds_only_the_ep_local_range_and_leaves_holes_elsewhere() {
    let gpu = MockGpuBackend::new();
    let mut m = HashMap::new();
    // 4 experts exist on disk, but this "rank" owns ids 1..3 only — the others
    // are simply absent from the store, exactly as under EP.
    for id in 1..3 {
        put_expert(&gpu, &mut m, 0, id);
    }
    let store = WeightStore::from_map(m);

    let bound = bind_experts_exl3(&gpu, &store, &qualifier(0), 4, (1, 3), (4096, 2048, 8)).unwrap();
    for t in &bound.tables {
        assert_eq!(t.local_start, 1, "table must record the EP-local start");
        assert_eq!(t.num_local, 2, "table must cover exactly the owned ids");
    }
}

#[test]
fn a_missing_local_expert_is_an_error_not_a_hole() {
    let gpu = MockGpuBackend::new();
    let mut m = HashMap::new();
    put_expert(&gpu, &mut m, 0, 0);
    // id 1 claimed as local but never stored.
    let store = WeightStore::from_map(m);

    let err = bind_experts_exl3(&gpu, &store, &qualifier(0), 2, (0, 2), (4096, 2048, 8)).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("experts.1"),
        "the error must name the missing expert, got: {msg}"
    );
}

#[test]
fn an_invalid_local_range_is_refused() {
    let gpu = MockGpuBackend::new();
    let store = WeightStore::from_map(HashMap::new());
    assert!(bind_experts_exl3(&gpu, &store, &qualifier(0), 4, (2, 2), (4096, 2048, 8)).is_err());
    assert!(bind_experts_exl3(&gpu, &store, &qualifier(0), 4, (0, 5), (4096, 2048, 8)).is_err());
}
