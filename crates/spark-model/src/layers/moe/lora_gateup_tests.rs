// SPDX-License-Identifier: AGPL-3.0-only

//! Host tests for the per-proj expert route builder that feeds the gate/up
//! folds. Uses `MockGpuBackend` (records `copy_h2d`) to read the packed A/scale
//! tables back; asserts gate/up carry `k_in=hidden`/`n_out=moe_inter` (the
//! TRANSPOSE of down), that a mixed-proj layer yields three INDEPENDENT
//! `n_experts` table lengths, and that each proj's table holds only its own
//! addresses. GPU-free (mock backend).

use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::moe::MoeLayer;
use crate::layers::ops::lora_delta::LoraPair;
use crate::lora::{ExpertLoraLayer, ExpertProj};
use crate::weight_map::DenseWeight;

// GPU-free LoraPair (A addr = tag, B addr = tag+1, scale 0.5, pooled rank 16).
fn dummy_pair(tag: u64, k_in: u32, n_out: u32) -> LoraPair {
    LoraPair {
        a: DenseWeight {
            weight: DevicePtr(tag),
        },
        b: DenseWeight {
            weight: DevicePtr(tag + 1),
        },
        rank: 8,
        k_in,
        n_out,
        scale: 0.5,
        max_rank: 16,
    }
}

fn u64s(gpu: &MockGpuBackend, p: DevicePtr, n: usize) -> Vec<u64> {
    let mut b = vec![0u8; n * 8];
    gpu.copy_d2h(p, &mut b).unwrap();
    b.chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn f32s(gpu: &MockGpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; n * 4];
    gpu.copy_d2h(p, &mut b).unwrap();
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// hidden=2048, moe_inter=512 (matches the target_tests factory cfg dims).
const H: u32 = 2048;
const INTER: u32 = 512;

fn gate_pair(tag: u64) -> LoraPair {
    // gate/up dims: (out=inter, in=hidden) => LoraPair k_in=hidden, n_out=inter.
    dummy_pair(tag, H, INTER)
}
fn down_pair(tag: u64) -> LoraPair {
    // down dims: (out=hidden, in=inter) => LoraPair k_in=inter, n_out=hidden.
    dummy_pair(tag, INTER, H)
}

#[test]
fn gate_route_dims_are_hidden_to_inter() {
    let gpu = MockGpuBackend::new();
    let mut el = ExpertLoraLayer::default();
    el.pairs.insert((5, ExpertProj::Gate), gate_pair(0x100));
    let route = MoeLayer::build_expert_route(&el, ExpertProj::Gate, &gpu)
        .unwrap()
        .expect("gate pair present => Some route");
    assert_eq!(route.k_in, H); // shrink contraction = hidden
    assert_eq!(route.n_out, INTER); // expand output = moe_inter
    assert_eq!(route.max_rank, 16);
    assert_eq!(route.n_experts, 6); // max adapted id (5) + 1
    // Table holds the gate A/B/scale only at index 5, zeros elsewhere.
    assert_eq!(u64s(&gpu, route.a_table, 6), vec![0, 0, 0, 0, 0, 0x100]);
    assert_eq!(u64s(&gpu, route.b_table, 6), vec![0, 0, 0, 0, 0, 0x101]);
    assert_eq!(
        f32s(&gpu, route.scale_table, 6),
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    );
}

#[test]
fn down_route_is_the_transpose_of_gate() {
    let gpu = MockGpuBackend::new();
    let mut el = ExpertLoraLayer::default();
    el.pairs.insert((2, ExpertProj::Down), down_pair(0x200));
    let route = MoeLayer::build_expert_route(&el, ExpertProj::Down, &gpu)
        .unwrap()
        .expect("down pair present => Some route");
    assert_eq!(route.k_in, INTER); // down contracts over moe_inter
    assert_eq!(route.n_out, H); // ...and outputs hidden — transpose of gate/up
    assert_eq!(route.n_experts, 3);
}

#[test]
fn mixed_proj_layer_yields_three_independent_tables() {
    let gpu = MockGpuBackend::new();
    let mut el = ExpertLoraLayer::default();
    // Distinct max ids per proj => distinct table lengths.
    el.pairs.insert((5, ExpertProj::Gate), gate_pair(0x10));
    el.pairs.insert((3, ExpertProj::Up), gate_pair(0x20));
    el.pairs.insert((7, ExpertProj::Down), down_pair(0x30));

    let gate = MoeLayer::build_expert_route(&el, ExpertProj::Gate, &gpu)
        .unwrap()
        .unwrap();
    let up = MoeLayer::build_expert_route(&el, ExpertProj::Up, &gpu)
        .unwrap()
        .unwrap();
    let down = MoeLayer::build_expert_route(&el, ExpertProj::Down, &gpu)
        .unwrap()
        .unwrap();
    assert_eq!(gate.n_experts, 6); // gate id 5 + 1
    assert_eq!(up.n_experts, 4); // up id 3 + 1
    assert_eq!(down.n_experts, 8); // down id 7 + 1
    // Gate/up are hidden->inter; down is the transpose.
    assert_eq!((gate.k_in, gate.n_out), (H, INTER));
    assert_eq!((up.k_in, up.n_out), (H, INTER));
    assert_eq!((down.k_in, down.n_out), (INTER, H));
    // Each proj's table carries only its own address (no cross-proj bleed).
    assert_eq!(u64s(&gpu, up.a_table, 4), vec![0, 0, 0, 0x20]);
}

#[test]
fn absent_proj_returns_none() {
    let gpu = MockGpuBackend::new();
    let mut el = ExpertLoraLayer::default();
    el.pairs.insert((1, ExpertProj::Down), down_pair(0x1));
    // No Gate/Up pairs installed => those routes are None (down-only adapter).
    assert!(
        MoeLayer::build_expert_route(&el, ExpertProj::Gate, &gpu)
            .unwrap()
            .is_none()
    );
    assert!(
        MoeLayer::build_expert_route(&el, ExpertProj::Up, &gpu)
            .unwrap()
            .is_none()
    );
    assert!(
        MoeLayer::build_expert_route(&el, ExpertProj::Down, &gpu)
            .unwrap()
            .is_some()
    );
}
