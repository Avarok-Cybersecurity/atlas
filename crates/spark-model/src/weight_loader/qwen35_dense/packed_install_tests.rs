// SPDX-License-Identifier: AGPL-3.0-only

//! The keep-packed FFN install: detection, the build order, and every refusal, on mock tensors with a
//! recording builder in place of the GGUF kernels.

use super::*;
use crate::port::w4::fake::{FakeW4, log};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::WeightDtype;
use std::collections::HashMap;
use std::sync::Arc;

const H: usize = 256;
const I: usize = 512;
const LP: &str = "model.layers.0";

fn tensor(gpu: &MockGpuBackend, shape: [usize; 2], dtype: WeightDtype) -> WeightTensor {
    let bytes = match dtype {
        WeightDtype::BF16 => shape[0] * shape[1] * 2,
        // Q6_K: 210 bytes per 256 weights.
        _ => shape[0] * shape[1] / 256 * 210,
    };
    WeightTensor {
        ptr: gpu.alloc(bytes.max(1)).expect("mock alloc"),
        shape: shape.to_vec(),
        dtype,
    }
}

/// Layer 0's FFN with the given dtypes (gate, up, down) and shapes.
fn store(gpu: &MockGpuBackend, dtypes: [WeightDtype; 3], shapes: [[usize; 2]; 3]) -> WeightStore {
    let mut m = HashMap::new();
    for ((proj, dt), shape) in FFN_NAMES.iter().zip(dtypes).zip(shapes) {
        m.insert(tensor_name(LP, proj), tensor(gpu, shape, dt));
    }
    WeightStore::from_map(m)
}

const PACKED: [WeightDtype; 3] = [WeightDtype::Q6K; 3];

/// A builder that wraps every tensor as a recording fake of the tensor's shape, and logs the order.
fn fake_builder(
    order: &mut Vec<Vec<usize>>,
) -> impl FnMut(&WeightTensor) -> Result<Option<W4Box>> + '_ {
    let l = log();
    move |t: &WeightTensor| {
        order.push(t.shape.clone());
        Ok(Some(
            Arc::new(FakeW4::new("packed", t.shape[0], t.shape[1], 8, &l)) as W4Box,
        ))
    }
}

#[test]
fn packed_dtypes_are_the_three_gguf_q4_types_and_nothing_else() {
    let gpu = MockGpuBackend::new();
    assert!(is_packed(&tensor(&gpu, [256, 256], WeightDtype::Q6K)));
    for dt in [
        WeightDtype::BF16,
        WeightDtype::FP8E4M3,
        WeightDtype::UInt8,
        WeightDtype::Q2K,
    ] {
        assert!(!is_packed(&tensor(&gpu, [256, 256], dt)), "{dt:?}");
    }
}

#[test]
fn detection_is_any_projection_packed() {
    let gpu = MockGpuBackend::new();
    let shapes = ffn_shapes(H, I);
    assert!(ffn_any_packed(&store(&gpu, PACKED, shapes), LP));
    let bf16 = [WeightDtype::BF16; 3];
    assert!(
        !ffn_any_packed(&store(&gpu, bf16, shapes), LP),
        "a BF16 layer is untouched"
    );
    let partly = [WeightDtype::BF16, WeightDtype::BF16, WeightDtype::Q6K];
    assert!(ffn_any_packed(&store(&gpu, partly, shapes), LP));
    assert!(
        !ffn_any_packed(&store(&gpu, PACKED, shapes), "model.layers.9"),
        "absent layer"
    );
}

#[test]
fn the_boxes_come_out_gate_up_down_with_the_config_shapes() {
    let gpu = MockGpuBackend::new();
    let s = store(&gpu, PACKED, ffn_shapes(H, I));
    let mut order = Vec::new();
    let [g, u, d] = ffn_boxes(&s, LP, H, I, 1, false, fake_builder(&mut order)).unwrap();
    assert_eq!(order, vec![vec![I, H], vec![I, H], vec![H, I]]);
    assert_eq!(
        (g.n(), g.k(), u.n(), u.k(), d.n(), d.k()),
        (I, H, I, H, H, I)
    );
    // The install borrows the store's buffers: every tensor is still allocated.
    for p in FFN_NAMES {
        let t = s.get(&tensor_name(LP, p)).unwrap();
        assert!(gpu.read_alloc(t.ptr).is_some(), "{p} stays the store's");
    }
}

#[test]
fn tensor_parallel_and_lora_are_refused_before_anything_is_built() {
    let gpu = MockGpuBackend::new();
    let s = store(&gpu, PACKED, ffn_shapes(H, I));
    let mut order = Vec::new();
    let e = ffn_boxes(&s, LP, H, I, 2, false, fake_builder(&mut order)).unwrap_err();
    assert!(format!("{e:#}").contains("tensor parallel 2"), "{e:#}");
    let e = ffn_boxes(&s, LP, H, I, 1, true, fake_builder(&mut order)).unwrap_err();
    assert!(format!("{e:#}").contains("LoRA"), "{e:#}");
    assert!(order.is_empty(), "nothing was built");
}

#[test]
fn a_partly_packed_layer_is_refused_by_name() {
    let gpu = MockGpuBackend::new();
    let partly = [WeightDtype::Q6K, WeightDtype::BF16, WeightDtype::Q6K];
    let s = store(&gpu, partly, ffn_shapes(H, I));
    let mut order = Vec::new();
    let e = ffn_boxes(&s, LP, H, I, 1, false, fake_builder(&mut order)).unwrap_err();
    let msg = format!("{e:#}");
    assert!(
        msg.contains("up_proj") && msg.contains("partly packed"),
        "{msg}"
    );
}

#[test]
fn a_shape_that_is_not_the_configs_is_refused() {
    let gpu = MockGpuBackend::new();
    let wrong = [[I, H], [I, H], [H, 2 * I]];
    let s = store(&gpu, PACKED, wrong);
    let mut order = Vec::new();
    let e = ffn_boxes(&s, LP, H, I, 1, false, fake_builder(&mut order)).unwrap_err();
    assert!(format!("{e:#}").contains("down_proj"), "{e:#}");
}

#[test]
fn a_declining_builder_is_a_load_error() {
    let gpu = MockGpuBackend::new();
    let s = store(&gpu, PACKED, ffn_shapes(H, I));
    let e = ffn_boxes(&s, LP, H, I, 1, false, |_t: &WeightTensor| Ok(None)).unwrap_err();
    assert!(format!("{e:#}").contains("no 4-bit kernel"), "{e:#}");
}

#[test]
fn a_builder_returning_the_wrong_shape_is_refused() {
    let gpu = MockGpuBackend::new();
    let s = store(&gpu, PACKED, ffn_shapes(H, I));
    let l = log();
    let e = ffn_boxes(&s, LP, H, I, 1, false, |_t: &WeightTensor| {
        Ok(Some(Arc::new(FakeW4::new("wrong", 8, 8, 8, &l)) as W4Box))
    })
    .unwrap_err();
    assert!(format!("{e:#}").contains("[8x8]"), "{e:#}");
}

#[test]
fn the_production_builder_declines_a_bf16_tensor_without_a_lookup() {
    // The production builder (`port::gguf_q4::from_weight_tensor`) resolves L19's kernels, so the CPU tests
    // above use a recording builder. This pins that it declines a non-packed dtype before any lookup.
    let gpu = MockGpuBackend::new();
    let t = tensor(&gpu, [256, 256], WeightDtype::BF16);
    assert!(
        crate::port::gguf_q4::from_weight_tensor(&gpu, &t)
            .unwrap()
            .is_none()
    );
    assert!(gpu.kernel_lookups_snapshot().is_empty());
}

const AP: &str = "model.layers.3.self_attn";
const LA: &str = "model.layers.0.linear_attn";
// A gated attention: q is [2 * heads * head_dim, hidden]; k and v are [kv heads * head_dim, hidden].
const ATTN_SHAPES: [(usize, usize); 4] = [(1024, H), (256, H), (256, H), (H, 512)];

/// The layer's q/k/v/o (`names`, `prefix`) with the given dtypes and shapes.
fn proj_store<const N: usize>(
    gpu: &MockGpuBackend,
    prefix: &str,
    names: [&str; N],
    dtypes: [WeightDtype; N],
    shapes: [(usize, usize); N],
) -> WeightStore {
    let mut m = HashMap::new();
    for ((proj, dt), (n, k)) in names.iter().zip(dtypes).zip(shapes) {
        m.insert(proj_tensor_name(prefix, proj), tensor(gpu, [n, k], dt));
    }
    WeightStore::from_map(m)
}

#[test]
fn attention_detection_is_any_of_q_k_v_o_packed() {
    let gpu = MockGpuBackend::new();
    let packed = [WeightDtype::Q6K; 4];
    let s = proj_store(&gpu, AP, ATTN_NAMES, packed, ATTN_SHAPES);
    assert!(attn_any_packed(&s, AP));
    let bf16 = [WeightDtype::BF16; 4];
    assert!(!attn_any_packed(
        &proj_store(&gpu, AP, ATTN_NAMES, bf16, ATTN_SHAPES),
        AP
    ));
    let partly = [
        WeightDtype::BF16,
        WeightDtype::BF16,
        WeightDtype::BF16,
        WeightDtype::Q6K,
    ];
    assert!(attn_any_packed(
        &proj_store(&gpu, AP, ATTN_NAMES, partly, ATTN_SHAPES),
        AP
    ));
    assert!(
        !attn_any_packed(&s, "model.layers.9.self_attn"),
        "absent layer"
    );
}

#[test]
fn the_attention_boxes_come_out_q_k_v_o_with_the_config_shapes() {
    let gpu = MockGpuBackend::new();
    let s = proj_store(&gpu, AP, ATTN_NAMES, [WeightDtype::Q6K; 4], ATTN_SHAPES);
    let mut order = Vec::new();
    let [q, k, v, o] = attn_boxes(&s, AP, ATTN_SHAPES, 1, false, fake_builder(&mut order)).unwrap();
    assert_eq!(
        order,
        vec![vec![1024, H], vec![256, H], vec![256, H], vec![H, 512]]
    );
    assert_eq!(
        (q.n(), q.k(), k.n(), v.n(), o.n(), o.k()),
        (1024, H, 256, 256, H, 512)
    );
}

#[test]
fn attention_refusals_name_the_reason_and_build_nothing() {
    let gpu = MockGpuBackend::new();
    let packed = [WeightDtype::Q6K; 4];
    let s = proj_store(&gpu, AP, ATTN_NAMES, packed, ATTN_SHAPES);
    let mut order = Vec::new();
    let e = attn_boxes(&s, AP, ATTN_SHAPES, 2, false, fake_builder(&mut order)).unwrap_err();
    assert!(format!("{e:#}").contains("tensor parallel 2"), "{e:#}");
    let e = attn_boxes(&s, AP, ATTN_SHAPES, 1, true, fake_builder(&mut order)).unwrap_err();
    assert!(format!("{e:#}").contains("LoRA"), "{e:#}");
    assert!(order.is_empty(), "nothing was built");
    // A partly packed layer, a wrong shape, a declining builder.
    let partly = [
        WeightDtype::Q6K,
        WeightDtype::BF16,
        WeightDtype::Q6K,
        WeightDtype::Q6K,
    ];
    let s = proj_store(&gpu, AP, ATTN_NAMES, partly, ATTN_SHAPES);
    let e = attn_boxes(&s, AP, ATTN_SHAPES, 1, false, fake_builder(&mut order)).unwrap_err();
    assert!(
        format!("{e:#}").contains("k_proj") && format!("{e:#}").contains("partly packed"),
        "{e:#}"
    );
    let wrong = [(1024, H), (256, H), (128, H), (H, 512)];
    let s = proj_store(&gpu, AP, ATTN_NAMES, packed, wrong);
    let e = attn_boxes(&s, AP, ATTN_SHAPES, 1, false, fake_builder(&mut order)).unwrap_err();
    assert!(format!("{e:#}").contains("v_proj"), "{e:#}");
    let s = proj_store(&gpu, AP, ATTN_NAMES, packed, ATTN_SHAPES);
    let e = attn_boxes(&s, AP, ATTN_SHAPES, 1, false, |_t: &WeightTensor| Ok(None)).unwrap_err();
    assert!(format!("{e:#}").contains("no 4-bit kernel"), "{e:#}");
}

#[test]
fn a_packed_gdn_projection_is_detected_and_refused_by_name() {
    let gpu = MockGpuBackend::new();
    let shapes = [(1024, H), (256, H)];
    let packed = proj_store(&gpu, LA, GDN_NAMES, [WeightDtype::Q6K; 2], shapes);
    assert!(gdn_any_packed(&packed, LA));
    let bf16 = proj_store(&gpu, LA, GDN_NAMES, [WeightDtype::BF16; 2], shapes);
    assert!(!gdn_any_packed(&bf16, LA), "a BF16 GDN layer is untouched");
    let msg = format!("{:#}", gdn_packed_refusal(LA));
    assert!(
        msg.contains(LA) && msg.contains("AVAROK_GGUF_NATIVE_IQ4=0"),
        "{msg}"
    );
}
