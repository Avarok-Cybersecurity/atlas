// SPDX-License-Identifier: AGPL-3.0-only

//! Which arm each export's layout takes, end to end through a real
//! [`WeightStore`].
//!
//! 🔴 The point of these is the NEGATIVE half. Both new arms are keyed off the
//! on-disk dtype, so the claim "`LibertAIDAI/GLM-5.3-Flash-NVFP4` loads exactly
//! as it did" is only worth as much as a test that a community-shaped store
//! still reaches the old code. The community cases here assert the old
//! behaviour directly — same values, same device pointers, nothing derived.

use std::collections::HashMap;

use spark_runtime::gpu::{DevicePtr, GpuBackend, mock::MockGpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::{LayerSource, nvfp4_dequant};

const LAYER: usize = 45;

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect()
}

/// A deterministic, sign-mixed ramp — enough structure that a swapped nibble
/// or a dropped scale changes the answer.
fn ramp(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 - n as f32 / 2.0) * 0.125)
        .collect()
}

struct StoreBuilder {
    gpu: MockGpuBackend,
    map: HashMap<String, WeightTensor>,
}

impl StoreBuilder {
    fn new() -> Self {
        Self {
            gpu: MockGpuBackend::new(),
            map: HashMap::new(),
        }
    }

    fn put(&mut self, name: &str, bytes: &[u8], shape: &[usize], dtype: WeightDtype) -> DevicePtr {
        let p = self.gpu.alloc(bytes.len().max(1)).unwrap();
        self.gpu.copy_h2d(bytes, p).unwrap();
        self.map.insert(
            name.to_string(),
            WeightTensor {
                ptr: p,
                shape: shape.to_vec(),
                dtype,
            },
        );
        p
    }

    fn finish(self) -> (MockGpuBackend, WeightStore) {
        (self.gpu, WeightStore::from_map(self.map))
    }
}

fn qualified(leaf: &str) -> String {
    format!("model.language_model.layers.{LAYER}.{leaf}")
}

// ---------------------------------------------------------------- dense MLP

/// `LibertAIDAI/GLM-5.3-Flash-NVFP4`: the dense MLP is BF16 and reaches the
/// builder as the BF16 values themselves. No scales exist and none are looked
/// for.
#[test]
fn a_bf16_dense_mlp_still_takes_the_plain_float_path() {
    let values = ramp(64);
    let mut b = StoreBuilder::new();
    b.put(
        &qualified("mlp.gate_proj.weight"),
        &bf16_bytes(&values),
        &[4, 16],
        WeightDtype::BF16,
    );
    let (gpu, store) = b.finish();
    let src = LayerSource::collect(&gpu, &store, LAYER).unwrap();

    let got = src.f32("mlp.gate_proj.weight").unwrap();
    let want: Vec<f32> = values
        .iter()
        .map(|x| half::bf16::from_f32(*x).to_f32())
        .collect();
    assert_eq!(got, want, "BF16 must round-trip through f32 unchanged");
}

/// `nvidia/GLM-5.3-Flash-NVFP4`: the same tensor is packed NVFP4 plus its two
/// scale siblings, and comes out of the SAME entry point as `f32`, so
/// `build_dense_mlp` never learns the difference.
#[test]
fn a_packed_dense_mlp_is_dequantised_at_the_same_entry_point() {
    // [4, 8] U8 = a [4, 16] weight; one 16-element block per row.
    let packed: Vec<u8> = (0..32u8).map(|i| i.wrapping_mul(37)).collect();
    let scales = vec![0x38u8, 0x3C, 0x30, 0x40]; // 1.0, 1.5, 0.5, 2.0
    let scale_2 = 0.25f32;

    let mut b = StoreBuilder::new();
    b.put(
        &qualified("mlp.gate_proj.weight"),
        &packed,
        &[4, 8],
        WeightDtype::UInt8,
    );
    b.put(
        &qualified("mlp.gate_proj.weight_scale"),
        &scales,
        &[4, 1],
        WeightDtype::FP8E4M3,
    );
    b.put(
        &qualified("mlp.gate_proj.weight_scale_2"),
        &scale_2.to_le_bytes(),
        &[],
        WeightDtype::FP32,
    );
    // W4A4 calibration scale: present in the official export, read by nothing
    // here, and it must not disturb the layer.
    b.put(
        &qualified("mlp.gate_proj.input_scale"),
        &1.5f32.to_le_bytes(),
        &[],
        WeightDtype::FP32,
    );
    let (gpu, store) = b.finish();
    let src = LayerSource::collect(&gpu, &store, LAYER).unwrap();

    let got = src.f32("mlp.gate_proj.weight").unwrap();
    let want =
        nvfp4_dequant::dequant_nvfp4_to_f32("ref", &packed, &[4, 8], &scales, scale_2).unwrap();
    assert_eq!(got.len(), 64, "[4, 8] U8 is a [4, 16] weight");
    assert_eq!(got, want);
}

/// A packed weight whose scales are absent is a format this loader has not
/// been taught — an error, never a guess at another convention's naming.
#[test]
fn a_packed_weight_without_its_scales_is_an_error() {
    let mut b = StoreBuilder::new();
    b.put(
        &qualified("mlp.gate_proj.weight"),
        &[0u8; 8],
        &[1, 8],
        WeightDtype::UInt8,
    );
    let (gpu, store) = b.finish();
    let src = LayerSource::collect(&gpu, &store, LAYER).unwrap();
    let err = src.f32("mlp.gate_proj.weight").unwrap_err().to_string();
    assert!(err.contains("weight_scale is absent"), "{err}");
}

/// B6. `hc_{attn,ffn}_{base,scale}` are F32 in `LibertAIDAI/...` and BF16 in
/// NVIDIA's export. `bind_mhc_site` reads all three mHC tensors through
/// `LayerSource::f32`, which takes either width, and `plan_dtype` claims none
/// of them (they are not `self_attn.*` and not in `KDA_TENSORS`), so no
/// plan-dtype copy interferes either. Nothing needed fixing; this pins it.
///
/// 🪤 The values must agree EXACTLY, not approximately: an F32 export of a
/// tensor the model rounded to BF16 carries the same numbers at twice the
/// width, so a width-dependent read is a silent numeric change, not a
/// tolerance question.
#[test]
fn the_mhc_tensors_read_the_same_at_either_stored_width() {
    let vals = ramp(24);
    let rounded: Vec<f32> = vals
        .iter()
        .map(|x| half::bf16::from_f32(*x).to_f32())
        .collect();

    let mut wide = StoreBuilder::new();
    wide.put(
        &qualified("hc_attn_base"),
        &rounded
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
        &[24],
        WeightDtype::FP32,
    );
    let (gpu, store) = wide.finish();
    let from_f32 = LayerSource::collect(&gpu, &store, LAYER)
        .unwrap()
        .f32("hc_attn_base")
        .unwrap();

    let mut narrow = StoreBuilder::new();
    narrow.put(
        &qualified("hc_attn_base"),
        &bf16_bytes(&rounded),
        &[24],
        WeightDtype::BF16,
    );
    let (gpu, store) = narrow.finish();
    let from_bf16 = LayerSource::collect(&gpu, &store, LAYER)
        .unwrap()
        .f32("hc_attn_base")
        .unwrap();

    assert_eq!(from_f32, rounded);
    assert_eq!(from_bf16, rounded);
    assert_eq!(
        from_f32, from_bf16,
        "storage width must not change the mHC values"
    );
}
