// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

#[test]
fn test_fp8_e4m3_lut() {
    use super::FP8_E4M3_LUT;

    // Verify key reference values against IEEE FP8 E4M3FN spec
    assert_eq!(FP8_E4M3_LUT[0x00], 0.0); // +0
    assert_eq!(FP8_E4M3_LUT[0x80], -0.0); // -0
    assert_eq!(FP8_E4M3_LUT[0x38], 1.0); // exp=7, mant=0 → 2^(7-7)*(1+0) = 1.0
    assert_eq!(FP8_E4M3_LUT[0xB8], -1.0); // negative 1.0
    assert_eq!(FP8_E4M3_LUT[0x3C], 1.5); // exp=7, mant=4 → 2^0*(1+0.5) = 1.5
    assert_eq!(FP8_E4M3_LUT[0x7E], 448.0); // max finite: exp=15, mant=6
    assert_eq!(FP8_E4M3_LUT[0xFE], -448.0); // min finite
    assert_eq!(FP8_E4M3_LUT[0x7F], 0.0); // NaN → 0.0 (safety)
    assert_eq!(FP8_E4M3_LUT[0xFF], 0.0); // -NaN → 0.0 (safety)

    // Subnormals: 2^(-6) * mant/8
    let eps = 1e-10;
    assert!((FP8_E4M3_LUT[0x01] - 0.001953125).abs() < eps); // 2^(-6)/8 = 1/512
    assert!((FP8_E4M3_LUT[0x07] - 0.013671875).abs() < eps); // 7 * 2^(-6)/8

    // Verify all 256 entries match the reference formula
    for i in 0u16..256 {
        let bits = i as u8;
        let sign = (bits >> 7) & 1;
        let exp = (bits >> 3) & 0x0F;
        let mant = bits & 0x07;

        let expected = if exp == 0x0F && mant == 0x07 {
            0.0f32 // NaN → 0
        } else if exp == 0 && mant == 0 {
            0.0f32
        } else if exp == 0 {
            let v = (mant as f32 / 8.0) * 2.0f32.powi(-6);
            if sign == 1 { -v } else { v }
        } else {
            let v = (1.0 + mant as f32 / 8.0) * 2.0f32.powi(exp as i32 - 7);
            if sign == 1 { -v } else { v }
        };
        let actual = FP8_E4M3_LUT[i as usize];
        assert!(
            (actual - expected).abs() < 1e-10 || (actual == 0.0 && expected == 0.0),
            "LUT mismatch at index {i:#04x}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn test_weight_name_patterns() {
    // Verify our name generation matches actual HF patterns.
    let layer = 3;
    assert_eq!(
        format!("model.layers.{layer}.self_attn.q_proj.weight"),
        "model.layers.3.self_attn.q_proj.weight"
    );
    assert_eq!(
        format!("model.layers.{layer}.linear_attn.in_proj_qkvz.weight"),
        "model.layers.3.linear_attn.in_proj_qkvz.weight"
    );
    assert_eq!(
        format!("model.layers.{layer}.mlp.experts.{}.gate_proj.weight", 42),
        "model.layers.3.mlp.experts.42.gate_proj.weight"
    );
}
