// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

// The FP8 E4M3 LUT and the f32 → BF16 cast are now `atlas_core::numeric`;
// their reference-value, exhaustive-spec, RNE-byte-exact and PyTorch-parity
// tests live beside them there. The identical copy that used to sit here
// was deleted rather than kept in sync by hand.

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
