// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for [`WeightStore::fp8_kv_scale_count`]. Sibling file for the
//! 500-LoC cap.

#[cfg(test)]
mod tests {
    use super::super::{WeightDtype, WeightStore, WeightTensor};
    use crate::gpu::GpuBackend;
    use crate::gpu::mock::MockGpuBackend;
    use std::collections::HashMap;

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
        let mut map = HashMap::new();
        for i in 0..4 {
            map.insert(
                format!("w{i}"),
                WeightTensor {
                    ptr: gpu.alloc(1024).expect("alloc"),
                    shape: vec![16, 16],
                    dtype: WeightDtype::BF16,
                },
            );
        }
        let store = WeightStore::from_map(map);
        assert_eq!(store.fp8_kv_scale_count(), 0);
    }
}
