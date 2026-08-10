// SPDX-License-Identifier: AGPL-3.0-only

//! Shared (`#[cfg(test)]`-only) fixtures for the LoRA seam tests. SSOT for the
//! factory config, the `SlotView` builder, and the GPU-free `LoraPair` factory
//! used across `slot_math_tests` / `key_tests` / `types_tests`.

use atlas_core::config::ModelConfig;
use spark_runtime::gpu::DevicePtr;

use crate::layers::ops::lora_delta::LoraPair;
use crate::lora::SlotView;
use crate::weight_map::DenseWeight;

// Real factory config: layers 3,7,…,47 are FullAttention. The pack offset
// math depends only on layer_type + projection dims.
pub(crate) fn cfg() -> ModelConfig {
    ModelConfig::qwen3_next_80b_nvfp4()
}

// Task #27: a per-slot view for the pure victim-selection policy tests.
pub(crate) fn view(filled: bool, ref_count: usize, last_used: u64) -> SlotView {
    SlotView {
        filled,
        ref_count,
        last_used,
    }
}

// `tag` distinguishes pairs by their A device pointer so a test can assert it
// selected the intended one (no GPU / no real weights).
pub(crate) fn dummy_pair(tag: u64, k_in: u32, n_out: u32) -> LoraPair {
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
