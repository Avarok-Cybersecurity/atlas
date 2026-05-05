// SPDX-License-Identifier: AGPL-3.0-only

//! NVIDIA TensorRT ModelOpt NVFP4 serialization.
//!
//! ModelOpt exports are emitted by NVIDIA's `nvidia-modelopt` toolkit and
//! appear on HuggingFace under `nvidia/*` and downstream community
//! re-quants (`lukealonso/*`, `saricles/*`, `NVIDIA/*`). The tensor-name
//! convention is distinct from compressed-tensors:
//!
//! | field                | tensor name          | dtype        |
//! | -------------------- | -------------------- | ------------ |
//! | packed FP4 payload   | `.weight`            | uint8 packed |
//! | per-group FP8 scales | `.weight_scale`      | float8_e4m3  |
//! | per-tensor scalar    | `.weight_scale_2`    | f32 scalar   |
//! | activation scale     | `.input_scale`       | f32 scalar   |
//!
//! Unquantized modules (typically `lm_head`, embeddings, and the full
//! attention tower on smaller re-quants) are declared in the top-level
//! `ignore` array of `hf_quant_config.json` / `config.json`'s
//! `quantization_config` block. Those modules ship as plain BF16 and
//! must NOT be run through the NVFP4 loader — reading uint8-packed FP4
//! at BF16 stride is a 4× byte overrun that lands as
//! `CUDA_ERROR_ILLEGAL_ADDRESS` later on, which is the bug this module
//! was written to eliminate.
//!
//! This maps to the existing [`Nvfp4Variant::Standard`] dispatch in
//! `weight_map.rs` (which was always the ModelOpt path — the misleading
//! name predates support for the compressed-tensors split).

use crate::quant_format::{QuantFormat, module_matches_pattern};
use crate::weight_map::Nvfp4Variant;

/// ModelOpt-style NVFP4 checkpoint.
#[derive(Debug)]
pub struct ModeloptFormat {
    /// `quant_algo` declared in config (`"NVFP4"`, `"FP8"`, …). Used for
    /// diagnostic logging — the actual dispatch uses `base_variant`.
    pub algo: String,
    /// Module-path globs to load as dense BF16 rather than NVFP4.
    pub ignore_modules: Vec<String>,
}

impl ModeloptFormat {
    pub fn new(algo: String, ignore_modules: Vec<String>) -> Self {
        Self {
            algo,
            ignore_modules,
        }
    }
}

impl QuantFormat for ModeloptFormat {
    fn name(&self) -> &'static str {
        "modelopt"
    }

    fn base_variant(&self) -> Nvfp4Variant {
        // `Standard` is the legacy name for ModelOpt NVFP4 layout.
        Nvfp4Variant::Standard
    }

    fn is_ignored(&self, module_path: &str) -> bool {
        self.ignore_modules
            .iter()
            .any(|pat| module_matches_pattern(module_path, pat))
    }
}
