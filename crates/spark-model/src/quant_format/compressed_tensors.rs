// SPDX-License-Identifier: AGPL-3.0-only

//! Neural Magic `llm-compressor` / compressed-tensors NVFP4 serialization.
//!
//! The dominant community NVFP4 format, shipped by `Sehyo/*`, `RedHatAI/*`,
//! `nm-testing/*`, and most third-party re-quants. Tensor-name convention:
//!
//! | field                | tensor name            | dtype         |
//! | -------------------- | ---------------------- | ------------- |
//! | packed FP4 payload   | `.weight_packed`       | uint8 packed  |
//! | per-group FP8 scales | `.weight_scale`        | float8_e4m3   |
//! | per-tensor scalar    | `.weight_global_scale` | f32 scalar    |
//! | activation scale     | `.input_global_scale`  | f32 scalar    |
//!
//! Scale convention: `weight_global_scale` is the RECIPROCAL of ModelOpt's
//! `weight_scale_2` (verified empirically — see `quantized_v2` comment in
//! `weight_map.rs`).
//!
//! Unquantized modules are declared either in the top-level `ignore`
//! array or in `config_groups.group_N.targets` / `exclude_modules`
//! (vLLM style). Both are folded into a single list during config parse.
//!
//! Maps to the existing [`Nvfp4Variant::CompressedTensors`] dispatch.

use crate::quant_format::{QuantFormat, module_matches_pattern};
use crate::weight_map::Nvfp4Variant;

/// compressed-tensors NVFP4 checkpoint.
#[derive(Debug)]
pub struct CompressedTensorsFormat {
    /// `format` string from config (e.g. `"nvfp4-pack-quantized"`). Log only.
    pub format: String,
    /// Module-path globs that stay BF16 rather than NVFP4.
    pub ignore_modules: Vec<String>,
}

impl CompressedTensorsFormat {
    pub fn new(format: String, ignore_modules: Vec<String>) -> Self {
        Self {
            format,
            ignore_modules,
        }
    }
}

impl QuantFormat for CompressedTensorsFormat {
    fn name(&self) -> &'static str {
        "compressed-tensors"
    }

    fn base_variant(&self) -> Nvfp4Variant {
        Nvfp4Variant::CompressedTensors
    }

    fn is_ignored(&self, module_path: &str) -> bool {
        self.ignore_modules
            .iter()
            .any(|pat| module_matches_pattern(module_path, pat))
    }
}
