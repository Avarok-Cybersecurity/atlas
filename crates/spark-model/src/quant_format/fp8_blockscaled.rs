// SPDX-License-Identifier: AGPL-3.0-only

//! FP8 E4M3 block-scaled serialization (DeepSeek-V3 / Qwen3.5-35B-A3B-FP8
//! convention).
//!
//! | field             | tensor name           | dtype          |
//! | ----------------- | --------------------- | -------------- |
//! | FP8 payload       | `.weight`             | float8_e4m3    |
//! | block scales      | `.weight_scale_inv`   | bf16 [N/BS,K/BS] |
//!
//! Atlas consumes these via runtime BF16→NVFP4 re-quantization inside
//! [`crate::weight_map::quantized_from_fp8`]. The `ignore_modules` list,
//! when present, flags modules that should stay BF16 after dequant
//! (skipping the NVFP4 step).
//!
//! Maps to the existing [`Nvfp4Variant::Fp8Dequanted`] dispatch.

use crate::quant_format::{QuantFormat, module_matches_pattern};
use crate::weight_map::Nvfp4Variant;

/// FP8 block-scaled checkpoint.
#[derive(Debug)]
pub struct Fp8BlockScaledFormat {
    pub ignore_modules: Vec<String>,
}

impl Fp8BlockScaledFormat {
    pub fn new(ignore_modules: Vec<String>) -> Self {
        Self { ignore_modules }
    }
}

impl QuantFormat for Fp8BlockScaledFormat {
    fn name(&self) -> &'static str {
        "fp8-blockscaled"
    }

    fn base_variant(&self) -> Nvfp4Variant {
        Nvfp4Variant::Fp8Dequanted
    }

    fn is_ignored(&self, module_path: &str) -> bool {
        self.ignore_modules
            .iter()
            .any(|pat| module_matches_pattern(module_path, pat))
    }
}
