// SPDX-License-Identifier: AGPL-3.0-only

//! Weight-quantization format abstraction.
//!
//! Atlas must load quantized checkpoints produced by several toolchains,
//! each of which serializes the same fundamental numeric format (e.g. NVFP4)
//! with a different tensor-name convention. Historically we sniffed those
//! names at load time via `detect_nvfp4_variant` in `weight_map.rs`, but
//! community re-quants that advertise their scheme in `quantization_config`
//! (HF standard) and keep some modules unquantized via an `ignore` list
//! (`lukealonso/MiniMax-M2.7-NVFP4`) broke that heuristic: the detector
//! saw MLP gates without `.weight_scale` and returned `Bf16Raw`, which
//! then read uint8-packed FP4 as BF16 — a 4× byte overrun that surfaced
//! as `CUDA_ERROR_ILLEGAL_ADDRESS` ten seconds into model construction
//! (reported on Discord 2026-04-17 by `energyburns`, `henryous`).
//!
//! The mitigation is to match vLLM / TensorRT-LLM / SGLang: **prefer the
//! `quantization_config` signal, fall back to tensor-name sniffing only
//! when it is absent**. This module formalizes that with a trait plus
//! one implementation per supported serialization layout:
//!
//!   * [`CompressedTensorsFormat`] — Neural Magic `llm-compressor`
//!     (`weight_packed` + `weight_global_scale` + `input_global_scale`)
//!   * [`ModeloptFormat`] — NVIDIA TensorRT ModelOpt
//!     (`weight` + `weight_scale` + `weight_scale_2` + `input_scale`)
//!   * [`Fp8BlockScaledFormat`] — FP8 E4M3 with `weight_scale_inv`
//!
//! [`detect_quant_format`] is the single entry point. It inspects
//! `config.quantization_config` first and only falls back to a heuristic
//! on the weight store when the config is silent (emitting a warning,
//! since a silent fallback is precisely what caused the original bug).

use atlas_core::config::ModelConfig;
use spark_runtime::weights::WeightStore;

use crate::weight_map::Nvfp4Variant;

mod compressed_tensors;
mod fp8_blockscaled;
mod modelopt;

pub use compressed_tensors::CompressedTensorsFormat;
pub use fp8_blockscaled::Fp8BlockScaledFormat;
pub use modelopt::ModeloptFormat;

/// A serialization layout for quantized weights.
///
/// Implementations describe a **loading policy** for a checkpoint: which
/// tensor names to read, which dtypes to expect, and which module paths
/// should stay BF16 (the ignore list). The heavy per-linear loading code
/// remains in `weight_map.rs`; this trait is a thin dispatch wrapper that
/// selects the right variant while honoring per-module overrides.
pub trait QuantFormat: Send + Sync + std::fmt::Debug {
    /// Human-readable name for logs (`"modelopt"`, `"compressed-tensors"`,
    /// `"fp8-blockscaled"`).
    fn name(&self) -> &'static str;

    /// The [`Nvfp4Variant`] that this format maps to in the existing
    /// `weight_map.rs` dispatch. Allows the trait to co-exist with the
    /// legacy variant-based call sites during incremental migration.
    fn base_variant(&self) -> Nvfp4Variant;

    /// Is `module_path` in the format's ignore list (should be loaded
    /// as dense BF16 rather than quantized)? `module_path` is the tensor
    /// name with the trailing `.weight_scale_2` / `.weight_packed` /
    /// etc. stripped — i.e. the `prefix` passed to
    /// `weight_map::quantized_any`.
    fn is_ignored(&self, module_path: &str) -> bool;

    /// Effective variant for a specific module: the base variant, or
    /// `Bf16Raw` if the module is in the ignore list. Loaders should
    /// consult this instead of `base_variant` when loading per-module.
    fn variant_for(&self, module_path: &str) -> Nvfp4Variant {
        if self.is_ignored(module_path) {
            Nvfp4Variant::Bf16Raw
        } else {
            self.base_variant()
        }
    }
}

/// Pick the right [`QuantFormat`] for a checkpoint.
///
/// Decision order:
///   1. If `config.quantization_config` is present, use its declared
///      `quant_method` / `quant_algo` — this is the authoritative signal
///      every other inference stack (vLLM, TRT-LLM, SGLang) keys off.
///   2. Otherwise scan the weight store for the convention actually on
///      disk (legacy path, preserved for the many checkpoints in the
///      wild that ship without a `quantization_config` block).
///   3. If neither the config nor the store yields a recognized scheme,
///      return a `ModeloptFormat` with empty ignore list but emit a
///      `tracing::warn!`. This preserves existing behavior for the
///      rarely-used pure-BF16 checkpoints while making the guess loud.
pub fn detect_quant_format(config: &ModelConfig, store: &WeightStore) -> Box<dyn QuantFormat> {
    // (1) Config-level dispatch — the path that fixes the
    // `lukealonso/MiniMax-M2.7-NVFP4` bug.
    if let Some(qc) = &config.quantization_config {
        let method = qc.quant_method.as_str();
        let algo = qc.quant_algo.as_str();
        let format = qc.format.as_str();
        let ignore = qc.ignore_modules.clone();

        match method {
            "modelopt" => {
                tracing::info!(
                    "QuantFormat: modelopt (algo={algo:?}), {} ignored module(s)",
                    ignore.len(),
                );
                return Box::new(ModeloptFormat::new(algo.to_string(), ignore));
            }
            "compressed-tensors" => {
                tracing::info!(
                    "QuantFormat: compressed-tensors (format={format:?}), {} ignored module(s)",
                    ignore.len(),
                );
                return Box::new(CompressedTensorsFormat::new(format.to_string(), ignore));
            }
            "fp8" => {
                tracing::info!(
                    "QuantFormat: fp8 (block-scaled), {} ignored module(s)",
                    ignore.len(),
                );
                return Box::new(Fp8BlockScaledFormat::new(ignore));
            }
            other if !other.is_empty() => {
                tracing::warn!(
                    "QuantFormat: config declares unrecognized quant_method={other:?}; \
                     falling back to tensor-name heuristic. Atlas currently understands \
                     {{compressed-tensors, modelopt, fp8}}. Checkpoint load may fail."
                );
                // fall through
            }
            _ => {
                // Empty method but non-empty ignore list — treat like
                // heuristic detection (common for older configs).
            }
        }
    }

    // (2) Heuristic fallback. Reuse the existing detector to preserve
    // every working checkpoint in Atlas's CI matrix; only the partial-
    // metadata footgun is patched separately in `weight_map.rs`.
    let variant = crate::weight_map::detect_nvfp4_variant(store, config);
    let ignore = config
        .quantization_config
        .as_ref()
        .map(|qc| qc.ignore_modules.clone())
        .unwrap_or_default();
    match variant {
        Nvfp4Variant::CompressedTensors => {
            tracing::info!("QuantFormat: compressed-tensors (detected from tensor names)");
            Box::new(CompressedTensorsFormat::new(String::new(), ignore))
        }
        Nvfp4Variant::Fp8Dequanted => {
            tracing::info!("QuantFormat: fp8-blockscaled (detected from tensor names)");
            Box::new(Fp8BlockScaledFormat::new(ignore))
        }
        Nvfp4Variant::Standard => {
            tracing::info!("QuantFormat: modelopt-style NVFP4 (detected from tensor names)");
            Box::new(ModeloptFormat::new(String::new(), ignore))
        }
        Nvfp4Variant::Bf16Raw => {
            // Pure BF16 / partial-metadata checkpoint. The ModelOpt
            // impl with empty ignore list will route every call to
            // `Bf16Raw` via `variant_for` — which is what we want.
            tracing::warn!(
                "QuantFormat: no quantization declared and no pre-quantized weights found; \
                 treating checkpoint as BF16 raw (weights will be runtime-quantized). \
                 Quality will be inferior to a calibrated NVFP4 release."
            );
            Box::new(ModeloptFormat::new(String::new(), ignore)) as Box<dyn QuantFormat>
        }
    }
}

/// HuggingFace-style glob match for module ignore-list entries.
///
/// Used by every [`QuantFormat::is_ignored`] impl so all three schemes
/// share the same semantics. `*` matches any run of characters including
/// empty; literal `.` matches `.`. Patterns without `*` must match
/// exactly (prefix match would be too lax — `lm_head` must not match
/// `lm_head_norm`). Patterns ending in `*` DO act as prefix matches,
/// which is the dominant HF case (`model.layers.*.self_attn*`).
pub(crate) fn module_matches_pattern(path: &str, pattern: &str) -> bool {
    let segments: Vec<&str> = pattern.split('*').collect();
    if segments.len() == 1 {
        return path == pattern;
    }
    let mut rest = path;
    // First segment anchors at the start unless the pattern begins with `*`.
    let first = segments[0];
    if !first.is_empty() {
        if !rest.starts_with(first) {
            return false;
        }
        rest = &rest[first.len()..];
    }
    // Intermediate segments must appear in order.
    for seg in &segments[1..segments.len() - 1] {
        if seg.is_empty() {
            continue;
        }
        match rest.find(seg) {
            Some(pos) => rest = &rest[pos + seg.len()..],
            None => return false,
        }
    }
    // Last segment: empty means pattern ended in `*` (any remainder OK);
    // non-empty means the remainder must END with it.
    let last = segments[segments.len() - 1];
    last.is_empty() || rest.ends_with(last)
}

#[cfg(test)]
mod tests {
    use super::module_matches_pattern as m;

    #[test]
    fn exact_match() {
        assert!(m("lm_head", "lm_head"));
        assert!(!m("lm_head_norm", "lm_head"));
    }

    #[test]
    fn prefix_star() {
        assert!(m(
            "model.layers.5.self_attn.q_proj",
            "model.layers.*.self_attn*"
        ));
        assert!(m(
            "model.layers.62.self_attn.out",
            "model.layers.*.self_attn*"
        ));
        assert!(!m(
            "model.layers.5.mlp.gate_proj",
            "model.layers.*.self_attn*"
        ));
    }

    #[test]
    fn suffix_star() {
        assert!(m("lm_head.weight", "lm_head*"));
    }

    #[test]
    fn middle_star() {
        assert!(m("model.layers.0.mlp.gate", "model.layers.*.mlp.gate"));
        assert!(!m("model.layers.0.attn.gate", "model.layers.*.mlp.gate"));
    }
}
