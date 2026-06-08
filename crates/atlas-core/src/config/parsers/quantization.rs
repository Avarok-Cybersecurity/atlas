// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `config.rs` for file-size budget. Parser for a model family.

#![allow(unused_imports)]

use anyhow::{Context, Result};
use serde_json::Value;

use super::super::{ModelConfig, QuantizationConfig};

pub fn parse_quantization_config(raw: &serde_json::Value) -> Option<QuantizationConfig> {
    let qc_raw = raw.get("quantization_config")?;
    // NVIDIA ModelOpt's sibling `hf_quant_config.json` (merged into the
    // `quantization_config` slot by `merge_sidecar_quant_config`) nests the
    // quant fields one level deep under a `"quantization"` object and never
    // emits a `quant_method` key — the scheme is implied by
    // `producer.name == "modelopt"`:
    //
    //   { "producer": { "name": "modelopt", ... },
    //     "quantization": { "quant_algo": "NVFP4",
    //                       "exclude_modules": ["lm_head", ...] } }
    //
    // Read at the top level this parses to all-empty and returns `None`,
    // which silently drops the checkpoint into tensor-name heuristics — for
    // a ModelOpt NVFP4 checkpoint (no `.weight_packed`, `.mixer.`-prefixed
    // Nemotron modules) those mis-detect as `Bf16Raw` and limp into a
    // runtime BF16->NVFP4 requant path, degrading a fully-calibrated NVFP4
    // release. `normalize_modelopt_sidecar` lifts the nested object and
    // synthesizes `quant_method` so the rest of this parser sees the
    // canonical flat shape. Already-flat configs pass through unchanged.
    let canonical = normalize_modelopt_sidecar(qc_raw);
    let qc = &canonical;

    // Either scheme may set any of these top-level strings:
    //   quant_method     — scheme name; both schemes set this.
    //   quant_algo       — ModelOpt-specific label (e.g. "NVFP4"). Also
    //                      propagated when producer.name=="modelopt".
    //   format           — compressed-tensors only
    //                      (e.g. "nvfp4-pack-quantized").
    let quant_method = qc
        .get("quant_method")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();
    let quant_algo = qc
        .get("quant_algo")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            // ModelOpt dumps sometimes put quant_algo under
            // `config_groups.group_0.weights.type` as `"float"` with a
            // `num_bits` sibling. Mine those for a best-effort label.
            let group = qc.get("config_groups")?.get("group_0")?;
            let weights = group.get("weights")?;
            let bits = weights.get("num_bits")?.as_u64()?;
            let ty = weights.get("type")?.as_str()?;
            match (bits, ty) {
                (4, "float") => Some("NVFP4"),
                (8, "float") => Some("FP8"),
                _ => None,
            }
        })
        .unwrap_or("")
        .to_string();
    let format = qc
        .get("format")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();

    // Ignore list: ModelOpt calls it `ignore`, compressed-tensors calls
    // it `ignore` too at the top level but also has `targets` inside
    // `config_groups`. Collect anything useful from both places.
    let mut ignore_modules: Vec<String> = Vec::new();
    if let Some(arr) = qc.get("ignore").and_then(serde_json::Value::as_array) {
        for v in arr {
            if let Some(s) = v.as_str() {
                ignore_modules.push(s.to_string());
            }
        }
    }
    // compressed-tensors can also use `exclude_modules` (vLLM-style).
    if let Some(arr) = qc
        .get("exclude_modules")
        .and_then(serde_json::Value::as_array)
    {
        for v in arr {
            if let Some(s) = v.as_str()
                && !ignore_modules.contains(&s.to_string())
            {
                ignore_modules.push(s.to_string());
            }
        }
    }

    // An empty quant_method with empty ignore list is not a real quant
    // config — skip so callers can fall through to heuristic detection.
    if quant_method.is_empty() && quant_algo.is_empty() && ignore_modules.is_empty() {
        return None;
    }

    Some(QuantizationConfig {
        quant_method,
        quant_algo,
        format,
        ignore_modules,
    })
}

/// Flatten a ModelOpt-style `hf_quant_config.json` payload into the canonical
/// shape the rest of [`parse_quantization_config`] reads.
///
/// Two transforms, both no-ops on an already-canonical (HF-standard) block:
///   1. If the payload has a nested `"quantization"` object, lift it to the
///      top level (ModelOpt nests `quant_algo` / `exclude_modules` there).
///   2. If no `quant_method` is present but `producer.name == "modelopt"`,
///      synthesize `quant_method = "modelopt"` so downstream config-first
///      dispatch (`detect_quant_format`, `detect_nvfp4_variant`) recognizes
///      the scheme. An explicit `quant_method` is never overwritten.
fn normalize_modelopt_sidecar(qc_raw: &serde_json::Value) -> serde_json::Value {
    // (1) Lift nested `quantization` object if present.
    let mut canonical = match qc_raw.get("quantization") {
        Some(serde_json::Value::Object(inner)) => serde_json::Value::Object(inner.clone()),
        _ => qc_raw.clone(),
    };

    let Some(obj) = canonical.as_object_mut() else {
        return canonical;
    };

    // (2) Synthesize `quant_method` from `producer.name` when absent.
    let has_method = obj
        .get("quant_method")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|s| !s.is_empty());
    if !has_method {
        let producer_modelopt = qc_raw
            .get("producer")
            .and_then(|p| p.get("name"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|n| n.eq_ignore_ascii_case("modelopt"));
        if producer_modelopt {
            obj.insert(
                "quant_method".to_string(),
                serde_json::Value::String("modelopt".to_string()),
            );
        }
    }

    canonical
}

#[cfg(test)]
mod tests {
    use super::parse_quantization_config;

    /// The exact `hf_quant_config.json` schema shipped by
    /// `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`, wrapped by
    /// `merge_sidecar_quant_config` into the `quantization_config` slot.
    #[test]
    fn modelopt_sidecar_nested_schema_parses() {
        let raw = serde_json::json!({
            "quantization_config": {
                "producer": { "name": "modelopt", "version": "0.29.0" },
                "quantization": {
                    "quant_algo": "NVFP4",
                    "kv_cache_quant_algo": "FP8",
                    "group_size": 16,
                    "exclude_modules": [
                        "lm_head",
                        "backbone.layers.4.mixer.in_proj",
                        "backbone.layers.0.mixer.conv1d"
                    ]
                }
            }
        });
        let qc = parse_quantization_config(&raw)
            .expect("ModelOpt nested sidecar must yield a QuantizationConfig");
        assert_eq!(qc.quant_method, "modelopt");
        assert_eq!(qc.quant_algo, "NVFP4");
        assert_eq!(qc.ignore_modules.len(), 3);
        assert!(qc.ignore_modules.iter().any(|m| m == "lm_head"));
        assert!(
            qc.ignore_modules
                .iter()
                .any(|m| m == "backbone.layers.4.mixer.in_proj")
        );
    }

    /// An already-flat HF-standard block (compressed-tensors) must be
    /// unaffected by the ModelOpt normalization.
    #[test]
    fn flat_compressed_tensors_block_unchanged() {
        let raw = serde_json::json!({
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "nvfp4-pack-quantized",
                "ignore": ["lm_head"]
            }
        });
        let qc = parse_quantization_config(&raw).expect("flat block must still parse");
        assert_eq!(qc.quant_method, "compressed-tensors");
        assert_eq!(qc.format, "nvfp4-pack-quantized");
        assert_eq!(qc.ignore_modules, vec!["lm_head".to_string()]);
    }

    /// ModelOpt mixed-precision sidecar (Super-120B) — nested, no
    /// `exclude_modules`. Must still resolve `quant_method = modelopt`.
    #[test]
    fn modelopt_mixed_precision_sidecar_parses() {
        let raw = serde_json::json!({
            "quantization_config": {
                "producer": { "name": "modelopt", "version": "0.43.0" },
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "kv_cache_quant_algo": "FP8"
                }
            }
        });
        let qc = parse_quantization_config(&raw)
            .expect("mixed-precision sidecar must yield a QuantizationConfig");
        assert_eq!(qc.quant_method, "modelopt");
        assert_eq!(qc.quant_algo, "MIXED_PRECISION");
    }
}
