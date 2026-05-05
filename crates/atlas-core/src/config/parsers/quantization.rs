// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `config.rs` for file-size budget. Parser for a model family.

#![allow(unused_imports)]

use anyhow::{Context, Result};
use serde_json::Value;

use super::super::{ModelConfig, QuantizationConfig};

pub fn parse_quantization_config(raw: &serde_json::Value) -> Option<QuantizationConfig> {
    let qc = raw.get("quantization_config")?;

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
