// SPDX-License-Identifier: AGPL-3.0-only
// After TensorFold docs/recipes/exl3.md ("What a family has to say", "Traps": the config's `bits` is
// the average and `head_bits` the head's; Apache-2.0, ashhart/TensorFold, commit 3885dce). Design
// only, no code copied. See EXL3-ATTRIBUTION.md.

//! Responsibility: the EXL3 block of a checkpoint's `config.json` (`quantization_config` with
//! `quant_method = "exl3"`): whether a checkpoint is EXL3, and the informative fields of its block.
//! Every field but the method is optional, and none of them decides a layer's decode: the per-tensor
//! widths and codebooks come from each group's trellis shape and marker ([`super::load`]).

use std::path::Path;

use anyhow::{Context, Result};

/// The EXL3 block of a `config.json`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Exl3Config {
    /// The quantizer version string.
    pub version: Option<String>,
    /// Average bits a weight (the model's average, not any one tensor's).
    pub bits: Option<f64>,
    /// Bits of the output head.
    pub head_bits: Option<f64>,
    /// The codebook the quantizer was asked for (`3inst`, `mcg`, `mul1`); informative only.
    pub codebook: Option<String>,
    /// Whether output scales were folded into `svh` (`always`, `never`, `auto`); the layer's formula
    /// is the same either way.
    pub out_scales: Option<String>,
}

/// The EXL3 block of a parsed `config.json`, at the top level or under `text_config`; `None` when the
/// checkpoint is not EXL3.
pub fn config_from_json(config: &serde_json::Value) -> Option<Exl3Config> {
    let q = config
        .get("quantization_config")
        .or_else(|| config.get("text_config")?.get("quantization_config"))?;
    let method = q.get("quant_method")?.as_str()?;
    if !method.eq_ignore_ascii_case("exl3") {
        return None;
    }
    let string = |key: &str| q.get(key).and_then(|v| v.as_str()).map(str::to_string);
    let number = |key: &str| q.get(key).and_then(|v| v.as_f64());
    Some(Exl3Config {
        version: string("version"),
        bits: number("bits"),
        head_bits: number("head_bits"),
        codebook: string("codebook"),
        out_scales: string("out_scales"),
    })
}

/// Reads `dir/config.json` and returns its EXL3 block, `Ok(None)` when the checkpoint is not EXL3.
pub fn read_config(dir: &Path) -> Result<Option<Exl3Config>> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(config_from_json(&json))
}

/// True when `dir/config.json` declares an EXL3 checkpoint. A missing or unreadable config is not EXL3.
pub fn is_exl3_checkpoint(dir: &Path) -> bool {
    matches!(read_config(dir), Ok(Some(_)))
}
