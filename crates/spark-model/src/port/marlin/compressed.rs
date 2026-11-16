// SPDX-License-Identifier: AGPL-3.0-only
// The tensor layout follows compressed-tensors (https://github.com/neuralmagic/compressed-tensors,
// Apache-2.0): compressors/quantized_compressors/pack_quantized.py (`pack_to_int32`: the signed int4
// value plus `1 << (num_bits - 1)`, eight codes per int32 along the input dimension, code `j` of a word
// at bits `4 j`; `weight_scale` `[out, in / group]`; `weight_g_idx` `[in]` only for `actorder: group`),
// and the scheme keys of quantization/quant_scheme.py and quant_args.py. Changes: restated as a Rust
// parser of raw little-endian safetensors bytes into Atlas's `LogicalInt4`; the scheme refusals are
// Atlas's.

//! Responsibility: compressed-tensors `pack-quantized` int4 (W4A16) checkpoints: recognize the scheme
//! from `config.json`, and parse one linear layer's `weight_packed` / `weight_scale` (/ `weight_g_idx`)
//! tensors into a [`LogicalInt4`], which the existing Marlin repack serves (`kU4B8`).
//!
//! # The mapping
//!
//! ```text
//! weight_packed  I32 [n, k / 8]   code of (out nn, in kk) = (word[nn][kk / 8] >> 4 (kk % 8)) & 0xF
//!                                 (the signed int4 plus 8, so w = (code - 8) * s: Marlin's kU4B8)
//! weight_scale   BF16|F16 [n, k / group]   -> LogicalInt4.scales[g * n + nn] (transposed)
//! weight_g_idx   I32 [k]          only with actorder "group"; must be the contiguous grouping
//! weight_shape   I64 [2]          [n, k]; not read (the config gives the shapes)
//! ```
//!
//! This is the GPTQ `qweight` layout transposed (GPTQ packs along K in a `[k / 8, n]` matrix), so a
//! symmetric compressed-tensors weight and the GPTQ weight of the same codes are the same
//! [`LogicalInt4`].
//!
//! # What is refused, by name
//!
//! Anything but 4-bit symmetric `int` weights with `strategy: group` at group 128 (the only group the
//! Ada Marlin build instantiates), static (not dynamic) weights, no activation quantization (W4A16
//! only), and an act-order checkpoint whose `g_idx` is not the contiguous grouping (Atlas's Marlin port
//! has no activation reorder, `checkpoint::check_g_idx`). `actorder: static` (llm-compressor's
//! "weight" order, the RedHatAI exports) quantizes in activation order but keeps the groups contiguous
//! and ships no `g_idx`, so it is served.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

use super::DEFAULT_GROUP;
use super::checkpoint::{ScaleDtype, check_g_idx, scales_bf16, words_le};
use super::reference::LogicalInt4;

/// compressed-tensors' `quant_method`.
pub const QUANT_METHOD: &str = "compressed-tensors";
/// The INT packer's `format` (the FP4 packers are `nvfp4-pack-quantized` and `mxfp4-pack-quantized`).
pub const INT_PACK_FORMAT: &str = "pack-quantized";

/// The tensor suffixes of one quantized linear (`<module>.<suffix>`).
pub const PACKED: &str = "weight_packed";
pub const SCALE: &str = "weight_scale";
pub const SHAPE: &str = "weight_shape";
pub const G_IDX: &str = "weight_g_idx";
pub const ZERO_POINT: &str = "weight_zero_point";

/// Codes per int32 word.
pub const PACK_FACTOR: usize = 8;

/// True for exactly the INT packer's format string (case and surrounding space ignored).
pub fn is_int_pack_format(format: &str) -> bool {
    format.trim().eq_ignore_ascii_case(INT_PACK_FORMAT)
}

/// True when `name` is one of a quantized linear's tensors (`*.weight_packed`, `*.weight_scale`, ...).
pub fn is_scheme_tensor(name: &str) -> bool {
    [PACKED, SCALE, SHAPE, G_IDX, ZERO_POINT]
        .iter()
        .any(|s| name.strip_suffix(s).is_some_and(|p| p.ends_with('.')))
}

/// `<module>.<suffix>`.
pub fn tensor_name(module: &str, suffix: &str) -> String {
    format!("{module}.{suffix}")
}

/// The checkpoint's activation-ordering setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActOrder {
    /// `null`: columns quantized in order.
    None,
    /// `"static"` / `"weight"`: quantized in activation order, groups kept contiguous, no `g_idx`.
    Static,
    /// `"group"` / `"dynamic"`: groups follow activation order; a `g_idx` is stored and must turn out
    /// contiguous to be served.
    Group,
}

/// The served scheme of a compressed-tensors `pack-quantized` checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CtW4Scheme {
    pub group: usize,
    pub actorder: ActOrder,
    /// The `ignore` list as written (module names and `re:` patterns); informational, the loader keys
    /// on the tensors present.
    pub ignore: Vec<String>,
}

fn s<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}

fn parse_actorder(w: &Value, gname: &str) -> Result<ActOrder> {
    match w.get("actorder") {
        None | Some(Value::Null) => Ok(ActOrder::None),
        Some(Value::Bool(false)) => Ok(ActOrder::None),
        Some(Value::Bool(true)) => Ok(ActOrder::Group),
        Some(Value::String(a)) => match a.to_ascii_lowercase().as_str() {
            "static" | "weight" => Ok(ActOrder::Static),
            "group" | "dynamic" => Ok(ActOrder::Group),
            other => bail!("compressed-tensors {gname}: unknown actorder {other:?}"),
        },
        Some(other) => bail!("compressed-tensors {gname}: unreadable actorder {other}"),
    }
}

/// Check one `config_groups` entry against what the Marlin W4A16 route serves.
fn parse_group(gname: &str, g: &Value) -> Result<(usize, ActOrder)> {
    if let Some(f) = s(g, "format") {
        ensure!(
            is_int_pack_format(f),
            "compressed-tensors {gname}: format {f:?} in a pack-quantized checkpoint is not served"
        );
    }
    let ia = g.get("input_activations").unwrap_or(&Value::Null);
    ensure!(
        ia.is_null(),
        "compressed-tensors {gname}: quantized input activations ({ia}) are not served; the Marlin \
         route is W4A16 only"
    );
    let w = g
        .get("weights")
        .with_context(|| format!("compressed-tensors {gname}: no `weights` block"))?;
    let bits = w.get("num_bits").and_then(Value::as_u64).unwrap_or(0);
    ensure!(
        bits == 4,
        "compressed-tensors {gname}: {bits}-bit weights are not served (Marlin W4A16 is 4-bit)"
    );
    let ty = s(w, "type").unwrap_or("int");
    ensure!(
        ty.eq_ignore_ascii_case("int"),
        "compressed-tensors {gname}: weight type {ty:?} is not served (int only)"
    );
    let symmetric = w.get("symmetric").and_then(Value::as_bool).unwrap_or(true);
    ensure!(
        symmetric,
        "compressed-tensors {gname}: asymmetric weights (zero points) are not served by this route; \
         only symmetric W4A16 (Marlin kU4B8)"
    );
    let dynamic = w.get("dynamic").and_then(Value::as_bool).unwrap_or(false);
    ensure!(
        !dynamic,
        "compressed-tensors {gname}: dynamic weight quantization is not a checkpoint format"
    );
    let strategy = s(w, "strategy").unwrap_or("group");
    ensure!(
        strategy.eq_ignore_ascii_case("group"),
        "compressed-tensors {gname}: strategy {strategy:?} is not served (group {DEFAULT_GROUP} only; \
         the Ada Marlin build has no channelwise instance)"
    );
    let group = w.get("group_size").and_then(Value::as_u64).unwrap_or(0) as usize;
    ensure!(
        group == DEFAULT_GROUP,
        "compressed-tensors {gname}: group_size {group} is not served (the Ada Marlin build \
         instantiates g{DEFAULT_GROUP} only)"
    );
    Ok((group, parse_actorder(w, gname)?))
}

impl CtW4Scheme {
    /// The scheme of a whole `config.json` document. `Ok(None)`: not a compressed-tensors
    /// `pack-quantized` checkpoint (every other checkpoint, untouched). `Err`: it is one, but a group
    /// asks for something this route does not serve (the message names it).
    pub fn from_config_json(raw: &Value) -> Result<Option<Self>> {
        let Some(qc) = raw.get("quantization_config") else {
            return Ok(None);
        };
        if s(qc, "quant_method") != Some(QUANT_METHOD) {
            return Ok(None);
        }
        if !s(qc, "format").is_some_and(is_int_pack_format) {
            return Ok(None);
        }
        let groups = qc
            .get("config_groups")
            .and_then(Value::as_object)
            .context("compressed-tensors pack-quantized: no `config_groups`")?;
        ensure!(
            !groups.is_empty(),
            "compressed-tensors pack-quantized: `config_groups` is empty"
        );
        let mut scheme: Option<(usize, ActOrder)> = None;
        for (gname, g) in groups {
            let this = parse_group(gname, g)?;
            match scheme {
                None => scheme = Some(this),
                Some(prev) => ensure!(
                    prev == this,
                    "compressed-tensors pack-quantized: groups differ ({prev:?} vs {gname} {this:?})"
                ),
            }
        }
        let (group, actorder) = scheme.context("compressed-tensors: no group")?;
        let ignore = qc
            .get("ignore")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        Ok(Some(Self {
            group,
            actorder,
            ignore,
        }))
    }

    /// [`Self::from_config_json`] of `<dir>/config.json`.
    pub fn from_model_dir(dir: &Path) -> Result<Option<Self>> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("compressed-tensors: reading {}", path.display()))?;
        let raw: Value = serde_json::from_str(&text)
            .with_context(|| format!("compressed-tensors: parsing {}", path.display()))?;
        Self::from_config_json(&raw)
    }
}

/// One linear layer's compressed-tensors int4 tensors, as raw little-endian bytes.
#[derive(Clone, Copy, Debug)]
pub struct CtPackedWeights<'a> {
    /// I32 LE `[n, k / 8]`.
    pub packed: &'a [u8],
    /// `[n, k / group]` in `scale_dtype`.
    pub scales: &'a [u8],
    pub scale_dtype: ScaleDtype,
    /// I32 LE `[k]`, present only for `actorder: group`.
    pub g_idx: Option<&'a [u8]>,
    /// Outputs.
    pub n: usize,
    /// Inputs.
    pub k: usize,
    pub group: usize,
}

/// A compressed-tensors tensor set as a logical weight (`kU4B8`, no zero points).
pub fn ct_to_logical(w: &CtPackedWeights<'_>) -> Result<LogicalInt4> {
    let (n, k, group) = (w.n, w.k, w.group);
    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(PACK_FACTOR),
        "marlin: compressed-tensors [{n}x{k}] is not a packed int4 shape (K a multiple of 8)"
    );
    ensure!(
        group > 0 && k.is_multiple_of(group),
        "marlin: compressed-tensors K={k} is not a multiple of the group {group}"
    );
    if let Some(g) = w.g_idx {
        check_g_idx(g, k, group).context(
            "compressed-tensors actorder \"group\" checkpoint: Atlas's Marlin port serves only the \
             contiguous grouping (re-export with actorder \"static\" or null)",
        )?;
    }
    let groups = k / group;
    let kw = k / PACK_FACTOR;
    let words = words_le(w.packed, n * kw, "compressed-tensors weight_packed")?;
    let mut q = vec![0u8; k * n];
    for nn in 0..n {
        let row = &words[nn * kw..(nn + 1) * kw];
        for kk in 0..k {
            q[kk * n + nn] = ((row[kk / PACK_FACTOR] >> ((kk % PACK_FACTOR) * 4)) & 0xF) as u8;
        }
    }
    let sc = scales_bf16(w.scales, n * groups, w.scale_dtype)?;
    let mut scales = vec![0u16; groups * n];
    for nn in 0..n {
        for g in 0..groups {
            scales[g * n + nn] = sc[nn * groups + g];
        }
    }
    let l = LogicalInt4 {
        n,
        k,
        group,
        q,
        scales,
        zeros: None,
    };
    l.validate()?;
    Ok(l)
}

/// `weight_packed` words of an `[n][k]` matrix of codes 0..=15 (`pack_to_int32` after its `+ 8`):
/// the inverse of [`ct_to_logical`]'s unpack, for tests and synthetic checkpoints.
pub fn ct_pack(codes_nk: &[u8], n: usize, k: usize) -> Vec<u32> {
    let kw = k / PACK_FACTOR;
    let mut out = vec![0u32; n * kw];
    for nn in 0..n {
        for kk in 0..k {
            out[nn * kw + kk / PACK_FACTOR] |=
                u32::from(codes_nk[nn * k + kk] & 0xF) << ((kk % PACK_FACTOR) * 4);
        }
    }
    out
}

#[cfg(test)]
#[path = "compressed_tests.rs"]
mod tests;
