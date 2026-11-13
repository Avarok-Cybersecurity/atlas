// SPDX-License-Identifier: AGPL-3.0-only
// After ExLlamaV3 (turboderp-org/exllamav3), MIT, Copyright (c) 2025 Turboderp, via TensorFold
// zig/src/core/exl3_load.zig:1-177 (Apache-2.0, ashhart/TensorFold, commit 3885dce; Copyright 2026
// TensorFold contributors). Changes: translated from Zig to Rust over Atlas's header-only shard reader;
// every refusal carries its reason; the checkpoint's plain (unquantized) tensors are listed; a group
// loads into an owned `Exl3Tensor`. See EXL3-ATTRIBUTION.md.

//! Responsibility: EXL3 checkpoint admission: every `{prefix}.trellis` tensor of a checkpoint is either
//! admitted as a group (its trellis layout, its codebook marker and value, its scales and optional bias
//! all valid) or refused with a reason, then loaded on demand into an [`Exl3Tensor`].
//!
//! Admission (`exl3_load.zig:133-177`): the trellis is I16 rank 3 with a last dimension of `8 * halves`,
//! halves 2..=16; `K = 16 * shape[0]` and `N = 16 * shape[1]` without overflow and multiples of 128;
//! at most one marker (`{prefix}.mcg` or `{prefix}.mul1`, a 4-byte I32/U32 scalar holding the
//! multiplier; none means `3inst`); odd halves only with `mul1`; fp16 `suh` `[K]` and `svh` `[N]`, or
//! both as packed sign words `su` `[K/16]` and `sv` `[N/16]` (one of each is a half-written group);
//! an optional fp16 `bias` `[N]`. Scales are checked by element count, as the Zig scan does.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use anyhow::{Context, Result, bail};
use safetensors::Dtype;

use super::format::{Codebook, Width};
use super::shards::{ShardEntry, ShardSet};
use super::tensor::{Exl3Tensor, ScaleForm};

/// Suffixes of a group's tensors after `{prefix}.`.
pub const GROUP_SUFFIXES: [&str; 8] = ["trellis", "suh", "svh", "su", "sv", "bias", "mcg", "mul1"];

/// Why a trellis tensor was not admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The trellis is not I16 of rank 3.
    NotI16Rank3,
    /// The last dimension is not `8 * halves` with halves 2..=16.
    BadWidth,
    /// `16 * shape` overflows.
    Overflow,
    /// K or N is not a multiple of 128.
    NotBlockAligned,
    /// Both codebook markers are present.
    TwoMarkers,
    /// A marker tensor that is not a 4-byte scalar holding its multiplier.
    BadMarker,
    /// A half width without the `mul1` codebook.
    HalfWidthNeedsMul1,
    /// A scale tensor (fp16 or packed) is missing.
    MissingScales,
    /// A scale tensor with the wrong dtype or element count.
    BadScales,
    /// fp16 input scales with packed output scales, or the reverse.
    MixedScales,
    /// A bias that is not fp16 `[N]`.
    BadBias,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NotI16Rank3 => "trellis is not I16 of rank 3",
            Self::BadWidth => "trellis words a tile is not 8 x (2..=16 half bits)",
            Self::Overflow => "trellis shape overflows 16 x tiles",
            Self::NotBlockAligned => "K or N is not a multiple of 128",
            Self::TwoMarkers => "both the mcg and the mul1 markers are present",
            Self::BadMarker => "codebook marker is not a 4-byte scalar holding its multiplier",
            Self::HalfWidthNeedsMul1 => "a half-bit width without the mul1 codebook",
            Self::MissingScales => "suh/svh (or su/sv) missing",
            Self::BadScales => "suh/svh (or su/sv) of the wrong dtype or length",
            Self::MixedScales => "one fp16 scale vector and one packed sign vector",
            Self::BadBias => "bias is not F16 [N]",
        })
    }
}

/// One admitted group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupInfo {
    /// The tensor prefix.
    pub prefix: String,
    /// Its bit width.
    pub width: Width,
    /// Its codebook.
    pub codebook: Codebook,
    /// Inputs.
    pub k: usize,
    /// Outputs.
    pub n: usize,
    /// How its scales are stored.
    pub scales: ScaleForm,
    /// True when it has a bias.
    pub bias: bool,
}

/// Counts of the admitted groups.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    /// Admitted groups.
    pub groups: usize,
    /// Groups on `3inst`.
    pub inst3: usize,
    /// Groups on `mcg`.
    pub mcg: usize,
    /// Groups on `mul1`.
    pub mul1: usize,
    /// Groups with fp16 scales.
    pub fp16_scales: usize,
    /// Groups with packed sign scales.
    pub sign_scales: usize,
    /// Groups with a bias.
    pub bias: usize,
    /// Groups by half-bit width (index 2..=16).
    pub by_halves: [usize; 17],
    /// Routed experts' down projections (`*.experts.*.down_proj`) by half-bit width.
    pub experts_down: [usize; 17],
    /// Bytes of every admitted trellis.
    pub trellis_bytes: u64,
    /// Weights of every admitted group (`sum K * N`).
    pub weights: u64,
}

impl Summary {
    fn bump(&mut self, g: &GroupInfo, trellis_bytes: u64) {
        self.weights += (g.k as u64) * (g.n as u64);
        self.groups += 1;
        match g.codebook {
            Codebook::Inst3 => self.inst3 += 1,
            Codebook::Mcg => self.mcg += 1,
            Codebook::Mul1 => self.mul1 += 1,
        }
        match g.scales {
            ScaleForm::Fp16 => self.fp16_scales += 1,
            ScaleForm::Signs => self.sign_scales += 1,
        }
        self.bias += usize::from(g.bias);
        let h = usize::from(g.width.halves());
        self.by_halves[h] += 1;
        if g.prefix.ends_with(".down_proj") && g.prefix.contains(".experts.") {
            self.experts_down[h] += 1;
        }
        self.trellis_bytes += trellis_bytes;
    }

    /// Average trellis bits a weight over the admitted groups (scales and bias excluded), or 0 with
    /// no groups.
    pub fn average_bits(&self) -> f64 {
        if self.weights == 0 {
            return 0.0;
        }
        self.trellis_bytes as f64 * 8.0 / self.weights as f64
    }
}

/// Every trellis tensor of a checkpoint, admitted or refused.
#[derive(Debug)]
pub struct Exl3Scan {
    shards: ShardSet,
    groups: Vec<GroupInfo>,
    refused: Vec<(String, Refusal)>,
    /// Prefixes of the admitted and the refused groups.
    prefixes: BTreeSet<String>,
    summary: Summary,
}

impl Exl3Scan {
    /// Scans `dir`'s shards (headers only).
    pub fn open(dir: &Path) -> Result<Self> {
        Ok(Self::from_shards(ShardSet::open(dir)?))
    }

    /// Scans an already-read shard set.
    pub fn from_shards(shards: ShardSet) -> Self {
        // Keyed by prefix: prefix order can differ from tensor-name order ("a-b.trellis" sorts before
        // "a.trellis", but "a" sorts before "a-b").
        let mut by_prefix = BTreeMap::new();
        let mut refused = Vec::new();
        let mut summary = Summary::default();
        for (name, entry) in shards.iter() {
            let Some(prefix) = name.strip_suffix(".trellis") else {
                continue;
            };
            match admit(&shards, prefix, entry) {
                Ok(g) => {
                    summary.bump(&g, entry.len);
                    by_prefix.insert(g.prefix.clone(), g);
                }
                Err(why) => refused.push((prefix.to_string(), why)),
            }
        }
        let groups: Vec<GroupInfo> = by_prefix.into_values().collect();
        let prefixes = groups
            .iter()
            .map(|g| g.prefix.clone())
            .chain(refused.iter().map(|(p, _)| p.clone()))
            .collect();
        Self {
            shards,
            groups,
            refused,
            prefixes,
            summary,
        }
    }

    /// The admitted groups, sorted by prefix.
    pub fn groups(&self) -> &[GroupInfo] {
        &self.groups
    }

    /// The refused trellis prefixes and why.
    pub fn refused(&self) -> &[(String, Refusal)] {
        &self.refused
    }

    /// Counts of the admitted groups.
    pub fn summary(&self) -> &Summary {
        &self.summary
    }

    /// The shard headers.
    pub fn shards(&self) -> &ShardSet {
        &self.shards
    }

    /// The index of the admitted group `prefix`.
    pub fn find(&self, prefix: &str) -> Option<usize> {
        self.groups
            .binary_search_by(|g| g.prefix.as_str().cmp(prefix))
            .ok()
    }

    /// True when `name` is one of a group's tensors (admitted or refused), so a BF16 loader must not
    /// load it as an ordinary weight.
    pub fn is_group_tensor(&self, name: &str) -> bool {
        split_group_name(name).is_some_and(|(prefix, _)| self.prefixes.contains(prefix))
    }

    /// The checkpoint's tensors that belong to no group (embeddings, norms, routers, sometimes a head):
    /// the family's ordinary loader reads these.
    pub fn plain_tensors(&self) -> Vec<&str> {
        self.shards
            .iter()
            .map(|(name, _)| name)
            .filter(|name| !self.is_group_tensor(name))
            .collect()
    }

    /// Loads admitted group `gi`.
    pub fn load(&self, gi: usize) -> Result<Exl3Tensor> {
        let g = self
            .groups
            .get(gi)
            .context("EXL3 group index out of range")?;
        load_group(&self.shards, g)
    }

    /// Loads the admitted group `prefix`.
    pub fn load_prefix(&self, prefix: &str) -> Result<Exl3Tensor> {
        let gi = self
            .find(prefix)
            .with_context(|| format!("no admitted EXL3 group {prefix}"))?;
        self.load(gi)
    }
}

/// `{prefix}.{suffix}` with a group suffix, split.
fn split_group_name(name: &str) -> Option<(&str, &str)> {
    let (prefix, suffix) = name.rsplit_once('.')?;
    GROUP_SUFFIXES.contains(&suffix).then_some((prefix, suffix))
}

fn tensor<'a>(shards: &'a ShardSet, prefix: &str, suffix: &str) -> Option<&'a ShardEntry> {
    shards.get(&format!("{prefix}.{suffix}"))
}

/// One scale vector's admission: the fp16 name, else the packed-words fallback; `None` when both are
/// absent and the part is optional (`exl3_load.zig:163-177`).
fn scale(
    shards: &ShardSet,
    prefix: &str,
    name: &str,
    len: usize,
    packed: Option<(&str, usize)>,
) -> Result<Option<ScaleForm>, Refusal> {
    let Some(e) = tensor(shards, prefix, name) else {
        let Some((packed_name, packed_len)) = packed else {
            return Ok(None);
        };
        let p = tensor(shards, prefix, packed_name).ok_or(Refusal::MissingScales)?;
        if p.dtype != Some(Dtype::I16) || p.numel() != Some(packed_len) {
            return Err(Refusal::BadScales);
        }
        return Ok(Some(ScaleForm::Signs));
    };
    if e.dtype != Some(Dtype::F16) || e.numel() != Some(len) {
        return Err(if packed.is_none() {
            Refusal::BadBias
        } else {
            Refusal::BadScales
        });
    }
    Ok(Some(ScaleForm::Fp16))
}

/// The admission of one trellis tensor.
pub fn admit(shards: &ShardSet, prefix: &str, e: &ShardEntry) -> Result<GroupInfo, Refusal> {
    if e.dtype != Some(Dtype::I16) || e.shape.len() != 3 {
        return Err(Refusal::NotI16Rank3);
    }
    let width = Width::from_tile_words(e.shape[2]).ok_or(Refusal::BadWidth)?;
    let k = e.shape[0].checked_mul(16).ok_or(Refusal::Overflow)?;
    let n = e.shape[1].checked_mul(16).ok_or(Refusal::Overflow)?;
    if k == 0 || n == 0 || !k.is_multiple_of(128) || !n.is_multiple_of(128) {
        return Err(Refusal::NotBlockAligned);
    }
    let mut codebook = Codebook::Inst3;
    let mut marker = None;
    for cb in [Codebook::Mcg, Codebook::Mul1] {
        if let Some(m) = cb.marker_suffix().and_then(|s| tensor(shards, prefix, s)) {
            if marker.is_some() {
                return Err(Refusal::TwoMarkers);
            }
            marker = Some((cb, m));
            codebook = cb;
        }
    }
    if width.is_half() && codebook != Codebook::Mul1 {
        return Err(Refusal::HalfWidthNeedsMul1);
    }
    if let Some((cb, m)) = marker {
        let scalar = matches!(m.dtype, Some(Dtype::I32 | Dtype::U32)) && m.len == 4;
        let value = if scalar {
            match shards.read(m) {
                Ok(b) if b.len() == 4 => Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
                _ => None,
            }
        } else {
            None
        };
        if value != cb.marker_value() {
            return Err(Refusal::BadMarker);
        }
    }
    let su = scale(shards, prefix, "suh", k, Some(("su", k / 16)))?;
    let sv = scale(shards, prefix, "svh", n, Some(("sv", n / 16)))?;
    let (Some(su), Some(sv)) = (su, sv) else {
        return Err(Refusal::MissingScales);
    };
    if su != sv {
        return Err(Refusal::MixedScales);
    }
    let bias = scale(shards, prefix, "bias", n, None)?.is_some();
    Ok(GroupInfo {
        prefix: prefix.to_string(),
        width,
        codebook,
        k,
        n,
        scales: su,
        bias,
    })
}

fn i16s(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn u16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn read_part(shards: &ShardSet, prefix: &str, suffix: &str) -> Result<Vec<u8>> {
    let e = tensor(shards, prefix, suffix)
        .with_context(|| format!("EXL3 group {prefix} lost its {suffix} tensor"))?;
    shards.read(e)
}

/// Reads an admitted group's tensors into an [`Exl3Tensor`], re-checking the trellis layout.
pub fn load_group(shards: &ShardSet, g: &GroupInfo) -> Result<Exl3Tensor> {
    let prefix = g.prefix.as_str();
    let want = [g.k / 16, g.n / 16, g.width.tile_words()];
    let te = tensor(shards, prefix, "trellis")
        .with_context(|| format!("EXL3 group {prefix} has no trellis"))?;
    if !te.is(Dtype::I16, &want) {
        bail!(
            "EXL3 {prefix}.trellis is {} {:?}, want I16 {want:?}",
            te.dtype_name,
            te.shape
        );
    }
    let trellis = i16s(&shards.read(te)?);
    let (suh, svh) = match g.scales {
        ScaleForm::Fp16 => (
            u16s(&read_part(shards, prefix, "suh")?),
            u16s(&read_part(shards, prefix, "svh")?),
        ),
        ScaleForm::Signs => (
            super::format::unpack_signs(&i16s(&read_part(shards, prefix, "su")?)),
            super::format::unpack_signs(&i16s(&read_part(shards, prefix, "sv")?)),
        ),
    };
    let bias = if g.bias {
        Some(u16s(&read_part(shards, prefix, "bias")?))
    } else {
        None
    };
    Exl3Tensor::new(
        g.prefix.clone(),
        g.codebook,
        g.width,
        g.k,
        g.n,
        trellis,
        suh,
        svh,
        bias,
        g.scales,
    )
}
