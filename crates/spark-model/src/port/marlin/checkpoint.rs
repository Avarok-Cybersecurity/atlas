// SPDX-License-Identifier: AGPL-3.0-only
// The tensor layouts follow vLLM (https://github.com/vllm-project/vllm, commit 8a36f08):
// vllm/model_executor/layers/quantization/utils/quant_utils.py:834-941 (`pack_rows`, `pack_cols`,
// `unpack_cols`, `awq_pack`: GPTQ packs eight codes along K, AWQ eight along N with the
// 0,2,4,6,1,3,5,7 interleave), marlin_utils.py:577-601 (`awq_to_marlin_zero_points`) and the act-order
// limit of csrc/quantization/marlin/marlin_template.h:320-321. Copyright contributors to the vLLM
// project, Apache-2.0. Changes: restated as Rust parsers of raw little-endian safetensors bytes into
// Atlas's `LogicalInt4`; GPTQ v1 zero offsets, symmetric collapse and the act-order refusal are Atlas's.

//! Responsibility: parse GPTQ and AWQ int4 checkpoint tensors (raw little-endian bytes, as a
//! safetensors loader holds them) into a [`LogicalInt4`].
//!
//! * GPTQ: `qweight` I32 `[k / 8, n]` (code `kk` of a word at bits `4 (kk % 8)`), `scales` F16 or BF16
//!   `[k / group, n]`, optional `qzeros` I32 `[k / group, n / 8]` (zero `nn` at bits `4 (nn % 8)`;
//!   AutoGPTQ v1 checkpoints store `z - 1`), optional `g_idx` I32 `[k]`.
//! * AWQ: `qweight` I32 `[k, n / 8]` and `qzeros` I32 `[k / group, n / 8]`, both with column `nn` at
//!   in-word position `AWQ_UNDO_PACK[nn % 8]`; `scales` `[k / group, n]`.
//!
//! When every zero point is 8, the weight is the `kU4B8` type (no zero-point buffer): `(q - 8) * s` is
//! the same arithmetic either way.

use anyhow::{Result, bail, ensure};

use super::reference::{LogicalInt4, bf16_to_f32, f16_to_bf16};
use super::repack::AWQ_UNDO_PACK;

/// The dtype of a checkpoint's scale tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleDtype {
    /// FP16 (GPTQ and AWQ exports); converted to BF16 with round to nearest even.
    F16,
    Bf16,
}

/// How a GPTQ checkpoint stores zero points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GptqZeroFormat {
    /// AutoGPTQ v1 (`checkpoint_format = "gptq"`, the default): stored value `z - 1` (mod 16).
    V1,
    /// `checkpoint_format = "gptq_v2"`: stored value `z`.
    V2,
}

/// A GPTQ int4 tensor set for one linear layer.
#[derive(Clone, Copy, Debug)]
pub struct GptqWeights<'a> {
    /// I32 LE `[k / 8, n]`.
    pub qweight: &'a [u8],
    /// `[k / group, n]` in `scale_dtype`.
    pub scales: &'a [u8],
    pub scale_dtype: ScaleDtype,
    /// I32 LE `[k / group, n / 8]`; `None` means symmetric (zero 8).
    pub qzeros: Option<&'a [u8]>,
    /// I32 LE `[k]`; `None` means `g_idx[kk] = kk / group`.
    pub g_idx: Option<&'a [u8]>,
    /// Outputs.
    pub n: usize,
    /// Inputs.
    pub k: usize,
    pub group: usize,
    pub zero_format: GptqZeroFormat,
}

/// An AWQ int4 tensor set for one linear layer.
#[derive(Clone, Copy, Debug)]
pub struct AwqWeights<'a> {
    /// I32 LE `[k, n / 8]`, AWQ-interleaved.
    pub qweight: &'a [u8],
    /// I32 LE `[k / group, n / 8]`, AWQ-interleaved.
    pub qzeros: &'a [u8],
    /// `[k / group, n]` in `scale_dtype`.
    pub scales: &'a [u8],
    pub scale_dtype: ScaleDtype,
    pub n: usize,
    pub k: usize,
    pub group: usize,
}

/// Little-endian u32 words of a byte tensor of `words` elements.
pub fn words_le(b: &[u8], words: usize, what: &str) -> Result<Vec<u32>> {
    ensure!(
        b.len() == words * 4,
        "marlin: {what} has {} bytes, {} expected",
        b.len(),
        words * 4
    );
    Ok(b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// BF16 scales of a `count`-element scale tensor; non-finite scales are refused.
pub fn scales_bf16(b: &[u8], count: usize, dtype: ScaleDtype) -> Result<Vec<u16>> {
    ensure!(
        b.len() == count * 2,
        "marlin: scales have {} bytes, {} expected",
        b.len(),
        count * 2
    );
    let out: Vec<u16> = b
        .chunks_exact(2)
        .map(|c| {
            let v = u16::from_le_bytes([c[0], c[1]]);
            match dtype {
                ScaleDtype::F16 => f16_to_bf16(v),
                ScaleDtype::Bf16 => v,
            }
        })
        .collect();
    if let Some(i) = out.iter().position(|&s| !bf16_to_f32(s).is_finite()) {
        bail!("marlin: scale {i} is not finite");
    }
    Ok(out)
}

fn check_dims(n: usize, k: usize, group: usize) -> Result<()> {
    ensure!(
        n > 0 && k > 0 && n.is_multiple_of(8) && k.is_multiple_of(8),
        "marlin: [{n}x{k}] is not a packed int4 shape"
    );
    ensure!(
        group > 0 && k.is_multiple_of(group),
        "marlin: K={k} is not a multiple of the group {group}"
    );
    Ok(())
}

/// Refuse act-order checkpoints: the kernel's groups are contiguous runs of K
/// (`static_assert(group_blocks != 0, "runtime activation ordering is not supported")`).
pub fn check_g_idx(g_idx: &[u8], k: usize, group: usize) -> Result<()> {
    let g = words_le(g_idx, k, "g_idx")?;
    if let Some(kk) = (0..k).find(|&kk| g[kk] as usize != kk / group) {
        bail!(
            "marlin: act-order GPTQ (desc_act) is not supported: g_idx[{kk}] = {} where the \
             contiguous grouping gives {} (the Marlin kernel has no activation reorder)",
            g[kk] as i32,
            kk / group
        );
    }
    Ok(())
}

/// Decode a zero-point matrix to `[groups][n]`, or `None` when every zero is 8.
fn collapse_zeros(z: Vec<u8>) -> Option<Vec<u8>> {
    if z.iter().all(|&v| v == 8) {
        None
    } else {
        Some(z)
    }
}

/// A GPTQ tensor set as a logical weight.
pub fn gptq_to_logical(w: &GptqWeights<'_>) -> Result<LogicalInt4> {
    let (n, k, group) = (w.n, w.k, w.group);
    check_dims(n, k, group)?;
    if let Some(g) = w.g_idx {
        check_g_idx(g, k, group)?;
    }
    let groups = k / group;
    let qw = words_le(w.qweight, k / 8 * n, "GPTQ qweight")?;
    let mut q = vec![0u8; k * n];
    for kk in 0..k {
        let shift = (kk % 8) * 4;
        let row = &qw[(kk / 8) * n..(kk / 8 + 1) * n];
        for (nn, &word) in row.iter().enumerate() {
            q[kk * n + nn] = ((word >> shift) & 0xF) as u8;
        }
    }
    let scales = scales_bf16(w.scales, groups * n, w.scale_dtype)?;
    let zeros = match w.qzeros {
        None => None,
        Some(b) => {
            let zw = words_le(b, groups * n / 8, "GPTQ qzeros")?;
            let offset = match w.zero_format {
                GptqZeroFormat::V1 => 1u32,
                GptqZeroFormat::V2 => 0,
            };
            let mut z = vec![0u8; groups * n];
            for g in 0..groups {
                for nn in 0..n {
                    let stored = (zw[g * (n / 8) + nn / 8] >> ((nn % 8) * 4)) & 0xF;
                    z[g * n + nn] = ((stored + offset) & 0xF) as u8;
                }
            }
            collapse_zeros(z)
        }
    };
    let l = LogicalInt4 {
        n,
        k,
        group,
        q,
        scales,
        zeros,
    };
    l.validate()?;
    Ok(l)
}

/// An AWQ tensor set as a logical weight.
pub fn awq_to_logical(w: &AwqWeights<'_>) -> Result<LogicalInt4> {
    let (n, k, group) = (w.n, w.k, w.group);
    check_dims(n, k, group)?;
    let groups = k / group;
    let qw = words_le(w.qweight, k * (n / 8), "AWQ qweight")?;
    let zw = words_le(w.qzeros, groups * (n / 8), "AWQ qzeros")?;
    let awq_code = |words: &[u32], row: usize, nn: usize| -> u8 {
        ((words[row * (n / 8) + nn / 8] >> (AWQ_UNDO_PACK[nn % 8] * 4)) & 0xF) as u8
    };
    let mut q = vec![0u8; k * n];
    for kk in 0..k {
        for nn in 0..n {
            q[kk * n + nn] = awq_code(&qw, kk, nn);
        }
    }
    let mut z = vec![0u8; groups * n];
    for g in 0..groups {
        for nn in 0..n {
            z[g * n + nn] = awq_code(&zw, g, nn);
        }
    }
    let scales = scales_bf16(w.scales, groups * n, w.scale_dtype)?;
    let l = LogicalInt4 {
        n,
        k,
        group,
        q,
        scales,
        zeros: collapse_zeros(z),
    };
    l.validate()?;
    Ok(l)
}

/// GPTQ `qweight` words of a `[k][n]` code matrix (`pack_rows`): the inverse of the parser, for tests
/// and for building synthetic checkpoints in the microtest.
pub fn gptq_pack(q: &[u8], k: usize, n: usize) -> Vec<u32> {
    let mut out = vec![0u32; k / 8 * n];
    for kk in 0..k {
        for nn in 0..n {
            out[(kk / 8) * n + nn] |= u32::from(q[kk * n + nn] & 0xF) << ((kk % 8) * 4);
        }
    }
    out
}

/// AWQ words of a `[rows][n]` code matrix (`awq_pack`), for `qweight` (`rows = k`) and `qzeros`
/// (`rows = groups`).
pub fn awq_pack(q: &[u8], rows: usize, n: usize) -> Vec<u32> {
    let mut out = vec![0u32; rows * (n / 8)];
    for r in 0..rows {
        for nn in 0..n {
            out[r * (n / 8) + nn / 8] |=
                u32::from(q[r * n + nn] & 0xF) << (AWQ_UNDO_PACK[nn % 8] * 4);
        }
    }
    out
}

/// GPTQ `qzeros` words of a `[groups][n]` zero matrix (`pack_cols`), storing `z - offset`.
pub fn gptq_pack_zeros(z: &[u8], groups: usize, n: usize, format: GptqZeroFormat) -> Vec<u32> {
    let offset = match format {
        GptqZeroFormat::V1 => 15u32,
        GptqZeroFormat::V2 => 0,
    };
    let mut out = vec![0u32; groups * (n / 8)];
    for g in 0..groups {
        for nn in 0..n {
            let stored = (u32::from(z[g * n + nn]) + offset) & 0xF;
            out[g * (n / 8) + nn / 8] |= stored << ((nn % 8) * 4);
        }
    }
    out
}

/// Little-endian bytes of u32 words (the safetensors image of an I32 tensor).
pub fn le_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}
