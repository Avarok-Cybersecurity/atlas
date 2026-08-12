// SPDX-License-Identifier: AGPL-3.0-only
//! CPU reference dequantization for GGUF/GGML quantized weight blocks.
//!
//! This is the correctness oracle and portable fallback for the Atlas GGUF
//! loader. Every routine reproduces llama.cpp's `dequantize_row_*` index
//! arithmetic exactly (little-endian throughout). Clarity is preferred over
//! speed: the GPU dequant kernels are the fast path; these functions are what
//! those kernels are validated against, and what runs when a ggml type has no
//! GPU kernel or under `MockGpuBackend` unit tests (which do not execute
//! kernels).

use anyhow::{Context, Result, bail};

mod blocks;

#[cfg(test)]
mod tests;

/// A GGML block-quantization type.
///
/// The PrismML-private `Q2_0` (ggml id 42) is polymorphic in its group size
/// (128 in the shipped Ternary-Bonsai file, 64 in the fork master), so the
/// group is carried in the variant rather than hardcoded. Standard `Tq2_0`
/// (id 35, group-256, scale-at-end) is a *separate* entry and must not be
/// conflated with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Bf16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Tq1_0,
    Tq2_0,
    /// PrismML id 42: 2-bit, symmetric, scale-at-front, contiguous codes.
    /// `group` ∈ {64, 128}; block = 2 + group/4 bytes.
    Q2_0 {
        group: usize,
    },
}

impl GgmlType {
    /// Map a raw ggml type id to a `GgmlType`. `q2_0_group` supplies the group
    /// size for the id-42 PrismML type (ignored for every other id); pass the
    /// value read from the fork's GGUF metadata (128 default, 64 fork-master).
    pub fn from_id(id: u32, q2_0_group: usize) -> Result<Self> {
        Ok(match id {
            0 => Self::F32,
            1 => Self::F16,
            30 => Self::Bf16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            34 => Self::Tq1_0,
            35 => Self::Tq2_0,
            42 => Self::Q2_0 { group: q2_0_group },
            other => bail!("unsupported / unknown ggml type id {other}"),
        })
    }

    /// Number of dequantized elements produced by one block (`QK` / `QK_K`).
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::Tq1_0
            | Self::Tq2_0 => 256,
            Self::Q2_0 { group } => group,
        }
    }

    /// On-disk byte size of one block.
    pub fn block_bytes(self) -> Result<usize> {
        Ok(match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            Self::Tq1_0 => 54,
            Self::Tq2_0 => 66,
            Self::Q2_0 { group } => {
                if group == 0 || !group.is_multiple_of(4) {
                    bail!("Q2_0 group size must be a positive multiple of 4, got {group}");
                }
                2 + group / 4
            }
        })
    }
}

/// True if the CPU reference dequant handles this ggml type id.
pub(crate) fn supports(id: u32) -> bool {
    // The group value is irrelevant to the support decision (id 42 is always
    // supported once a group is known); probe with the default group.
    GgmlType::from_id(id, 128).is_ok()
}

/// Dequantize a tensor's raw block bytes into little-endian BF16 host bytes.
/// `q2_group` is the id-42 PrismML group size (128 default / 64 fork-master);
/// ignored for every other type. This is the loader's CPU-fallback entry point.
pub(crate) fn to_bf16_bytes(
    id: u32,
    q2_group: usize,
    raw: &[u8],
    n_elements: usize,
) -> Result<Vec<u8>> {
    let t = GgmlType::from_id(id, q2_group)?;
    let mut out = vec![0u16; n_elements];
    dequant_to_bf16(t, raw, n_elements, &mut out)?;
    let mut bytes = Vec::with_capacity(n_elements * 2);
    for v in &out {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Ok(bytes)
}

/// Dequantize `n_elements` from packed `block_bytes` into `out` as `f32`.
///
/// `block_bytes` must contain at least `ceil(n_elements / block_size)` whole
/// blocks; `n_elements` must be a whole multiple of the type's block size
/// (GGUF tensors always are — the last dim is padded to the block size).
pub fn dequant_to_f32(
    t: GgmlType,
    block_bytes: &[u8],
    n_elements: usize,
    out: &mut [f32],
) -> Result<()> {
    if out.len() < n_elements {
        bail!("out buffer too small: {} < {}", out.len(), n_elements);
    }

    // Passthrough / widen types operate on the whole run at once.
    match t {
        GgmlType::F32 => {
            let need = n_elements.checked_mul(4).context("f32 size overflow")?;
            require_len(block_bytes, need)?;
            blocks::widen_f32(&block_bytes[..need], &mut out[..n_elements]);
            return Ok(());
        }
        GgmlType::F16 => {
            let need = n_elements.checked_mul(2).context("f16 size overflow")?;
            require_len(block_bytes, need)?;
            blocks::widen_f16(&block_bytes[..need], &mut out[..n_elements]);
            return Ok(());
        }
        GgmlType::Bf16 => {
            let need = n_elements.checked_mul(2).context("bf16 size overflow")?;
            require_len(block_bytes, need)?;
            blocks::widen_bf16(&block_bytes[..need], &mut out[..n_elements]);
            return Ok(());
        }
        _ => {}
    }

    let qk = t.block_size();
    if !n_elements.is_multiple_of(qk) {
        bail!("n_elements {n_elements} is not a multiple of block size {qk} for {t:?}");
    }
    let bb = t.block_bytes()?;
    let n_blocks = n_elements / qk;
    let need = n_blocks
        .checked_mul(bb)
        .context("block byte count overflow")?;
    require_len(block_bytes, need)?;

    for b in 0..n_blocks {
        let blk = &block_bytes[b * bb..b * bb + bb];
        let o = &mut out[b * qk..b * qk + qk];
        dequant_block(t, blk, o);
    }
    Ok(())
}

/// Dequantize into BF16 bit patterns (round-to-nearest-even), the form the
/// GGUF loader hands to `WeightTensor { dtype: BF16 }`. Goes via `f32` for a
/// single, obviously-correct code path.
pub fn dequant_to_bf16(
    t: GgmlType,
    block_bytes: &[u8],
    n_elements: usize,
    out: &mut [u16],
) -> Result<()> {
    if out.len() < n_elements {
        bail!("out buffer too small: {} < {}", out.len(), n_elements);
    }
    let mut tmp = vec![0f32; n_elements];
    dequant_to_f32(t, block_bytes, n_elements, &mut tmp)?;
    for (o, &f) in out[..n_elements].iter_mut().zip(tmp.iter()) {
        *o = f32_to_bf16_bits(f);
    }
    Ok(())
}

#[inline]
fn require_len(buf: &[u8], need: usize) -> Result<()> {
    if buf.len() < need {
        bail!("input block bytes too small: {} < {}", buf.len(), need);
    }
    Ok(())
}

/// Dispatch one block to its per-type routine. Passthrough arms are reachable
/// only via callers other than `dequant_to_f32` (which short-circuits them);
/// they remain correct for a single-element slice.
fn dequant_block(t: GgmlType, blk: &[u8], out: &mut [f32]) {
    match t {
        GgmlType::F32 => blocks::widen_f32(blk, out),
        GgmlType::F16 => blocks::widen_f16(blk, out),
        GgmlType::Bf16 => blocks::widen_bf16(blk, out),
        GgmlType::Q4_0 => blocks::dequant_q4_0(blk, out),
        GgmlType::Q4_1 => blocks::dequant_q4_1(blk, out),
        GgmlType::Q5_0 => blocks::dequant_q5_0(blk, out),
        GgmlType::Q5_1 => blocks::dequant_q5_1(blk, out),
        GgmlType::Q8_0 => blocks::dequant_q8_0(blk, out),
        GgmlType::Q8_1 => blocks::dequant_q8_1(blk, out),
        GgmlType::Q2K => blocks::dequant_q2_k(blk, out),
        GgmlType::Q3K => blocks::dequant_q3_k(blk, out),
        GgmlType::Q4K => blocks::dequant_q4_k(blk, out),
        GgmlType::Q5K => blocks::dequant_q5_k(blk, out),
        GgmlType::Q6K => blocks::dequant_q6_k(blk, out),
        GgmlType::Q8K => blocks::dequant_q8_k(blk, out),
        GgmlType::Tq1_0 => blocks::dequant_tq1_0(blk, out),
        GgmlType::Tq2_0 => blocks::dequant_tq2_0(blk, out),
        GgmlType::Q2_0 { group } => blocks::dequant_q2_0_gn(blk, out, group),
    }
}

// ---- little-endian scalar helpers (no external deps) ----------------------

/// IEEE binary16 → f32, handling zero/subnormal/inf/nan.
#[inline]
pub(super) fn f16_to_f32(h: u16) -> f32 {
    let sign = if (h >> 15) & 1 == 1 { -1.0f32 } else { 1.0f32 };
    let exp = (h >> 10) & 0x1f;
    let mant = (h & 0x3ff) as f32;
    match exp {
        0 => sign * mant * 2f32.powi(-24), // subnormal / zero
        0x1f => {
            if mant == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + mant / 1024.0) * 2f32.powi(exp as i32 - 15),
    }
}

#[inline]
pub(super) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[inline]
pub(super) fn rd_f16(b: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}

/// f32 → bf16 bits, round-to-nearest-even, NaN preserved (quieted).
#[inline]
pub(super) fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if f.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounding_bias = 0x0000_7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}
