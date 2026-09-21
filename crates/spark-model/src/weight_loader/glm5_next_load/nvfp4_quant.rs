// SPDX-License-Identifier: AGPL-3.0-only

//! `f32` → packed NVFP4, on the host, in ModelOpt's convention — the inverse of
//! [`super::nvfp4_dequant`].
//!
//! # Why this exists
//!
//! `LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3` quantises the MTP block's routed
//! experts (`layers.45.mlp.experts.E.{gate,up,down}_proj`) exactly like the
//! text stack's. NVIDIA's official export, `nvidia/GLM-5.3-Flash-NVFP4`, leaves
//! them at full width:
//!
//! ```text
//! model.language_model.layers.45.mlp.experts.0.gate_proj.weight
//!   LibertAIDAI : U8 [2048, 2048] + .weight_scale F8_E4M3 + .weight_scale_2
//!   nvidia      : BF16 [2048, 4096], no scales at all
//! ```
//!
//! The routed-expert forward is w4a16 end to end — `Nvfp4Proj`, the global-id
//! pointer tables, `w4a16_gemv_sw_moe` — and there is no BF16 expert path to
//! fall back to. So the BF16 originals are quantised here, once, at load, into
//! exactly the operand triple [`super::bind_expert`] already hands the kernels.
//!
//! # The convention, and why it must be THIS one
//!
//! ModelOpt's NVFP4, reproduced so a re-quantised expert is indistinguishable
//! from a checkpoint-quantised one to the kernel:
//!
//! ```text
//! weight_scale_2 = amax(tensor) / (6 * 448)          per TENSOR, f32
//! block_scale    = amax(block16) / 6 / weight_scale_2  per 16, encoded E4M3
//! code           = rne_e2m1( w / (e4m3(block_scale) * weight_scale_2) )
//! ```
//!
//! `6` is the largest `E2M1` magnitude and `448` the largest finite `E4M3`, so
//! the product is the largest value the pair can represent and the tensor's
//! amax lands on it. Both constants are read off the runtime's own tables
//! rather than typed here, for the same reason
//! [`super::nvfp4_dequant`] reads them: three separate places — this
//! quantiser, that dequantiser and the CUDA kernels — must agree about the
//! same bytes.
//!
//! 🪤 **Round to nearest, ties to EVEN, on the code index.** Consecutive
//! `E2M1` codes (and consecutive `E4M3` codes) differ in the mantissa LSB, so
//! ties-to-even-code IS the IEEE-style ties-to-even the format specifies.
//! Truncation instead would bias every expert's weights toward zero — a
//! uniformly slightly-wrong model with no failing assertion anywhere.
//!
//! 🪤 The packing is the same as the dequantiser's: even flat index = LOW
//! nibble.

use anyhow::{Result, bail};
use spark_runtime::kv_dequant::{NVFP4_E2M1_LUT, NVFP4_GROUP_SIZE, e4m3_lut};

/// `E4M3` codes `0x00..=0x7E`: every finite non-negative value, ascending.
/// `0x7F` is NaN and `0x80..` are the negatives, neither of which a block
/// scale may be.
const E4M3_FINITE_CODES: usize = 0x7F;

/// The largest `E2M1` magnitude (6.0) — the top of the codebook's first half.
fn e2m1_max() -> f32 {
    NVFP4_E2M1_LUT[7]
}

/// The largest finite `E4M3` value (448.0).
fn e4m3_max() -> f32 {
    e4m3_lut()[E4M3_FINITE_CODES - 1]
}

/// One quantised projection, in the three pieces `Nvfp4Proj` is built from.
#[derive(Debug)]
pub(super) struct Nvfp4Blob {
    /// `[rows, cols / 2]` U8, two `e2m1` codes per byte.
    pub packed: Vec<u8>,
    /// `[rows, cols / 16]` `E4M3` block scales, one byte each.
    pub scales: Vec<u8>,
    /// The per-tensor global scale.
    pub scale_2: f32,
}

/// Quantise a row-major `f32 [rows, cols]` weight to ModelOpt NVFP4.
pub(super) fn quantize_to_nvfp4(
    what: &str,
    values: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Nvfp4Blob> {
    if values.len() != rows * cols {
        bail!(
            "{what}: {} elements, expected [{rows}, {cols}] = {}",
            values.len(),
            rows * cols
        );
    }
    // 🪤 `cols == 0` passes `is_multiple_of` and then makes the band width
    // zero, which `chunks` panics on rather than rejects.
    if cols == 0 || !cols.is_multiple_of(NVFP4_GROUP_SIZE) {
        bail!(
            "{what}: {cols} columns is not a whole number of {NVFP4_GROUP_SIZE}-element \
             NVFP4 blocks"
        );
    }
    // 🪤 `f32::max` IGNORES NaN — it returns the other operand — so folding the
    // amax and then testing it for finiteness catches an infinity and misses
    // every NaN. The scan has to reject per element.
    let mut amax = 0.0f32;
    for &v in values {
        if !v.is_finite() {
            bail!("{what}: weight contains a non-finite value; refusing to quantise it");
        }
        amax = amax.max(v.abs());
    }
    // 🪤 An all-zero tensor has no amax to scale by. `1.0` keeps every product
    // finite and every code zero; `0.0` would make the dequant a 0 * inf NaN.
    let scale_2 = if amax > 0.0 {
        amax / (e2m1_max() * e4m3_max())
    } else {
        1.0
    };

    let groups_per_row = cols / NVFP4_GROUP_SIZE;
    let mut packed = vec![0u8; rows * cols / 2];
    let mut scales = vec![0u8; rows * groups_per_row];

    // Rows are INDEPENDENT once `scale_2` is known — a block never reaches
    // past its own 16 columns — so the sweep splits by row band. Measured on
    // the 20-core GB10, one `[2048, 4096]` projection: 69 ms serial, 10 ms
    // over 20 bands; across the 432 projections of one EP=2 rank's MTP layer
    // that is 30 s of boot recovered. `std::thread::scope` rather than a new
    // dependency: the bands are disjoint `chunks_mut` slices, which is the
    // one case scoped threads make trivially sound.
    let bands = std::thread::available_parallelism().map_or(1, |n| n.get());
    let band_rows = rows.div_ceil(bands).max(1);
    std::thread::scope(|s| {
        for ((v, p), sc) in values
            .chunks(band_rows * cols)
            .zip(packed.chunks_mut(band_rows * cols / 2))
            .zip(scales.chunks_mut(band_rows * groups_per_row))
        {
            s.spawn(move || quantize_rows(v, p, sc, cols, scale_2));
        }
    });
    Ok(Nvfp4Blob {
        packed,
        scales,
        scale_2,
    })
}

/// One band of whole rows. `values`, `packed` and `scales` are the band's
/// slices of the three buffers, so every index here is band-local.
fn quantize_rows(values: &[f32], packed: &mut [u8], scales: &mut [u8], cols: usize, scale_2: f32) {
    let groups_per_row = cols / NVFP4_GROUP_SIZE;
    let e4m3 = e4m3_lut();
    for (r, row) in values.chunks(cols).enumerate() {
        for g in 0..groups_per_row {
            let base = g * NVFP4_GROUP_SIZE;
            let block = &row[base..base + NVFP4_GROUP_SIZE];
            let bmax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let sb = encode_e4m3_rne(bmax / e2m1_max() / scale_2);
            scales[r * groups_per_row + g] = sb;
            // The DECODED scale, not the requested one: the kernel will read
            // the byte, so the codes must be chosen against what that byte
            // means, never against the real number it was rounded from.
            let eff = e4m3[sb as usize] * scale_2;
            let inv = if eff > 0.0 { 1.0 / eff } else { 0.0 };
            for (i, &v) in block.iter().enumerate() {
                let code = encode_e2m1_rne(v * inv);
                let flat = r * cols + base + i;
                if flat.is_multiple_of(2) {
                    packed[flat / 2] |= code;
                } else {
                    packed[flat / 2] |= code << 4;
                }
            }
        }
    }
}

/// Nearest `E2M1` code to `v`, ties to even, sign preserved.
///
/// 🪤 The sign survives a magnitude that rounds to zero, so `-0.0` is code
/// `0x8` and not `0x0`. That is what the hardware convert does and what the
/// codebook's `-0.0` entry exists for; both decode to a zero the GEMM adds.
fn encode_e2m1_rne(v: f32) -> u8 {
    let sign = if v.is_sign_negative() { 0x8u8 } else { 0 };
    let mag = &NVFP4_E2M1_LUT[..8];
    sign | nearest_even_code(mag, v.abs()) as u8
}

/// Nearest `E4M3` code to a non-negative `v`, ties to even.
fn encode_e4m3_rne(v: f32) -> u8 {
    nearest_even_code(&e4m3_lut()[..E4M3_FINITE_CODES], v) as u8
}

/// Index of the entry of an ASCENDING ladder nearest to `v`, ties to the even
/// index. `v` at or past the top saturates; `v` at or below zero is index 0.
///
/// Ties go to the even index because consecutive codes of both `E2M1` and
/// `E4M3` differ in the mantissa's low bit — so "even index" and "even
/// mantissa" are the same rule, and this is round-to-nearest-even as the
/// format defines it.
fn nearest_even_code(ladder: &[f32], v: f32) -> usize {
    let top = ladder.len() - 1;
    // NaN never reaches here (`quantize_to_nvfp4` rejects non-finite input and
    // every other argument is an `abs()` of a finite value), but say so rather
    // than let a partial order decide it silently.
    if v.is_nan() || v <= ladder[0] {
        return 0;
    }
    if v >= ladder[top] {
        return top;
    }
    let hi = ladder.partition_point(|&x| x < v);
    let lo = hi - 1;
    let below = v - ladder[lo];
    let above = ladder[hi] - v;
    if below < above || (below == above && lo.is_multiple_of(2)) {
        lo
    } else {
        hi
    }
}

#[cfg(test)]
#[path = "nvfp4_quant_tests.rs"]
mod tests;
