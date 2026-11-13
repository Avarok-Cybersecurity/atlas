// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/core/affine4_host.zig:41-66 (quantizeMlx), zig/kernels/cuda/affine_codes.cuh:12-45 (code_at,
// pair_at) and zig/kernels/cuda/affine8.cu:319-385 (the two packers). Copyright 2026 TensorFold
// contributors. Licensed under the Apache License, Version 2.0; code written before TensorFold 0.6.0 is
// also available under the MIT License. See TENSORFOLD-ATTRIBUTION.md. Changes for Atlas: Zig and CUDA
// to Rust; the quantizer generalized to 2^bits - 1 levels with the warp-order clip search of
// `affine_n_ada.cu`; a lane emulation of the GEMV's operand reads, a float64 GEMV with its error
// bound, and an fp32 model of the kernel's epilogue are Atlas additions.
//! Responsibility: the CPU oracle of the 2/3/5/6/8-bit affine path: bf16 rounding, MLX's quantizer
//! (bit-exact with `affine_n_quantize_ada`), MLX dequantization, the two tile packers (bit-exact with
//! `affine_n_pack_ada` / `affine8_pack_ada`), the dequant kernel's index math, a lane-by-lane emulation
//! of how the GEMV reads its MMA operands, a float64 reference GEMV with the gate's error bound, and
//! an fp32 model of the GEMV's arithmetic in both the unsplit and the split order. Used by
//! `widths_tests.rs` and by `examples/affine_n_microtest.rs`.

use super::layout;

/// bf16 bits of `x`, round to nearest even; NaN stays a quiet NaN.
pub fn bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    if x.is_nan() {
        return ((u >> 16) | 0x40) as u16;
    }
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

/// The f32 value of bf16 bits.
pub fn bf16_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// `x` rounded to bf16 and back.
pub fn round_bf16(x: f32) -> f32 {
    bf16_f32(bf16_bits(x))
}

/// An MLX affine weight on the host: `(n, k * bits / 32)` words, `(n, k / 64)` bf16 scales and biases.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlxWeight {
    pub words: Vec<u32>,
    pub scales: Vec<u16>,
    pub biases: Vec<u16>,
    pub n: usize,
    pub k: usize,
    pub bits: u8,
}

/// The packed tiles and transposed tables the GEMV reads (`[k / 64][npad]` tables).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tiles {
    pub words: Vec<u32>,
    pub scales_t: Vec<u16>,
    pub biases_t: Vec<u16>,
    pub n: usize,
    pub k: usize,
    pub bits: u8,
}

fn levels(bits: u8) -> f32 {
    ((1u32 << bits) - 1) as f32
}

/// Code `i` of an MLX row packed `bits` wide (a code may cross a word boundary).
pub fn code_at(row: &[u32], i: usize, bits: u8) -> u32 {
    let b = usize::from(bits);
    let bit = i * b;
    let sh = bit & 31;
    let wd = bit >> 5;
    let mut v = row[wd] >> sh;
    if sh + b > 32 {
        v |= row[wd + 1] << (32 - sh);
    }
    v & ((1u32 << bits) - 1)
}

/// MLX's scale and bias for the range `[l, h]` at `nb` levels, or `None` when the group cannot be
/// represented. Every operation is the same IEEE fp32 operation as `mlx_params` in the kernel.
pub fn mlx_params(l: f32, h: f32, nb: f32) -> Option<(f32, f32)> {
    let step = ((h - l) / nb).max(1e-7);
    if !step.is_finite() {
        return None;
    }
    let low = l.abs() > h.abs();
    let sg = if low { step } else { -step };
    let edge = if low { l } else { h };
    let q0 = (edge / sg).round();
    let s = if q0 == 0.0 { sg } else { edge / q0 };
    let b = if q0 == 0.0 { 0.0 } else { edge };
    if !s.is_finite() || s == 0.0 || !b.is_finite() {
        return None;
    }
    (round_bf16(s).is_finite() && round_bf16(b).is_finite()).then_some((s, b))
}

/// The code of `v`: round half away from zero of `(v - b) / s`, clamped to `[0, nb]`.
pub fn mlx_code(v: f32, s: f32, b: f32, nb: f32) -> u32 {
    ((v - b) / s).round().clamp(0.0, nb) as u32
}

/// Squared error of one group under `(s, b)` with bf16-stored parameters, the kernel's fp32 sum:
/// lane l adds its entries 2l and 2l + 1, then a butterfly over partners XOR 16, 8, 4, 2, 1.
pub fn group_sse(g: &[f32], s: f32, b: f32, nb: f32) -> f32 {
    let (sb, bb) = (round_bf16(s), round_bf16(b));
    let err = |v: f32| (sb * mlx_code(v, s, b, nb) as f32 + bb) - v;
    let mut p = [0f32; 32];
    for (l, slot) in p.iter_mut().enumerate() {
        let (d0, d1) = (err(g[2 * l]), err(g[2 * l + 1]));
        // The kernel adds d0^2 to a zero first; 0 + d0^2 is d0^2 exactly, so it is left out here.
        *slot = d0 * d0 + d1 * d1;
    }
    for off in [16usize, 8, 4, 2, 1] {
        let prev = p;
        for (l, slot) in p.iter_mut().enumerate() {
            *slot = prev[l] + prev[l ^ off];
        }
    }
    p[0]
}

/// Most clip-search candidates (shrink ratios 0.95 down to 0.50), the 4-bit quantizer's limit.
pub const MAX_CLIP_SEARCH: u8 = 10;

/// MLX's affine quantizer for row-major `values` `[n][k]` (bf16-representable) at `bits`, groups of
/// 64, with the kernel's clip search: `[lo, hi]` shrunk about its midpoint by `1 - 0.05 i` for
/// `i = 1..=clip` (at most [`MAX_CLIP_SEARCH`]), keeping the least stored-dequantization error, ties
/// to the earlier candidate. `Err` is the first failing group's linear index (row * k / 64 + group),
/// as the kernel reports it.
pub fn quantize_mlx(
    values: &[f32],
    n: usize,
    k: usize,
    bits: u8,
    clip: u8,
) -> Result<MlxWeight, usize> {
    let kg = layout::groups(k);
    let rw = layout::mlx_row_words(k, bits);
    let nb = levels(bits);
    let mut w = MlxWeight {
        words: vec![0; n * rw],
        scales: vec![0; n * kg],
        biases: vec![0; n * kg],
        n,
        k,
        bits,
    };
    for gi in 0..n * kg {
        let (row, g) = (gi / kg, gi % kg);
        let v = &values[row * k + g * layout::GROUP..row * k + (g + 1) * layout::GROUP];
        if v.iter().any(|x| !x.is_finite()) {
            return Err(gi);
        }
        // The kernel's -0.0 canonicalization (`lo + 0.0`), so the zero sign never matters.
        let lo = v.iter().copied().fold(f32::INFINITY, f32::min) + 0.0;
        let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) + 0.0;
        let Some((mut s, mut b)) = mlx_params(lo, hi, nb) else {
            return Err(gi);
        };
        if clip > 0 {
            let mut best = group_sse(v, s, b, nb);
            let (mid, half_w) = ((lo + hi) * 0.5, (hi - lo) * 0.5);
            for i in 1..=clip.min(MAX_CLIP_SEARCH) {
                let h = half_w * (1.0 - 0.05 * f32::from(i));
                let Some((cs, cb)) = mlx_params(mid - h, mid + h, nb) else {
                    continue;
                };
                let e = group_sse(v, cs, cb, nb);
                if e < best {
                    (best, s, b) = (e, cs, cb);
                }
            }
        }
        w.scales[gi] = bf16_bits(s);
        w.biases[gi] = bf16_bits(b);
        let words = &mut w.words[row * rw..(row + 1) * rw];
        for (j, &x) in v.iter().enumerate() {
            let q = mlx_code(x, s, b, nb);
            let bit = (g * layout::GROUP + j) * usize::from(bits);
            let (wd, sh) = (bit >> 5, bit & 31);
            words[wd] |= q << sh;
            if sh + usize::from(bits) > 32 {
                words[wd + 1] |= q >> (32 - sh);
            }
        }
    }
    Ok(w)
}

/// Dequantized values in fp32: `s * q + b`, product and sum each rounded (the dequant kernel's ops).
pub fn dequant_mlx(w: &MlxWeight) -> Vec<f32> {
    let (kg, rw) = (layout::groups(w.k), layout::mlx_row_words(w.k, w.bits));
    let mut out = vec![0f32; w.n * w.k];
    for row in 0..w.n {
        let words = &w.words[row * rw..(row + 1) * rw];
        for i in 0..w.k {
            let gi = row * kg + i / layout::GROUP;
            let q = code_at(words, i, w.bits) as f32;
            out[row * w.k + i] = bf16_f32(w.scales[gi]) * q + bf16_f32(w.biases[gi]);
        }
    }
    out
}

/// `(n, kg)` tables to the tiles' `(kg, npad)` layout, padded rows zero.
pub fn transpose_pad(table: &[u16], n: usize, kg: usize) -> Vec<u16> {
    let np = layout::npad(n);
    let mut out = vec![0u16; kg * np];
    for g in 0..kg {
        for col in 0..n {
            out[g * np + col] = table[col * kg + g];
        }
    }
    out
}

/// The tiles of `w`, word for word what `affine_n_pack_ada` (narrow) or `affine8_pack_ada` writes.
pub fn pack(w: &MlxWeight) -> Tiles {
    let (n, k, bits) = (w.n, w.k, w.bits);
    let (kg, rw, b) = (
        layout::groups(k),
        layout::mlx_row_words(k, bits),
        usize::from(bits),
    );
    let mut words = vec![0u32; layout::tile_words(n, k, bits)];
    for (idx, out) in words.iter_mut().enumerate() {
        let mut word = 0u32;
        if layout::is_wide(bits) {
            let (kt, lane, j, tg) = (idx & 3, (idx >> 2) & 31, (idx >> 7) & 7, idx >> 10);
            let (g, t) = (tg % kg, tg / kg);
            let row = t * 64 + j * 8 + (lane >> 2);
            if row < n {
                let src = &w.words[row * rw..(row + 1) * rw];
                let k0 = g * 64 + 16 * kt + 2 * (lane & 3);
                for (p, off) in [0usize, 1, 8, 9].into_iter().enumerate() {
                    word |= code_at(src, k0 + off, bits) << (8 * p);
                }
            }
        } else {
            let (v, rest) = (idx % b, idx / b);
            let (lane, wn, tg) = (rest & 31, (rest >> 5) & 3, rest >> 7);
            let (g, t, c) = (tg % kg, tg / kg, lane & 3);
            for bit in 0..16 {
                let sb = 16 * v + bit;
                let (i, at) = (sb / b, sb % b);
                let (j, kt, h) = (i >> 3, (i >> 1) & 3, i & 1);
                let row = t * 64 + (wn * 2 + j) * 8 + (lane >> 2);
                if row >= n {
                    continue;
                }
                let src = &w.words[row * rw..(row + 1) * rw];
                let kk = g * 64 + 16 * kt + 2 * c + 8 * h;
                word |= ((code_at(src, kk, bits) >> at) & 1) << bit;
                word |= ((code_at(src, kk + 1, bits) >> at) & 1) << (16 + bit);
            }
        }
        *out = word;
    }
    Tiles {
        words,
        scales_t: transpose_pad(&w.scales, n, kg),
        biases_t: transpose_pad(&w.biases, n, kg),
        n,
        k,
        bits,
    }
}

/// The code at `(row, kk)` by the index math of `affine_n_dequant_bf16_ada`.
pub fn tile_code(t: &Tiles, row: usize, kk: usize) -> u32 {
    let kg = layout::groups(t.k);
    let (g, inner, tile, rr) = (kk / 64, kk % 64, row / 64, row % 64);
    let (kt, rem) = (inner / 16, inner % 16);
    let (h, c, e) = (rem / 8, (rem % 8) / 2, rem % 2);
    let lane = (rr % 8) * 4 + c;
    if layout::is_wide(t.bits) {
        let word = t.words[(((tile * kg + g) * 8 + rr / 8) * 32 + lane) * 4 + kt];
        return (word >> (8 * (h * 2 + e))) & 0xff;
    }
    let b = usize::from(t.bits);
    let (wn, j) = (rr / 16, (rr / 8) % 2);
    let i = j * 8 + kt * 2 + h;
    let base = ((tile * kg + g) * 128 + wn * 32 + lane) * b;
    (0..b).fold(0u32, |q, at| {
        let sb = i * b + at;
        q | (((t.words[base + sb / 16] >> (sb % 16 + 16 * e)) & 1) << at)
    })
}

/// `__byte_perm(a, b, 0x5410)` and `0x7632`: the low or the high halves of `a` and `b` joined.
fn halves(a: u32, b: u32, high: bool) -> u32 {
    if high {
        (a >> 16) | (b & 0xffff_0000)
    } else {
        (a & 0xffff) | (b << 16)
    }
}

/// The two codes of operand pair `i` of a narrow lane's words (the kernel's `pair_at`).
pub fn pair_at(w: &[u32], bits: u8, i: usize) -> (u32, u32) {
    let b = usize::from(bits);
    let m = (1u32 << bits) - 1;
    let (o, h) = (i * b, (i * b) >> 4);
    let sh = o & 15;
    if sh + b <= 16 {
        return ((w[h] >> sh) & m, (w[h] >> (16 + sh)) & m);
    }
    let (lo, hi) = (halves(w[h], w[h + 1], false), halves(w[h], w[h + 1], true));
    ((lo >> sh) & m, (hi >> sh) & m)
}

/// The codes of column tile `tile`, group `g`, as the GEMV's 128 lanes hand them to the MMA:
/// `out[col][kk]` for the tile's 64 columns and the group's 64 inputs.
pub fn lane_codes(t: &Tiles, tile: usize, g: usize) -> Vec<[u8; 64]> {
    let kg = layout::groups(t.k);
    let mut out = vec![[0u8; 64]; 64];
    for wn in 0..4 {
        for lane in 0..32 {
            let c = lane & 3;
            for j in 0..2 {
                let col = wn * 16 + j * 8 + (lane >> 2);
                for kt in 0..4 {
                    let pairs: [(u32, u32); 2] = if layout::is_wide(t.bits) {
                        let w = t.words[(((tile * kg + g) * 8 + wn * 2 + j) * 32 + lane) * 4 + kt];
                        [(w & 0xff, (w >> 8) & 0xff), ((w >> 16) & 0xff, w >> 24)]
                    } else {
                        let b = usize::from(t.bits);
                        let base = ((tile * kg + g) * 128 + wn * 32 + lane) * b;
                        let lw = &t.words[base..base + b];
                        let i = (j * 4 + kt) * 2;
                        [pair_at(lw, t.bits, i), pair_at(lw, t.bits, i + 1)]
                    };
                    for (h, (c0, c1)) in pairs.into_iter().enumerate() {
                        let kk = 16 * kt + 2 * c + 8 * h;
                        out[col][kk] = c0 as u8;
                        out[col][kk + 1] = c1 as u8;
                    }
                }
            }
        }
    }
    out
}

pub use super::reference_gemv::{gemv_f64, gemv_model, group_sums};
