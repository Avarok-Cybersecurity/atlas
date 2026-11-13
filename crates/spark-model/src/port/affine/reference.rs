// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/core/affine4_host.zig:7-66 (f32of, bf16of, dequantize, quantize, quantizeMlx) and
// zig/kernels/cuda/affine4_pack.cu:5-36 (the tiled layout's index map). Copyright 2026 TensorFold
// contributors. Licensed under the Apache License, Version 2.0; code written before TensorFold 0.6.0 is
// also available under the MIT License. See TENSORFOLD-ATTRIBUTION.md.
// Changes for Atlas: translated from Zig and CUDA to Rust; the group size is a parameter; the quantizer
// reproduces the GPU kernel's warp order (lane l holds inputs l*V..l*V+V-1, the clip-search error is a
// per-lane in-order sum then an xor butterfly), canonicalizes a -0.0 edge to +0.0, and gains the clip
// search; the pack inverse, the transpose inverse and the tiled dequantizer are new.

//! Responsibility: the host mirrors of `kernels/ada/common/affine_pack_ada.cu`, bit for bit: the MLX
//! affine 4-bit quantizer, the tiled pack and its inverse, the scale transpose, the BF16 dequantizer and
//! the group sums. The float64 GEMV reference and its error bound are in `reference_gemv.rs`.
//!
//! Every float operation here is a single IEEE binary32 operation (Rust never contracts `a * b + c`),
//! the same operations the kernels issue through `__f*_rn`, so the results are bit-identical.

pub use super::reference_gemv::{GemvRef, count_outside, gemv_bound, gemv_f64};

/// The f32 value of bf16 bits.
pub fn f32_of(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// The nearest bf16, ties to even; NaN stays NaN (`bf16of`, the kernels' rounding).
pub fn bf16_of(x: f32) -> u16 {
    let u = x.to_bits();
    if x.is_nan() {
        return ((u >> 16) | 0x40) as u16;
    }
    (u.wrapping_add(0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

/// Round half away from zero (`roundf`, Zig `@round`).
fn round(x: f32) -> f32 {
    x.round()
}

/// An MLX affine 4-bit matrix: words `[n, k/8]` (low nibble first), scales and biases `[n, k/group]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mlx {
    pub words: Vec<u32>,
    pub scales: Vec<u16>,
    pub biases: Vec<u16>,
    pub n: usize,
    pub k: usize,
    pub group: usize,
}

impl Mlx {
    /// The 4-bit code of `(row, col)`.
    pub fn code(&self, row: usize, col: usize) -> u32 {
        let i = row * self.k + col;
        (self.words[i / 8] >> (4 * (i % 8))) & 0xF
    }
}

/// MLX's scale and bias for `[lo, hi]` (`affine4_host.zig:48-56`); `None` when not representable.
pub fn mlx_params(lo: f32, hi: f32) -> Option<(f32, f32)> {
    let d = (hi - lo) / 15.0;
    let step = if d > 1e-7 { d } else { 1e-7 };
    if !step.is_finite() {
        return None;
    }
    let low_edge = lo.abs() > hi.abs();
    let sg = if low_edge { step } else { -step };
    let edge = if low_edge { lo } else { hi };
    let q0 = round(edge / sg);
    let (s, b) = if q0 == 0.0 {
        (sg, 0.0)
    } else {
        (edge / q0, edge)
    };
    (s.is_finite() && s != 0.0 && b.is_finite()).then_some((s, b))
}

/// The code of `x` under the unrounded f32 scale and bias.
pub fn code_of(x: f32, s: f32, b: f32) -> u32 {
    round((x - b) / s).clamp(0.0, 15.0) as u32
}

fn storable(s: f32, b: f32) -> bool {
    f32_of(bf16_of(s)).is_finite() && f32_of(bf16_of(b)).is_finite()
}

/// The kernel's warp sum: lane partials `p`, then `p[l] + p[l ^ off]` for off = 16, 8, 4, 2, 1.
pub fn warp_sum(mut p: [f32; 32]) -> f32 {
    for off in [16usize, 8, 4, 2, 1] {
        let prev = p;
        for l in 0..32 {
            p[l] = prev[l] + prev[l ^ off];
        }
    }
    p[0]
}

/// The clip-search error of one group under `(s, b)`: squared error of the stored dequantization, summed
/// as the kernel sums it.
pub fn group_err(v: &[f32], s: f32, b: f32) -> f32 {
    let lanes = v.len() / 32;
    let (sb, bb) = (f32_of(bf16_of(s)), f32_of(bf16_of(b)));
    let mut p = [0f32; 32];
    for (l, part) in p.iter_mut().enumerate() {
        let mut e = 0f32;
        for &x in &v[l * lanes..(l + 1) * lanes] {
            let d = (sb * code_of(x, s, b) as f32 + bb) - x;
            e += d * d;
        }
        *part = e;
    }
    warp_sum(p)
}

/// One group's `(scale, bias)` as the kernel picks them, or `None` for a bad group.
pub fn group_params(v: &[f32], clip: u8) -> Option<(f32, f32)> {
    if v.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let lo = v.iter().copied().fold(f32::INFINITY, f32::min) + 0.0;
    let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) + 0.0;
    let (mut s, mut b) = mlx_params(lo, hi)?;
    if !storable(s, b) {
        return None;
    }
    if clip > 0 {
        let mut best = group_err(v, s, b);
        let mid = (lo + hi) * 0.5;
        let half = (hi - lo) * 0.5;
        for i in 1..=clip {
            let ratio = 1.0f32 - 0.05f32 * f32::from(i);
            let h = half * ratio;
            let Some((cs, cb)) = mlx_params(mid - h, mid + h) else {
                continue;
            };
            if !storable(cs, cb) {
                continue;
            }
            let e = group_err(v, cs, cb);
            if e < best {
                best = e;
                (s, b) = (cs, cb);
            }
        }
    }
    Some((s, b))
}

/// The GPU quantizer (`affine4_quantize_bf16_g<group>`) on the host. `values` is `[n, k]` row-major
/// (bf16-exact f32 for a bit-equal comparison). `Err(gid)` names the first bad group, as the kernel's
/// flag does (its other groups are still written there).
pub fn quantize_mlx(
    values: &[f32],
    n: usize,
    k: usize,
    group: usize,
    clip: u8,
) -> Result<Mlx, usize> {
    let kg = k / group;
    let mut q = Mlx {
        words: vec![0; n * k / 8],
        scales: vec![0; n * kg],
        biases: vec![0; n * kg],
        n,
        k,
        group,
    };
    for gid in 0..n * kg {
        let base = (gid / kg) * k + (gid % kg) * group;
        let v = &values[base..base + group];
        let (s, b) = group_params(v, clip).ok_or(gid)?;
        q.scales[gid] = bf16_of(s);
        q.biases[gid] = bf16_of(b);
        for (j, &x) in v.iter().enumerate() {
            let i = base + j;
            q.words[i / 8] |= code_of(x, s, b) << (4 * (i % 8));
        }
    }
    Ok(q)
}

/// Upstream's simple min/max quantizer (`affine4_host.zig:22-38`: bias = min, scale = range / 15), at
/// group 64; kept for the ported upstream test.
pub fn quantize_minmax(values: &[f32], words: &mut [u32], scales: &mut [u16], biases: &mut [u16]) {
    const G: usize = 64;
    words.fill(0);
    for g in 0..values.len() / G {
        let v = &values[g * G..(g + 1) * G];
        let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        scales[g] = bf16_of((hi - lo) / 15.0);
        biases[g] = bf16_of(lo);
        let (s, b) = (f32_of(scales[g]), f32_of(biases[g]));
        for (j, &x) in v.iter().enumerate() {
            let q = if s == 0.0 {
                0
            } else {
                round((x - b) / s).clamp(0.0, 15.0) as u32
            };
            let i = g * G + j;
            words[i / 8] |= q << (4 * (i % 8));
        }
    }
}

/// Row-major f32 values of MLX words, scales and biases at group 64 (`affine4_host.zig:15-20`).
pub fn dequantize_minmax(words: &[u32], scales: &[u16], biases: &[u16], out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        let q = (words[i / 8] >> (4 * (i % 8))) & 0xF;
        *o = f32_of(scales[i / 64]) * q as f32 + f32_of(biases[i / 64]);
    }
}

/// `n` rounded up to 128.
pub fn npad_of(n: usize) -> usize {
    n.div_ceil(128) * 128
}

/// The B-fragment input order of one tiled word's nibbles (`affine4_pack.cu:16`).
pub const OFFS: [usize; 8] = [0, 8, 16, 24, 1, 9, 17, 25];

/// Index of tiled word `(t, g, j, lane, v)`.
fn tiled_index(
    t: usize,
    g: usize,
    j: usize,
    lane: usize,
    v: usize,
    kg: usize,
    lanes: usize,
) -> usize {
    (((t * kg + g) * 8 + j) * 32 + lane) * lanes + v
}

/// MLX words `[n, k/8]` -> the tiled codes `[npad/64][kg][8][32][group/32]` (`affine4_pack_tiles`).
pub fn pack_tiles(words: &[u32], n: usize, k: usize, group: usize) -> Vec<u32> {
    let (npad, kg, lanes) = (npad_of(n), k / group, group / 32);
    let mut out = vec![0u32; npad * k / 8];
    for (idx, o) in out.iter_mut().enumerate() {
        let v = idx % lanes;
        let lane = (idx / lanes) % 32;
        let j = (idx / (lanes * 32)) % 8;
        let tg = idx / (lanes * 256);
        let (g, t) = (tg % kg, tg / kg);
        let row = t * 64 + j * 8 + lane / 4;
        if row >= n {
            continue;
        }
        let c = lane % 4;
        let mut word = 0u32;
        for (p, off) in OFFS.iter().enumerate() {
            let input = g * group + 32 * v + 2 * c + off;
            let src = words[row * (k / 8) + input / 8];
            word |= ((src >> (4 * (input % 8))) & 0xF) << (4 * p);
        }
        *o = word;
    }
    out
}

/// Where `(row, col)`'s code sits in the tiled layout: `(word index, nibble)` (the dequant kernel's map).
pub fn tiled_position(row: usize, col: usize, k: usize, group: usize) -> (usize, usize) {
    let (kg, lanes) = (k / group, group / 32);
    let (g, ic) = (col / group, col % group);
    let (v, rem) = (ic / 32, ic % 32);
    let c = (rem % 8) / 2;
    let p = rem / 8 + 4 * (rem % 2);
    let lane = (row % 8) * 4 + c;
    let j = (row % 64) / 8;
    (tiled_index(row / 64, g, j, lane, v, kg, lanes), p)
}

/// The tiled codes back to MLX words `[n, k/8]`.
pub fn unpack_tiles(tiled: &[u32], n: usize, k: usize, group: usize) -> Vec<u32> {
    let mut words = vec![0u32; n * k / 8];
    for row in 0..n {
        for col in 0..k {
            let (wi, p) = tiled_position(row, col, k, group);
            let q = (tiled[wi] >> (4 * p)) & 0xF;
            let i = row * k + col;
            words[i / 8] |= q << (4 * (i % 8));
        }
    }
    words
}

/// `(n, kg)` -> `(kg, npad)` with zero columns past n (`affine_transpose_pad16`).
pub fn transpose_pad16(x: &[u16], n: usize, kg: usize, npad: usize) -> Vec<u16> {
    let mut out = vec![0u16; kg * npad];
    for g in 0..kg {
        for col in 0..n {
            out[g * npad + col] = x[col * kg + g];
        }
    }
    out
}

/// `(kg, npad)` -> `(n, kg)`.
pub fn untranspose16(t: &[u16], n: usize, kg: usize, npad: usize) -> Vec<u16> {
    let mut out = vec![0u16; n * kg];
    for col in 0..n {
        for g in 0..kg {
            out[col * kg + g] = t[g * npad + col];
        }
    }
    out
}

/// BF16 bits of `s * q + b` for every `[n, k]` element (`affine4_dequant_bf16`), from MLX layout.
pub fn dequant_bf16(q: &Mlx) -> Vec<u16> {
    let kg = q.k / q.group;
    let mut out = vec![0u16; q.n * q.k];
    for row in 0..q.n {
        for col in 0..q.k {
            let gi = row * kg + col / q.group;
            let val = f32_of(q.scales[gi]) * q.code(row, col) as f32 + f32_of(q.biases[gi]);
            out[row * q.k + col] = bf16_of(val);
        }
    }
    out
}

/// Group sums of `rows` bf16 rows `k` wide (`affine_group_sums`): in-order f32 sums, `[rows, k/group]`.
pub fn group_sums(x: &[u16], rows: usize, k: usize, group: usize) -> Vec<f32> {
    let kg = k / group;
    let mut out = vec![0f32; rows * kg];
    for r in 0..rows {
        for g in 0..kg {
            let mut s = 0f32;
            for &e in &x[r * k + g * group..r * k + (g + 1) * group] {
                s += f32_of(e);
            }
            out[r * kg + g] = s;
        }
    }
    out
}
