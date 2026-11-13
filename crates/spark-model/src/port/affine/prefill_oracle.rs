// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/kernels/cuda/w4a8_quantize.cuh:1-51 and w4a8_normalize.h:1-14 (the activation quantizer),
// zig/kernels/cuda/w4a8_prefill.cu:22-28 and 125-147 (`bytes4` and the int8 epilogue) and
// zig/kernels/cuda/affine4_pack.cu:5-36 (the tiled layout).
// Copyright 2026 TensorFold contributors. Licensed under the Apache License, Version 2.0; code written
// before TensorFold 0.6.0 is also available under the MIT License. See TENSORFOLD-ATTRIBUTION.md.
// Changes for Atlas: CUDA translated to Rust host models for the oracles (the warp butterfly emulated lane by
// lane, explicit fused multiply-adds via `f32::mul_add`); the float64 references and the per-group error bound of
// the int8 path are new.

//! Responsibility: CPU oracles of the affine prompt GEMMs (lane L17b). Pure host code, no GPU:
//!
//! * [`quantize_rows`] / [`swiglu_quantize`]: bit-exact models of `affine4_act_quant_ada` and
//!   `affine4_swiglu_act_quant_ada` (codes in slot order, scales, sums, the exceptional verdict); the SwiGLU
//!   itself uses the host `exp`, which is not bit-exact to `__expf`, so the GPU compares the fused kernel with the
//!   GPU pair instead.
//! * [`int8_gemm`]: a bit-exact model of `affine4_w4a8_prefill_ada` (exact integer group dots, the same fma
//!   chain in group order), so the GPU microtest compares output bytes.
//! * [`exact`], [`int8_bound`]: the float64 affine product and the derived bound on the int8 path's distance
//!   from it (activation rounding, the offset trick, the group sum, accumulation, the BF16 store).
//! * [`bf16_reference`]: the float64 product over BF16-rounded weights and its bound (`|want| * 2^-8 +
//!   gamma(K) * sum |x w|`), the gate of `affine4_prefill_bf16_ada`.
//! * [`HostAffine`]: logical MLX weights and their tiled device layout (codes, scales, biases).

use half::bf16;

/// Inputs a group.
pub const GROUP: usize = 64;
/// 12582912 = 1.5 * 2^23: the int8 accumulators' starting bit pattern read as a float.
pub const OFFSET: f32 = 12_582_912.0;
/// Code offset of nibble p within a lane's word, relative to `32 * word + 2 * (lane % 4)` (tf_pack_dense).
pub const NIBBLE_OFFS: [usize; 8] = [0, 8, 16, 24, 1, 9, 17, 25];
const U24: f64 = 1.0 / 16_777_216.0;
const U23: f64 = 2.0 * U24;
/// BF16 unit roundoff (half an ulp relative to the value).
pub const U_BF16: f64 = 1.0 / 256.0;
/// Smallest positive BF16 subnormal, 2^-133: the absolute floor of every BF16 bound.
pub const BF16_TINY: f64 = 9.183_549_615_799_121e-41;
/// Half the FP32 subnormal spacing, 2^-150.
const F32_HALF_TINY: f64 = 7.006_492_321_624_085e-46;

pub fn round_bf16(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}

pub fn bf16_bits(v: f32) -> u16 {
    bf16::from_f32(v).to_bits()
}

pub fn bf16_value(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}

/// `gamma(n) = n u / (1 - n u)` for unit roundoff `u`.
pub fn gamma(n: usize, u: f64) -> f64 {
    let nu = n as f64 * u;
    nu / (1.0 - nu)
}

/// Deterministic generator for oracle inputs.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    /// A BF16 value of roughly normal shape with standard deviation `sigma`.
    pub fn bf16_normal(&mut self, sigma: f32) -> f32 {
        let s = self.unit() + self.unit() + self.unit() + self.unit() - 2.0;
        round_bf16(s * 1.7 * sigma)
    }
}

/// Within each 32-input step the int8 mma's slot s holds input `slot_input(s)` (w4a8_quantize.cuh).
pub fn slot_input(s: usize) -> usize {
    let (q, i) = ((s & 15) >> 2, s & 3);
    (s & 16) + 2 * q + (i & 1) + 8 * (i >> 1)
}

/// The slot of input `i` (0..32) of a step: the inverse of [`slot_input`].
pub fn slot_of(i: usize) -> usize {
    (0..32).find(|&s| slot_input(s) == i).unwrap_or(usize::MAX)
}

/// `bytes4` of affine_prefill_ada.cu: the nibbles at s, s + 16, s + 4, s + 20 of `w` as four int8 bytes.
pub fn bytes4(w: u32, s: u32) -> [u8; 4] {
    [s, s + 16, s + 4, s + 20].map(|b| ((w >> b) & 15) as u8)
}

/// `pair` of affine_prefill_ada.cu: the codes at s (low half) and s + 16 (high half) of `w`.
pub fn pair(w: u32, s: u32) -> [u8; 2] {
    [s, s + 16].map(|b| ((w >> b) & 15) as u8)
}

/// Logical MLX affine 4-bit weights: `codes[col * k + i]` in 0..15, `scales` / `biases` `[n][k / 64]` BF16
/// values. Value of a weight: `s * q + b`.
#[derive(Clone, Debug)]
pub struct HostAffine {
    pub n: usize,
    pub k: usize,
    pub codes: Vec<u8>,
    pub scales: Vec<f32>,
    pub biases: Vec<f32>,
}

impl HostAffine {
    /// Random weights shaped like an MLX quantization of a zero-mean matrix: signed scales of about `scale`,
    /// biases near `-7.5 s` so the values center on zero.
    pub fn random(rng: &mut Lcg, n: usize, k: usize, scale: f32) -> Self {
        let kg = k / GROUP;
        let codes = (0..n * k).map(|_| (rng.next_u32() % 16) as u8).collect();
        let mut scales = Vec::with_capacity(n * kg);
        let mut biases = Vec::with_capacity(n * kg);
        for _ in 0..n * kg {
            let sign = if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
            let s = round_bf16(sign * scale * (0.5 + rng.unit()));
            scales.push(s);
            biases.push(round_bf16(-7.5 * s + 0.1 * scale * (rng.unit() - 0.5)));
        }
        Self {
            n,
            k,
            codes,
            scales,
            biases,
        }
    }

    pub fn kg(&self) -> usize {
        self.k / GROUP
    }

    pub fn npad(&self) -> usize {
        self.n.div_ceil(128) * 128
    }

    /// The exact affine value of weight (col, i).
    pub fn weight(&self, col: usize, i: usize) -> f64 {
        let g = col * self.kg() + i / GROUP;
        f64::from(self.codes[col * self.k + i]) * f64::from(self.scales[g])
            + f64::from(self.biases[g])
    }

    /// The BF16 operand the BF16 prompt GEMM forms: `fma.rn.bf16x2(q, s, b)`, one rounding of the exact value.
    pub fn weight_bf16(&self, col: usize, i: usize) -> f64 {
        bf16::from_f64(self.weight(col, i)).to_f64()
    }

    /// The tiled codes `[npad/64][k/64][8][32][2]` u32 (tf_pack_dense).
    pub fn tiled_words(&self) -> Vec<u32> {
        let kg = self.kg();
        let total = self.npad() / 64 * kg * 512;
        (0..total)
            .map(|idx| {
                let (v, lane, j) = (idx & 1, (idx >> 1) & 31, (idx >> 6) & 7);
                let (g, t) = ((idx >> 9) % kg, (idx >> 9) / kg);
                let row = t * 64 + j * 8 + (lane >> 2);
                if row >= self.n {
                    return 0;
                }
                (0..8).fold(0u32, |w, p| {
                    let input = g * 64 + 32 * v + 2 * (lane & 3) + NIBBLE_OFFS[p];
                    w | (u32::from(self.codes[row * self.k + input]) << (4 * p))
                })
            })
            .collect()
    }

    fn tiled(&self, vals: &[f32]) -> Vec<u16> {
        let (kg, npad) = (self.kg(), self.npad());
        (0..kg * npad)
            .map(|idx| {
                let (g, col) = (idx / npad, idx % npad);
                if col < self.n {
                    bf16_bits(vals[col * kg + g])
                } else {
                    0
                }
            })
            .collect()
    }

    /// The tiled scales `[k/64][npad]` BF16 bits, zero past n (tf_transpose_pad16).
    pub fn tiled_scales(&self) -> Vec<u16> {
        self.tiled(&self.scales)
    }

    pub fn tiled_biases(&self) -> Vec<u16> {
        self.tiled(&self.biases)
    }
}

/// The quantizer's output for `m` rows of `k` inputs, in device order.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantRows {
    pub m: usize,
    pub k: usize,
    /// `[m][k]` int8 codes as bytes, slot order within each 32-input step.
    pub xq: Vec<u8>,
    /// `[m][k/64]` scales and sums (0 for an exceptional group, which the kernel leaves unwritten).
    pub xa: Vec<f32>,
    pub xs: Vec<f32>,
    /// `[m][k/64]`: false where the group is exceptional.
    pub group_ok: Vec<bool>,
    /// flags[0]: any group exceptional.
    pub exceptional: bool,
}

impl QuantRows {
    /// The int8 code of input `i` of row `r`, in logical order.
    pub fn code(&self, r: usize, i: usize) -> i32 {
        let (g, w) = (i / GROUP, i % GROUP);
        i32::from(self.xq[r * self.k + g * GROUP + (w / 32) * 32 + slot_of(w % 32)] as i8)
    }

    /// Row `r`'s codes in logical order.
    pub fn row_codes(&self, r: usize) -> Vec<i32> {
        (0..self.k).map(|i| self.code(r, i)).collect()
    }
}

/// w4a8_normalize.h: the multiply whenever the reciprocal is finite.
fn normalized(v: f32, amax: f32, inv: f32) -> f32 {
    if inv > f32::MAX {
        (v / amax) * 127.0
    } else {
        v * inv
    }
}

/// One warp's group (`quantize_group`): `lanes[l] = (v0, v1)`, lane l's slots `2l % 32` and `+1` of step
/// `l / 16`. Returns the 64 code bytes in slot order and `(a, sum)`, or `None` when exceptional.
fn quantize_group(lanes: &[(f32, f32); 32]) -> Option<([u8; 64], f32, f32)> {
    if lanes.iter().any(|(a, b)| !a.is_finite() || !b.is_finite()) {
        return None;
    }
    let mut amax: [f32; 32] = std::array::from_fn(|l| lanes[l].0.abs().max(lanes[l].1.abs()));
    let mut sum: [f32; 32] = std::array::from_fn(|l| lanes[l].0 + lanes[l].1);
    for o in [16, 8, 4, 2, 1] {
        let (pa, ps) = (amax, sum);
        amax = std::array::from_fn(|l| pa[l].max(pa[l ^ o]));
        sum = std::array::from_fn(|l| ps[l] + ps[l ^ o]);
    }
    let (amax, sum) = (amax[0], sum[0]);
    let (a, inv) = if amax > 0.0 {
        (amax / 127.0, 127.0 / amax)
    } else {
        (1.0, 1.0)
    };
    if !a.is_finite() || !sum.is_finite() {
        return None;
    }
    let mut bytes = [0u8; 64];
    for (l, &(v0, v1)) in lanes.iter().enumerate() {
        let at = (l >> 4) * 32 + ((2 * l) & 31);
        bytes[at] = normalized(v0, amax, inv).round_ties_even() as i32 as u8;
        bytes[at + 1] = normalized(v1, amax, inv).round_ties_even() as i32 as u8;
    }
    Some((bytes, a, sum))
}

/// The input offsets within a group that lane l reads: `(step * 32 + slot_input(s0), .. s0 + 1)`.
fn lane_inputs(l: usize) -> (usize, usize) {
    let (s0, step) = ((2 * l) & 31, l >> 4);
    (step * 32 + slot_input(s0), step * 32 + slot_input(s0 + 1))
}

fn quantize_with(m: usize, k: usize, mut value: impl FnMut(usize, usize) -> f32) -> QuantRows {
    let kg = k / GROUP;
    let mut q = QuantRows {
        m,
        k,
        xq: vec![0; m * k],
        xa: vec![0.0; m * kg],
        xs: vec![0.0; m * kg],
        group_ok: vec![true; m * kg],
        exceptional: false,
    };
    for r in 0..m {
        for g in 0..kg {
            let lanes: [(f32, f32); 32] = std::array::from_fn(|l| {
                let (i0, i1) = lane_inputs(l);
                (value(r, g * GROUP + i0), value(r, g * GROUP + i1))
            });
            match quantize_group(&lanes) {
                Some((bytes, a, s)) => {
                    q.xq[r * k + g * GROUP..r * k + (g + 1) * GROUP].copy_from_slice(&bytes);
                    q.xa[r * kg + g] = a;
                    q.xs[r * kg + g] = s;
                }
                None => {
                    q.group_ok[r * kg + g] = false;
                    q.exceptional = true;
                }
            }
        }
    }
    q
}

/// `affine4_act_quant_ada` on `m` rows of `x` (BF16 values, row stride `ldx`).
pub fn quantize_rows(x: &[f32], ldx: usize, m: usize, k: usize) -> QuantRows {
    quantize_with(m, k, |r, i| x[r * ldx + i])
}

/// moe_silu_mul's expression with the host `exp`, rounded to BF16.
pub fn swiglu_bf16(g: f32, u: f32) -> f32 {
    let sigmoid_g = 1.0 / (1.0 + (-g).exp());
    round_bf16(g * sigmoid_g * u)
}

/// The fused `affine4_swiglu_act_quant_ada`, lane by lane: h (`[m][k]`, row stride k) and h's quantization.
pub fn swiglu_quantize(
    gate: &[f32],
    up: &[f32],
    ld_in: usize,
    m: usize,
    k: usize,
) -> (Vec<f32>, QuantRows) {
    let mut h = vec![0.0f32; m * k];
    let q = quantize_with(m, k, |r, i| {
        let v = swiglu_bf16(gate[r * ld_in + i], up[r * ld_in + i]);
        h[r * k + i] = v;
        v
    });
    (h, q)
}

/// The bit-exact model of `affine4_w4a8_prefill_ada`: BF16 bits of `out[r][0..n]` for every r in `rows`
/// (row-major, `rows.len() * n`). Requires no exceptional group in those rows.
pub fn int8_gemm(w: &HostAffine, q: &QuantRows, rows: &[usize]) -> Vec<u16> {
    let (k, kg) = (w.k, w.kg());
    let mut out = Vec::with_capacity(rows.len() * w.n);
    for &r in rows {
        let xr = q.row_codes(r);
        for col in 0..w.n {
            let codes = &w.codes[col * k..(col + 1) * k];
            let mut acc = 0.0f32;
            for g in 0..kg {
                let span = g * GROUP..(g + 1) * GROUP;
                let dot: i32 = codes[span.clone()]
                    .iter()
                    .zip(&xr[span])
                    .map(|(c, x)| i32::from(*c) * x)
                    .sum();
                let (a, rs) = (q.xa[r * kg + g], q.xs[r * kg + g]);
                let rc = -OFFSET * a;
                let d = (OFFSET + dot as f32).mul_add(a, rc);
                let (s, b) = (w.scales[col * kg + g], w.biases[col * kg + g]);
                acc = b.mul_add(rs, s.mul_add(d, acc));
            }
            out.push(bf16_bits(acc));
        }
    }
    out
}

/// The float64 affine product `sum_i (s q_i + b) x[r][i]` for every r in `rows` (row-major, `rows.len() * n`).
pub fn exact(w: &HostAffine, x: &[f32], ldx: usize, rows: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows.len() * w.n);
    for &r in rows {
        let xr = &x[r * ldx..r * ldx + w.k];
        for col in 0..w.n {
            out.push((0..w.k).map(|i| w.weight(col, i) * f64::from(xr[i])).sum());
        }
    }
    out
}

/// The derived bound on `|int8 output - exact|` for every r in `rows`, given the exact values `want` (from
/// [`exact`], same order). Per group g of row r and column c, with a the group scale and q the codes:
///
/// * activation rounding: `|s| a (1/2 + 2^-14) sum q` (`|x - a xq| <= a (1/2 + 2^-14)` per input);
/// * the offset trick: `|s| (1.0001 a + 2^-23 a |dot|)` (`-12582912 a` rounds by at most a, then one fma);
/// * the FP32 group sum: `|b| gamma(7) sum |x|` (six additions deep);
///
/// plus `gamma(2 KG + 2)` times the sum of the groups' term magnitudes (two fmas a group), plus the BF16 store
/// (`2^-8` of the result) and the BF16 subnormal floor.
pub fn int8_bound(
    w: &HostAffine,
    x: &[f32],
    ldx: usize,
    q: &QuantRows,
    rows: &[usize],
    want: &[f64],
) -> Vec<f64> {
    let (k, kg) = (w.k, w.kg());
    let mut out = Vec::with_capacity(rows.len() * w.n);
    for (ri, &r) in rows.iter().enumerate() {
        let xr = q.row_codes(r);
        let xv = &x[r * ldx..r * ldx + k];
        let sx: Vec<f64> = (0..kg)
            .map(|g| {
                xv[g * GROUP..(g + 1) * GROUP]
                    .iter()
                    .map(|v| f64::from(v.abs()))
                    .sum()
            })
            .collect();
        for col in 0..w.n {
            let codes = &w.codes[col * k..(col + 1) * k];
            let (mut e, mut t) = (0.0f64, 0.0f64);
            for g in 0..kg {
                let span = g * GROUP..(g + 1) * GROUP;
                let sq: f64 = codes[span.clone()].iter().map(|c| f64::from(*c)).sum();
                let dot: f64 = codes[span.clone()]
                    .iter()
                    .zip(&xr[span])
                    .map(|(c, v)| f64::from(*c) * f64::from(*v))
                    .sum();
                let a = f64::from(q.xa[r * kg + g]);
                let (s, b) = (
                    f64::from(w.scales[col * kg + g]).abs(),
                    f64::from(w.biases[col * kg + g]).abs(),
                );
                // (+ the absolute rounding of a subnormal FP32 scale: half its spacing per code unit)
                e += s * (a * (0.5 + 1.0 / 16_384.0) + 127.0 * F32_HALF_TINY) * sq;
                e += s * (1.0001 * a + U23 * a * dot.abs());
                e += b * gamma(7, U24) * sx[g];
                t += s * a * dot.abs() + b * f64::from(q.xs[r * kg + g]).abs();
            }
            e += gamma(2 * kg + 2, U24) * t;
            let y = want[ri * w.n + col].abs();
            out.push(e + U_BF16 * (y + e) + BF16_TINY);
        }
    }
    out
}

/// The BF16 prompt GEMM's reference: `(want, bound)` per output of every r in `rows`, with `want` the float64
/// product over the BF16-rounded weights and `bound = 2^-8 (|want| + G) + G + tiny`, `G = gamma(K, 2^-23)
/// sum |x w|` (the tensor-core FP32 accumulation taken at twice the FP32 unit roundoff).
pub fn bf16_reference(
    w: &HostAffine,
    x: &[f32],
    ldx: usize,
    rows: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let wb: Vec<f64> = (0..w.n * w.k)
        .map(|j| w.weight_bf16(j / w.k, j % w.k))
        .collect();
    let (mut want, mut bound) = (Vec::new(), Vec::new());
    for &r in rows {
        let xr = &x[r * ldx..r * ldx + w.k];
        for col in 0..w.n {
            let wr = &wb[col * w.k..(col + 1) * w.k];
            let (v, mag) = wr
                .iter()
                .zip(xr)
                .fold((0.0f64, 0.0f64), |(v, m), (wv, xv)| {
                    let p = wv * f64::from(*xv);
                    (v + p, m + p.abs())
                });
            let g = gamma(w.k, U23) * mag;
            want.push(v);
            bound.push(U_BF16 * (v.abs() + g) + g + BF16_TINY);
        }
    }
    (want, bound)
}

/// Outputs (as BF16 bits) farther than their bound from `want`; NaN on either side counts as a violation.
pub fn violations(got: &[u16], want: &[f64], bound: &[f64]) -> usize {
    got.iter()
        .zip(want.iter().zip(bound))
        .filter(|(g, (w, b))| {
            let v = f64::from(bf16_value(**g));
            let within = matches!(
                (v - **w).abs().partial_cmp(*b),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            );
            !within
        })
        .count()
}
