// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from llama.cpp (ggml-org/llama.cpp), MIT License, Copyright (c) 2023-2026 The ggml
// authors: ggml/src/ggml-cuda/quantize.cu (the q8_1 block: d = amax / 127, q = round(x / d), d stored as
// f16; quantize_mmq_q8_1: d_inv = 127 / amax, the D4 and DS4 layouts), ggml/src/ggml-quants.c
// (get_scale_min_k4). Changes: a host mirror in Rust for the oracle of the Ada kernels, by Atlas
// contributors.

//! Responsibility: the CPU oracle of the keep-packed GGUF 4-bit route (port lane L19): the bit-exact host
//! mirrors of the decode and MMQ activation quantizers, the weight values (through spark-runtime's
//! bit-exact GGUF decoders), the float64 GEMV, the Q4_K partial-sum term of the MMQ tiles, and the error
//! bounds a GPU output is held to.
//!
//! The GPU GEMV multiplies the packed weight by the q8_1-QUANTIZED activation (dp4a on int8 codes), so the
//! reference is the float64 dot of the dequantized weight with the dequantized q8_1 activation
//! (`q * f16(d)`); that leaves only the kernel's f32 accumulation and the bf16 store as error, which
//! [`within_bound`] bounds. Against the original bf16 activation the GPU additionally carries the q8_1
//! rounding (about 0.4% of a block's largest value per element), reported separately by the microtest.

use anyhow::{Result, ensure};
use spark_runtime::weights::dequant_cpu::{GgmlType, dequant_to_f32};

use super::launch::{Q4Type, QK_K, QK8_1};

/// The `GgmlType` of a packed type.
pub fn ggml_type(ty: Q4Type) -> GgmlType {
    match ty {
        Q4Type::Iq4Xs => GgmlType::Iq4Xs,
        Q4Type::Q4K => GgmlType::Q4K,
        Q4Type::Q6K => GgmlType::Q6K,
    }
}

/// bf16 bits to f32 (exact).
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// One `block_q8_1` as a quantizer writes it: 32 codes, the scale the dot product reads, and the
/// partial sum of the ORIGINAL values (stored as f16 by both the decode rows form and the MMQ DS4 form).
#[derive(Clone, Debug, PartialEq)]
pub struct Q8Block {
    pub qs: [i8; 32],
    /// The scale as the kernels read it: `f16(amax / 127)` for the decode rows and the MMQ DS4 layout,
    /// the f32 `1 / (127 / amax)` for the MMQ D4 layout.
    pub d: f32,
    /// `f16(sum of the 32 bf16 values)`, summed here in index order (the GPU sums in a butterfly, so its
    /// f16 can differ by one unit in the last place; [`q4_k_sum_correction`] allows for that).
    pub s: f32,
}

fn f16_round(v: f32) -> f32 {
    half::f16::from_f32(v).to_f32()
}

/// Mirror of `atlas_q8_1_rows_bf16_ada` for one row of `k` bf16 values (`k % 32 == 0`).
pub fn quantize_q8_1_row(x: &[u16]) -> Vec<Q8Block> {
    x.chunks_exact(QK8_1)
        .map(|blk| {
            let xf: Vec<f32> = blk.iter().map(|&b| bf16_to_f32(b)).collect();
            let amax = xf.iter().fold(0f32, |a, v| a.max(v.abs()));
            let d = amax / 127.0;
            let mut qs = [0i8; 32];
            if amax != 0.0 {
                for (q, v) in qs.iter_mut().zip(&xf) {
                    // CUDA `roundf` rounds half away from zero, as `f32::round` does.
                    *q = (v / d).round() as i8;
                }
            }
            Q8Block {
                qs,
                d: f16_round(d),
                s: f16_round(xf.iter().sum()),
            }
        })
        .collect()
}

/// The MMQ activation layout a weight type's tiles read (`mmq_get_q8_1_ds_layout`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmqLayout {
    /// One f32 scale per 32 values (IQ4_XS, Q6_K).
    D4,
    /// One `half2(scale, partial sum)` per 32 values (Q4_K).
    Ds4,
}

/// The MMQ layout of a packed type (the host side of `Q4Type::mmq_act_entry`).
pub fn mmq_layout(ty: Q4Type) -> MmqLayout {
    match ty {
        Q4Type::Q4K => MmqLayout::Ds4,
        Q4Type::Iq4Xs | Q4Type::Q6K => MmqLayout::D4,
    }
}

/// Mirror of the vendored `quantize_mmq_q8_1_worker` (the prefill activations) for one row of bf16
/// values (`k % 32 == 0`): `d_inv = 127 / amax`, `q = round(x * d_inv)` (a product, not the decode rows'
/// quotient, so a code can differ from [`quantize_q8_1_row`] by one), `d = 1 / d_inv`. An all-zero block
/// gives `d_inv = inf`, `0 * inf = NaN`, which both the GPU conversion and Rust's `as i8` turn into 0.
pub fn quantize_q8_1_mmq_row(x: &[u16], layout: MmqLayout) -> Vec<Q8Block> {
    x.chunks_exact(QK8_1)
        .map(|blk| {
            let xf: Vec<f32> = blk.iter().map(|&b| bf16_to_f32(b)).collect();
            let amax = xf.iter().fold(0f32, |a, v| a.max(v.abs()));
            let d_inv = 127.0f32 / amax;
            let mut qs = [0i8; 32];
            for (q, v) in qs.iter_mut().zip(&xf) {
                *q = (v * d_inv).round() as i8;
            }
            let d = 1.0f32 / d_inv;
            Q8Block {
                qs,
                d: match layout {
                    MmqLayout::D4 => d,
                    MmqLayout::Ds4 => f16_round(d),
                },
                s: f16_round(xf.iter().sum()),
            }
        })
        .collect()
}

/// ggml `get_scale_min_k4`: the 6-bit scale and min of sub-block `j` (0..8) of a K-quant block.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// `dmin * m` of every 32-weight sub-block of a raw Q4_K `[n, k]` weight, laid out `[n][k / 32]`: the
/// constant part of each weight value (`w = d * sc * q - dmin * m`).
pub fn q4_k_mins(raw: &[u8], n: usize, k: usize) -> Vec<f32> {
    let per_row = k / QK_K;
    let mut out = Vec::with_capacity(n * k / QK8_1);
    for blk in raw
        .chunks_exact(Q4Type::Q4K.block_bytes())
        .take(n * per_row)
    {
        let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        let scales = &blk[4..16];
        for j in 0..8 {
            out.push(dmin * f32::from(scale_min_k4(j, scales).1));
        }
    }
    out
}

/// The MMQ Q4_K tile pairs each sub-block's min with the activation's stored partial sum `s` (DS4) where
/// the dequantized reference pairs it with `d * sum(q)`. Returns, per output `[m][n]`, the difference to
/// add to [`gemv_f64`]'s `want` (`sum over sub-blocks of min * (d * sum(q) - s)`), and the slack for the
/// GPU's own f16 sum (two f16 units of `|min * s|` per sub-block).
pub fn q4_k_sum_correction(
    mins: &[f32],
    n: usize,
    k: usize,
    rows: &[Vec<Q8Block>],
) -> (Vec<f64>, Vec<f64>) {
    let blocks = k / QK8_1;
    let mut corr = vec![0f64; rows.len() * n];
    let mut slack = vec![0f64; rows.len() * n];
    for (r, row) in rows.iter().enumerate() {
        for j in 0..n {
            let (mut c, mut sl) = (0f64, 0f64);
            for (b, q8) in row.iter().enumerate().take(blocks) {
                let m = f64::from(mins[j * blocks + b]);
                let sum_q: i32 = q8.qs.iter().map(|&q| i32::from(q)).sum();
                c += m * (f64::from(q8.d) * f64::from(sum_q) - f64::from(q8.s));
                sl += (m * f64::from(q8.s)).abs() * 2f64.powi(-10);
            }
            corr[r * n + j] = c;
            slack[r * n + j] = sl;
        }
    }
    (corr, slack)
}

/// The prefill (MMQ) bound: [`bound`] plus `2^-10` of the magnitude, because the MMA tiles of the K-quant
/// types hold each sub-block's `d * sc` and `dmin * m` as f16 (`load_tiles_q4_K`), which the f32 CPU
/// decoder does not round.
pub fn prefill_bound(want: f64, mag: f64, k: usize) -> f64 {
    bound(want, mag, k) + mag * 2f64.powi(-10)
}

/// The activation values the GPU dot product sees for one row: `q * f16(d)`.
pub fn q8_1_values(row: &[Q8Block]) -> Vec<f32> {
    row.iter()
        .flat_map(|b| b.qs.iter().map(move |&q| f32::from(q) * b.d))
        .collect()
}

/// The `[n, k]` weight values of raw packed bytes (spark-runtime's bit-exact decoders).
pub fn dequant_weight(ty: Q4Type, raw: &[u8], n: usize, k: usize) -> Result<Vec<f32>> {
    ensure!(
        k.is_multiple_of(QK_K),
        "k = {k} is not a multiple of {QK_K}"
    );
    let mut out = vec![0f32; n * k];
    dequant_to_f32(ggml_type(ty), raw, n * k, &mut out)?;
    Ok(out)
}

/// `(want, mag)` per output `[m][n]`: the float64 dot of weight row `j` with activation row `r`, and the
/// sum of the absolute products (the scale of the accumulation error).
pub fn gemv_f64(w: &[f32], n: usize, k: usize, x: &[f32], m: usize) -> (Vec<f64>, Vec<f64>) {
    let mut want = vec![0f64; m * n];
    let mut mag = vec![0f64; m * n];
    for r in 0..m {
        let xr = &x[r * k..(r + 1) * k];
        for j in 0..n {
            let wr = &w[j * k..(j + 1) * k];
            let (mut s, mut a) = (0f64, 0f64);
            for (wv, xv) in wr.iter().zip(xr) {
                let p = f64::from(*wv) * f64::from(*xv);
                s += p;
                a += p.abs();
            }
            want[r * n + j] = s;
            mag[r * n + j] = a;
        }
    }
    (want, mag)
}

/// The bound a GPU output `got` (already widened from bf16) is held to: the bf16 store (unit roundoff
/// 2^-8 of the value) plus f32 accumulation over the kernel's partial sums (one product per 32-value
/// block per lane and a reduction tree: at most `k / 32 + 64` f32 roundings, each at most 2^-24 of the
/// running magnitude) plus the f32 scale products inside each block's dot (a few roundings per term).
pub fn bound(want: f64, mag: f64, k: usize) -> f64 {
    let bf16_store = want.abs() * 2f64.powi(-8);
    let accum = (k / QK8_1 + 64) as f64 * f64::from(f32::EPSILON) * mag;
    bf16_store + accum + f64::MIN_POSITIVE
}

/// True when `got` is within [`bound`] of `want`.
pub fn within_bound(got: f32, want: f64, mag: f64, k: usize) -> bool {
    got.is_finite() && (f64::from(got) - want).abs() <= bound(want, mag, k)
}

/// `(count over the bound, worst error in units of the bound)` of a GPU output against the reference.
pub fn grade(got: &[f32], want: &[f64], mag: &[f64], k: usize) -> (usize, f64) {
    grade_with(got, want, mag, k, bound)
}

/// [`grade`] against an explicit bound function (for example [`prefill_bound`]).
pub fn grade_with(
    got: &[f32],
    want: &[f64],
    mag: &[f64],
    k: usize,
    bound_fn: fn(f64, f64, usize) -> f64,
) -> (usize, f64) {
    let mut over = 0usize;
    let mut worst = 0f64;
    for ((g, w), a) in got.iter().zip(want).zip(mag) {
        let e = (f64::from(*g) - w).abs() / bound_fn(*w, *a, k);
        if !g.is_finite() || e > 1.0 {
            over += 1;
        }
        if e.is_finite() {
            worst = worst.max(e);
        }
    }
    (over, worst)
}
